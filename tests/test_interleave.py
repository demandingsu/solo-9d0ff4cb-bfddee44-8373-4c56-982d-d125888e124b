"""同号重试交错与重启恢复的回归测试。

针对事故场景:旧失败线程标记失败后继续清理,误删同号重试新尝试已发布的文件,
导致页面显示"已发布/对数正常"而下载 404。修复后的不变量:
- 旧尝试只能清理自己的残留(凭 .attempt 归属标记);
- 新尝试的文件、对数与 published 状态始终一致;
- 进程中断后可安全重试;已发布运行不可改写。
"""
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import db_path, published_dir, staging_dir  # noqa: E402
from app.db import db_conn  # noqa: E402
from app.main import (  # noqa: E402
    ATTEMPT_MARKER,
    TRASH_PREFIX,
    adopt_stale_publish,
    attempt_token,
    reclaim_published,
    write_attempt_marker,
)

from test_app import (  # noqa: E402
    counts_of,
    make_inputs,
    run_id_of,
    submit,
    submit_good,
    wait_done,
)


# ------------------------------------------------------------ 认领逻辑单元测试

def test_reclaim_deletes_own_residue(tmp_path):
    pub = tmp_path / "published"
    pub.mkdir()
    final = pub / "RUNA"
    final.mkdir()
    (final / "S1_R1.fastq").write_text("x")
    write_attempt_marker(final, "tok-old")

    reclaim_published(final, "tok-old")  # 标记是自己的 -> 删

    assert not final.exists()
    assert not list(pub.iterdir())


def test_reclaim_removes_unmarked_legacy_dir(tmp_path):
    """无标记目录(旧版本/手工遗留的无主半成品)可安全回收。"""
    pub = tmp_path / "published"
    pub.mkdir()
    final = pub / "RUNB"
    final.mkdir()
    (final / "S1_R1.fastq").write_text("x")

    reclaim_published(final, "tok-new")

    assert not final.exists()


def test_reclaim_restores_foreign_published_dir(tmp_path):
    """核心不变量:别人(同号重试新尝试)的已发布目录必须原样还原。"""
    pub = tmp_path / "published"
    pub.mkdir()
    final = pub / "RUNC"
    final.mkdir()
    (final / "S1_R1.fastq").write_text("new-attempt-files")
    (final / "demux.log").write_text("log")
    write_attempt_marker(final, "tok-new")

    reclaim_published(final, "tok-old")  # 旧尝试来清理

    assert final.is_dir()
    assert (final / "S1_R1.fastq").read_text() == "new-attempt-files"
    assert (final / ATTEMPT_MARKER).read_text() == "tok-new"
    leftovers = [p for p in pub.iterdir() if p.name.startswith(TRASH_PREFIX)]
    assert leftovers == []


def test_adopt_stale_publish_removes_confirmed_orphan(client):
    """新尝试发布前,旧尝试来不及回收的无主残留可被认领清除。"""
    final = published_dir() / "RUNZ"
    final.mkdir(parents=True)
    (final / "half.fastq").write_text("orphan")
    write_attempt_marker(final, "dead-token")

    with db_conn(db_path()) as conn:
        cur = conn.execute(
            "INSERT INTO runs(run_number, status, staging_dir, created_at, updated_at) "
            "VALUES ('RUNZ', 'processing', ?, 't', 't')",
            (str(staging_dir() / "live-token"),),
        )
        live_id = cur.lastrowid

    adopt_stale_publish(final, "live-token", live_id)
    assert not final.exists()


# ------------------------------------------------------------ 端到端:失败清理 × 同号重试交错

def test_failed_attempt_cleanup_does_not_touch_new_publish(client, monkeypatch):
    """旧线程卡在清理半路时同号重传成功:旧清理结束后新发布文件必须完好。"""
    import app.main as main

    gate = threading.Event()
    real_reclaim = main.reclaim_published

    def gated_reclaim(final, token):
        # 只阻塞第一次(失败旧尝试)的认领,新尝试的调用直接放行
        if not getattr(gated_reclaim, "armed", False):
            gated_reclaim.armed = True
            gate.wait(timeout=15)
        return real_reclaim(final, token)

    monkeypatch.setattr(main, "reclaim_published", gated_reclaim)

    r1, r2, i1, csv = make_inputs()
    bad_r2 = r2[: r2.rfind(b"\n", 0, r2.rfind(b"\n"))]  # 末条记录缺质量行
    rid1 = run_id_of(submit(client, "RUN-I", r1, bad_r2, i1, csv))
    st1 = wait_done(client, rid1)
    assert st1["status"] == "failed"
    # 此刻旧线程正阻塞在 gated_reclaim,失败状态已落库,同号重试被接受
    rid2 = run_id_of(submit(client, "RUN-I", r1, r2, i1, csv))
    st2 = wait_done(client, rid2)
    assert st2["status"] == "published", st2.get("error")
    assert counts_of(st2) == {
        "S1": (3, 1), "S2": (1, 0), "S3": (1, 1), "undetermined": (2, 0),
    }

    final = client.data_dir / "published" / "RUN-I"
    assert final.is_dir()
    dl = client.get(f"/runs/{rid2}/download/S1_R1.fastq")
    assert dl.status_code == 200

    gate.set()  # 放行旧线程的延迟清理
    time.sleep(0.5)  # 等旧线程把认领流程跑完

    # 旧清理跑完后:目录、文件、下载、状态、对数仍然全部一致
    assert final.is_dir(), "旧尝试的清理误删了新尝试的发布目录"
    with db_conn(db_path()) as conn:
        new_row = conn.execute("SELECT * FROM runs WHERE id=?", (rid2,)).fetchone()
    new_token = attempt_token(new_row["staging_dir"])
    assert (final / ATTEMPT_MARKER).read_text() == new_token
    assert client.get(f"/runs/{rid2}/download/S1_R1.fastq").status_code == 200
    assert client.get(f"/runs/{rid2}/download/demux.log").status_code == 200
    again = client.get(f"/api/runs/{rid2}").json()
    assert again["status"] == "published" and again["total_pairs"] == 7
    # 归属标记不属于可下载文件名
    assert client.get(f"/runs/{rid2}/download/{ATTEMPT_MARKER}").status_code == 404
    # 没有遗留隔离目录
    trash = [p for p in published_dir().iterdir() if p.name.startswith(TRASH_PREFIX)]
    assert trash == []
    # 旧尝试记录已在同号重试提交时按设计删除(行被新尝试取代),其 id 不再可见
    assert client.get(f"/api/runs/{rid1}").status_code == 404


def test_retry_while_old_residue_still_published_succeeds(client, monkeypatch):
    """旧线程还没来得及认领发布残留(rename 后、清理前的窗口)时同号重试:
    新尝试通过 adopt_stale_publish 清掉无主残留后正常发布。"""
    import app.main as main

    proceed = threading.Event()
    real_reclaim = main.reclaim_published

    def slow_reclaim(final, token):
        if not getattr(slow_reclaim, "held", False):
            slow_reclaim.held = True
            proceed.wait(timeout=15)
        return real_reclaim(final, token)

    monkeypatch.setattr(main, "reclaim_published", slow_reclaim)

    r1, r2, i1, csv = make_inputs()
    bad_r2 = r2[: r2.rfind(b"\n", 0, r2.rfind(b"\n"))]
    rid1 = run_id_of(submit(client, "RUN-J", r1, bad_r2, i1, csv))
    assert wait_done(client, rid1)["status"] == "failed"

    rid2 = run_id_of(submit(client, "RUN-J", r1, r2, i1, csv))
    st2 = wait_done(client, rid2)  # 新尝试须自行清掉残留并发布成功
    assert st2["status"] == "published", st2.get("error")
    assert client.get(f"/runs/{rid2}/download/S1_R2.fastq").status_code == 200

    proceed.set()
    time.sleep(0.4)
    assert client.get(f"/runs/{rid2}/download/S1_R2.fastq").status_code == 200
    final = client.data_dir / "published" / "RUN-J"
    assert final.is_dir()
    assert [p for p in published_dir().iterdir() if p.name.startswith(TRASH_PREFIX)] == []


# ------------------------------------------------------------ 重启恢复

def test_restart_sweeps_trash_and_preserves_published(client):
    """启动恢复:.trash 隔离目录与无主发布目录被清扫,已发布运行原样保留。"""
    from app.main import recover_interrupted

    rid = run_id_of(submit_good(client, "RUN-K"))
    assert wait_done(client, rid)["status"] == "published"
    good_dir = published_dir() / "RUN-K"
    assert good_dir.is_dir()

    trash_dir = published_dir() / f"{TRASH_PREFIX}deadbeef"
    trash_dir.mkdir()
    (trash_dir / "leftover.fastq").write_text("kill 时认领流程的遗留")
    orphan = published_dir() / "RUN-ORPHAN"
    orphan.mkdir()
    (orphan / "x.fastq").write_text("无主发布目录")
    stray_staging = staging_dir() / "orphan-token"
    stray_staging.mkdir(parents=True)
    (stray_staging / "uploads").mkdir()

    recover_interrupted()  # 模拟重启

    assert not trash_dir.exists()
    assert not orphan.exists()
    assert not stray_staging.exists()
    assert good_dir.is_dir()  # 已发布目录不被恢复逻辑触碰
    assert client.get(f"/runs/{rid}/download/S1_R1.fastq").status_code == 200
    with db_conn(db_path()) as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_number='RUN-K'").fetchone()
    assert row["status"] == "published"


def test_restart_reclaims_processing_residue_without_touching_published(tmp_path, monkeypatch):
    """重启时:processing 遗留被标失败,其自带 token 的发布残留被认领清除,
    而已发布运行的目录原样保留(即便两个处理的清理在同一次恢复中执行)。"""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data2"))
    from fastapi.testclient import TestClient
    from app.main import app, recover_interrupted

    with TestClient(app) as c2:
        rid = run_id_of(submit_good(c2, "RUN-L"))
        assert wait_done(c2, rid)["status"] == "published"
        good = published_dir() / "RUN-L"
        good_token = (good / ATTEMPT_MARKER).read_text()

        # 伪造一个中断的 processing 尝试,并留下带其自身 token 的发布残留
        stale_staging = staging_dir() / "stale-token"
        stale_staging.mkdir(parents=True)
        residue = published_dir() / "RUN-STALE"
        residue.mkdir(parents=True)
        (residue / "S1_R1.fastq").write_text("改名后来不及落库的半成品")
        write_attempt_marker(residue, "stale-token")
        with db_conn(db_path()) as conn:
            conn.execute(
                "INSERT INTO runs(run_number, status, staging_dir, created_at, updated_at) "
                "VALUES ('RUN-STALE', 'processing', ?, 't', 't')",
                (str(stale_staging),),
            )

        recover_interrupted()  # 模拟再次重启

        assert not residue.exists()  # 本尝试自己的残留被认领清除
        assert not stale_staging.exists()
        assert good.is_dir()  # 他人(已发布)的目录不得被中断清理误删
        assert (good / ATTEMPT_MARKER).read_text() == good_token
        with db_conn(db_path()) as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_number='RUN-STALE'"
            ).fetchone()
        assert row["status"] == "failed" and "中断" in row["error"]
        assert c2.get(f"/runs/{rid}/download/S1_R1.fastq").status_code == 200
        # 恢复后同号重试可成功
        rid2 = run_id_of(submit_good(c2, "RUN-STALE"))
        assert wait_done(c2, rid2)["status"] == "published"


def test_interrupted_then_retry_runs_to_published(client):
    """进程中断(processing 遗留)重启标失败后,同号重试可完整发布。"""
    from app.main import recover_interrupted

    staging = staging_dir() / "deadbeef2"
    (staging / "out").mkdir(parents=True)
    (staging / "out" / "S1_R1.fastq").write_text("半成品")
    stray_pub = published_dir() / "RUN-M"
    stray_pub.mkdir(parents=True)
    (stray_pub / "S1_R1.fastq").write_text("改名后来不及落库的文件")
    write_attempt_marker(stray_pub, "deadbeef2")
    with db_conn(db_path()) as conn:
        conn.execute(
            "INSERT INTO runs(run_number, status, staging_dir, created_at, updated_at) "
            "VALUES ('RUN-M', 'processing', ?, 't', 't')",
            (str(staging),),
        )

    recover_interrupted()

    assert not staging.exists() and not stray_pub.exists()
    rid = run_id_of(submit_good(client, "RUN-M"))
    st = wait_done(client, rid)
    assert st["status"] == "published" and st["total_pairs"] == 7
    assert client.get(f"/runs/{rid}/download/S1_R1.fastq").status_code == 200
