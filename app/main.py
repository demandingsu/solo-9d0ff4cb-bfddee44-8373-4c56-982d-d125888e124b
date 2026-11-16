"""泳道拆样台:上传 R1/R2/I1 与样本索引 CSV,按样本拆出成对读段。

发布模型:
- 处理中:上传件与半成品输出都在 staging/<token>/,不对外可见。每次尝试有
  自己的随机 token,暂存目录彼此隔离。
- 成功:输出目录(含本尝试的 .attempt 归属标记)整体 os.rename 到
  published/<运行号>/,同事务写入计数,之后状态置为 published。
  任何一步失败都不会留下半套已发布文件。
- 失败/中断:状态置 failed,只清理**本尝试**的残留——暂存目录按 token 隔离
  可直接删;发布目录经 reclaim_published 凭归属标记认领,属于同号重试新尝试
  的已发布文件一律原样还原,绝不误删。同一运行号可重试。
- 已发布:同一运行号拒绝改写。
"""
import os
import re
import shutil
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .config import db_path, published_dir, staging_dir
from .db import db_conn, init_db
from .demux import demultiplex
from .samples import load_samples

RUN_NUMBER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
STATUS_LABELS = {"processing": "处理中", "published": "已发布", "failed": "失败"}
UNDETERMINED = "undetermined"
ATTEMPT_MARKER = ".attempt"  # 发布目录内的尝试归属标记,内容为暂存 token
TRASH_PREFIX = ".trash-"     # 认领发布目录时临时挪入的隔离目录前缀


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def attempt_token(staging: Path) -> str:
    """一次处理尝试的身份:暂存目录名(随机 token)。"""
    return Path(staging).name


def write_attempt_marker(out_dir: Path, token: str) -> None:
    """在半成品输出目录写入归属标记,随 os.rename 一并原子进入发布区。"""
    (Path(out_dir) / ATTEMPT_MARKER).write_text(token, encoding="ascii")


def _remove(path) -> None:
    p = Path(path)
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p, ignore_errors=True)
    else:
        try:
            p.unlink()
        except (FileNotFoundError, NotADirectoryError):
            pass


def reclaim_published(final: Path, token: str) -> None:
    """失败/中断清理时,只回收**本尝试**可能留下的发布目录。

    同号重试的新尝试可能已经把自己的输出原子改名到同一 final 路径。为避免
    "旧尝试删新尝试文件"的交错事故,这里先把目标整体挪到同卷的 .trash-<token>
    隔离目录(同卷 rename 原子),再凭 .attempt 归属标记判定:
    - 标记属于本 token,或目录无标记(旧版本遗留的无主半成品):确认是本尝试
      的残留,安全删除;
    - 标记属于其他尝试:这是新尝试的已发布文件,原样还原,绝不删除;
    - 还原失败时宁可留在 .trash 里(启动恢复会统一清扫),也不能误删他人文件。
    """
    final = Path(final)
    if not final.exists() and not final.is_symlink():
        return
    trash = _trash_path(final, token)
    try:
        os.rename(final, trash)  # 同一数据卷内原子挪走,检查与删除之间无人能改归属
    except FileNotFoundError:
        return
    owner = _read_owner(trash)
    if owner == token or owner == "":
        _remove(trash)
        return
    try:  # 属于其他尝试(同号重试的新尝试):完整还原
        os.rename(trash, final)
    except OSError:
        # 极端情况下还原不回原路径;保留在隔离目录交由启动恢复处理,
        # 绝不能 rmtree 掉不属于本尝试的文件。
        pass


def adopt_stale_publish(final: Path, token: str, run_id: int) -> None:
    """新尝试发布前,清掉同路径上**确认无主**的旧残留。

    能走到这里说明本运行号当前只有本尝试这一行 processing 记录(run_number
    唯一;已发布行会在提交时被 409 挡下)。因此路径上若已有目录,只可能是
    已被替换的旧尝试来不及回收的残留:挪走后凭标记再确认一次归属,
    只要发现仍有别的存活处理行就原样还原,绝不强删。
    """
    final = Path(final)
    if not final.exists() and not final.is_symlink():
        return
    trash = _trash_path(final, token)
    try:
        os.rename(final, trash)
    except FileNotFoundError:
        return
    owner = _read_owner(trash)
    if owner == token:  # 自己的输出不应已在发布路径,还原并交上层报错
        try:
            os.rename(trash, final)
        except OSError:
            pass
        raise RuntimeError(f"发布目录已存在且属于当前尝试: {final}")
    with db_conn(db_path()) as conn:
        other_live = conn.execute(
            "SELECT 1 FROM runs WHERE run_number=? AND id<>? AND status='processing' LIMIT 1",
            (final.name, run_id),
        ).fetchone()
    if other_live:  # 理论上被唯一约束挡死,保守处理:原样还原
        os.rename(trash, final)
        raise RuntimeError(f"发布目录正被其他尝试占用: {final}")
    _remove(trash)


def _trash_path(final: Path, token: str) -> Path:
    trash = final.parent / f"{TRASH_PREFIX}{token}"
    n = 1
    while trash.exists() or trash.is_symlink():
        trash = final.parent / f"{TRASH_PREFIX}{token}-{n}"
        n += 1
    return trash


def _read_owner(trash: Path) -> str:
    try:
        return (trash / ATTEMPT_MARKER).read_text(encoding="ascii").strip()
    except (FileNotFoundError, UnicodeDecodeError, OSError):
        return ""


def recover_interrupted() -> None:
    """启动恢复:凡未走到 published 的运行一律标失败并清掉半成品,
    保证发布区里只存在完整运行的完整文件。

    启动时本进程没有任何存活的处理线程,因此:
    - .trash-* 隔离目录都是进程被杀时认领流程的遗留,可无条件清扫;
    - 暂存目录与发布区孤儿目录同理,此时不存在"新尝试在用"的可能。
    """
    with db_conn(db_path()) as conn:
        stale = conn.execute(
            "SELECT id, run_number, staging_dir FROM runs WHERE status = 'processing'"
        ).fetchall()
        for row in stale:
            conn.execute(
                "UPDATE runs SET status='failed', error=?, updated_at=? "
                "WHERE id=? AND status='processing'",
                ("进程中断,未完成即停止;可用同一运行号重新提交", utcnow(), row["id"]),
            )
        published = {
            r["run_number"]
            for r in conn.execute("SELECT run_number FROM runs WHERE status='published'")
        }
    for row in stale:
        _remove(row["staging_dir"])
        # 仅认领本尝试(凭暂存 token)的发布残留;他人的目录会被原样还原
        reclaim_published(published_dir() / row["run_number"], attempt_token(row["staging_dir"]))
    if staging_dir().exists():
        for child in staging_dir().iterdir():  # 无主的暂存目录一并清理
            _remove(child)
    pub = published_dir()
    if pub.exists():
        for child in list(pub.iterdir()):
            if child.name.startswith(TRASH_PREFIX):
                _remove(child)  # 启动期无存活线程,隔离目录全是无主遗留
            elif child.name not in published:
                _remove(child)
    with db_conn(db_path()) as conn:  # 已发布但文件丢失:不允许提供残缺下载
        for num in published:
            if not (published_dir() / num).is_dir():
                conn.execute(
                    "UPDATE runs SET status='failed', error=?, updated_at=? "
                    "WHERE run_number=? AND status='published'",
                    ("已发布文件缺失,运行标记为失败", utcnow(), num),
                )


def setup() -> None:
    staging_dir().mkdir(parents=True, exist_ok=True)
    published_dir().mkdir(parents=True, exist_ok=True)
    init_db(db_path())
    recover_interrupted()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup()
    yield


app = FastAPI(title="泳道拆样台", lifespan=lifespan)


# ---------------------------------------------------------------- 后台处理

def write_log(path: Path, run, sample_list, stats) -> None:
    lines = [
        f"运行号: {run['run_number']}",
        f"提交时间: {run['created_at']}",
        f"完成时间: {utcnow()}",
        f"输入文件: R1={run['r1_name']} R2={run['r2_name']} "
        f"I1={run['i1_name']} 样本表={run['samples_name']}",
        "样本索引:",
    ]
    lines += [f"  {sid}\t{idx}" for sid, idx in sample_list]
    lines.append(f"总读段对: {stats['total_pairs']}")
    lines.append("分组对数(对数\t其中单碱基纠错):")
    for sid in stats["order"]:
        pairs, corrected = stats["counts"][sid]
        lines.append(f"  {sid}\t{pairs}\t{corrected}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_run(run_id: int) -> None:
    with db_conn(db_path()) as conn:
        run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if run is None:
        return
    run_number = run["run_number"]
    staging = Path(run["staging_dir"])
    token = attempt_token(staging)
    uploads = staging / "uploads"
    out = staging / "out"
    final = published_dir() / run_number
    try:
        sample_list = load_samples(uploads / "samples.csv")
        out.mkdir(parents=True)
        stats = demultiplex(
            str(uploads / "r1.fastq"), str(uploads / "r2.fastq"),
            str(uploads / "i1.fastq"), sample_list, str(out),
        )
        write_log(out / "demux.log", run, sample_list, stats)
        write_attempt_marker(out, token)  # 归属标记随输出一起原子进入发布区
        # 同路径若残留旧尝试的发布目录,凭标记确认无主后清走;有主则报错不抢
        adopt_stale_publish(final, token, run_id)
        os.rename(out, final)  # 同一数据卷内原子移动:先完整落盘,再发布
        with db_conn(db_path()) as conn:
            updated = conn.execute(
                "UPDATE runs SET status='published', total_pairs=?, published_at=?, "
                "updated_at=? WHERE id=? AND status='processing'",
                (stats["total_pairs"], utcnow(), utcnow(), run_id),
            ).rowcount
            if updated != 1:
                raise RuntimeError("运行状态已变化,放弃发布")
            index_of = dict(sample_list)
            for sid in stats["order"]:
                pairs, corrected = stats["counts"][sid]
                conn.execute(
                    "INSERT INTO sample_counts(run_id, sample_id, index_seq, pairs, "
                    "corrected) VALUES (?,?,?,?,?)",
                    (run_id, sid, index_of.get(sid, ""), pairs, corrected),
                )
        _remove(staging)
    except Exception as exc:  # 任何失败:标失败 + 清本尝试的半成品,不发布
        with db_conn(db_path()) as conn:
            # 仅当本行仍是 processing 时才落失败:同号重试已删行/改状态时,
            # 旧尝试不得覆盖新尝试的状态
            conn.execute(
                "UPDATE runs SET status='failed', error=?, updated_at=? "
                "WHERE id=? AND status='processing'",
                (str(exc), utcnow(), run_id),
            )
        _remove(staging)  # 暂存目录按 token 隔离,只可能是本尝试自己的
        # 发布目录须凭归属标记认领:若同号重试的新尝试已在此发布,原样保留
        reclaim_published(final, token)


# ---------------------------------------------------------------- 页面与接口

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if "text/html" in request.headers.get("accept", ""):
        return templates.TemplateResponse(
            request, "error.html",
            {"status": exc.status_code, "detail": exc.detail},
            status_code=exc.status_code,
        )
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    with db_conn(db_path()) as conn:
        runs = conn.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
    return templates.TemplateResponse(
        request, "index.html", {"runs": runs, "labels": STATUS_LABELS}
    )


async def save_upload(upload: UploadFile, dest: Path) -> None:
    with open(dest, "wb") as fh:
        while chunk := await upload.read(1 << 20):
            fh.write(chunk)


@app.post("/runs")
async def create_run(
    run_number: str = Form(...),
    r1: UploadFile = File(...),
    r2: UploadFile = File(...),
    i1: UploadFile = File(...),
    samples: UploadFile = File(...),
):
    run_number = run_number.strip()
    if not RUN_NUMBER_RE.fullmatch(run_number):
        raise HTTPException(400, "运行号格式不正确:字母或数字开头,可含 - _ .,最长 64 字符")
    for label, up in (("R1", r1), ("R2", r2), ("I1", i1), ("样本表", samples)):
        if not up.filename:
            raise HTTPException(400, f"请上传{label}文件")

    token = uuid.uuid4().hex
    staging = staging_dir() / token
    uploads = staging / "uploads"
    names = {k: Path(u.filename or "").name for k, u in
             (("r1", r1), ("r2", r2), ("i1", i1), ("samples", samples))}
    try:
        with db_conn(db_path()) as conn:
            row = conn.execute(
                "SELECT id, status FROM runs WHERE run_number=?", (run_number,)
            ).fetchone()
            if row and row["status"] == "published":
                raise HTTPException(409, f"运行 {run_number} 已发布,不可改写")
            if row and row["status"] == "processing":
                raise HTTPException(409, f"运行 {run_number} 正在处理中,请稍后查看")
            if row:  # failed:允许同号重试
                conn.execute("DELETE FROM runs WHERE id=?", (row["id"],))
            cur = conn.execute(
                "INSERT INTO runs(run_number, status, staging_dir, r1_name, r2_name, "
                "i1_name, samples_name, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (run_number, "processing", str(staging), names["r1"], names["r2"],
                 names["i1"], names["samples"], utcnow(), utcnow()),
            )
            run_id = cur.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(409, f"运行 {run_number} 已存在,请刷新后查看")

    try:
        uploads.mkdir(parents=True)
        await save_upload(r1, uploads / "r1.fastq")
        await save_upload(r2, uploads / "r2.fastq")
        await save_upload(i1, uploads / "i1.fastq")
        await save_upload(samples, uploads / "samples.csv")
        load_samples(uploads / "samples.csv")  # 先校验样本表,尽早报错
    except HTTPException:
        raise
    except Exception as exc:
        with db_conn(db_path()) as conn:
            conn.execute(
                "UPDATE runs SET status='failed', error=?, updated_at=? WHERE id=?",
                (f"上传或样本表错误: {exc}", utcnow(), run_id),
            )
        _remove(staging)
        raise HTTPException(400, f"上传或样本表错误: {exc}")

    threading.Thread(target=process_run, args=(run_id,), daemon=True).start()
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


def get_run_or_404(run_id: int):
    with db_conn(db_path()) as conn:
        run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if run is None:
        raise HTTPException(404, "运行不存在")
    return run


def get_counts(run_id: int):
    with db_conn(db_path()) as conn:
        return conn.execute(
            "SELECT * FROM sample_counts WHERE run_id=? "
            "ORDER BY sample_id = 'undetermined', sample_id",
            (run_id,),
        ).fetchall()


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: int):
    run = get_run_or_404(run_id)
    groups = [
        {
            "sample_id": c["sample_id"],
            "index_seq": c["index_seq"],
            "pairs": c["pairs"],
            "corrected": c["corrected"],
            "r1": f"{c['sample_id']}_R1.fastq",
            "r2": f"{c['sample_id']}_R2.fastq",
        }
        for c in get_counts(run_id)
    ]
    return templates.TemplateResponse(
        request, "run.html", {"run": run, "groups": groups, "labels": STATUS_LABELS}
    )


@app.get("/api/runs/{run_id}")
def run_status(run_id: int):
    run = get_run_or_404(run_id)
    counts = get_counts(run_id)
    return {
        "run_number": run["run_number"],
        "status": run["status"],
        "error": run["error"],
        "total_pairs": run["total_pairs"],
        "counts": [
            {"sample_id": c["sample_id"], "index_seq": c["index_seq"],
             "pairs": c["pairs"], "corrected": c["corrected"]}
            for c in counts
        ],
    }


@app.get("/runs/{run_id}/download/{filename}")
def download(run_id: int, filename: str):
    run = get_run_or_404(run_id)
    if run["status"] != "published":
        raise HTTPException(409, "运行未发布,没有完整文件可下载")
    valid = {"demux.log"}
    for c in get_counts(run_id):
        valid.add(f"{c['sample_id']}_R1.fastq")
        valid.add(f"{c['sample_id']}_R2.fastq")
    if filename not in valid:
        raise HTTPException(404, "文件不存在")
    path = published_dir() / run["run_number"] / filename
    if not path.is_file():
        raise HTTPException(404, "文件不存在")
    return FileResponse(path, filename=filename, media_type="application/octet-stream")
