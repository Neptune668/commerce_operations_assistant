from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models import CreativePlan, GenerationJob, GenerationJobEvent, GenerationJobKind, GenerationJobStatus
from app.services.generated_assets import create_generated_asset
from app.services.image_generation import generate_main_image_asset
from app.services.video_generation import generate_video_asset


# 默认最大尝试次数，包含第一次执行
DEFAULT_MAX_ATTEMPTS = 3
# 默认运行超时阈值，超过后由调度器回收
DEFAULT_RUNNING_TIMEOUT_MINUTES = 30


def enqueue_generation_job(
    db: Session,
    product_id: int,
    creative_plan_id: int,
    kind: GenerationJobKind,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> GenerationJob:
    """创建生成任务记录，并写入第一条 queued 事件。"""
    job = GenerationJob(
        product_id=product_id,
        creative_plan_id=creative_plan_id,
        kind=kind,
        status=GenerationJobStatus.PENDING,
        result={},
        attempts=0,
        max_attempts=max_attempts,
        next_run_at=datetime.now(UTC),
    )
    db.add(job)
    record_generation_job_event(
        job,
        "queued",
        message="生成任务已入队。",
        payload={"kind": kind.value, "max_attempts": max_attempts},
    )
    return job


def retry_generation_job_record(job: GenerationJob) -> None:
    """清理上一次执行的痕迹，让同一个任务 ID 重新回到待执行状态。"""
    # 保留任务 ID 与历史事件，页面与审计都能串起完整的生命周期。
    job.status = GenerationJobStatus.PENDING
    job.result = {}
    job.error = None
    job.locked_at = None
    job.locked_by = None
    job.started_at = None
    job.finished_at = None
    job.next_run_at = datetime.now(UTC)
    record_generation_job_event(
        job,
        "retry_requested",
        message="用户请求重试生成任务。",
        payload={"next_run_at": job.next_run_at.isoformat()},
    )


def cancel_generation_job_record(job: GenerationJob, *, reason: str | None = None) -> bool:
    """取消尚未结束的任务，返回是否取消成功。"""
    # 已成功或已失败的记录是历史事实，不做状态回滚。
    if job.status not in {GenerationJobStatus.PENDING, GenerationJobStatus.RUNNING}:
        return False
    job.status = GenerationJobStatus.CANCELLED
    job.error = reason or "任务已取消。"
    job.locked_at = None
    job.locked_by = None
    job.next_run_at = None
    job.finished_at = datetime.now(UTC)
    record_generation_job_event(
        job,
        "cancelled",
        message=job.error,
        payload={"reason": reason or "user_cancelled"},
    )
    return True


def mark_stale_running_generation_jobs(
    db: Session,
    *,
    timeout_minutes: int = DEFAULT_RUNNING_TIMEOUT_MINUTES,
    now: datetime | None = None,
) -> int:
    """回收运行超时的任务，按重试策略重新排队或直接失败，返回处理条数。"""
    current_time = now or datetime.now(UTC)
    deadline = current_time - timedelta(minutes=timeout_minutes)
    jobs = db.scalars(
        select(GenerationJob).where(
            GenerationJob.status == GenerationJobStatus.RUNNING,
            GenerationJob.started_at.is_not(None),
            GenerationJob.started_at <= deadline,
        )
    ).all()
    for job in jobs:
        # 超时视为执行链路中断：释放锁并按重试策略处理，避免长期卡在 running。
        _mark_failed_or_retry(job, RuntimeError(f"生成任务运行超过 {timeout_minutes} 分钟，已按超时处理。"))
        job.locked_at = None
        job.locked_by = None
        job.finished_at = current_time
        record_generation_job_event(
            job,
            "timeout_checked",
            message="调度器已处理运行超时任务。",
            payload={"timeout_minutes": timeout_minutes},
        )
    db.commit()
    return len(jobs)


def dispatch_generation_job(job_id: int, background_tasks: object | None = None) -> None:
    """按配置选择执行方式：投递到 Celery 队列，或交给后台任务。"""
    settings = get_settings()
    if settings.task_queue_backend == "celery":
        # 耗时生成交给独立 Worker，避免阻塞 Web 进程。
        from app.worker.tasks import run_generation_job_task
        run_generation_job_task.delay(job_id)
        return

    if background_tasks is not None and hasattr(background_tasks, "add_task"):
        # 本地与轻量部署使用后台任务，保持部署简单。
        background_tasks.add_task(run_generation_job, job_id)
        return
    # 兜底
    run_generation_job(job_id)


def process_next_generation_job(worker_id: str = "fastapi-background") -> int | None:
    """取出一条最早可执行的待执行任务并执行，返回任务 ID。"""
    now = datetime.now(UTC)
    with SessionLocal() as db:
        # 单次只取一条，便于用定时器或单 Worker 轮询时控制并发与重试节奏。
        statement = (
            select(GenerationJob)
            .where(
                GenerationJob.status == GenerationJobStatus.PENDING,
                (GenerationJob.next_run_at.is_(None)) | (GenerationJob.next_run_at <= now),
            )
            .order_by(GenerationJob.created_at.asc())
            .limit(1)
        )
        job = db.scalars(statement).first()
        if job is None:
            return None
        job_id = job.id

    run_generation_job(job_id, worker_id=worker_id)
    return job_id


def run_generation_job(job_id: int, worker_id: str = "fastapi-background") -> None:
    """执行一次生成任务：加锁、检查状态、调用媒体服务并记录结果。"""
    with SessionLocal() as db:
        # 行锁让 API 后台、调度器与 Worker 的重复消息只能有一个真正开始执行。
        job = db.get(GenerationJob, job_id, with_for_update=True)
        # 任务可能已被取消、重试或被其他 Worker 抢先处理，非 pending 直接返回。
        if job is None or job.status != GenerationJobStatus.PENDING:
            return

        # 部分驱动会返回不带时区的时间，统一按 UTC 解释后再比较。
        due_at = job.next_run_at
        if due_at is not None and due_at.tzinfo is None:
            due_at = due_at.replace(tzinfo=UTC)
        if due_at is not None and due_at > datetime.now(UTC):
            return

        _mark_running(job, worker_id)
        db.commit()

        try:
            creative_plan = db.scalars(
                select(CreativePlan).where(CreativePlan.id == job.creative_plan_id)
            ).first()
            if creative_plan is None:
                raise RuntimeError("Creative plan not found.")

            asset = _generate_asset(db, job, creative_plan)
            job.result = asset
            job.status = GenerationJobStatus.SUCCEEDED
            job.error = None
            job.next_run_at = None
            record_generation_job_event(
                job,
                "succeeded",
                message="生成任务执行成功。",
                payload={"result_url": asset.get("url"), "attempts": job.attempts},
                worker_id=worker_id,
            )
        except Exception as exc:
            _mark_failed_or_retry(job, exc)
        finally:
            job.locked_at = None
            job.locked_by = None
            job.finished_at = datetime.now(UTC)
            db.commit()


def _mark_running(job: GenerationJob, worker_id: str) -> None:
    """把任务切换到 running，记录本次尝试与执行者。"""
    job.status = GenerationJobStatus.RUNNING
    job.attempts += 1
    job.started_at = datetime.now(UTC)
    job.finished_at = None
    job.locked_at = job.started_at
    job.locked_by = worker_id
    record_generation_job_event(
        job,
        "started",
        message="worker 已开始执行生成任务。",
        payload={"attempts": job.attempts},
        worker_id=worker_id,
    )


def _generate_asset(db: Session, job: GenerationJob, creative_plan: CreativePlan) -> dict:
    """按任务种类调用对应的媒体服务，并把结果写入方案内容与素材记录。"""
    if job.kind == GenerationJobKind.MAIN_IMAGE_ASSET:
        asset = generate_main_image_asset(creative_plan)
        _append_generated_asset(creative_plan, "generated_images", asset)
        create_generated_asset(db, creative_plan, "image", asset, job.id)
        return asset
    if job.kind == GenerationJobKind.VIDEO_ASSET:
        asset = generate_video_asset(creative_plan)
        _append_generated_asset(creative_plan, "generated_videos", asset)
        create_generated_asset(db, creative_plan, "video", asset, job.id)
        return asset
    raise RuntimeError(f"Unsupported generation job kind: {job.kind}")


def _mark_failed_or_retry(job: GenerationJob, exc: Exception) -> None:
    """记录失败原因，未达上限时安排重试，达到上限时置为失败。"""
    job.error = str(exc)[:2000]
    if job.attempts < job.max_attempts:
        # 供应商偶发超时较常见，采用短退避重试，最长退避 10 分钟。
        job.status = GenerationJobStatus.PENDING
        job.next_run_at = datetime.now(UTC) + timedelta(minutes=min(job.attempts * 2, 10))
        record_generation_job_event(
            job,
            "retry_scheduled",
            message=job.error,
            payload={"attempts": job.attempts, "next_run_at": job.next_run_at.isoformat()},
        )
        return
    job.status = GenerationJobStatus.FAILED
    job.next_run_at = None
    record_generation_job_event(
        job,
        "failed",
        message=job.error,
        payload={"attempts": job.attempts},
    )


def _append_generated_asset(creative_plan: CreativePlan, key: str, asset: dict) -> None:
    """把素材字典追加到方案内容的指定列表中，最新结果排在最前面。"""
    content = dict(creative_plan.content or {})
    existing = content.get(key, [])
    if not isinstance(existing, list):
        existing = []
    content[key] = [asset, *existing]
    creative_plan.content = content


def record_generation_job_event(
    job: GenerationJob,
    event_type: str,
    *,
    message: str | None = None,
    payload: dict | None = None,
    worker_id: str | None = None,
) -> GenerationJobEvent:
    """写入一条任务事件，记录当时的状态、说明与结构化细节。"""
    # 事件只保存安全摘要：不写入供应商密钥，也不保存完整的外部响应。
    event = GenerationJobEvent(
        event_type=event_type,
        status=job.status,
        message=message,
        payload=payload or {},
        worker_id=worker_id,
    )
    job.events.append(event)
    return event