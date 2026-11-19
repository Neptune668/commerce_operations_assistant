import logging
import time
from datetime import UTC, datetime

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models import GenerationJob, GenerationJobStatus
from app.services.generation_jobs import dispatch_generation_job

logger = logging.getLogger(__name__)

SCAN_INTERVAL_SECONDS = 5
BATCH_SIZE = 20



def dispatch_due_generation_jobs()-> int:
    now = datetime.now(UTC)

    with SessionLocal() as db:
        # 查询所有状态为pending的GenerationJob，返回BATCH_SIZE条
        job_ids = list(
            db.scalars(
                select(GenerationJob.id)
                .where(
                    GenerationJob.status == GenerationJobStatus.PENDING,
                    (GenerationJob.next_run_at.is_(None)) | (GenerationJob.next_run_at <= now),
                )
                .order_by(GenerationJob.created_at.asc(), GenerationJob.id.asc())
                .limit(BATCH_SIZE)
            )
        )
        # 投递生成任务
        dispatched = 0
        for job_id in job_ids:
            try:
                dispatch_generation_job(job_id)
                dispatched += 1
            except Exception as e:
                pass
    return dispatched

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("媒体任务调度器启动：每 %s 秒扫描一次到期任务。", SCAN_INTERVAL_SECONDS)
    try:
        while True:
            try:
                count = dispatch_due_generation_jobs()
                if count:
                    logger.info("本轮处理 %s 条到期任务。", count)
            except Exception as exc:
                logger.warning("扫描失败，异常类型：%s；下一轮继续。", type(exc).__name__)
            time.sleep(SCAN_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        logger.info("媒体任务调度器已停止。")


if __name__ == "__main__":
    main()
