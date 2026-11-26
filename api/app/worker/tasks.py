from app.services.agent_maintenance import maintain_agent_runtime
from app.services.agent_runs import run_agent_workflow, mark_agent_run_task_error
from app.services.agent_workflows import run_registered_agent_workflow, mark_registered_workflow_error
from app.services.outbox import dispatch_pending_outbox_events
from app.worker.celery_app import celery_app
from app.services.generation_jobs import run_generation_job

@celery_app.task(name="generation.run_generation_job")
def run_generation_job_task(job_id):
    run_generation_job(job_id)


@celery_app.task(bind=True, name="agent.run_product_operation", max_retries=2)
def run_agent_workflow_task(
    self,
    run_id: int,
    resume_payload: dict | None = None,
) -> None:
    """执行Agent工作流任务，把队列消息转换为可重试、可记录状态的后台任务执行。"""
    # 先处理输入与边界条件，再执行核心逻辑，最后返回稳定且可验证的结果。
    lease_owner = f"celery:{self.request.id}"
    try:
        run_agent_workflow(run_id, resume_payload, lease_owner=lease_owner)
    except Exception as exc:
        will_retry = self.request.retries < self.max_retries
        should_retry = mark_agent_run_task_error(
            run_id,
            exc,
            will_retry=will_retry,
            is_resume=resume_payload is not None,
            lease_owner=lease_owner,
        )
        if should_retry:
            raise self.retry(exc=exc, countdown=2 ** (self.request.retries + 1))
        raise


@celery_app.task(name="maintenance.dispatch_outbox")
def dispatch_outbox_events_task() -> int:
    # 发件箱投递
    return dispatch_pending_outbox_events()


@celery_app.task(bind=True, name="agent.run_registered_workflow", max_retries=2)
def run_registered_agent_workflow_task(
    self,
    run_id: int,
    resume_payload: dict | None = None,
) -> None:
    lease_owner = f"celery:{self.request.id}"
    try:
        run_registered_agent_workflow(run_id, resume_payload, lease_owner=lease_owner)
    except Exception as exc:
        will_retry = self.request.retries < self.max_retries
        should_retry = mark_registered_workflow_error(
            run_id,
            exc,
            will_retry=will_retry,
            lease_owner=lease_owner,
        )
        if should_retry:
            raise self.retry(exc=exc, countdown=2 ** (self.request.retries + 1))
        raise


@celery_app.task(name="maintenance.agent_runtime")
def maintain_agent_runtime_task() -> dict[str, int]:
    """维护Agentruntime任务，把队列消息转换为可重试、可记录状态的后台任务执行。"""
    return maintain_agent_runtime()