from app.worker.celery_app import celery_app
from app.services.generation_jobs import run_generation_job

@celery_app.task(name="generation.run_generation_job")
def run_generation_job_task(job_id):
    run_generation_job(job_id)


@celery_app.task(bind=True, name="agent.run_registered_workflow", max_retries=2)
def run_registered_agent_workflow_task(
    self,
    run_id: int,
    resume_payload: dict | None = None,
) -> None:
    pass