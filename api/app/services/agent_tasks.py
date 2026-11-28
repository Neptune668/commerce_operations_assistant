from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agents.harness import redact_snapshot
from app.agents.registry import WorkflowDefinition, get_workflow, is_workflow_enabled, route_intent
from app.core.config import get_settings
from app.models import (
    AgentApproval,
    AgentApprovalStatus,
    AgentConversation,
    AgentMemory,
    AgentMemoryStatus,
    AgentRiskLevel,
    AgentRun,
    AgentRunEvent,
    AgentRunStatus,
    AgentTask,
    AgentTaskStatus,
    OutboxEvent,
    Product,
    Store,
    User,
    UserRole,
    Workspace,
)
from app.schemas import AgentMemoryCreate, AgentTaskCreate

# 定义AgentRun的非终态的集合
ACTIVE_RUN_STATUSES = {
    AgentRunStatus.QUEUED,
    AgentRunStatus.RUNNING,
    AgentRunStatus.AWAITING_REVIEW,
    AgentRunStatus.RESUME_QUEUED,
}

class AgentTaskConflictError(RuntimeError):
    pass


class AgentTaskStateError(RuntimeError):
    pass


class AgentApprovalExpiredError(AgentTaskStateError):
    pass

# 解析工作流
def _resolve_workflow(db: Session, *,workspace_id: int,payload: AgentTaskCreate,) -> WorkflowDefinition:
    if payload.workflow_key:
        try:
            workflow = get_workflow(payload.workflow_key)
        except KeyError as exc:
            raise AgentTaskStateError(str(exc)) from exc
    else:
        workflow = route_intent(payload.intent or "")
    workspace = db.get(Workspace, workspace_id)
    if workspace is None or not is_workflow_enabled(workspace.settings, workflow.key):
        raise AgentTaskStateError(f"Workflow is disabled for current Workspace: {workflow.key}")
    return workflow

# 校验上下文作用域
def _validate_context_scope(db: Session,*,workspace_id: int,workflow: WorkflowDefinition,context: dict[str, Any],) -> None:
    """校验上下文作用域"""
    # 如果当前workflow需要context上下文（例如：product_id）但是没有提供，则抛出AgentTaskStateError
    if not any(context.get(key) is not None for key in workflow.required_context_any):
        joined = ", ".join(workflow.required_context_any)
        raise AgentTaskStateError(f"Workflow requires one context field: {joined}")
    # 从context中获取商品ID，如果商品id不为空，则查询商品，验证商品是否存在
    product = None
    product_id = _optional_int(context.get("product_id"))
    if product_id is not None:
        product = db.get(Product, product_id)
        if product is None or product.workspace_id != workspace_id:
            raise AgentTaskStateError(f"Product {product_id} does not exist in workspace {workspace_id}")
    # 从context中获取店铺ID，如果店铺id不为空，则查询店铺，验证店铺是否存在
    store = None
    store_id = _optional_int(context.get("store_id"))
    if store_id is not None:
        store = db.get(Store, store_id)
        if store is None or store.workspace_id != workspace_id:
            raise AgentTaskStateError("Store does not belong to current Workspace")
    if product is not None and store is not None and product.store_id != store.id:
        raise AgentTaskStateError("Product does not belong to selected Store")

# 验证对话
def _validate_conversation(db: Session, *, workspace_id: int, conversation_id: int | None) -> None:
    """校验会话；
    1.如果没有conversation_id则直接返回
    2.如果conversation_id存在，但是id对应的会话不存在，则抛出AgentTaskStateError
    3.如果对话存在，但是对话不属于当前工作区，则抛出AgentTaskStateError
    """
    if conversation_id is None:
        return
    conversation = db.get(AgentConversation, conversation_id)
    if conversation is None or conversation.workspace_id != workspace_id:
        raise AgentTaskStateError("Conversation does not belong to current Workspace")

# workspace下并发任务数校验
def _enforce_workspace_concurrency(db: Session, *, workspace_id: int) -> None:
    # 锁定 Workspace 行，把“计数 + 新建 Run”串行化，防止并发请求穿透十任务上限。
    workspace = db.scalar(
        select(Workspace).where(Workspace.id == workspace_id).with_for_update()
    )
    if workspace is None:
        raise AgentTaskStateError("Workspace does not exist")
    active_count = db.scalar(
        select(func.count(AgentRun.id)).where(
            AgentRun.workspace_id == workspace_id,
            AgentRun.status.in_(ACTIVE_RUN_STATUSES),
        )
    )
    if int(active_count or 0) >= 10:
        raise AgentTaskConflictError("Workspace has reached the 10-run concurrency limit")

# 创建AgentTask
def create_agent_task(db:Session,*, workspace_id: int, requested_by: User,payload: AgentTaskCreate)->tuple[AgentTask,AgentRun,OutboxEvent]:
    # 1.解析payload，获取对应的workflow_key
    workflow: WorkflowDefinition = _resolve_workflow(db, workspace_id=workspace_id, payload=payload)
    # 一系列校验
    settings = get_settings()
    # 2.保留负载中可回放信息
    context_scope = redact_snapshot(payload.context_scope)
    # 3.验证上下文中是否存在当前workflow所需的店铺id或商品id，并验证店铺或商品是否存在
    _validate_context_scope(db, workspace_id=workspace_id, workflow=workflow, context=context_scope)
    # 4.验证对话
    _validate_conversation(db, workspace_id=workspace_id, conversation_id=payload.conversation_id)
    # 5.workspace下并发任务数校验（最多10个）
    _enforce_workspace_concurrency(db, workspace_id=workspace_id)

    title = payload.title or workflow.title
    # 6.创建AgentTask
    task = AgentTask(
        workspace_id=workspace_id,
        conversation_id=payload.conversation_id, # 对话ID，可以为None
        requested_by_user_id=requested_by.id,   # 创建Agent任务的用户ID
        workflow_key=workflow.key,  # AgentTask绑定的工作流
        workflow_version=workflow.version,  # 工作流版本
        title=title,    # 任务标题
        intent=payload.intent,  # 任务意图，可以为None
        status=AgentTaskStatus.QUEUED, # 任务状态：初始状态为 queued
        risk_level=workflow.default_risk_level, # 任务风险等级（工作流的风险等级）
        context_scope=context_scope,    # 上下文（product_id,store_id）
        input_payload=redact_snapshot(payload.input_payload),
        result_summary={},
        version_manifest=workflow.version_manifest,
    )
    db.add(task)
    db.flush()
    # 7.创建AgentRun
    product_id = _optional_int(context_scope.get("product_id"))
    run = AgentRun(
        thread_id=str(uuid4()),  # 执行当前AgentRun对应的AgentTask中的workflow的LangGraph的  thread_id
        workspace_id=workspace_id,
        task_id=task.id,    # 关联的AgentTask ID
        product_id=product_id,  # 商品ID，可以为None
        requested_by_user_id=requested_by.id,
        workflow_key=workflow.key,
        workflow_version=workflow.version,
        attempt=1,
        queue_name=settings.celery_agent_workflow_queue,  # 默认值：agent.workflow
        status=AgentRunStatus.QUEUED,   # AgentRun的状态，初始状态为 queued （等待执行）
        revision_count=0,
        max_revisions=1,
        budget={"max_total_tokens": 12000},
        usage={},
        version_manifest=workflow.version_manifest,
        context_snapshot={},
        draft_output={},
        quality_report={},
        final_output={},
        timeout_at=datetime.now(UTC) + timedelta(minutes=30),
    )
    db.add(run)
    db.flush()
    # 3.创建AgentRunEvent
    event = AgentRunEvent(
        run_id=run.id,
        event_type="queued",
        status=AgentRunStatus.QUEUED.value,
        message=f"{workflow.title}已进入受控工作流队列。",
        output_snapshot={"workflow": workflow.key, "version": workflow.version},
    )
    db.add(event)
    db.flush()
    # 4.创建OutboxEvent
    outbox_event = _enqueue_run_event(db, run=run, resume_payload=None, suffix="start")
    return task, run, outbox_event


def _is_due(value: datetime, now: datetime) -> bool:
    """判断到期"""
    if value.tzinfo is None:
        return value <= now.replace(tzinfo=None)
    return value <= now

# 审批业务
def decide_agent_approval(
    db: Session,
    *,
    approval: AgentApproval,
    reviewer: User,
    reviewer_role: UserRole,
    decision: str,
    note: str | None,
    decision_payload: dict[str, Any],
) -> OutboxEvent | None:
    # 1.判断当前审批消息状态是否为 pending，如果不是则抛出 AgentTaskStateError
    if approval.status != AgentApprovalStatus.PENDING:
        raise AgentTaskStateError("Approval is not pending")
    # 2.判断当前审批消息是否为 R3 级别，如果是，则要求审批人必须是 Workspace Owner
    if approval.risk_level == AgentRiskLevel.R3 and reviewer_role != UserRole.OWNER:
        raise AgentTaskStateError("R3 approval requires Workspace Owner")
    # 3.判断审批消息关联的AgentTask和AgentRun是否存在
    now = datetime.now(UTC)
    task = db.get(AgentTask, approval.task_id)
    run = db.get(AgentRun, approval.run_id)
    if task is None or run is None:
        raise AgentTaskStateError("Approval target does not exist")
    # 4.检验审批消息关联的AgentTask和AgentRun是否属于当前工作区
    if (
        task.workspace_id != approval.workspace_id
        or run.workspace_id != approval.workspace_id
        or run.task_id != task.id
    ):
        raise AgentTaskStateError("Approval target is outside current Workspace")
    # 5.审批消息是否过期，如果过期则设置审批消息状态为 expired，
    #   设置关联的 AgentTask 状态为 cancelled
    #   设置关联的 AgentRun 状态为 cancelled
    if approval.expires_at is not None and _is_due(approval.expires_at, now):
        approval.status = AgentApprovalStatus.EXPIRED
        approval.decided_at = now
        approval.note = "审批已超过有效期，未执行受控动作。"
        task.status = AgentTaskStatus.CANCELLED
        task.finished_at = now
        run.status = AgentRunStatus.CANCELLED
        run.finished_at = now
        run.lease_owner = None
        run.lease_expires_at = None
        raise AgentApprovalExpiredError("Approval has expired")
    # 6.检查审批消息关联的AgentTask和AgentRun是否处于待审批状态
    if (
        task.status != AgentTaskStatus.AWAITING_APPROVAL
        or run.status != AgentRunStatus.AWAITING_REVIEW
        or run.cancel_requested_at is not None
    ):
        raise AgentTaskStateError("Approval target is no longer awaiting review")
    # 7.执行审批操作
    approval.status = ( AgentApprovalStatus.APPROVED if decision == "approve" else AgentApprovalStatus.REJECTED  )
    approval.decided_by_user_id = reviewer.id
    approval.decision_payload = redact_snapshot(decision_payload)
    approval.note = note
    approval.decided_at = now
    # 8.如果审批决定是 reject，则取消关联的 AgentTask 和 AgentRun
    if decision == "reject":
        task.status = AgentTaskStatus.CANCELLED
        task.finished_at = now
        run.status = AgentRunStatus.REJECTED
        run.finished_at = now
        run.final_output = {"decision": "reject", "note": note}
        db.add(
            AgentRunEvent(
                run_id=run.id,
                event_type="approval_rejected",
                status=AgentRunStatus.REJECTED.value,
                message=note or "审批已拒绝。",
            )
        )
        return None

    # 9.如果审批决定是 approve，则恢复关联的 AgentTask 和 AgentRun
    task.status = AgentTaskStatus.QUEUED
    run.status = AgentRunStatus.RESUME_QUEUED
    db.add(
        AgentRunEvent(
            run_id=run.id,
            event_type="approval_accepted",
            status=AgentRunStatus.RESUME_QUEUED.value,
            message="审批已通过，等待恢复工作流。",
        )
    )
    # resume_payload = {"decision": "approve", **redact_snapshot(decision_payload)}
    resume_decision = "edit_and_approve" if decision_payload.get("edited_output") else decision
    resume_payload = {"decision": resume_decision, **redact_snapshot(decision_payload)}
    return _enqueue_run_event(db, run=run, resume_payload=resume_payload, suffix=f"approval-{approval.id}")

# 取消任务
def cancel_agent_task(db: Session, *, task: AgentTask) -> None:
    if task.status in {AgentTaskStatus.COMPLETED, AgentTaskStatus.FAILED, AgentTaskStatus.CANCELLED}:
        raise AgentTaskStateError("Agent task is already terminal")
    now = datetime.now(UTC)
    task.status = AgentTaskStatus.CANCELLED
    task.cancelled_at = now
    task.finished_at = now
    runs = db.scalars(select(AgentRun).where(AgentRun.task_id == task.id)).all()
    for run in runs:
        if run.status in ACTIVE_RUN_STATUSES:
            run.cancel_requested_at = now
            if run.status != AgentRunStatus.RUNNING:
                run.status = AgentRunStatus.CANCELLED
                run.finished_at = now
            db.add(
                AgentRunEvent(
                    run_id=run.id,
                    event_type="cancel_requested",
                    status=run.status.value,
                    message="用户请求取消 Agent Task。",
                )
            )
    approvals = db.scalars(
        select(AgentApproval).where(
            AgentApproval.task_id == task.id,
            AgentApproval.status == AgentApprovalStatus.PENDING,
        )
    ).all()
    for approval in approvals:
        approval.status = AgentApprovalStatus.CANCELLED
        approval.decided_at = now
        approval.note = "任务已取消，待处理审批同步关闭。"


# 创建记忆
def create_agent_memory(
    db: Session,
    *,
    workspace_id: int,
    created_by: User,
    payload: AgentMemoryCreate,
) -> AgentMemory:
    """创建Agent记忆；只交换结构化 Agent 状态或受控工具结果，支持 checkpoint、审计与恢复。"""
    # 节点只通过结构化状态和受控工具交换数据，禁止绕过策略边界写入结果。
    effective_at = payload.effective_at or datetime.now(UTC)
    if payload.expires_at is not None and _is_due(payload.expires_at, effective_at):
        raise AgentTaskStateError("Memory expires_at must be later than effective_at")
    _validate_memory_scope(
        db,
        workspace_id=workspace_id,
        scope_type=payload.scope_type,
        scope_id=payload.scope_id,
        user_id=created_by.id,
    )
    _validate_memory_source(
        db,
        workspace_id=workspace_id,
        source_type=payload.source_type,
        source_reference=payload.source_reference,
    )
    memory = AgentMemory(
        workspace_id=workspace_id,
        created_by_user_id=created_by.id,
        scope_type=payload.scope_type,
        scope_id=payload.scope_id,
        memory_key=payload.memory_key,
        schema_version=payload.schema_version,
        content=redact_snapshot(payload.content),
        source_type=payload.source_type,
        source_reference=redact_snapshot(payload.source_reference),
        status=AgentMemoryStatus.ACTIVE,
        effective_at=effective_at,
        expires_at=payload.expires_at,
    )
    db.add(memory)
    db.flush()
    return memory

# 删除记忆
def revoke_agent_memory(memory: AgentMemory) -> None:
    """撤销Agent记忆；只交换结构化 Agent 状态或受控工具结果，支持 checkpoint、审计与恢复。"""
    if memory.status != AgentMemoryStatus.ACTIVE:
        raise AgentTaskStateError("Agent memory is not active")
    memory.status = AgentMemoryStatus.REVOKED
    memory.revoked_at = datetime.now(UTC)

def _validate_memory_scope(
    db: Session,
    *,
    workspace_id: int,
    scope_type: str,
    scope_id: str | None,
    user_id: int | None = None,
) -> None:
    """校验记忆作用域"""
    if scope_type == "workspace":
        if scope_id not in {None, str(workspace_id)}:
            raise AgentTaskStateError("Workspace memory scope does not match current Workspace")
        return
    if scope_id is None:
        raise AgentTaskStateError(f"{scope_type} memory requires scope_id")
    if scope_type == "product":
        product = db.get(Product, int(scope_id))
        if product is None or product.workspace_id != workspace_id:
            raise AgentTaskStateError("Product memory scope is outside current Workspace")
    elif scope_type == "store":
        store = db.get(Store, int(scope_id))
        if store is None or store.workspace_id != workspace_id:
            raise AgentTaskStateError("Store memory scope is outside current Workspace")
    elif scope_type == "user":
        if user_id is None or scope_id != str(user_id):
            raise AgentTaskStateError("User memory can only target the current user")

def _validate_memory_source(
    db: Session,
    *,
    workspace_id: int,
    source_type: str,
    source_reference: dict[str, Any],
) -> None:
    """校验记忆source"""
    if source_type != "approved_task":
        return
    task_id = _optional_int(source_reference.get("task_id"))
    if task_id is None:
        raise AgentTaskStateError("approved_task memory requires source_reference.task_id")
    task = db.get(AgentTask, task_id)
    if (
        task is None
        or task.workspace_id != workspace_id
        or task.status != AgentTaskStatus.COMPLETED
    ):
        raise AgentTaskStateError("Memory source task is not an approved completed task")



# 创建OutboxEvent
def _enqueue_run_event(db: Session,*,run: AgentRun,resume_payload: dict[str, Any] | None,suffix: str,) -> OutboxEvent:
    """enqueue执行事件"""
    event = OutboxEvent(
        workspace_id=run.workspace_id,
        aggregate_type="agent_run",
        aggregate_id=str(run.id),
        event_type="workflow.run.requested",
        payload={"run_id": run.id, "resume_payload": resume_payload},
        idempotency_key=f"agent-run:{run.id}:{suffix}",
        status="pending",
        attempts=0,
        available_at=datetime.now(UTC),
    )
    db.add(event)
    db.flush()
    return event

# 辅助函数：将value转换成int
def _optional_int(value: Any) -> int | None:
    """可选int"""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AgentTaskStateError("Context identifiers must be integers") from exc