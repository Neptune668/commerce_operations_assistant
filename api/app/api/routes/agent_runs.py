from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.agents.contracts import ReviewResumePayload
from app.api.routes.auth import require_edit_permission, require_view_permission
from app.db.session import get_db
from app.models import AgentRun, Product, User
from app.schemas import AgentRunRead, AgentRunReviewCreate
from app.services.agent_runs import (
    AgentRunConflictError,
    AgentRunStateError,
    create_agent_run_record,
    dispatch_agent_run,
    mark_agent_run_dispatch_failed,
    submit_agent_review,
)


router = APIRouter(tags=["agent-runs"])
SessionDep = Annotated[Session, Depends(get_db)]


@router.post("/products/{product_id}/agent-runs",response_model=AgentRunRead,status_code=status.HTTP_202_ACCEPTED,)
def create_agent_run(
    product_id: int,
    db: SessionDep,
    current_user: Annotated[User, Depends(require_edit_permission)],
) -> AgentRun:
    # 1.查询商品信息并校验 商品归属的workspace_id与当前用户的workspace_id是否一致
    product = db.get(Product, product_id)
    workspace_id = db.info.get("workspace_id")
    if product is None or (workspace_id is not None and product.workspace_id != workspace_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
    run_id: int | None = None
    # 2.创建任务
    try:
        run = create_agent_run_record(db, product=product, requested_by=current_user)
        db.commit()
        run_id = run.id
        # 3.分发任务
        dispatch_agent_run(run_id)
    except AgentRunConflictError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        if run_id is not None:
            mark_agent_run_dispatch_failed(run_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Agent task could not be queued: {exc}",
        ) from exc
    return _get_run_or_404(db, run_id)


# 根据product_id查询与之关联的AgentRun
@router.get("/products/{product_id}/agent-runs", response_model=list[AgentRunRead])
def list_product_agent_runs(product_id: int,db: SessionDep,_current_user: Annotated[User, Depends(require_view_permission)],limit: Annotated[int, Query(ge=1, le=50)] = 10) -> list[AgentRun]:
    # 1.根据商品ID查询商品信息
    product = db.get(Product, product_id)
    # 2.校验商品归属的workspace_id与当前用户的workspace_id是否一致
    workspace_id = db.info.get("workspace_id")
    if product is None or (workspace_id is not None and product.workspace_id != workspace_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
    # 3.根据run_id查询AgentRun信息
    statement = (
        select(AgentRun)
        .where(AgentRun.product_id == product_id)
        .options(selectinload(AgentRun.events), selectinload(AgentRun.reviews))
        .order_by(AgentRun.id.desc())
        .limit(limit)
    )
    return list(db.scalars(statement).all())


@router.get("/agent-runs/{run_id}", response_model=AgentRunRead)
def get_agent_run(
    run_id: int,
    db: SessionDep,
    _current_user: Annotated[User, Depends(require_view_permission)],
) -> AgentRun:
    """处理获取Agent执行接口；完成身份、Workspace、资源归属和输入校验后返回结构化结果。"""
    return _get_run_or_404(db, run_id)


@router.post(
    "/agent-runs/{run_id}/review",
    response_model=AgentRunRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def review_agent_run(
    run_id: int,
    payload: AgentRunReviewCreate,
    db: SessionDep,
    current_user: Annotated[User, Depends(require_edit_permission)],
) -> AgentRun:
    # 1.根据run_id查询AgentRun(审批的任务的AgentRun)
    run = _get_run_or_404(db, run_id)
    # 2.从前端提交的审批信息的载荷中获取并校验 审批信息
    resume_payload = ReviewResumePayload.model_validate(payload.model_dump(mode="json"))
    try:
        # 将AgentRun的状态设置为审批中（RESUME_QUEUED）
        submit_agent_review(db, run=run, reviewer=current_user, payload=resume_payload)
        db.commit()
        # 分发审批任务: 如果AgentRun没有关联AgentTask，走直投
        dispatch_agent_run(run_id, resume_payload.model_dump(mode="json"))
    except AgentRunStateError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        mark_agent_run_dispatch_failed(run_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Agent resume task could not be queued: {exc}",
        ) from exc
    return _get_run_or_404(db, run_id)


def _get_run_or_404(db: Session, run_id: int) -> AgentRun:
    """处理获取执行or404接口；完成身份、Workspace、资源归属和输入校验后返回结构化结果。"""
    # 先完成认证、Workspace 与资源归属校验，再进入会产生副作用的服务调用。
    statement = (
        select(AgentRun)
        .where(AgentRun.id == run_id)
        .options(selectinload(AgentRun.events), selectinload(AgentRun.reviews))
    )
    workspace_id = db.info.get("workspace_id")
    if workspace_id is not None:
        statement = statement.where(AgentRun.workspace_id == workspace_id)
    run = db.scalar(statement)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent run not found")
    return run
