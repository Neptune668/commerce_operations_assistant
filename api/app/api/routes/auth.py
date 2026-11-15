from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.security import HTTPBearer
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import User, UserRole, Workspace
from app.schemas import AuthTokenRead, LoginRequest, UserRead


from app.service.auth import ensure_user_can_login, AuthTokenError, create_access_token
from app.service.users import verify_password, user_to_read

# from app.services.workspaces import WorkspaceAccess, WorkspaceAccessError, resolve_workspace_access

router = APIRouter(prefix="auth", tags=["auth"])
SessionDep = Annotated[Session, Depends(get_db)]


@router.post("/login", response_model=AuthTokenRead)
def login(payload: LoginRequest, db: SessionDep) -> dict:
    # 邮箱统一归一化后查询，避免大小写差异导致同一运营账号出现多套登录态。
    """处理登录接口；完成身份、Workspace、资源归属和输入校验后返回结构化结果。"""
    # 先完成认证、Workspace 与资源归属校验，再进入会产生副作用的服务调用。
    user = db.scalars(select(User).where(User.email == payload.email.strip().lower())).first()
    if user is None or not user.password_hash or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    try:
        ensure_user_can_login(user)
    except AuthTokenError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    return {
        "access_token": create_access_token(user),
        "token_type": "bearer",
        "user": user_to_read(user),
    }