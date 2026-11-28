from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable

from sqlalchemy import select

from app.agents.harness import redact_snapshot
from app.agents.policy import AgentPolicyError, authorize_tool
from app.db.session import SessionLocal
from app.models import (
    AgentRunOperation,
    CompetitorMonitor,
    InventoryItem,
    MembershipStatus,
    Product,
    ProductDiagnosis,
    ProductSku,
    Store,
    User,
    UserRole,
    UserStatus,
    Workspace,
    WorkspaceMembership,
    WorkspaceStatus,
)
from app.schemas import StoreInventoryRow
from app.services.ad_recommendations import generate_ad_recommendation
from app.services.creative_plans import generate_main_image_plans, generate_video_script_plans
from app.services.competitor_monitoring import run_competitor_monitor
from app.services.diagnosis import generate_product_diagnosis
from app.services.inventory_advice import generate_inventory_advice
from app.services.promotion_links import build_promotion_link_payloads
from app.services.review import generate_review
from app.services.agent_leases import require_active_lease


class ToolApprovalRequired(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolExecutionContext:
    workspace_id: int
    user_id: int
    workspace_role: UserRole
    run_id: int
    lease_owner: str | None = None


class ToolGateway:

    def __init__(self, context: ToolExecutionContext) -> None:
        self.context = context

    def execute(
        self,
        *,
        agent_key: str,
        tool_key: str,
        input_payload: dict[str, Any],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        # 1.生成工具操作记录的key
        operation_key = f"tool:{agent_key}:{tool_key}"
        with SessionLocal() as db:
            # 2.获取AgentRun
            run = require_active_lease(
                db,
                run_id=self.context.run_id,
                lease_owner=self.context.lease_owner,
                for_update=True,
            )
            # 3.租户权限校验
            if run.workspace_id != self.context.workspace_id:
                raise AgentPolicyError("Agent run is outside current Workspace")

            # 4.验证agent工具权限
            current_role = self._current_workspace_role(db)
            decision = authorize_tool(
                agent_key=agent_key,
                tool_key=tool_key,
                workspace_role=current_role,
            )
            # 如果agent不具备调用权限
            if not decision.allowed:
                raise AgentPolicyError(decision.reason)
            # 如果此工具调用需要人工确认，抛出了ToolApprovalRequired异常
            if decision.requires_approval:
                raise ToolApprovalRequired(
                    f"Tool {tool_key} requires {decision.risk_level.value} approval"
                )

            # 5.工具调用的幂等性校验
            operation = db.scalar(
                select(AgentRunOperation).where(
                    AgentRunOperation.run_id == self.context.run_id,
                    AgentRunOperation.operation_key == operation_key,
                )
            )
            # 如果同一个AgentRun中的同一个Agent调用过这个工具，并且状态是completed
            # 则直接返回上一次的调用结果
            if operation is not None and operation.status == "completed":
                return operation.result.get("value", operation.result)
            if operation is None:
                operation = AgentRunOperation(
                    run_id=self.context.run_id,
                    operation_key=operation_key,
                    status="running",
                    result={},
                )
                db.add(operation)
                db.flush()

            # 6.执行工具
            try:
                result = self._execute_tool(db, tool_key=tool_key, input_payload=input_payload)
            except Exception as exc:
                # 如果工具调用失败，则先回滚
                db.rollback()
                # 然后记录操作失败
                self._record_operation_failure(operation_key, exc)
                raise
            operation.status = "completed"
            operation.result = {"value": redact_snapshot(result)}
            operation.error = None
            operation.completed_at = datetime.now(UTC)
            db.commit()
            return result

    def _record_operation_failure(self, operation_key: str, error: Exception) -> None:
        """记录操作failure，在工具执行前复核租户、角色、租约和幂等键，并记录操作结果。"""
        # 节点只通过结构化状态和受控工具交换数据，禁止绕过策略边界写入结果。
        with SessionLocal() as db:
            operation = db.scalar(
                select(AgentRunOperation).where(
                    AgentRunOperation.run_id == self.context.run_id,
                    AgentRunOperation.operation_key == operation_key,
                )
            )
            if operation is None:
                operation = AgentRunOperation(
                    run_id=self.context.run_id,
                    operation_key=operation_key,
                    status="failed",
                    result={},
                )
                db.add(operation)
            operation.status = "failed"
            operation.error = str(error)[:2000]
            db.commit()

    def _current_workspace_role(self, db) -> UserRole:
        """当前Workspace角色，在工具执行前复核租户、角色、租约和幂等键，并记录操作结果。"""
        # 节点只通过结构化状态和受控工具交换数据，禁止绕过策略边界写入结果。
        membership = db.scalar(
            select(WorkspaceMembership)
            .join(Workspace, Workspace.id == WorkspaceMembership.workspace_id)
            .join(User, User.id == WorkspaceMembership.user_id)
            .where(
                WorkspaceMembership.workspace_id == self.context.workspace_id,
                WorkspaceMembership.user_id == self.context.user_id,
                WorkspaceMembership.status == MembershipStatus.ACTIVE,
                Workspace.status == WorkspaceStatus.ACTIVE,
                User.status == UserStatus.ACTIVE,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if membership is None:
            raise AgentPolicyError("Agent requester no longer has active Workspace access")
        return membership.role

    def _execute_tool(self, db, *, tool_key: str, input_payload: dict[str, Any]):
        """执行工具，在工具执行前复核租户、角色、租约和幂等键，并记录操作结果。"""
        # 节点只通过结构化状态和受控工具交换数据，禁止绕过策略边界写入结果。
        handlers: dict[str, Callable[[], Any]] = {
            "diagnosis.draft": lambda: generate_product_diagnosis(
                self._product(db, input_payload["product_id"])
            ),
            "competitor.monitor.run": lambda: self._run_competitor_monitors(
                db,
                product_id=input_payload.get("product_id"),
                store_id=input_payload.get("store_id"),
            ),
            "creative.plan.draft": lambda: self._creative_plans(
                db,
                product_id=input_payload["product_id"],
            ),
            "ad.recommendation.draft": lambda: generate_ad_recommendation(
                self._product(db, input_payload["product_id"])
            ),
            "promotion.link.draft": lambda: build_promotion_link_payloads(
                self._product(db, input_payload["product_id"])
            ),
            "performance.review.draft": lambda: generate_review(
                self._product(db, input_payload["product_id"])
            ),
            "inventory.advice.draft": lambda: self._inventory_advice(
                db,
                store_id=input_payload["store_id"],
            ),
        }
        handler = handlers.get(tool_key)
        if handler is None:
            raise AgentPolicyError(f"Tool has no executable adapter: {tool_key}")
        return redact_snapshot(handler())

    def _product(self, db, product_id: int) -> Product:
        """商品，在工具执行前复核租户、角色、租约和幂等键，并记录操作结果。"""
        product = db.get(Product, int(product_id))
        if product is None or product.workspace_id != self.context.workspace_id:
            raise AgentPolicyError("Product tool scope is outside current Workspace")
        return product

    def _store(self, db, store_id: int) -> Store:
        """店铺，在工具执行前复核租户、角色、租约和幂等键，并记录操作结果。"""
        store = db.get(Store, int(store_id))
        if store is None or store.workspace_id != self.context.workspace_id:
            raise AgentPolicyError("Store tool scope is outside current Workspace")
        return store

    def _run_competitor_monitors(self,db, *,product_id: int | None,store_id: int | None,) -> list[dict[str, Any]]:
        # 1.查询当前商品的竞品信息
        statement = select(CompetitorMonitor).join(
            Product,
            Product.id == CompetitorMonitor.product_id,
        )
        statement = statement.where(
            Product.workspace_id == self.context.workspace_id,
            CompetitorMonitor.status == "active",
        )
        # 2.查询商品信息和店铺信息
        if product_id is not None:
            self._product(db, product_id)
            statement = statement.where(CompetitorMonitor.product_id == int(product_id))
        elif store_id is not None:
            self._store(db, store_id)
            statement = statement.where(Product.store_id == int(store_id))
        else:
            raise AgentPolicyError("Competitor monitor tool requires product_id or store_id")
        # 3.调用原有的业务能力进行竞品分析
        snapshots = [run_competitor_monitor(db, monitor) for monitor in db.scalars(statement).all()]
        return [
            {
                "snapshot_id": snapshot.id,
                "monitor_id": snapshot.monitor_id,
                "status": snapshot.status,
                "changes": snapshot.changes,
                "error": snapshot.error,
            }
            for snapshot in snapshots
        ]

    def _creative_plans(self, db, *, product_id: int) -> dict[str, Any]:
        # 1.查询商品信息
        product = self._product(db, product_id)
        # 2.获取商品的诊断信息
        diagnosis = db.scalar(
            select(ProductDiagnosis)
            .where(ProductDiagnosis.product_id == product.id)
            .order_by(ProductDiagnosis.id.desc())
            .limit(1)
        )
        # 3.生成主图方案和视频脚本
        return {
            "main_image_plans": generate_main_image_plans(product, diagnosis),
            "video_script_plans": generate_video_script_plans(product, diagnosis),
        }

    def _inventory_advice(self, db, *, store_id: int) -> dict[str, Any]:
        """库存建议，在工具执行前复核租户、角色、租约和幂等键，并记录操作结果。"""
        store = self._store(db, store_id)
        rows = self._inventory_rows(db, store_id=store.id)
        return generate_inventory_advice(store, rows)

    def _inventory_rows(self, db, *, store_id: int) -> list[StoreInventoryRow]:
        # 根据店铺ID查询sku库存信息
        statement = (
            select(ProductSku, Product, InventoryItem)
            .join(Product, ProductSku.product_id == Product.id)
            .outerjoin(InventoryItem, InventoryItem.sku_id == ProductSku.id)
            .where(
                Product.store_id == store_id,
                Product.workspace_id == self.context.workspace_id,
            )
            .order_by(Product.id.asc(), ProductSku.id.asc())
        )
        rows = []
        for sku, product, item in db.execute(statement).all():
            quantity = item.quantity if item else 0
            locked = item.locked_quantity if item else 0
            warning = item.warning_threshold if item else 0
            available = quantity - locked
            rows.append(
                StoreInventoryRow(
                    product_id=product.id,
                    product_name=product.name,
                    sku_id=sku.id,
                    sku_code=sku.sku_code,
                    sku_name=sku.name,
                    quantity=quantity,
                    locked_quantity=locked,
                    available_quantity=available,
                    warning_threshold=warning,
                    is_low_stock=available <= warning,
                    location=item.location if item else None,
                )
            )
        return rows