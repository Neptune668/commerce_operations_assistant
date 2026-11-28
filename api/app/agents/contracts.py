from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CompetitorContext(StrictModel):
    name: str
    platform: str | None = None
    price: str | None = None
    sales_hint: str | None = None
    title: str | None = None
    selling_points: list[str] = Field(default_factory=list)
    review_keywords: list[str] = Field(default_factory=list)


class ProductContext(StrictModel):
    """商品+竞品经营上下文，诊断/创意的只读输入"""
    product_id: int
    name: str
    platform: str
    category: str | None = None
    price: str | None = None
    cost: str | None = None
    target_audience: str | None = None
    selling_points: list[str] = Field(default_factory=list)
    product_url: str | None = None
    images: list[str] = Field(default_factory=list)
    competitors: list[CompetitorContext] = Field(default_factory=list)


class DiagnosisDraft(StrictModel):
    """诊断草稿（定位/价格带/痛点/建议），`source` 区分 `llm`/`rules`"""
    source: Literal["llm", "rules"]
    positioning: str = Field(min_length=1)
    price_band: str = Field(min_length=1)
    audience_insights: list[str] = Field(default_factory=list)
    pain_points: list[str] = Field(default_factory=list)
    selling_point_analysis: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class CreativePlanDraft(StrictModel):
    """创意方向草稿"""
    title: str = Field(min_length=1)
    angle: str = Field(min_length=1)
    layout: str = Field(min_length=1)
    headline: str = Field(min_length=1)
    sub_headline: str = Field(min_length=1)
    visual_focus: str = Field(min_length=1)
    proof_points: list[str] = Field(default_factory=list)
    suitable_audience: str = Field(min_length=1)
    image_prompt: str = Field(min_length=1)
    source: Literal["llm", "rules"]
    provenance: dict[str, Any] = Field(default_factory=dict)


class AgentDraftOutput(StrictModel):
    """``diagnosis` + `creative_plans`（`3–6` 条），人审前的草稿整体"""
    diagnosis: DiagnosisDraft
    creative_plans: list[CreativePlanDraft] = Field(min_length=3, max_length=6)


class QualityIssue(StrictModel):
    code: str
    severity: Literal["warning", "error"]
    message: str
    field: str | None = None


class QualityReport(StrictModel):
    """质量门输出"""
    passed: bool
    score: int = Field(ge=0, le=100)
    issues: list[QualityIssue] = Field(default_factory=list)
    revision_recommended: bool = False
    checked_rules: list[str] = Field(default_factory=list)


class ModelUsage(StrictModel):
    """模型用量（输入/输出/总 Token/调用次数），用于预算"""
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    calls: int = Field(default=0, ge=0)


class ReviewResumePayload(StrictModel):
    """人工审批恢复载荷，`decision` 三选一，并带校验器确保 `edit_and_approve` 必须有 `edited_output`"""
    decision: Literal["approve", "edit_and_approve", "reject"]
    edited_output: AgentDraftOutput | None = None
    note: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_edited_output(self) -> "ReviewResumePayload":
        """校验编辑后输出，保证 Agent 输入、草稿和审批恢复载荷满足可序列化合同。"""
        # 节点只通过结构化状态和受控工具交换数据，禁止绕过策略边界写入结果。
        if self.decision == "edit_and_approve" and self.edited_output is None:
            raise ValueError("edit_and_approve requires edited_output")
        if self.decision != "edit_and_approve" and self.edited_output is not None:
            raise ValueError("edited_output is only allowed for edit_and_approve")
        return self


class HarnessEvent(StrictModel):
    """可观测事件（步骤开始/完成/失败），含脱敏后的输入/输出快照与 trace/span"""
    event_type: str
    agent_name: str | None = None
    operation_name: str | None = None
    status: str | None = None
    message: str | None = None
    input_snapshot: dict[str, Any] = Field(default_factory=dict)
    output_snapshot: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None
    span_id: str | None = None


class AgentWorkflowState(TypedDict, total=False):
    """
    【基础工作流的共享状态 TypedDict（所有节点读写的中枢）】
    run_id：本次工作流运行唯一 ID（agent_runs 主键），所有可观测事件与落库均挂在此 ID 上做审计追踪
    product_id：待分析运营的商品 ID，商品上下文节点据此加载完整经营数据
    context：商品上下文快照（ProductContext 序列化 dict），诊断/创意节点的只读输入
    diagnosis：诊断节点产出（DiagnosisDraft）：定位/价格带/痛点/卖点分析/风险/建议，source 区分 llm/rules
    creative_plans：创意节点产出（3-6 条 CreativePlanDraft）：标题、切入角度、版式、主副标题、画面提示词等
    quality_report：质量门报告（QualityReport）：passed/评分/问题清单，决定走受控返修还是进入人工审核
    revision_feedback：质量报告问题清单序列化，诊断/创意重跑时注入 prompt，实现带问题的受控返修
    revision_count：已执行返修次数，prepare_revision 节点累加，用于限制诊断-质量审核循环
    max_revisions：返修次数上限（配置 AGENT_MAX_REVISIONS），超限后强制进入人审，防止无限循环
    budget：本次运行 Token 预算（配置 AGENT_MAX_TOTAL_TOKENS），调用模型前计算剩余额度做成本护栏
    usage：累计模型用量（ModelUsage：输入/输出/总 Token、调用次数），每个 LLM 节点执行后合并累加
    review：人工审核结果（ReviewResumePayload）：approve/edit_and_approve/reject 三选一 + 备注
    approved_output：人审放行的最终草稿（AgentDraftOutput）；reject 时为空 dict，编辑通过时取人工编辑版
    final_output：finalize 节点落库后返回的最终结果快照（写入数据库的成品输出）
    final_status：工作流终态："completed"（通过/编辑后通过）或 "rejected"（人工驳回）
    """
    run_id: int
    product_id: int
    context: dict[str, Any]
    diagnosis: dict[str, Any]
    creative_plans: list[dict[str, Any]]
    quality_report: dict[str, Any]
    revision_feedback: list[dict[str, Any]]
    revision_count: int
    max_revisions: int
    budget: dict[str, Any]
    usage: dict[str, Any]
    review: dict[str, Any]
    approved_output: dict[str, Any]
    final_output: dict[str, Any]
    final_status: str