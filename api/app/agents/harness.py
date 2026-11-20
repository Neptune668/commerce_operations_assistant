import re
from collections.abc import Mapping, Sequence
from typing import Any

from app.agents.contracts import ModelUsage

# 定义敏感字段匹配正则
SENSITIVE_KEY_PATTERN = re.compile(
    r"(api[_-]?key|authorization|password|secret|cookie|credential|"
    r"(access|refresh|auth|bearer|api)[_-]?token)",
    re.IGNORECASE,
)

MAX_SNAPSHOT_STRING_LENGTH = 2000  # 某个字段值是一段超大文本，截断到 2000 字符
MAX_SNAPSHOT_DEPTH = 8      # 防止数据结构嵌套极深（甚至自引用）导致递归爆炸
MAX_SNAPSHOT_ITEMS = 200    # 单个列表/字典包含成千上万条目，只保留前 200 条，剩余用计数标注


class AgentBudgetExceededError(RuntimeError):
    pass


def redact_snapshot(value: Any, *, depth: int = 0) -> Any:
    """
    保留可回放结构，同时移除密钥类字段并限制异常大文本。
    redact_snapshot 递归遍历任意输入值，在超过深度上限时截断、
    遇到字典/列表时限制条目数并逐项脱敏（命中 SENSITIVE_KEY_PATTERN 或 key 为 token 的字段值替换为 [redacted]）、
    遇到超长字符串时截到 2000 字符，其余标量原样保留，最终返回一份结构可回放、密钥不泄漏、体积有硬上界的安全快照。
    """
    # 【深度闸】嵌套超过 8 层就直接返回占位符，不再向下递归
    # 防御两类问题：超深嵌套导致递归爆炸、自引用结构导致无限循环
    if depth >= MAX_SNAPSHOT_DEPTH:
        return "[truncated-depth]"
    # 【字典分支】字典/列表超过 200 项就截断、不再向下递归
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            # 【项脱敏】值为字典/列表时递归脱敏，否则原样保留
            if index >= MAX_SNAPSHOT_ITEMS:
                result["[truncated-items]"] = len(value) - MAX_SNAPSHOT_ITEMS
                break
            # 【键脱敏】key 为 token 或命中 SENSITIVE_KEY_PATTERN 的字段值替换为 [redacted]
            key_text = str(key)[:200]
            if key_text.lower() == "token" or SENSITIVE_KEY_PATTERN.search(key_text):
                result[key_text] = "[redacted]"
            else:
                result[key_text] = redact_snapshot(item, depth=depth + 1)
        return result
    # 【列表分支】列表超过 200 项就截断、不再向下递归
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for index, item in enumerate(value):
            if index >= MAX_SNAPSHOT_ITEMS:
                break
            result.append(redact_snapshot(item, depth=depth + 1))
        if len(value) > MAX_SNAPSHOT_ITEMS:
            result.append(f"[truncated-items:{len(value) - MAX_SNAPSHOT_ITEMS}]")
        return result
    # 【字符串分支】字符串超过 2000 字符就截断
    if isinstance(value, str) and len(value) > MAX_SNAPSHOT_STRING_LENGTH:
        return f"{value[:MAX_SNAPSHOT_STRING_LENGTH]}...[truncated]"
    # 【标量分支】其他标量类型原样保留
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # 【其他分支】其他类型转换为字符串保留
    return str(value)


def normalize_model_usage(raw_usage: Mapping[str, Any] | None) -> ModelUsage:
    """归一化不同模型供应商返回的用量字段，收敛成统一的内部计数结构。"""

    # normalize_model_usage 是一个【LLM 用量适配器/归一化器】
    # 把不同模型供应商返回的、字段名和形状各异的原始 usage 字典，
    # 收敛成项目内部统一的 ModelUsage Pydantic 结构，供后续的预算扣减、事件记录使用。
    raw_usage = raw_usage or {}
    input_tokens = _non_negative_int(
        raw_usage.get("prompt_tokens", raw_usage.get("input_tokens", 0))
    )
    output_tokens = _non_negative_int(
        raw_usage.get("completion_tokens", raw_usage.get("output_tokens", 0))
    )
    total_tokens = _non_negative_int(raw_usage.get("total_tokens", input_tokens + output_tokens))
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        calls=1,
    )


def merge_model_usage(current: Mapping[str, Any] | None, addition: ModelUsage) -> dict[str, int]:
    """
    merge_model_usage 把已有的 usage 字典（通过 ModelUsage.model_validate 反序列化为对象）
    和本次新增的 ModelUsage 在 input/output/total/calls 四个维度逐项相加，
    再以 model_dump() 返回普通 dict，
    用于在整个 Agent 运行生命周期中持续累加多轮 LLM 调用的总用量账本。
    """
    existing = ModelUsage.model_validate(current or {})
    return ModelUsage(
        input_tokens=existing.input_tokens + addition.input_tokens,
        output_tokens=existing.output_tokens + addition.output_tokens,
        total_tokens=existing.total_tokens + addition.total_tokens,
        calls=existing.calls + addition.calls,
    ).model_dump()


def remaining_model_tokens(
    budget: Mapping[str, Any] | None,
    usage: Mapping[str, Any] | None,
    requested_output_tokens: int,
) -> int:
    """计算本次模型调用可用的输出 Token，并在总预算耗尽时阻止继续调用。"""
    # 用配置的运行总预算 max_total_tokens 减去已累计消耗 consumed 得到剩余额度，
    # 若 ≤ 0 直接抛 AgentBudgetExceededError 熔断本次运行，否则返回节点申请量与剩余额度的较小值，
    # 从而保证任意单次 LLM 调用都无法绕过全局 Token 预算上限。
    max_total = _non_negative_int((budget or {}).get("max_total_tokens", 12000))
    consumed = ModelUsage.model_validate(usage or {}).total_tokens
    remaining = max_total - consumed
    if remaining <= 0:
        raise AgentBudgetExceededError("Agent run token budget exhausted")
    return min(requested_output_tokens, remaining)


def _non_negative_int(value: Any) -> int:
    """把任意取值转成非负整数，无法转换或为空时按零处理。"""
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0