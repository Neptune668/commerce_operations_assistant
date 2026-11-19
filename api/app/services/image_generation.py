import time
import uuid
import json
from typing import Any

import httpx
from sqlalchemy import select
from app.core.config import API_DIR, Settings
from app.db.session import SessionLocal
from app.models import CreativePlan, CreativePlanType
from app.services.model_settings import get_runtime_settings
from app.services.volcengine import VolcengineCredentialsError, parse_access_key_pair, signed_json_headers


UPLOADS_DIR = API_DIR / "uploads"
GENERATED_IMAGES_DIR = UPLOADS_DIR / "generated"


class ImageGenerationNotConfiguredError(RuntimeError):
    """图片生成本地配置缺失或不被支持时抛出。"""
    pass


class ImageGenerationError(RuntimeError):
    """图片生成过程失败时抛出，例如任务失败、轮询超时或下载失败。"""
    pass


def is_image_generation_configured(settings: Settings | None = None) -> bool:
    """判断本地是否已具备调用图片生成服务的完整配置。"""
    settings = settings or get_runtime_settings()
    return bool(
        settings.image_provider
        and settings.image_base_url
        and settings.image_api_key
        and settings.image_model
    )


def generate_main_image_asset(
    creative_plan: CreativePlan,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """把一条主图创意方案生成为一张图片，返回可直接入库的素材字典。"""
    settings = settings or get_runtime_settings()
    # 检查是否配置了图片模型
    if not is_image_generation_configured(settings):
        raise ImageGenerationNotConfiguredError("Image generation provider is not configured.")
    # 判断image_provider是否为jimeng
    if settings.image_provider != "jimeng":
        raise ImageGenerationNotConfiguredError(
            f"Unsupported image provider: {settings.image_provider}"
        )
    # 判断创意方案是否为主图方案
    if creative_plan.type != CreativePlanType.MAIN_IMAGE:
        raise ImageGenerationError("Only main image plans can generate image assets.")

    # 生成前清洗提示词，减少不可见字符与特殊货币符号导致接口解析失败的概率。
    prompt = _main_image_prompt(creative_plan)
    prompt = _sanitize_image_prompt(prompt)
    # 判断图片模型是否是jimeng_t2i_v30
    if settings.image_model != "jimeng_t2i_v30":
        raise ImageGenerationNotConfiguredError("本工程图片模型必须为 jimeng_t2i_v30。")
    # 请求图片模型，获取任务ID
    task_id = _create_jimeng_image_task(prompt, settings)
    # 轮询请求图片生成结果
    result = _poll_jimeng_image_task(task_id, settings)
    # 获取图片url
    image_url = _extract_jimeng_image_url(result)
    # 下载图片到 uploads目录
    public_path = _download_generated_image(image_url, creative_plan.id)
    return {
        "provider": settings.image_provider,
        "model": settings.image_model,
        "size": settings.image_size,
        "prompt": prompt,
        "task_id": task_id,
        "source_url": image_url,
        "url": public_path,
        "created_at": int(time.time()),
    }

def _create_jimeng_image_task(prompt: str, settings: Settings) -> str:
    """向服务商提交一次图片生成任务，返回供应商任务 ID。"""
    width, height = _parse_image_size(settings.image_size)
    payload: dict[str, Any] = {
        "req_key": settings.image_model,
        "prompt": prompt,
        "use_pre_llm": True,
        "seed": -1,
        "width": width,
        "height": height,
    }
    data = _post_jimeng_json(
        settings=settings,
        action="CVSync2AsyncSubmitTask",
        payload=payload,
        timeout=120,
    )
    task_id = _extract_jimeng_task_id(data)
    if not task_id:
        raise ImageGenerationError("Jimeng image response did not contain task_id.")
    return task_id


def _post_jimeng_json(
    *,
    settings: Settings,
    action: str,
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    """按统一流程发送一次服务商请求：签名、发送、校验业务状态码。"""
    try:
        access_key, secret_key = parse_access_key_pair(settings.image_api_key)
    except VolcengineCredentialsError as exc:
        raise ImageGenerationNotConfiguredError(str(exc)) from exc

    base_url = settings.image_base_url or "https://visual.volcengineapi.com"
    query = {"Action": action, "Version": "2022-08-31"}
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    headers = signed_json_headers(
        access_key=access_key,
        secret_key=secret_key,
        method="POST",
        url=base_url,
        query=query,
        body=body,
    )
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(base_url, params=query, content=body, headers=headers)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        raise ImageGenerationError(_provider_error_detail(exc.response)) from exc
    except httpx.HTTPError as exc:
        raise ImageGenerationError(_http_error_detail("Jimeng image request failed", exc)) from exc

    code = data.get("code")
    if code != 10000:
        raise ImageGenerationError(str(data)[:500])
    return data

def _poll_jimeng_image_task(task_id: str, settings: Settings) -> dict[str, Any]:
    """按配置间隔轮询任务结果，成功返回响应字典，失败或超时抛错。"""
    max_attempts = max(settings.image_poll_max_attempts, 1)
    interval = max(settings.image_poll_interval_seconds, 1)
    payload = {
        "req_key": settings.image_model,
        "task_id": task_id,
        "req_json": json.dumps({"return_url": True}, ensure_ascii=False),
    }

    for _attempt in range(max_attempts):
        data = _post_jimeng_json(
            settings=settings,
            action="CVSync2AsyncGetResult",
            payload=payload,
            timeout=30,
        )
        status_value = _jimeng_status(data)
        if status_value in {"done", "succeeded", "success", ""} and _extract_jimeng_image_url(data):
            return data
        if status_value in {"failed", "expired", "not_found"}:
            raise ImageGenerationError(f"Jimeng image task ended with status: {status_value}")
        time.sleep(interval)

    raise ImageGenerationError("Jimeng image task timed out.")

def _parse_image_size(size: str) -> tuple[int, int]:
    """把形如 1024*1024 的尺寸字符串解析为宽高元组，无法解析时回退默认尺寸。"""
    normalized = size.replace("*", "x").lower()
    parts = normalized.split("x", 1)
    if len(parts) != 2:
        return 1328, 1328
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError:
        return 1328, 1328
    return max(width, 1), max(height, 1)

def _main_image_prompt(creative_plan: CreativePlan) -> str:
    """优先使用方案里的 image_prompt，缺失时按方案字段拼出可用提示词。"""
    content = creative_plan.content or {}
    image_prompt = content.get("image_prompt")
    if isinstance(image_prompt, str) and image_prompt.strip():
        return image_prompt.strip()

    prompt_parts = [
        f"电商商品主图，方案标题：{creative_plan.title}",
        f"主文案：{content.get('headline', '')}",
        f"副文案：{content.get('sub_headline', '')}",
        f"画面布局：{content.get('layout', '')}",
        f"视觉重点：{content.get('visual_focus', '')}",
        "干净背景，商品主体清晰，商业摄影质感，适合电商首图。",
    ]
    return "\n".join(part for part in prompt_parts if part.strip())

def _sanitize_image_prompt(prompt: str) -> str:
    """清理服务商协议可能不接受的特殊字符，保留提示词的业务语义。"""
    replacements = {
        "\u00a5": "yuan",
        "\uffe5": "yuan",
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2028": "\n",
        "\u2029": "\n",
    }
    for old, new in replacements.items():
        prompt = prompt.replace(old, new)

    return "".join(
        char for char in prompt if char in {"\n", "\t"} or ord(char) >= 32
    ).strip()

def _download_generated_image(image_url: str, creative_plan_id: int) -> str:
    """把供应商临时地址的图片下载到本地，返回项目对外可访问的本地路径。"""
    GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"main-image-plan-{creative_plan_id}-{uuid.uuid4().hex}.png"
    output_path = GENERATED_IMAGES_DIR / filename

    # 把供应商临时 URL 下载到本地 uploads，避免外部链接过期后素材记录失效。
    try:
        with httpx.Client(timeout=120, follow_redirects=True) as client:
            response = client.get(image_url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ImageGenerationError(_http_error_detail("Generated image download failed", exc)) from exc

    output_path.write_bytes(response.content)
    return f"/uploads/generated/{filename}"

def _provider_error_detail(response: httpx.Response) -> str:
    """把服务商的错误响应整理成一段可写入日志与异常的简短文本。"""
    try:
        data = response.json()
    except ValueError:
        return response.text[:500]
    return str(data)[:500]

def _http_error_detail(message: str, exc: httpx.HTTPError) -> str:
    """统一格式化 HTTP 异常信息，便于区分网络错误与业务错误。"""
    return f"{message}: {type(exc).__name__}: {exc}"

def _extract_jimeng_task_id(data: dict[str, Any]) -> str:
    """从服务商响应的多个可能容器中提取任务 ID，取不到返回空字符串。"""
    for container in (data.get("data"), data.get("Result"), data.get("output"), data):
        if isinstance(container, dict) and isinstance(container.get("task_id"), str):
            return container["task_id"]
    return ""

def _jimeng_status(data: dict[str, Any]) -> str:
    """读取任务状态并统一转为小写，取不到返回空字符串。"""
    container = data.get("data")
    if isinstance(container, dict) and isinstance(container.get("status"), str):
        return container["status"].lower()
    if isinstance(data.get("status"), str):
        return data["status"].lower()
    return ""

def _extract_jimeng_image_url(data: dict[str, Any]) -> str:
    """从响应中提取图片地址，支持单值与列表两种形式。"""
    containers = [data.get("data"), data.get("Result"), data.get("output"), data]
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in ("image_url", "url"):
            value = container.get(key)
            if isinstance(value, str) and value:
                return value
        for key in ("image_urls", "urls", "result_urls"):
            value = container.get(key)
            if isinstance(value, list) and value and isinstance(value[0], str):
                return value[0]
        binary_data = container.get("binary_data_base64")
        if isinstance(binary_data, list) and binary_data:
            raise ImageGenerationError("Jimeng returned base64 image data; set return_url=true.")
    return ""

if __name__ == '__main__':
    with SessionLocal() as db:
        creative_plan = db.scalars(
            select(CreativePlan).where(CreativePlan.id == 1)
        ).first()

        result = generate_main_image_asset(creative_plan)
        print(result)