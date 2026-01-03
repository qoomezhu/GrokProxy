import json
import time
import logging
import os
from typing import List, Optional, Union
import yaml
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette import status
from grok import GrokRequest
from pydantic import BaseModel, Field
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.responses import StreamingResponse, JSONResponse

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== 数据模型 =====================
class Message(BaseModel):
    role: str
    content: str

class OpenAIRequest(BaseModel):
    model: str = Field(default="grok-3", description="模型名称")
    stream: bool = Field(default=False, description="是否流式输出")
    max_tokens: Optional[int] = Field(default=None, description="最大token数")
    temperature: Optional[float] = Field(default=None, description="温度参数")
    messages: List[Message]

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 1700000000
    owned_by: str = "xai"

# ===================== FastAPI 实例 =====================
app = FastAPI(
    title="GrokProxy",
    description="Grok Web to OpenAI API Proxy",
    version="1.1.0"
)

# CORS 中间件 - 支持跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== 全局变量 =====================
grok_request = GrokRequest()
security = HTTPBearer(auto_error=False)

# 加载配置文件
def load_config():
    config_path = os.getenv("CONFIG_PATH", "cookies.yaml")
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using environment variables")
        return {}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

config = load_config()

# 修复密码校验 - 确保是列表格式
def get_valid_api_keys():
    keys = config.get('password', [])
    if keys is None:
        return []
    if isinstance(keys, str):
        return [keys] if keys else []
    if isinstance(keys, list):
        return [k for k in keys if k]  # 过滤空值
    return []

valid_api_keys = get_valid_api_keys()
logger.info(f"Loaded {len(valid_api_keys)} API key(s)")

# ===================== 消息格式化 =====================
def format_messages(messages: List[Message]) -> str:
    """
    将 OpenAI 格式的消息转换为 Grok 需要的纯文本格式
    """
    if not messages:
        return ""
    
    formatted_parts = []
    system_prompt = None
    
    for msg in messages:
        role = msg.role.lower()
        content = msg.content.strip() if msg.content else ""
        
        if role == "system":
            system_prompt = content
        elif role == "user":
            formatted_parts.append(f"User: {content}")
        elif role == "assistant":
            formatted_parts.append(f"Assistant: {content}")
        else:
            formatted_parts.append(f"{role.capitalize()}: {content}")
    
    # 系统提示放在最前面
    if system_prompt:
        result = f"[System Instructions: {system_prompt}]\n\n" + "\n\n".join(formatted_parts)
    else:
        result = "\n\n".join(formatted_parts)
    
    return result

def get_last_user_message(messages: List[Message]) -> str:
    """
    获取最后一条用户消息（某些场景下 Grok 只需要最后一条）
    """
    for msg in reversed(messages):
        if msg.role.lower() == "user":
            return msg.content
    return ""

# ===================== 模型映射 =====================
MODEL_MAP = {
    # Grok 3 系列
    "grok-3": "grok-3",
    "grok-3-latest": "grok-3",
    "grok-3-thinking": "grok-3",
    # Grok 2 系列
    "grok-2": "grok-2",
    "grok-2-1212": "grok-2",
    "grok-2-latest": "grok-2",
    # 兼容 OpenAI 模型名
    "gpt-4": "grok-3",
    "gpt-4o": "grok-3",
    "gpt-4-turbo": "grok-3",
    "gpt-3.5-turbo": "grok-2",
    "claude-3-opus": "grok-3",
    "claude-3-sonnet": "grok-3",
}

def map_model_name(model: str) -> str:
    """将请求的模型名映射为 Grok 支持的模型"""
    return MODEL_MAP.get(model, "grok-3")

# ===================== 认证中间件 =====================
async def verify_api_key(authorization: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """验证 API Key"""
    # 如果没有配置密码，跳过验证
    if not valid_api_keys:
        logger.debug("No API keys configured, skipping authentication")
        return None
    
    # 检查是否提供了认证信息
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Missing Authorization header",
                    "type": "invalid_request_error",
                    "code": "missing_api_key"
                }
            }
        )
    
    # 严格比较 API Key
    if authorization.credentials not in valid_api_keys:
        logger.warning(f"Invalid API key attempt: {authorization.credentials[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Invalid API key provided",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key"
                }
            }
        )
    
    return authorization.credentials

# ===================== 流式响应生成器 =====================
async def generate_stream_response(message: str, model: str):
    """生成 OpenAI 兼容的流式响应"""
    mapped_model = map_model_name(model)
    request_id = f"chatcmpl-{int(time.time() * 1000)}"
    
    try:
        async for token in grok_request.get_grok_request(message, mapped_model):
            if token:
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        
        # 发送结束标记
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Stream error: {e}")
        error_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": f"\n\n[Error: {str(e)}]"},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

async def generate_response(message: str, model: str) -> List[str]:
    """生成非流式响应"""
    mapped_model = map_model_name(model)
    tokens = []
    try:
        async for token in grok_request.get_grok_request(message, mapped_model):
            if token:
                tokens.append(token)
    except Exception as e:
        logger.error(f"Response error: {e}")
        tokens.append(f"[Error: {str(e)}]")
    return tokens

# ===================== API 路由 =====================
@app.get("/")
async def root():
    """根路径 - 服务信息"""
    return {
        "message": "GrokProxy is running",
        "version": "1.1.0",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "ok",
        "timestamp": int(time.time()),
        "version": "1.1.0"
    }

@app.get("/v1/models")
async def list_models():
    """列出可用模型 - OpenAI 兼容"""
    models = [
        ModelInfo(id="grok-3"),
        ModelInfo(id="grok-3-thinking"),
        ModelInfo(id="grok-2"),
        ModelInfo(id="grok-2-1212"),
    ]
    return {
        "object": "list",
        "data": [m.model_dump() for m in models]
    }

@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """获取模型详情"""
    return ModelInfo(id=model_id).model_dump()

@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: OpenAIRequest):
    """
    聊天补全接口 - OpenAI 兼容
    """
    logger.info(f"Request: model={request.model}, stream={request.stream}, messages={len(request.messages)}")
    
    # 格式化消息
    formatted_message = format_messages(request.messages)
    
    if not formatted_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message": "No valid messages provided"}}
        )
    
    if request.stream:
        return StreamingResponse(
            generate_stream_response(formatted_message, request.model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        tokens = await generate_response(formatted_message, request.model)
        content = ''.join(tokens)
        
        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(formatted_message) // 4,
                "completion_tokens": len(content) // 4,
                "total_tokens": (len(formatted_message) + len(content)) // 4
            }
        }

# ===================== 兼容旧路由 =====================
@app.post("/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions_legacy(request: OpenAIRequest):
    """兼容不带 /v1 前缀的请求"""
    return await chat_completions(request)

# ===================== 错误处理 =====================
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "internal_error"
            }
        }
    )

# ===================== 启动入口 =====================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
