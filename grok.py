import json
import os
import logging
from typing import AsyncGenerator, Optional
import httpx
import yaml

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== Cookie 管理 =====================
class CookieManager:
    def __init__(self, config_path: str = "cookies.yaml"):
        self.config_path = config_path
        self.cookies = []
        self.current_index = 0
        self.load_cookies()
    
    def load_cookies(self):
        """加载 Cookie 列表"""
        try:
            # 优先从环境变量读取
            env_cookies = os.getenv("COOKIES")
            if env_cookies:
                # 支持逗号分隔的多个 Cookie
                self.cookies = [c.strip() for c in env_cookies.split(",") if c.strip()]
                logger.info(f"Loaded {len(self.cookies)} cookie(s) from environment")
                return
            
            # 从配置文件读取
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                cookies_data = config.get('cookies', [])
                
                if isinstance(cookies_data, list):
                    self.cookies = [c for c in cookies_data if c]
                elif isinstance(cookies_data, str):
                    self.cookies = [cookies_data] if cookies_data else []
                else:
                    self.cookies = []
                
                logger.info(f"Loaded {len(self.cookies)} cookie(s) from {self.config_path}")
        
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found")
            self.cookies = []
        except Exception as e:
            logger.error(f"Error loading cookies: {e}")
            self.cookies = []
    
    def get_current_cookie(self) -> Optional[str]:
        """获取当前 Cookie"""
        if not self.cookies:
            return None
        return self.cookies[self.current_index % len(self.cookies)]
    
    def rotate_cookie(self):
        """切换到下一个 Cookie"""
        if len(self.cookies) > 1:
            old_index = self.current_index
            self.current_index = (self.current_index + 1) % len(self.cookies)
            logger.info(f"Rotated cookie: {old_index} -> {self.current_index}")
    
    def mark_cookie_failed(self, index: int = None):
        """标记当前 Cookie 失败并切换"""
        if index is None:
            index = self.current_index
        logger.warning(f"Cookie {index} marked as failed, rotating...")
        self.rotate_cookie()
    
    def get_cookie_count(self) -> int:
        """获取 Cookie 总数"""
        return len(self.cookies)

# ===================== Grok 请求处理 =====================
class GrokRequest:
    def __init__(self):
        self.cookie_manager = CookieManager()
        
        # API 端点 - 支持环境变量覆盖
        self.grok_url = os.getenv(
            "GROK_API_URL", 
            "https://grok.com/rest/app-chat/conversations/new"
        )
        
        # HTTP 客户端配置
        self.timeout = httpx.Timeout(
            connect=10.0,
            read=120.0,  # Grok 响应可能很慢
            write=10.0,
            pool=10.0
        )
        
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            http2=True  # 启用 HTTP/2
        )
        
        logger.info(f"GrokRequest initialized with URL: {self.grok_url}")
    
    def _build_headers(self, cookie: str) -> dict:
        """构建请求头"""
        return {
            "accept": "text/event-stream",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": "https://grok.com",
            "referer": "https://grok.com/",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "cookie": cookie,
            # 可选的额外头部
            "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
        }
    
    def _build_payload(self, message: str, model: str) -> dict:
        """构建请求体"""
        return {
            "temporary": False,
            "modelName": model,
            "message": message,
            "fileAttachments": [],
            "imageAttachments": [],
            "disableSearch": False,
            "enableImageGeneration": True,
            "returnImageBytes": False,
            "returnRawGrokInXaiRequest": False,
            "enableImageStreaming": True,
            "imageGenerationCount": 2,
            "forceConcise": False,
            "toolOverrides": {},
            "enableSideBySide": True,
            "isPreset": False,
            "sendFinalMetadata": True,
            "customInstructions": "",
            "deepsearchPreset": "",
            "isReasoning": False
        }
    
    async def get_grok_request(self, message: str, model: str = "grok-3") -> AsyncGenerator[str, None]:
        """
        发送请求到 Grok 并流式返回响应
        """
        cookie = self.cookie_manager.get_current_cookie()
        
        if not cookie:
            logger.error("No valid cookie available")
            yield "[Error: No valid cookie configured. Please check cookies.yaml]"
            return
        
        headers = self._build_headers(cookie)
        payload = self._build_payload(message, model)
        
        max_retries = min(3, self.cookie_manager.get_cookie_count())
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Sending request to Grok (attempt {attempt + 1}/{max_retries})")
                
                async with self.client.stream(
                    "POST",
                    self.grok_url,
                    headers=headers,
                    json=payload
                ) as response:
                    
                    # 检查响应状态
                    if response.status_code == 401:
                        logger.warning("Cookie expired or invalid (401)")
                        self.cookie_manager.mark_cookie_failed()
                        cookie = self.cookie_manager.get_current_cookie()
                        if cookie:
                            headers = self._build_headers(cookie)
                            continue
                        else:
                            yield "[Error: All cookies are invalid]"
                            return
                    
                    if response.status_code == 429:
                        logger.warning("Rate limited (429)")
                        self.cookie_manager.rotate_cookie()
                        cookie = self.cookie_manager.get_current_cookie()
                        headers = self._build_headers(cookie)
                        continue
                    
                    if response.status_code == 403:
                        logger.warning("Forbidden (403) - IP may be blocked")
                        yield "[Error: Access forbidden. Your IP may be blocked by Grok.]"
                        return
                    
                    if response.status_code != 200:
                        logger.error(f"Unexpected status code: {response.status_code}")
                        yield f"[Error: Grok returned status {response.status_code}]"
                        return
                    
                    # 处理 SSE 流
                    buffer = ""
                    async for chunk in response.aiter_text():
                        buffer += chunk
                        
                        # 按行分割处理
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            
                            if not line:
                                continue
                            
                            # 解析 SSE 数据
                            token = self._parse_sse_line(line)
                            if token:
                                yield token
                    
                    # 处理剩余的 buffer
                    if buffer.strip():
                        token = self._parse_sse_line(buffer.strip())
                        if token:
                            yield token
                    
                    # 成功完成，退出重试循环
                    logger.info("Request completed successfully")
                    return
            
            except httpx.TimeoutException as e:
                logger.error(f"Request timeout: {e}")
                if attempt < max_retries - 1:
                    self.cookie_manager.rotate_cookie()
                    cookie = self.cookie_manager.get_current_cookie()
                    if cookie:
                        headers = self._build_headers(cookie)
                        continue
                yield "[Error: Request timeout. Please try again.]"
                return
            
            except httpx.HTTPError as e:
                logger.error(f"HTTP error: {e}")
                if attempt < max_retries - 1:
                    continue
                yield f"[Error: HTTP error - {str(e)}]"
                return
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                yield f"[Error: {str(e)}]"
                return
        
        yield "[Error: All retry attempts failed]"
    
    def _parse_sse_line(self, line: str) -> Optional[str]:
        """
        解析 SSE 行并提取文本内容
        """
        try:
            # 尝试直接解析 JSON
            if line.startswith("data:"):
                line = line[5:].strip()
            
            if not line or line == "[DONE]":
                return None
            
            data = json.loads(line)
            
            # 尝试多种可能的字段路径
            # 路径 1: result.response.text
            if "result" in data:
                result = data["result"]
                if isinstance(result, dict):
                    if "response" in result:
                        response = result["response"]
                        if isinstance(response, dict) and "text" in response:
                            return response["text"]
                        elif isinstance(response, str):
                            return response
                    if "text" in result:
                        return result["text"]
                    if "token" in result:
                        return result["token"]
            
            # 路径 2: response.text
            if "response" in data:
                response = data["response"]
                if isinstance(response, dict) and "text" in response:
                    return response["text"]
                elif isinstance(response, str):
                    return response
            
            # 路径 3: 直接的 text 或 token 字段
            if "text" in data:
                return data["text"]
            if "token" in data:
                return data["token"]
            if "content" in data:
                return data["content"]
            
            # 路径 4: choices[0].delta.content (OpenAI 格式)
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "delta" in choice and "content" in choice["delta"]:
                    return choice["delta"]["content"]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]
            
            return None
        
        except json.JSONDecodeError:
            # 不是 JSON，可能是纯文本
            if line and not line.startswith(":"):
                return line
            return None
        except Exception as e:
            logger.debug(f"Error parsing SSE line: {e}")
            return None
    
    async def close(self):
        """关闭 HTTP 客户端"""
        await self.client.aclose()


# ===================== 模块测试 =====================
if __name__ == "__main__":
    import asyncio
    
    async def test():
        grok = GrokRequest()
        async for token in grok.get_grok_request("Hello, Grok!", "grok-3"):
            print(token, end="", flush=True)
        print()
        await grok.close()
    
    asyncio.run(test())
