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
