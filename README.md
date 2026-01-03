# 🚀 GrokProxy

将 Grok 网页版转换为 OpenAI 兼容 API 的反向代理服务。

## ✨ 功能特性

- ✅ 完全兼容 OpenAI API 格式
- ✅ 支持流式输出 (SSE)
- ✅ 多 Cookie 轮询（自动切换）
- ✅ 支持 Grok-3 / Grok-2 模型
- ✅ Docker 一键部署
- ✅ 支持 Zeabur / Railway / ClawCloud 等平台

## 📦 快速开始

### 方式一：Docker 部署（推荐）

```bash
# 1.‌ 克隆项目
git clone https://github.com/your-repo/GrokProxy.git
cd GrokProxy

# 2.‌ 配置 Cookie
cp cookies.yaml.example cookies.yaml
# 编辑 cookies.yaml，填入你的 Cookie

# 3.‌ 启动服务
docker-compose up -d

# 4.‌ 查看日志
docker-compose logs -f
