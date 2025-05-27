---
title: Mcp Discussion Bot
emoji: 👀
colorFrom: purple
colorTo: yellow
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: false
base_path: /gradio
---

# 🤖 Hugging Face Discussion Bot

A FastAPI and Gradio application that automatically responds to Hugging Face Hub discussion comments using AI-powered responses via Hugging Face Inference API with MCP integration.

## ✨ Features

- **Webhook Integration**: Receives real-time webhooks from Hugging Face Hub when new discussion comments are posted
- **AI-Powered Responses**: Uses Hugging Face Inference API with MCP support for intelligent, context-aware responses
- **Interactive Dashboard**: Beautiful Gradio interface to monitor comments and test functionality
- **Automatic Posting**: Posts AI responses back to the original discussion thread
- **Testing Tools**: Built-in webhook simulation and AI testing capabilities
- **MCP Server**: Includes a Model Context Protocol server for advanced tool integration

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd mcp-course-unit3-example

# Install dependencies
pip install -e .
```

### 2. Environment Setup

Copy the example environment file and configure your API keys:

```bash
cp env.example .env
```

Edit `.env` with your credentials:

```env
# Webhook Configuration
WEBHOOK_SECRET=your-secure-webhook-secret

# Hugging Face Configuration  
HF_TOKEN=hf_your_hugging_face_token_here

# Model Configuration (optional)
HF_MODEL=microsoft/DialoGPT-medium
HF_PROVIDER=huggingface
```

### 3. Run the Application

```bash
python server.py
```

The application will start on `http://localhost:8000` with:
- 📊 **Gradio Dashboard**: `http://localhost:8000/gradio`
- 🔗 **Webhook Endpoint**: `http://localhost:8000/webhook`
- 📋 **API Documentation**: `http://localhost:8000/docs`

## 🔧 Configuration

### Hugging Face Hub Webhook Setup

1. Go to your Hugging Face repository settings
2. Navigate to the "Webhooks" section
3. Create a new webhook with:
   - **URL**: `https://your-domain.com/webhook`
   - **Secret**: Same as `WEBHOOK_SECRET` in your `.env`
   - **Events**: Subscribe to "Community (PR & discussions)"

### Required API Keys

#### Hugging Face Token
1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Write" permissions
3. Add it to your `.env` as `HF_TOKEN`

## 📊 Dashboard Features

### Recent Comments Tab
- View all processed discussion comments
- See AI responses in real-time
- Refresh and filter capabilities

### Test HF Inference Tab
- Direct testing of the Hugging Face Inference API
- Custom prompt input
- Response preview

### Simulate Webhook Tab
- Test webhook processing without real HF events
- Mock discussion scenarios
- Validate AI response generation

### Configuration Tab
- View current setup status
- Check API key configuration
- Monitor processing statistics

## 🔌 API Endpoints

### POST `/webhook`
Receives webhooks from Hugging Face Hub.

**Headers:**
- `X-Webhook-Secret`: Your webhook secret

**Body:** HF Hub webhook payload

### GET `/comments`
Returns all processed comments and responses.

### GET `/`
Basic API information and available endpoints.

## 🤖 MCP Server

The application includes a Model Context Protocol (MCP) server that provides tools for:

- **get_discussions**: Retrieve discussions from HF repositories
- **get_discussion_details**: Get detailed information about specific discussions
- **comment_on_discussion**: Add comments to discussions
- **generate_ai_response**: Generate AI responses using HF Inference
- **respond_to_discussion**: Generate and post AI responses automatically

### Running the MCP Server

```bash
python mcp_server.py
```

The MCP server uses stdio transport and can be integrated with MCP clients following the [Tiny Agents pattern](https://huggingface.co/blog/python-tiny-agents).

## 🧪 Testing

### Local Testing
Use the "Simulate Webhook" tab in the Gradio dashboard to test without real webhooks.

### Webhook Testing
You can test the webhook endpoint directly:

```bash
curl -X POST http://localhost:8000/webhook \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Secret: your-webhook-secret" \
  -d '{
    "event": {"action": "create", "scope": "discussion.comment"},
    "comment": {
      "content": "@discussion-bot How do I use this model?",
      "author": "test-user",
      "created_at": "2024-01-01T00:00:00Z"
    },
    "discussion": {
      "title": "Test Discussion",
      "num": 1,
      "url": {"api": "https://huggingface.co/api/repos/test/repo/discussions"}
    },
    "repo": {"name": "test/repo"}
  }'
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   HF Hub        │───▶│   FastAPI       │───▶│   HF Inference  │
│   Webhook       │    │   Server        │    │   API           │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Gradio        │
                       │   Dashboard     │
                       └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   MCP Server    │
                       │   (Tools)       │
                       └─────────────────┘
```

## 🔒 Security

- Webhook secret verification prevents unauthorized requests
- Environment variables keep sensitive data secure
- CORS middleware configured for safe cross-origin requests

## 🚀 Deployment

### Using Docker (Recommended)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["python", "server.py"]
```

### Using Cloud Platforms

The application can be deployed on:
- **Hugging Face Spaces** (recommended for HF integration)
- **Railway**
- **Render**
- **Heroku**
- **AWS/GCP/Azure**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Support

If you encounter issues:

1. Check the Configuration tab in the dashboard
2. Verify your API keys are correct
3. Ensure webhook URL is accessible
4. Check the application logs

For additional help, please open an issue in the repository.

## 🔗 Related Links

- [Hugging Face Webhooks Guide](https://huggingface.co/docs/hub/en/webhooks-guide-discussion-bot)
- [Hugging Face Hub Python Library](https://huggingface.co/docs/huggingface_hub/en/guides/community)
- [Tiny Agents in Python Blog Post](https://huggingface.co/blog/python-tiny-agents)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Gradio Documentation](https://gradio.app/)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
