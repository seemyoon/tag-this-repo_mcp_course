import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import uvicorn
from pydantic import BaseModel
from huggingface_hub.inference._mcp.agent import Agent
from dotenv import load_dotenv

load_dotenv()

# Configuration
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "your-webhook-secret")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "microsoft/DialoGPT-medium")
HF_PROVIDER = os.getenv("HF_PROVIDER", "huggingface")

# Simple storage for processed comments
comments_store: List[Dict[str, Any]] = []

# Agent instance
agent_instance: Optional[Agent] = None


class WebhookEvent(BaseModel):
    event: Dict[str, str]
    comment: Dict[str, Any]
    discussion: Dict[str, Any]
    repo: Dict[str, str]


app = FastAPI(title="HF Discussion Bot")
app.add_middleware(CORSMiddleware, allow_origins=["*"])


async def get_agent():
    """Get or create Agent instance"""
    global agent_instance
    if agent_instance is None and HF_TOKEN:
        agent_instance = Agent(
            model=HF_MODEL,
            provider=HF_PROVIDER,
            api_key=HF_TOKEN,
            servers=[
                {
                    "type": "stdio",
                    "config": {"command": "python", "args": ["mcp_server.py"]},
                }
            ],
        )
        await agent_instance.load_tools()
    return agent_instance


async def process_webhook_comment(webhook_data: Dict[str, Any]):
    """Process webhook using Agent with MCP tools"""
    comment_content = webhook_data["comment"]["content"]
    discussion_title = webhook_data["discussion"]["title"]
    repo_name = webhook_data["repo"]["name"]
    discussion_num = webhook_data["discussion"]["num"]

    agent = await get_agent()
    if not agent:
        ai_response = "Error: Agent not configured (missing HF_TOKEN)"
    else:
        # Use Agent to respond to the discussion
        prompt = f"""
        Please respond to this HuggingFace discussion comment using the available tools.
        
        Repository: {repo_name}
        Discussion: {discussion_title} (#{discussion_num})
        Comment: {comment_content}
        
        First use generate_discussion_response to create a helpful response, then use post_discussion_comment to post it.
        """

        try:
            response_parts = []
            async for item in agent.run(prompt):
                # Collect the agent's response
                if hasattr(item, "content") and item.content:
                    response_parts.append(item.content)
                elif isinstance(item, str):
                    response_parts.append(item)

            ai_response = (
                " ".join(response_parts) if response_parts else "No response generated"
            )
        except Exception as e:
            ai_response = f"Error using agent: {str(e)}"

    # Store the interaction with reply link
    discussion_url = f"https://huggingface.co/{repo_name}/discussions/{discussion_num}"

    interaction = {
        "timestamp": datetime.now().isoformat(),
        "repo": repo_name,
        "discussion_title": discussion_title,
        "discussion_num": discussion_num,
        "discussion_url": discussion_url,
        "original_comment": comment_content,
        "ai_response": ai_response,
        "comment_author": webhook_data["comment"]["author"],
    }

    comments_store.append(interaction)
    return ai_response


@app.post("/webhook")
async def webhook_handler(request: Request, background_tasks: BackgroundTasks):
    """Handle HF Hub webhooks"""
    webhook_secret = request.headers.get("X-Webhook-Secret")
    if webhook_secret != WEBHOOK_SECRET:
        return {"error": "Invalid webhook secret"}

    payload = await request.json()
    event = payload.get("event", {})

    if event.get("action") == "create" and event.get("scope") == "discussion.comment":
        background_tasks.add_task(process_webhook_comment, payload)
        return {"status": "processing"}

    return {"status": "ignored"}


async def simulate_webhook(
    repo_name: str, discussion_title: str, comment_content: str
) -> str:
    """Simulate webhook for testing"""
    if not all([repo_name, discussion_title, comment_content]):
        return "Please fill in all fields."

    mock_payload = {
        "event": {"action": "create", "scope": "discussion.comment"},
        "comment": {
            "content": comment_content,
            "author": "test-user",
            "created_at": datetime.now().isoformat(),
        },
        "discussion": {
            "title": discussion_title,
            "num": len(comments_store) + 1,
        },
        "repo": {"name": repo_name},
    }

    response = await process_webhook_comment(mock_payload)
    return f"✅ Processed! AI Response: {response}"


def create_gradio_app():
    """Create Gradio interface"""
    with gr.Blocks(title="HF Discussion Bot", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 HF Discussion Bot Dashboard")
        gr.Markdown("*Powered by HuggingFace Tiny Agents + FastMCP*")

        with gr.Column():
            sim_repo = gr.Textbox(label="Repository", value="microsoft/DialoGPT-medium")
            sim_title = gr.Textbox(label="Discussion Title", value="Test Discussion")
            sim_comment = gr.Textbox(
                label="Comment",
                lines=3,
                value="How do I use this model?",
            )
            sim_btn = gr.Button("📤 Test Webhook")

        with gr.Column():
            sim_result = gr.Textbox(label="Result", lines=8)

        sim_btn.click(
            fn=simulate_webhook,
            inputs=[sim_repo, sim_title, sim_comment],
            outputs=[sim_result],
        )

    return demo


# Mount Gradio app
gradio_app = create_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/")


if __name__ == "__main__":
    print("🚀 Starting HF Discussion Bot with Tiny Agents...")
    print("📊 Dashboard: http://localhost:7860")
    print("🔗 Webhook: http://localhost:7860/webhook")
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
