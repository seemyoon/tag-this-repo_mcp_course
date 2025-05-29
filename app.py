import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal

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
# Use a valid provider literal from the documentation
DEFAULT_PROVIDER: Literal["hf-inference"] = "hf-inference"
HF_PROVIDER = os.getenv("HF_PROVIDER", DEFAULT_PROVIDER)

# Simple storage for processed tag operations
tag_operations_store: List[Dict[str, Any]] = []

# Agent instance
agent_instance: Optional[Agent] = None

# Common ML tags that we recognize for auto-tagging
RECOGNIZED_TAGS = {
    "pytorch",
    "tensorflow",
    "jax",
    "transformers",
    "diffusers",
    "text-generation",
    "text-classification",
    "question-answering",
    "text-to-image",
    "image-classification",
    "object-detection",
    "conversational",
    "fill-mask",
    "token-classification",
    "translation",
    "summarization",
    "feature-extraction",
    "sentence-similarity",
    "zero-shot-classification",
    "image-to-text",
    "automatic-speech-recognition",
    "audio-classification",
    "voice-activity-detection",
    "depth-estimation",
    "image-segmentation",
    "video-classification",
    "reinforcement-learning",
    "tabular-classification",
    "tabular-regression",
    "time-series-forecasting",
    "graph-ml",
    "robotics",
    "computer-vision",
    "nlp",
    "cv",
    "multimodal",
}


class WebhookEvent(BaseModel):
    event: Dict[str, str]
    comment: Dict[str, Any]
    discussion: Dict[str, Any]
    repo: Dict[str, str]


app = FastAPI(title="HF Tagging Bot")
app.add_middleware(CORSMiddleware, allow_origins=["*"])


async def get_agent():
    """Get or create Agent instance"""
    global agent_instance
    if agent_instance is None and HF_TOKEN:
        agent_instance = Agent(
            model=HF_MODEL,
            provider=DEFAULT_PROVIDER,
            api_key=HF_TOKEN,
            servers=[
                {
                    "type": "stdio",
                    "config": {
                        "command": "python",
                        "args": ["mcp_server.py"],
                        "cwd": ".",  # Ensure correct working directory
                        "env": {"HF_TOKEN": HF_TOKEN} if HF_TOKEN else {},
                    },
                }
            ],
        )
        await agent_instance.load_tools()
    return agent_instance


def extract_tags_from_text(text: str) -> List[str]:
    """Extract potential tags from discussion text"""
    text_lower = text.lower()

    # Look for explicit tag mentions like "tag: pytorch" or "#pytorch"
    explicit_tags = []

    # Pattern 1: "tag: something" or "tags: something"
    tag_pattern = r"tags?:\s*([a-zA-Z0-9-_,\s]+)"
    matches = re.findall(tag_pattern, text_lower)
    for match in matches:
        # Split by comma and clean up
        tags = [tag.strip() for tag in match.split(",")]
        explicit_tags.extend(tags)

    # Pattern 2: "#hashtag" style
    hashtag_pattern = r"#([a-zA-Z0-9-_]+)"
    hashtag_matches = re.findall(hashtag_pattern, text_lower)
    explicit_tags.extend(hashtag_matches)

    # Pattern 3: Look for recognized tags mentioned in natural text
    mentioned_tags = []
    for tag in RECOGNIZED_TAGS:
        if tag in text_lower:
            mentioned_tags.append(tag)

    # Combine and deduplicate
    all_tags = list(set(explicit_tags + mentioned_tags))

    # Filter to only include recognized tags or explicitly mentioned ones
    valid_tags = []
    for tag in all_tags:
        if tag in RECOGNIZED_TAGS or tag in explicit_tags:
            valid_tags.append(tag)

    return valid_tags


async def process_webhook_comment(webhook_data: Dict[str, Any]):
    """Process webhook to detect and add tags"""
    comment_content = webhook_data["comment"]["content"]
    discussion_title = webhook_data["discussion"]["title"]
    repo_name = webhook_data["repo"]["name"]
    discussion_num = webhook_data["discussion"]["num"]
    comment_author = webhook_data["comment"]["author"]

    # Extract potential tags from the comment and discussion title
    comment_tags = extract_tags_from_text(comment_content)
    title_tags = extract_tags_from_text(discussion_title)
    all_tags = list(set(comment_tags + title_tags))

    result_messages = []

    if not all_tags:
        result_messages.append("No recognizable tags found in the discussion.")
    else:
        agent = await get_agent()
        if not agent:
            msg = "Error: Agent not configured (missing HF_TOKEN)"
            result_messages.append(msg)
        else:
            # Process each tag
            for tag in all_tags:
                try:
                    # Get response from agent
                    responses = []
                    prompt = (
                        f"Add the tag '{tag}' to repository {repo_name} "
                        "using add_new_tag"
                    )

                    async for item in agent.run(prompt):
                        # Just collect the response content
                        responses.append(str(item))

                    response_text = " ".join(responses) if responses else "Completed"

                    # Try to parse JSON from response if possible
                    try:
                        # Look for JSON in the response
                        json_found = False
                        for response_part in responses:
                            response_str = str(response_part)
                            if "{" in response_str and "}" in response_str:
                                # Try to extract JSON from the response
                                start_idx = response_str.find("{")
                                end_idx = response_str.rfind("}") + 1
                                json_str = response_str[start_idx:end_idx]

                                try:
                                    json_response = json.loads(json_str)
                                    status = json_response.get("status")
                                    if status == "success":
                                        pr_url = json_response.get("pr_url", "")
                                        msg = f"Tag '{tag}': PR created - {pr_url}"
                                    elif status == "already_exists":
                                        msg = f"Tag '{tag}': Already exists"
                                    else:
                                        tag_msg = json_response.get(
                                            "message", "Processed"
                                        )
                                        msg = f"Tag '{tag}': {tag_msg}"
                                    json_found = True
                                    break
                                except json.JSONDecodeError:
                                    continue

                        if not json_found:
                            # If no JSON found, use the response as is
                            msg = f"Tag '{tag}': {response_text}"

                    except Exception as parse_error:
                        msg = f"Tag '{tag}': Response parse error - {response_text}"

                    result_messages.append(msg)

                except Exception as e:
                    error_msg = f"Error processing tag '{tag}': {str(e)}"
                    result_messages.append(error_msg)

    # Store the interaction
    base_url = "https://huggingface.co"
    discussion_url = f"{base_url}/{repo_name}/discussions/{discussion_num}"

    interaction = {
        "timestamp": datetime.now().isoformat(),
        "repo": repo_name,
        "discussion_title": discussion_title,
        "discussion_num": discussion_num,
        "discussion_url": discussion_url,
        "original_comment": comment_content,
        "comment_author": comment_author,
        "detected_tags": all_tags,
        "results": result_messages,
    }

    tag_operations_store.append(interaction)
    return " | ".join(result_messages)


@app.post("/webhook")
async def webhook_handler(request: Request, background_tasks: BackgroundTasks):
    """Handle HF Hub webhooks"""
    webhook_secret = request.headers.get("X-Webhook-Secret")
    if webhook_secret != WEBHOOK_SECRET:
        return {"error": "Invalid webhook secret"}

    payload = await request.json()
    event = payload.get("event", {})

    scope_check = event.get("scope") == "discussion.comment"
    if event.get("action") == "create" and scope_check:
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
            "num": len(tag_operations_store) + 1,
        },
        "repo": {"name": repo_name},
    }

    response = await process_webhook_comment(mock_payload)
    return f"✅ Processed! Results: {response}"


def create_gradio_app():
    """Create Gradio interface"""
    with gr.Blocks(title="HF Tagging Bot", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏷️ HF Tagging Bot Dashboard")
        gr.Markdown("*Automatically adds tags to models when mentioned in discussions*")

        gr.Markdown("""
        ## How it works:
        - Monitors HuggingFace Hub discussions
        - Detects tag mentions in comments (e.g., "tag: pytorch", 
          "#transformers")
        - Automatically adds recognized tags to the model repository
        - Supports common ML tags like: pytorch, tensorflow, 
          text-generation, etc.
        """)

        with gr.Column():
            sim_repo = gr.Textbox(
                label="Repository",
                value="burtenshaw/play-mcp-repo-bot",
                placeholder="username/model-name",
            )
            sim_title = gr.Textbox(
                label="Discussion Title",
                value="Add pytorch tag",
                placeholder="Discussion title",
            )
            sim_comment = gr.Textbox(
                label="Comment",
                lines=3,
                value="This model should have tags: pytorch, text-generation",
                placeholder="Comment mentioning tags...",
            )
            sim_btn = gr.Button("🏷️ Test Tag Detection")

        with gr.Column():
            sim_result = gr.Textbox(label="Result", lines=8)

        sim_btn.click(
            simulate_webhook,
            inputs=[sim_repo, sim_title, sim_comment],
            outputs=sim_result,
        )

        gr.Markdown(f"""
        ## Recognized Tags:
        {", ".join(sorted(RECOGNIZED_TAGS))}
        """)

    return demo


# Mount Gradio app
gradio_app = create_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")


if __name__ == "__main__":
    print("🚀 Starting HF Tagging Bot...")
    print("📊 Dashboard: http://localhost:7860/gradio")
    print("🔗 Webhook: http://localhost:7860/webhook")
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
