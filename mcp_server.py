#!/usr/bin/env python3
"""
Simplified MCP Server for HuggingFace Hub Operations using FastMCP
"""

import os
from fastmcp import FastMCP
from huggingface_hub import comment_discussion, InferenceClient
from dotenv import load_dotenv

load_dotenv()

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
DEFAULT_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")

# Initialize HF client
inference_client = (
    InferenceClient(model=DEFAULT_MODEL, token=HF_TOKEN) if HF_TOKEN else None
)

# Create the FastMCP server
mcp = FastMCP("hf-discussion-bot")


@mcp.tool()
def generate_discussion_response(
    discussion_title: str, comment_content: str, repo_name: str
) -> str:
    """Generate AI response for a HuggingFace discussion comment"""
    if not inference_client:
        return "Error: HF token not configured for inference"

    prompt = f"""
    Discussion: {discussion_title}
    Repository: {repo_name}
    Comment: {comment_content}
    
    Provide a helpful response to this comment.
    """

    try:
        messages = [
            {
                "role": "system",
                "content": ("You are a helpful AI assistant for ML discussions."),
            },
            {"role": "user", "content": prompt},
        ]

        response = inference_client.chat_completion(messages=messages, max_tokens=150)
        content = response.choices[0].message.content
        ai_response = content.strip() if content else "No response generated"
        return ai_response

    except Exception as e:
        return f"Error generating response: {str(e)}"


@mcp.tool()
def post_discussion_comment(repo_id: str, discussion_num: int, comment: str) -> str:
    """Post a comment to a HuggingFace discussion"""
    if not HF_TOKEN:
        return "Error: HF token not configured"

    try:
        comment_discussion(
            repo_id=repo_id,
            discussion_num=discussion_num,
            comment=comment,
            token=HF_TOKEN,
        )
        success_msg = f"Successfully posted comment to discussion #{discussion_num}"
        return success_msg

    except Exception as e:
        return f"Error posting comment: {str(e)}"


if __name__ == "__main__":
    mcp.run()
