#!/usr/bin/env python3
"""
Simplified MCP Server for HuggingFace Hub Tagging Operations using FastMCP
"""

import os
import json
from fastmcp import FastMCP
from huggingface_hub import HfApi, model_info, ModelCard, ModelCardData
from huggingface_hub.utils import HfHubHTTPError
from dotenv import load_dotenv

load_dotenv()

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize HF API client
hf_api = HfApi(token=HF_TOKEN) if HF_TOKEN else None

# Create the FastMCP server
mcp = FastMCP("hf-tagging-bot")


@mcp.tool()
def get_current_tags(repo_id: str) -> str:
    """Get current tags from a HuggingFace model repository"""
    if not hf_api:
        return json.dumps({"error": "HF token not configured"})

    try:
        info = model_info(repo_id=repo_id, token=HF_TOKEN)
        current_tags = info.tags if info.tags else []

        result = {
            "status": "success",
            "repo_id": repo_id,
            "current_tags": current_tags,
            "count": len(current_tags),
        }
        return json.dumps(result)

    except Exception as e:
        error_result = {"status": "error", "repo_id": repo_id, "error": str(e)}
        return json.dumps(error_result)


@mcp.tool()
def add_new_tag(repo_id: str, new_tag: str) -> str:
    """Add a new tag to a HuggingFace model repository via PR"""
    if not hf_api:
        return json.dumps({"error": "HF token not configured"})

    try:
        # Get current model info and tags
        info = model_info(repo_id=repo_id, token=HF_TOKEN)
        current_tags = info.tags if info.tags else []

        # Check if tag already exists
        if new_tag in current_tags:
            result = {
                "status": "already_exists",
                "repo_id": repo_id,
                "tag": new_tag,
                "message": f"Tag '{new_tag}' already exists",
            }
            return json.dumps(result)

        # Add the new tag to existing tags
        updated_tags = current_tags + [new_tag]

        # Create model card content with updated tags
        try:
            # Load existing model card
            card = ModelCard.load(repo_id, token=HF_TOKEN)
            if not hasattr(card, "data") or card.data is None:
                card.data = ModelCardData()
        except HfHubHTTPError:
            # Create new model card if none exists
            card = ModelCard("")
            card.data = ModelCardData()

        # Update tags - create new ModelCardData with updated tags
        card_dict = card.data.to_dict()
        card_dict["tags"] = updated_tags
        card.data = ModelCardData(**card_dict)

        # Create a pull request with the updated model card
        pr_title = f"Add '{new_tag}' tag"
        pr_description = f"""
## Add tag: {new_tag}

This PR adds the `{new_tag}` tag to the model repository.

**Changes:**
- Added `{new_tag}` to model tags
- Updated from {len(current_tags)} to {len(updated_tags)} tags

**Current tags:** {", ".join(current_tags) if current_tags else "None"}
**New tags:** {", ".join(updated_tags)}
"""

        # Create commit with updated model card using CommitOperationAdd
        from huggingface_hub import CommitOperationAdd

        commit_info = hf_api.create_commit(
            repo_id=repo_id,
            operations=[
                CommitOperationAdd(
                    path_in_repo="README.md", path_or_fileobj=str(card).encode("utf-8")
                )
            ],
            commit_message=pr_title,
            commit_description=pr_description,
            token=HF_TOKEN,
            create_pr=True,
        )

        # Extract PR URL from commit info
        pr_url_attr = commit_info.pr_url
        pr_url = pr_url_attr if hasattr(commit_info, "pr_url") else str(commit_info)

        result = {
            "status": "success",
            "repo_id": repo_id,
            "tag": new_tag,
            "pr_url": pr_url,
            "previous_tags": current_tags,
            "new_tags": updated_tags,
            "message": f"Created PR to add tag '{new_tag}'",
        }
        return json.dumps(result)

    except Exception as e:
        error_result = {
            "status": "error",
            "repo_id": repo_id,
            "tag": new_tag,
            "error": str(e),
        }
        return json.dumps(error_result)


if __name__ == "__main__":
    mcp.run()
