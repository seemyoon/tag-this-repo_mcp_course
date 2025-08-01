import json
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from huggingface_hub import HfApi, model_info, ModelCard, ModelCardData, CommitOperationAdd
from huggingface_hub.utils import HfHubHTTPError
import traceback

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')

HF_API = HfApi(token=HF_TOKEN) if HF_TOKEN else None

mcp = FastMCP('hf-tagging-bot')


@mcp.tool()
def get_current_tags(repo_id: str) -> str:
    print(f'get_current_tags called with repo_id: {repo_id}')

    if not HF_API:
        err_result = {'error:' "HF token not configured"}
        json_str = json.dumps(err_result)
        print(f"No HF API token - returning: {json_str}")
        return json_str

    try:
        print(f'fetching model info for: {repo_id}')
        info = model_info(repo_id=repo_id, token=HF_TOKEN)
        current_tags = info.tags if info.tags else []
        print(f'found {len(current_tags)} tags: {current_tags}')

        result = {
            "status": "success",
            "repo_id": repo_id,
            "current_tags": current_tags,
            "count": len(current_tags)
        }

        json_str = json.dumps(result)
        print(f'get_current_tags returning: {str(json_str)}')
        return json_str

    except Exception as e:
        print(f'error in get_current_tags: {str(e)}')
        error_res = {'status': 'error', "repo_id": repo_id, 'error': str(e)}
        json_str = json.dumps(error_res)
        print(f"get_current_tags error returning: {json_str}")
        return json_str


@mcp.tool()
def add_new_tag(repo_id: str, new_tag: str) -> str:
    if not HF_API:
        err_result = {"error": "HF token not configured"}
        json_str = json.dumps(err_result)
        print(f"no HF API token - returning: {json_str}")
        return json_str

    try:
        print(f'fetching model info for: {repo_id}')
        info = model_info(repo_id=repo_id, token=HF_TOKEN)
        current_tags = info.tags if info.tags else []
        print(f'current tags: {current_tags}')

        if new_tag in current_tags:
            print(f"tag {new_tag} already exists in {current_tags}")
            res = {
                "status": "already_exists",
                "repo_id": repo_id,
                "tag": new_tag,
                "message": f"tag '{new_tag}' already exists",
            }
            json_str = json.dumps(res)
            print(f"add_new_tag (already exists) returning: {json_str}")
            return json_str

        updated_tags = current_tags + [new_tag]
        print(f"will update tags from {current_tags} to {updated_tags}")

        try:
            print(f'loading existing model card...')
            card = ModelCard.load(repo_id, token=HF_TOKEN)
            if not hasattr(card, 'data') or card.data is None:
                card.data = ModelCardData()

        except HfHubHTTPError:
            print('creating new model card (none exists)')
            card = ModelCard("")
            card.data = ModelCardData()

        card_dict = card.data.to_dict()
        card_dict['tags'] = updated_tags
        card.data = ModelCardData(**card_dict)

        pr_title = f"add '{new_tag}' tag"
        pr_description = f"""
        
        ## Add tag: {new_tag}
        
        This PR adds the `{new_tag}` tag to the model repository.
        
        **Changes:**
        - Added `{new_tag}` to model tags
        - Updated from {len(current_tags)} to {len(updated_tags)} tags
        
        **Current tags:** {", ".join(current_tags) if current_tags else "None"}
        **New tags:** {", ".join(updated_tags)}
        
        🤖 This is a pull request created by the Hugging Face Hub Tagging Bot.
        
        """

        print(f"creating PR with title: {pr_title}")

        commit_info = HF_API.create_commit(
            repo_id=repo_id,
            operations=[
                CommitOperationAdd(
                    path_in_repo='README.md', path_or_fileobj=str(card).encode('urf-8')
                )
            ],
            commit_message=pr_title,
            commit_description=pr_description,
            token=HF_TOKEN,
            create_pr=True
        )

        pr_url_attr = commit_info.pr_url
        pr_url = pr_url_attr if hasattr(commit_info, 'pr_url') else str(commit_info)

        print(f"PR created successfully! URL: {pr_url}")

        res = {
            'repo_id': repo_id,
            "status": "success",
            "tag": new_tag,
            "pr_url": pr_url,
            "previous_tags": current_tags,
            'new_tags': updated_tags,
            "message": f"created PR to add tag '{new_tag}'"
        }

        json_str = json.dumps(res)
        print(f"add_new_tag success returning: {json_str}")
        return json_str

    except Exception as e:
        print(f"❌ Error in add_new_tag: {str(e)}")
        print(f"❌ Error type: {type(e)}")

        print(f"traceback: {traceback.format_exc()}")

        err_res = {
            "status": 'error',
            "repo_id": repo_id,
            "tag": new_tag,
            "error": str(e),
        }

        json_str = json.dumps(err_res)
        print(f'add_new_tag error returning {json_str}')
        return json_str
