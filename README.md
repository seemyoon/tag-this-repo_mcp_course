---
title: tag-a-repo bot
emoji: 👀
colorFrom: purple
colorTo: yellow
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: false
base_path: /gradio
---

# HF Tagging Bot

This is a bot that tags HuggingFace models when they are mentioned in discussions.

## How it works

1. The bot listens to discussions on the HuggingFace Hub
2. When a discussion is created, the bot checks for tag mentions in the comment
3. If a tag is mentioned, the bot adds the tag to the model repository via a PR