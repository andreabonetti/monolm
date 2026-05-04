"""git commit

usage:
    git diff --staged | python git_commit.py
"""

import sys

from monolm import chat, git_commit_stream, git_commit_user, load_model

if __name__ == '__main__':
    git_diff = sys.stdin.read()

    llm = load_model(model_path='../models/gemma-4-E4B-it-Q4_K_M.gguf')

    tools_user = [git_commit_user]

    tools_stream = [git_commit_stream]

    prompt = '/git_commit\n\n' + git_diff

    chat(llm, prompt=prompt, tools_user=tools_user, tools_stream=tools_stream)
