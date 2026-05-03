"""git commit

usage:
    git diff --staged | python git_commit.py
"""

import sys

from monolm import chat, git_commit, load_model

if __name__ == '__main__':
    git_diff = sys.stdin.read()

    llm = load_model(
        model_path='../models/gemma-4-E4B-it-Q4_K_M.gguf',
        n_ctx=131072,
        n_threads=8,
        n_gpu_layers=43,
    )

    tools_user = [git_commit]

    prompt = git_diff + '\n\n/git_commit'

    chat(llm, prompt=prompt, tools_user=tools_user)
