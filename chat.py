import argparse
import contextlib
import os
import re

import requests
from bs4 import BeautifulSoup
from llama_cpp import Llama

MODEL_CONFIGS = {
    'gemma': {
        'model_path': 'models/gemma-4-E4B-it-Q4_K_M.gguf',
        'n_ctx': 131072,
        'n_gpu_layers': 43,
        'n_threads': 8,
    },
    'phi3': {
        'model_path': 'models/Phi-3-mini-4k-instruct-q4.gguf',
        'n_ctx': 4096,
        'n_gpu_layers': 33,
        'n_threads': 8,
    },
    'tiny': {
        'model_path': 'models/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf',
        'n_ctx': 2048,
        'n_gpu_layers': 22,
        'n_threads': 8,
    },
}


def load_model(config):
    """Load the LLaMA model with error output suppressed."""
    with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f):
        return Llama(
            model_path=config['model_path'],
            n_ctx=config['n_ctx'],
            n_threads=config['n_threads'],
            n_gpu_layers=config['n_gpu_layers'],
            verbose=False,
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Local chat with selectable model presets')
    parser.add_argument(
        '--preset',
        choices=sorted(MODEL_CONFIGS),
        default='phi3',
        help='Model preset to load',
    )
    return parser.parse_args()


def fetch_url_text(url: str) -> str:
    """Fetch the text content of a webpage, removing scripts and styles."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, 'html.parser')

    # remove junk
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()

    text = soup.get_text(separator='\n')
    return '\n'.join(line.strip() for line in text.splitlines() if line.strip())


def extract_url(text):
    """Extract the first URL from the given text."""
    match = re.search(r'https?://\S+', text)
    return match.group(0) if match else None


if __name__ == '__main__':
    args = parse_args()
    config = MODEL_CONFIGS[args.preset]

    llm = load_model(config)

    messages = []

    print(f"Local LLM ready ({args.preset}: {config['model_path']}). Type 'exit' to quit.\n")

    while True:
        user_input = input('you: ').strip()

        if user_input.lower() in ['exit', 'quit']:
            break

        url = extract_url(user_input)

        if url:
            page_text = fetch_url_text(url)

            user_input = (
                'You are given a webpage content. Use it as context.\n\n'
                f'URL: {url}\n\n'
                f'CONTENT:\n{page_text[:12000]}\n\n'
                f'QUESTION:\n{user_input}'
            )

        messages.append({'role': 'user', 'content': user_input})

        stream = llm.create_chat_completion(messages=messages, stream=True)

        print('assistant: ', end='', flush=True)

        assistant_text = ''

        for chunk in stream:
            delta = chunk['choices'][0]['delta'].get('content', '')
            print(delta, end='', flush=True)
            assistant_text += delta

        print('\n')

        messages.append({'role': 'assistant', 'content': assistant_text})
