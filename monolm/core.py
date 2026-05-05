import contextlib
import os
from pathlib import Path

import yaml
from llama_cpp import Llama


def load_config(yaml_path: str = '~/.monolm/config.yaml') -> dict:
    path = Path(yaml_path).expanduser()

    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_model(
    config: dict,
    seed: int = 42,  # Seed for reproducibility
    verbose: bool = False,  # Disable verbose logging
):
    """Load the LLaMA model with error output suppressed."""
    with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f):
        return Llama(
            model_path=config['model_path'],
            n_ctx=config.get('n_ctx', 2048),
            n_threads=config.get('n_threads', 8),
            n_gpu_layers=config.get('n_gpu_layers', 50),
            seed=config.get('seed', 42),
            verbose=config.get('verbose', False),
        )


def print_stream(stream):
    """Print the streamed response from the model."""
    for chunk in stream:
        delta = chunk['choices'][0]['delta'].get('content', '')
        print(delta, end='', flush=True)
    print()


def chat(llm, prompt=None, tools_user: list = [], tools_stream: list = []):
    """Start an interactive chat session with the LLM, optionally using tools."""

    messages = []
    tools_state = {}

    while True:
        user_input = prompt if prompt is not None else input('you: ').strip()

        if user_input.lower() in ['/exit', '/quit']:
            break

        # tools run on every user input
        for tool in tools_user:
            user_input, tools_state = tool(user_input, tools_state)

        # append the user's input to the conversation history
        messages.append({'role': 'user', 'content': user_input})

        # create a streaming chat completion
        stream = llm.create_chat_completion(messages=messages, stream=True)

        # print the assistant's response as it streams in
        if prompt is None:
            print('assistant: ', end='', flush=True)
        assistant_text = ''
        for chunk in stream:
            delta = chunk['choices'][0]['delta'].get('content', '')
            print(delta, end='', flush=True)
            assistant_text += delta
        print('\n')

        # add assistant_text to tools_state
        tools_state['assistant_text'] = assistant_text

        # tools run on every stream
        for tool in tools_stream:
            stream, tools_state = tool(stream, tools_state)

        # append the assistant's full response to the conversation history
        messages.append({'role': 'assistant', 'content': assistant_text})

        # exit after single prompt mode
        if prompt is not None:
            break
