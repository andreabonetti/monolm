import contextlib
import os
from pathlib import Path

from llama_cpp import Llama


def load_model(
    model_path: str,
    n_ctx: int = 2048,  # Context length
    n_threads: int = 8,  # Number of threads to use for inference
    n_gpu_layers: int = 1,  # Metal offload
    seed: int = None,  # Seed for reproducibility
    verbose: bool = False,  # Disable verbose logging
):
    """Load the LLaMA model with error output suppressed."""
    with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f):
        BASE_DIR = Path(__file__).resolve().parent
        model_path = BASE_DIR / model_path
        model_path = str(model_path.resolve())

        return Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=verbose,
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
