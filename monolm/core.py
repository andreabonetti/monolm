import contextlib
import os

from llama_cpp import Llama


def load_model(
    model_path: str,
    n_ctx: int = 2048,  # Context length
    n_threads: int = 8,  # Number of threads to use for inference
    n_gpu_layers: int = 1,  # Metal offload
    verbose: bool = False,  # Disable verbose logging
):
    """Load the LLaMA model with error output suppressed."""
    with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f):
        return Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )


def print_stream(stream):
    """Print the streamed response from the model."""
    for chunk in stream:
        delta = chunk['choices'][0]['delta'].get('content', '')
        print(delta, end='', flush=True)
    print()


def chat(llm, tools: list =[]):
    """Start an interactive chat session with the LLM, optionally using tools."""

    messages = []

    while True:
        user_input = input('you: ').strip()

        if user_input.lower() in ['exit', 'quit']:
            break

        # tools run on every user input
        for tool in tools:
            user_input = tool(user_input)

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