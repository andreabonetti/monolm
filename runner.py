"""Runner script to load the LLM model and generate a response to a user query."""

import contextlib
import os

from llama_cpp import Llama


def load_model():
    """Load the LLaMA model with error output suppressed."""
    with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f):
        return Llama(
            model_path='models/Phi-3-mini-4k-instruct-q4.gguf',
            n_ctx=2048,  # Context length
            n_threads=8,  # Number of threads to use for inference
            n_gpu_layers=1,  # Metal offload
            verbose=False,  # Disable verbose logging
        )


if __name__ == '__main__':
    llm = load_model()

    stream = llm.create_chat_completion(
        messages=[
            {
                'role': 'user',
                'content': 'What is Paris known for? Give me a one sentence answer.',
            }
        ],
        stream=True,
    )

    for chunk in stream:
        delta = chunk['choices'][0]['delta'].get('content', '')
        print(delta, end='', flush=True)
