"""Runner script to load the LLM model and generate a response to a user query."""

from monolm import load_model, print_stream

if __name__ == '__main__':
    llm = load_model(
        model_path='models/gemma-4-E4B-it-Q4_K_M.gguf',
        n_ctx=131072,
        n_threads=8,
        n_gpu_layers=43,
    )

    stream = llm.create_chat_completion(
        messages=[
            {
                'role': 'user',
                'content': 'What is Paris known for? Give me a one sentence answer.',
            }
        ],
        stream=True,
    )

    print_stream(stream)
