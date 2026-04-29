
from monolm import load_model, url_context

if __name__ == '__main__':

    config = {
        'model': 'models/gemma-4-E4B-it-Q4_K_M.gguf',
        'n_ctx': 131072,
        'n_threads': 8,
        'n_gpu_layers': 43,
    }

    llm = load_model(
        model_path=config['model'],
        n_ctx=config['n_ctx'],
        n_threads=config['n_threads'],
        n_gpu_layers=config['n_gpu_layers'],
    )

    print(f"Local LLM ready: {config['model']}. Type 'exit' to quit.\n")

    messages = []

    while True:
        user_input = input('you: ').strip()

        if user_input.lower() in ['exit', 'quit']:
            break

        user_input = url_context(user_input)

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
