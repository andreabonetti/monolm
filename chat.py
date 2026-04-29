
from monolm import load_model, chat, url_context

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

    tools=[url_context]

    print(f"monolm chat")
    print(f"→ llm: {config['model']}")
    print(f"→ tools: {', '.join([tool.__name__ for tool in tools])}")
    print(f"type 'exit' to quit.\n")



    chat(llm, tools=[url_context])