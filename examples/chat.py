"""monolm chat"""

from monolm import chat, read, load_model, url_context

if __name__ == '__main__':
    llm = load_model(
        model_path='../models/gemma-4-E4B-it-Q4_K_M.gguf',
        n_ctx=131072,
        n_threads=8,
        n_gpu_layers=43,
    )

    tools = [url_context, read]

    print("monolm chat - type 'exit' to quit.\n")

    chat(llm, tools=tools)
