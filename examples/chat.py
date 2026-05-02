"""monolm chat"""

from monolm import chat, read, load_model, url_context, write

if __name__ == '__main__':
    llm = load_model(
        model_path='../models/gemma-4-E4B-it-Q4_K_M.gguf',
        n_ctx=131072,
        n_threads=8,
        n_gpu_layers=43,
    )

    tools_user = [url_context, read, write]

    print("monolm chat - type 'exit' to quit.\n")

    chat(llm, tools_user=tools_user)
