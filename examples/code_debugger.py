"""code debugger"""

from monolm import chat, load_model, read, url_context, write_stream, write_user

if __name__ == '__main__':
    llm = load_model(model_path='../models/gemma-4-E4B-it-Q4_K_M.gguf')

    tools_user = [url_context, read, write_user]
    tools_stream = [write_stream]

    prompt = 'debug /write data/bugged_hello_world.py'

    chat(llm, prompt=prompt, tools_user=tools_user, tools_stream=tools_stream)
