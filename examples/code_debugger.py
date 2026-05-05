"""code debugger"""

from monolm import chat, load_config, load_model, read, url_context, write_stream, write_user

if __name__ == '__main__':
    config = load_config()

    llm = load_model(config)

    tools_user = [url_context, read, write_user]
    tools_stream = [write_stream]

    prompt = 'debug /write data/bugged_hello_world.py'

    chat(llm, prompt=prompt, tools_user=tools_user, tools_stream=tools_stream)
