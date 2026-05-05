"""monolm chat"""

from monolm import chat, load_model, read, url_context, write_stream, write_user, load_config

if __name__ == '__main__':
    config = load_config()

    llm = load_model(config)

    tools_user = [url_context, read, write_user]
    tools_stream = [write_stream]

    print("monolm chat - type 'exit' to quit.\n")

    chat(llm, tools_user=tools_user, tools_stream=tools_stream)
