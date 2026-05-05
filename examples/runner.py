"""monolm runner generates a response to a user query."""

from monolm import load_model, print_stream, load_config

if __name__ == '__main__':
    config = load_config()

    llm = load_model(config)

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
