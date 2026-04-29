import argparse

from monolm import _fetch_url_text, load_model, print_stream


def parse_args():
    parser = argparse.ArgumentParser(description='URL to markdown')
    parser.add_argument('url', type=str, help='URL to convert')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    llm = load_model(
        model_path='../models/gemma-4-E4B-it-Q4_K_M.gguf',
        n_ctx=131072,
        n_threads=8,
        n_gpu_layers=43,
    )

    content = ''
    content += 'Convert the following URL content into markdown format. No explanations.\n\n'
    page_text = _fetch_url_text(args.url)
    content += page_text[:12000]  # TODO: smarter truncation

    messages = [{'role': 'user', 'content': content}]

    stream = llm.create_chat_completion(
        messages=messages,
        stream=True,
    )

    print_stream(stream)
