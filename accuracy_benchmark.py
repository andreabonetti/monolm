import argparse
import contextlib
import json
import os
import re

from llama_cpp import Llama

DEFAULT_DATASET = [
    {'prompt': 'What is the capital of France?', 'expected': 'Paris'},
    {'prompt': 'What is 2 + 2?', 'expected': '4'},
    {'prompt': 'Water freezes at what temperature in Celsius?', 'expected': '0'},
    {
        'prompt': 'What is the largest planet in the Solar System?',
        'expected': 'Jupiter',
    },
    {'prompt': 'Which planet is known as the Red Planet?', 'expected': 'Mars'},
    {'prompt': 'What is the chemical symbol for gold?', 'expected': 'Au'},
    {'prompt': 'Who wrote Hamlet?', 'expected': 'William Shakespeare'},
    {'prompt': 'How many days are in a leap year?', 'expected': '366'},
    {
        'prompt': 'Which gas do plants absorb from the atmosphere?',
        'expected': 'carbon dioxide',
    },
    {'prompt': 'What is H2O commonly called?', 'expected': 'water'},
]


def load_model(model_path, n_ctx=2048, n_threads=8, n_gpu_layers=1, seed=0):
    """Load the model with noisy stderr output suppressed."""
    with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f):
        return Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=False,
        )


def load_dataset(path=None, limit=None):
    """Load benchmark examples from JSON/JSONL or fall back to a built-in set."""
    if path is None:
        dataset = DEFAULT_DATASET
    elif path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f if line.strip()]
    else:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a list of objects with 'prompt' and 'expected'.")

    normalized_dataset = []
    for i, item in enumerate(dataset, start=1):
        if not isinstance(item, dict):
            raise ValueError(f'Dataset item {i} is not an object.')
        if 'prompt' not in item or 'expected' not in item:
            raise ValueError(f"Dataset item {i} must include 'prompt' and 'expected'.")
        normalized_dataset.append(
            {
                'prompt': str(item['prompt']).strip(),
                'expected': str(item['expected']).strip(),
            }
        )

    if limit is not None:
        normalized_dataset = normalized_dataset[:limit]

    return normalized_dataset


def normalize_text(text):
    """Normalize text so small formatting differences do not dominate scoring."""
    text = text.strip().lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def contains_expected(actual, expected):
    """Check whether the normalized expected answer appears in the normalized output."""
    actual_normalized = normalize_text(actual)
    expected_normalized = normalize_text(expected)
    return expected_normalized in actual_normalized


def exact_match(actual, expected):
    """Check whether the normalized response exactly matches the expected answer."""
    return normalize_text(actual) == normalize_text(expected)


def generate_answer(llm, prompt, max_tokens=32, temperature=0.0, instruction=None):
    """Generate a single response for a benchmark example."""
    user_prompt = prompt
    if instruction:
        user_prompt = f'{prompt}\n\n{instruction}'

    response = llm.create_chat_completion(
        messages=[{'role': 'user', 'content': user_prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )

    return response['choices'][0]['message']['content'].strip()


def benchmark(llm, dataset, max_tokens=32, temperature=0.0, instruction=None):
    """Run the accuracy benchmark and collect aggregate metrics."""
    exact_hits = 0
    contains_hits = 0
    rows = []

    print('\n--- benchmark ---')

    for i, example in enumerate(dataset, start=1):
        actual = generate_answer(
            llm,
            prompt=example['prompt'],
            max_tokens=max_tokens,
            temperature=temperature,
            instruction=instruction,
        )

        is_exact = exact_match(actual, example['expected'])
        has_expected = contains_expected(actual, example['expected'])

        exact_hits += int(is_exact)
        contains_hits += int(has_expected)

        rows.append(
            {
                'prompt': example['prompt'],
                'expected': example['expected'],
                'actual': actual,
                'exact_match': is_exact,
                'contains_expected': has_expected,
            }
        )

        status = 'OK' if has_expected else 'MISS'
        print(
            f'example {i}/{len(dataset)} | {status} | '
            f'exact={is_exact} | expected={example["expected"]!r} | actual={actual!r}'
        )

    total = len(dataset)
    return {
        'examples': total,
        'exact_match_accuracy': exact_hits / total if total else 0.0,
        'contains_expected_accuracy': contains_hits / total if total else 0.0,
        'rows': rows,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Local LLM accuracy benchmark')

    parser.add_argument('--model', type=str, required=True, help='Path to GGUF model')
    parser.add_argument('--dataset', type=str, default=None, help='Optional JSON or JSONL dataset')
    parser.add_argument('--ctx', type=int, default=2048, help='Context size')
    parser.add_argument('--threads', type=int, default=8, help='CPU threads')
    parser.add_argument('--gpu', type=int, default=1, help='GPU layers')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducible runs')
    parser.add_argument('--tokens', type=int, default=32, help='Max tokens per answer')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument(
        '--instruction',
        type=str,
        default='Answer with only the final answer and no explanation.',
        help='Instruction appended to each prompt',
    )
    parser.add_argument(
        '--limit', type=int, default=None, help='Only evaluate the first N examples'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dataset = load_dataset(path=args.dataset, limit=args.limit)

    llm = load_model(
        model_path=args.model,
        n_ctx=args.ctx,
        n_threads=args.threads,
        n_gpu_layers=args.gpu,
        seed=args.seed,
    )

    stats = benchmark(
        llm,
        dataset=dataset,
        max_tokens=args.tokens,
        temperature=args.temperature,
        instruction=args.instruction,
    )

    print('\n--- summary ---')
    print(f'Examples:                  {stats["examples"]}')
    print(f'Exact match accuracy:      {stats["exact_match_accuracy"]:.2%}')
    print(f'Contains-expected accuracy:{stats["contains_expected_accuracy"]:.2%}')
