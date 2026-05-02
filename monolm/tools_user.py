import re
from pathlib import Path

import requests
import trafilatura

# ------------------------------------------------------------
# common
# ------------------------------------------------------------


def _extract_paths(text: str, cmd: str = '/read'):
    """
    Extract paths from commands like:
    /read path/to/file
    /read "path with spaces/file.txt"
    """
    pattern = rf'{cmd}\s+(?:"([^"]+)"|(\S+))'
    matches = re.findall(pattern, text)

    # each match is a tuple (quoted, unquoted)
    paths = [m[0] or m[1] for m in matches]
    return paths


def _read_file(path: str, max_chars: int = 12000) -> str:
    """Safely read a file from disk."""
    path = Path(path).expanduser().resolve()

    if not path.exists() or not path.is_file():
        raise ValueError('File not found')

    content = path.read_text(encoding='utf-8', errors='ignore')
    return content[:max_chars]


def _file_context(paths: list, user_input: str) -> str:
    for path in paths:
        try:
            content = _read_file(path)

            user_input += (
                'You are given a local file. Use it as context.\n\n'
                f'PATH: {path}\n\n'
                f'CONTENT:\n{content}\n\n'
            )
        except Exception as e:
            user_input += f'\n\n[File read error: {e}]'

    return user_input


# ------------------------------------------------------------
# url_context
# ------------------------------------------------------------


def _extract_url(text):
    """Extract the first URL from the given text."""
    match = re.search(r'https?://\S+', text)
    return match.group(0) if match else None


def _fetch_html(url: str) -> str:
    """Fetch the HTML content of a webpage."""
    response = requests.get(url, timeout=3)
    response.raise_for_status()
    html = response.text
    return html


def url_context(user_input):
    """If URL is present, fetch its content for context."""
    url = _extract_url(user_input)

    if url:
        html = _fetch_html(url)

        markdown = trafilatura.extract(
            html,
            output_format='markdown',
            include_links=True,
            include_images=False,
            include_formatting=True,
        )

        user_input = (
            'You are given a webpage content. Use it as context.\n\n'
            f'URL: {url}\n\n'
            f'CONTENT:\n{markdown}\n\n'
            f'QUESTION:\n{user_input}'
        )

    return user_input


# ------------------------------------------------------------
# read
# ------------------------------------------------------------


def read(user_input: str) -> str:
    """Read file from path, load it as context."""
    paths = _extract_paths(user_input, cmd='/read')

    user_input = _file_context(paths, user_input)

    return user_input


# ------------------------------------------------------------
# write
# ------------------------------------------------------------


def write(user_input: str) -> str:
    """Write content to a file."""
    paths = _extract_paths(user_input, cmd='/write')

    user_input = _file_context(paths, user_input)

    user_input += """
You are a coding assistant with access to a file editing tool.

When you want to modify a file, you MUST emit an edit block using the exact format below.

Rules:
- Emit the FULL new file content.
- Do NOT emit partial diffs.
- Do NOT explain the changes inside the edit block.
- Do NOT truncate the file.
- Preserve existing code unless intentionally changing it.
- Always include the file path.
- Use UTF-8 text only.
- You may include normal conversational text before or after the edit block.

Exact format:

/wite path/to/file.py
<<<
FULL FILE CONTENT HERE
>>>

Example:

/write hello.py
<<<
def main():
    print("hello")

if __name__ == "__main__":
    main()
>>>

Never use markdown code fences around edit blocks.

If multiple files must be changed, emit multiple edit blocks.

Only emit edit blocks when explicitly asked to create or modify files.
"""

    return user_input
