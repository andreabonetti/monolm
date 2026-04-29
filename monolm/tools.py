import re

import requests
from bs4 import BeautifulSoup


def _extract_url(text):
    """Extract the first URL from the given text."""
    match = re.search(r'https?://\S+', text)
    return match.group(0) if match else None


def _fetch_url_text(url: str) -> str:
    """Fetch the text content of a webpage, removing scripts and styles."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, 'html.parser')

    # remove junk
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()

    text = soup.get_text(separator='\n')
    return '\n'.join(line.strip() for line in text.splitlines() if line.strip())


def url_context(user_input):
    """If URL is present, fetch its content for context."""
    url = _extract_url(user_input)

    if url:
        page_text = _fetch_url_text(url)
        user_input = (
            'You are given a webpage content. Use it as context.\n\n'
            f'URL: {url}\n\n'
            f'CONTENT:\n{page_text[:12000]}\n\n'
            f'QUESTION:\n{user_input}'
        )

    return user_input
