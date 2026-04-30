import re

import requests
from bs4 import BeautifulSoup


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
            output_format="markdown",
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
