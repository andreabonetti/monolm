# ------------------------------------------------------------
# write
# ------------------------------------------------------------

def write_stream(stream: str, file_path: str) -> str:
    """Write content to a file."""
    code_block = False

    for chunk in stream:
        delta = chunk['choices'][0]['delta'].get('content', '')

        if delta.startswith('<<<'):
            code_block = True
        elif delta.endswith('>>>'):
            code_block = False
        
        if code_block:
            # Extract content between <<< and >>>
            if delta.startswith('<<<'):
                content = delta[3:].strip()  # remove <<< from the start
            elif delta.endswith('>>>'):
                content = delta[:-3].strip()  # remove >>> from the end
            else:
                content = delta.strip()

            # write the content to the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

    return stream