# ------------------------------------------------------------
# write
# ------------------------------------------------------------

def write_stream(stream: str, tools_state: dict) -> str:
    """Write content to a file."""
    if 'write_path' not in tools_state:
        return stream, tools_state  # No file specified, skip writing
    
    file_path = tools_state['write_path']
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

    return stream, tools_state