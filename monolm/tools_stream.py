# ------------------------------------------------------------
# write
# ------------------------------------------------------------


def write_stream(stream: str, tools_state: dict) -> str:
    """Write content to a file."""
    if 'write_path' not in tools_state:
        return stream, tools_state  # No file specified, skip writing

    file_path = tools_state['write_path']
    code_block = False
    content = ''

    for line in tools_state['assistant_text'].splitlines():
        if line.startswith('<<<'):
            code_block = True
        elif line.endswith('>>>'):
            code_block = False

        if code_block:
            # Extract content between <<< and >>>
            if line.startswith('<<<'):
                content += line[3:].strip()  # remove <<< from the start
            elif line.endswith('>>>'):
                content += line[:-3].strip()  # remove >>> from the end
            else:
                content += line.strip()

    # write the content to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return stream, tools_state


# ------------------------------------------------------------
# git_commit
# ------------------------------------------------------------

def git_commit_stream(stream: str, tools_state: dict) -> str:
    """Stream the git commit message and write it to .git/COMMIT_EDITMSG."""

    if 'git_commit' not in tools_state or tools_state['git_commit'] is False:
        return stream, tools_state

    # execute stream as bash command
    import sys
    import subprocess

    print("Shall I execute the command? (Y/n)")

    try:
        with open("/dev/tty") as tty:
            confirm = tty.readline().strip()
    except OSError:
        print("No terminal available, aborting.")
        confirm = "n"
    
    if confirm == "Y":
        subprocess.run(["bash"], input=tools_state['assistant_text'], text=True)
    else:
        print("Aborted.")

    return stream, tools_state