# ------------------------------------------------------------
# write
# ------------------------------------------------------------


def write(user_input: str) -> str:
    """Write content to a file."""
    paths = _extract_paths(user_input, cmd='/write')

    user_input = _file_context(paths, user_input)

    return user_input