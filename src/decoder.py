from llm_sdk import Small_LLM_Model
from .models import FunctionDef


def find_candidates(token_texts: list[str], targets: list[str]) -> list[int]:
    """Find the token ids whose text appears inside at least one target.

    Args:
        token_texts: Real text of each token id (from load_token_texts).
        targets: The strings that may be generated, e.g. 'fn_greet"'.

    Returns:
        The ids of the tokens that could be used to build a target.
    """
    ids: list[int] = []
    for token_id, text in enumerate(token_texts):
        if text and any(text in target for target in targets):
            ids.append(token_id)
    return ids


def choose_name(
    model: Small_LLM_Model,
    token_texts: list[str],
    prompt_ids: list[int],
    names: list[str],
) -> str:
    """Let the model pick one of the function names, token by token.

    Args:
        model: The loaded language model.
        token_texts: Real text of each token id.
        prompt_ids: Token ids of the prompt, ending right where the
            name should start (after '{"name": "').
        names: The valid function names.

    Returns:
        The function name chosen by the model.

    Raises:
        ValueError: If no allowed token can continue the name.
    """
    targets = [name + '"' for name in names]
    candidates = find_candidates(token_texts, targets)
    ids = list(prompt_ids)
    generated = ""
    while generated not in targets:
        logits = model.get_logits_from_input_ids(ids)
        allowed = [
            t for t in candidates
            if any(target.startswith(generated + token_texts[t])
                   for target in targets)
        ]
        if not allowed:
            raise ValueError(f"no valid continuation after {generated!r}")
        best = max(allowed, key=lambda t: logits[t])
        ids.append(best)
        generated += token_texts[best]
    return generated[:-1]


def build_prompt(functions: list[FunctionDef], request: str) -> str:
    """Build the text the model reads before choosing a function.

    Lists every available function with its parameters and description,
    then the user's request, and ends right where the name must start.

    Args:
        functions: The available function definitions.
        request: The user's natural-language request.

    Returns:
        The prompt text, ending with '{"name": "'.
    """
    lines = [
        "You translate requests into function calls.",
        "Available functions:",
    ]
    for f in functions:
        params = ", ".join(
            f"{name}: {p.type}" for name, p in f.parameters.items()
        )
        lines.append(f"- {f.name}({params}): {f.description}")
    lines.append("")
    lines.append(f"Request: {request}")
    lines.append('Call: {"name": "')
    return "\n".join(lines)
