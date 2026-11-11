import re
from typing import Any, Callable

from llm_sdk import Small_LLM_Model

from .models import FunctionDef

# Text that can still become a number, and a complete number.
# Both allow one leading space, because the model often writes " -4".
NUMBER_START = re.compile(r" ?-?\d*(\.\d*)?")
NUMBER_DONE = re.compile(r" ?-?\d+(\.\d+)?")


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


def generate(
    model: Small_LLM_Model,
    token_texts: list[str],
    ids: list[int],
    candidates: list[int],
    is_valid: Callable[[str], bool],
    is_finished: Callable[[str], bool],
    max_tokens: int = 50,
) -> str:
    """Generate text token by token, allowing only valid continuations.

    Args:
        model: The loaded language model.
        token_texts: Real text of each token id.
        ids: Token ids so far. Chosen tokens are appended to it.
        candidates: The only token ids that may be chosen.
        is_valid: Says whether a generated text is still acceptable.
        is_finished: Says whether a generated text is complete.
        max_tokens: Safety limit on the number of generated tokens.

    Returns:
        The generated text.

    Raises:
        ValueError: If no candidate is valid, or the limit is reached.
    """
    generated = ""
    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(ids)
        allowed = [t for t in candidates
                   if is_valid(generated + token_texts[t])]
        if not allowed:
            raise ValueError(f"no valid continuation after {generated!r}")
        best = max(allowed, key=lambda t: logits[t])
        ids.append(best)
        generated += token_texts[best]
        if is_finished(generated):
            return generated
    raise ValueError(f"value too long: {generated!r}")


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
    text = generate(
        model, token_texts, prompt_ids, candidates,
        is_valid=lambda s: any(t.startswith(s) for t in targets),
        is_finished=lambda s: s in targets,
    )
    return text[:-1]


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


def force(model: Small_LLM_Model, ids: list[int], text: str) -> None:
    """Append the tokens of a fixed text to the ids, without the model.

    Args:
        model: The loaded language model (used for its tokenizer).
        ids: Token ids so far. The new tokens are appended to it.
        text: The text to append.
    """
    ids.extend(model.encode(text)[0].tolist())


def generate_number(
    model: Small_LLM_Model,
    token_texts: list[str],
    ids: list[int],
    end: str,
) -> float:
    """Let the model write a number, followed by the end character.

    Args:
        model: The loaded language model.
        token_texts: Real text of each token id.
        ids: Token ids so far. Chosen tokens are appended to it.
        end: The character that closes the value ("," or "}").

    Returns:
        The generated number.
    """
    candidates = [
        t for t, s in enumerate(token_texts)
        if s and (s == end or all(c in " 0123456789-." for c in s))
    ]

    def is_finished(s: str) -> bool:
        return s.endswith(end) and NUMBER_DONE.fullmatch(s[:-1]) is not None

    def is_valid(s: str) -> bool:
        return NUMBER_START.fullmatch(s) is not None or is_finished(s)

    text = generate(model, token_texts, ids, candidates,
                    is_valid, is_finished, max_tokens=30)
    return float(text[:-1])


def generate_boolean(
    model: Small_LLM_Model,
    token_texts: list[str],
    ids: list[int],
    end: str,
) -> bool:
    """Let the model write true or false, followed by the end character.

    Args:
        model: The loaded language model.
        token_texts: Real text of each token id.
        ids: Token ids so far. Chosen tokens are appended to it.
        end: The character that closes the value ("," or "}").

    Returns:
        The generated boolean.
    """
    targets = [p + w + end for p in ("", " ") for w in ("true", "false")]
    text = generate(
        model, token_texts, ids, find_candidates(token_texts, targets),
        is_valid=lambda s: any(t.startswith(s) for t in targets),
        is_finished=lambda s: s in targets,
    )
    return text.strip().startswith("true")


def generate_arguments(
    model: Small_LLM_Model,
    token_texts: list[str],
    ids: list[int],
    function: FunctionDef,
) -> dict[str, Any]:
    """Generate a value for every parameter of the chosen function.

    The JSON structure (keys, colons, commas, braces) is forced; the model
    only writes the values.

    Args:
        model: The loaded language model.
        token_texts: Real text of each token id.
        ids: Token ids so far, ending right after the function name.
        function: The definition of the chosen function.

    Returns:
        A dict mapping each parameter name to its generated value.

    Raises:
        ValueError: If a parameter has an unsupported type.
    """
    force(model, ids, ', "parameters": {')
    args: dict[str, Any] = {}
    names = list(function.parameters)
    for i, name in enumerate(names):
        end = "}" if i == len(names) - 1 else ","
        force(model, ids, f'"{name}":' if i == 0 else f' "{name}":')
        ptype = function.parameters[name].type
        if ptype == "number":
            args[name] = generate_number(model, token_texts, ids, end)
        elif ptype == "boolean":
            args[name] = generate_boolean(model, token_texts, ids, end)
        else:
            raise ValueError(f"unsupported parameter type: {ptype}")
    return args
