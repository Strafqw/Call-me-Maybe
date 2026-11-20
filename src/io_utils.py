import json
from .models import FunctionDef, Prompt, FunctionCall
from pathlib import Path


def load_functions(path: str) -> list[FunctionDef]:
    """Load and validate the function definitions file.

    Args:
        path: Path to the JSON file with the function definitions.

    Returns:
        The list of validated function definitions.

    Raises:
        OSError: If the file cannot be opened (e.g. it does not exist).
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the top-level JSON value is not a list.
        pydantic.ValidationError: If an item does not match FunctionDef.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of functions")
    return [FunctionDef.model_validate(item) for item in data]


def load_prompts(path: str) -> list[Prompt]:
    """Load and validate the prompts file.

    Args:
        path: Path to the JSON file with the natural-language prompts.

    Returns:
        The list of validated prompts.

    Raises:
        OSError: If the file cannot be opened (e.g. it does not exist).
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the top-level JSON value is not a list.
        pydantic.ValidationError: If an item does not match Prompt.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of prompts")
    return [Prompt.model_validate(item) for item in data]


def write_results(path: str, results: list[FunctionCall]) -> None:
    """Write the generated function calls to a JSON file.

    Creates the parent directory of the file if it does not exist yet.

    Args:
        path: Path of the output JSON file.
        results: The function calls to write, one per prompt.

    Raises:
        OSError: If the directory or the file cannot be written.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = [call.model_dump() for call in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
