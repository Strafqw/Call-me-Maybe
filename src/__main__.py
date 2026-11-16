import argparse
import json
import sys

from llm_sdk import Small_LLM_Model
from pydantic import ValidationError

from .decoder import build_prompt, choose_name, generate_arguments
from .io_utils import load_functions, load_prompts, write_results
from .models import FunctionCall
from .vocab import load_token_texts

DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calls.json"


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns:
        The parsed arguments with the three file paths.
    """
    parser = argparse.ArgumentParser(
        description="Translate natural language prompts into function calls"
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to the JSON file with the prompts."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to the output file with the results."
    )
    parser.add_argument(
        "--functions_definition",
        default=DEFAULT_FUNCTIONS,
        help="Path to the JSON file with the available function definitions."
    )
    return parser.parse_args()


def main() -> None:
    """Run the function calling program from the command line."""
    args = parse_args()
    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
    except FileNotFoundError as e:
        print(f"Error: file not found: {e.filename}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error: cannot read file: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except (ValidationError, ValueError) as e:
        print(f"Error: invalid input format: {e}", file=sys.stderr)
        sys.exit(1)

    if not functions:
        print("Error: no functions defined, nothing to call.",
              file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(functions)} functions and {len(prompts)} prompts.")

    try:
        model = Small_LLM_Model()
        vocab_size = len(model.get_logits_from_input_ids([0]))
        token_texts = load_token_texts(
            model.get_path_to_vocab_file(), vocab_size
        )
    except Exception as e:
        print(f"Error: cannot load the model: {e}", file=sys.stderr)
        sys.exit(1)

    names = [f.name for f in functions]
    results: list[FunctionCall] = []
    for p in prompts:
        try:
            text = build_prompt(functions, p.prompt)
            ids = model.encode(text)[0].tolist()
            name = choose_name(model, token_texts, ids, names)
            function = next(f for f in functions if f.name == name)
            parameters = generate_arguments(model, token_texts, ids,
                                            function)
        except Exception as e:
            print(f"Error on prompt {p.prompt!r}: {e}", file=sys.stderr)
            continue
        results.append(FunctionCall(prompt=p.prompt, name=name,
                                    parameters=parameters))
        print(f"{p.prompt!r} -> {name} {parameters}")

    try:
        write_results(args.output, results)
    except OSError as e:
        print(f"Error: cannot write output: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
