import argparse
import json
import sys

from pydantic import ValidationError

from .io_utils import load_functions, load_prompts, write_results
from .models import FunctionCall

DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"


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

    print(f"Loaded {len(functions)} functions and {len(prompts)} prompts.")

    # Placeholder until the decoder exists: no function chosen yet.
    results = [
        FunctionCall(prompt=p.prompt, name="TODO", parameters={})
        for p in prompts
    ]

    try:
        write_results(args.output, results)
    except OSError as e:
        print(f"Error: cannot write output: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
