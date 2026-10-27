from pydantic import BaseModel
from typing import Any


class Prompt(BaseModel):
    """A natural-language request read from the prompts input file.

    Attributes:
        prompt: The text of the request, e.g. "What is the sum of 2 and 3?".
    """

    prompt: str


class ParamType(BaseModel):
    """The type description of a function parameter or return value.

    Attributes:
        type: The JSON type name, e.g. "number", "string" or "boolean".
    """

    type: str


class FunctionDef(BaseModel):
    """A function the model is allowed to call, from the definitions file.

    Attributes:
        name: The function name, e.g. "fn_add_numbers".
        description: What the function does, in plain language.
        parameters: Maps each parameter name to its type description.
        returns: The type description of the return value.
    """

    name: str
    description: str
    parameters: dict[str, ParamType]
    returns: ParamType


class FunctionCall(BaseModel):
    """One generated function call, written to the output file.

    Attributes:
        prompt: The original natural-language request.
        name: The name of the function chosen by the model.
        parameters: Maps each parameter name to its generated value.
    """

    prompt: str
    name: str
    parameters: dict[str, Any]
