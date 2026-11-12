*This project has been created as part of the 42 curriculum by joasampa.*

# call me maybe

## Description

**call me maybe** turns natural-language requests into structured function calls
using a small language model (Qwen3-0.6B).

Given the request `"What is the sum of 2 and 3?"` and a list of available functions,
the program does not answer `5`. It outputs the function to call and its typed
arguments:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

A 0.6B model asked to "answer in JSON" often produces invalid JSON, picks a function
that does not exist, or keeps writing after the answer. This project uses
**constrained decoding** instead: at every generation step, only the tokens that
keep the output valid are allowed. The model still makes every choice (which
function, which values), but the output is guaranteed to be valid JSON that matches
the function definitions.

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/)
- Internet access on the first run (the model, about 1.5 GB, is downloaded from
  Hugging Face and cached)

### Installation

```bash
make install        # same as: uv sync
```

### Running

```bash
make run            # same as: uv run python -m src
```

With custom files:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

All three arguments are optional; the values above are the defaults.

### Other Makefile rules

| Rule | Effect |
|---|---|
| `make debug` | Run the program under `pdb` |
| `make lint` | `flake8` and `mypy` with the flags required by the subject |
| `make lint-strict` | `flake8` and `mypy --strict` |
| `make clean` | Remove `__pycache__` and `.mypy_cache` |

### Input and output

- `functions_definition.json`: a list of functions, each with a `name`, a
  `description`, `parameters` (name → `{"type": ...}`) and `returns`.
- `function_calling_tests.json`: a list of `{"prompt": "..."}` objects.
- The output is a list of `{"prompt", "name", "parameters"}` objects, one per
  prompt.

Supported parameter types: `number` (written as a float), `string` and `boolean`.

## Example usage

```
$ make run
Loaded 5 functions and 11 prompts.
'What is the sum of 2 and 3?' -> fn_add_numbers {'a': 2.0, 'b': 3.0}
'Greet shrek' -> fn_greet {'name': 'shrek'}
"Reverse the string 'hello'" -> fn_reverse_string {'s': 'hello'}
'What is the square root of 16?' -> fn_get_square_root {'a': 16.0}
"Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'" -> fn_substitute_string_with_regex {'source_string': 'The cat sat on the mat with another cat', 'regex': 'cat', 'replacement': 'dog'}
...
Wrote 11 results to data/output/function_calling_results.json
```

Errors in the input files are reported without a crash:

```
$ uv run python -m src --input missing.json
Error: file not found: missing.json

$ uv run python -m src --input broken.json
Error: invalid JSON: Expecting property name enclosed in double quotes: line 1 column 17 (char 16)
```

## Algorithm explanation

### How the model generates text

The model never sees text, only **tokens**: small pieces of text with an id
(`"Greet shrek"` becomes `G`, `reet`, ` sh`, `rek`). Given a list of token ids,
`get_logits_from_input_ids` returns one score (**logit**) for each of the 151936
possible next tokens. Generation is a loop: get the scores, pick a token, append it,
repeat.

### Constrained decoding

Constrained decoding changes only the "pick a token" step: instead of taking the
best token overall, the program takes the best token **among the ones that keep the
output valid**. This is equivalent to setting the logits of all invalid tokens to
negative infinity.

All of this lives in one function, `generate` (in `src/decoder.py`), which takes
the rules as parameters:

- `candidates`: the only token ids that may ever be chosen,
- `is_valid(text)`: whether the text generated so far can still become a correct value,
- `is_finished(text)`: whether the value is complete.

```
generated = ""
repeat (at most max_tokens times):
    logits  = model scores for the next token
    allowed = [t in candidates if is_valid(generated + text_of(t))]
    best    = allowed token with the highest logit
    append best to the input, add its text to generated
    if is_finished(generated): return generated
```

### Forced structure vs. model choices

The program writes every part of the JSON it already knows, and the model writes
only the parts that depend on the request:

```
{"name": "  fn_add_numbers"  , "parameters": {"a":   2,  "b":   3}
 forced     model             forced          model forced model
```

Forced text is appended directly with `encode`, without calling the model. This is
faster and cannot go wrong.

### Rules for each part

| Part | Allowed tokens | Finished when |
|---|---|---|
| function name | pieces of the valid names, as long as the text is still the start of `<name>"` | the text equals one full `<name>"` |
| number | digits, `-`, `.`, an optional leading space, then the end character | a complete number (`-?\d+(\.\d+)?`) followed by `,` or `}` |
| string | any token without `"`, `\`, newline or broken characters, plus a token that closes the string | the text ends with `"` followed by `,` or `}` |
| boolean | pieces of `true` / `false` (optional leading space) | `true` or `false` followed by `,` or `}` |

The "end character" is `,` if another parameter follows and `}` for the last one,
so the model also decides where each value stops.

The **function name is chosen by the model**: at every step several names are
still possible, and the model's scores decide which way to continue.

### Decoding the vocabulary

The vocabulary file stores tokens in a disguised form: invisible bytes are replaced
by visible characters (a space is stored as `Ġ`, a newline as `Ċ`). This is the
byte-to-character table from GPT-2, which Qwen reuses. `src/vocab.py` rebuilds that
table (`bytes_to_unicode`), reverses it, and decodes every token into its real text
once at startup. The result is a list `token_texts` where `token_texts[id]` is the
real text of a token, which is what the rules above check.

## Design decisions

- **One generic loop, small rule functions.** Names, numbers, strings and booleans
  all use the same `generate` function with different rules, so the decoding logic
  exists only once.
- **Force everything that is known.** Keys, quotes, colons, commas and braces come
  from the function definition, so the model is never asked to produce them. This
  removes a whole class of errors and saves model calls.
- **Pre-filtered candidates.** Each rule first selects the few tokens that could
  ever be valid (for example, only digit-like tokens for a number), so each step
  checks a short list instead of all 151936 tokens.
- **The prompt lists every function with its description.** Constraints guarantee
  the format; the descriptions let the model pick the right function even when the
  names are not self-explanatory.
- **No backslashes, quotes or newlines inside strings.** This avoids JSON escaping
  entirely. The cost: regexes are written as `[0-9]+` rather than `\d+`.
- **Optional leading space in values.** The model naturally writes `" -4"` as the
  token `" -"` followed by `4`. Forcing a lone space before the value made the minus
  sign unlikely, so the space is left to the model instead.
- **Robust error handling.** Missing files, invalid JSON, wrong structure (checked
  with pydantic) and model loading problems print a clear message instead of a
  traceback. Each prompt is processed in its own `try`, so one failure does not stop
  the others.
- **Vocabulary size measured, not hardcoded.** The number of logits is read from one
  model call, so the token list always matches the model.

## Performance analysis

- **Accuracy:** 11/11 prompts of the provided test set get the correct function and
  correct arguments, including the three regex substitutions.
- **Validity:** 100% of the outputs are valid JSON matching the definitions, by
  construction: an invalid token can never be chosen.
- **Speed:** the full test set (11 prompts) runs in about **4 min 30 s** on a laptop
  CPU (Intel Core Ultra 7 255H, no GPU, plugged in), including about 15 s of model
  loading.
  Each model call reprocesses the whole input, so the cost is driven by the prompt
  length (all function descriptions) and the number of generated tokens. Forcing
  the known parts of the JSON keeps the number of model calls low. Running on
  battery in power-saving mode was 2–3 times slower.

## Challenges faced

- **Understanding the vocabulary file.** Tokens like `Ġthe` looked like noise until
  the GPT-2 byte table explained them. Solution: decode every token once into its
  real text.
- **Numbers are split into digits.** Qwen has no `42` token, only `4` and `2`, so
  numbers are generated one digit at a time and the number rule must accept partial
  numbers like `-` or `4.`.
- **Negative numbers came out positive.** Forcing the space before a value turned
  `-4.5` into `4.5`, because the model writes a minus after a colon as the single
  token `" -"`. Solution: let the value start with an optional space.
- **Strings that never closed.** On the regex prompts the model repeated
  `([0-9]+) | ([0-9]+) | ...` forever. Inspecting the logits showed it actually
  wanted to close with the token `' ",'` (space, quote, comma), which the first
  version did not allow. Solution: accept any closing token whose quote is followed
  by nothing or by the end character, and strip surrounding spaces from the value.
- **Speed.** Each logits call takes from a fraction of a second to a few seconds
  depending on input length and power mode. Forcing known text and pre-filtering
  candidates kept the full run within the 5-minute limit.

## Testing strategy

- **Exploration scripts** (not submitted) to understand the SDK: tokenization,
  the vocabulary, logits, plain greedy generation, and a first constraint
  (digits only).
- **Model validation tests** with pydantic: valid input files load, and broken
  items (wrong type, missing key, wrong key) are rejected.
- **Error handling tests:** missing file, invalid JSON, a JSON object instead of a
  list, and items with the wrong shape, each giving a clear message.
- **Prototype tests** for every value type before integrating it: numbers
  (negative, decimal, large), strings (single and multiple parameters), and the
  full test set.
- **Full runs** with `make run`, checking every function name and argument by hand,
  and `make lint` after every change.

## Resources

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Hugging Face course: tokenizers and byte-level BPE](https://huggingface.co/learn/nlp-course/chapter6/5)
- [OpenAI GPT-2 `encoder.py`](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
  (origin of `bytes_to_unicode`)
- [Outlines: "Efficient Guided Generation for LLMs"](https://arxiv.org/abs/2307.09702)
  (the idea of constraining tokens with a grammar)
- [pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)
- [Python `argparse`](https://docs.python.org/3/library/argparse.html) and
  [`re`](https://docs.python.org/3/library/re.html) documentation

### How AI was used

An AI assistant (Claude) was used throughout the project as a tutor and pair
programmer:

- **Learning:** explaining tokens, logits, constrained decoding, the GPT-2 byte
  table, pydantic, argparse and regular expressions, and proposing small
  exploration experiments to run against the SDK.
- **Planning:** proposing the project structure and a day-by-day plan.
- **Code:** writing or co-writing parts of the code (the byte table in `vocab.py`,
  the generation functions in `decoder.py`, parts of `__main__.py` and
  `io_utils.py`), and reviewing and fixing code written by hand (`models.py`,
  `io_utils.py`, argument parsing).
- **Debugging:** diagnosing the negative-number and unclosed-string problems by
  inspecting the model's logits, and the mypy configuration for `llm_sdk`.
- **Documentation:** writing docstrings and this README.

Every part was run, tested and reviewed, and can be explained line by line.
