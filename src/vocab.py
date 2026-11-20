import json


def bytes_to_unicode() -> dict[int, str]:
    """Build the GPT-2 byte-to-character table used by the vocab file.

    Printable bytes map to themselves; the other bytes are shifted to
    unused characters starting at 256 (so byte 32, a space, becomes 'Ġ').

    Returns:
        A mapping from each byte value (0-255) to its stand-in character.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    table: dict[int, str] = {}
    shift = 0
    for b in range(256):
        if b in printable:
            table[b] = chr(b)
        else:
            table[b] = chr(256 + shift)
            shift += 1
    return table


def load_token_texts(vocab_path: str, size: int) -> list[str]:
    """Load the vocab file and decode every token into its real text.

    Args:
        vocab_path: Path to the model's vocab.json (token string -> id).
        size: Number of logits the model returns. Ids without an entry
            in the vocab file (special tokens, padding) get "".

    Returns:
        A list where index i holds the text of token id i.

    Raises:
        OSError: If the vocab file cannot be read.
        json.JSONDecodeError: If the vocab file is not valid JSON.
    """
    with open(vocab_path, encoding="utf-8") as f:
        vocab: dict[str, int] = json.load(f)
    char_to_byte = {c: b for b, c in bytes_to_unicode().items()}
    texts = [""] * size
    for token, token_id in vocab.items():
        if 0 <= token_id < size:
            raw = bytes(char_to_byte[c] for c in token)
            texts[token_id] = raw.decode("utf-8", errors="replace")
    return texts
