"""Match requested catalogue models and capacities using product data."""

import re
from typing import Any


def model_name(product: dict[str, Any]) -> str:
    return str(product.get("itemName") or product.get("name") or product.get("title") or "").strip()


def model_tokens(text: str) -> tuple[str, ...]:
    # Separate generations and camel-case variants without naming brands or variants.
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    return tuple(part.casefold() for part in re.findall(r"[A-Za-z]+|\d+|[一-鿿]+", text))


def has_model_reference(question: str) -> bool:
    words = model_tokens(question)
    return any(i > 0 and word.isdigit() and words[i - 1].isascii()
               and words[i - 1].isalpha() for i, word in enumerate(words))


def requested_models(question: str, rows: list[dict[str, Any]], *, exact: bool = False) -> list[dict[str, Any]]:
    """Keep the catalogue model mentioned in the question; a base model includes its series."""
    words = model_tokens(question)
    models = [(row, model_tokens(model_name(row["product"]))) for row in rows]
    matches = []
    for row, tokens in models:
        if not tokens:
            continue
        for start in range(len(words)):
            size = 0
            while size < len(tokens) and start + size < len(words) and words[start + size] == tokens[size]:
                size += 1
            # A model must include its generation, when one exists. Otherwise require
            # at least two words to avoid matching a generic brand in ordinary prose.
            core = next((i + 1 for i, token in enumerate(tokens) if token.isdigit()), min(2, len(tokens)))
            if size >= core and (core > 1 or size == len(tokens)):
                matches.append((row, tokens, size, core))
    if not matches:
        # An explicit alphanumeric model that is absent from the catalogue must
        # not silently turn into an unrelated recommendation.
        return [] if has_model_reference(question) else rows
    best = max(size for _, _, size, _ in matches)
    matching = [row for row, tokens, size, core in matches
                if size == best and (size == len(tokens) or (not exact and size == core))]
    return matching


def same_model(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return model_tokens(model_name(left)) == model_tokens(model_name(right))


def memory_gb(value: Any) -> int | None:
    match = re.fullmatch(r"\s*(\d+)\s*(GB?|TB?)\s*", str(value), re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1)) * (1024 if match.group(2).upper().startswith("T") else 1)


def requested_memory(question: str) -> int | None:
    match = re.search(r"(?<![\w])(\d+)\s*(GB?|TB?)(?![A-Za-z])", question, re.IGNORECASE)
    return memory_gb(match.group(0)) if match else None


def product_memories(product: dict[str, Any]) -> list[int]:
    values = product.get("memory") or []
    if isinstance(values, (str, int)):
        values = [values]
    return [size for value in values if (size := memory_gb(value)) is not None]


def rental(product: dict[str, Any]) -> float | None:
    value = product.get("dailyRental")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
