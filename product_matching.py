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
    matches = []
    for row in rows:
        tokens = model_tokens(model_name(row["product"]))
        if not tokens:
            continue
        model = "".join(tokens)
        # The first model number defines the family boundary. A bare family may
        # include variants, while an explicit variant must match the full model.
        core_size = next((i + 1 for i, token in enumerate(tokens) if token.isdigit()), min(2, len(tokens)))
        core = "".join(tokens[:core_size])
        for start in range(len(words)):
            prefix = ""
            consumed = 0
            for word in words[start:]:
                if not model.startswith(prefix + word):
                    break
                prefix += word
                consumed += 1
            next_word = words[start + consumed] if start + consumed < len(words) else ""
            # "iPhone18 Pro" is not the base model; "Pro Max" is not Pro.
            explicit_suffix = next_word.isascii() and next_word.isalpha()
            if len(prefix) >= len(core) and not (explicit_suffix and prefix in (core, model)):
                matches.append((row, len(prefix), prefix == model, prefix == core))
    if not matches:
        # An explicit alphanumeric model that is absent from the catalogue must
        # not silently turn into an unrelated recommendation.
        return [] if has_model_reference(question) else rows
    best = max(size for _, size, _, _ in matches)
    return [row for row, size, full, family in matches
            if size == best and (full or (not exact and family))]


def same_model(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return "".join(model_tokens(model_name(left))) == "".join(model_tokens(model_name(right)))


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
    # Listings such as "12GB+512GB" encode RAM + storage. Match the storage
    # component, not the RAM size, when the user asks for a capacity.
    return [size for value in values if (size := memory_gb(str(value).split("+")[-1])) is not None]


def rental(product: dict[str, Any]) -> float | None:
    value = product.get("dailyRental")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
