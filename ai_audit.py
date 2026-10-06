"""AI re-labelling: Gemini labels a sample independently so we can compare."""

from gemini_client import AIError, generate_json
from lint import normalize

BATCH_SIZE = 15

SYSTEM_INSTRUCTION = """You are a careful data annotator.

For each item, choose exactly one label from the allowed label list, following
the labelling guidelines if any are given. Judge only the text itself.
Give a confidence from 0 to 100 and a one-sentence rationale.
Reply with JSON only, in the requested structure."""


def build_schema(labels):
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "label": {"type": "string", "enum": labels},
                        "confidence": {"type": "integer"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["id", "label", "confidence", "rationale"],
                },
            }
        },
        "required": ["items"],
    }


def build_prompt(batch, labels, guidelines):
    lines = ["Allowed labels: " + ", ".join(labels)]
    if guidelines and guidelines.strip():
        lines += ["", "Labelling guidelines:", guidelines.strip()]
    lines += ["", "Items:"]
    for item_id, text in batch:
        lines.append(f"[{item_id}] {text}")
    return "\n".join(lines)


def validate_batch(data, batch_ids, labels):
    """Keep only well-formed answers for ids we asked about, with allowed labels."""
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise AIError("The AI's reply didn't have the expected format. Try again.")
    allowed = {normalize(label): label for label in labels}
    results = {}
    for item in data["items"]:
        if not isinstance(item, dict):
            continue
        try:
            item_id = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        label = allowed.get(normalize(item.get("label", "")))
        if item_id not in batch_ids or label is None or item_id in results:
            continue
        try:
            confidence = max(0, min(100, int(item.get("confidence", 0))))
        except (TypeError, ValueError):
            confidence = 0
        results[item_id] = {
            "ai_label": label,
            "confidence": confidence,
            "rationale": str(item.get("rationale", "")).strip(),
        }
    return results


def relabel(items, labels, guidelines="", progress=None):
    """Label (id, text) pairs in batches. Returns {id: result}. Skips unanswered ids."""
    if not items:
        raise AIError("There are no rows to audit.")
    results = {}
    batches = [items[i : i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    schema = build_schema(labels)
    for number, batch in enumerate(batches, start=1):
        data = generate_json(build_prompt(batch, labels, guidelines), SYSTEM_INSTRUCTION, schema)
        results.update(validate_batch(data, {item_id for item_id, _ in batch}, labels))
        if progress:
            progress(number / len(batches))
    if not results:
        raise AIError("The AI didn't return any usable labels. Try again.")
    return results
