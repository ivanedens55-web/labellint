"""Rule-based annotation checks and agreement statistics (no AI needed)."""

from collections import Counter

import pandas as pd

SEVERITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def normalize(value):
    """Lowercase and collapse whitespace so 'Positive ' and 'positive' compare equal."""
    return " ".join(str(value).split()).casefold()


def is_missing(value):
    return pd.isna(value) or not str(value).strip()


def csv_row(index):
    """Row number as it appears in a spreadsheet (header is row 1)."""
    return int(index) + 2


def _issue(df, index, text_col, label_col, check, severity, detail):
    text = "" if is_missing(df.at[index, text_col]) else str(df.at[index, text_col])
    label = "" if is_missing(df.at[index, label_col]) else str(df.at[index, label_col])
    return {
        "row": csv_row(index),
        "check": check,
        "severity": severity,
        "text": text[:150],
        "label": label,
        "detail": detail,
    }


def run_checks(df, text_col, label_col):
    """Run every rule-based check. Returns a list of issue dicts."""
    issues = []

    # 1. Missing text or label
    for index in df.index:
        if is_missing(df.at[index, text_col]):
            issues.append(_issue(df, index, text_col, label_col, "Missing text", "High", "Row has no text."))
        if is_missing(df.at[index, label_col]):
            issues.append(_issue(df, index, text_col, label_col, "Missing label", "High", "Row has no label."))

    valid = df[~df[text_col].apply(is_missing) & ~df[label_col].apply(is_missing)]

    # 2. Label variants: same label written differently ("Positive " vs "positive")
    raw_by_norm = {}
    for raw in valid[label_col].astype(str).unique():
        raw_by_norm.setdefault(normalize(raw), set()).add(raw)
    canonical = {}
    for norm, raws in raw_by_norm.items():
        if len(raws) > 1:
            counts = Counter(valid[label_col].astype(str))
            canonical[norm] = max(raws, key=lambda r: counts[r])
    for index in valid.index:
        raw = str(valid.at[index, label_col])
        norm = normalize(raw)
        if norm in canonical and raw != canonical[norm]:
            issues.append(_issue(
                df, index, text_col, label_col, "Label variant", "Medium",
                f'"{raw}" looks like a variant of "{canonical[norm]}".',
            ))

    # 3. Duplicate texts: conflicting labels (High) or repeated exactly (Low)
    norm_text = valid[text_col].apply(normalize)
    norm_label = valid[label_col].apply(normalize)
    for _, group_index in norm_text.groupby(norm_text).groups.items():
        if len(group_index) < 2:
            continue
        labels = set(norm_label[group_index])
        rows = ", ".join(str(csv_row(i)) for i in group_index)
        for index in group_index:
            if len(labels) > 1:
                issues.append(_issue(
                    df, index, text_col, label_col, "Conflicting duplicate", "High",
                    f"Same text labelled differently in rows {rows}.",
                ))
            else:
                issues.append(_issue(
                    df, index, text_col, label_col, "Exact duplicate", "Low",
                    f"Same text and label repeated in rows {rows}.",
                ))

    # 4. Rare labels: used so rarely they may be typos or taxonomy gaps
    label_counts = norm_label.value_counts()
    rare_cutoff = max(2, int(0.01 * len(valid)))
    for index in valid.index:
        count = label_counts[norm_label[index]]
        if count <= rare_cutoff and len(label_counts) > 1:
            issues.append(_issue(
                df, index, text_col, label_col, "Rare label", "Low",
                f"Label used only {count} time(s) in the dataset.",
            ))

    issues.sort(key=lambda i: (SEVERITY_ORDER[i["severity"]], i["row"]))
    return issues


def label_set(df, label_col):
    """Distinct labels after normalization, keeping the most common spelling of each."""
    values = df[label_col][~df[label_col].apply(is_missing)].astype(str)
    counts = Counter(values)
    best = {}
    for raw, count in counts.items():
        norm = normalize(raw)
        if norm not in best or count > counts[best[norm]]:
            best[norm] = raw.strip()
    return sorted(best.values())


def cohens_kappa(labels_a, labels_b):
    """Cohen's kappa between two equal-length label lists (normalized)."""
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("Need two non-empty label lists of the same length.")
    a = [normalize(x) for x in labels_a]
    b = [normalize(x) for x in labels_b]
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum(count_a[k] * count_b[k] for k in set(a) | set(b)) / (n * n)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def describe_kappa(kappa):
    """Landis & Koch (1977) interpretation bands."""
    if kappa < 0:
        return "Poor (worse than chance)"
    for cutoff, label in [(0.20, "Slight"), (0.40, "Fair"), (0.60, "Moderate"), (0.80, "Substantial")]:
        if kappa <= cutoff:
            return label
    return "Almost perfect"
