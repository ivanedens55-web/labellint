"""Streamlit interface for LabelLint."""

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from ai_audit import relabel
from gemini_client import AIError
from lint import cohens_kappa, csv_row, describe_kappa, is_missing, label_set, normalize, run_checks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MAX_FILE_MB = 5
MAX_ROWS = 10_000
MAX_GUIDELINE_CHARS = 2_000
SAMPLE_PATH = Path(__file__).parent / "sample_data.csv"
MAX_LABELS = 25
REPORT_COLUMNS = ["row", "check", "severity", "text", "label", "detail"]


def load_data():
    """Return the uploaded or sample DataFrame, or None."""
    uploaded = st.file_uploader("Upload a CSV with a text column and a label column", type="csv")
    if st.button("Use sample data instead"):
        st.session_state.use_sample = True
    if uploaded is not None:
        st.session_state.use_sample = False
        if uploaded.size > MAX_FILE_MB * 1024 * 1024:
            st.error(f"That file is larger than {MAX_FILE_MB} MB. Upload a smaller CSV or a sample of it.")
            return None
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            logger.exception("Failed to read uploaded CSV")
            st.error("That file couldn't be read as a CSV. Check it opens in a spreadsheet app.")
            return None
        if len(df) > MAX_ROWS:
            st.error(f"That file has {len(df):,} rows; the limit is {MAX_ROWS:,}. Upload a smaller sample.")
            return None
        return df
    if st.session_state.get("use_sample"):
        return pd.read_csv(SAMPLE_PATH)
    return None


def guess_column(columns, candidates):
    for i, column in enumerate(columns):
        if column.strip().lower() in candidates:
            return i
    return 0


def render_rule_checks(issues):
    st.subheader("1. Rule-based checks")
    if not issues:
        st.success("No rule-based issues found.")
        return
    counts = pd.Series([i["severity"] for i in issues]).value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("High", int(counts.get("High", 0)))
    c2.metric("Medium", int(counts.get("Medium", 0)))
    c3.metric("Low", int(counts.get("Low", 0)))
    st.dataframe(pd.DataFrame(issues, columns=REPORT_COLUMNS), hide_index=True, width="stretch")


def run_ai_audit(df, text_col, label_col, labels, sample_size, guidelines):
    valid = df[~df[text_col].apply(is_missing) & ~df[label_col].apply(is_missing)]
    sample = valid.sample(n=min(sample_size, len(valid)), random_state=42)
    items = [(int(i), str(sample.at[i, text_col])) for i in sample.index]
    bar = st.progress(0.0, text="Re-labelling sample...")
    results = relabel(items, labels, guidelines, progress=lambda p: bar.progress(p, text="Re-labelling sample..."))
    bar.empty()

    rows = []
    for index, result in results.items():
        human = str(df.at[index, label_col]).strip()
        rows.append({
            "row": csv_row(index),
            "text": str(df.at[index, text_col])[:150],
            "human_label": human,
            "ai_label": result["ai_label"],
            "agree": normalize(human) == normalize(result["ai_label"]),
            "confidence": result["confidence"],
            "rationale": result["rationale"],
        })
    return pd.DataFrame(rows), len(items)


def render_ai_results(audit, requested):
    st.subheader("2. AI audit results")
    if len(audit) < requested:
        st.caption(f"{requested - len(audit)} item(s) got no usable answer from the AI and were skipped.")
    agreement = audit["agree"].mean()
    kappa = cohens_kappa(list(audit["human_label"]), list(audit["ai_label"]))
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows audited", len(audit))
    c2.metric("Agreement", f"{agreement:.0%}")
    c3.metric("Cohen's kappa", f"{kappa:.2f}", help=describe_kappa(kappa))
    st.caption(f"Kappa interpretation: **{describe_kappa(kappa)}** (Landis & Koch).")

    st.markdown("**Confusion table** (rows = human label, columns = AI label)")
    confusion = pd.crosstab(audit["human_label"].map(normalize), audit["ai_label"].map(normalize))
    st.dataframe(confusion, width="stretch")

    disagreements = audit[~audit["agree"]].sort_values("confidence", ascending=False)
    st.markdown(f"**Disagreements to review ({len(disagreements)})**, most confident first")
    if disagreements.empty:
        st.success("The AI agreed with every audited label.")
    else:
        st.dataframe(disagreements.drop(columns="agree"), hide_index=True, width="stretch")


def build_report(issues, audit):
    rows = list(issues)
    if audit is not None:
        for _, r in audit[~audit["agree"]].iterrows():
            rows.append({
                "row": r["row"], "check": "AI disagreement",
                "severity": "Medium" if r["confidence"] >= 80 else "Low",
                "text": r["text"], "label": r["human_label"],
                "detail": f'AI suggests "{r["ai_label"]}" ({r["confidence"]}%): {r["rationale"]}',
            })
    return pd.DataFrame(rows, columns=REPORT_COLUMNS).to_csv(index=False)


def main():
    st.set_page_config(page_title="LabelLint", page_icon="🏷️", layout="wide")
    st.title("🏷️ LabelLint")
    st.markdown("Audit a labelled dataset: rule-based checks for common annotation errors, plus an AI second opinion on a sample.")

    df = load_data()
    if df is None:
        st.info("Upload a CSV or use the sample data to get started.")
        return
    if df.empty or len(df.columns) < 2:
        st.error("The CSV needs at least two columns and one row.")
        return

    columns = list(df.columns)
    c1, c2 = st.columns(2)
    text_col = c1.selectbox("Text column", columns, index=guess_column(columns, {"text", "content", "sentence", "review"}))
    label_col = c2.selectbox("Label column", columns, index=guess_column(columns, {"label", "labels", "class", "category", "sentiment"}))
    if text_col == label_col:
        st.warning("Pick two different columns for text and label.")
        return

    with st.expander(f"Preview ({len(df)} rows)"):
        st.dataframe(df.head(20), width="stretch")

    issues = run_checks(df, text_col, label_col)
    render_rule_checks(issues)

    st.divider()
    labels = label_set(df, label_col)
    st.markdown(f"**Labels found ({len(labels)}):** " + ", ".join(labels))
    key = (text_col, label_col, len(df))
    if st.session_state.get("audit_key") != key:
        st.session_state.audit, st.session_state.audit_error = None, None

    if len(labels) < 2:
        st.info("The AI audit needs at least two distinct labels.")
    elif len(labels) > MAX_LABELS:
        st.info(f"The AI audit supports up to {MAX_LABELS} labels; this dataset has {len(labels)}.")
    else:
        sample_size = st.slider("Rows to audit with AI", 5, 100, 30, step=5)
        guidelines = st.text_area("Labelling guidelines (optional)", height=80, max_chars=MAX_GUIDELINE_CHARS,
                                  placeholder="e.g. 'neutral' = factual statements with no opinion")
        if st.button("Run AI audit", type="primary"):
            try:
                audit, requested = run_ai_audit(df, text_col, label_col, labels, sample_size, guidelines)
                st.session_state.audit, st.session_state.requested = audit, requested
                st.session_state.audit_key, st.session_state.audit_error = key, None
            except AIError as error:
                st.session_state.audit, st.session_state.audit_error = None, str(error)
            except Exception:
                logger.exception("Unexpected error during AI audit")
                st.session_state.audit = None
                st.session_state.audit_error = "Something unexpected went wrong. Try again."

    if st.session_state.get("audit_error"):
        st.error(st.session_state.audit_error)
    audit = st.session_state.get("audit")
    if audit is not None and not audit.empty:
        render_ai_results(audit, st.session_state.requested)

    st.divider()
    st.download_button("Download issue report (CSV)", build_report(issues, audit),
                       file_name="labellint_report.csv", mime="text/csv")
    st.caption("The AI is a second opinion, not ground truth. Review flagged rows before changing labels.")


if __name__ == "__main__":
    main()
