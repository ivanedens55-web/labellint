# 🏷️ LabelLint

![Tests](https://github.com/ivanedens55-web/labellint/actions/workflows/tests.yml/badge.svg)

A Streamlit app that audits labelled datasets for annotation quality. It runs rule-based checks for common labelling errors, then uses Google Gemini to independently re-label a sample and measure how well the existing labels hold up, using percent agreement and Cohen's kappa.

Built for anyone who works with labelled data for machine learning: catch mislabels, inconsistent spellings and conflicting duplicates before they reach a model.

## Features

**Rule-based checks (no API key needed)**
- Missing text or missing labels
- Label variants: the same label written differently (`positive` vs `Positive `)
- Conflicting duplicates: the same text labelled differently
- Exact duplicates: the same text and label repeated
- Rare labels: labels used so rarely they may be typos or taxonomy gaps
- Issues sorted by severity (High, Medium, Low), with spreadsheet row numbers

**AI audit**
- Gemini re-labels a random sample (5–100 rows) using only your existing label set, with optional labelling guidelines
- Percent agreement and Cohen's kappa, with the standard Landis & Koch interpretation
- Confusion table showing which labels get mixed up
- Disagreements listed with the AI's suggested label, confidence and rationale, most confident first

**Report**
- Download a single CSV combining rule-based issues and AI disagreements
- Includes a sample dataset with deliberate errors for a quick demo

## Screenshots

<img width="1600" height="452" alt="WhatsApp Image 2026-10-06 at 17 51 20" src="https://github.com/user-attachments/assets/4bc3ea9c-6c93-4636-9a7f-c3ed90d003e5" />
<img width="1600" height="791" alt="WhatsApp Image 2026-10-06 at 17 52 09" src="https://github.com/user-attachments/assets/afaa3b11-1a8f-42a6-8ef4-0561dbea56e0" />
<img width="1600" height="796" alt="WhatsApp Image 2026-10-06 at 17 52 33" src="https://github.com/user-attachments/assets/7136918b-42e3-4e54-b540-c4323d56d761" />

## Technologies used

- Python 3.10+
- [Streamlit](https://streamlit.io/) for the interface
- pandas for data handling
- [Google Gemini API](https://ai.google.dev/) via the official `google-genai` SDK
- `python-dotenv` for configuration
- `pytest` for tests (development only)

## Project structure

```
labellint/
├── app.py              # Streamlit UI: upload, column selection, results, report download
├── lint.py             # Rule-based checks, label normalisation, Cohen's kappa
├── ai_audit.py         # Batched AI re-labelling and response validation
├── gemini_client.py    # Gemini API call, JSON parsing, friendly error handling
├── sample_data.csv     # Demo dataset with deliberate labelling errors
├── tests/
│   └── test_lint.py    # Unit tests for checks, kappa and AI response validation
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

## Setup

```bash
git clone https://github.com/ivanedens55-web/labellint.git
cd labellint
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The rule-based checks work without an API key. For the AI audit, create a free Gemini API key at [Google AI Studio](https://aistudio.google.com/apikey), copy `.env.example` to `.env` and set:

```
AI_API_KEY=your_real_key_here
```

The default model is `gemini-3.1-flash-lite`. Set `AI_MODEL` in `.env` to use another one.

## Run

```bash
streamlit run app.py
```

Upload a CSV with a text column and a label column, or click **Use sample data instead**.

## Run the tests

```bash
pip install pytest
python -m pytest
```

## How it works

1. The rule-based checks in `lint.py` normalise text and labels (lowercase, collapsed whitespace) and look for missing values, spelling variants, duplicates and rare labels.
2. For the AI audit, a random sample is sent to Gemini in batches of 15. The response schema restricts answers to your existing labels, and every reply is validated: unknown IDs, labels outside the set and malformed items are dropped.
3. The app compares the AI labels with the human labels to calculate agreement and Cohen's kappa, which corrects for agreement that would happen by chance.

## Limitations

The AI's label is a second opinion, not ground truth. A disagreement means a row is worth a human look, not that the original label is wrong. Kappa from a small sample is noisy, so audit more rows for a more reliable estimate. The AI audit supports up to 25 distinct labels. Free-tier usage is rate-limited, and on the free tier Google may use requests to improve its models, so don't upload sensitive or personal data.

## Future improvements

- Multi-annotator datasets: agreement between human annotators (Fleiss' kappa)
- Excel and JSONL input
- Re-check only the flagged rows with a second model

## License

MIT — see [LICENSE](LICENSE).
