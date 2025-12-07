# eval-streamlit

Prototype Streamlit app for running structured evaluations over conversation logs using Anthropic.

## Local Setup

1. **Create and activate a virtualenv**:

   ```bash
   python -m venv evalenv
   source evalenv/bin/activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set required environment variables** (e.g. in a `.env` file or your shell):

   - `ANTHROPIC_API_KEY` – API key used by the evaluator.
   - `EVAL_APP_ACCESS_CODE` – simple access code required to use the Streamlit UI.

## Running the Streamlit app

From the `eval-streamlit` directory:

```bash
streamlit run app.py
```

Then open the URL printed in the terminal (typically `http://localhost:8501`) and enter the access code when prompted.

## Running the CLI evaluator

You can also run evaluations from the command line using the existing script:

```bash
python scripts/run_evaluations.py configs/config1.json data/your_conversations.json -o results.csv
```

- **Config JSON** (`configs/config1.json`) defines the evaluator system prompt, model, and categories.
- **Data path** can be a single JSON file or a directory containing multiple JSON files.
- Results are written to the specified CSV file.
