import glob
import re
from google import genai
import pandas as pd
import streamlit as st

MODEL = "gemini-2.5-flash"

st.set_page_config(page_title="Grind and Crumb Hub", layout="wide")
st.title("Grind and Crumb - Multi-Dataset AI Intelligence Hub")
st.caption("loader v3")  # remove once you've confirmed the new version is deployed

# Strict encodings first so text decodes correctly; latin-1 is the last
# resort because it maps every byte and can never raise UnicodeDecodeError.
ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")

SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "len", "sum", "min", "max", "round", "sorted", "list", "dict", "set",
        "tuple", "str", "int", "float", "bool", "range", "enumerate", "zip",
        "abs", "any", "all", "isinstance", "reversed", "map", "filter",
    )
}


@st.cache_data
def load_csv_safely(file_path):
  for enc in ENCODINGS:
    try:
      return pd.read_csv(file_path, encoding=enc, low_memory=False)
    except UnicodeDecodeError:
      continue
  raise ValueError(f"Could not decode {file_path}")


def describe(dfs):
  parts = []
  for f, df in dfs.items():
    sample = df.head(3).to_string(max_colwidth=40)
    parts.append(
        f'dfs["{f}"]: {df.shape[0]} rows\n'
        f"columns and dtypes: {dict(df.dtypes.astype(str))}\n"
        f"first rows:\n{sample}"
    )
  return "\n\n".join(parts)


def extract_code(text):
  m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
  return (m.group(1) if m else text).strip()


def run_code(code, dfs):
  env = {
      "pd": pd,
      "dfs": {k: v.copy() for k, v in dfs.items()},
      "__builtins__": SAFE_BUILTINS,
  }
  exec(code, env)
  if "result" not in env:
    raise ValueError("The code did not set a variable called `result`.")
  return env["result"]


def generate(client, prompt):
  return client.models.generate_content(model=MODEL, contents=prompt).text


def answer_question(client, dfs, question):
  schema = describe(dfs)
  code_prompt = (
      "You are a data analyst. The following pandas DataFrames are in a dict "
      "called `dfs`, keyed by file name:\n\n"
      f"{schema}\n\n"
      f"Question: {question}\n\n"
      "Write Python (pandas) code that answers the question using only `dfs` "
      "and `pd` (both already available; do not import anything, read files, "
      "or plot). Assign the final answer to a variable named `result` "
      "(a DataFrame, Series or scalar; keep it small, e.g. top 20 rows). "
      "If a column needs cleaning (dates, currency strings), do it in the "
      "code. Return only one ```python code block."
  )
  code = extract_code(generate(client, code_prompt))

  try:
    result = run_code(code, dfs)
  except Exception as first_error:
    # One retry: show the model its own error
    retry_prompt = (
        f"{code_prompt}\n\nYour previous code:\n```python\n{code}\n```\n"
        f"failed with: {first_error!r}\nFix it and return only the corrected "
        "```python code block."
    )
    code = extract_code(generate(client, retry_prompt))
    result = run_code(code, dfs)  # if this fails too, the caller shows it

  result_text = (
      result.to_string() if hasattr(result, "to_string") else str(result)
  )[:4000]
  explain_prompt = (
      f"Question: {question}\n\nThis code was run on the real data:\n"
      f"```python\n{code}\n```\nIt produced:\n{result_text}\n\n"
      "Answer the question clearly for a business owner using only these "
      "results. Quote the actual numbers. Do not invent anything that is "
      "not in the results."
  )
  explanation = generate(client, explain_prompt)
  return explanation, code, result


csv_files = glob.glob("*.csv")
if not csv_files:
  st.error("No CSV files found in the directory!")
else:
  selected = st.sidebar.multiselect(
      "Select datasets:", csv_files, default=csv_files[:5]
  )

  dataframes = {}
  for file in selected:
    try:
      dataframes[file] = load_csv_safely(file)
    except Exception as e:
      st.sidebar.error(f"Error loading {file}: {e}")

  tab1, tab2 = st.tabs(["Data Previews", "Gemini AI"])

  with tab1:
    st.subheader("Data Inspector")
    if selected:
      chosen = st.selectbox("Inspect file:", selected)
      if chosen in dataframes:
        st.dataframe(dataframes[chosen].head(20))

  with tab2:
    st.subheader("Ask Gemini")
    key = st.text_input("Gemini API Key:", type="password")
    query = st.text_input("Ask a business question:")
    if query and key:
      if not dataframes:
        st.warning("Select at least one dataset in the sidebar first.")
      else:
        try:
          client = genai.Client(api_key=key)
          with st.spinner("Analysing your data..."):
            explanation, code, result = answer_question(
                client, dataframes, query
            )
          st.write(explanation)
          with st.expander("Show the data behind this answer"):
            if isinstance(result, (pd.DataFrame, pd.Series)):
              st.dataframe(result)
            else:
              st.write(result)
            st.code(code, language="python")
        except Exception as e:
          st.error(f"Error: {e}")