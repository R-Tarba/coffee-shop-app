import glob
import os
import re

import pandas as pd
import streamlit as st

try:
  from google import genai
except ImportError:  # shown as a friendly message in the AI tab
  genai = None

MODEL = "gemini-3.8-flash"

st.set_page_config(page_title="Grind and Crumb Hub", layout="wide")
st.title("Grind and Crumb - AI Data Hub")
st.caption("v4 - dashboard + AI chat")  # remove once you've seen it

# Strict encodings first so text decodes correctly; latin-1 is the last
# resort because it maps every byte and can never raise UnicodeDecodeError.
ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")

SAFE_NAMES = (
    "len", "sum", "min", "max", "round", "sorted", "list", "dict", "set",
    "tuple", "str", "int", "float", "bool", "range", "enumerate", "zip",
    "abs", "any", "all", "isinstance", "reversed", "map", "filter",
)
_b = __builtins__
SAFE_BUILTINS = {
    n: (_b[n] if isinstance(_b, dict) else getattr(_b, n)) for n in SAFE_NAMES
}


@st.cache_data
def load_csv_safely(file_path):
  for enc in ENCODINGS:
    try:
      return pd.read_csv(file_path, encoding=enc, low_memory=False)
    except UnicodeDecodeError:
      continue
  raise ValueError(f"Could not decode {file_path}")


# ---------------------------------------------------------------- charts
def to_chart_frame(result, max_rows=30):
  """Turn an AI result into a frame st.bar_chart/line_chart can draw."""
  if isinstance(result, pd.Series):
    df = result.rename(result.name if result.name is not None else "value")
    df = df.reset_index()
  elif isinstance(result, pd.DataFrame):
    df = result.copy()
    if not isinstance(df.index, pd.RangeIndex):
      df = df.reset_index()
  else:
    return None, False
  if df.empty:
    return None, False
  df.columns = [str(c) for c in df.columns]
  num_cols = df.select_dtypes("number").columns.tolist()
  if not num_cols:
    return None, False
  others = [c for c in df.columns if c not in num_cols]
  is_time = False
  if others:
    x = others[0]
    is_time = pd.api.types.is_datetime64_any_dtype(df[x])
    df = df.set_index(x)
  df = df[num_cols].head(max_rows)
  return df, is_time


def show_result(result, code=None):
  chart_df, is_time = to_chart_frame(result)
  if chart_df is not None:
    if is_time:
      st.line_chart(chart_df)
    else:
      st.bar_chart(chart_df)
  if isinstance(result, (pd.DataFrame, pd.Series)):
    st.dataframe(result)
  else:
    st.write(result)
  if code:
    with st.expander("Show the code Gemini ran"):
      st.code(code, language="python")


# -------------------------------------------------------------------- AI
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
      "or plot). Assign the final answer to a variable named `result`. "
      "Prefer a Series or DataFrame with one label column and numeric "
      "columns (so it can be charted), keep it small (e.g. top 20 rows). "
      "If a column needs cleaning (dates, currency strings), do it in the "
      "code. Return only one ```python code block."
  )
  code = extract_code(generate(client, code_prompt))
  try:
    result = run_code(code, dfs)
  except Exception as first_error:
    retry_prompt = (
        f"{code_prompt}\n\nYour previous code:\n```python\n{code}\n```\n"
        f"failed with: {first_error!r}\nFix it and return only the corrected "
        "```python code block."
    )
    code = extract_code(generate(client, retry_prompt))
    result = run_code(code, dfs)

  result_text = (
      result.to_string() if hasattr(result, "to_string") else str(result)
  )[:4000]
  explain_prompt = (
      f"Question: {question}\n\nThis code was run on the real data:\n"
      f"```python\n{code}\n```\nIt produced:\n{result_text}\n\n"
      "Answer the question clearly for a business owner in a few sentences, "
      "using only these results. Quote the actual numbers. Do not invent "
      "anything that is not in the results."
  )
  return generate(client, explain_prompt), code, result


# ------------------------------------------------------------------- app
csv_files = sorted(glob.glob("*.csv"))
if not csv_files:
  st.error("No CSV files found in the directory!")
  st.stop()

selected = st.sidebar.multiselect(
    "Select datasets:", csv_files, default=csv_files[:5]
)
api_key = st.sidebar.text_input(
    "Gemini API Key:", type="password", value=os.environ.get("GEMINI_API_KEY", "")
)

dataframes = {}
for file in selected:
  try:
    dataframes[file] = load_csv_safely(file)
  except Exception as e:
    st.sidebar.error(f"Error loading {file}: {e}")

tab_dash, tab_chat, tab_data = st.tabs(
    ["Dashboard", "Ask the AI", "Raw Data"]
)

# ---- Dashboard: pick columns, get a chart
with tab_dash:
  if not dataframes:
    st.info("Pick at least one dataset in the sidebar.")
  else:
    name = st.selectbox("Dataset:", list(dataframes), key="dash_file")
    df = dataframes[name]
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")

    numeric_cols = df.select_dtypes("number").columns.tolist()
    st.markdown("#### Build a chart")
    a, b, c, d = st.columns(4)
    x = a.selectbox("Group by", list(df.columns))
    y = b.selectbox("Measure", ["(count rows)"] + numeric_cols)
    agg = c.selectbox("Calculation", ["sum", "mean", "max", "min"])
    kind = d.radio("Chart", ["Bar", "Line"], horizontal=True)
    top = st.slider("Max groups", 5, 100, 15)

    try:
      if y == "(count rows)":
        data = df.groupby(x).size().rename("rows")
      else:
        data = df.groupby(x)[y].agg(agg)
      if kind == "Bar":
        data = data.sort_values(ascending=False).head(top)
        st.bar_chart(data)
      else:
        st.line_chart(data.sort_index().head(top))
      st.dataframe(data)
    except Exception as e:
      st.warning(f"Could not draw that chart: {e}")

    if numeric_cols:
      st.markdown("#### Quick stats")
      st.dataframe(df[numeric_cols].describe().T)

# ---- AI chat
with tab_chat:
  st.subheader("Ask your data a question")
  if genai is None:
    st.error("The Gemini library is missing. Run: pip install google-genai")
  elif not api_key:
    st.info("Paste your Gemini API key in the sidebar to start.")
  elif not dataframes:
    st.info("Pick at least one dataset in the sidebar.")
  else:
    if "chat" not in st.session_state:
      st.session_state.chat = []
    for m in st.session_state.chat:
      with st.chat_message(m["role"]):
        st.write(m["text"])
        if m.get("show"):
          show_result(m["result"], m.get("code"))

    question = st.chat_input("e.g. Which product sold the most?")
    if question:
      st.session_state.chat.append({"role": "user", "text": question})
      with st.chat_message("user"):
        st.write(question)
      with st.chat_message("assistant"):
        try:
          client = genai.Client(api_key=api_key)
          with st.spinner("Analysing your data..."):
            text, code, result = answer_question(client, dataframes, question)
          st.write(text)
          show_result(result, code)
          st.session_state.chat.append({
              "role": "assistant", "text": text, "result": result,
              "code": code, "show": True,
          })
        except Exception as e:
          msg = f"Sorry, that failed: {e}"
          st.error(msg)
          st.session_state.chat.append({"role": "assistant", "text": msg})

# ---- Raw data
with tab_data:
  if dataframes:
    chosen = st.selectbox("Inspect file:", list(dataframes), key="raw_file")
    st.dataframe(dataframes[chosen].head(100))