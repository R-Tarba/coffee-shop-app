import datetime
import glob
import os
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

try:
  from google import genai
except ImportError:  # shown as a friendly message in the AI tab
  genai = None

MODEL = "gemini-3.6-flash"

st.set_page_config(page_title="Grind and Crumb Hub", layout="wide")
st.title("Grind and Crumb - AI Data Hub")
st.caption("v6 - TESTING AUTO REFRESH")

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

GREEN = "#2e9e5b"
RED = "#d64545"


@st.cache_data
def load_csv_safely(file_path):
  for enc in ENCODINGS:
    try:
      return pd.read_csv(file_path, encoding=enc, low_memory=False)
    except UnicodeDecodeError:
      continue
  raise ValueError(f"Could not decode {file_path}")


# ------------------------------------------------------ data manipulation
PERIODS = {"Day": "D", "Week": "W", "Month": "M"}


def parse_dates(series):
  return pd.to_datetime(series, errors="coerce", dayfirst=True)


def over_time(df, date_col, measure, agg, period, split=None):
  """Totals per Day/Week/Month, with % change vs the previous period."""
  work = pd.DataFrame({
      "date": parse_dates(df[date_col]),
      "value": pd.to_numeric(df[measure], errors="coerce"),
  })
  if split:
    work["group"] = df[split].astype(str)
  work = work.dropna(subset=["date"])
  if work.empty:
    return None
  work["period"] = work["date"].dt.to_period(PERIODS[period]).dt.start_time
  keys = ["period"] + (["group"] if split else [])
  g = work.groupby(keys)["value"].agg(agg).reset_index().sort_values(keys)
  if split:
    g["change_pct"] = g.groupby("group")["value"].pct_change(fill_method=None)
  else:
    g["change_pct"] = g["value"].pct_change(fill_method=None)
  g["change_pct"] = g["change_pct"] * 100
  g["change_pct"] = g["change_pct"].where(np.isfinite(g["change_pct"]))
  return g


def compare_periods(df, date_col, cat_col, measure, agg, a_range, b_range):
  """Value per category in period A vs period B, plus change and change %."""
  work = pd.DataFrame({
      "date": parse_dates(df[date_col]).dt.date,
      "cat": df[cat_col].astype(str),
      "value": pd.to_numeric(df[measure], errors="coerce"),
  }).dropna(subset=["date"])

  def part(r):
    m = work[(work["date"] >= r[0]) & (work["date"] <= r[1])]
    return m.groupby("cat")["value"].agg(agg)

  comp = pd.concat(
      [part(a_range).rename("Period A"), part(b_range).rename("Period B")],
      axis=1,
  ).fillna(0)
  comp["Change"] = comp["Period B"] - comp["Period A"]
  comp["Change %"] = comp["Change"] / comp["Period A"].replace(0, np.nan) * 100
  comp.index.name = "cat"
  return comp.sort_values("Change", key=abs, ascending=False)


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


def change_color(field):
  return alt.condition(
      f"datum.{field} >= 0", alt.value(GREEN), alt.value(RED)
  )


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
  import time

  for attempt in range(4):
    try:
      return client.models.generate_content(model=MODEL, contents=prompt).text
    except Exception as e:
      if attempt == 3 or not any(c in str(e) for c in ("503", "429")):
        raise
      time.sleep(2 * (attempt + 1))


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

tab_dash, tab_cmp, tab_chat, tab_data = st.tabs(
    ["Dashboard", "Compare", "Ask the AI", "Raw Data"]
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

# ---- Compare: change over time, period vs period, categories
with tab_cmp:
  if not dataframes:
    st.info("Pick at least one dataset in the sidebar.")
  else:
    name = st.selectbox("Dataset:", list(dataframes), key="cmp_file")
    df = dataframes[name]
    num_cols = df.select_dtypes("number").columns.tolist()
    if not num_cols:
      st.warning("This file has no number columns to compare.")
    else:
      mode = st.radio(
          "What do you want to see?",
          ["Change over time", "Period A vs Period B", "Compare categories"],
          horizontal=True,
      )
      m1, m2 = st.columns(2)
      measure = m1.selectbox("Measure", num_cols, key="cmp_measure")
      agg = m2.selectbox(
          "Calculation", ["sum", "mean", "count", "max", "min"], key="cmp_agg"
      )
      all_cols = list(df.columns)

      try:
        if mode == "Change over time":
          t1, t2, t3 = st.columns(3)
          date_col = t1.selectbox("Date column", all_cols, key="ot_date")
          period = t2.selectbox("Per", list(PERIODS), index=2, key="ot_period")
          split = t3.selectbox(
              "Split by (optional)", ["(none)"] + all_cols, key="ot_split"
          )
          split = None if split == "(none)" else split
          if split and df[split].nunique() > 12:
            st.warning(
                f"'{split}' has {df[split].nunique()} different values, which "
                "is too many lines. Pick a column with fewer groups."
            )
            split = None
          g = over_time(df, date_col, measure, agg, period, split)
          if g is None:
            st.warning(
                f"Could not read '{date_col}' as dates. Pick the date column."
            )
          else:
            tip = ["period:T", "value:Q"] + (["group:N"] if split else [])
            line_enc = dict(
                x=alt.X("period:T", title=period),
                y=alt.Y("value:Q", title=f"{agg} of {measure}"),
                tooltip=tip,
            )
            if split:
              line_enc["color"] = alt.Color("group:N", title=split)
            st.altair_chart(
                alt.Chart(g).mark_line(point=True).encode(**line_enc)
                .interactive()
            )
            st.markdown("**% change vs the previous " + period.lower() + "**")
            ch = g.dropna(subset=["change_pct"])
            if ch.empty:
              st.info("Need at least two periods to show a change.")
            else:
              bar_enc = dict(
                  x=alt.X("period:T", title=period),
                  y=alt.Y("change_pct:Q", title="% change"),
                  tooltip=tip + ["change_pct:Q"],
              )
              if split:
                bar_enc["color"] = alt.Color("group:N", title=split)
                bar_enc["xOffset"] = "group:N"
              else:
                bar_enc["color"] = change_color("change_pct")
              st.altair_chart(
                  alt.Chart(ch).mark_bar().encode(**bar_enc).interactive()
              )
              if not split:
                last = g.iloc[-1]
                st.metric(
                    f"Latest {period.lower()} ({last['period']:%d %b %Y})",
                    f"{last['value']:,.2f}",
                    None if pd.isna(last["change_pct"])
                    else f"{last['change_pct']:.1f}%",
                )
            st.dataframe(g)

        elif mode == "Period A vs Period B":
          t1, t2 = st.columns(2)
          date_col = t1.selectbox("Date column", all_cols, key="ab_date")
          cat_col = t2.selectbox("Compare by", all_cols, key="ab_cat")
          dates = parse_dates(df[date_col]).dropna().dt.date
          if dates.empty:
            st.warning(
                f"Could not read '{date_col}' as dates. Pick the date column."
            )
          else:
            dmin, dmax = dates.min(), dates.max()
            mid = dmin + (dmax - dmin) / 2
            d1, d2 = st.columns(2)
            a_rng = d1.date_input(
                "Period A", value=(dmin, mid), min_value=dmin, max_value=dmax,
                key="ab_a",
            )
            b_rng = d2.date_input(
                "Period B",
                value=(mid + datetime.timedelta(days=1), dmax),
                min_value=dmin, max_value=dmax, key="ab_b",
            )
            if len(a_rng) != 2 or len(b_rng) != 2:
              st.info("Pick a start and an end date for both periods.")
            else:
              top = st.slider("Show top N biggest changes", 3, 40, 10)
              comp = compare_periods(
                  df, date_col, cat_col, measure, agg, a_rng, b_rng
              ).head(top)
              long = comp.reset_index().melt(
                  id_vars="cat", value_vars=["Period A", "Period B"],
                  var_name="period", value_name="value",
              )
              st.altair_chart(
                  alt.Chart(long).mark_bar().encode(
                      x=alt.X("cat:N", sort=list(comp.index), title=cat_col),
                      xOffset="period:N",
                      y=alt.Y("value:Q", title=f"{agg} of {measure}"),
                      color=alt.Color("period:N", title=""),
                      tooltip=["cat:N", "period:N", "value:Q"],
                  ).interactive()
              )
              st.markdown("**Change (B minus A)**")
              chg = comp.reset_index().rename(
                  columns={"Change": "change", "Change %": "change_pct"}
              )
              st.altair_chart(
                  alt.Chart(chg).mark_bar().encode(
                      x=alt.X("cat:N", sort=list(comp.index), title=cat_col),
                      y=alt.Y("change:Q", title="Change"),
                      color=change_color("change"),
                      tooltip=["cat:N", "change:Q", "change_pct:Q"],
                  ).interactive()
              )
              st.dataframe(comp.style.format("{:,.2f}"))

        else:  # Compare categories
          cat_col = st.selectbox("Compare by", all_cols, key="cc_cat")
          g = (
              df.assign(_c=df[cat_col].astype(str))
              .groupby("_c")[measure].agg(agg)
              .sort_values(ascending=False)
          )
          picks = st.multiselect(
              "Items to compare", list(g.index), default=list(g.index[:8]),
              key="cc_picks",
          )
          if picks:
            data = g.loc[picks].rename("value").rename_axis("item")
            data = data.reset_index()
            avg = float(g.mean())
            bars = alt.Chart(data).mark_bar().encode(
                x=alt.X("item:N", sort="-y", title=cat_col),
                y=alt.Y("value:Q", title=f"{agg} of {measure}"),
                tooltip=["item:N", "value:Q"],
            )
            rule = alt.Chart(pd.DataFrame({"avg": [avg]})).mark_rule(
                color="orange", strokeDash=[6, 4]
            ).encode(y="avg:Q")
            st.altair_chart((bars + rule).interactive())
            st.caption(f"Dashed line = average across all {len(g)} items.")
            st.dataframe(data)
      except Exception as e:
        st.warning(f"Could not build that comparison: {e}")

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
          client = genai.Client(
              api_key=api_key, http_options=dict(timeout=60000)
          )
          with st.spinner("Analysing your data..."):
            text, code, result = answer_question(client, dataframes, question)
          st.write(text)
          show_result(result, code)
          st.session_state.chat.append({
              "role": "assistant", "text": text, "result": result,
              "code": code, "show": True,
          })
        except Exception as e:
          st.error(f"Sorry, that failed: {e}")

# ---- Raw data
with tab_data:
  if dataframes:
    chosen = st.selectbox("Inspect file:", list(dataframes), key="raw_file")
    st.dataframe(dataframes[chosen].head(100))