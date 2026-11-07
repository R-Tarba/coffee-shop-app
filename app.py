import streamlit as st
import pandas as pd
import glob
from google import genai

st.set_page_config(page_title="Grind and Crumb Hub", layout="wide")
st.title("Grind and Crumb - Multi-Dataset AI Intelligence Hub")

csv_files = glob.glob("*.csv")
if not csv_files:
    st.error("No CSV files found in the directory!")
else:
    selected = st.sidebar.multiselect("Select datasets:", csv_files, default=csv_files[:5])
    
    dataframes = {}
    for file in selected:
        try:
            dataframes[file] = pd.read_csv(file, low_memory=False)
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
            try:
                client = genai.Client(api_key=key)
                ctx = "Available data summaries:\n"
                for f, df in dataframes.items():
                    ctx += f"File: {f}, Columns: {list(df.columns)}\n"
                
                res = client.models.generate_content(
                    model="gemini-2.5-flash", 
                    contents=f"{ctx}\nQuestion: {query}"
                )
                st.write(res.text)
            except Exception as e:
                st.error(f"Error: {e}")