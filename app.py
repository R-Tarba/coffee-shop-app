import streamlit as st
import pandas as pd
import glob
import os
from google import genai

st.set_page_config(page_title="Grind & Crumb - Executive Command Center", layout="wide")

st.title("☕ Grind & Crumb - Multi-Dataset AI Intelligence Hub")
st.markdown("### Powered by Streamlit & Gemini 2.5 Flash")

# Automatically discover all local CSV files
csv_files = glob.glob("*.csv")

if not csv_files:
    st.error("⚠️ No CSV files found in the project root directory!")
else:
    # Define your exact core hackathon datasets as defaults
    target_defaults = [
        "starter_store_monthly_pnl.csv",
        "stores.csv",
        "footfall.csv",
        "weather.csv",
        "staff_rota_weekly.csv",
        "waste_log.csv",
        "store_costs_monthly.csv"
    ]
    
    # Filter defaults to only include ones actually present in the folder
    valid_defaults = [f for f in target_defaults if f in csv_files]

    # Sidebar for Dataset Management
    st.sidebar.header("📁 Core Datasets")
    selected_files = st.sidebar.multiselect(
        "Select datasets to include in analysis:",
        options=csv_files,
        default=valid_defaults
    )
    
    # Load selected dataframes into memory
    dataframes = {}
    for file in selected_files:
        try:
            df = pd.read_csv(file, low_memory=False)
            dataframes[file] = df
        except Exception as e:
            st.sidebar.error(f"Error loading {file}: {e}")

    # Main Layout Tabs
    tab1, tab2 = st.tabs(["📊 Dataset Previews", "🤖 Gemini AI Multi-Data Analyst"])
    
    with tab1:
        st.subheader("Data Inspector")
        if selected_files:
            chosen_file = st.selectbox("Choose a dataset to inspect:", selected_files)
            if chosen_file in dataframes:
                df_to_show = dataframes[chosen_file]
                st.write(f"**Shape:** {df_to_show.shape[0]:,} rows × {df_to_show.shape[1]} columns")
                st.dataframe(df_to_show.head(50), use_container_width=True)
        else:
            st.info("Select datasets from the sidebar to preview them.")

    with tab2:
        st.subheader("Ask Gemini Across Your Selected Datasets")
        st.markdown("Query P&L performance, footfall trends, weather impacts, labor costs, and waste leaks simultaneously.")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            gemini_api_key = st.text_input("Gemini API Key:", type="password", help="Enter your Google AI Studio API key")
        with col2:
            user_query = st.text_input("Ask a business question:", placeholder="e.g., How does bad weather impact daily footfall and store revenue?")

        if user_query and gemini_api_key:
            with st.spinner("Gemini is cross-analyzing your selected datasets..."):
                try:
                    client = genai.Client(api_key=gemini_api_key)
                    
                    # Build structured context summarizing the selected core files
                    context_payload = "Here is summary data from the selected Grind & Crumb datasets:\n\n"
                    for fname, df in dataframes.items():
                        context_payload += f"=== FILE: {fname} (Columns: {list(df.columns)}) ===\n"
                        context_payload += df.head(15).to_string() + "\n\n"
                    
                    system_prompt = f"""
                    You are the lead executive data consultant for Grind & Crumb (£8.66m turnover across UK coffee shops). 
                    You have access to the core operational and financial datasets selected by the user. Use them to provide precise, rigorous, and financially sound answers.
                    
                    {context_payload}
                    """
                    
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=f"{system_prompt}\n\nQuestion: {user_query}"
                    )
                    
                    st.markdown("### 💡 Gemini Strategic Insight")
                    st.write(response.text)
                    
                except Exception as e:
                    st.error(f"API Error: {e}")
                    
        elif user_query and not gemini_api_key:
            st.warning("⚠️ Please enter your Gemini API key above to execute queries.")