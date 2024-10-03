import os
import pandas as pd
import streamlit as st
from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM

# Streamlit App Layout
st.title("Chat with Dataset 📊")

# Placeholder to store the dataset
if 'df' not in st.session_state:
    st.session_state.df = None

# File Upload Section
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

# Display refresh button to clear everything
if st.button("🔄 Refresh"):
    st.session_state.df = None
    st.rerun()

# Load file and display preview
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df  # Save the dataframe in session state
        st.success("Dataset loaded successfully!")
        
        with st.expander("🔎 Data Preview"):
            st.write(st.session_state.df.head(3))
        
    except Exception as e:
        st.error(f"Error loading file: {e}")

# If a file is loaded, show the query section
if st.session_state.df is not None:
    query = st.text_area("🗣️ Ask a question about your dataset")
    
    # Placeholder for output
    output_container = st.empty()
    
    if query:
        try:
            # Set up LLM and SmartDataframe
            ollama_llm = LocalLLM(api_base="http://localhost:11434/v1", model="llama3")
            df_llm = SmartDataframe(st.session_state.df, config={"llm": ollama_llm})
            
            # Get the response from PandasAI
            answer = df_llm.chat(query)
            
            # Display the response
            output_container.write(f"Response: {answer}")
            
            # New Question button allows users to ask new queries without resetting the dataset
            if st.button("New Question"):
                query = st.text_area("🗣️ Ask a new question")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
