import streamlit as st
import pandas as pd

st.set_page_config(page_title="Financial Document Q&A Assistant")

st.title("Financial Document Q&A Assistant")

uploaded = st.file_uploader("Upload a financial PDF/Excel", type=["pdf", "xls", "xlsx"])

if uploaded:
    st.success(f"File {uploaded.name} uploaded successfully!")

    # Placeholder for now
    st.subheader("Demo Output")
    st.write("Document processed successfully (dummy pipeline).")
    st.write("Next: Connect extraction + RAG pipeline.")
