__import__("pysqlite3")
import sys
sys.modules["sqlite3"]=sys.modules.pop("pysqlite3")

import os
from huggingface_hub import login

login(os.environ["HUGGINGFACEHUB_API_TOKEN"])

import streamlit as st
import torch
torch.classes.__path__ = []

from summarization import summarize
from assessement_retriever import find_matches
from most_accurate import find_most_accurate
# Streamlit UI
st.set_page_config(page_title="Assessment Finder", layout="wide")

st.title("SHL Assessment Recommendation Engine")
st.markdown("Enter a job title or description and find relevant SHL assessments.")

job_query = st.text_area("🔍 Job Title / Description", height=150, placeholder="e.g., I’m hiring a backend developer who knows Python and system design.")
search = st.button("➡️ Search")

if search:
    if job_query.strip():
        with st.spinner("Summarizing and finding matches..."):
            summary = summarize(job_query)
            matches = find_matches(summary)

            matches=find_most_accurate(matches,job_query)

        if matches:
            st.success(f"Found {len(matches)} related assessment(s) for your input.")

            for match in matches:
                st.markdown(f"""
                    <div style="margin-bottom: 1rem; padding: 1rem; border: 1px solid #ccc; border-radius: 10px;">
                        <h4><a href="{match['Link']}" target="_blank">{match['Assessment Name']}</a></h4>
                        <p><b>Duration:</b> {match['Duration']}<br>
                        <b>Remote Testing:</b> {match['Remote Testing']}<br>
                        <b>Adaptive Testing:</b> {match['Adaptive Testing']}<br>
                        <b>Language:</b> {match['Language']}</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No close matches found. Try using a more general or different job description.")
    else:
        st.warning("Please enter a job description to search.")
