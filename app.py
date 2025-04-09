import streamlit as st
from summarization import summarizer
from most_accurate import find_most_accurate
from assessment_api import find_matches
# from huggingface_hub import login
import torch

st.set_page_config(page_title="Assessment Finder", layout="wide")
# import os
# login(os.environ["HUGGINGFACEHUB_API_TOKEN"])

torch.classes.__path__ = [] 


st.title("SHL Assessment Recommendation Engine")
st.markdown("Enter a job title or description and find relevant SHL assessments.")

job_query = st.text_area("🔍 Job Title / Description", height=150, placeholder="e.g., I’m hiring a backend developer who knows Python and system design.")
search = st.button("➡️ Search")

if search:
    if job_query.strip():
        with st.spinner("Summarizing and finding matches..."):
            summary = summarizer(job_query)
            matches = find_matches(summary)

            matches=find_most_accurate(matches,job_query)

        if matches:
            st.success(f"Found {len(matches)} related assessment(s) for your input.")

            import pandas as pd
            df = pd.DataFrame(matches)
            df = df[["Assessment_Name", "Link","Duration", "Remote_Testing", "Adaptive_Testing", "Language"]]
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No close matches found. Try using a more general or different job description.")
    else:
        st.warning("Please enter a job description to search.")
