"""
app.py
Streamlit UI: paste a careers-page/job-posting URL, scrape it, extract
structured job postings with an LLM, retrieve matching portfolio links
from ChromaDB, and generate a personalized cold email for each posting.
"""

import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import EmailChain
from portfolio import Portfolio
from utils import clean_text


def run_pipeline(url, chain, portfolio):
    loader = WebBaseLoader([url])
    page_data = clean_text(loader.load().pop().page_content)

    portfolio.load_portfolio()
    jobs = chain.extract_jobs(page_data)

    if not jobs:
        st.warning(
        "No job postings were found on that page. This usually means the page "
        "needs JavaScript to load its content (the scraper only reads static HTML), "
        "or the URL isn't a job posting page. Try a simpler, text-based careers page."
    )

    outputs = []
    for job in jobs:
        links = portfolio.query_links(job.get("skills", []))
        email = chain.write_email(job, links)
        outputs.append((job, email))
    return outputs


st.set_page_config(layout="wide", page_title="Cold Mail Generator")
st.title("📧 Cold Mail Generator")
st.caption(
    "Paste a job posting URL — the app scrapes it, extracts the role details, "
    "finds matching portfolio links via a ChromaDB RAG lookup, and drafts a "
    "personalized cold email."
)

url_input = st.text_input(
    "Job posting URL", value="https://jobs.example.com/careers/software-engineer"
)
submit = st.button("Generate Email")

if submit:
    try:
        chain = EmailChain()
        portfolio = Portfolio()
        results = run_pipeline(url_input, chain, portfolio)
        for job, email in results:
            st.subheader(job.get("role", "Extracted Role"))
            st.json(job)
            st.markdown("**Generated Email:**")
            st.code(email, language="markdown")
    except Exception as e:
        st.error(f"Error: {e}")
