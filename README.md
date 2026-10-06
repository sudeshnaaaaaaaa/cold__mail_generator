# Cold Mail Generator (LLM + RAG)

Scrapes a job posting, extracts structured role details with an LLM, retrieves the most relevant portfolio links from a ChromaDB vector store, and generates a personalized cold outreach email.

## How it works (RAG pipeline)

1. **Scrape** — `WebBaseLoader` pulls raw text from a job posting URL; `utils.clean_text()` strips HTML/links/noise.
2. **Extract** — an LLM (Llama 3.1 via Groq's free API) parses the cleaned text into structured JSON: role, experience, skills, description.
3. **Retrieve** — the extracted skills are embedded and matched against a ChromaDB collection built from `resource/my_portfolio.csv` (tech-stack → project link), returning the most relevant portfolio links. This is the "R" in RAG.
4. **Generate** — the LLM writes a personalized email using the job details and retrieved links, following a prompt template designed to avoid generic, templated output.

## What's actually been tested — read this before you claim it in an interview

Being upfront, exactly like the churn project:

- ✅ **Tested and confirmed working:** the CSV portfolio data loads correctly with pandas; all four Python files (`portfolio.py`, `chains.py`, `utils.py`, `app.py`) are syntactically valid; `utils.clean_text()` was run on sample messy HTML and correctly stripped tags, links, and noise.
- ⚠️ **Not tested end-to-end in this environment:** the ChromaDB embedding step and the actual LLM calls (job extraction + email generation). ChromaDB needs to download an embedding model from Hugging Face, and the Groq API needs a live key — neither is reachable from the sandbox this was built in. This is not a flaw in the code; it's a network restriction of the build environment.

**Before you say "I built and ran this" in an interview, you need to actually run it yourself** — see setup below. It should take about 10 minutes and a free Groq API key.

## Setup — do this yourself to make it real

```bash
pip install langchain langchain-groq langchain-community chromadb pandas streamlit

# Get a free Groq API key (no credit card needed): https://console.groq.com/keys
export GROQ_API_KEY=your_key_here      # on Windows PowerShell: $env:GROQ_API_KEY="your_key_here"

streamlit run app.py
```

Paste any public job-posting URL into the app and click Generate Email. The first run will download ChromaDB's embedding model (~90MB) automatically.

## Customize before pushing to GitHub

- Edit `resource/my_portfolio.csv` — this currently points to placeholder GitHub links for your other projects; swap in your real repo URLs once they're live.
- Adjust `sender_name` / `sender_role` in `chains.py` if needed.

## Interview talking points

- **Why Groq instead of OpenAI?** Groq offers free, fast inference for Llama 3.1 — no billing required, which matters for a student project you're paying for out of pocket.
- **What makes this RAG, not just "call an LLM"?** The portfolio-link retrieval step (ChromaDB similarity search over your skills/projects) happens *before* the email is generated, and its output is injected into the generation prompt — the LLM doesn't just improvise, it's grounded in retrieved, relevant context.
- **How would you evaluate email quality?** Manually score a batch of generated emails on relevance, specificity, and whether the referenced portfolio link genuinely matches the job — this is a good "what would you improve" answer.
- **What breaks this in production?** Very long job postings can exceed context/token limits during extraction; the prompt currently doesn't chunk long pages, which would be a real next step.
