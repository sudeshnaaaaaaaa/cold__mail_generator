"""
portfolio.py
Loads (Techstack -> Link) rows from a CSV into a persistent ChromaDB
collection, and retrieves the most relevant portfolio links for a
given set of skills extracted from a job posting. This is the
"retrieval" half of the RAG pipeline.
"""

import uuid
import pandas as pd
import chromadb


class Portfolio:
    def __init__(self, csv_path="resource/my_portfolio.csv", persist_dir=".chroma_store"):
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        """Embed and store each (techstack -> link) row, if not already stored."""
        if not self.collection.count():
            for _, row in self.data.iterrows():
                self.collection.add(
                    documents=[row["Techstack"]],
                    metadatas=[{"links": row["Links"]}],
                    ids=[str(uuid.uuid4())],
                )

    def query_links(self, skills, n_results=2):
        """
        Given a list of skills (strings) extracted from a job posting,
        return the most relevant portfolio links.
        """
        if isinstance(skills, str):
            skills = [skills]
        skills = [s for s in (skills or []) if s and str(s).strip()]
        if not skills:
            return []
        results = self.collection.query(query_texts=skills, n_results=n_results)
        return results.get("metadatas", [])