"""
chains.py
The LLM half of the pipeline: (1) extracts structured job postings from
raw scraped page text, and (2) generates a personalized cold email using
the extracted job details plus retrieved portfolio links.

Uses Groq's free-tier Llama 3.1 API via langchain-groq. Requires a
GROQ_API_KEY environment variable (free to obtain at console.groq.com).
"""

import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException


class EmailChain:
    def __init__(self, model="openai/gpt-oss-120b", temperature=0):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys "
                "and set it with: export GROQ_API_KEY=your_key_here"
            )
        self.llm = ChatGroq(model=model, temperature=temperature, groq_api_key=api_key)

    def extract_jobs(self, page_text: str):
        """Parse raw scraped job-posting page text into structured JSON."""
        prompt = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM A CAREERS PAGE:
            {page_data}
            ### INSTRUCTION:
            Extract the job posting(s) from the text above and return them as a JSON
            array. Each object must have the keys: `role`, `experience`,
            `skills` (a list), and `description`.
            Only return valid JSON — no preamble or commentary.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain = prompt | self.llm
        response = chain.invoke({"page_data": page_text})
        try:
            parser = JsonOutputParser()
            parsed = parser.parse(response.content)
        except OutputParserException:
            raise OutputParserException("Job posting text too large or unparsable — try a shorter excerpt.")
        return parsed if isinstance(parsed, list) else [parsed]

    def write_email(self, job: dict, portfolio_links, sender_name="Sudeshna Samale",
                     sender_role="Computer Science student"):
        """Generate a personalized cold email for one extracted job posting."""
        prompt = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### RELEVANT PORTFOLIO LINKS:
            {link_list}

            ### INSTRUCTION:
            You are {sender_name}, a {sender_role} writing a cold outreach email
            to apply for or express interest in the role above. Write a concise,
            genuine, non-generic email (under 150 words) that:
            - Opens with a specific, relevant hook tied to the role
            - Briefly highlights matching skills/experience
            - References 1-2 of the portfolio links naturally where relevant
            - Ends with a clear, low-pressure call to action
            Do not sound like a template. No preamble — output only the email body.
            ### EMAIL (NO PREAMBLE):
            """
        )
        chain = prompt | self.llm
        response = chain.invoke({
            "job_description": str(job),
            "link_list": str(portfolio_links),
            "sender_name": sender_name,
            "sender_role": sender_role,
        })
        return response.content
