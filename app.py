import json
import os
from typing import Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# -------------------- LLM (GROQ) --------------------

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# -------------------- TOOLS --------------------

# ✅ REAL WEB SEARCH TOOL (Tavily)
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_community.tools import (
    WikipediaQueryRun,
    ArxivQueryRun
)
from langchain_community.utilities import (
    WikipediaAPIWrapper,
    ArxivAPIWrapper
)

# -------------------- INPUT MODEL --------------------

class WorkflowInput(BaseModel):
    query: str

# -------------------- TOOLS SETUP --------------------

# ✅ Web Search Tool (Tavily)
tavily_tool = TavilySearchResults(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=3
)

# Wikipedia Tool
wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=1500
)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# arXiv Tool
arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=2000
)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# -------------------- MAIN WORKFLOW --------------------

def run_multi_agent_workflow(workflow_input: WorkflowInput) -> Dict[str, Any]:
    query = workflow_input.query

    # ===== Research Agent =====
    research_data = []

    # ✅ Web search (Tavily)
    try:
        research_data.append(str(tavily_tool.run(query)))
    except Exception:
        research_data.append("Web search unavailable.")

    # Wikipedia
    try:
        research_data.append(wiki_tool.run(query))
    except Exception:
        research_data.append("Wikipedia data unavailable.")

    # arXiv
    try:
        research_data.append(arxiv_tool.run(query))
    except Exception:
        research_data.append("arXiv data unavailable.")

    research_output = "\n\n".join(research_data)

    # ===== Summarizer Agent =====
    summary_prompt = f"""
Convert the following research notes into structured JSON.

Research:
{research_output}

Your output MUST be a valid JSON object with these keys:
- "executive_summary": A single paragraph of max 150–250 words.

Rules:
- Write the executive summary as ONE proper paragraph.
- Do NOT use bullet points.
- Do NOT add line breaks.
- Do NOT include any other keys.
- Return ONLY the JSON object.
"""

    summary_response = llm.invoke(summary_prompt)
    raw_summary = summary_response.content.strip()

    try:
        summary_json = json.loads(raw_summary)
    except Exception:
        summary_json = {
            "executive_summary": raw_summary
        }

    # ===== Email Agent =====
    email_prompt = f"""
Based on the research provided above, draft a professional business email.

- Include a clear subject line.
- Summarize the key trends from the research.
- Suggest appropriate next steps.
- Do not include extra sections.

Research:
{research_output}

Write the email in this format:

Subject: [Your Subject Line]

Dear [Recipient Name],

[Your Email Content Here]

Best regards,
AI Research Agent
"""

    email_response = llm.invoke(email_prompt)

    return {
        "raw_research": research_output,
        "summary": summary_json,
        "final_email": email_response.content.strip()
    }

# -------------------- TEST RUN --------------------

if __name__ == "__main__":
    result = run_multi_agent_workflow(
        WorkflowInput(query="Latest AI trends in India")
    )
    print(json.dumps(result, indent=2))
