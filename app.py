import json
from typing import Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# Tools
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_core.tools import Tool


# -------------------- LLM --------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# -------------------- INPUT MODEL --------------------

class WorkflowInput(BaseModel):
    query: str

# -------------------- TOOLS SETUP --------------------

# DuckDuckGo
search_tool = DuckDuckGoSearchRun()

# Wikipedia
wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=1500
)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# arXiv
arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=2000
)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# (Tool list kept for clarity / mentor reference)
tools = [
    Tool(
        func=search_tool.run,
        name="web_search",
        description="Search the web for current and real-time information."
    ),
    Tool(
        func=wiki_tool.run,
        name="wikipedia",
        description="Search Wikipedia for general and historical facts."
    ),
    Tool(
        func=arxiv_tool.run,
        name="arxiv_research",
        description="Search academic and scholarly research papers."
    )
]

# -------------------- MAIN WORKFLOW --------------------

def run_multi_agent_workflow(workflow_input: WorkflowInput) -> Dict[str, Any]:
    query = workflow_input.query

    # ===== Research Agent =====
    research_data = []

    # DuckDuckGo (most reliable)
    try:
        research_data.append(search_tool.run(query))
    except Exception as e:
        research_data.append("DuckDuckGo search failed.")

    # Wikipedia (can fail due to API limits)
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

Return JSON EXACTLY in this format:
{{
  "executive_summary": "...",
  "action_items": ["...", "..."]
}}
"""

    summary_response = llm.invoke(summary_prompt)
    raw_summary = summary_response.content.strip()

    try:
        summary_json = json.loads(raw_summary)
    except Exception:
        summary_json = {
            "executive_summary": raw_summary,
            "action_items": []
        }

    # ===== Email Agent =====
    email_prompt = f"""
Write a short professional business email based on the executive summary below.

{summary_json["executive_summary"]}

Return only the email body. No greeting and no signature.
"""

    email_response = llm.invoke(email_prompt)

    return {
        "raw_research": research_output,
        "summary": summary_json,
        "final_email": email_response.content.strip()
    }

# -------------------- TEST --------------------

if __name__ == "__main__":
    result = run_multi_agent_workflow(
        WorkflowInput(query="Latest AI trends in India")
    )
    print(json.dumps(result, indent=2))
