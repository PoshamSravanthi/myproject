import json
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

from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
    ArxivQueryRun
)
from langchain_community.utilities import (
    WikipediaAPIWrapper,
    ArxivAPIWrapper
)
from langchain_core.tools import Tool

# -------------------- INPUT MODEL --------------------

class WorkflowInput(BaseModel):
    query: str

# -------------------- TOOLS SETUP --------------------

search_tool = DuckDuckGoSearchRun()

wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=1500
)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=2000
)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

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

    try:
        research_data.append(search_tool.run(query))
    except:
        research_data.append("DuckDuckGo search failed.")

    try:
        research_data.append(wiki_tool.run(query))
    except:
        research_data.append("Wikipedia data unavailable.")

    try:
        research_data.append(arxiv_tool.run(query))
    except:
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
- Do NOT write it as a single long line without structure.
- Do NOT use bullet points.
- Do NOT add line breaks.
- Do NOT include any other keys.
- Return ONLY the JSON object.
"""

    summary_response = llm.invoke(summary_prompt)
    raw_summary = summary_response.content.strip()

    try:
        summary_json = json.loads(raw_summary)
    except:
        summary_json = {
            "executive_summary": raw_summary
        }

    # ===== Email Agent =====
    email_prompt = f"""
Based on the research provided above, draft a professional business email.

- Include a clear subject line.
- Summarize the key trends from the research.
- Suggest appropriate next steps.
- Do not include 'Tone Recommendation' or 'Action Items' sections.

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
