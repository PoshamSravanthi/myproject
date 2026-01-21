import json
from typing import Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from ddgs import DDGS
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class WorkflowInput(BaseModel):
    query: str


def web_search(query: str) -> str:
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No research data found."
        bodies = [res.get("body", "") for res in results if "body" in res]
        return "\n\n".join(bodies) if bodies else "No readable body found."
    except Exception as e:
        return f"Search failed due to error: {str(e)}"


def run_multi_agent_workflow(workflow_input: WorkflowInput) -> Dict[str, Any]:
    query = workflow_input.query

    # Step 1: Web Research
    research = web_search(query)

    # Step 2: Summarization Prompt
    summary_prompt = f"""
Convert the following research notes into structured JSON:

Research:
{research}

Return JSON EXACTLY like:
{{
   "executive_summary": "...",
   "action_items": ["...", "..."]
}}
"""

    summary_result = llm.invoke(summary_prompt)
    raw_summary = summary_result.content.strip()

    # Safe JSON parsing
    try:
        summary_json = json.loads(raw_summary)
    except:
        summary_json = {
            "executive_summary": raw_summary,
            "action_items": []
        }

    # Step 3: Email Generation
    email_prompt = f"""
Write a short professional business email based on this executive summary:

{summary_json['executive_summary']}

Return only the email body content without greetings like "Dear" or signatures.
"""

    email_result = llm.invoke(email_prompt)
    email_body = email_result.content.strip()

    return {
        "raw_research": research,
        "raw_summary": summary_json,
        "final_email": email_body
    }


if __name__ == "__main__":
    test = run_multi_agent_workflow(WorkflowInput(query="Latest AI trends in India"))
    print(json.dumps(test, indent=2))
