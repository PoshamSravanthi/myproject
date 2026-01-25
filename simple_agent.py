import requests
import os
import numexpr

from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables
load_dotenv()

# -------------------- TOOLS --------------------

# Search tool
search_tool = DuckDuckGoSearchRun()

# Weather tool (SAFE)
@tool
def get_place_temperature(city: str) -> dict:
    """Fetch current weather safely"""

    api_key = os.getenv("WEATHERSTACK_API_KEY")

    if not api_key:
        return {"city": city, "error": "Weather API key missing"}

    try:
        response = requests.get(
            "http://api.weatherstack.com/current",
            params={
                "access_key": api_key,
                "query": city
            },
            timeout=10
        )

        if response.status_code != 200:
            return {"city": city, "error": "Weather service unavailable"}

        data = response.json()

        if "current" not in data:
            return {"city": city, "error": "Weather data not available"}

        return {
            "city": city,
            "temperature": data["current"]["temperature"],
            "condition": data["current"]["weather_descriptions"][0]
        }

    except Exception:
        return {"city": city, "error": "Failed to fetch weather"}

# Calculator tool
@tool
def calculator(expression: str) -> float:
    """Evaluate math expression safely"""
    return float(numexpr.evaluate(expression))


# -------------------- LLM --------------------

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# -------------------- AGENT --------------------

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_place_temperature, calculator],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_place_temperature, calculator],
    verbose=True,
    handle_parsing_errors=True
)

# -------------------- TEST --------------------
if __name__ == "__main__":
    # Correct numexpr usage
    print(numexpr.evaluate("10+5"))

    result = agent_executor.invoke({
        "input": "find the capital of India, then find its current temperature and subtract 5 from it"
    })

    print("\nFINAL RESULT:\n", result)
