import requests
import os

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

@tool
def get_place_temperature(city: str) -> dict:
    """Fetch current weather for a city"""

    data = requests.get(
        "http://api.weatherstack.com/current",
        params={
            "access_key": os.environ["WEATHERSTACK_API_KEY"],
            "query": city
        },
        timeout=10
    ).json()

    return {
        "city": city,
        "temp_c": data["current"]["temperature"],
        "condition": data["current"]["weather_descriptions"][0]
    }

# -------------------- LLM --------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# -------------------- AGENT --------------------

# Pull ReAct prompt
prompt = hub.pull("hwchase17/react")

# Create agent
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_place_temperature],
    prompt=prompt
)

# Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_place_temperature],
    verbose=True,
    handle_parsing_errors=True
)

if __name__ == "__main__":
    result = agent_executor.invoke({
        "input": "find the capital of India, then find its current weather condition"
    })

    print("\nFINAL RESULT:\n", result)
