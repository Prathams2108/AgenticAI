import os
import ast
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

from tools import image_search_tool

load_dotenv()

# -------------------- LLM --------------------
llm = ChatOpenAI(
    model="nvidia/nemotron-nano-12b-v2-vl:free",
    temperature=0.3,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# -------------------- CUSTOM REACT PROMPT --------------------
REACT_PROMPT = PromptTemplate.from_template("""
You are an AI agent that can use tools.

You have access to the following tools:
{tools}

Tool names:
{tool_names}

Use the following format:

Thought: you should think about what to do
Action: the tool name
Action Input: the input to the tool
Observation: the result of the tool
... (repeat Thought/Action/Observation as needed)
Final Answer: the final answer

IMPORTANT:
- If a tool is required, DO NOT provide a Final Answer
- When using DishImageSearch, return ONLY the tool output

Begin!

Question:
{input}

{agent_scratchpad}
""")

# -------------------- CREATE AGENT --------------------
agent = create_react_agent(
    llm=llm,
    tools=[image_search_tool],
    prompt=REACT_PROMPT
)

# -------------------- AGENT EXECUTOR --------------------
agent_executor = AgentExecutor(
    agent=agent,
    tools=[image_search_tool],
    verbose=True,
    handle_parsing_errors=True  # 🔥 REQUIRED
)

# -------------------- PUBLIC FUNCTION --------------------
def fetch_dish_images(dish_name: str):
    """
    Fetch image URLs for a dish.
    Always returns list[str]
    """

    prompt = f"""
You MUST use the DishImageSearch tool.

Dish name:
{dish_name}
"""

    result = agent_executor.invoke({"input": prompt})

    output = result.get("output", [])

    # -------------------- NORMALIZE OUTPUT --------------------
    if isinstance(output, str):
        try:
            output = ast.literal_eval(output)
        except Exception:
            output = []

    if not isinstance(output, list):
        output = []

    return [url for url in output if isinstance(url, str)]
