import os
import markdown
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap


load_dotenv()

app = Flask(__name__)

# -------------------- LLM --------------------
llm = ChatOpenAI(
    model="google/gemma-3-27b-it:free",
    temperature=0.4,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

parser = StrOutputParser()

# ==========================================================
# CHAIN 1: ROUTE + TIME + WAYPOINTS
# ==========================================================
route_examples = [
    {
        "input": "Bangalore to Mysore",
        "output": (
            "Estimated Travel Time:\n"
            "3 to 3.5 hours\n\n"
            "Planned Route:\n"
            "Bangalore → NICE Road → NH275 → Mysore\n\n"
            "Waypoints:\n"
            "- Bidadi\n"
            "- Ramanagara\n"
            "- Mandya"
        )
    }
]

route_example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Trip: {input}\nResponse:\n{output}"
)

route_prompt = FewShotPromptTemplate(
    examples=route_examples,
    example_prompt=route_example_prompt,
    prefix=(
        "You are a motorcycle route planner.\n"
        "Estimate time, route, and waypoints.\n\n"
    ),
    suffix="Trip: {start} to {destination}\nResponse:",
    input_variables=["start", "destination"]
)

route_chain = route_prompt | llm | parser

# ==========================================================
# CHAIN 2: RIDING TIPS (USES ROUTE OUTPUT)
# ==========================================================
tips_prompt = PromptTemplate(
    input_variables=["route_plan"],
    template="""
You are a motorcycle safety expert.

Based on the route plan below, provide 4–5 riding tips.

Route Plan:
{route_plan}

Riding Tips:
"""
)

tips_chain = tips_prompt | llm | parser

# ==========================================================
# CHAIN 3: PLACES TO VISIT (USES ROUTE + DESTINATION)
# ==========================================================
places_prompt = PromptTemplate(
    input_variables=["destination", "route_plan"],
    template="""
You are a biker travel guide.

Based on the destination and route plan below,
suggest 4–5 scenic places to visit near the destination.

Destination:
{destination}

Route Plan:
{route_plan}

Places:
"""
)

places_chain = places_prompt | llm | parser

# ==========================================================
# SEQUENTIAL PIPELINE
# ==========================================================
biker_trip_chain = (
    RunnableMap({
        "route_plan": route_chain,
        "destination": RunnablePassthrough()
    })
    | RunnableMap({
        "route_plan": RunnablePassthrough(),
        "riding_tips": tips_chain,
        "places_to_visit": places_chain
    })
)


# -------------------- ROUTES --------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    start = request.json.get("start")
    destination = request.json.get("destination")

    # 1️⃣ Route planning
    route_plan = route_chain.invoke({
        "start": start,
        "destination": destination
    })

    # 2️⃣ Riding tips (uses route output)
    riding_tips = tips_chain.invoke({
        "route_plan": route_plan
    })

    # 3️⃣ Places to visit (uses route + destination)
    places_to_visit = places_chain.invoke({
        "destination": destination,
        "route_plan": route_plan
    })

    return jsonify({
        "route_plan": markdown.markdown(route_plan),
        "riding_tips": markdown.markdown(riding_tips),
        "places_to_visit": markdown.markdown(places_to_visit)
    })


if __name__ == "__main__":
    app.run(debug=True)
