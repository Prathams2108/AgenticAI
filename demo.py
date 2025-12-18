import os
import markdown
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

app = Flask(__name__)

llm = ChatOpenAI(
    model="google/gemma-3-27b-it:free",
    temperature=0.7,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# -------------------- LOCATION → DISHES --------------------
location_examples = [
    {"input": "Italy", "output": "Some classic dishes from Italy include pizza, pasta carbonara, and risotto."},
    {"input": "India", "output": "Traditional Indian dishes include Chicken Kebab, Masala Dosa and Chole Bhature."},
]

location_example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Location: {input}\nResponse: {output}"
)

location_prompt = FewShotPromptTemplate(
    examples=location_examples,
    example_prompt=location_example_prompt,
    prefix="Suggest two or three classic dishes from the location.\n\n",
    suffix="Location: {location}\nResponse:",
    input_variables=["location"]
)

location_chain = location_prompt | llm | StrOutputParser()

# -------------------- DISH → RECIPE --------------------
recipe_examples = [
    {
        "input": "Pizza, pasta carbonara, risotto",
        "output": (
            "The easiest dish to cook at home is Pasta Carbonara because it requires minimal ingredients "
            "and simple techniques.\n\n"
            "Ingredients:\n"
            "- Spaghetti\n"
            "- Eggs\n"
            "- Parmesan cheese\n"
            "- Bacon or pancetta\n"
            "- Black pepper\n\n"
            "Steps:\n"
            "- Boil the spaghetti until al dente\n"
            "- Cook the bacon until crisp\n"
            "- Whisk eggs with cheese\n"
            "- Combine everything off the heat and season with pepper"
        )
    }
]


recipe_example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Meals: {input}\nResponse: {output}"
)

recipe_prompt = FewShotPromptTemplate(
    examples=recipe_examples,
    example_prompt=recipe_example_prompt,
    prefix=(
        "From the given list of meals, identify the easiest dish to cook at home.\n"
        "Explain briefly why it is easiest, then provide a detailed recipe.\n"
        "Use clear section headings like Ingredients and Steps.\n"
        "Use bullet points for ingredients and steps.\n\n"
    ),
    suffix="Meals: {meal}\nResponse:",
    input_variables=["meal"]
)

dish_chain = recipe_prompt | llm | StrOutputParser()

# -------------------- TIME & COST --------------------
time_prompt = PromptTemplate(
    input_variables=["recipe"],
    template="Estimate cooking time and cost for this recipe:\n{recipe}"
)

time_chain = time_prompt | llm | StrOutputParser()

# -------------------- ROUTES --------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    location = request.json.get("location")

    meal = location_chain.invoke({"location": location})
    recipe = dish_chain.invoke({"meal": meal})
    time = time_chain.invoke({"recipe": recipe})

    return jsonify({
        "meal": markdown.markdown(meal),
        "recipe": markdown.markdown(recipe),
        "time": markdown.markdown(time)
    })

if __name__ == "__main__":
    app.run(debug=True)
