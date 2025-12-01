from openai import OpenAI
import json

# Load openai api key
with open('/home/c18ty/.api-keys.json') as f:
    keys = json.load(f)
API_KEY = keys['openai']

client = OpenAI(api_key=API_KEY)

# Prompt
def generate_nutrition_advice(macros, user_profile, goal, user_notes = ""):
    prompt = f"""You are a nutritional assistant, the user has eaten a meal with the following estimated macros. 

    {macros}    

    User Profile:
    Height: {user_profile["height"]}
    Weight: {user_profile["weight"]}
    Age: {user_profile["age"]}
    Goal: {goal}

    Additional notes from user:
    "{user_notes}"

    Provide a nutritional recommendation considering the following:
    - Provide friendly advice that is encouraging to the user.
    - Find something positive to say about the meal.
    - Find at least one thing to improve upon. 
    - Say how the meal they just ate fits into their goal if applicable ({goal}).
    - Provide tips for the user next meal. Provide general tips such as types of food or specific ingredients, not entire recipes/meals.
    - Only make recommendations based on the information available to you. If the user doesn't provide certain information, use what you have. 
    """

    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [{"role": "user", "content": prompt}])

    return response.choices[0].message.content