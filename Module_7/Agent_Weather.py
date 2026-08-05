import json
import requests
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

MODEL = "llama3.1" 

def get_weather(city: str) -> dict:
  
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_resp = requests.get(geo_url, params={"name": city, "count": 1}).json()

    if "results" not in geo_resp or not geo_resp["results"]:
        return {"error": f"Could not find location: {city}"}

    location = geo_resp["results"][0]
    lat, lon = location["latitude"], location["longitude"]

    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_resp = requests.get(
        weather_url,
        params={
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
        },
    ).json()

    current = weather_resp.get("current_weather", {})
    return {
        "city": location.get("name", city),
        "country": location.get("country", ""),
        "temperature_C": current.get("temperature"),
        "windspeed_kmh": current.get("windspeed"),
        "weather_code": current.get("weathercode"),
    }


weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given city name.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Chennai'",
                }
            },
            "required": ["city"],
        },
    },
}


def ask(question: str):
    messages = [{"role": "user", "content": question}]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=[weather_tool],
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)  

        for tool_call in msg.tool_calls:
            if tool_call.function.name == "get_weather":
                args = json.loads(tool_call.function.arguments)
                result = get_weather(args["city"])

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )

        final = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        return final.choices[0].message.content

    return msg.content


if __name__ == "__main__":
    answer = ask("Is monsoon started at chennai?")
    print(answer)
