from dotenv import load_dotenv
load_dotenv()
import os
import requests
from tavily import TavilyClient
from langchain.tools import tool 

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
open_weather = os.getenv("OPENWEATHER_API_KEY")

@tool
def weather_agent(city:str):
    """
    Fetches the current weather information for a given city using the OpenWeatherMap API.

    Args:
        city (str): The name of the city for which to retrieve weather information.
    """
    
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={open_weather}"
    
    params = {
        "q": city,
        "appid": open_weather,
        "units": "metric"  # You can change this to 'imperial' for Fahrenheit
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        return f"Error fetching weather data for {city}"
    else:
     data = response.json()
     weather_description = data["weather"][0]["description"]
     temperature = data["main"]["temp"]
     humidity = data["main"]["humidity"]
     wind_speed = data["wind"]["speed"]

    weather_info = (
            f"Weather in {city}:\n"
            f"Description: {weather_description}\n"
            f"Temperature: {temperature}°C\n"
            f"Humidity: {humidity}%\n"
            f"Wind Speed: {wind_speed} m/s"
        )
        
    return weather_info
  
  
@tool
def news_agent(city:str) ->str:
  """
  Returns latest news about a city
  """
  # pyright: ignore[reportUndefinedVariable]
  result = tavily.search(query=f"Latest news about {city}" , max_results=5)
  
  articles = result["results"]
  
  if not articles:
    return "No recent news was found from {city} city"
  
  output = ""
  
  for i, article in enumerate(articles,start=1):
    output+=f"""
    ({i})
    Title: {article['title']}
    Content: {article['content']}
    Source: {article['url']}
    """
    
    return output