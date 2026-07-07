from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langgraph.prebuilt import create_react_agent

from Weather_News import weather_agent, news_agent

# Initialize the LLM
llm = ChatMistralAI(
    model="mistral-small-2603",
    temperature=0
)

# Create the agent
agent = create_react_agent(
    model=llm,
    tools=[
        weather_agent,
        news_agent
    ]
)


def get_city_report(city: str) -> str:
    """
    Returns the weather and latest news for a given city.
    """

    response = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"""
                    Give me:

                    1. Current weather
                    2. Latest news
For each news article:
• Give the headline.
• Write a detailed summary of around 100 to 150 words.
• Explain what happened, why it matters, and any important background.
• Do not simply copy the search snippet.
• Use the retrieved information to generate a clear, natural summary.          
Finally, present everything using Markdown headings.              
                    for {city}.

                    Use both available tools and present the information in a clean, well formatted manner.
                    """
                )
            ]
        }
    )

    return response["messages"][-1].content


# Allows the file to be run directly from the terminal
if __name__ == "__main__":
    city = input("Enter a city name: ")
    print(get_city_report(city))