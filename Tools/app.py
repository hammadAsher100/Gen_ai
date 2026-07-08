import streamlit as st
from main import agent

# 1. Page Configuration
st.set_page_config(
    page_title="AI City News & Weather",
    page_icon="🌤️",
    layout="wide"
)

# 2. Main UI Header
st.title("🌤️ AI City Weather & News Agent")
st.write(
    "Enter the name of any city, and the AI agent will automatically orchestrate "
    "tool calls to retrieve live weather data and top current news."
)

# 3. Input Section
city = st.text_input(
    "City Name",
    placeholder="e.g., Tokyo, New York, Karachi"
)

# 4. Agent Execution & Results
if st.button("Search", use_container_width=True):
    if not city.strip():
        st.warning("Please enter a valid city name.")
        st.stop()

    with st.spinner("AI Agent is working..."):
        try:
            response = agent.invoke(
                {
                    "messages": [
                        (
                            "user",
                            f"Give me the current weather and latest news for {city}. Use both tools."
                        )
                    ]
                }
            )
            
            # Extract final response content safely
            answer = response["messages"][-1].content
            
            # Render enhanced UI presentation layout
            st.divider()
            st.subheader(f"📍 {city.strip().title()}")
            st.markdown(answer)
            
        except Exception as e:
            st.error(f"An error occurred while executing the agent: {e}")