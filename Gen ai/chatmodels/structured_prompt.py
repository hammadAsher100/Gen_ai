import streamlit as st
import json
import re
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

# Load environment variables
load_dotenv()

@st.cache_resource
def get_model():
    return ChatMistralAI(model="ministral-3b-latest", temperature=0.7, max_tokens=512)

model = get_model()

class MovieModel(BaseModel):
    title: str = Field(description="The title of the movie")
    release_year: int = Field(description="The year the movie was released")
    genres: List[str] = Field(description="The genres of the movie")
    director: Optional[str] = Field(description="The director of the movie")
    rating: float = Field(description="The rating of the movie")
    summary: Optional[str] = Field(description="A brief summary of the movie")

parser = PydanticOutputParser(pydantic_object=MovieModel)

prompt = ChatPromptTemplate.from_messages([
    ('system', 
     """You are a movie expert. Please provide detailed information about the movie based on the user's query. 
      {format_instructions}  
     """),
    ('human', 'Please provide information about the movie "{movie_name}".')
])

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Movie Expert", page_icon="🎬", layout="centered")

st.title("🎬 AI Movie Expert")
st.write("Enter the name of any movie below to get structured, AI-powered insights.")

with st.form(key="movie_form"):
    movie_name = st.text_input("Enter the name of the movie:", placeholder="e.g., Inception, The Godfather...")
    submit_button = st.form_submit_button(label="Search Movie")

if submit_button and movie_name.strip():
    with st.spinner(f"Fetching details for '{movie_name}'..."):
        try:
            final_prompt = prompt.invoke({
                "movie_name": movie_name.strip(),
                "format_instructions": parser.get_format_instructions()
            })

            response = model.invoke(final_prompt)

            # Strip markdown code fences if present
            raw = response.content
            raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

            parsed_data: MovieModel = parser.parse(raw)

            st.success(f"Found information for **{parsed_data.title}**!")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Release Year", value=parsed_data.release_year)
            with col2:
                st.metric(label="Director", value=parsed_data.director or "Unknown")
            with col3:
                st.metric(label="Rating", value=f"⭐ {parsed_data.rating}/10")

            st.markdown("---")

            st.write("**Genres:**")
            genres_html = "".join([f"<span style='background-color:#f0f2f6; padding:5px 10px; margin-right:5px; border-radius:5px;'>{g}</span>" for g in parsed_data.genres])
            st.markdown(genres_html, unsafe_allow_html=True)

            st.markdown("---")

            st.subheader("Summary")
            st.write(parsed_data.summary or "No summary provided.")

            with st.expander("View Raw JSON Output"):
                try:
                    st.json(json.loads(raw))
                except Exception:
                    st.code(raw, language="json")

        except Exception as e:
            st.error(f"An error occurred while parsing the movie data: {e}")
            if 'response' in locals():
                st.text_area("Raw Response:", value=response.content)

elif submit_button:
    st.warning("Please enter a valid movie name.")