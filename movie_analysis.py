import os
from dotenv import load_dotenv
from openai import OpenAI
import json
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import pandas as pd
from io import StringIO
from langchain_openai import ChatOpenAI



memory = SqliteSaver.from_conn_string(":memory:")
from langgraph.graph import StateGraph
#with SqliteSaver.from_conn_string("sqlite:///chat_memory.db"):
 #   graph = graph_builder.compile(checkpointer=memory)
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()


# Load environment variables from .env file
load_dotenv()

#client = OpenAI()

from openai import OpenAI
with open('./config.json') as f:
    config = json.load(f)
    os.environ["OPENAI_API_KEY"] = config['OPENAI_KEY']
openai_key = os.getenv("OPENAI_API_KEY")

tavily = os.getenv("TAVILY_API_KEY")

llm_name = "gpt-3.5-turbo"
model = ChatOpenAI(api_key=openai_key, model=llm_name)

from tavily import TavilyClient

tavily = TavilyClient(api_key=tavily)


from typing import TypedDict, List
#from langchain_core.pydantic_v1 import BaseModel
from pydantic import BaseModel


class AgentState(TypedDict):
    task: str
    similar: List[str]
    csv_file: str
    movie_data: str
    analysis: str
    similar_data: str
    comparison: str
    feedback: str
    report: str
    content: List[str]
    revision_number: int
    max_revisions: int


class Queries(BaseModel):
    queries: List[str]


# Define the prompts for each node - IMPROVE AS NEEDED
GATHER_MOVIE_PROMPT = """You are an expert movie analyst. Gather the movie data for the given years. Provide detailed  data."""
ANALYZE_DATA_PROMPT = """You are an expert movie analyst. Analyze the provided movie data and provide detailed insights and analysis."""
RESEARCH_SIMILAR_PROMPT = """You are a researcher tasked with providing information about similar companies for performance comparison. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""
COMPETE_PERFORMANCE_PROMPT = """You are an expert movie analyst. Compare the movie performance of the given movie with its similar based on the provided data.
**MAKE SURE TO INCLUDE THE NAMES OF THE SIMILAR IN THE COMPARISON.**"""
FEEDBACK_PROMPT = """You are a reviewer. Provide detailed feedback and critique for the provided movie comparison report. Include any additional information or revisions needed."""
WRITE_REPORT_PROMPT = """You are a movie report writer. Write a comprehensive movie report based on the analysis, similar research, comparison, and feedback provided."""
RESEARCH_CRITIQUE_PROMPT = """You are a researcher tasked with providing information to address the provided critique. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""


def gather_movie_node(state: AgentState):
    # Read the CSV file into a pandas DataFrame
    csv_file = state["csv_file"]
    df = pd.read_csv(StringIO(csv_file))

    # Convert the DataFrame to a string
    movie_data_str = df.to_string(index=False)

    # Combine the movie data string with the task
    combined_content = (
        f"{state['task']}\n\nHere is the movie data:\n\n{movie_data_str}"
    )

    messages = [
        SystemMessage(content=GATHER_MOVIE_PROMPT),
        HumanMessage(content=combined_content),
    ]

    response = model.invoke(messages)
    return {"movie_data": response.content}


def analyze_data_node(state: AgentState):
    messages = [
        SystemMessage(content=ANALYZE_DATA_PROMPT),
        HumanMessage(content=state["movie_data"]),
    ]
    response = model.invoke(messages)
    return {"analysis": response.content}


def research_similar_node(state: AgentState):
    content = state["content"] or []
    for similar in state["similar"]:
        queries = model.with_structured_output(Queries).invoke(
            [
                SystemMessage(content=RESEARCH_SIMILAR_PROMPT),
                HumanMessage(content=similar),
            ]
        )
        for q in queries.queries:
            response = tavily.search(query=q, max_results=2)
            for r in response["results"]:
                content.append(r["content"])
    return {"content": content}


def compare_performance_node(state: AgentState):
    content = "\n\n".join(state["content"] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is the movie analysis:\n\n{state['analysis']}"
    )
    messages = [
        SystemMessage(content=COMPETE_PERFORMANCE_PROMPT.format(content=content)),
        user_message,
    ]
    response = model.invoke(messages)
    return {
        "comparison": response.content,
        "revision_number": state.get("revision_number", 1) + 1,
    }


def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state["feedback"]),
        ]
    )
    content = state["content"] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response["results"]:
            content.append(r["content"])
    return {"content": content}


def collect_feedback_node(state: AgentState):
    messages = [
        SystemMessage(content=FEEDBACK_PROMPT),
        HumanMessage(content=state["comparison"]),
    ]
    response = model.invoke(messages)
    return {"feedback": response.content}


def write_report_node(state: AgentState):
    messages = [
        SystemMessage(content=WRITE_REPORT_PROMPT),
        HumanMessage(content=state["comparison"]),
    ]
    response = model.invoke(messages)
    return {"report": response.content}


def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "collect_feedback"


builder = StateGraph(AgentState)

builder.add_node("gather_movie", gather_movie_node)
builder.add_node("analyze_data", analyze_data_node)
builder.add_node("research_similar", research_similar_node)
builder.add_node("compare_performance", compare_performance_node)
builder.add_node("collect_feedback", collect_feedback_node)
builder.add_node("research_critique", research_critique_node)

builder.add_node("write_report", write_report_node)


builder.set_entry_point("gather_movie")


builder.add_conditional_edges(
    "compare_performance",
    should_continue,
    {END: END, "collect_feedback": "collect_feedback"},
)

builder.add_edge("gather_movie", "analyze_data")
builder.add_edge("analyze_data", "research_similar")
builder.add_edge("research_similar", "compare_performance")
builder.add_edge("collect_feedback", "research_critique")
builder.add_edge("research_critique", "compare_performance")
builder.add_edge("compare_performance", "write_report")

graph = builder.compile(checkpointer=memory)

import streamlit as st


def main():
    st.title("Movie recommendation Agent")

    task = st.text_input(
        "Enter the task:",
        "Analyze all Movies available on Internet to suggest some good movie for weekend",
    )
    similar = st.text_area("Enter favourite movie names (one per line):").split("\n")
    max_revisions = st.number_input("Max Revisions", min_value=1, value=2)
    uploaded_file = st.file_uploader(
        "Upload a CSV file with last few movies watched", type=["csv"]
    )

    if st.button("Start Analysis") and uploaded_file is not None:
        # Read the uploaded CSV file
        csv_data = uploaded_file.getvalue().decode("utf-8")

        initial_state = {
            "task": task,
            "similar": [comp.strip() for comp in similar if comp.strip()],
            "csv_file": csv_data,
            "max_revisions": max_revisions,
            "revision_number": 1,
        }
        thread = {"configurable": {"thread_id": "1"}}

        final_state = None
        for s in graph.stream(initial_state, thread):
            st.write(s)
            final_state = s

        if final_state and "recommendation" in final_state:
            st.subheader("Final Recommendation")
            st.write(final_state["recommendation"])


if __name__ == "__main__":
    main()
