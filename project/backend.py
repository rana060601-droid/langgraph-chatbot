from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv


load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash-lite')

checkpointer = InMemorySaver()

# Create a State

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]


def chat_node(state : ChatState) -> ChatState:
    message = state['messages']

    response = model.invoke(message)

    return {'messages' : [response]}


# Create a Graph
graph = StateGraph(ChatState)

# Create a Node
graph.add_node('chat_node', chat_node)

# Create Edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# Compile the graph
chatbot = graph.compile(checkpointer=checkpointer)