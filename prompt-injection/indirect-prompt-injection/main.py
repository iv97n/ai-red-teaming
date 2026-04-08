from typing import TypedDict, List
from typing import Annotated

from langchain.tools import tool
from langchain_core.tools import create_retriever_tool
from langchain_core.messages import HumanMessage, AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings


# -------- Vector Database creation and initialization -----------
# Create the vector database
embeddings = OllamaEmbeddings(model="nomic-embed-text")
# Initialize the vector database
loader = DirectoryLoader(
    "./database",           
    glob="**/*.txt",   
    loader_cls=TextLoader
)
docs = loader.load()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#docs = splitter.split_documents(docs)

# -------- Tools definition --------
# Tool for retrieving the documents
retriever_tool = create_retriever_tool(
    retriever,
    name="search_docs",
    description="Search and return relevant info from the knowledge base.",
)
# Tool for summing two numbers
@tool
def sum(a: int, b: int) -> int:
    """Sum two integers a and b"""
    return a + b

tools = [retriever_tool, sum]


# --------  Model definition and tool binding -------- 
model = ChatOllama(model="qwen3:8b").bind_tools(tools)

# -------- LangGraph pipeline definition -------- 
# State of the agent, basically the conversation history
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def llm(state: AgentState):
    return {"messages": [model.invoke(state["messages"])]}

# LangGraph graph definition
builder = StateGraph(AgentState)
builder.add_node("assistant", llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

# -------- Testing -------- 
messages = [HumanMessage(content="How is the weather in california? Reply using a tool")]
print(messages)
messages = react_graph.invoke({"messages": messages, "input_file": None})
print(messages)
