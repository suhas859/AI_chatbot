"""
Unified LangGraph Chatbot Backend
Combines: RAG, Tools, MCP, Database Persistence, Async Support
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import asyncio
import threading
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool, BaseTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

# Optional MCP support
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  MCP not available. Install with: pip install langchain-mcp-adapters")

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
DATABASE_PATH = "chatbot.db"
MODEL_NAME = "gpt-4o-mini"  # Fast and cost-effective
EMBEDDING_MODEL = "text-embedding-3-small"

# =============================================================================
# 1. LLM + EMBEDDINGS
# =============================================================================
llm = ChatOpenAI(model=MODEL_NAME, temperature=0.7)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# =============================================================================
# 2. PDF/RAG MANAGEMENT (Thread-specific retrievers)
# =============================================================================
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.
    
    Args:
        file_bytes: PDF file content as bytes
        thread_id: Unique thread identifier
        filename: Original filename
        
    Returns:
        Dictionary with ingestion summary
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# =============================================================================
# 3. TOOLS DEFINITION
# =============================================================================

# Web Search Tool
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform basic arithmetic operations on two numbers.
    
    Supported operations: add, sub, mul, div
    
    Args:
        first_num: First number
        second_num: Second number
        operation: Operation to perform (add/sub/mul/div)
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'TSLA', 'GOOGL')
        
    Returns:
        Dictionary with stock price information
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    try:
        r = requests.get(url, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": f"Failed to fetch stock price: {str(e)}"}


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    
    Args:
        query: Search query
        thread_id: Thread identifier (automatically provided)
        
    Returns:
        Dictionary with retrieved context and metadata
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    try:
        result = retriever.invoke(query)
        context = [doc.page_content for doc in result]
        metadata = [doc.metadata for doc in result]

        return {
            "query": query,
            "context": context,
            "metadata": metadata,
            "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
        }
    except Exception as e:
        return {"error": f"RAG retrieval failed: {str(e)}"}


@tool
def wikipedia_tool(query: str) -> dict:
    """
    Search Wikipedia for detailed information about a topic.
    
    Args:
        query: Search topic
        
    Returns:
        Dictionary with wikipedia summary
    """
    try:
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
        tool = WikipediaQueryRun(api_wrapper=api_wrapper)
        return {"result": tool.run(query)}
    except Exception as e:
        return {"error": f"Wikipedia search failed: {str(e)}"}


@tool
def weather_tool(city: str) -> dict:
    """
    Get current weather for a specific city using OpenMeteo (Free).
    
    Args:
        city: City name (e.g., "London", "New York", "Tokyo")
        
    Returns:
        Dictionary with weather data
    """
    try:
        # First get coordinates
        geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_res = requests.get(geocoding_url, timeout=5)
        geo_data = geo_res.json()
        
        if not geo_data.get("results"):
            return {"error": f"City '{city}' not found"}
            
        location = geo_data["results"][0]
        lat, lon = location["latitude"], location["longitude"]
        
        # Get weather
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m&timezone=auto"
        weather_res = requests.get(weather_url, timeout=5)
        weather_data = weather_res.json()
        
        current = weather_data.get("current", {})
        units = weather_data.get("current_units", {})
        
        return {
            "city": location["name"],
            "country": location.get("country"),
            "temperature": f"{current.get('temperature_2m')} {units.get('temperature_2m')}",
            "feels_like": f"{current.get('apparent_temperature')} {units.get('apparent_temperature')}",
            "humidity": f"{current.get('relative_humidity_2m')} {units.get('relative_humidity_2m')}",
            "wind_speed": f"{current.get('wind_speed_10m')} {units.get('wind_speed_10m')}",
            "condition_code": current.get("weather_code")
        }
    except Exception as e:
        return {"error": f"Weather fetch failed: {str(e)}"}


@tool
def url_reader(url: str) -> dict:
    """
    Read and summarize the content of a specific URL.
    
    Args:
        url: The website URL to read
        
    Returns:
        Dictionary with page title and text content
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
            
        # Get text
        text = soup.get_text()
        
        # Clean text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return {
            "url": url,
            "title": soup.title.string if soup.title else "No title",
            "content": clean_text[:5000] + "..." if len(clean_text) > 5000 else clean_text
        }
    except Exception as e:
        return {"error": f"Failed to read URL: {str(e)}"}


# =============================================================================
# 4. MCP TOOLS (Optional)
# =============================================================================
def load_mcp_tools() -> list[BaseTool]:
    """Load MCP tools if available and configured."""
    if not MCP_AVAILABLE:
        return []
    
    # Configure your MCP servers here
    # Example configuration (commented out - update with your servers)
    # try:
    #     client = MultiServerMCPClient({
    #         "math": {
    #             "transport": "stdio",
    #             "command": "python3",
    #             "args": ["path/to/mcp-math-server/main.py"],
    #         },
    #     })
    #     
    #     # Synchronous loading for simplicity
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)
    #     tools = loop.run_until_complete(client.get_tools())
    #     loop.close()
    #     return tools
    # except Exception as e:
    #     print(f"⚠️  Failed to load MCP tools: {e}")
    #     return []
    
    return []


# Combine all tools
mcp_tools = load_mcp_tools()
tools = [search_tool, wikipedia_tool, weather_tool, url_reader, get_stock_price, calculator, rag_tool, *mcp_tools]
llm_with_tools = llm.bind_tools(tools)

# =============================================================================
# 5. STATE DEFINITION
# =============================================================================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# =============================================================================
# 6. GRAPH NODES
# =============================================================================
def chat_node(state: ChatState, config=None):
    """
    LLM node that processes messages and may request tool calls.
    
    Includes system message with instructions for RAG and tool usage.
    """
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    # Dynamic system message based on available document
    has_document = thread_id and thread_id in _THREAD_RETRIEVERS
    doc_info = ""
    if has_document:
        meta = _THREAD_METADATA.get(str(thread_id), {})
        doc_info = f" A document '{meta.get('filename')}' is available for this chat."
    
    system_message = SystemMessage(
        content=(
            "You are a helpful AI assistant with access to multiple tools:\n"
            "You are a helpful AI assistant with access to multiple tools:\n"
            "- 🔍 Web search (DuckDuckGo)\n"
            "- 📖 Wikipedia for detailed topics\n"
            "- 🌤️ Weather for any city\n"
            "- 🔗 Read URL for webpage content\n"
            "- 📊 Stock price lookup\n"
            "- 🧮 Calculator for arithmetic\n"
            "- 📄 RAG tool for querying uploaded documents\n"
            f"{doc_info}\n\n"
            "When answering questions about uploaded documents, always use the rag_tool "
            f"with thread_id='{thread_id}'. Be helpful, accurate, and concise."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# =============================================================================
# 7. CHECKPOINTER (SQLite Persistence)
# =============================================================================
conn = sqlite3.connect(database=DATABASE_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# =============================================================================
# 8. GRAPH CONSTRUCTION
# =============================================================================
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# =============================================================================
# 9. HELPER FUNCTIONS
# =============================================================================
def retrieve_all_threads() -> list[str]:
    """Get all thread IDs from the database."""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    """Check if a thread has an indexed document."""
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    """Get metadata for a thread's indexed document."""
    return _THREAD_METADATA.get(str(thread_id), {})


def clear_thread_document(thread_id: str) -> bool:
    """Clear the indexed document for a thread."""
    thread_key = str(thread_id)
    if thread_key in _THREAD_RETRIEVERS:
        del _THREAD_RETRIEVERS[thread_key]
    if thread_key in _THREAD_METADATA:
        del _THREAD_METADATA[thread_key]
        return True
    return False


# =============================================================================
# INITIALIZATION MESSAGE
# =============================================================================
if __name__ == "__main__":
    print("✅ Unified LangGraph Chatbot Backend Initialized")
    print(f"📁 Database: {DATABASE_PATH}")
    print(f"🤖 Model: {MODEL_NAME}")
    print(f"🔧 Tools: {len(tools)} available")
    print(f"📊 MCP Tools: {len(mcp_tools)} loaded")
    print(f"💾 Existing threads: {len(retrieve_all_threads())}")
