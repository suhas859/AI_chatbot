"""
Unified LangGraph Chatbot Frontend
Professional Streamlit UI with all features integrated
"""
import uuid
from datetime import datetime

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from unified_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
    thread_has_document,
    clear_thread_document,
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="LangGraph AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .thread-button {
        width: 100%;
        text-align: left;
        font-family: monospace;
        font-size: 0.85rem;
    }
    div[data-testid="stStatusWidget"] {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def generate_thread_id() -> str:
    """Generate a unique thread ID."""
    return str(uuid.uuid4())


def format_thread_name(thread_id: str) -> str:
    """Create a readable thread name from the first user message."""
    # Try to get first user message as the title (like ChatGPT)
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
        for msg in messages:
            if isinstance(msg, HumanMessage) and msg.content:
                # Truncate to 30 chars for sidebar display
                content = msg.content.strip()
                if len(content) > 30:
                    return content[:27] + "..."
                return content
    except Exception:
        pass
    # Fallback to "New Chat" for empty conversations
    return "New Chat"


def reset_chat():
    """Start a new chat thread."""
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.session_state["last_activity"] = datetime.now()


def add_thread(thread_id: str):
    """Add thread to the list if not already present."""
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id: str):
    """Load conversation history for a thread from the database."""
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        return state.values.get("messages", [])
    except Exception as e:
        st.error(f"Failed to load conversation: {e}")
        return []


def switch_thread(thread_id: str):
    """Switch to a different conversation thread."""
    st.session_state["thread_id"] = thread_id
    messages = load_conversation(thread_id)

    temp_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        # Only show user and assistant messages, not system or tool messages
        if hasattr(msg, 'content') and msg.content:
            temp_messages.append({"role": role, "content": msg.content})
    
    st.session_state["message_history"] = temp_messages
    st.session_state["last_activity"] = datetime.now()


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "last_activity" not in st.session_state:
    st.session_state["last_activity"] = datetime.now()

# Ensure current thread is in the list
add_thread(st.session_state["thread_id"])

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### 🤖 LangGraph Chatbot")
    st.markdown("---")
    
    # Current thread info
    current_thread = st.session_state["thread_id"]
    st.markdown(f"**Current Thread:**")
    st.code(format_thread_name(current_thread), language=None)
    
    # New chat button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        reset_chat()
        st.rerun()
    
    st.markdown("---")
    
    # PDF Upload Section
    st.markdown("### 📄 Document Upload")
    
    thread_key = str(current_thread)
    has_doc = thread_has_document(thread_key)
    
    if has_doc:
        doc_meta = thread_document_metadata(thread_key)
        st.success(
            f"✅ **{doc_meta.get('filename', 'Document')}**\n\n"
            f"📄 {doc_meta.get('documents', 0)} pages\n\n"
            f"📊 {doc_meta.get('chunks', 0)} chunks"
        )
        
        if st.button("🗑️ Clear Document", use_container_width=True):
            clear_thread_document(thread_key)
            st.rerun()
    else:
        st.info("No document uploaded for this chat")
    
    uploaded_pdf = st.file_uploader(
        "Upload PDF to chat about",
        type=["pdf"],
        help="Upload a PDF to enable document Q&A"
    )
    
    if uploaded_pdf:
        if has_doc and doc_meta.get('filename') == uploaded_pdf.name:
            st.info(f"'{uploaded_pdf.name}' already indexed")
        else:
            with st.status("📑 Indexing PDF...", expanded=True) as status_box:
                try:
                    summary = ingest_pdf(
                        uploaded_pdf.getvalue(),
                        thread_id=thread_key,
                        filename=uploaded_pdf.name,
                    )
                    status_box.update(
                        label=f"✅ Indexed {summary['chunks']} chunks",
                        state="complete",
                        expanded=False
                    )
                    st.rerun()
                except Exception as e:
                    status_box.update(
                        label=f"❌ Indexing failed",
                        state="error",
                        expanded=True
                    )
                    st.error(f"Error: {e}")
    
    st.markdown("---")
    
    # Conversation History
    st.markdown("### 💬 Conversation History")
    
    threads = st.session_state["chat_threads"][::-1]  # Most recent first
    
    if not threads:
        st.write("No conversations yet")
    else:
        # Show max 10 most recent threads
        for idx, thread_id in enumerate(threads[:10]):
            thread_name = format_thread_name(thread_id)
            
            # Highlight current thread
            is_current = thread_id == current_thread
            button_type = "primary" if is_current else "secondary"
            
            # Show document indicator
            doc_indicator = "📄 " if thread_has_document(str(thread_id)) else ""
            
            if st.button(
                f"{doc_indicator}{thread_name}",
                key=f"thread-{thread_id}",
                use_container_width=True,
                type=button_type,
                disabled=is_current
            ):
                switch_thread(thread_id)
                st.rerun()
        
        if len(threads) > 10:
            st.caption(f"Showing 10 of {len(threads)} conversations")
    
    st.markdown("---")
    
    # Help section
    with st.expander("ℹ️ Available Tools"):
        st.markdown("""
        **Your AI assistant can:**
        - 🔍 Search the web for current information
        - 📊 Get real-time stock prices
        - 🧮 Perform calculations
        - 📄 Answer questions about uploaded PDFs
        
        **Tips:**
        - Upload a PDF to enable document Q&A
        - Ask about stocks: "What's the price of AAPL?"
        - Search the web: "Latest news about AI"
        - Calculate: "What's 15% of 250?"
        """)

# =============================================================================
# MAIN CHAT INTERFACE
# =============================================================================
st.markdown('<div class="main-header">🤖 AI Assistant</div>', unsafe_allow_html=True)

# Display conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask me anything... 💬", key="chat_input")

if user_input:
    # Add user message to history
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Prepare config
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }
    
    # Assistant response with streaming and tool status
    with st.chat_message("assistant"):
        status_holder = {"box": None, "tool_name": None}
        
        def ai_stream_with_tools():
            """Generator that yields AI tokens and shows tool usage."""
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Handle tool execution
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    
                    # Create or update status box
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}`...",
                            expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}`...",
                            state="running",
                            expanded=True,
                        )
                    status_holder["tool_name"] = tool_name
                
                # Stream AI response tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content
        
        # Stream the response
        ai_message = st.write_stream(ai_stream_with_tools())
        
        # Finalize tool status if any tool was used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label=f"✅ `{status_holder['tool_name']}` completed",
                state="complete",
                expanded=False
            )
    
    # Add assistant message to history
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })
    
    # Update last activity
    st.session_state["last_activity"] = datetime.now()

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"🕒 Thread: {format_thread_name(st.session_state['thread_id'])}")

with col2:
    msg_count = len(st.session_state["message_history"])
    st.caption(f"💬 Messages: {msg_count}")

with col3:
    if thread_has_document(str(st.session_state["thread_id"])):
        st.caption("📄 Document available")
    else:
        st.caption("📄 No document")
