#!/usr/bin/env python3
"""
Test script to verify LangGraph Chatbot setup
Run this after setup to ensure everything works
"""

import sys
import os
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ required")
        return False
    print("✅ Python version OK")
    return True

def check_required_files():
    """Check if required files exist"""
    print_header("Checking Required Files")
    
    required_files = [
        "unified_backend.py",
        "unified_frontend.py",
        "requirements.txt",
        ".env"
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} not found")
            all_exist = False
    
    return all_exist

def check_environment_variables():
    """Check environment variables"""
    print_header("Checking Environment Variables")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return False
    
    if api_key == "sk-your-openai-api-key-here":
        print("❌ OPENAI_API_KEY not configured (still using placeholder)")
        return False
    
    if not api_key.startswith("sk-"):
        print("❌ OPENAI_API_KEY format looks incorrect")
        return False
    
    print(f"✅ OPENAI_API_KEY configured ({api_key[:10]}...)")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Python Packages")
    
    required_packages = [
        ("langchain", "LangChain"),
        ("langgraph", "LangGraph"),
        ("streamlit", "Streamlit"),
        ("openai", "OpenAI"),
        ("faiss", "FAISS"),
        ("langchain_openai", "LangChain OpenAI"),
    ]
    
    all_installed = True
    for module, name in required_packages:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} not installed")
            all_installed = False
    
    return all_installed

def test_backend_import():
    """Test if backend can be imported"""
    print_header("Testing Backend Import")
    
    try:
        import unified_backend
        print("✅ Backend imports successfully")
        
        # Check if chatbot is compiled
        if hasattr(unified_backend, 'chatbot'):
            print("✅ Chatbot graph compiled")
        else:
            print("❌ Chatbot graph not found")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Backend import failed: {e}")
        return False

def test_openai_connection():
    """Test OpenAI API connection"""
    print_header("Testing OpenAI Connection")
    
    try:
        from langchain_openai import ChatOpenAI
        from dotenv import load_dotenv
        
        load_dotenv()
        
        llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=10)
        response = llm.invoke("Say 'test successful' and nothing else")
        
        print(f"✅ OpenAI connection OK")
        print(f"   Response: {response.content}")
        return True
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        return False

def test_database():
    """Test database creation"""
    print_header("Testing Database")
    
    try:
        import sqlite3
        
        # Test database creation
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("SELECT sqlite_version()")
        version = cursor.fetchone()[0]
        conn.close()
        
        print(f"✅ SQLite version {version}")
        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def run_basic_chatbot_test():
    """Run a basic chatbot test"""
    print_header("Testing Chatbot Functionality")
    
    try:
        from unified_backend import chatbot
        from langchain_core.messages import HumanMessage
        import uuid
        
        # Create test config
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        # Send test message
        print("Sending test message...")
        response = chatbot.invoke(
            {"messages": [HumanMessage(content="Say 'test' and nothing else")]},
            config=config
        )
        
        # Check response
        if response and "messages" in response:
            last_message = response["messages"][-1]
            print(f"✅ Chatbot response: {last_message.content}")
            return True
        else:
            print("❌ No response from chatbot")
            return False
            
    except Exception as e:
        print(f"❌ Chatbot test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n🤖 LangGraph Chatbot - Setup Verification")
    print("="*60)
    
    results = []
    
    # Run all checks
    results.append(("Python Version", check_python_version()))
    results.append(("Required Files", check_required_files()))
    results.append(("Environment Variables", check_environment_variables()))
    results.append(("Python Packages", check_dependencies()))
    results.append(("Backend Import", test_backend_import()))
    results.append(("Database", test_database()))
    results.append(("OpenAI Connection", test_openai_connection()))
    results.append(("Chatbot Functionality", run_basic_chatbot_test()))
    
    # Print summary
    print_header("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All tests passed! Your chatbot is ready to use.")
        print("\nTo start the chatbot, run:")
        print("  streamlit run unified_frontend.py")
        print("\nOr use the run script:")
        print("  ./run.sh")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Configure .env file with your API key")
        print("3. Check SETUP_GUIDE.md for detailed help")
        return 1

if __name__ == "__main__":
    sys.exit(main())
