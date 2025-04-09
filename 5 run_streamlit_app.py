"""
Run script for the News RAG Streamlit app
This script avoids PyTorch and Streamlit file watcher conflicts
"""
import os
import sys
import subprocess

# Disable Streamlit's file watcher to avoid PyTorch conflicts
os.environ["STREAMLIT_SERVER_WATCH_FILES"] = "false"

# Run the Streamlit app
subprocess.run(
    [sys.executable, "-m", "streamlit", "run", "news_rag_app.py", "--server.fileWatcherType", "none"],
    check=True
)
