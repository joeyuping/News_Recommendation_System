import streamlit as st
import pickle
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.load import loads

# Disable Streamlit's file watcher to avoid PyTorch conflicts
os.environ["STREAMLIT_SERVER_WATCH_FILES"] = "false"

# Set page configuration
st.set_page_config(
    page_title="News RAG Assistant",
    page_icon="📰",
    layout="wide"
)

device = "cpu"

# Function to load documents and create retriever
@st.cache_resource
def load_retriever():
    try:
        # Try to load FAISS index if it exists
        if os.path.exists("faiss_news_index"):
            # Load embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-l6-v2", 
                model_kwargs={'device': device}
            )
            # Load FAISS index with allow_dangerous_deserialization set to True
            # This is safe because we created this index ourselves
            db = FAISS.load_local("faiss_news_index", embeddings, allow_dangerous_deserialization=True)
            st.sidebar.success("Successfully loaded FAISS index!")
        else:
            # Load from pickle file if FAISS index doesn't exist
            try:
                # First try using langchain's safe loads method
                with open('processed_news_docs.pkl', 'rb') as f:
                    file_content = f.read()
                    docs = loads(file_content)
            except Exception as load_error:
                st.sidebar.warning(f"Could not load with langchain safe loader: {str(load_error)}")
                # Fall back to regular pickle with warning
                with open('processed_news_docs.pkl', 'rb') as f:
                    st.sidebar.warning("Using pickle.load() with your own file - this should be safe if you created this file yourself.")
                    docs = pickle.load(f)
            
            # Create embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-l6-v2", 
                model_kwargs={'device': device}
            )
            
            # Create FAISS index
            db = FAISS.from_documents(docs, embeddings)
            
            # Save for future use
            db.save_local("faiss_news_index")
            st.sidebar.success("Created and saved new FAISS index!")
            
        return db.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.sidebar.error(f"Error loading retriever: {str(e)}")
        return None

# Main UI
st.title("📰 News RAG Recommender System")
st.markdown("""
This app uses Retrieval Augmented Generation (RAG) with the Gemini 2.0 API to recommend news articles.
""")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# API Key handling
try:
    # Try to load API key from secrets
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    st.sidebar.success("API key loaded from secrets successfully!")
    model_configured = True
except Exception as e:
    # Fall back to manual input if secrets not available
    api_key = st.sidebar.text_input("Enter your Google API Key:", type="password")
    if api_key:
        genai.configure(api_key=api_key)
        st.sidebar.success("API key configured successfully!")
        model_configured = True
    else:
        st.sidebar.warning("Please enter your Google API Key to continue")
        model_configured = False

# Load retriever
retriever = load_retriever()

# Add toggle for using retriever
use_retriever = st.sidebar.checkbox("Use Retriever (RAG)", value=True, 
                                   help="Toggle to switch between using retrieval augmented generation or direct questioning")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the news articles..."):
    if not model_configured:
        st.error("Please enter your Google API Key in the sidebar to continue.")
    elif retriever is None and use_retriever:
        st.error("Retriever not loaded. Check the sidebar for errors.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # Create Gemini model
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                if use_retriever and retriever is not None:
                    # Get relevant documents using the newer invoke method instead of deprecated get_relevant_documents
                    try:
                        # Try the newer invoke method first
                        docs = retriever.invoke(prompt)
                    except (AttributeError, TypeError):
                        # Fall back to the deprecated method if invoke is not available
                        docs = retriever.get_relevant_documents(prompt)
                    
                    # Format context from retrieved documents
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Create prompt with context
                    system_prompt = f"""You are an AI assistant that helps recommend news articles related to the user query and summarizes the articles.

                    NOTE: 
                    - Answer in the language of the USER QUESTION, either in English or Traditional Chinese.
                    - Ignore part of the news that is not related to the question.
                    - If no retrieved news are related to the question, say so.
                    
                    RELATED NEWS:
                    {context}
                    
                    USER QUESTION:
                    {prompt}
                    
                    Answer the question based only on the provided news articles. Structure your response by first providing the list of recommended news articles in the format :
                    - **title** - agency (hyperlink to url)
                    Then summarize the articles, while also answering the user's query.

                    ANSWER:
                    """
                else:
                    # Direct questioning without retrieval
                    system_prompt = f"""You are an AI assistant that helps recommend news articles related to the user query.

                    NOTE: Answer in the language of the USER QUESTION, either in English or Traditional Chinese.
                    
                    USER QUESTION:
                    {prompt}.
                    
                    ANSWER:
                    """
                
                # Generate response
                response = model.generate_content(system_prompt)
                answer = response.text
                
                # Display response
                message_placeholder.markdown(answer)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                message_placeholder.error(f"Error: {str(e)}")

# Sidebar information
with st.sidebar:
    st.header("About")
    
    st.header("System Status")
    st.write(f"Device: {device}")
    
    if retriever is not None:
        st.success("Retriever: Loaded ✅")
    else:
        st.error("Retriever: Not loaded ❌")
    
    st.write(f"RAG Mode: {'Enabled' if use_retriever else 'Disabled'}")
        
    st.header("Instructions")
    st.markdown("""
    1. Enter your Google API key in the sidebar
    2. Choose whether to use the retriever (RAG) or not
    3. Ask questions about natural disasters in the chat
    4. The system will generate a response, with or without context from news articles
    """)
