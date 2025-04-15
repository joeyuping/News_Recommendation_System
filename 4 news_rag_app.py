import streamlit as st
import pickle
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.load import loads
import time

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
                model_name="sentence-transformers/distiluse-base-multilingual-cased-v1", 
                model_kwargs={'device': device}
            )
            # Load FAISS index with allow_dangerous_deserialization set to True
            # This is safe because we created this index ourselves
            db = FAISS.load_local("faiss_news_index", embeddings, allow_dangerous_deserialization=True)
            st.sidebar.success("Successfully loaded FAISS index!")
            
            # Debug info
            print(f"FAISS index loaded with {db.index.ntotal} vectors")
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
                model_name="sentence-transformers/distiluse-base-multilingual-cased-v1", 
                model_kwargs={'device': device}
            )
            
            # Create FAISS index
            db = FAISS.from_documents(docs, embeddings)
            
            # Save for future use
            db.save_local("faiss_news_index")
            st.sidebar.success("Created and saved new FAISS index!")
            
            # Debug info
            print(f"FAISS index created with {db.index.ntotal} vectors")
            
        return db.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.sidebar.error(f"Error loading retriever: {str(e)}")
        return None

# Main UI
st.title("📰 News RAG Recommender System")
st.markdown("""
This app uses Retrieval Augmented Generation (RAG) with the Gemini 1.5 Pro API to recommend news articles.
""")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# API Key handling
try:
    # Try to load API key from secrets
    api_key = st.secrets["google"]["api_key"]
    if api_key:
        genai.configure(api_key=api_key)
        st.sidebar.success("API Key loaded from secrets!")
        model_configured = True
    else:
        raise ValueError("API key is empty")
except Exception as e:
    api_key = st.sidebar.text_input("Enter your Google API Key:", type="password")
    if api_key:
        genai.configure(api_key=api_key)
        model_configured = True
    else:
        st.sidebar.warning("Please enter your Google API Key to continue")
        model_configured = False

# Load retriever
retriever = load_retriever()

# Add toggle for using retriever
use_retriever = st.sidebar.checkbox("Use Retriever (RAG)", value=True, 
                                   help="Toggle to switch between using retrieval augmented generation or direct questioning")

# Add button to rebuild index
if st.sidebar.button("Rebuild FAISS Index"):
    try:
        st.sidebar.info("Rebuilding FAISS index... This may take a moment.")
        
        # Load documents from pickle
        try:
            with open('processed_news_docs.pkl', 'rb') as f:
                docs = pickle.load(f)
            
            # Create embeddings with current model
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/distiluse-base-multilingual-cased-v1", 
                model_kwargs={'device': device}
            )
            
            # Remove old index if it exists
            if os.path.exists("faiss_news_index"):
                import shutil
                shutil.rmtree("faiss_news_index")
            
            # Create new FAISS index
            db = FAISS.from_documents(docs, embeddings)
            
            # Save for future use
            db.save_local("faiss_news_index")
            
            # Reload retriever
            retriever = load_retriever()
            
            st.sidebar.success("FAISS index rebuilt successfully!")
        except Exception as e:
            st.sidebar.error(f"Error rebuilding index: {str(e)}")
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")

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
                model = genai.GenerativeModel('gemini-1.5-pro')
                
                retrieval_time = None
                llm_time = None
                docs = None
                # --- Time the retrieval step ---
                if use_retriever and retriever is not None:
                    retrieval_start = time.time()
                    try:
                        docs = retriever.invoke(prompt)
                    except (AttributeError, TypeError) as e:
                        docs = retriever.get_relevant_documents(prompt)
                    retrieval_end = time.time()
                    retrieval_time = retrieval_end - retrieval_start
                    context = "\n\n".join([doc.page_content for doc in docs])
                    system_prompt = f"""You are an AI assistant that helps recommend news articles related to the user question and answer the question.
                    
                    RELATED NEWS:
                    {context}
                    
                    YOUR TASK:
                    - Answer using ONLY the retrieved news articles.
                    - If NO retrieved news are related to the question, just say that the news archive does not contain news related to the question.
                    - You MUST answer in the language of the USER QUESTION, either in English or Traditional Chinese.
                    
                    Structure your response by first providing the list of relevant news articles in the format :
                    - **title** - agency (hyperlink to url)
                    Then answer the question. If it is not a question, summarize the news articles.
                    
                    USER QUESTION:
                    {prompt}
                    
                    THINK: What is the user asking? what language is the user asking in?
                    
                    ANSWER:
                    """
                else:
                    system_prompt = f"""You are an AI assistant that helps recommend news articles related to the user query.
                    
                    YOUR TASK:
                    - You MUST answer in the language of the USER QUESTION, either in English or Traditional Chinese.
                    
                    Structure your response by first providing the list of relevant news articles in the format :
                    - **title** - agency (hyperlink to url)
                    Then answer the question. If it is not a question, summarize the news articles.
                    
                    USER QUESTION:
                    {prompt}.
                    
                    ANSWER:
                    """
                # --- Time the LLM response step ---
                llm_start = time.time()
                response = model.generate_content(system_prompt)
                llm_end = time.time()
                llm_time = llm_end - llm_start
                answer = response.text
                # Display response
                timing_info = ""
                if retrieval_time is not None:
                    timing_info += f"\n\n**Retrieval time:** {retrieval_time:.2f} seconds"
                if llm_time is not None:
                    timing_info += f"\n**LLM response time:** {llm_time:.2f} seconds"
                message_placeholder.markdown(answer + timing_info)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer + timing_info})
            except Exception as e:
                message_placeholder.error(f"Error: {str(e)}")

# Sidebar information
with st.sidebar:
    if retriever is not None:
        st.success("Retriever: Loaded ✅")
    else:
        st.error("Retriever: Not loaded ❌")
