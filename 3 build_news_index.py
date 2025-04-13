import json
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the JSON file with explicit UTF-8 encoding
with open('./news.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract the full content from each item
texts = []
metadata_list = []
for item in data:
    if isinstance(item, dict) and 'full_content' in item and item['full_content'] is not None:
        # Include title, source name, and URL in the content
        title = item.get('title', 'No Title')
        source_name = item.get('source', {}).get('name', 'Unknown Source')
        url = item.get('url', 'No URL')
        
        # Format the content with the additional information
        formatted_content = f"TITLE: {title}\nSOURCE: {source_name}\nURL: {url}\n\nCONTENT:\n{item['full_content']}"
        texts.append(formatted_content)
        
        # Create metadata for additional context
        metadata = {
            'title': title,
            'source': source_name,
            'url': url,
            'publishedAt': item.get('publishedAt', '')
        }
        metadata_list.append(metadata)

# Create documents from the texts with metadata
archive_data = [
    Document(page_content=text, metadata=metadata) 
    for text, metadata in zip(texts, metadata_list) 
    if text
]

print(f"Loaded {len(archive_data)} documents")

# Instead of splitting into chunks, use the whole articles
docs = archive_data

# Build FAISS vectorstore from documents using HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/distiluse-base-multilingual-cased-v1", 
    model_kwargs={'device': device}
)
print(f"Embedding model loaded on {device}")

try:
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    print("FAISS index built successfully")
    
    # Save the FAISS index for future use
    db.save_local("faiss_news_index")
    print("FAISS index saved to 'faiss_news_index'")
except Exception as e:
    print(f"Error building FAISS index: {str(e)}")
    
    # Save documents to pickle as fallback
    import pickle
    with open('processed_news_docs.pkl', 'wb') as f:
        pickle.dump(docs, f)
    print("Saved processed documents to 'processed_news_docs.pkl' as fallback")
