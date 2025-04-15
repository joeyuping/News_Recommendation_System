import numpy as np
import faiss
import pickle

# Load the FAISS index
faiss_index = faiss.read_index("faiss_news_index/index.faiss")

# Print info about the FAISS index
print("FAISS index type:", type(faiss_index))
print("Number of vectors:", faiss_index.ntotal)

# Try to get the vectors (for IndexFlatIP/L2)
try:
    xb = faiss_index.reconstruct_n(0, min(5, faiss_index.ntotal))  # Get up to 5 example vectors
    print("Example vectors (first 5):\n", xb)
    print("Shape of vectors:", xb.shape)
    print("Data type:", xb.dtype)
except Exception as e:
    print("Could not extract vectors directly:", e)

index = np.load("faiss_news_index/index.pkl", allow_pickle=True)

print("Type of loaded object:", type(index))

# If it's a dict or has keys, print them
if hasattr(index, 'keys'):
    print("Keys in index.pkl:", index.keys())
    # Try to print the indices or id mapping if present
    if 'index' in index:
        print("FAISS index object:", index['index'])
    if 'ids' in index:
        print("FAISS ids:", index['ids'])
    if 'docstore' in index:
        print("Docstore:", index['docstore'])
else:
    print("Loaded object:", index)

# Load ID mapping and docstore from index.pkl
index_pkl = np.load("faiss_news_index/index.pkl", allow_pickle=True)
if hasattr(index_pkl, 'keys'):
    print("Keys in index.pkl:", index_pkl.keys())
    # Try to print the title of the news article for the first few IDs
    if 'docstore' in index_pkl and 'ids' in index_pkl:
        docstore = index_pkl['docstore']
        ids = index_pkl['ids']
        print("Example FAISS IDs:", ids[:5])
        for i, doc_id in enumerate(ids[:5]):
            # The docstore may be a dict or an object with a _dict attribute
            doc_dict = docstore.get('_dict', docstore) if hasattr(docstore, 'get') else docstore._dict
            doc = doc_dict.get(doc_id)
            if doc is not None and hasattr(doc, 'metadata') and 'title' in doc.metadata:
                print(f"ID {doc_id}: Title = {doc.metadata['title']}")
            else:
                print(f"ID {doc_id}: No title found")
else:
    print("Loaded object from index.pkl:", index_pkl)