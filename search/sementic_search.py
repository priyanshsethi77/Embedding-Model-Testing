from embeddings.embed_utils import cosine_similarity
from mongo.mongo_utils import get_all_documents
import numpy as np

def search(query, embed_fn, top_k=6):
    query_vec = embed_fn(query)
    docs = get_all_documents()
    results = []
    for doc in docs:
        doc_vec = np.array(doc["embedding"])
        score = cosine_similarity(query_vec, doc_vec)
        results.append((doc["text"], score))

    return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
