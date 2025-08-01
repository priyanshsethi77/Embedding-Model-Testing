from pymongo import MongoClient
import numpy as np

def get_mongo_collection():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["embedding_test"]
    return db["documents"]

# def insert_documents(docs, embed_fn):
#     collection = get_mongo_collection()
#     collection.delete_many({})
#     for doc in docs:
#         vec = embed_fn(doc)
#         collection.insert_one({
#             "text": doc,
#             "embedding": vec.tolist()
#         })

from pymongo import MongoClient

def get_mongo_collection(db_name="embedding_db", collection_name="documents"):
    client = MongoClient("mongodb://localhost:27017")  # adjust URI if needed
    db = client[db_name]
    return db[collection_name]

def insert_documents(documents, embed_fn):
    collection = get_mongo_collection()  # ✅ Define it here
    collection.delete_many({})  # Clear previous docs (optional)

    for doc in documents:
        if "text" not in doc:
            continue  # skip if no 'text' key
        vec = embed_fn(doc["text"])
        doc["embedding"] = vec.tolist()
        collection.insert_one(doc)

def get_all_documents():
    collection = get_mongo_collection()
    return list(collection.find({}))
