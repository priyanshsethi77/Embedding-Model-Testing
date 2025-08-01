import json
from config.model_list import MODEL_REGISTRY
from embeddings.embed_utils import load_model, compute_embedding
from mongo.mongo_utils import insert_documents
# from ocr.extract_text import extract_text_from_pdf

from search.sementic_search import search

# Load sample data
import fitz  # PyMuPDF
import os
import json

# Path to documents
doc_folder = "./pdf_documents"
output = []
doc_files = sorted([f for f in os.listdir(doc_folder) if f.endswith(".pdf")])

id_counter = 1  # start id from 1

for doc_file in doc_files:
    doc_path = os.path.join(doc_folder, doc_file)
    doc = fitz.open(doc_path)

    text_accumulator = []

    for page in doc:
        text = page.get_text()
        if text.strip():  # skip empty pages
            text_accumulator.append(text.strip())

    full_text = " ".join(text_accumulator).replace("\n", " ")

    output.append({
        "id": id_counter,
        "text": full_text
    })
    id_counter += 1

# Save to JSON file

with open("output_documents.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

query = "What are qubits and how do they function?"

# Test each model
for key, model_name in MODEL_REGISTRY.items():
    print(f"\n--- Testing Model: {key} ({model_name}) ---")
    tokenizer, model = load_model(model_name)

    # Embedding function
    embed_fn = lambda text: compute_embedding(text, tokenizer, model)

    # Insert documents
    insert_documents(output, embed_fn)

    # Search
    results = search(query, embed_fn)
    for text, score in results:
        print(f"{score:.4f} | {text}")

# import fitz  # PyMuPDF
# import os
# import json

# # Path to documents
# doc_folder = "./pdf_documents"
# output = []
# doc_files = sorted([f for f in os.listdir(doc_folder) if f.endswith(".pdf")])

# id_counter = 1  # start id from 1

# for doc_file in doc_files:
#     doc_path = os.path.join(doc_folder, doc_file)
#     doc = fitz.open(doc_path)

#     text_accumulator = []

#     for page in doc:
#         text = page.get_text()
#         if text.strip():  # skip empty pages
#             text_accumulator.append(text.strip())

#     full_text = " ".join(text_accumulator).replace("\n", " ")

#     output.append({
#         "id": id_counter,
#         "text": full_text
#     })
#     id_counter += 1

# # Save to JSON file
# with open("output_documents.json", "w", encoding="utf-8") as f:
#     json.dump(output, f, ensure_ascii=False, indent=2)

# print("✅ Extraction complete. Data saved in 'output_documents.json'")

# import os
# import json
# from config.model_list import MODEL_REGISTRY
# from embeddings.embed_utils import load_model, compute_embedding
# from mongo.mongo_utils import insert_documents
# from ocr.extract_text import extract_text_from_pdf
# from search.sementic_search import search

# # Define directory for PDFs
# DOCUMENT_DIR = "documents/"
# QUERY = "What are qubits and how do they function?"

# # Iterate over each model
# for key, model_name in MODEL_REGISTRY.items():
#     print(f"\n--- Testing Model: {key} ({model_name}) ---")

#     # Load tokenizer & model
#     tokenizer, model = load_model(model_name)

#     # Embedding function
#     embed_fn = lambda text: compute_embedding(text, tokenizer, model)

#     # Process each PDF
#     for file in os.listdir(DOCUMENT_DIR):
#         if file.endswith(".pdf"):
#             file_path = os.path.join(DOCUMENT_DIR, file)
#             print(f"\nProcessing {file}...")

#             # Extract pages
#             pages = extract_text_from_pdf(file_path)

#             # Format as list of dicts for insert
#             page_docs = [
#                 {"doc_id": file, "page": idx + 1, "text": pg}
#                 for idx, pg in enumerate(pages)
#             ]

#             # Insert documents into MongoDB with embeddings
#             insert_documents(page_docs, embed_fn)

#     # Run semantic search on the inserted data
#     print(f"\nSemantic Search Result for Query: '{QUERY}'")
#     results = search(QUERY, embed_fn)

#     for text, score in results:
#         print(f"{score:.4f} | {text[:80]}...")
