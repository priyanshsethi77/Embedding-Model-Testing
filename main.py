import json
import os
import fitz  # PyMuPDF

from config.model_list import MODEL_REGISTRY
from embeddings.embed_utils import load_model, compute_embedding
from mongo.mongo_utils import insert_documents
from ocr.extract_text import extract_text_from_pdf
from search.sementic_search import search
import time

# === Setup folders ===
doc_folder = "./pdf_documents"
results_folder = "./results"
img_output_folder = "./extracted_images"
os.makedirs(img_output_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)

# === Step 1: Load and preprocess documents ===
output = []
doc_files = sorted([f for f in os.listdir(doc_folder) if f.endswith(".pdf")])
id_counter = 1

for doc_file in doc_files:
    doc_path = os.path.join(doc_folder, doc_file)
    doc = fitz.open(doc_path)

    text_accumulator = []
    images = []
    tables = []  # Placeholder

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            text_accumulator.append(text.strip())

        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"{os.path.splitext(doc_file)[0]}_page{page_num}_img{img_index + 1}.{image_ext}"
            image_path = os.path.join(img_output_folder, image_filename)

            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

            images.append(image_filename)

    full_text = " ".join(text_accumulator).replace("\n", " ")
    output.append({
        "id": id_counter,
        "file_name": doc_file,
        "text": full_text,
        "tables": tables,
        "images": images
    })
    id_counter += 1

# Save preprocessed documents
with open("sample_documents.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# === Step 2: Define Test Queries ===
queries = [
    "Which 19th-century metallic structure faced artistic backlash during its initial years despite now being celebrated globally?",
    "What is the widely accepted identity of the person depicted, and what is the reason behind the alternative name given to the portrait?",
    "Which components in the system are responsible for semantic segmentation of document content, and how are their outputs standardized for cross-model compatibility?",
    "What was Emperor Qin Shi Huang’s strategy for defending a unified territory against outside aggression?",
    "What open-source tool allows developers to implement numerical simulations and scientific analysis with extensive third-party package support?",
    "Which Renaissance technique contributed to the subtle transitions and emotional ambiguity in the subject’s face?",
    "What methods or models are employed to detect non-linear content flows, such as tables or multi-column layouts, in complex page formats?",
    "How do novel physical properties in subatomic systems enable massively parallel problem solving in next-generation processors?",
    "What towering creation was initially built to commemorate a major revolution’s centennial through a world exhibition?",
    "Which beginner-friendly coding language has gained traction in educational curricula due to its human-readable syntax?",
    "Which ancient structure in East Asia relied on smoke and fire to transmit military messages over long distances?",
    "What scientific phenomenon allows information units to remain linked across long distances, enhancing computational accuracy?"
]

# === Step 3: Run all models against all queries ===
def truncate_text(text, num_words=15):
    words = str(text).strip().split()
    return " ".join(words[:num_words]) + " ..." if len(words) > num_words else text

for key, model_name in MODEL_REGISTRY.items():
    print(f"\n=== Testing Model: {key} ({model_name}) ===")
    # Start time tracking
    start_time = time.time()
    tokenizer, model = load_model(model_name)
    embed_fn = lambda text: compute_embedding(text, tokenizer, model)

    # Refresh DB with latest embeddings
    insert_documents(output, embed_fn)

    result_file = os.path.join(results_folder, f"{key}_results.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"Model: {key} ({model_name})\n\n")

        for i, query in enumerate(queries, start=1):
            results = search(query, embed_fn)

            if not results:
                continue

            # Extract top result (text, score)
            top_text, top_score = results[0]

            # Find corresponding document file name
            file_match = next(
                (doc['file_name'] for doc in output if str(top_text).strip() in doc['text']), 
                "unknown"
            )

            # Truncate the matched text to 15 words
            short_text = truncate_text(top_text, num_words=15)

            # Print to console
            print(f"\nQuery {i}: {query}")
            print(f"{top_score:.4f} | {file_match} | {short_text}")

            # Write to result file
            f.write(f"Query {i}: {query}\n")
            f.write(f"{top_score:.4f} | {file_match} | {short_text}\n\n")
        # End time tracking
        end_time = time.time()
        elapsed_time = end_time - start_time
        f.write(f"\n⏱ Total Time Taken: {elapsed_time:.2f} seconds\n")
    print(f"✅ Results saved to {result_file} in {elapsed_time:.2f} seconds.")

# import json
# import os
# import fitz  # PyMuPDF

# from config.model_list import MODEL_REGISTRY
# from embeddings.embed_utils import load_model, compute_embedding
# from mongo.mongo_utils import insert_documents
# from search.sementic_search import search

# # --- Folder Setup ---
# doc_folder = "./pdf_documents"
# img_output_folder = "./extracted_images"
# results_folder = "./results"
# os.makedirs(img_output_folder, exist_ok=True)
# os.makedirs(results_folder, exist_ok=True)

# # --- Document Extraction ---
# output = []
# doc_files = sorted([f for f in os.listdir(doc_folder) if f.endswith(".pdf")])
# id_counter = 1

# for doc_file in doc_files:
#     doc_path = os.path.join(doc_folder, doc_file)
#     doc = fitz.open(doc_path)

#     text_accumulator = []
#     images = []
#     tables = []  # Placeholder for future table extraction

#     for page_num, page in enumerate(doc, start=1):
#         # Extract text
#         text = page.get_text()
#         if text.strip():
#             text_accumulator.append(text.strip())

#         # Extract images
#         for img_index, img in enumerate(page.get_images(full=True)):
#             xref = img[0]
#             base_image = doc.extract_image(xref)
#             image_bytes = base_image["image"]
#             image_ext = base_image["ext"]
#             image_filename = f"{os.path.splitext(doc_file)[0]}_page{page_num}_img{img_index + 1}.{image_ext}"
#             image_path = os.path.join(img_output_folder, image_filename)

#             with open(image_path, "wb") as img_file:
#                 img_file.write(image_bytes)

#             images.append(image_filename)

#     full_text = " ".join(text_accumulator).replace("\n", " ")

#     output.append({
#         "id": id_counter,
#         "file_name": doc_file,
#         "text": full_text,
#         "tables": tables,
#         "images": images
#     })
#     id_counter += 1

# # Save structured output to JSON
# with open("output_documents.json", "w", encoding="utf-8") as f:
#     json.dump(output, f, ensure_ascii=False, indent=2)

# print("✅ Document extraction complete.")

# # --- Semantic Search Evaluation ---
# query = "What are qubits and how do they function?"

# for key, model_name in MODEL_REGISTRY.items():
#     print(f"\n--- Testing Model: {key} ({model_name}) ---")
#     tokenizer, model = load_model(model_name)
#     embed_fn = lambda text: compute_embedding(text, tokenizer, model)

#     # Insert into MongoDB
#     insert_documents(output, embed_fn)

#     # Run semantic search
#     results = search(query, embed_fn)

#     # Save results to file
#     result_path = os.path.join(results_folder, f"{key}_results.txt")
#     with open(result_path, "w", encoding="utf-8") as f:
#         f.write(f"Semantic Search Results for Model: {key} ({model_name})\n")
#         f.write(f"Query: {query}\n\n")
#         for text_chunk, score in results:
#             f.write(f"{score:.4f} | {text_chunk[:200].replace('\n', ' ')}\n")

#     print(f"📄 Results saved to {result_path}")


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


