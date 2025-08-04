# List of open-source embedding models

MODEL_REGISTRY = {
    # BGE Series (BAAI)
    "bge-base-en": "BAAI/bge-base-en",
    "bge-large-en": "BAAI/bge-large-en",
    "bge-m3": "BAAI/bge-m3",  # multilingual

    # E5 Series (intfloat)
    "e5-base-v2": "intfloat/e5-base-v2",
    "e5-large-v2": "intfloat/e5-large-v2",

    # GTE Series (Alibaba)
    "gte-base": "thenlper/gte-base",
    "gte-large": "thenlper/gte-large",
    "gte-multilingual-base": "Alibaba-NLP/gte-multilingual-base",

    # Nomic Embed (MoE)
    "nomic-v1": "nomic-ai/nomic-embed-text-v1",
    "nomic-v2": "nomic-ai/nomic-embed-text-v2-moe",

    # Sentence Transformers (SBERT)
    "sbert-mini": "sentence-transformers/all-MiniLM-L6-v2",
    "sbert-mpnet": "sentence-transformers/all-mpnet-base-v2",

    # MixedBread
    "mixedbread": "mixedbread-ai/mxbai-embed-large-v1",

    # Stella
    # "stella": "NovaSearch/stella_en_1.5B_v5"
}


# MODEL_REGISTRY = {
#     # BGE Series (BAAI)

#     "bge-m3": "BAAI/bge-m3"  # multilingual


# }