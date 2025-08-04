from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name,low_cpu_mem_usage=True,trust_remote_code=True).eval()
    return tokenizer, model

# def compute_embedding(text, tokenizer, model):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)

#     if hasattr(outputs, "pooler_output"):
#         return outputs.pooler_output[0].numpy()
#     return outputs.last_hidden_state.mean(dim=1)[0].numpy()

def compute_embedding(text, tokenizer, model):

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tokenize input and move to correct device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512  # explicitly set max length
    ).to(device)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        # Use pooler_output if available (like in BERT-style models)
        return outputs.pooler_output[0].cpu().numpy()
    elif hasattr(outputs, "last_hidden_state"):
        # Apply mean pooling manually
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked_embeddings = last_hidden_state * attention_mask
        summed = torch.sum(masked_embeddings, dim=1)
        counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        mean_pooled = summed / counts
        return mean_pooled[0].cpu().numpy()
    else:
        raise ValueError("Model output does not contain `pooler_output` or `last_hidden_state`.")

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
