import tiktoken
import torch
from Gpt.GPTModel import GPTModel
from GenTxt.generate_text_simple import generate_text_simple

tokenizer = tiktoken.get_encoding("gpt2")
GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}
torch.manual_seed(123)          #set randomization seed to make predictions reproducible
model = GPTModel(GPT_CONFIG_124M)

start_context = "Every effort moves you"
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
model.eval() #A
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)

