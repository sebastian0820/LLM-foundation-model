import re
from Practise.SimpleTokenizerV2 import SimpleTokenizerV2
import torch
from Gpt.GPTModel import GPTModel
from GenTxt.generate_text_simple import generate_text_simple

with open("Practise/short story.txt", "r", encoding="utf-8") as f : raw_text = f.read()
preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_words = sorted(list(set(preprocessed)))
all_words.extend(["<|endoftext|>", "<|unk|>", "<|BOS|>", "<|EOS|>", "<|PAD|>"])
vocab_size = len(all_words)
vocab = {token:integer for integer , token in enumerate(all_words)}
tokenizer = SimpleTokenizerV2(vocab)

GPT_CONFIG_124M = {
    "vocab_size": vocab_size, # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}
torch.manual_seed(123)          #set randomization seed to make predictions reproducible
model = GPTModel(GPT_CONFIG_124M)


text = "I HAD always thought Jack Gisburn"
encoded = tokenizer.encode(text)
tensor = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
model.eval()
out = generate_text_simple(
    model=model,
    idx=tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)