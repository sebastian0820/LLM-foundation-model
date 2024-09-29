import sys
import os
# Add the parent directory of the current file to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import re
import torch
import tiktoken
from Gpt.GPTModel import GPTModel
from GenTxt.generate_text_simple import generate_text_simple
from Practise.GPTDatasetV1 import GPTDatasetV1
from torch.utils.data import Dataset, DataLoader

GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 256, # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}
torch.manual_seed(123)          #set randomization seed to make predictions reproducible
model = GPTModel(GPT_CONFIG_124M)
model.eval()

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

# tokenizer = tiktoken.get_encoding("gpt2")
# start_context = "Every effort moves you"
# token_ids = generate_text_simple(
#     model=model,
#     idx=text_to_token_ids(start_context, tokenizer),
#     max_new_tokens=10,
#     context_size=GPT_CONFIG_124M["context_length"]
# )
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
#                        [40, 1107, 588]]) # "I really like"]
# targets = torch.tensor([[3626, 6100, 345 ], # [" effort moves you",
#                         [588, 428, 11311]]) # " really like chocolate"]
# with torch.no_grad():logits = model(inputs)

# #show result of evaluation

# probas = torch.softmax(logits, dim=-1) # Probability of each token in vocabulary
# print(probas.shape)                                                                   #print tokens shape
# token_ids = torch.argmax(probas, dim=-1, keepdim=True)
# print("Token IDs:\n", token_ids)                                                    #print tokens shape
# print()

# #manual calculation of loss

# print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")               #print result for first token
# print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
# print()
# print(f"Targets batch 2: {token_ids_to_text(targets[1], tokenizer)}")               #print result for second token
# print(f"Outputs batch 2: {token_ids_to_text(token_ids[1].flatten(), tokenizer)}")
# print()
# text_idx = 0                                                                        #probability array for each token
# target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
# print("Text 1:", target_probas_1)
# print()
# text_idx = 1
# target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
# print("Text 2:", target_probas_2)
# print()
# log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
# print(log_probas)
# print()
# avg_log_probas = torch.mean(log_probas)
# print(avg_log_probas)
# print()
# neg_avg_log_probas = avg_log_probas * -1
# print(neg_avg_log_probas)
# print()

# logits_flat = logits.flatten(0, 1)                                      # python library based calculation of loss
# targets_flat = targets.flatten()
# print("Flattened logits:", logits_flat.shape)
# print("Flattened targets:", targets_flat.shape)
# print()
# loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
# print(loss)
# perplexity = torch.exp(loss)
# print(perplexity)                                                       # python library based calculation of loss
# print()

vocab = {
"closer": 0,
"every": 1,
"effort": 2,
"forward": 3,
"inches": 4,
"moves": 5,
"pizza": 6,
"toward": 7,
"you": 8,
}
inverse_vocab = {v: k for k, v in vocab.items()}
next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])
probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
print(inverse_vocab[next_token_id])

torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token_id])

print_sampled_tokens(probas)

temperatures = [1, 0.1, 5] # Original, higher, and lower temperature
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
    rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')
ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
plt.show()