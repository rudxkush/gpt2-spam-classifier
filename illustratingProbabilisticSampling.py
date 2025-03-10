# To illustrate the probabilistic sampling with a concrete example, 
# let's briefly discuss the next-token generation process using 
# a very small vocabulary for illustration purposes:

import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

vocab = {      # 9 tokens -> their respective token_id
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

inverse_vocab = {v: k for k, v in vocab.items()} # token_id -> decodes their respective 9 tokens  

# Next, assume the LLM is given the start context "every effort moves you" and
# generates the following next-token logits:

next_token_logits = torch.tensor(
[4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

next_token_logits2 = next_token_logits/0.1

next_token_logits3 = next_token_logits/5

probas = torch.softmax(next_token_logits2, dim=0)
print(probas)


probas = torch.softmax(next_token_logits3, dim=0)
print(probas)

probas = torch.softmax(next_token_logits, dim=0)

print(probas)

next_token_id = torch.argmax(probas).item()

print(next_token_id)

print(inverse_vocab[next_token_id])

torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token_id])

def print_sampled_tokens(probas):
    torch.manual_seed(123) # Manual seed for reproducibility
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

print_sampled_tokens(probas)

# As we can see based on the output, the word "forward" is sampled most of the time (582
# out of 1000 times), but other tokens such as "closer", "inches", and "toward" will also
# be sampled some of the time. 

# This means that if we replaced the argmax function with the
# multinomial function inside the generate_and_print_sample function, the LLM would
# sometimes generate texts such as "every effort moves you toward", "every effort
# moves you inches", and "every effort moves you closer" instead of "every effort
# moves you forward".

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

# Temperature values
temperatures = [1, 0.1, 5]  # Original, higher confidence, and lower confidence

# Calculate scaled probabilities
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]

##Multinomial
# Plotting
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
plt.savefig("temperature-plot.pdf")
plt.show()

# In the previous section, we implemented a probabilistic sampling approach coupled with
# temperature scaling to increase the diversity of the outputs. 

# We saw that higher
# temperature values result in more uniformly distributed next-token probabilities, which
# result in more diverse outputs as it reduces the likelihood of the model repeatedly selecting
# the most probable token. 

# This method allows for exploring less likely but potentially more
# interesting and creative paths in the generation process. 

# However, One downside of this
# approach is that it sometimes leads to grammatically incorrect or completely nonsensical
# outputs such as "every effort moves you pizza".

top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k) 
print("Top logits:", top_logits)
print("Top positions:", top_pos)

new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],
    input=torch.tensor(float("-inf")), 
    other=next_token_logits
)

print(new_logits)

topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)

# In top-k sampling, we can restrict the sampled tokens to the top-k most likely tokens
# and exclude all other tokens from the selection process by masking their probability scores.

# logits -> top-k -> logits/temperature -> softmax -> sample from multinomial.

