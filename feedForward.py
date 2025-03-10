import torch
import torch.nn as nn
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

gelu, relu = GELU(), nn.ReLU()

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), ## Expansion -> Expands the dimensionality (typically from emd_dim to 4*emd_dim). 
            GELU(), ## Activation -> Gaussian Error Linear Unit is applied element-wise to introduce non-linearity.
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), ## Contraction -> Projects back to the original model size.
        )

    def forward(self, x):
        return self.layers(x)
    
ffn = FeedForward(GPT_CONFIG_124M) # creating an instance ff class
x = torch.rand(2, 3, 768)  # input -> x
out = ffn(x) # passing the input to the instance.
print(out.shape) # printing the output shape -> same as input.


# Suppose you’re designing a house, and you have 3 basic features: Length, Width, Height

# Now, imagine you want to extract more meaningful features before finalizing your design. Instead of just using these 3 raw values, you temporarily expand them into 12 features that capture more nuanced details, like:
# Floor Area (Length × Width)
# Volume (Length × Width × Height)
# Wall Surface Area
# Number of rooms
# Ceiling height category
# Natural light index
# ...(other complex attributes)

# Now, after computing these 12 richer features, you realize that you don’t need all of them—some may be redundant. 
# So, you apply transformations (like non-linearity and learned weights) and finally compress them back into the original 3D space, 
# but now with better and more informative values.

# this gets applied on weights*input_val in order to express their more features together