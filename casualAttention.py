import torch
import torch.nn as nn
import torch.nn.functional as F

# Sample Inputs
inputs = torch.tensor(
  [[0.43, 0.15, 0.89],  # Your     (x^1)
   [0.55, 0.87, 0.66],  # journey  (x^2)
   [0.57, 0.85, 0.64],  # starts   (x^3)
   [0.22, 0.58, 0.33],  # with     (x^4)
   [0.77, 0.25, 0.10],  # one      (x^5)
   [0.05, 0.80, 0.55]]  # step     (x^6)
)

batch = torch.stack((inputs, inputs), dim=0)
# so the input is 2*(6*3) we multiply it with 2*(3*2) 
# random weights to get queries matrix of 2*(6*2) and 
# then we will multiply this with key matrix and to make 
# that happen we have two transpose the
# 2 columns to row and row to columns
# and get the attention scores
class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) 
        # creating a matrix of 6*6 that would upper 1's triangular matrix that we would use to get the required masked attention scores.
        # so wherever the matrix returns 1 we will set the value to be -inf, and rest stays the same(refer to PyTorch Masked_fill implementation).
        # what's the need of buffer? -> The use of register_buffer in PyTorch is not strictly necessary for all use cases but offers several advantages here. 
        # For instance, when we use the CausalAttention class in our LLM, buffers are automatically
        # moved to the appropriate device (CPU or GPU) along with our model, which will be relevant
        # when training the LLM in future chapters. This means we don't need to manually ensure
        # these tensors are on the same device as your model parameters, avoiding device mismatch errors.


    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        print("inputs: \n", x)
        keys = self.W_key(x)
        print(keys)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # Changed transpose
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) 
        # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        #  e^-(inf) = 0 so while masking these value won't factor.
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights) # randomly switches off neurons so that each neuron gets to participate -> better gerneralization.

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123) 
context_length = batch.shape[1]
d_in = inputs.shape[1] # column size = 6.
d_out = 3
# 2*(->6*3)
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)