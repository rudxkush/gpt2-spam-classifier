##########################################################################################
# GPT placeholder architecture -> implementing a gpt model from scratch to generate text #
##########################################################################################
import torch
import torch.nn as nn
import tiktoken

GPT_CONFIG_124M = {  # dictionary 
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # nn.Embedding is optimise for intialising weights
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) 
        # nn.embedding in torch-> it creates a token embedding matrix-> 50257*768(random values-> later using backpropagation we are gonna train these) and for each input token id we get the 768 dimension -> look up table
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]) # cfg -> from the dictionary
        # positional embedding layer that maps position indices (i.e., the token’s position in the sequence) to dense vectors (similar to token embeddings). -> 1024*768
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Use a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        # Use a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx): # It takes in an input(vocab_size(=number of tokens) * emb_dim(=number of features)) -> tokens and at the end of it is going to printout the output -> next word.
        batch_size, seq_len = in_idx.shape 
        # number of rows of the same sentence and sequence_len is the number of token we are considering in each row.->(2*4)
        tok_embeds = self.tok_emb(in_idx)  
        # we are going go get the 768 dimension for each input token id from the look up table and gonna form a token embedding weight matrix. ->(2*(4*768))
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device)) # for each token (assigned a row from 0 to 3) and with their respective token id their corresponding positional vectors are retrieved.
        # For each input token, we are adding the position embeddings (which encode the token’s position in the sequence) to the token embeddings (which encode the semantic meaning of the token). 
        # This helps the model learn not only the meaning of each token but also its position in the sequence.->(2*(4*768))
        x = tok_embeds + pos_embeds # The resulting vector incorporates both semantic and positional information. -> (2*(4*768))
        x = self.drop_emb(x) # for generlisation purpose, we are gonna switch on and off some weight values. prevents overfitting :/
        # trf_blocks(x) -> passing your input through a Transformer block 
        # where it gets contextualized (through self-attention)-> context_vector,
        # and then further processed by a feedforward neural network 
        # to make predictions.
        x = self.trf_blocks(x) # 12 of these in gpt 2(small).
        # output matrix would be rows = no of i/p tokens & each of these token will have 768 dimensional vector(representation).
        # output matrix -> final output matrix?
        # 
        # final output matrix -> rows = no of i/p tokens, columns = vocabulary size(50257)
        # for token 1 -> Every ,output -> we will choose the word with the highest probability in the vocabulary.
        # for token 2 -> Every effort, output -> moves 
        # for token 3 -> Every effort moves, output -> you
        # for token 4 -> Every effort moves you ,output -> forward.
        x = self.final_norm(x) # normalisation layer -> output matrix.
        # output is called logits.
        logits = self.out_head(x) # imp. step-> final output matrix
        return logits


class DummyTransformerBlock(nn.Module):  
    def __init__(self, cfg):
        super().__init__()
        # A simple placeholder

    def forward(self, x):
        # This block does nothing and just returns its input.
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # The parameters here are just to mimic the LayerNorm interface.

    def forward(self, x):
        # This layer does nothing and just returns its input.
        return x

# I. Tokenisation  
tokenizer = tiktoken.get_encoding("gpt2") # byte pair encoder for tokenisation
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1))) # txt1 -> token_ids 
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

# II. creating an instance of dummy gpt model 
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M) # creating an instance of the class dummyGPTModel; 
# Explicitly passing the dictionary `GPT_CONFIG_124M` as `cfg`, which contains model hyperparameters  
logits = model(batch)  
# Passing the batch of tokenized input sequences to the model  
# logits = model(batch) 
print("Output shape:", logits.shape)
print(logits)

# Training weights -> incorporating which words to pay more attention to.
# so at some point we have generalised the context vector 
# that is by passing the sub space of features  of the same input 
# to multi head attention transformer and this is done 
# so that we get the weight matrices-> key, query, value 
# to work on any kind of input data.
# so that is just from one transformer block and 
# we pass the same input in gpt- 2(small) with 12 transformer 
# all producing different generalised matrix that would again 
# help us average those weights