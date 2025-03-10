import importlib.metadata
import re
import importlib
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class gptDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        #tokenise the entire text
        token_ids = tokenizer.encode(text, allowed_special = {"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i+1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self): #as we are using dataloader here
        return len(self.input_ids)
    
    def __getitem__(self, idx): #as we are using dataloader here
        return self.input_ids[idx], self.target_ids[idx]

#for implementing batch processing    
def create_dataloaderV1(text, batch_size = 4, max_length = 256, stride = 128, shuffle = True, drop_last = True, num_workers = 0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    #create dataset
    dataset = gptDatasetV1(text, tokenizer, max_length, stride)
    #create dataloader
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last, num_workers = num_workers)
    return dataloader

with open("/Users/rudxkush/LLM/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

#stride of 4 is a good scenario as our batches are not overlapping and also we do not miss a single word.(prevents from both overfitting and underfitting models)
dataloader = create_dataloaderV1(raw_text, batch_size = 8, max_length = 4, stride = 4, shuffle = False) 
data_iter = iter(dataloader) #we are going to iterate through the dataloader
inputs, targets = next(data_iter)
print(inputs) 
print("size of inputs (batch_size * #tokens(=max_length)): ", inputs.shape)

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim) #creates a random weighted 50257(tokens 0 -> 50256) * 256(features)
token_embeddings = token_embedding_layer(inputs) 
print("size of the token embedding; (4 tokens a row) in a batch(size = 8) with 256 features -> 8*(4*256): ", token_embeddings.shape)

#Now we will add position embedding to each of these token embeddings.
max_length = 4
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim) #Randomly initialized weights for 4 positions (0 to 3) with 256 features each.
pos_embeddings = pos_embedding_layer(torch.arange(max_length))    #4 vector(0, 1, 2, 3) each of size 256.
print("size of the position embedding: ",pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
