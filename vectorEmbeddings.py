import gensim.downloader as api
import torch
# Pre-trained vectors trained with about 100 billion words. 300 -> dimensions.
model = api.load("word2vec-google-news-300")

word_vectors = model # Assign a dictionary to model(word-2-vector embeddings) 
# Every word in this dictionary will be assigned to a 300 dimensional vector.

print(word_vectors['computer'])
print(word_vectors['computer'].shape) # Every word is encoded into a 300 dimensional vector.

# Let us demonstrate that vector embedding actually captures the semantic meaning of the words(tokens).
print(word_vectors.most_similar(positive = ['king', 'women'], negative = ['man'], topn = 10))

print(word_vectors.similarity('women', 'men'))
print(word_vectors.similarity('paper', 'water'))  #not close to each other at all!

# We can also find the difference between their vector magnitude to demonstrate how much related they are. the closer they are the more they are related.

# Now, let's demonstrate how the token ID to vector conversion works.
input_ids = torch.tensor([2,3,5,1])
# Here we are using tensor because ultimately we are going to use back propagation to optimize the embedding layer weights.
vocab_size = 6 # For demonstration purposes, we are taking a very small vocabulary.
output_dim = 3 # We wanna create embedding of size 3(number of dimensions)
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(embedding_layer.weight) #random weights
# embedding_layer.weight stores the embedding matrix of shape (vocab_size, output_dim) -> updated during backpropagation.
# It is called a lookup matrix as it retrieves rows from the embedding matrix via token_id
print(embedding_layer.weight.data[3])  # Get embedding for token ID 3
# .data is just a Tensor that holds the underlying numerical values of the Parameter(weight).
# We will optimise these values by using back propagation(to produce a certain output the weights are manipulated) to get the required weights.

# weight (torch.nn.Parameter)
#  ├── data          (torch.Tensor)       -> The raw numbers stored in memory
#  ├── requires_grad (bool)               -> If True, PyTorch tracks gradients
#  ├── grad_fn       (function or None)   -> Tracks how it was created
#  ├── grad          (torch.Tensor)       -> Stores gradients after backprop

