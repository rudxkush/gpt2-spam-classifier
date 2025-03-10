#######################################################
# IMPLEMENTING SELF ATTENTION WITH TRAINABLE WEIGHTS  #
#######################################################
import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
# for illustration purposes.
d_in = inputs.shape[1] # row size = 3.
d_out = 2 
torch.manual_seed(123)
w_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False) # 3*2, weights will not be updated during training
w_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
w_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)

print(w_query)

x_2 = inputs[1] 
query_vector_journey = x_2 @ w_query 
print("query vector for journey", query_vector_journey) 
# we are in abstract world
query_vector = inputs @ w_query
print("Query: \n", query_vector)
key_vector = inputs @ w_key
print("Key: \n",key_vector)   
value_vector = inputs @ w_value
print("Value: \n",value_vector)

# finding how much importance should we paying to other key.
atten_score_journey = query_vector[1] @ key_vector.T
# encodes info about how much journey attends with our, journey, starts, with, one, step respectively.
print(atten_score_journey)
# similar to dot product
atten_score = query_vector @ key_vector.T  # this is basically the attention_scores
print("Attention scores: \n", atten_score)

# Normalisation: serves two purposes:
# 1st -> make things interpretable.
# 2nd -> helps in back propagation. 
#we also need to scale it two 6*2 matrix from 6*6.

d_key = key_vector.shape[-1] #6*2, -1 for columns => 2.
atten_weights_journey = torch.softmax(atten_score_journey / d_key**0.5, dim = -1) #d_key ^ 0.5
print("Normalised attention score for journey: \n", atten_weights_journey)


atten_weights = torch.softmax(atten_score / d_key**0.5, dim = -1) # Every element will first be divided by d_key ^ 0.5.
print("Normalised attention scores: \n", atten_weights)
# why divide by sqrt(dimension) -> to avoid peaky output, as softmax is sensitive to the magnitude of its input
# when inputs of large there, the difference values of each inputs becomes much more pronounced; giving highest value almost all
# the probability mass.
# why sqrt? -> to make the variance of dot product stable. dot product of q and k increases the variance to reduce that we 
# particularly use sqrt(dimension).

#val_vector is just the input vector kinda like before how we were multiplying the vector

context_vector = atten_weights @ value_vector
print("product of attention weights and value matrix: \n", context_vector)