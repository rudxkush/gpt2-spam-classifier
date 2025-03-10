###################################
# SIMPLIFIED ATTENTION MECHANISM  #
###################################
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
print(inputs.shape)

# Corresponding words
words = ['Your', 'journey', 'starts', 'with', 'one', 'step']

# Extract x, y, z coordinates
x_coords = inputs[:, 0].numpy()
y_coords = inputs[:, 1].numpy()
z_coords = inputs[:, 2].numpy()

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each point and annotate with corresponding word
for x, y, z, word in zip(x_coords, y_coords, z_coords, words):
    ax.scatter(x, y, z)
    ax.text(x, y, z, word, fontsize=10)

# Set labels for axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.title('3D Plot of Word Embeddings')
plt.show()

# Create 3D plot with vectors from origin to each point, using different colors
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define a list of colors for the vectors
colors = ['r', 'g', 'b', 'c', 'm', 'y']

# Plot each vector with a different color and annotate with the corresponding word
for (x, y, z, word, color) in zip(x_coords, y_coords, z_coords, words, colors):
    # Draw vector from origin to the point (x, y, z) with specified color and smaller arrow length ratio
    ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.05)
    ax.text(x, y, z, word, fontsize=10, color=color)

# Set labels for axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set plot limits to keep arrows within the plot boundaries
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

plt.title('3D Plot of Word Embeddings with Colored Vectors')
plt.show()

query = inputs[1]  # 2nd input token is the query

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product (transpose not necessary here since they are 1-dim vectors)

print(attn_scores_2)

attention_scores = torch.empty(6,6) #empty tensor of 6*6

#for i, x_i in enumerate(inputs):   #enumerate indexes the row of input from 0 to 6. for 0, 0 in the indexing which is the first row.
#    for j, x_j in enumerate(inputs):
#        attention_scores[i, j] = torch.dot(x_i, x_j)

#to make the it much efficient we can use matrix multiplication
attention_scores = inputs @ inputs.T
print(attention_scores)


#so while finding similarity between two words we use Euclidean Distance between 
#two token embedding and if we wanna find how much they support 
#each other when taken together we take their dot product

#Lets normalise the these score in the range[0,1].
#def softmax_naive(x):
#    return torch.exp(x)/torch.exp(x).sum(dim = 0) #taking the exponent of each attention_scores_2 and dividing it by their summation.
#dim = 0, summing the entries in the row.

#attention_weights = softmax_naive(attention_scores_2)

#using pytorch impl. of softmax for optimisation over extensively large data.
attention_weights = torch.softmax(attention_scores, dim = -1)  #the dim function specifies the dimension of the input tensor along which the 
#function will be computed.
print(attention_weights)

context_vector = attention_weights @ inputs
print(context_vector)


