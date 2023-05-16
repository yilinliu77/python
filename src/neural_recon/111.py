import torch

# Assuming the following variables are defined:
# M: number of vertices
# N: number of edges
# S: number of times each edge is repeated with perturbations
# I: index tensor of shape (N, 2)
# total_loss: total loss tensor of shape (N, 2, S, 1)

# Define M, N, and S
M = 4  # Number of vertices
N = 6   # Number of edges
S = 4   # Number of times each edge is repeated with perturbations

# Generate the index tensor I with shape (N, 2)
I = torch.randint(0, M, (N, 2))

# Generate the total loss tensor with shape (N, 2, S, 1)
total_loss = torch.rand(N, 2, S, 1)

# Calculate the mean loss along the S dimension
mean_loss = torch.mean(total_loss, dim=2)  # Shape: (N, 2, 1)

# Expand the index tensor I along the last dimension to match mean_loss shape
expanded_I = I.unsqueeze(2)  # Shape: (N, 2, 1)

# Use scatter_add to accumulate mean_loss values for each vertex
O = torch.zeros(M, 1, device=expanded_I.device)
O.scatter_add_(0, expanded_I, mean_loss)

# Count the number of occurrences of each vertex in the edges
vertex_counts = torch.zeros(M, 1, device=expanded_I.device)
vertex_counts.scatter_add_(0, expanded_I, torch.ones_like(expanded_I))

# Calculate the mean loss value for each vertex by dividing accumulated losses by vertex_counts
O /= vertex_counts

print("I:", I)
print("total_loss:", total_loss)
print(O)  # Output tensor O with shape (M, 1)
