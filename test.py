import torch

# Example tensor with shape [1, 2, 768]
output1 = [torch.randn(1, 2, 768),torch.randn(1, 2, 768),torch.randn(1, 2, 768)]  # Simulating one tensor in a list

print("Before concatenation:")
print(f"Shape of each element in output1: {output1[0].shape}")

# Concatenate and reshape
output1 = torch.cat(output1, dim=0)
output1 = output1.view(output1.size(0), -1)
# .view(output1[0].size(0), -1)  # Concatenate along entity dimension and reshape

print("After concatenation and reshaping:")
print(f"Shape of output1: {output1.shape}")  # Should print torch.Size([3, 1536])
