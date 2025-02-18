import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class SmoothTopKGate(nn.Module):
    def __init__(self, s_dim, k, tau=1e-2):
        super().__init__()
        self.s_dim = s_dim
        self.k = k
        self.tau = tau
    
    def forward(self, s):
        """
        s: Tensor of shape (batch_size, s_dim)
        """
        # Solve for threshold \theta using Newton's method
        theta = self._compute_threshold(s)
        
        # Smooth thresholding
        g = torch.sigmoid((s - theta) / self.tau)
        return g
    
    def _compute_threshold(self, s):
        """
        Compute \theta such that approximately k elements are active.
        Using an iterative approach.
        """
        batch_size = s.shape[0]
        theta = torch.zeros(batch_size, device=s.device)
        
        for _ in range(10):  # Newton-Raphson iterations
            sigma = torch.sigmoid((s - theta.unsqueeze(1)) / self.tau)
            sum_sigma = sigma.sum(dim=1)
            
            # Newton step update
            theta = theta + self.tau * (sum_sigma - self.k)
        
        return theta.unsqueeze(1)  # Shape (batch_size, 1)
    
class SoftTopKGate(nn.Module):
    def __init__(self, s_dim, k, tau=1e-2):
        super().__init__()
        self.s_dim = s_dim
        self.k = k
        self.tau = tau
    
    def forward(self, s):
        """
        s: Tensor of shape (batch_size, s_dim)
        """
        # Smooth thresholding
        g = torch.softmax(s/self.tau, dim=-1)
        return g

class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, s_dim, k, tau=1e-2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, s_dim)
        )
        self.gate = SoftTopKGate(s_dim, k, tau)
    
    def forward(self, x):
        s = self.mlp(x)
        g = self.gate(s)
        return g

# Example usage
N = 1000   # Input dimension
s = 8    # Output dimension
k = 2    # Number of active elements
hidden = 256  # Hidden layer size

# torch.manual_seed(0)
batch_size = 300
x_i = torch.randn(N)
x_f = torch.randn(N)
ts = torch.linspace(0, 1, batch_size)
x = torch.stack([x_i + t * (x_f - x_i)
                 for t in ts], dim=0)

model = TwoLayerMLP(N, hidden, s, k)
g = model(x)

print(x.shape)
print(g.shape)

for i in range(batch_size):
    g_sorted = g[i, torch.argsort(g[i])]
    print(f"Sample {i}: Top {k+2} values:", g_sorted[-k-2:])
    top_k_sum = g_sorted[-k-4:].sum()
    if top_k_sum > 0:  # Avoid division by zero
        print(f"Ratio: {g_sorted[-k-4]/top_k_sum:.5f}")

# # Plot histograms in a grid
# fig, axes = plt.subplots(4, 4, figsize=(20, 10))
# for i, ax in enumerate(axes.flat):
#     g_sorted = g[i, torch.argsort(g[i])]
#     print(g_sorted)
#     ax.hist(g[i].detach().numpy(), bins=20)
#     ax.set_title(f'Sample {i}')
# plt.tight_layout()
# plt.show()
plt.figure(figsize=(10, 7))
plt.imshow(g.detach(), aspect='auto', cmap='viridis')
plt.colorbar(label='Gate Value')
plt.xlabel('Feature')
plt.ylabel('Sample')
plt.title('Gate Values Across Samples')
plt.show()

print(g)

plt.show()

print(g.shape)  # Expected output: (32, s)
