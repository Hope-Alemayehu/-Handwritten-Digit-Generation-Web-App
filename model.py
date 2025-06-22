import torch
import torch.nn as nn

class Generator(nn.Module):
    """Conditional GAN generator for MNIST digits (input: noise + digit label)."""
    def __init__(self):
        super(Generator, self).__init__()
        
        # Embedding for the digit label (0-9)
        self.label_embedding = nn.Embedding(num_embeddings=10, embedding_dim=10)
        
        # Neural network layers
        self.model = nn.Sequential(
            nn.Linear(in_features=110, out_features=256),  # 100 (noise) + 10 (label)
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=1024, out_features=28*28),  # MNIST image size
            nn.Tanh()  # Output pixels in [-1, 1] (will rescale to [0, 1] later)
        )

    def forward(self, noise, labels):
        # Concatenate noise and embedded labels
        label_embed = self.label_embedding(labels)
        x = torch.cat([noise, label_embed], dim=-1)  # Shape: (batch_size, 110)
        
        # Generate image
        img = self.model(x)
        return img.view(-1, 1, 28, 28)  # Reshape to (batch_size, 1, 28, 28)