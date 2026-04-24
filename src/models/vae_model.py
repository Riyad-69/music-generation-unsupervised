import torch
import torch.nn as nn

LATENT_DIM = 128

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(128*128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(512, LATENT_DIM)
        self.fc_logvar = nn.Linear(512, LATENT_DIM)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128*128),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar



def loss_function(x, x_hat, mu, logvar, beta=0.001):
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction='mean')

    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss