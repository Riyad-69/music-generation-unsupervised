import torch
import torch.nn as nn

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size=129, d_model=256, nhead=8, num_layers=4):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 128, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding[:, :x.size(1), :]
        x = x.permute(1, 0, 2)

        x = self.transformer(x)

        x = x.permute(1, 0, 2)
        return self.fc(x)