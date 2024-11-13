import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):
        """
        :param max_len: Input length sequence.
        :param d_model: Embedding dimension.
        :param dropout: Dropout value (default=0.1)
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Adds the positional encoding to the input embeddings at the corresponding positions.
        Inputs of forward function
        :param x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        seq_len,
        embed_dim,
        output_dim,
        num_layers,
        num_heads,
        dropout=0.1,
    ):
        super(TransformerClassifier, self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(max_len=seq_len, d_model=embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers,
        )
        self.linear = nn.Linear(embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.emb(x)
        x = self.pos_encoder(emb)
        x = self.dropout(x)
        x = x.max(dim=1)[0]  # check this
        out = self.linear(x)
        return out
