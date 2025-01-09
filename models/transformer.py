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

        # Create learnable [CLS] token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embed_dim)
        )  # Shape: [1, 1, embed_dim]

    def forward(self, x):
        # Step 1: Get embeddings of input tokens
        emb = self.emb(x)  # Shape: [batch_size, seq_len, embed_dim]

        # Step 2: Add positional encoding (if needed)
        x = self.pos_encoder(emb)  # Shape: [batch_size, seq_len, embed_dim]

        # Step 3: Add the [CLS] token at the beginning of the sequence
        cls_token_expanded = self.cls_token.expand(
            x.size(0), -1, -1
        )  # Expand [1, 1, embed_dim] to [batch_size, 1, embed_dim]
        x_with_cls = torch.cat(
            (cls_token_expanded, x), dim=1
        )  # Concatenate the [CLS] token to the input sequence

        # Step 4: Apply Transformer Encoder
        x = self.encoder(x_with_cls)  # Shape: [batch_size, seq_len + 1, embed_dim]

        # Step 5: Take the embedding of the [CLS] token (first token)
        cls_embedding = x[:, 0, :]  # Shape: [batch_size, embed_dim]

        # Step 6: Apply dropout
        cls_embedding = self.dropout(cls_embedding)

        # Step 7: Pass the [CLS] token embedding through the final output layer
        out = self.linear(cls_embedding)  # Shape: [batch_size, output_dim]

        return out
