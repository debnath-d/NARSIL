import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_printoptions(precision=3, sci_mode=False)


def expand_mask(mask: torch.Tensor):
    assert mask.ndim >= 2, (
        "Mask must be at least 2-dimensional with seq_length x seq_length"
    )
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class MultiheadAttention(nn.Module):
    def __init__(self, q_dim: int, embed_dim: int, num_heads: int, kv_dim: int = None):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            "Embedding dimension must be 0 modulo number of heads."
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if isinstance(kv_dim, int):
            self.q_proj = nn.Linear(q_dim, embed_dim, bias=False)
            self.kv_proj = nn.Linear(kv_dim, 2 * embed_dim, bias=False)
        else:
            self.qkv_proj = nn.Linear(q_dim, 3 * embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, q_dim, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        try:
            nn.init.xavier_uniform_(self.qkv_proj.weight)
        except Exception:
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def forward(self, Q: torch.Tensor, mask: torch.Tensor, KV: torch.Tensor = None):
        batch_size, seq_length, _ = Q.size()
        if isinstance(mask, torch.Tensor):
            mask = expand_mask(mask)
        if isinstance(KV, torch.Tensor):  # Cross Attention
            q = self.q_proj(Q)
            q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
            q = q.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]

            kv = self.kv_proj(KV)
            kv = kv.reshape(batch_size, seq_length, self.num_heads, 2 * self.head_dim)
            kv = kv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
            k, v = kv.chunk(2, dim=-1)
        else:
            qkv = self.qkv_proj(Q)
            qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
            qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
            q, k, v = qkv.chunk(3, dim=-1)

        with nn.attention.sdpa_kernel(nn.attention.SDPBackend.EFFICIENT_ATTENTION):
            values = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0
            )
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        return o


class EncoderBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        num_heads,
        dim_feedforward,
        dropout=0.0,
    ):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(**block_args) for _ in range(num_layers)]
        )

    def forward(self, x, mask: torch.Tensor = None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        num_heads,
        dim_feedforward,
        dropout=0.0,
    ):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Self Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Cross attention layer
        self.cross_attn = MultiheadAttention(
            input_dim, input_dim, num_heads, kv_dim=input_dim
        )

        # MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, mask=None):
        # Self Attention part
        x = x + self.dropout(self.self_attn(x, mask=mask))
        x = self.norm1(x)

        # Cross attention part
        x = x + self.dropout(self.cross_attn(Q=x, KV=memory, mask=mask))
        x = self.norm2(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm3(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderBlock(**block_args) for _ in range(num_layers)]
        )

    def forward(self, x, memory, mask: torch.Tensor = None):
        for layer in self.layers:
            x = layer(x, memory=memory, mask=mask)
        return x


class NARSILModel(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_layers: int,
        input_dim: int,
        num_heads: int,
        feedforward_factor=4,
        dropout=0.0,
    ):
        super().__init__()
        embed_dim = num_heads * head_dim
        dim_feedforward = int(embed_dim * feedforward_factor)
        # self.proj_features = torch.nn.Linear(encoder_input_dim, embed_dim)
        self.proj_features = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, embed_dim),
        )

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            input_dim=embed_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.decoder = TransformerDecoder(
            num_layers=num_layers,
            input_dim=embed_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def calculate_logits(self, q: torch.Tensor, k: torch.Tensor = None) -> torch.Tensor:
        if not isinstance(k, torch.Tensor):
            k = q
        _, graph_size, embed_dim = q.size()
        logits = (q @ k.mT) / embed_dim**0.5

        logits.masked_fill_(
            mask=torch.eye(graph_size, dtype=torch.bool, device=logits.device),
            value=-torch.inf,
        )

        return logits

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        x = self.proj_features(x)

        encoded_nodes = self.encoder(x, mask=mask)
        decoded_nodes = self.decoder(encoded_nodes, memory=encoded_nodes, mask=mask)
        logits = self.calculate_logits(q=decoded_nodes, k=encoded_nodes)
        # logits = self.calculate_logits(q=encoded_nodes)
        # probs = self.calculate_probs(q=encoded_nodes)

        return logits.softmax(dim=-1)
