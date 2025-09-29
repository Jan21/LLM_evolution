import torch
import torch.nn as nn
import math




class PositionalEncoding_v2(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 128):
        super().__init__()
        # Evolutionary improvements:
        # - Combine sinusoidal base (stable, extrapolatable) with a small learnable
        #   positional embedding to let the model correct/adjust positions.
        # - Provide a learnable scaling factor for positional contribution.
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Sinusoidal positional encoding (fixed)
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # When odd, last column remains zero for cos
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe_fixed", pe)  # (max_seq_length, d_model)

        # Small learnable positional corrections
        self.pe_learned = nn.Parameter(torch.zeros(max_seq_length, d_model))
        nn.init.normal_(self.pe_learned, mean=0.0, std=0.02)

        # Learnable scale between token embeddings and positional signal
        self.pos_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected shape: (seq_len, batch_size, d_model)
        seq_len = x.size(0)
        if seq_len > self.max_seq_length:
            # If sequence longer than buffer, interpolate learned positions
            # to fit the new length (simple linear interpolation)
            pe_fixed = self.pe_fixed
            pe_fixed = torch.nn.functional.interpolate(
                pe_fixed.transpose(0, 1).unsqueeze(0),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).squeeze(0).transpose(0, 1)
            pe_learned = torch.nn.functional.interpolate(
                self.pe_learned.transpose(0, 1).unsqueeze(0),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).squeeze(0).transpose(0, 1)
        else:
            pe_fixed = self.pe_fixed[:seq_len, :]  # (seq_len, d_model)
            pe_learned = self.pe_learned[:seq_len, :]

        # Combine fixed + learned, apply learnable scale, and broadcast to batch
        pe_combined = (pe_fixed + pe_learned) * self.pos_scale  # (seq_len, d_model)
        pe_combined = pe_combined.unsqueeze(1)  # (seq_len, 1, d_model)
        return x + pe_combined.to(x.dtype).to(x.device)


class GeneratedTransformerModel_v2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        print("Using GeneratedTransformerModel_v2")  # indicate this evolved model is used

        # Evolutionary improvements summary (in-code comments below):
        # - Use GELU activations (smoother than ReLU).
        # - Use pre-norm Transformer (norm_first=True) to improve training stability for deeper stacks.
        # - Combine fixed sinusoidal + small learned positional embeddings (see PositionalEncoding_v2).
        # - Introduce a small learned embedding scale instead of fixed sqrt(d_model) to allow adaptive rescaling.
        # - Tie input embeddings and output projection weights to improve generalization.
        # - Use Xavier initialization for linear layers and small normal init for embeddings.
        # - Add final LayerNorm before output projection (pre-norm head).
        # - Keep batch_first=True for compatibility with existing training code.
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Learned embedding scale (initialized to sqrt(d_model) like common practice, but learnable)
        self.embed_scale = nn.Parameter(torch.tensor(math.sqrt(d_model), dtype=torch.float))

        # Positional encoding v2: combines sinusoidal + small learned corrections
        self.pos_encoding = PositionalEncoding_v2(d_model, max_seq_length)

        # Transformer encoder layer: use GELU, pre-norm (norm_first=True), and batch_first for compatibility
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # pre-norm improves training stability for deeper models
        )

        # Transformer encoder stack
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Final layer norm (pre-head) to stabilize logits
        self.final_layer_norm = nn.LayerNorm(d_model, eps=1e-5)

        # Output projection; bias kept to allow flexibility, will tie weights after init
        self.output_projection = nn.Linear(d_model, vocab_size, bias=True)

        # Dropout layers for regularization
        self.embedding_dropout = nn.Dropout(dropout * 0.5)  # slightly milder dropout on embeddings
        self.output_dropout = nn.Dropout(dropout)

        # Initialize weights with improved strategies
        self._init_weights()

        # Tie embeddings and output projection weights to improve generalization and reduce params
        # (embedding.weight shape: [vocab_size, d_model]; output_projection.weight shape: [vocab_size, d_model])
        # Assign the same Parameter object so they share gradients/weights
        self.output_projection.weight = self.embedding.weight

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform is a good default for linear layers with GELU
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Keep embeddings small and centered
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch_size, seq_len)
        # attention_mask: (batch_size, seq_len), 1 for real tokens, 0 for padding

        # Embed tokens and apply learned scale
        x = self.embedding(input_ids) * self.embed_scale  # (batch_size, seq_len, d_model)

        # Apply positional encoding. PositionalEncoding_v2 expects (seq_len, batch, d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)  # back to (batch, seq_len, d_model)
        x = self.embedding_dropout(x)

        # Create causal mask for autoregressive generation (upper triangular)
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device), diagonal=1)

        # Convert attention_mask to key_padding_mask expected by transformer:
        # key_padding_mask: (batch_size, seq_len) with True for positions that should be masked (padding)
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        # Transformer expects src (batch_first=True) shape: (batch_size, seq_len, d_model)
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask
        )

        # Final normalization and dropout before projection (pre-head norm)
        x = self.final_layer_norm(x)
        x = self.output_dropout(x)

        # Project to vocabulary. Because weights are tied, this shares parameters with input embedding.
        logits = self.output_projection(x)  # (batch_size, seq_len, vocab_size)
        return logits


class PositionalEncoding_v3(nn.Module):
    """
    Sinusoidal positional encoding with small learnable scaling parameter.
    Improvements:
    - Uses classic sinusoidal encoding (no extra params) but adds a learnable
      scale to allow the model to adjust contribution of positional signals.
    - Supports inputs in (seq_len, batch, d_model) format to match Transformer API.
    """
    def __init__(self, d_model: int, max_seq_length: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # handle odd d_model: last dim uses cos on a zero-padded term
            pe[:, 1::2] = torch.cos(position * div_term)[:,:pe[:,1::2].shape[1]]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_seq_length, 1, d_model)
        self.register_buffer('pe', pe, persistent=False)
        # learnable scale for positional enc, initialized near 1.0
        self.scale = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, d_model)
        seq_len = x.size(0)
        if seq_len > self.max_seq_length:
            # fallback: recompute on the fly if exceeding precomputed max
            pe = torch.zeros(seq_len, self.d_model, device=x.device)
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * (-math.log(10000.0) / self.d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(1)
        else:
            pe = self.pe[:seq_len].to(x.device)
        x = x + self.scale * pe
        return self.dropout(x)

class GeneratedTransformerModel_v3(nn.Module):
    """
    Evolution iteration 3 of the Transformer model.
    Improvements introduced:
    - GELU activation (better smoothness than ReLU)
    - Pre-norm architecture via norm_first=True for improved training stability in deep stacks
    - Xavier (Glorot) uniform initialization for linear layers and embeddings for better signal propagation
    - LayerNorm applied after encoder stack to stabilize final representations
    - Embedding LayerNorm and dropout to regularize token embeddings
    - Weight tying between input embedding and output projection to reduce parameters and improve generalization
    - Sinusoidal positional encoding with a learnable scaling factor (PositionalEncoding_v3)
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embedding with small init scale
        self.embedding = nn.Embedding(vocab_size, d_model)
        # embedding layer normalization (pre-encoder)
        self.embedding_layer_norm = nn.LayerNorm(d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Positional encoding module (v3)
        self.pos_encoding = PositionalEncoding_v3(d_model, max_seq_length, dropout=dropout)

        # Build a stack of TransformerEncoderLayer with pre-norm and GELU activation
        # norm_first=True implements pre-norm which often improves training stability
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ) for _ in range(num_layers)
        ])

        # Final layer norm after the encoder stack (stabilizes outputs)
        self.final_layer_norm = nn.LayerNorm(d_model)

        # Output projection tied to embedding weights (weight tying)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=True)
        # Tie weights: output_projection.weight shape (vocab_size, d_model) matches embedding.weight
        self.output_projection.weight = self.embedding.weight

        # Additional dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights with Xavier for better conditioning
        self._init_weights()

        print(f"GeneratedTransformerModel_v3 initialized with d_model={d_model}, num_layers={num_layers}, num_heads={num_heads}")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Xavier uniform for embeddings
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm weights to 1 and bias to 0
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch_size, seq_len)
        # attention_mask: (batch_size, seq_len) with 1 for real tokens and 0 for padding

        # Embedding + scaling
        x = self.embedding(input_ids) * math.sqrt(self.d_model)  # (batch, seq, d_model)
        x = self.embedding_layer_norm(x)
        # Transformer expects (seq_len, batch, d_model) for positional encoding here
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.emb_dropout(x)

        # Create causal mask for autoregressive generation
        seq_len = input_ids.size(1)
        # Using float mask with -inf for masked positions to be compatible with Transformer API
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        causal_mask = causal_mask.to(input_ids.device)

        # Convert attention_mask to the format expected by transformer encoder layers
        if attention_mask is not None:
            # attention_mask: 1 for real tokens, 0 for padding
            # src_key_padding_mask expects True for positions that should be masked (padding)
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        # Pass through each encoder layer sequentially
        for layer in self.layers:
            # Each layer handles its own residual connections; using pre-norm improves stability
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=key_padding_mask)

        # Final normalization for stable outputs
        x = self.final_layer_norm(x)
        x = self.dropout(x)

        logits = self.output_projection(x)
        return logits


class PositionalEncoding_v4(nn.Module):
    """
    Evolution v4 positional encoding.
    - Sinusoidal base (deterministic, good inductive bias)
    - Learnable global scaling parameter to allow the model to adapt positional magnitude
    - Dropout for regularization
    Interface: same as previous positional encoding versions
    """
    def __init__(self, d_model: int, max_seq_length: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.dropout = nn.Dropout(dropout)

        # Create sinusoidal positional encodings (max_seq_length x d_model)
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # If odd, handle last col for cos
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer (not a parameter)
        self.register_buffer("pe", pe)  # shape (max_seq_length, d_model)

        # Learnable scaling factor for positional encodings
        self.scale = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, d_model)
        seq_len = x.size(0)
        if seq_len > self.max_seq_length:
            # If longer sequence encountered, expand positional encodings on the fly
            pe = self._extend_pe(seq_len)
        else:
            pe = self.pe[:seq_len, :]

        # pe: (seq_len, d_model) -> (seq_len, 1, d_model) to broadcast across batch
        pe = pe.unsqueeze(1).to(x.dtype).to(x.device)
        x = x + self.scale * pe
        return self.dropout(x)

    def _extend_pe(self, seq_len: int) -> torch.Tensor:
        # Create extended pe when seq_len exceeds precomputed max
        pe = torch.zeros(seq_len, self.d_model, device=self.pe.device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=self.pe.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=self.pe.device).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if self.d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe

class GeneratedTransformerModel_v4(nn.Module):
    """
    Evolution iteration 4 of the Transformer model.
    Improvements:
    - Custom pre-norm transformer stack built from primitives (MultiheadAttention + FFN)
      to allow insertion of LayerScale and stochastic depth (DropPath) per layer.
    - SiLU (Swish) activation for smoother gradients and better empirical performance.
    - LayerScale (small learnable scaling) on both attention and FFN residuals to stabilize deep networks.
    - Stochastic depth (DropPath) regularization across encoder layers to reduce overfitting.
    - Xavier initialization for linear layers and embeddings; dedicated init for MHA parameters.
    - PositionalEncoding_v4 (sinusoidal + learnable scaling).
    - Weight tying between input embedding and output projection.
    - Pre-norm architecture for better training stability.
    Interface preserved exactly to remain compatible with existing training code.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # Token embedding and light regularization
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_layer_norm = nn.LayerNorm(d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Positional encoding (v4)
        self.pos_encoding = PositionalEncoding_v4(d_model, max_seq_length, dropout=dropout)

        # Build custom encoder layers using primitives so we can add LayerScale & DropPath
        self.layers = nn.ModuleList()
        # Linearly increase drop path probability across layers (0 -> max_drop_path)
        max_drop_path = 0.1  # modest stochastic depth for regularization
        for i in range(num_layers):
            layer = nn.Module()  # lightweight container to hold submodules; registrations will work
            # Multihead attention (batch_first=True for convenience)
            layer.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
            # Feed-forward network
            layer.linear1 = nn.Linear(d_model, d_ff)
            layer.linear2 = nn.Linear(d_ff, d_model)
            # Layer norms (pre-norm architecture)
            layer.norm1 = nn.LayerNorm(d_model)
            layer.norm2 = nn.LayerNorm(d_model)
            # Dropout for sublayers
            layer.dropout = nn.Dropout(dropout)
            # LayerScale parameters (initialized to small values to stabilize early training)
            init_scale = 1e-4
            layer.layer_scale_1 = nn.Parameter(torch.ones(d_model) * init_scale)
            layer.layer_scale_2 = nn.Parameter(torch.ones(d_model) * init_scale)
            # Per-layer drop path probability
            layer.drop_path_prob = max_drop_path * float(i) / max(1, num_layers - 1)
            self.layers.append(layer)

        # Final layer norm after the encoder stack
        self.final_layer_norm = nn.LayerNorm(d_model)

        # Output projection tied to embedding weights
        self.output_projection = nn.Linear(d_model, vocab_size, bias=True)
        self.output_projection.weight = self.embedding.weight

        # Additional dropout before logits
        self.dropout = nn.Dropout(dropout)

        # Activation function (SiLU = Swish)
        self.activation = nn.SiLU()

        # Initialize weights
        self._init_weights()

        print(f"GeneratedTransformerModel_v4 initialized with d_model={d_model}, num_layers={num_layers}, num_heads={num_heads}")

    def _drop_path(self, x: torch.Tensor, drop_prob: float) -> torch.Tensor:
        """
        DropPath / Stochastic Depth per sample (only when training).
        x: (batch, seq, d_model)
        """
        if drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - drop_prob
        # sample shape: (batch, 1, 1) to broadcast across seq and feature dims
        shape = (x.size(0), 1, 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        x = x / keep_prob * binary_tensor
        return x

    def _init_weights(self):
        # Xavier initialization for linear layers and embeddings, LayerNorm, and MHA parameters
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # in_proj_weight: (3*embed_dim, embed_dim) ; out_proj.weight: (embed_dim, embed_dim)
                if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if hasattr(module, 'in_proj_bias') and module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if hasattr(module, 'out_proj') and isinstance(module.out_proj, nn.Linear):
                    nn.init.xavier_uniform_(module.out_proj.weight)
                    if module.out_proj.bias is not None:
                        nn.init.zeros_(module.out_proj.bias)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch_size, seq_len)
        # attention_mask: (batch_size, seq_len) with 1 for real tokens and 0 for padding

        # Embedding + scaling
        x = self.embedding(input_ids) * math.sqrt(self.d_model)  # (batch, seq, d_model)
        x = self.embedding_layer_norm(x)
        # Positional encoding expects (seq_len, batch, d_model) as in prior versions
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)  # back to (batch, seq, d_model)
        x = self.emb_dropout(x)

        # Create causal mask for autoregressive generation
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1).to(input_ids.device)

        # Convert attention_mask to src_key_padding_mask format expected by MHA
        if attention_mask is not None:
            # attention_mask: 1 for real tokens, 0 for padding
            key_padding_mask = (attention_mask == 0)  # True for positions to mask
        else:
            key_padding_mask = None

        # Pass through custom encoder layers with pre-norm, LayerScale, and DropPath
        for layer in self.layers:
            # Pre-norm for attention
            x_norm = layer.norm1(x)
            # MultiheadAttention expects (batch, seq, embed) with batch_first=True
            attn_out, _ = layer.self_attn(x_norm, x_norm, x_norm, attn_mask=causal_mask, key_padding_mask=key_padding_mask, need_weights=False)
            attn_out = layer.dropout(attn_out)
            # Apply LayerScale (broadcast across batch and seq)
            attn_out = attn_out * layer.layer_scale_1.unsqueeze(0).unsqueeze(0)
            # Stochastic depth (DropPath)
            attn_out = self._drop_path(attn_out, layer.drop_path_prob)
            # Residual connection
            x = x + attn_out

            # Pre-norm for FFN
            x_norm2 = layer.norm2(x)
            ff = layer.linear1(x_norm2)
            ff = self.activation(ff)
            ff = layer.dropout(ff)
            ff = layer.linear2(ff)
            ff = layer.dropout(ff)
            # LayerScale on FFN
            ff = ff * layer.layer_scale_2.unsqueeze(0).unsqueeze(0)
            # DropPath on FFN
            ff = self._drop_path(ff, layer.drop_path_prob)
            # Residual connection
            x = x + ff

        # Final normalization for stable outputs
        x = self.final_layer_norm(x)
        x = self.dropout(x)

        logits = self.output_projection(x)
        return logits


class PositionalEncoding_v5(nn.Module):
    """
    Evolution v5 positional encoding:
    - Sinusoidal base (deterministic) for good extrapolation
    - Learnable global scaling (alpha) to let the model tune positional magnitude
    - Learnable phase shift per dimension to allow slight learnable offsets
    - Dropout for regularization of positional contribution
    - Expects input shape (seq_len, batch, d_model) and returns same shape
    """
    def __init__(self, d_model: int, max_seq_length: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.dropout = nn.Dropout(dropout)

        # Create deterministic sinusoidal positional encodings (max_seq_length x d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # (d_model/2)

        pe = torch.zeros(max_seq_length, d_model)  # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it's moved with the module but not a parameter
        self.register_buffer("pe", pe)  # (max_len, d_model)

        # Learnable global scale for positional encodings (initialized near 1.0)
        self.alpha = nn.Parameter(torch.tensor(1.0))

        # Learnable phase shift to slightly adjust the sinusoidal basis per feature
        self.phase = nn.Parameter(torch.zeros(d_model))

        # Small initialization noise to avoid symmetry
        nn.init.normal_(self.phase, mean=0.0, std=1e-3)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        seq_len = x.size(0)
        if seq_len > self.max_seq_length:
            # If dynamic longer sequences occur, generate extra positional encodings on the fly
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * -(math.log(10000.0) / self.d_model))
            pe = torch.zeros(seq_len, self.d_model, device=x.device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe = self.pe[:seq_len].to(x.device)  # (seq_len, d_model)

        # Apply learnable phase shift
        pe = pe + (self.phase * 1e-3)

        # Scale positional encodings with learnable alpha
        pe = pe * self.alpha

        # Convert to (seq_len, 1, d_model) so it can broadcast over batch
        pe = pe.unsqueeze(1)

        x = x + pe  # (seq_len, batch, d_model)
        x = self.dropout(x)
        return x


class GeneratedTransformerModel_v5(nn.Module):
    """
    Evolution iteration 5 of the Transformer model.
    Evolutionary improvements included:
    - Learnable embedding scale (lets model adjust embedding magnitude instead of fixed sqrt(d_model))
    - PositionalEncoding_v5 with learnable scale and phase shift for flexible positional signals
    - Layer (stochastic) layer-drop: randomly skip entire encoder layers during training (regularization similar to stochastic depth)
      with expectation-preserving scaling when layers are active
    - Pre-norm TransformerEncoderLayer (norm_first=True) retained for stability
    - Xavier uniform initialization with small gain tweak and LayerNorm init carefully set
    - Weight tying between input embedding and output projection preserved
    - Slightly stronger regularization via embedding LayerNorm + dropout and final dropout
    - Maintains same interface as previous versions
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Learnable scalar to replace fixed sqrt(d_model) scaling; initialized to sqrt(d_model)
        self.embedding_scale = nn.Parameter(torch.tensor(math.sqrt(d_model), dtype=torch.float))
        # Embedding LayerNorm and dropout for regularization/stability
        self.embedding_layer_norm = nn.LayerNorm(d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Positional encoding module (v5) - expects (seq_len, batch, d_model)
        self.pos_encoding = PositionalEncoding_v5(d_model, max_seq_length, dropout=dropout)

        # Transformer encoder layers (pre-norm for stability in deep stacks)
        # Keep batch_first=True so tensors are (batch, seq, d_model) for encoder layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation='gelu',  # GELU retained for smoothness; stable and widely effective
                batch_first=True,
                norm_first=True
            ) for _ in range(num_layers)
        ])

        # Per-layer stochastic depth rates (increasing deeper into the network)
        # Using a gentle schedule; max drop is a fraction of dropout to avoid destabilizing training
        max_layer_drop = min(0.2, dropout * 1.0)  # cap the maximum
        if num_layers > 1:
            # linearly increase drop rate from 0 to max_layer_drop across layers
            self.layer_drop_rates = [float(r) for r in torch.linspace(0.0, max_layer_drop, steps=num_layers)]
        else:
            self.layer_drop_rates = [0.0 for _ in range(num_layers)]

        # Final layer norm for stable outputs
        self.final_layer_norm = nn.LayerNorm(d_model)

        # Output projection tied to embedding weights (weight tying)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=True)
        # Tie weights
        self.output_projection.weight = self.embedding.weight

        # Final dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

        print(f"GeneratedTransformerModel_v5 initialized with d_model={d_model}, num_layers={num_layers}, num_heads={num_heads}")

    def _init_weights(self):
        # Xavier initialization for linear layers and embeddings, LayerNorm set to sensible defaults
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform with small gain to keep activations in stable range
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Xavier uniform for embeddings
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm weights to 1 and bias to 0
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch_size, seq_len)
        # attention_mask: (batch_size, seq_len) with 1 for real tokens and 0 for padding

        # Embedding + learnable scaling
        x = self.embedding(input_ids) * self.embedding_scale  # (batch, seq, d_model)
        x = self.embedding_layer_norm(x)
        # Transformer positional encoding expects (seq_len, batch, d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.emb_dropout(x)

        # Create causal mask for autoregressive generation (seq_len x seq_len)
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device), diagonal=1)

        # Convert attention_mask to src_key_padding_mask (True for positions to mask)
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # (batch, seq) boolean
        else:
            key_padding_mask = None

        # Pass through each encoder layer with stochastic depth regularization
        for i, layer in enumerate(self.layers):
            p_drop = self.layer_drop_rates[i] if i < len(self.layer_drop_rates) else 0.0
            if self.training and p_drop > 0.0:
                # Decide whether to drop this layer (skip applying it)
                if torch.rand(1, device=x.device).item() < p_drop:
                    # Skip layer entirely (identity)
                    continue
                else:
                    # Apply layer and scale outputs to preserve expectation
                    x = layer(x, src_mask=causal_mask, src_key_padding_mask=key_padding_mask)
                    # Scale to keep expected magnitude stable (divide by survival probability)
                    survival = 1.0 - p_drop
                    if survival > 0.0:
                        x = x / survival
                    else:
                        # Extremely unlikely due to clamping, but guard division by zero
                        x = x
            else:
                # Deterministic path (evaluation or no drop)
                x = layer(x, src_mask=causal_mask, src_key_padding_mask=key_padding_mask)

        # Final normalization and dropout
        x = self.final_layer_norm(x)
        x = self.dropout(x)

        logits = self.output_projection(x)
        return logits