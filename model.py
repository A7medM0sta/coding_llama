from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    # Later Set in the Build Method
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for kv cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Root Mean Square Layer Normalization (RMSNorm) module.

        RMSNorm is a normalization technique similar to Layer Normalization (LayerNorm) but
        without the mean subtraction step. It normalizes the input based on its root mean
        square (RMS) value, which can stabilize the training of deep neural networks.

        :param dim: The dimension of the input tensor.
        :param eps: A small epsilon value to avoid division by zero during normalization.
                    The default value is 1e-6.
        """
        super().__init__()
        self.eps = eps

        # Gamma parameter: A learnable scaling parameter initialized to ones with the same
        # dimension as the input tensor. This helps to re-scale the normalized values.
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Root Mean Square (RMS) normalization of the input tensor.

        :param x: The input tensor of shape (B, Seq_len, Dim), where:
                  B is the batch size,
                  Seq_len is the sequence length,
                  Dim is the feature dimension.
        :return: The normalized tensor of the same shape as the input.

        Formula: RMSNorm(x) = x / sqrt(mean(x^2) + eps)
        """

        # (B, Seq_len, Dim) * (B, Seq_len, 1) = (B, Seq_len, Dim)
        # `rsqrt`: Computes the reciprocal of the square root, i.e., 1/sqrt(x).
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RMSNorm module.

        Applies RMS normalization followed by a re-scaling using the learned gamma
        parameter (weight).

        :param x: The input tensor of shape (B, Seq_len, Dim).
        :return: The re-scaled, normalized tensor of the same shape as the input.
        """

        # Element-wise multiplication with the learned weight (gamma) parameter.
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float=10000.0):
    """
    Precompute the positional frequencies for the rotary positional embeddings, as described in the paper.

    This function calculates positional frequencies based on the given head dimension and sequence length.
    It is specifically designed to work with models that use rotary positional embeddings, such as Transformer-based models.

    The frequencies are computed using the formula:
        theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ..., dim/2]
    where `dim` refers to the head dimension.

    The function returns a complex tensor representing the positional frequencies.

    :param head_dim: The dimension of the attention head. It must be divisible by 2.
    :param seq_len: The length of the input sequence.
    :param device: The device on which to compute the tensor (e.g., 'cpu' or 'cuda').
    :param theta: The base frequency parameter used in the frequency calculation. Default is 10000.0.
    :return: A tensor of complex values representing the precomputed positional frequencies with shape (Seq_Len, Head_Dim / 2).
    """
    # as written in the paragraph 3.2.2 of the paper
    # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"

    # Build the theta parameter
    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()

    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)  # (Dim / 2)
    # Construct the positions (the "m" parameter)
    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device=device)

    # Multiply each theta by each position using the outer product.
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()

    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    """
    Apply rotary positional embeddings to the input tensor `x` by rotating the input using precomputed complex positional frequencies.

    This function uses the rotary embeddings technique described in Transformer models for positional encoding.
    The input tensor is first converted to a complex representation, then rotated using the precomputed positional frequencies (`freqs_complex`).
    Finally, the rotated tensor is converted back to a real-valued tensor.

    :param x: Input tensor of shape (B, Seq_Len, H, Head_Dim), where B is the batch size, Seq_Len is the sequence length, H is the number of heads, and Head_Dim is the head dimension.
    :param freqs_complex: Precomputed complex tensor of positional frequencies with shape (Seq_Len, Head_Dim / 2).
    :param device: The device on which to perform the computation (e.g., 'cpu' or 'cuda').
    :return: The input tensor `x` after applying the rotary positional embeddings, with the same shape as the input tensor.
    """
    # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
    # Two consecutive values will become a single complex number
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # Reshape the freqs_complex tensor to match the shape of the x_complex tensor.
    # Add the batch dimension and the head dimension
    # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
    # This results in the rotation of the complex number as shown in Figure 1 of the paper
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex

    # Convert the complex number back to the real number
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)

    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats the keys/values to match the number of query heads.

    :param x: Input tensor of shape (batch_size, seq_len, n_kv_heads, head_dim).
    :param n_rep: The number of repetitions for each key/value head.
    :return: Output tensor of shape (batch_size, seq_len, n_kv_heads * n_rep, head_dim).
    """
    batch_size, seq_len, n_kv_heads, head_dim = x.shape

    # If no repetition is needed, return the input tensor as is
    if n_rep == 1:
        return x

    # Repeat the heads and reshape the tensor
    return (
        x.unsqueeze(2)  # (B, Seq_Len, N_KV_Heads) -> (B, Seq_Len, 1, N_KV_Heads, Head_Dim)
        .expand(batch_size, seq_len, n_rep, n_kv_heads, head_dim)  # Expand for repetition
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)  # Reshape to combine repeated heads
    )


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for the Transformer model. This block includes query, key, and value projections,
    rotary embeddings for positional encoding, and attention score computation.

    :param args: A `ModelArgs` object containing model configuration such as `dim`, `n_heads`, and `max_seq_len`.

    Methods:
    - forward(x, start_pos, freqs_complex): Applies self-attention to the input tensor `x`.
    """

    def __init__(self, args: ModelArgs):
        """
        Initializes the SelfAttention block.

        :param args: A `ModelArgs` object containing the necessary configuration for the model. This includes:
                     - `dim`: The dimension of the input tensor.
                     - `n_heads`: The number of attention heads.
                     - `n_kv_heads`: The number of heads for keys and values (defaults to `n_heads` if not specified).
                     - `max_batch_size`: The maximum batch size for caching.
                     - `max_seq_len`: The maximum sequence length for caching.
        """
        super().__init__()
        # Set the number of heads for keys and values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        # Set the number of heads for queries
        self.n_heads_q = args.n_heads

        # Determine the replication factor for keys and values
        self.n_rep = self.n_heads_q // self.n_kv_heads

        # Set the dimension of each head
        self.head_dim = args.dim // args.n_heads

        # Linear projections for queries, keys, and values
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # Cache for keys and values
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        """
        Forward pass for the SelfAttention block.

        :param x: The input tensor of shape (batch_size, seq_len, dim).
        :param start_pos: The starting position in the sequence.
        :param freqs_complex: The complex frequencies for rotary embeddings.

        :return: The output tensor of shape (batch_size, seq_len, dim) after applying self-attention.
        """
        batch_size, seq_len, _ = x.shape  # (B, 1, Dim)

        # Project the input tensor to queries, keys, and values
        # (B, 1, Dim) -> (B, 1, H_Q * Head_Dim)
        xq = self.wq(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xk = self.wk(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xv = self.wv(x)

        # Reshape projections for multi-head attention
        # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply rotary embeddings to queries and keys
        # (B, 1, H_Q, Head_Dim) --> (B, 1, H_Q, Head_Dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        # (B, 1, H_KV, Head_Dim) --> (B, 1, H_KV, Head_Dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # Update the key and value caches
        self.cache_k[:batch_size, start_pos: start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos: start_pos + seq_len] = xv

        # Retrieve cached keys and values
        # (B, Seq_Len_KV, H_KV, Head_Dim)
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        # (B, Seq_Len_KV, H_KV, Head_Dim)
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # Repeat keys and values to match the number of query heads
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = repeat_kv(keys, self.n_rep)
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        values = repeat_kv(values, self.n_rep)

        # Transpose tensors for attention score computation
        # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention scores
        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # Apply softmax to attention scores
        # (B, H_Q, 1, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # Compute attention-weighted sum of values
        # (B, H_Q, 1, Seq_Len) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)
        # Reshape output to match the input shape
        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))

        # Apply final linear projection
        return self.wo(output)  # (B, 1, Dim) -> (B, 1, Dim)




class FeedForward(nn.Module):
    """
    Feed-forward network block for the Transformer model. This block includes two linear transformations with
    a Swish (SiLU) activation function in between, commonly used in Transformer-based models.

    :param args: A `ModelArgs` object containing model configuration such as `dim`, `ffn_dim_multiplier`, and `multiple_of`.

    Methods:
    - forward(x): Applies the feed-forward network to the input tensor `x`.
    """

    def __init__(self, args: ModelArgs):
        """
        Initializes the FeedForward block.

        :param args: A `ModelArgs` object containing the necessary configuration for the model. This includes:
                     - `dim`: The dimension of the input tensor.
                     - `ffn_dim_multiplier`: Multiplier for adjusting the hidden dimension of the feed-forward layers.
                     - `multiple_of`: The multiple to which the hidden dimension is rounded.
        """
        super().__init__()
        hidden_dim = 4 * args.dim
        # Adjust the hidden dimension as per the Transformer model architecture
        hidden_dim = int(2 * hidden_dim / 3)

        # If a multiplier is provided, scale the hidden dimension accordingly
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        # Round the hidden dimension to the nearest multiple of the `multiple_of` parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        # First linear transformation layer
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        # Second linear transformation layer (output layer)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        # Third linear transformation layer (used for element-wise multiplication with the Swish activation)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for the FeedForward block.

        :param x: The input tensor of shape (batch_size, seq_len, dim).

        :return: The output tensor of shape (batch_size, seq_len, dim) after applying the feed-forward network.
        """
        # Apply the first linear transformation followed by the Swish activation function
        # (B, Seq_len, Dim) --> (B, Seq_len, Hidden_Dim)
        swish = F.silu(self.w1(x))

        # Apply the third linear transformation
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)

        # Perform element-wise multiplication between the Swish-activated tensor and the third linear transformation result
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V

        # Apply the second linear transformation to get the final output
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)

        return x

class EncoderBlock(nn.Module):
    """
    Encoder block for a Transformer model. This block consists of self-attention and feed-forward layers,
    each followed by RMS normalization. It processes input tensors and returns the output after applying attention
    and feed-forward transformations.

    :param args: A `ModelArgs` object containing model configuration such as `dim`, `n_heads`, and `norm_eps`.

    Methods:
    - forward(x, start_pos, freqs_complex): Applies self-attention and feed-forward layers to the input tensor `x`.
    """

    def __init__(self, args: ModelArgs):
        """
        Initializes the EncoderBlock.

        :param args: A `ModelArgs` object containing the necessary configuration for the model. This includes:
                     - `dim`: The dimension of the input tensor.
                     - `n_heads`: The number of attention heads.
                     - `norm_eps`: The epsilon value for normalization layers.
        """
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        # Self-attention layer
        self.attention = SelfAttention(args)
        # Feed-forward layer
        self.feed_forward = FeedForward(args)

        # RMS normalization layer applied before the attention block
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # RMS normalization layer applied before the feed-forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        """
        Forward pass for the EncoderBlock.

        :param x: The input tensor of shape (batch_size, seq_len, dim).
        :param start_pos: The starting position in the sequence for computing positional encodings.
        :param freqs_complex: The precomputed complex frequencies for rotary embeddings, used in the attention mechanism.

        :return: The output tensor of the same shape as the input, after applying attention and feed-forward transformations.
        """
        # Apply attention normalization and self-attention, then add the residual connection
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex
        )
        # Apply feed-forward normalization and feed-forward layer, then add the residual connection
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out




class Transformer(nn.Module):
    """
    A Transformer model for sequence processing. This implementation uses rotary embeddings and consists of multiple encoder layers.

    :param args: A `ModelArgs` object containing model configuration such as `vocab_size`, `dim`, `n_layers`, `n_heads`, `max_seq_len`, `norm_eps`, and `device`.

    Methods:
    - forward(tokens, start_pos): Processes input tokens and returns the model's output.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads,
                                                              self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, Seq_Len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output