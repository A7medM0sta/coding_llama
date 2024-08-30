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
