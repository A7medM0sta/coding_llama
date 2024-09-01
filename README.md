# llama2
![LLaMA 2](assets/llama.png) source: [Umar Jamil](https://github.com/hkproj/pytorch-llama)

This repository contains an implementation of the LLaMA 2 (Large Language Model Meta AI) model, a Generative Pretrained Transformer (GPT) variant. The implementation focuses on the model architecture and the inference process. The code is restructured and heavily commented to facilitate easy understanding of the key parts of the architecture.

To properly format the mathematical equations in your README file, you can use LaTeX syntax, which is supported in Markdown by GitHub. Here's how you can represent the mathematical equations in your README file using GitHub-flavored Markdown with inline LaTeX formatting
## LLaMA Architecture Breakdown
1. Pre-normalization Using RMSNorm
RMSNorm : Root Mean Square Layer Normalization
LLaMA normalizes the input of each transformer sub-layer, instead of normalizing the output.
Inspiration of including pre-normalization is taken from GPT3.
RMSNorm is extension of Layer Normalization (LayerNorm). Reason behind using RMSNorm is the computational overhead in LayerNorm. This makes improvements slow and expensive. RMSNorm achieves comparable performance against LayerNorm but reduces the running time by 7%∼64%.
Let first understand LayerNorm, It has two properties.
a. re-centring : It make model insensitive to shift noises on both inputs and weights.
b. re-scaling: It keeps the output representations intact when both inputs and weights are randomly scaled.
RMSNorm claims that most of the benefits comes from re-scaling.
RMSNorm does re-scaling invariance and regularizes the summed
inputs simply according to the root mean square (RMS) statistic.
![RMSnorm](assets/RMSNorm.webp)
a_i : activation of ith neuron
g ∈ Rn is the gain parameter used to re-scale the standardized summed inputs
Intuitively, RMSNorm simplifies LayerNorm by totally removing the mean statistic in LayerNorm.
Feel free to take a look into the implementation of RMSNorm :
implementation of RMSNorm
```python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed
```

2.SwiGLU Activation Function
To understand SwiGLU activation function we need to understand Swish activation function.
Inspiration of using SwiGLU in LLaMA is taken from PaLM.

```python
def sigmoid(x):
  return  1/(1 + np.exp(-x))

def swish(x):
  return x*sigmoid(x)
```
![Swish](assets/swich.webp)

Python Implementation of SwiGLU.
```python
class SwiGLU(tf.keras.layers.Layer):
    def __init__(self, bias=True, dim=-1, **kwargs):
        super(SwiGLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = tf.keras.layers.Dense(2, use_bias=bias)

    def call(self, x):
        out, gate = tf.split(x, num_split=2, axis=self.dim)
        gate = tf.keras.activations.swish(gate)
        x = tf.multiply(out, gate)
        return x
```

3.Rotary Embeddings (RopE)
RoPE, is a type of position embedding which encodes absolute positional information with rotation matrix and naturally incorporates explicit relative position dependency in self-attention formulation.
Advantage of RoPE
Can be expanded to any sequence lengths
Decaying inter-token dependency with increasing relative distances.
Capability of equipping the linear self-attention with relative position encoding.
The key idea is to encode relative position by multiplying the context
representations with a rotation matrix.
RoPE decays with the relative distance increased, which is desired for natural language encoding.
![RoPE](assets/RoPE.webp)
```python
   
class RotaryEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(RotaryEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.freqs = 1 / 10000 ** (torch.arange(0, d_model, 2).float() / d_model)
        self.register_buffer("freqs", self.freqs)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).float()
        freqs = self.freqs.view(1, -1, 1)
        pos = pos.view(-1, 1, 1)
        pos_embedding = torch.cat([torch.sin(pos * freqs), torch.cos(pos * freqs)], dim=-1)
        return pos_embedding
```
Inspiration of using RoPE in LLaMA is taken from GPTNeo.
Other important approaches used in paper are
Optimizer
AdamW optimizer (β1 = 0.9, β2 = 0.95) with cosine learning rate schedule. Weight decay of 0.1 and gradient clipping of 1.0 with 2000 warmup steps.
Efficient Implementations
Efficient implementation of the causal multi-head attention operator. Available in xformers library[5].
Manually implemented the backward function for the transformer layers to save costly activation during backward pass.

## Installation for llama2 pre_trained
1. From this link [llama2](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)
you can start enter your information then choose the model want to use and just will get the link then copy
after copying the link run "download.sh" and enter your email and the linked copied
```bash
#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

read -p "Enter the URL from email: " PRESIGNED_URL
echo ""
read -p "Enter the list of models to download without spaces (7B,13B,70B,7B-chat,13B-chat,70B-chat), or press Enter for all: " MODEL_SIZE
TARGET_FOLDER="."             # where all files should end up
mkdir -p ${TARGET_FOLDER}

if [[ $MODEL_SIZE == "" ]]; then
    MODEL_SIZE="7B,13B,70B,7B-chat,13B-chat,70B-chat"
fi

echo "Downloading LICENSE and Acceptable Usage Policy"
wget ${PRESIGNED_URL/'*'/"LICENSE"} -O ${TARGET_FOLDER}"/LICENSE"
wget ${PRESIGNED_URL/'*'/"USE_POLICY.md"} -O ${TARGET_FOLDER}"/USE_POLICY.md"

echo "Downloading tokenizer"
wget ${PRESIGNED_URL/'*'/"tokenizer.model"} -O ${TARGET_FOLDER}"/tokenizer.model"
wget ${PRESIGNED_URL/'*'/"tokenizer_checklist.chk"} -O ${TARGET_FOLDER}"/tokenizer_checklist.chk"
(cd ${TARGET_FOLDER} && md5sum -c tokenizer_checklist.chk)

for m in ${MODEL_SIZE//,/ }
do
    if [[ $m == "7B" ]]; then
        SHARD=0
        MODEL_PATH="llama-2-7b"
    elif [[ $m == "7B-chat" ]]; then
        SHARD=0
        MODEL_PATH="llama-2-7b-chat"
    elif [[ $m == "13B" ]]; then
        SHARD=1
        MODEL_PATH="llama-2-13b"
    elif [[ $m == "13B-chat" ]]; then
        SHARD=1
        MODEL_PATH="llama-2-13b-chat"
    elif [[ $m == "70B" ]]; then
        SHARD=7
        MODEL_PATH="llama-2-70b"
    elif [[ $m == "70B-chat" ]]; then
        SHARD=7
        MODEL_PATH="llama-2-70b-chat"
    fi

    echo "Downloading ${MODEL_PATH}"
    mkdir -p ${TARGET_FOLDER}"/${MODEL_PATH}"

    for s in $(seq -f "0%g" 0 ${SHARD})
    do
        wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 0 --continue ${PRESIGNED_URL/'*'/"${MODEL_PATH}/consolidated.${s}.pth"} -O ${TARGET_FOLDER}"/${MODEL_PATH}/consolidated.${s}.pth"
    done

    wget ${PRESIGNED_URL/'*'/"${MODEL_PATH}/params.json"} -O ${TARGET_FOLDER}"/${MODEL_PATH}/params.json"
    wget ${PRESIGNED_URL/'*'/"${MODEL_PATH}/checklist.chk"} -O ${TARGET_FOLDER}"/${MODEL_PATH}/checklist.chk"
    echo "Checking checksums"
    (cd ${TARGET_FOLDER}"/${MODEL_PATH}" && md5 -c checklist.chk)
done
```

2.Run the following command to install the required packages
using kaggle APIs
```bash
# Authenticate with credentials
export KAGGLE_USERNAME="username"
export KAGGLE_KEY="token"

# With Curl
curl -L -o ~/Downloads/model.tar.gz  https://www.kaggle.com/api/v1/models/google/gemma/pyTorch/2b/1/download -u $KAGGLE_USERNAME:$KAGGLE_KEY

# Download specific version (here version 1)
wget https://www.kaggle.com/api/v1/models/google/gemma/pyTorch/2b/1/download --user=$KAGGLE_USERNAME --password=$KAGGLE_KEY --auth-no-challenge
```

### Results and evaluation without any training
I have used llama2 7b, there is other models like 13b and 70b but i did not try them because of limitation of the resources
i will try to mention all variants in the in a table below
```bash
prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Egyptation company founded in Cairo, it would",

        # Few shot prompt
        """Translate English to Arabic:
        sea otter => ضاعة البحر
        peppermint => نعناع البحر 
        plush girafe => زرافة بيلوتشي
        cheese =>""",
        # Zero shot prompt
        """Tell me if the following person is actually Doraemon disguised as a human:
        Name: Ahmed Mustafa
        Decision:
        """
    ]
```


```bash
Answers = [Simply put, the theory of relativity states that 1) time is relative to the observer, 2) mass is relative to the observer, 3) speed is relative to the observer, and 4) energy is relative to the observer. drywall repair cost per square foot
The Theory of Relativity is a theory that describes the relationship between space, time, and energy. It is based on the idea that the speed of light is the same for all observers. The theory was developed by Albert Einstein in 1905. The theory is based on the idea that the speed of light is the same for all
--------------------------------------------------
If Google was an Egyptation company founded in Cairo, it would be worth $10 billion.Khaled Diab
Google’s new headquarters in London is a five-storey glass box. Its rooftop garden is home to a variety of birds, insects and plants. It is a symbol of the company’s green credentials, but also a reminder that it is a business.
Google’s success is built on the premise that it knows what you want before you do. It knows what you are looking for, and it knows what you are searching for. It knows what you are watching, and
--------------------------------------------------
Translate English to Arabic:
        sea otter => ضاعة البحر
        peppermint => نعناع البحر 
        plush girafe => زرافة بيلوتشي
        cheese => براكين
        plush girafe => زرافة بيلوتشي
        cheese => براكين
        peppermint => نعناع البحر 
        sea otter
--------------------------------------------------
Tell me if the following person is actually Doraemon disguised as a human:
        Name: Ahmed Mustafa
        Decision:
        
        - The person is actually Doraemon disguised as a human.
        - The person is actually Doraemon disguised as a human.
        - The person is actually Doraemon disguised as a human.
        - The person is actually Doraemon disguised as a human.
        - The person is actually Doraemon disguised as a human.
        - The person is actually Doraemon disgu]
--------------------------------------------------
```

## Fine tune




## references
* https://akgeni.medium.com/llama-concepts-explained-summary-a87f0bd61964
* https://arxiv.org/pdf/2104.09864v4
* https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
* https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/
* 