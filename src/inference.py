from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from model import ModelArgs, Transformer


class LLaMA:
    """
    LLaMA class handles the model, tokenizer, and arguments for text completion tasks.
    """

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        """
        Initializes the LLaMA class with a model, tokenizer, and model arguments.

        Args:
            model (Transformer): The transformer model used for text completion.
            tokenizer (SentencePieceProcessor): The tokenizer for encoding and decoding text.
            model_args (ModelArgs): The model arguments that define the configuration of the model.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int,
              device: str):
        """
        Builds the LLaMA model from checkpoints and tokenizer.

        Args:
            checkpoints_dir (str): Directory containing the model checkpoints.
            tokenizer_path (str): Path to the tokenizer model.
            load_model (bool): Flag indicating whether to load the model from checkpoints.
            max_seq_len (int): Maximum sequence length for the model.
            max_batch_size (int): Maximum batch size for processing.
            device (str): The device to load the model onto (e.g., 'cpu', 'cuda').

        Returns:
            LLaMA: An instance of the LLaMA class with the model, tokenizer, and arguments initialized.
        """
        prev_time = time.time()

        if load_model:
            # Load the model checkpoint from the specified directory.
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()

        # Load model parameters from the JSON configuration file.
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        # Initialize model arguments with the given parameters.
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        # Load the tokenizer from the specified path.
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        # Set the default tensor type based on the device type.
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        # Create a Transformer model instance with the specified arguments.
        model = Transformer(model_args).to(device)

        if load_model:
            # Remove unmatched keys from the checkpoint and load the model's state dictionary.
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")

        return LLaMA(model, tokenizer, model_args)

    def text_completion(self, prompts: list[str], temperature: float = 0.6, top_p: float = 0.9,
                        max_gen_len: Optional[int] = None):
        """
        Generates text completions for the given prompts.

        Args:
            prompts (list[str]): List of input text prompts for generation.
            temperature (float, optional): Temperature for sampling (higher values make the output more random).
            top_p (float, optional): Top-p sampling (nucleus sampling) probability.
            max_gen_len (Optional[int], optional): Maximum number of tokens to generate.

        Returns:
            tuple: Generated tokens and corresponding text completions.
        """
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1

        # Convert each prompt into tokens using the tokenizer.
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]

        # Determine the batch size and ensure it does not exceed the maximum allowed batch size.
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"batch size must be less than or equal to {self.args.max_batch_size}"

        # Find the maximum length of the prompts and ensure it does not exceed the maximum sequence length.
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"

        # Calculate the total length for generation (prompt length + generated length).
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # Create a tensor to hold the tokens, initializing it with padding tokens.
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)

        # Populate the initial tokens tensor with the prompt tokens.
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        # Track whether EOS (end of sequence) tokens have been reached for each sequence.
        eos_reached = torch.tensor([False] * batch_size, device=device)

        # Create a mask to distinguish prompt tokens from generated tokens.
        prompt_tokens_mask = tokens != pad_id

        # Iterate through each position in the sequence and generate tokens.
        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
        for cur_pos in cur_iterator:
            with torch.no_grad():
                # Get the logits from the model for the current position.
                logits = self.model.forward(tokens[:, cur_pos - 1:cur_pos], cur_pos)

            if temperature > 0:
                # Apply temperature scaling and sample from the top-p distribution.
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedily select the token with the maximum probability.
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)

            # Only replace tokens that are padding tokens with the generated token.
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            # Update EOS reached status when the EOS token is found in non-prompt positions.
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)

            # Stop generation if EOS has been reached for all sequences.
            if all(eos_reached):
                break

        # Decode the generated tokens into text and return them.
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))

        return (out_tokens, out_text)

    def _sample_top_p(self, probs, p):
        """
        Samples the next token using top-p (nucleus) sampling.

        Args:
            probs (torch.Tensor): The probabilities of the tokens for the current position.
            p (float): The cumulative probability threshold for top-p sampling.

        Returns:
            torch.Tensor: The sampled token's index.
        """
        # Sort the probabilities in descending order.
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        # Compute the cumulative sum of the sorted probabilities.
        probs_sum = torch.cumsum(probs_sort, dim=-1)

        # Create a mask to zero out probabilities that exceed the cumulative probability threshold.
        mask = probs_sum - probs_sort > p

        # Set the masked probabilities to zero.
        probs_sort[mask] = 0.0

        # Normalize the remaining probabilities to sum to 1.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

        # Sample a token index from the top-p distribution.
        next_token = torch.multinomial(probs_sort, num_samples=1)

        # Retrieve the original token index from the sorted indices.
        next_token = torch.gather(probs_idx, -1, next_token)

        return next_token


if __name__ == '__main__':
    """
    The main execution script for running the LLaMA model. This script sets the device (CPU or GPU), 
    defines a list of example prompts for text completion, builds the LLaMA model, generates text completions,
    and prints the output texts.
    """

    torch.manual_seed(0)

    # Determine the device to run the model on (CPU or GPU).
    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    # Define some example prompts for text completion.
    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",

        # Few shot prompt
        """Translate English to French:
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        # Zero shot prompt
        """Tell me if the following person is actually Doraemon disguised as a human:
        Name: Ahmed Mustafa
        Decision: 
        """
    ]

    # Build the LLaMA model with the specified configuration.
    model = LLaMA.build(
        checkpoints_dir='../llama-2-7b/',
        tokenizer_path='llama-2-7b/tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device
    )

    # Generate text completions for the prompts.
    out_tokens, out_texts = (model.text_completion(prompts, max_gen_len=64))

    # Ensure the number of outputs matches the number of prompts.
    assert len(out_texts) == len(prompts)

    # Print the generated text for each prompt.
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)

