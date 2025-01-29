from dataclasses import dataclass

import torch
from torch.nn import TransformerDecoder
from torchtune.models.qwen2_5 import qwen2_5_3b, qwen2_5_tokenizer
from torchtune.models.qwen2_5._tokenizer import Qwen2_5Tokenizer
import safetensors.torch as stt


def load_model(weights_path: str, device: torch.device | str = 'cpu') -> torch.nn.Module:
    weights = stt.load_file(weights_path)
    model = qwen2_5_3b()
    model.to(device)
    for key in model.state_dict().keys():
        model.state_dict()[key] = weights[key]
    return model


def load_tokenizer(vocab_json_path: str, merges_txt_path: str, tokenizer_json_path: str) -> Qwen2_5Tokenizer:
    tokenizer = qwen2_5_tokenizer(path=vocab_json_path, merges_file=merges_txt_path, special_tokens_path=tokenizer_json_path)
    return tokenizer


@dataclass
class Qwen25GeneratorParams:
    # Maximum number of tokens to generate
    max_length: int = 1000
    # Controls randomness in sampling (higher = more random)
    temperature: float = 0.7
    # Nucleus sampling parameter
    top_p: float = 0.9
    # Top-k sampling parameter
    top_k: int = 50


class Qwen25Generator:
    def __init__(self, model: TransformerDecoder, tokenizer: Qwen2_5Tokenizer, params: Qwen25GeneratorParams, device: torch.device | str = 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.params = params
        self.device = device

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(self.device)

    def _top_p_filtering(self, logits, top_p=0.9):
        """
        Filter logits using nucleus (top-p) sampling
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        return logits

    def generate(self, prompt: str) -> str:
        """
        Generate text using the Qwen model autoregressively.

        Args:
            prompt (str): Input text to condition the generation

        Returns:
            str: Generated text including the prompt
        """
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt)[:-1]
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)

        with torch.no_grad():
            for _ in range(self.params.max_length):
                # Get model outputs
                outputs = self.model(input_ids)
                next_token_logits = outputs[:, -1, :].float()  # Ensure float type

                # Apply temperature
                next_token_logits = next_token_logits / self.params.temperature

                # Apply top-k filtering
                if self.params.top_k > 0:
                    values, _ = torch.topk(next_token_logits, self.params.top_k)
                    min_values = values[:, -1].unsqueeze(-1).expand_as(next_token_logits)
                    next_token_logits = torch.where(
                        next_token_logits < min_values,
                        torch.full_like(next_token_logits, float('-inf')),
                        next_token_logits
                    )

                # Apply top-p filtering
                if self.params.top_p < 1.0:
                    next_token_logits = self._top_p_filtering(next_token_logits, self.params.top_p)

                # Sample next token
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append next token to input_ids
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Check if we've generated an end token
                if next_token.item() == self.tokenizer.eos_id:
                    break

        # Decode the generated tokens
        generated_text = self.tokenizer.decode(input_ids[0].tolist())
        return generated_text
