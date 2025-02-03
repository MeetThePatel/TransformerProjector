import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from datasets import load_dataset
from tqdm import tqdm

from deep_reorder import register_buffers, register_hooks

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Using {DEVICE}.")


def main():
    # Load model and freeze weights.
    model: Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", torch_dtype="bfloat16", device_map="auto")
    model.model.eval()
    for param in model.model.parameters():
        param.requires_grad = False

    # Create necessary buffers.
    register_buffers(model.model)
    register_hooks(model.model)

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

    # DATA LOADING CODE HERE
    train_dataset = load_dataset("sedthh/gutenberg_english", split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    train_bar = tqdm(train_dataloader, desc=f"[TRAIN] ")
    for batch in train_bar:
        batch = tokenizer(batch["TEXT"], padding=True, padding_side="left", truncation=True, return_tensors="pt").to(DEVICE)
        _ = model.generate(batch.input_ids, attention_mask=batch.attention_mask, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        loss = torch.sum(model.model.layers[0].activation_correlations.to(DEVICE) @ model.model.layers[0].linear_positions.to(DEVICE))
        loss.backward()

        # TODO: Need to add optimizer.


if __name__ == "__main__":
    main()
