"""Training script for DeepReorder."""

import argparse
import os
import io

import numpy as np
import torch
import accelerate
from torch import nn
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import deep_reorder

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_DATASET = "oivlisnet/c4-en"


def main(model_name: str, dataset_name: str):
    """Run the main training loop."""
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=8)
    device = accelerator.device

    log_dir = "runs/deepreorderexperiment"
    writer = SummaryWriter(log_dir=log_dir)

    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Load model and freeze weights.
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16", device_map="auto")
    model.model.eval()
    for param in model.model.parameters():
        param.requires_grad = False

    # Create necessary buffers.
    deep_reorder.register_buffers(model.model)
    deep_reorder.register_hooks(model.model)
    model = torch.compile(model, backend="eager")
    model.to(device)

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set up dataloaders.
    train_dataset = load_dataset(dataset_name, split="train").select_columns(["text"])
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    optimizer = torch.optim.AdamW(model.model.parameters(), lr=0.1)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1)

    # initial_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1, end_factor=0.1, total_iters=500)
    # warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.001, end_factor=0.1, total_iters=50)

    model, optimizer, scheduler, train_dataloader = accelerator.prepare(model, optimizer, scheduler, train_dataloader)

    train_bar = tqdm(train_dataloader, desc=f"[TRAIN] ", disable=not accelerator.is_local_main_process)
    for idx, batch in enumerate(train_bar):
        with accelerator.accumulate(model):
            # Tokenize.
            batch = batch["text"]
            batch = tokenizer(
                batch,
                padding=True,
                padding_side="left",
                truncation=True,
                max_length=AutoConfig.from_pretrained(model_name).max_position_embeddings,
                return_tensors="pt",
            )
            batch.to(device)

            # Run forward pass. Don't really care about what token was generated, so discard it.
            _ = model.generate(batch.input_ids, attention_mask=batch.attention_mask, do_sample=False, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id)
            loss = deep_reorder.compute_model_loss(model.model)
            train_bar.write(f"{loss.item(): >.3f}")

            accelerator.backward(loss)

            optimizer.step()
            scheduler.step(loss)
            optimizer.zero_grad()

            # Log Metrics
            writer.add_scalar("Loss/train", loss.item(), idx)
            writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], idx)

            # Log Charts
            if idx % 100 == 0:
                fig, axes = plt.subplots(len(model.model.layers), sharex=True, figsize=(20, 20))
                fig.tight_layout()
                for layer_idx, layer in enumerate(model.model.layers):
                    axes[layer_idx].clear()
                    axes[layer_idx].scatter(
                        layer.linear_positions.detach().cpu().numpy(),
                        torch.zeros_like(layer.linear_positions).detach().cpu().numpy(),
                        c=layer.colors.cpu().numpy(),
                        s=1,
                    )
                    axes[layer_idx].set_yticks([])
                    axes[layer_idx].set_ylabel(f"{layer_idx}")
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                image = np.array(plt.imread(buf))
                image = np.transpose(image, (2, 0, 1))
                writer.add_image("Linear Positions Scatter Plots", image, idx)
                plt.close(fig)

            # Checkpoint
            if idx % 2500 == 0 and idx != 0:
                accelerator.wait_for_everyone()
                output_checkpoints_dir = os.path.join(checkpoint_dir, f"step_{idx}")
                accelerator.save_state(output_checkpoints_dir)
                print(f"Checkpoint saved at step {idx} to {output_checkpoints_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="DeepReorder")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    args = parser.parse_args()

    main(
        model_name=args.model,
        dataset_name=args.dataset,
    )
