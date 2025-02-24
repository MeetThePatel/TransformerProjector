"""Example training script for DeepReorder wrapper model."""

import os

import accelerate
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from transformers import AutoTokenizer
from rich import progress

from deep_reorder import DeepReorderModel, DeepReorderModelParams

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "vicgalle/alpaca-gpt4"
SYSTEM_PROMPT = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

GRADIENT_ACCUMULATION_STEPS = 1
BATCH_SIZE = 4


def construct_messages_batch(batch, system_prompt=SYSTEM_PROMPT):
    """Prepare prompt."""
    batch_messages = []
    instructions = batch["instruction"]
    inputs = batch["input"]

    for i in range(len(instructions)):
        instruction = instructions[i]
        input_text = inputs[i]

        messages = []
        messages.append({"role": "system", "content": system_prompt})
        if input_text:
            messages.append({"role": "user", "content": f"Instruction: {instruction}\nInput: {input_text}"})
        else:
            messages.append({"role": "user", "content": f"Instruction: {instruction}"})
        messages.append({"role": "assistant", "content": ""})

        batch_messages.append(messages)
    return batch_messages


def train(model_name: str, dataset_name: str):
    """Run the main training loop."""
    # Set up devices with Accelerate.
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    device = accelerator.device

    # Tensorboard and model checkpointing.
    log_dir = f"runs/deep_reorder_{model_name}"
    writer = SummaryWriter(log_dir=log_dir)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load model and freeze weights.
    model_params = DeepReorderModelParams()
    model = DeepReorderModel(model_name, model_params)
    for component in model_params.component_list:
        model.register_hooks(component)
    model.hf_model = torch.compile(model.hf_model, backend="eager")
    model.hf_model.to(device)

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set up dataloaders.
    train_dataset = load_dataset(dataset_name, split="train").select_columns(["instruction", "input", "output"])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Set up optimizer.
    params = []
    for layer in model.hf_model.model.layers:
        for component in model_params.component_list:
            params.append(layer.__getattr__(component).neuron_projections)
    for param in params:
        param.requires_grad = True

    optimizer = torch.optim.AdamW(params, lr=0.003)

    model.hf_model, optimizer, train_dataloader = accelerator.prepare(model.hf_model, optimizer, train_dataloader)

    # Training loop.
    step = 1
    with progress.Progress() as p_bar:
        for batch in p_bar.track(train_dataloader):
            with accelerator.accumulate(model):
                # Tokenize.
                messages_list = construct_messages_batch(batch)
                tokenized_inputs = tokenizer(
                    [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) for messages in messages_list],
                    padding=True,
                    padding_side="left",
                    truncation=True,
                    max_length=model.hf_model.config.max_position_embeddings,
                    return_tensors="pt",
                )
                tokenized_inputs.to(device)

                # Run forward pass. Don't really care about what token was generated, so discard it.
                _ = model.hf_model.generate(
                    tokenized_inputs.input_ids,
                    attention_mask=tokenized_inputs.attention_mask,
                    max_new_tokens=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                loss = model._compute_model_loss()
                p_bar.console.print(f"{loss.item(): >.3f}")

                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

                # Log Metrics
                writer.add_scalar("Loss/train", loss.item(), step)
                writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], step)

                # Checkpoint
                if step % 500 == 0:
                    accelerator.wait_for_everyone()
                    output_checkpoints_dir = os.path.join(checkpoint_dir, f"step_{step}")
                    accelerator.save_state(output_checkpoints_dir)
                    print(f"Checkpoint saved at step {step} to {output_checkpoints_dir}")
                step += 1


if __name__ == "__main__":
    train(model_name=MODEL_NAME, dataset_name=DATASET_NAME)
