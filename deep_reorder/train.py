"""Training script for DeepReorder."""

import argparse
import os

import torch
import accelerate
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
from rich import progress

from torch.utils.tensorboard import SummaryWriter

import deep_reorder
from deep_reorder import ProjectionInitialization

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DATASET = "vicgalle/alpaca-gpt4"
SYSTEM_PROMPT = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

PROJECTION_DIM = 3
NORM_ORDER = 2.0
INIT_SCHEME = ProjectionInitialization.RandUniform


def main(model_name: str, dataset_name: str):
    """Run the main training loop."""
    # Set up devices with Accelerate.
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=1)
    device = accelerator.device

    # Tensorboard and model checkpointing.
    log_dir = "runs/deepreorderexperimentInstruct3TESTParamsSet"
    writer = SummaryWriter(log_dir=log_dir)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load model and freeze weights.
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16", device_map="auto")
    model.model.eval()
    for param in model.model.parameters():
        param.requires_grad = False

    # Create necessary buffers and register necessary hooks.
    for layer in model.model.layers:
        deep_reorder.register_buffers(
            layer.self_attn,
            hidden_dim=model.config.hidden_size,
            projection_dim=PROJECTION_DIM,
            projection_initialization=INIT_SCHEME,
        )
        layer.self_attn.register_forward_hook(deep_reorder.correlation_calculation_hook)

        deep_reorder.register_buffers(
            layer.mlp,
            hidden_dim=model.config.hidden_size,
            projection_dim=PROJECTION_DIM,
            projection_initialization=INIT_SCHEME,
        )
        layer.mlp.register_forward_hook(deep_reorder.correlation_calculation_hook)

    # Compile model.
    model = torch.compile(model, backend="eager")
    model.to(device)

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set up dataloaders.
    train_dataset = load_dataset(dataset_name, split="train").select_columns(["instruction", "input", "output"])
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Set up optimizer.
    params = [layer.self_attn.neuron_projection for layer in model.model.layers] + [layer.mlp.neuron_projection for layer in model.model.layers]
    optimizer = torch.optim.AdamW(params, lr=0.003)

    # Set up learning rate schedules.
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1)

    # Set up Accelerate.
    model: AutoModelForCausalLM
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    train_dataloader: DataLoader
    # model, optimizer, scheduler, train_dataloader = accelerator.prepare(model, optimizer, scheduler, train_dataloader)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

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
                    max_length=AutoConfig.from_pretrained(model_name).max_position_embeddings,
                    return_tensors="pt",
                )
                tokenized_inputs.to(device)

                # Run forward pass. Don't really care about what token was generated, so discard it.
                _ = model.generate(
                    tokenized_inputs.input_ids,
                    attention_mask=tokenized_inputs.attention_mask,
                    max_new_tokens=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                loss = deep_reorder.compute_model_loss(model.model, p=NORM_ORDER)
                p_bar.console.print(f"{loss.item(): >.3f}")

                accelerator.backward(loss)
                # torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)

                optimizer.step()
                # scheduler.step()
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


def construct_messages_batch(batch, system_prompt=SYSTEM_PROMPT):
    """Prepare prompt."""
    batch_messages = []
    instructions = batch["instruction"]
    inputs = batch["input"]

    for i in range(len(instructions)):  # Iterate over the items in the batch
        instruction = instructions[i]
        input_text = inputs[i]  # Use the index to get corresponding inputs

        messages = []
        messages.append({"role": "system", "content": system_prompt})
        if input_text:
            messages.append(
                {
                    "role": "user",
                    "content": f"Instruction: {instruction}\nInput: {input_text}",
                }
            )
        else:
            messages.append({"role": "user", "content": f"Instruction: {instruction}"})
        messages.append({"role": "assistant", "content": ""})

        batch_messages.append(messages)
    return batch_messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="DeepReorder")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    args = parser.parse_args()

    main(
        model_name=args.model,
        dataset_name=args.dataset,
    )
