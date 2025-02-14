"""Deep Reorder is an experiment for reordering MLP blocks based on activation correlation."""

from typing import Tuple, Callable
import math

import torch
from torch import nn

batched_corrcoef = torch.func.vmap(torch.corrcoef)
# batched_corrcoef = torch.compile(torch.func.vmap(torch.corrcoef), backend='eager')


def register_buffers(model: nn.Module):
    """Register necessary buffers and parameters in the model.

    Args:
        model (nn.Module):
            model to add buffers and parameters to.
    """
    for idx, layer in enumerate(model.layers):
        idx: int
        layer: nn.Module
        linear_positions = nn.Parameter(math.sqrt(layer.hidden_size) * torch.rand(layer.hidden_size, requires_grad=True, dtype=torch.float32))
        original_positions = linear_positions.detach()

        sorted_indices = torch.argsort(linear_positions)
        ranked_positions = torch.zeros_like(linear_positions, dtype=torch.long)
        ranked_positions[sorted_indices] = torch.arange(layer.hidden_size)
        colors = nn.functional.normalize(ranked_positions.float(), dim=-1)
        colors.requires_grad = False

        correlation_matrix = torch.zeros((layer.hidden_size, layer.hidden_size), requires_grad=False, dtype=torch.float32)
        layer.register_buffer("colors", colors)
        layer.register_parameter("linear_positions", linear_positions)
        layer.register_buffer("original_positions", original_positions)
        layer.register_buffer("correlation_matrix", correlation_matrix)


def register_hooks(model: nn.Module):
    """Register necessary hooks in the model.

    Args:
        model (nn.Module):
            model to add hooks to.
    """

    def save_activation_hook(_module_name: str) -> Callable[[nn.Module, Tuple[torch.Tensor], Tuple[torch.Tensor]], None]:
        """Generate hook to attach to modules.

        Args:
            _module_name (str):
                name of the module.

        Returns:
            A hook that computes and stores correlation matrices for activations of a module.
        """

        def hook(module: nn.Module, _input_tensor: Tuple[torch.Tensor], output_tensor: Tuple[torch.Tensor]):
            """Compute correlation matrix for activations and store in buffer.

            Args:
                module (nn.Module):
                    module that is calling this hook
                _input_tensor (Tuple[torch.Tensor]):
                    input that is fed into the module
                output_tensor (Tuple[torch.Tensor]):
                    output from the module
            """
            module.correlation_matrix = batched_corrcoef(torch.transpose(output_tensor[0], 1, 2))

        return hook

    for idx, layer in model.layers.named_children():
        layer.register_forward_hook(save_activation_hook(f"{layer.__class__.__name__}_{idx}"))


@torch.compile(backend="eager")
def _compute_distance_matrix(positions: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distance matrix, given a positions vector."""
    col_vec = positions.unsqueeze(1)
    row_vec = positions.unsqueeze(0)
    return torch.abs(col_vec - row_vec)


@torch.compile(backend="eager")
def _compute_layer_loss(layer: nn.Module) -> torch.Tensor:
    """Compute the loss for a layer."""
    distance_matrix = _compute_distance_matrix(layer.linear_positions)
    target_distance_matrix = torch.clamp(1 - layer.correlation_matrix, min=0)
    loss_matrix = (distance_matrix - target_distance_matrix) ** 2
    return torch.mean(loss_matrix)


@torch.compile(backend="eager")
def compute_model_loss(model: nn.Module) -> torch.Tensor:
    """Compute the DeepReorder loss for a model."""
    return torch.mean(torch.stack([_compute_layer_loss(layer) for _, layer in model.layers.named_children()]))
