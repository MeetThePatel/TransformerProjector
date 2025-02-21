"""Deep Reorder is an experiment for reordering MLP blocks based on activation correlation."""

import math
from enum import Enum
from typing import Tuple

import torch
from torch import nn

batched_corrcoef = torch.func.vmap(torch.corrcoef)


class ProjectionInitialization(Enum):
    """Initialization scheme to use for neuron projections.

    Attributes:
        SinglePoint (-1): Initialize all vectors to a single point.
        RandUniform (0): Initialize vectors using uniform random distribution.
        RandNormal (1): Initialize vectors using normal distribution.
    """

    SinglePoint = "single_point"
    RandUniform = "rand"
    RandNormal = "randn"


def register_buffers(
    module: nn.Module,
    hidden_dim: int,
    projection_dim: int = 1,
    projection_initialization: ProjectionInitialization = ProjectionInitialization.RandUniform,
):
    """Register necessary buffers and parameters in the model.

    Args:
        module (nn.Module):
            module to add buffers and parameters to.
        hidden_dim (int):
            number of outgoing activations.
        projection_dim (int):
            dimension of the neuron projections.
        projection_initialization (ProjectionInitialization):
            initialization scheme to use for the neuron projections.
    """
    match projection_initialization:
        case ProjectionInitialization.SinglePoint:
            neuron_projection = nn.Parameter(torch.zeros((hidden_dim, projection_dim), requires_grad=True, dtype=torch.float32), requires_grad=True)
        case ProjectionInitialization.RandUniform:
            neuron_projection = nn.Parameter(math.sqrt(hidden_dim) * torch.rand((hidden_dim, projection_dim), requires_grad=True, dtype=torch.float32), requires_grad=True)
        case ProjectionInitialization.RandNormal:
            neuron_projection = nn.Parameter(math.sqrt(hidden_dim) * torch.randn((hidden_dim, projection_dim), requires_grad=True, dtype=torch.float32), requires_grad=True)
        case _:
            raise ValueError("unknown projection_initialization scheme.")
    module.register_parameter("neuron_projection", neuron_projection)

    correlation_matrix = torch.zeros((hidden_dim, hidden_dim), requires_grad=False, dtype=torch.float32)
    module.register_buffer("correlation_matrix", correlation_matrix)


def correlation_calculation_hook(
    module: nn.Module,
    input_tensor: torch.Tensor | Tuple[torch.Tensor],
    output_tensor: torch.Tensor | Tuple[torch.Tensor],
):
    """Compute correlation matrix for activations and store in buffer.

    Args:
        module (nn.Module):
            module to add hook to.
        input_tensor (torch.Tensor | Tuple[torch.Tensor]):
            preactivations for the module.
        output_tensor (torch.Tensor | Tuple[torch.Tensor]):
            activations from the module.
    """
    if isinstance(output_tensor, torch.Tensor):
        module.correlation_matrix = batched_corrcoef(output_tensor.transpose(1, 2)).mean(dim=0)
    elif isinstance(output_tensor, list):
        module.correlation_matrix = batched_corrcoef(output_tensor[0].transpose(1, 2))
    else:
        raise TypeError("Unknown output tensor type. Expected torch.Tensor or List[torch.Tensor].")


# @torch.compile(backend="eager")
def _compute_distance_matrix(projections: torch.Tensor, p: float = 2.0, eps: float = 1e-12) -> torch.Tensor:
    """Compute pairwise distance matrix, using a more stable Lp-norm gradient.

    Args:
        projections (torch.Tensor):
            neuron projections.
        p (float):
            order of Lp norm to compute distances. Default is 2.0 (Euclidean norm).
        eps (float):
            small value added for numerical stability of gradients. Default is 1e-12.

    Returns:
        Pairwise distance matrix.
    """
    if p < 1.0:
        raise ValueError("p must be greater than or equal to 1 for Lp norm.")

    col_vec = projections.unsqueeze(1)
    row_vec = projections.unsqueeze(0)
    diff = col_vec - row_vec

    abs_diff = torch.abs(diff)
    pow_diff_p = torch.pow(abs_diff, p)
    sum_pow_diff_p = torch.sum(pow_diff_p, dim=-1)

    if p == 2.0:
        distance = torch.sqrt(sum_pow_diff_p + eps)
    elif p == 1.0:
        distance = torch.sum(abs_diff, dim=-1)
    else:
        sign = torch.sign(sum_pow_diff_p)
        sum_pow_diff_p_clamped = torch.clamp(sum_pow_diff_p, min=eps)
        distance = sign * torch.pow(sum_pow_diff_p_clamped, 1 / p)

    return distance


# @torch.compile(backend="eager")
def _compute_module_loss(module: nn.Module, p: float = 2.0) -> torch.Tensor:
    """Compute the loss for a module.

    Args:
        module (nn.Module):
            module to compute loss for.
        p (float):
            order of Lp norm to compute distances. Default is 2.0 (Euclidean norm).

    Returns:
        DeepReorder loss for the module.
    """
    distance_matrix = _compute_distance_matrix(module.neuron_projection, p)
    target_distance_matrix = 1 - module.correlation_matrix
    loss = (distance_matrix - target_distance_matrix).pow(2).sum().sqrt()
    return loss


# @torch.compile(backend="eager")
def compute_model_loss(model: nn.Module, p: float = 2.0) -> torch.Tensor:
    """Compute the DeepReorder loss for the model.

    Args:
        model (nn.Module):
            model to compute loss for.
        p (float):
            order of Lp norm to compute distances. Default is 2.0 (Euclidean norm).

    Returns:
        DeepReorder loss for the model.
    """
    loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=model.device)
    for layer in model.layers:
        self_attention_loss = _compute_module_loss(layer.self_attn, p)
        mlp_loss = _compute_module_loss(layer.mlp, p)
        loss = loss + self_attention_loss + mlp_loss
    return loss / (2 * len(model.layers))
