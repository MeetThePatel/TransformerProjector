from typing import Tuple, Callable

import torch
from torch import nn

batched_corrcoef = torch.func.vmap(torch.corrcoef)


def register_buffers(model: nn.Module, device: torch.device | str = "cpu"):
    """Register necessary buffers and parameters in the model.

    Args:
        model (nn.Module):
            model to add buffers and parameters to.
        device (torch.device or str):
            device to store buffers on.
    """
    for idx, layer in enumerate(model.layers):
        idx: int
        layer: nn.Module
        linear_positions = nn.Parameter(torch.linspace(0, 1, layer.hidden_size, dtype=torch.float32, device=device))
        activation_correlations = torch.zeros((layer.hidden_size, layer.hidden_size), dtype=torch.float32, device=device)
        layer.register_parameter("linear_positions", linear_positions)
        layer.register_buffer("activation_correlations", activation_correlations)


def _save_activation_hook(_module_name: str) -> Callable[[nn.Module, Tuple[torch.Tensor], Tuple[torch.Tensor]], None]:
    """Generate hook to attach to modules.

    Args:
        _module_name (str):
            name of the module.

    Returns:
        A hook that computes and stores correlation matrices for activations of a module.
    """

    def hook(module: nn.Module, _input_tensor: Tuple[torch.Tensor], output_tensor: Tuple[torch.Tensor]):
        """Compute correlation matrix for activations.

        Args:
            module (nn.Module):
                module that is calling this hook
            _input_tensor (Tuple[torch.Tensor]):
                input that is fed into the module
            output_tensor (Tuple[torch.Tensor]):
                output from the module
        """
        correlation_matrix: torch.Tensor = batched_corrcoef(torch.transpose(output_tensor[0], 1, 2))
        module.activation_correlations = torch.mean(correlation_matrix, dim=0)

    return hook


def register_hooks(model: nn.Module):
    """Register necessary hooks in the model.

    Args:
        model (nn.Module):
            model to add hooks to.
    """
    for idx, layer in model.layers.named_children():
        layer.register_forward_hook(_save_activation_hook(f"{layer.__class__.__name__}_{idx}"))
