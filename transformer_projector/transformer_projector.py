"""TransformerProjector is an experiment for projecting transformer components using activation correlation."""

import dataclasses
import math
from enum import Enum
from typing import List, Optional

import safetensors
import torch
from torch import nn
from transformers import PreTrainedModel, AutoModelForCausalLM


class ProjectionInitialization(Enum):
    """Initialization scheme to use for neuron projections.

    Attributes:
        SinglePoint: Initialize all vectors to a single point.

        RandUniform: Initialize vectors using uniform random distribution.

        RandNormal: Initialize vectors using normal distribution.
    """

    SinglePoint = "single_point"
    RandUniform = "rand"
    RandNormal = "randn"


@dataclasses.dataclass
class TransformerProjectorModelParams:
    """Parameters for the TransformerProjector model.

    Attributes:
        component_list (Optional[List[str]]):
            List of component names to apply TransformerProjector to.
            Defaults to ["self_attn", "mlp"].

        hidden_dim (Optional[int]):
            Hidden dimension of the model.
            Defaults to None, which will use `model.config.hidden_size`.

        projection_dim (Optional[int]):
            Dimension of the neuron projections.
            Defaults to 3.

        projection_init_scheme (Optional[`ProjectionInitialization`]):
            Initialization scheme for the neuron projections.
            Defaults to `ProjectionInitialization.RandUniform`.

        norm_order (Optional[float]):
            Order of the Lp norm used for distance calculations.
            Defaults to 2.0 (Euclidean norm).
    """

    component_list: Optional[List[str]] = dataclasses.field(default_factory=lambda: ["self_attn", "mlp"])
    hidden_dim: Optional[int] = None
    projection_dim: Optional[int] = 3
    projection_init_scheme: Optional[ProjectionInitialization] = ProjectionInitialization.RandUniform
    norm_order: Optional[float] = 2.0


class TransformerProjectorModel:
    """TransformerProjector model."""

    def __init__(self, model: str, params: TransformerProjectorModelParams):
        """Initialize the TransformerProjector model."""
        self.hf_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model, torch_dtype="bfloat16", device_map="auto")
        # Freeze all weights from the HuggingFace model. We are only interested in training/loading weights of neuron_projections.
        self.freeze()

        self.params = params
        if self.params.hidden_dim is None:
            self.params.hidden_dim = self.hf_model.config.hidden_size

        # Register necessary buffers/parameters on components.
        for component in self.params.component_list:
            self.register_buffers(component)
            self.register_parameters(component)

    def load_state(self, model_state_path: str):
        """Load state from trained model.

        Args:
            model_state_path (str):
                path to trained model state SafeTensors file.
        """
        state_dict = self.hf_model.state_dict()

        with safetensors.safe_open(model_state_path, framework="pt") as f:
            for k in f.keys():
                if "neuron_projection" in k:
                    target_key = "model." + k[k.find("layers") :]
                    if target_key not in state_dict:
                        raise KeyError(f"Key {target_key} not found in model state dict.")
                    tensor = f.get_tensor(k)
                    state_dict[target_key].copy_(tensor)

    def freeze(self):
        """Freeze the state of the model.

        This operation should be done in the following scenarios:
        - immediately after loading weights.
        - immediately before training.
        - immediately after training.
        """
        self.hf_model.model.eval()
        for param in self.hf_model.parameters():
            param.requires_grad = False

    def register_parameters(self, component: str):
        """Register neuron projection parameters in the model.

        Args:
            component (str):
                component to add parameters to.
        """
        for layer in self.hf_model.model.layers:
            self._register_neuron_projections_parameter(layer.__getattr__(component))

    def register_buffers(self, component: str):
        """Register correlation matrix buffer in the model.

        Args:
            component (str):
                component to add buffer to.
        """
        for layer in self.hf_model.model.layers:
            self._register_correlation_matrix_buffer(layer.__getattr__(component))

    def register_hooks(self, component: str):
        """Register correlation calculation hooks in the model.

        Args:
            component (str):
                component to add hook to.
        """

        def _correlation_calculation_hook(module: nn.Module, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
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
                module.correlation_matrix = torch.vmap(torch.corrcoef)(output_tensor.transpose(1, 2)).mean(dim=0)
            elif isinstance(output_tensor, tuple):
                module.correlation_matrix = torch.vmap(torch.corrcoef)(output_tensor[0].transpose(1, 2))
            else:
                raise TypeError(f"Unknown output tensor type: {type(output_tensor)}. Expected torch.Tensor or Tuple[torch.Tensor].")

        hooks = []
        for layer in self.hf_model.model.layers:
            hook = layer.__getattr__(component).register_forward_hook(_correlation_calculation_hook)
            hooks.append(hook)
        self.hooks = hooks

    def unregister_hooks(self):
        """Unregister correlation calculation hooks in the model."""
        for hook in self.hooks:
            hook.remove()

    def _register_neuron_projections_parameter(self, module: nn.Module):
        """Register neuron projection parameter in the model.

        Args:
            module (nn.Module):
                module to add parameter to.
        """
        match self.params.projection_init_scheme:
            case ProjectionInitialization.SinglePoint:
                neuron_projections = nn.Parameter(
                    torch.zeros((self.params.hidden_dim, self.params.projection_dim), requires_grad=True, dtype=torch.float32),
                    requires_grad=True,
                )
            case ProjectionInitialization.RandUniform:
                neuron_projections = nn.Parameter(
                    math.sqrt(self.params.hidden_dim) * torch.rand((self.params.hidden_dim, self.params.projection_dim), requires_grad=True, dtype=torch.float32),
                    requires_grad=True,
                )
            case ProjectionInitialization.RandNormal:
                neuron_projections = nn.Parameter(
                    math.sqrt(self.params.hidden_dim) * torch.randn((self.params.hidden_dim, self.params.projection_dim), requires_grad=True, dtype=torch.float32),
                    requires_grad=True,
                )
            case _:
                raise ValueError("Unknown projection_initialization scheme.")
        module.register_parameter("neuron_projections", neuron_projections)

    def _register_correlation_matrix_buffer(self, module: nn.Module):
        """Register buffer for correlation matrix in the model.

        Args:
            module (nn.Module):
                module to add buffer to.
        """
        correlation_matrix = torch.zeros((self.params.hidden_dim, self.params.hidden_dim), requires_grad=False, dtype=torch.float32)
        module.register_buffer("correlation_matrix", correlation_matrix)

    def _compute_distance_matrix(self, module: nn.Module, eps: float = 1e-12) -> torch.Tensor:
        """Compute pairwise distance matrix, using a more stable Lp-norm gradient.

        Args:
            module (nn.Module):
                module to compute distance matrix for.

            eps (float):
                small value added for numerical stability of gradients.
                Defaults to 1e-12.

        Returns:
            Pairwise distance matrix.
        """
        if self.params.norm_order < 1.0:
            raise ValueError("norm_order must be greater than or equal to 1 for Lp norm.")

        col_vec = module.neuron_projections.unsqueeze(1)
        row_vec = module.neuron_projections.unsqueeze(0)
        diff = col_vec - row_vec

        abs_diff = torch.abs(diff)
        pow_diff_p = torch.pow(abs_diff, self.params.norm_order)
        sum_pow_diff_p = torch.sum(pow_diff_p, dim=-1)

        if self.params.norm_order == 2.0:
            distance = torch.sqrt(sum_pow_diff_p + eps)
        elif self.params.norm_order == 1.0:
            distance = torch.sum(abs_diff, dim=-1)
        else:
            sign = torch.sign(sum_pow_diff_p)
            sum_pow_diff_p_clamped = torch.clamp(sum_pow_diff_p, min=eps)
            distance = sign * torch.pow(sum_pow_diff_p_clamped, 1 / self.params.norm_order)

        return distance

    def _compute_module_loss(self, module: nn.Module) -> torch.Tensor:
        """Compute the loss for a module.

        Args:
            module (nn.Module):
                module to compute loss for.

            p (float):
                order of Lp norm to compute distances.
                Defaults to 2.0 (Euclidean norm).

        Returns:
            TransformerProjector loss for the module.
        """
        distance_matrix = self._compute_distance_matrix(module)
        target_distance_matrix = 1 - module.correlation_matrix
        loss = (distance_matrix - target_distance_matrix).pow(2).sum().sqrt()
        return loss

    def _compute_model_loss(self) -> torch.Tensor:
        """Compute the TransformerProjector loss for the model.

        Returns:
            TransformerProjector loss for the model.
        """
        loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=self.hf_model.model.device)
        for layer in self.hf_model.model.layers:
            for component in self.params.component_list:
                component_loss = self._compute_module_loss(layer.__getattr__(component))
                loss = loss + component_loss
        return loss / (len(self.params.component_list) * len(self.hf_model.model.layers))
