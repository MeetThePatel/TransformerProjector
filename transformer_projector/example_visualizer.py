"""Example visualization script for TransformerProjector wrapper model."""

import math

import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import Slider

from transformer_projector import TransformerProjectorModel, TransformerProjectorModelParams


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "vicgalle/alpaca-gpt4"


def visualize(model: TransformerProjectorModel):
    """Visualize a model's neuron projections."""
    initial_layer = 0

    mlp_x_lim, mlp_y_lim, mlp_z_lim = 0.0, 0.0, 0.0
    self_attn_x_lim, self_attn_y_lim, self_attn_z_lim = 0.0, 0.0, 0.0

    for layer in model.hf_model.model.layers:
        mlp_x_lim = max(mlp_x_lim, torch.max(layer.mlp.neuron_projections[:, 0]).item())
        self_attn_x_lim = max(self_attn_x_lim, torch.max(layer.self_attn.neuron_projections[:, 0]).item())
        mlp_y_lim = max(mlp_y_lim, torch.max(layer.mlp.neuron_projections[:, 1]).item())
        self_attn_y_lim = max(self_attn_y_lim, torch.max(layer.self_attn.neuron_projections[:, 1]).item())
        mlp_z_lim = max(mlp_z_lim, torch.max(layer.mlp.neuron_projections[:, 2]).item())
        self_attn_z_lim = max(self_attn_z_lim, torch.max(layer.self_attn.neuron_projections[:, 2]).item())

    mlp_x_lim, mlp_y_lim, mlp_z_lim = math.ceil(mlp_x_lim), math.ceil(mlp_y_lim), math.ceil(mlp_z_lim)
    self_attn_x_lim, self_attn_y_lim, self_attn_z_lim = math.ceil(self_attn_x_lim), math.ceil(self_attn_y_lim), math.ceil(self_attn_z_lim)

    fig = plt.figure(figsize=(16, 8))

    ax_mlp = fig.add_subplot(121, projection="3d")
    ax_mlp.set_title("MLP Projection")
    ax_self_attn = fig.add_subplot(122, projection="3d")
    ax_self_attn.set_title("Self Attention Projection")

    mlp_data = model.hf_model.model.layers[initial_layer].mlp.neuron_projections.numpy()
    mlp_data = mlp_data - mlp_data.mean(axis=0)
    scatter_mlp = ax_mlp.scatter(mlp_data[:, 0], mlp_data[:, 1], mlp_data[:, 2], s=10, c="blue", cmap="viridis")

    self_attn_data = model.hf_model.model.layers[initial_layer].self_attn.neuron_projections.numpy()
    self_attn_data = self_attn_data - self_attn_data.mean(axis=0)
    scatter_self_attn = ax_self_attn.scatter(self_attn_data[:, 0], self_attn_data[:, 1], self_attn_data[:, 2], s=10, c="red", cmap="viridis")

    ax_mlp.set_xlim([-mlp_x_lim, mlp_x_lim])
    ax_mlp.set_ylim([-mlp_y_lim, mlp_y_lim])
    ax_mlp.set_zlim([-mlp_z_lim, mlp_z_lim])

    ax_self_attn.set_xlim([-self_attn_x_lim, self_attn_x_lim])
    ax_self_attn.set_ylim([-self_attn_y_lim, self_attn_y_lim])
    ax_self_attn.set_zlim([-self_attn_z_lim, self_attn_z_lim])

    plt.subplots_adjust(bottom=0.25)

    ax_layer = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_layer.set_zorder(10)

    layer_indices = range(len(model.hf_model.model.layers))
    layer_min = layer_indices[0]
    layer_max = layer_indices[-1]

    layer_slider = Slider(ax=ax_layer, label="Layer", valmin=layer_min, valmax=layer_max, valinit=initial_layer, valstep=1, valfmt="%d")

    def update(_):
        layer_index = int(layer_slider.val)

        mlp_data = model.hf_model.model.layers[layer_index].mlp.neuron_projections.numpy()
        mlp_data = mlp_data - mlp_data.mean(axis=0)
        scatter_mlp._offsets3d = (mlp_data[:, 0], mlp_data[:, 1], mlp_data[:, 2])

        self_attn_data = model.hf_model.model.layers[layer_index].self_attn.neuron_projections.numpy()
        self_attn_data = self_attn_data - self_attn_data.mean(axis=0)
        scatter_self_attn._offsets3d = (self_attn_data[:, 0], self_attn_data[:, 1], self_attn_data[:, 2])

        fig.canvas.draw()

    # Register the update function with the slider
    layer_slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    model = TransformerProjectorModel(MODEL_NAME, TransformerProjectorModelParams())

    model.load_state("./runs/transformer_projector_experimentInstruct3Scaled/checkpoints/step_13000/model.safetensors")
    model.freeze()

    visualize(model)
