import sys
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import accelerate
import matplotlib
import numpy as np
import torch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from transformer_projector.transformer_projector import TransformerProjectorModel, TransformerProjectorModelParams
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

matplotlib.use("QtAgg")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_SYSTEM_PROMPT = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

try:
    MAX_GENERATION_LENGTH = int(os.environ.get("MAX_TOKENS", 500))
except ValueError:
    MAX_GENERATION_LENGTH = 500


class MplCanvas(FigureCanvas):
    """A Matplotlib canvas for visualizing MLP and Self-Attention projections.

    This class creates a FigureCanvas with two 3D subplots: one for visualizing MLP
    projections and another for self-attention projections. It provides methods to
    initialize and update the plots with new data.

    Attributes:
        fig (matplotlib.figure.Figure):
            the main figure.

        mlp_ax (mpl_toolkits.mplot3d.Axes3D):
            the 3D subplot for MLP projections.

        self_attn_ax (mpl_toolkits.mplot3d.Axes3D):
            the 3D subplot for Self-Attention.

        mlp_scatter (mpl_toolkits.mplot3d.art3d.Path3DCollection):
            scatter plot for MLP.

        self_attn_scatter (mpl_toolkits.mplot3d.art3d.Path3DCollection):
            scatter plot for self-attention.
    """

    def __init__(
        self,
        mlp_values: np.ndarray,
        self_attn_values: np.ndarray,
        mlp_sizes: Optional[np.ndarray] = None,
        self_attn_sizes: Optional[np.ndarray] = None,
        mlp_colors: np.ndarray | str | None = None,
        self_attn_colors: np.ndarray | str | None = None,
    ) -> None:
        """Initializes the MplCanvas with MLP and self-attention data.

        Args:
            mlp_values (np.ndarray):
                array of shape (N, 3) representing the 3D coordinates
                of MLP projections.  N is the number of points.

            self_attn_values (np.ndarray):
                array of shape (N, 3) representing the 3D coordinates of
                self-attention projections. M is the number of points.

            mlp_sizes (Optional[np.ndarray]):
                array of shape (N,) containing the sizes of the points
                in the MLP scatter plot.
                If no sizes are provided, it will default to 2.

            self_attn_sizes (Optional[np.ndarray]):
                array of shape (N,) containing the sizes of the points
                in the self-attention scatter plot.
                If no sizes are provided, it will default to 2.

            mlp_colors (np.ndarray | str | None):
                array or a color specification for the MLP scatter plot.
                If array, it should be of shape (N, 3) or (N, 4)
                for RGB or RGBA colors, or a string for a single color
                for all points. Defaults to blue ('b').

            self_attn_colors (np.ndarray | str | None):
                array or a color specification for the self-attention
                scatter plot. If array, it should be of shape (N, 3)
                or (N, 4) for RGB or RGBA, or a string for a single color
                for all points. Defaults to red ('r').
        """

        self.fig = Figure()

        self.mlp_ax: Axes3D = self.fig.add_subplot(121, projection="3d")
        self.mlp_ax.set_title("MLP Projection")
        mlp_sizes = np.ones(len(mlp_values)) * 2 if mlp_sizes is None else mlp_sizes
        mlp_colors = "b" if mlp_colors is None else mlp_colors

        self.self_attn_ax: Axes3D = self.fig.add_subplot(122, projection="3d")
        self.self_attn_ax.set_title("Self Attention Projection")
        self_attn_sizes = np.ones(len(self_attn_values)) * 2 if self_attn_sizes is None else self_attn_sizes
        self_attn_colors = "r" if self_attn_colors is None else self_attn_colors

        super(MplCanvas, self).__init__(self.fig)

        self.mlp_scatter = self.mlp_ax.scatter(
            mlp_values[:, 0],
            mlp_values[:, 1],
            mlp_values[:, 2],
            s=mlp_sizes,
            c=mlp_colors,
        )
        self.self_attn_scatter = self.self_attn_ax.scatter(
            self_attn_values[:, 0],
            self_attn_values[:, 1],
            self_attn_values[:, 2],
            s=self_attn_sizes,
            c=self_attn_colors,
        )

        self.fig.canvas.draw()

    def update_data(
        self,
        mlp_values: np.ndarray,
        self_attn_values: np.ndarray,
        mlp_sizes: Optional[np.ndarray] = None,
        self_attn_sizes: Optional[np.ndarray] = None,
        mlp_colors: np.ndarray | str | None = None,
        self_attn_colors: np.ndarray | str | None = None,
    ) -> None:
        """Updates the data in the scatter plots and redraws the canvas.

        Args:
            mlp_values (np.ndarray):
                array of shape (N, 3) representing the 3D coordinates
                of MLP projections.  N is the number of points.

            self_attn_values (np.ndarray):
                array of shape (N, 3) representing the 3D coordinates of
                self-attention projections. M is the number of points.

            mlp_sizes (Optional[np.ndarray]):
                array of shape (N,) containing the sizes of the points
                in the MLP scatter plot.
                If no sizes are provided, it will default to 2.

            self_attn_sizes (Optional[np.ndarray]):
                array of shape (N,) containing the sizes of the points
                in the self-attention scatter plot.
                If no sizes are provided, it will default to 2.

            mlp_colors (np.ndarray | str | None):
                array or a color specification for the MLP scatter plot.
                If array, it should be of shape (N, 3) or (N, 4)
                for RGB or RGBA colors, or a string for a single color
                for all points. Defaults to blue ('b').

            self_attn_colors (np.ndarray | str | None):
                array or a color specification for the self-attention
                scatter plot. If array, it should be of shape (N, 3)
                or (N, 4) for RGB or RGBA, or a string for a single color
                for all points. Defaults to red ('r').
        """
        mlp_sizes = np.ones(len(mlp_values)) * 2 if mlp_sizes is None else mlp_sizes
        mlp_colors = "b" if mlp_colors is None else mlp_colors
        self_attn_sizes = np.ones(len(self_attn_values)) * 2 if self_attn_sizes is None else self_attn_sizes
        self_attn_colors = "r" if self_attn_colors is None else self_attn_colors

        self.mlp_scatter._offsets3d = (
            mlp_values[:, 0],
            mlp_values[:, 1],
            mlp_values[:, 2],
        )
        self.mlp_scatter.set_sizes(mlp_sizes)
        self.mlp_scatter.set_facecolors(mlp_colors)

        self.self_attn_scatter._offsets3d = (
            self_attn_values[:, 0],
            self_attn_values[:, 1],
            self_attn_values[:, 2],
        )
        self.self_attn_scatter.set_sizes(self_attn_sizes)
        self.self_attn_scatter.set_facecolors(self_attn_colors)

        self.fig.canvas.draw_idle()


class Visualizer(QMainWindow):
    """A visualizer for model projections and activations.

    This class provides an interactive visualization tool for exploring
    MLP and self-attention projections, as well as activations at different
    layers and tokens.

    Attributes:
        is_activation_visualization (bool):
            True when visualizing with activations.

        mpl_canvas (MplCanvas):
            canvas instance for plotting.

        layer_slider (QSlider):
            slider to move between model layers.

        mlp_values (np.ndarray):
            MLP projections.

        self_attn_values (np.ndarray):
            self-attention projections.

        mlp_activation_tensor (np.ndarray):
            MLP activations per layer and token.

        self_attn_activation_tensor (np.ndarray):
            self-attention activations per layer and token.

        tokenizer (AutoTokenizer):
            tokenizer to decode the tokens.

        output_tokens (np.ndarray):
            token-ids of text.

        tokenized_text_indices (List[Tuple[int, int]]):
            index start and end of the text token.
    """

    def __init__(self) -> None:
        """Initializes the Visualizer window."""
        super().__init__()
        self.setWindowTitle(f"Interactive Visualization: {MODEL_NAME}")
        self.setGeometry(100, 100, 1500, 1200)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        self.is_activation_visualization: bool = False
        self.mpl_canvas: Optional[MplCanvas] = None
        self.layer_slider: Optional[QSlider] = None
        self.mlp_values: Optional[np.ndarray] = None
        self.self_attn_values: Optional[np.ndarray] = None
        self.mlp_activation_tensor: Optional[np.ndarray] = None
        self.self_attn_activation_tensor: Optional[np.ndarray] = None
        self.tokenizer: AutoTokenizer = None
        self.output_tokens: Optional[np.ndarray] = None
        self.tokenized_text_indices: List[Tuple[int, int]] = []

    def visualize(self, mlp_projections: torch.Tensor, self_attn_projections: torch.Tensor) -> None:
        """Visualizes MLP and self-attention projections without activations.

        Args:
            mlp_projections (torch.Tensor):
                tensor of shape (num_layers, num_points, 3) representing the
                3D MLP projections for each layer.

            self_attn_projections (torch.Tensor):
                tensor of shape (num_layers, num_points, 3) representing the
                3D self-attention projections for each layer.
        """
        self.is_activation_visualization = False
        initial_layer = 0

        self.mlp_values = mlp_projections.float().cpu().numpy()
        self.self_attn_values = self_attn_projections.float().cpu().numpy()
        n_layers = self.mlp_values.shape[0]

        if self.mpl_canvas is not None:
            self.layout.removeWidget(self.mpl_canvas)
            self.mpl_canvas.deleteLater()
            self.mpl_canvas = None

        self.mpl_canvas = MplCanvas(
            mlp_values=self._demean(self.mlp_values[initial_layer]),
            self_attn_values=self._demean(self.self_attn_values[initial_layer]),
        )
        self.layout.addWidget(self.mpl_canvas)

        if self.layer_slider is not None:
            self.slider_layout.removeWidget(self.layer_slider)
            self.layer_slider.deleteLater()
            self.layer_slider = None

        self.slider_layout = QHBoxLayout()
        self.layer_label = QLabel(f"Layer: {initial_layer}")

        self.layer_slider = QSlider(Qt.Orientation.Horizontal)
        self.layer_slider.setMinimum(0)
        self.layer_slider.setMaximum(n_layers - 1)
        self.layer_slider.setValue(initial_layer)
        self.layer_slider.valueChanged.connect(self._on_slider_change)

        self.slider_layout.addWidget(self.layer_label)
        self.slider_layout.addWidget(self.layer_slider)
        self.layout.addLayout(self.slider_layout)

    def visualize_activations(
        self,
        mlp_projections: torch.Tensor,
        self_attn_projections: torch.Tensor,
        activations: Dict[str, torch.Tensor],
        tokenizer,
        output_tokens: torch.Tensor,
    ) -> None:
        """Visualizes MLP and self-attention projections with activations.

        Args:
            mlp_projections (torch.Tensor):
                tensor of shape (num_layers, num_points, 3) representing the
                3D MLP projections for each layer.

            self_attn_projections (torch.Tensor):
                tensor of shape (num_layers, num_points, 3) representing the
                3D self-attention projections for each layer.

            activations (Dict[str, torch.Tensor]):
                dictionary where keys are strings like "layer_component"
                (e.g., "0_mlp", "1_self_attn") and values are tensors representing
                the activations for that layer and component.

            tokenizer (AutoTokenizer):
                tokenizer used to process the input text.

            output_tokens (torch.Tensor):
                tensor of token-ids representing the output tokens.

        Raises:
             ValueError: If output_tokens is not of type np.ndarray.
        """
        if not isinstance(output_tokens, torch.Tensor):
            raise ValueError(f"output_tokens must be a torch.Tensor, got {type(output_tokens)}")

        self.is_activation_visualization = True
        initial_layer, initial_token = 0, 0
        n_layers = mlp_projections.shape[0]

        self.mlp_values = mlp_projections.float().cpu().numpy()
        self.self_attn_values = self_attn_projections.float().cpu().numpy()
        (
            self.mlp_activation_tensor,
            self.self_attn_activation_tensor,
        ) = self._collect_activations_into_tensors(activations)

        self.tokenizer = tokenizer
        self.output_tokens = output_tokens

        mlp_activations = self.mlp_activation_tensor[initial_layer]
        self_attn_activations = self.self_attn_activation_tensor[initial_layer]

        if self.mpl_canvas is not None:
            self.layout.removeWidget(self.mpl_canvas)
            self.mpl_canvas.deleteLater()
            self.mpl_canvas = None

        self.help_text = QLabel(
            "For the 3D scatter plots, size of the point is based on the absolute "
            "value of the activation, and color is based on sign. For example, "
            "values near zero will be small points with neutral color, large "
            "values will be large green points, and large negative values will be "
            "large red points."
        )
        self.layout.addWidget(self.help_text)

        self.mpl_canvas = MplCanvas(
            mlp_values=self._demean(self.mlp_values[initial_layer]),
            self_attn_values=self._demean(self.self_attn_values[initial_layer]),
            mlp_sizes=self._convert_to_sizes(mlp_activations[initial_token]),
            self_attn_sizes=self._convert_to_sizes(self_attn_activations[initial_token]),
            mlp_colors=self._convert_to_color(mlp_activations[initial_token]),
            self_attn_colors=self._convert_to_color(self_attn_activations[initial_token]),
        )
        self.mpl_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.mpl_canvas.setMinimumHeight(500)
        self.layout.addWidget(self.mpl_canvas)

        if self.layer_slider is not None:
            self.layer_slider_layout.removeWidget(self.layer_slider)
            self.layer_slider.deleteLater()
            self.layer_slider = None

        self.layer_slider_layout = QHBoxLayout()
        self.layer_label = QLabel(f"Layer: {initial_layer}")
        self.layer_slider = QSlider(Qt.Orientation.Horizontal)
        self.layer_slider.setMinimum(0)
        self.layer_slider.setMaximum(n_layers - 1)
        self.layer_slider.setValue(initial_layer)
        self.layer_slider_layout.addWidget(self.layer_label)
        self.layer_slider_layout.addWidget(self.layer_slider)
        self.layout.addLayout(self.layer_slider_layout)

        if hasattr(self, "activation_slider"):
            self.activation_slider_layout.removeWidget(self.activation_slider)
            self.activation_slider.deleteLater()

        self.activation_slider_layout = QHBoxLayout()
        self.activation_label = QLabel(f"Activation: {initial_token}")
        self.activation_slider = QSlider(Qt.Orientation.Horizontal)
        self.activation_slider.setMinimum(0)
        self.activation_slider.setMaximum(output_tokens.shape[0] - 1)
        self.activation_slider.setValue(initial_token)
        self.activation_slider_layout.addWidget(self.activation_label)
        self.activation_slider_layout.addWidget(self.activation_slider)
        self.layout.addLayout(self.activation_slider_layout)

        self.layer_slider.valueChanged.connect(self._on_slider_change)
        self.activation_slider.valueChanged.connect(self._on_slider_change)

        self.output_text_area = QTextEdit()
        self.output_text_area.setFontPointSize(18.0)
        self.output_text_area.setReadOnly(True)
        self.output_text_area.setText(tokenizer.decode(output_tokens, skip_special_tokens=True))
        self.layout.addWidget(self.output_text_area)

        self._compute_token_boundaries()
        self._highlight_current_token(initial_token)

        self.layout.setStretchFactor(self.help_text, 1)
        self.layout.setStretchFactor(self.mpl_canvas, 10)
        self.layout.setStretchFactor(self.output_text_area, 2)

        self.show()

    def _on_slider_change(self) -> None:
        """Handles slider value changes and updates the visualization."""

        if not (self.layer_slider and self.mpl_canvas and self.mlp_values is not None and self.self_attn_values is not None):
            return

        if self.is_activation_visualization:
            if not (self.activation_slider and self.mlp_activation_tensor is not None and self.self_attn_activation_tensor is not None):
                return

            layer_index = self.layer_slider.value()
            activation_index = self.activation_slider.value()

            self.layer_label.setText(f"Layer: {layer_index}")
            self.activation_label.setText(f"Activation: {activation_index}")

            mlp_activations = self.mlp_activation_tensor[layer_index]
            self_attn_activations = self.self_attn_activation_tensor[layer_index]

            self.mpl_canvas.update_data(
                mlp_values=self._demean(self.mlp_values[layer_index]),
                self_attn_values=self._demean(self.self_attn_values[layer_index]),
                mlp_sizes=self._convert_to_sizes(mlp_activations[activation_index]),
                self_attn_sizes=self._convert_to_sizes(self_attn_activations[activation_index]),
                mlp_colors=self._convert_to_color(mlp_activations[activation_index]),
                self_attn_colors=self._convert_to_color(self_attn_activations[activation_index]),
            )

            self._highlight_current_token(activation_index)
        else:
            layer_index = self.layer_slider.value()
            self.layer_label.setText(f"Layer: {layer_index}")
            self.mpl_canvas.update_data(
                mlp_values=self._demean(self.mlp_values[layer_index]),
                self_attn_values=self._demean(self.self_attn_values[layer_index]),
            )

    def _highlight_current_token(self, token_index: int) -> None:
        """Highlights the currently selected token in the output text area.

        Args:
            token_index (int):
                index of the token to highlight.
        """
        if not self.tokenized_text_indices:
            self._compute_token_boundaries()

        cursor = self.output_text_area.textCursor()
        cursor.select(cursor.SelectionType.Document)
        format = cursor.charFormat()
        format.setFontUnderline(False)
        format.setBackground(Qt.GlobalColor.transparent)
        cursor.setCharFormat(format)
        cursor.clearSelection()

        if 0 <= token_index < len(self.tokenized_text_indices):
            start_pos, end_pos = self.tokenized_text_indices[token_index]

            cursor.setPosition(start_pos)
            cursor.setPosition(end_pos, cursor.MoveMode.KeepAnchor)

            highlight_format = cursor.charFormat()
            highlight_format.setFontUnderline(True)
            highlight_format.setBackground(Qt.GlobalColor.yellow)
            cursor.setCharFormat(highlight_format)

    def _compute_token_boundaries(self) -> None:
        """Computes the start and end positions of each token in the text."""

        if not (self.output_tokens is not None and self.tokenizer):
            return
        text = self.output_text_area.toPlainText()
        self.tokenized_text_indices = []
        pos = 0
        for token_id in self.output_tokens:
            token_string = self.tokenizer.decode([token_id], skip_special_tokens=True)
            if token_string:
                start_pos = text.find(token_string, pos)
                if start_pos != -1:
                    end_pos = start_pos + len(token_string)
                    self.tokenized_text_indices.append((start_pos, end_pos))
                    pos = end_pos
                else:
                    estimated_start = pos
                    estimated_end = estimated_start + max(1, len(token_string))
                    self.tokenized_text_indices.append((estimated_start, estimated_end))
                    pos = estimated_end

    def _convert_to_sizes(self, x: np.ndarray) -> np.ndarray:
        """Converts activation values to point sizes for visualization.

        Args:
            x (np.ndarray):
                array of activation values.

        Returns:
            array of point sizes.
        """
        tmp = np.abs(x)
        if np.min(tmp) == np.max(tmp):
            return np.zeros_like(tmp, dtype=np.float32)
        else:
            return 20 * (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp)) + 1.5

    def _convert_to_color(self, x: np.ndarray) -> np.ndarray:
        """Converts activation values to colors for visualization.

        Args:
            x (np.ndarray):
                array of activation values.

        Returns:
            array of colors (RGBA).
        """
        if np.max(x) == np.min(x):
            return np.array([[0.5, 0.5, 0.5, 1.0]] * len(x))

        normalized_x = (x - np.min(x)) / (np.max(x) - np.min(x))
        colormap = matplotlib.colormaps.get_cmap("PiYG")
        return colormap(normalized_x)

    def _demean(self, x: np.ndarray) -> np.ndarray:
        """Centers array around origin.

        Args:
            x (np.ndarray):
                input array.

        Returns:
            centered array.
        """
        return x - x.mean(axis=0)

    def _collect_activations_into_tensors(self, activations: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """Collects activations into separate MLP and Self-Attention tensors.

        Args:
            activations (Dict[str, torch.Tensor]):
                dictionary where keys are strings like "layer_component"
                (e.g., "0_mlp", "1_self_attn") and values are tensors representing
                the activations for that layer and component.

        Returns:
            A tuple containing two arrays:
                - mlp_tensors: (num_layers, ..., activation_dim)
                - self_attn_tensors: (num_layers, ..., activation_dim)
        """
        layer_numbers = [int(key.split("_")[0]) for key in activations.keys()]
        n_layers = max(layer_numbers) + 1

        mlp_tensors = torch.zeros((n_layers, *activations["0_mlp"].shape))
        self_attn_tensors = torch.zeros((n_layers, *activations["0_self_attn"].shape))

        for key, tensor in activations.items():
            layer_number = int(key.split("_")[0])
            module_type = key.split("_")[1]

            if module_type == "mlp":
                mlp_tensors[layer_number] = tensor
            else:
                self_attn_tensors[layer_number] = tensor

        return (
            mlp_tensors.float().cpu().numpy(),
            self_attn_tensors.float().cpu().numpy(),
        )

    @staticmethod
    def _register_activation_hooks(model: TransformerProjectorModel, activations: Dict[str, List]) -> List[torch.utils.hooks.RemovableHandle]:
        """Registers hooks to collect activations during a forward pass.

        Args:
            model (TransformerProjectorModel):
                model instance.
            activations (Dict[str, List]):
                dictionary to store the collected activations.
                Keys should be strings like "layer_component" (e.g., "0_mlp").

        Returns:
            A list of hook handles that can be used to remove the hooks later.
        """

        def _generate_hook(key: str):
            def _hook(_module, _input, output: torch.Tensor | Tuple[torch.Tensor]) -> None:
                if isinstance(output, torch.Tensor):
                    activations[key].append(output)
                elif isinstance(output, tuple):
                    activations[key].append(output[0])

            return _hook

        hooks: List[torch.utils.hooks.RemovableHandle] = []
        for layer_idx, layer in enumerate(model.hf_model.model.layers):
            for component in ["mlp", "self_attn"]:
                hook = layer.__getattr__(component).register_forward_hook(_generate_hook(f"{layer_idx}_{component}"))
                hooks.append(hook)
        return hooks


class MessageConstructor:
    """Handles construction of prompts and messages for the model."""

    @staticmethod
    def construct_messages(
        instruction: str,
        input_text: str = "",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> List[List[Dict[str, str]]]:
        """Construct a message for model inference.

        Args:
            instruction (str):
                instruction for the model.
            input_text (str):
                optional auxilliary information.
            system_prompt (str):
                system prompt to use for the model.
                Defaults to DEFAULT_SYSTEM_PROMPT.
        """
        return MessageConstructor.construct_messages_batch({"instruction": [instruction], "input": [input_text]}, system_prompt)

    @staticmethod
    def construct_messages_batch(batch: Dict[str, List[str]], system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> List[List[Dict[str, str]]]:
        """Prepare batch of messages for model inference.

        Args:
            batch (Dict[str, List[str]]):
                batch of messages to send to the model.
            system_prompt (str):
                system prompt to use for the model.
                Defaults to DEFAULT_SYSTEM_PROMPT.
        """
        batch_messages = []
        instructions = batch["instruction"]
        inputs = batch["input"]

        for i in range(len(instructions)):
            instruction = instructions[i]
            input_text = inputs[i]

            messages = [{"role": "system", "content": system_prompt}]
            if input_text:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Instruction: {instruction}\nInput: {input_text}",
                    }
                )
            else:
                messages.append({"role": "user", "content": f"Instruction: {instruction}"})

            batch_messages.append(messages)

        return batch_messages


class TransformerProjectorApp(QWidget):
    """Main application for TransformerProjector visualizations."""

    def __init__(self):
        """Initialize the application."""
        super().__init__()

        self.setWindowTitle("TransformerProjector")
        self.setGeometry(100, 100, 800, 600)

        self.loaded_model = None
        self.loaded_tokenizer = None

        self.accelerator = accelerate.Accelerator()
        self.device = self.accelerator.device

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._add_model_selection_section(main_layout)
        self._add_file_selection_section(main_layout)
        self._add_control_buttons_section(main_layout)
        self._add_separator(main_layout)
        self._add_inference_section(main_layout)
        main_layout.addStretch()
        self.setLayout(main_layout)

    def _add_model_selection_section(self, parent_layout: QVBoxLayout) -> None:
        """Add model selection section to the UI."""
        hf_model_layout = QHBoxLayout()
        hf_model_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.hf_model_label = QLabel("HF Model:")
        hf_model_layout.addWidget(self.hf_model_label)
        hf_model_layout.setAlignment(self.hf_model_label, Qt.AlignmentFlag.AlignTop)

        self.hf_model_text_box = QLineEdit()
        hf_model_layout.addWidget(self.hf_model_text_box)
        hf_model_layout.setAlignment(self.hf_model_text_box, Qt.AlignmentFlag.AlignTop)
        self.hf_model_text_box.setText(MODEL_NAME)

        parent_layout.addLayout(hf_model_layout)

    def _add_file_selection_section(self, parent_layout: QVBoxLayout) -> None:
        """Add file selection section to the UI."""
        file_selector_layout = QHBoxLayout()
        file_selector_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.file_path_label = QLabel("Model SafeTensors file path:")
        file_selector_layout.addWidget(self.file_path_label)
        file_selector_layout.setAlignment(self.file_path_label, Qt.AlignmentFlag.AlignTop)

        self.file_path_line_edit = QLineEdit()
        self.file_path_line_edit.setReadOnly(True)
        file_selector_layout.addWidget(self.file_path_line_edit)
        file_selector_layout.setAlignment(self.file_path_line_edit, Qt.AlignmentFlag.AlignTop)

        self.file_button = QPushButton("Select File")
        self.file_button.clicked.connect(self.open_file_dialog)
        file_selector_layout.addWidget(self.file_button)
        file_selector_layout.setAlignment(self.file_button, Qt.AlignmentFlag.AlignTop)

        parent_layout.addLayout(file_selector_layout)

    def _add_control_buttons_section(self, parent_layout: QVBoxLayout) -> None:
        """Add control buttons section to the UI."""
        load_visualize_layout = QHBoxLayout()
        load_visualize_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_model)
        load_visualize_layout.addWidget(self.load_button)

        self.visualize_button = QPushButton("Visualize")
        self.visualize_button.clicked.connect(self.visualize_model)
        load_visualize_layout.addWidget(self.visualize_button)

        parent_layout.addLayout(load_visualize_layout)

    def _add_separator(self, parent_layout: QVBoxLayout) -> None:
        """Add a horizontal separator line to the UI."""
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        parent_layout.addWidget(separator)

    def _add_inference_section(self, parent_layout: QVBoxLayout) -> None:
        """Add inference section to the UI."""
        self.inference_widget = QWidget()
        self.inference_widget.setVisible(False)
        self.inference_layout = QVBoxLayout(self.inference_widget)
        self.inference_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.inference_layout.addWidget(QLabel("Inference Inputs:", alignment=Qt.AlignmentFlag.AlignTop))
        self._add_system_prompt_section()
        self._add_instruction_section()
        self._add_content_section()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_inference)
        self.inference_layout.addWidget(self.run_button, alignment=Qt.AlignmentFlag.AlignTop)

        parent_layout.addWidget(self.inference_widget)

    def _add_system_prompt_section(self) -> None:
        """Add system prompt section to the inference UI."""
        self.system_prompt_label = QLabel("System Prompt:")
        self.inference_layout.addWidget(self.system_prompt_label, alignment=Qt.AlignmentFlag.AlignTop)

        self.system_prompt_text_box = QLineEdit()
        self.system_prompt_text_box.setText(DEFAULT_SYSTEM_PROMPT)
        self.inference_layout.addWidget(self.system_prompt_text_box, alignment=Qt.AlignmentFlag.AlignTop)

    def _add_instruction_section(self) -> None:
        """Add instruction section to the inference UI."""
        self.instruction_label = QLabel("Instruction:")
        self.inference_layout.addWidget(self.instruction_label, alignment=Qt.AlignmentFlag.AlignTop)

        self.instruction_text_box = QTextEdit()
        self.inference_layout.addWidget(self.instruction_text_box, alignment=Qt.AlignmentFlag.AlignTop)

    def _add_content_section(self) -> None:
        """Add content section to the inference UI."""
        self.content_label = QLabel("Content:")
        self.inference_layout.addWidget(self.content_label, alignment=Qt.AlignmentFlag.AlignTop)

        self.content_text_box = QTextEdit()
        self.inference_layout.addWidget(self.content_text_box, alignment=Qt.AlignmentFlag.AlignTop)

    def open_file_dialog(self) -> None:
        """Open a file dialog to select a SafeTensors file."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select SafeTensors File")
        if file_path:
            self.file_path_line_edit.setText(file_path)

    def load_model(self) -> None:
        """Load the model from the selected file."""
        model_name = self.hf_model_text_box.text()
        safetensors_path = self.file_path_line_edit.text()

        if not safetensors_path:
            QMessageBox.warning(self, "Warning", "Please select a SafeTensors file path.")
            return

        try:
            self._load_model_and_tokenizer(model_name, safetensors_path)
            self.inference_widget.setVisible(True)
            QMessageBox.information(self, "Success", "Model loaded successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")
            self.inference_widget.setVisible(False)
            self.loaded_model = None

    def _load_model_and_tokenizer(self, model_name: str, safetensors_path: str) -> None:
        """Load and prepare the model and tokenizer."""
        self.loaded_model = TransformerProjectorModel(model_name, TransformerProjectorModelParams())
        self.loaded_model.load_state(safetensors_path)
        self.loaded_model.freeze()
        self.loaded_model.hf_model = torch.compile(self.loaded_model.hf_model, backend="eager")
        self.loaded_model.hf_model = self.accelerator.prepare(self.loaded_model.hf_model)

        self.loaded_tokenizer = AutoTokenizer.from_pretrained(model_name)

    def visualize_model(self) -> None:
        """Visualize the loaded model."""
        if self.loaded_model is None:
            QMessageBox.warning(
                self,
                "Warning",
                "No model loaded. Please load a model first before visualizing.",
            )
            return

        try:
            n_layers = len(self.loaded_model.hf_model.model.layers)
            mlp_projections = torch.stack(
                [self.loaded_model.hf_model.model.layers[layer_idx].mlp.neuron_projections for layer_idx in range(n_layers)]
            )
            self_attn_projections = torch.stack(
                [self.loaded_model.hf_model.model.layers[layer_idx].self_attn.neuron_projections for layer_idx in range(n_layers)]
            )
            self.visualizer = Visualizer()
            self.visualizer.visualize(mlp_projections, self_attn_projections)
            self.visualizer.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to visualize model:\n{e}")

    def run_inference(self) -> None:
        """Run inference with the loaded model."""
        if self.loaded_model is None:
            QMessageBox.warning(self, "Warning", "Model not loaded. Please load a model first.")
            return

        system_prompt = self.system_prompt_text_box.text()
        instruction = self.instruction_text_box.toPlainText()
        content = self.content_text_box.toPlainText()
        if not instruction:
            QMessageBox.warning(self, "Warning", "Instruction cannot be empty.")
            return
        constructed_message = MessageConstructor.construct_messages(instruction, content, system_prompt)

        tokenized_inputs = self._tokenize_input_messages(constructed_message)
        tokenized_inputs.to(self.loaded_model.hf_model.device)

        activations = defaultdict(list)
        hooks = Visualizer._register_activation_hooks(self.loaded_model, activations)

        with torch.no_grad():
            output_tokens = self.loaded_model.hf_model.generate(
                tokenized_inputs.input_ids,
                attention_mask=tokenized_inputs.attention_mask,
                do_sample=True,
                pad_token_id=self.loaded_tokenizer.pad_token_id,
                eos_token_id=self.loaded_tokenizer.eos_token_id,
                max_new_tokens=MAX_GENERATION_LENGTH,
                temperature=0.7,
                top_p=0.9,
            )

        for hook in hooks:
            hook.remove()

        for key in activations.keys():
            activations[key] = torch.cat([tensor.squeeze(0) for tensor in activations[key]], dim=0)

        n_layers = len(self.loaded_model.hf_model.model.layers)
        mlp_projections = torch.stack([self.loaded_model.hf_model.model.layers[layer_idx].mlp.neuron_projections for layer_idx in range(n_layers)])
        self_attn_projections = torch.stack(
            [self.loaded_model.hf_model.model.layers[layer_idx].self_attn.neuron_projections for layer_idx in range(n_layers)]
        )

        input_token_length = tokenized_inputs.input_ids.shape[1]
        generated_output_tokens = output_tokens[0][input_token_length:]

        self.interactive_visualization = Visualizer()
        self.interactive_visualization.visualize_activations(
            mlp_projections=mlp_projections,
            self_attn_projections=self_attn_projections,
            activations=activations,
            tokenizer=self.loaded_tokenizer,
            output_tokens=generated_output_tokens,
        )
        self.interactive_visualization.show()

    def _tokenize_input_messages(self, messages: List[List[Dict[str, str]]]) -> BatchEncoding:
        """Tokenize input messages for the model."""
        return self.loaded_tokenizer(
            [self.loaded_tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages],
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=self.loaded_model.hf_model.config.max_position_embeddings,
            return_tensors="pt",
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TransformerProjectorApp()
    window.show()
    sys.exit(app.exec())
