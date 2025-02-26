import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import accelerate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import *
from rich import print
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from deep_reorder import DeepReorderModel, DeepReorderModelParams

matplotlib.use("QtAgg")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_SYSTEM_PROMPT = "Below is an instruction that describes a task. Write a response that appropriately completes the request."


class MplCanvas(FigureCanvas):
    def __init__(
        self,
        mlp_values: np.array,
        self_attn_values: np.array,
        mlp_sizes: Optional[np.array] = None,
        self_attn_sizes: Optional[np.array] = None,
        mlp_colors: Optional[np.array] = None,
        self_attn_colors: Optional[np.array] = None,
    ):
        self.fig = Figure()

        self.mlp_ax = self.fig.add_subplot(121, projection="3d")
        self.mlp_ax.set_title("MLP Projection")
        mlp_sizes = np.ones(len(mlp_values)) * 2 if mlp_sizes is None else mlp_sizes
        mlp_colors = "b" if mlp_colors is None else mlp_colors

        self.self_attn_ax = self.fig.add_subplot(122, projection="3d")
        self.self_attn_ax.set_title("Self Attention Projection")
        self_attn_sizes = np.ones(len(self_attn_values)) * 2 if self_attn_sizes is None else self_attn_sizes
        self_attn_colors = "r" if self_attn_colors is None else self_attn_colors

        super(MplCanvas, self).__init__(self.fig)

        self.mlp_scatter = self.mlp_ax.scatter(mlp_values[:, 0], mlp_values[:, 1], mlp_values[:, 2], s=mlp_sizes, c=mlp_colors)
        self.self_attn_scatter = self.self_attn_ax.scatter(self_attn_values[:, 0], self_attn_values[:, 1], self_attn_values[:, 2], s=self_attn_sizes, c=self_attn_colors)

        self.fig.canvas.draw()

    def update_data(
        self,
        mlp_values: np.array,
        self_attn_values: np.array,
        mlp_sizes: Optional[np.array] = None,
        self_attn_sizes: Optional[np.array] = None,
        mlp_colors: Optional[np.array] = None,
        self_attn_colors: Optional[np.array] = None,
    ):
        mlp_sizes = np.ones(len(mlp_values)) * 2 if mlp_sizes is None else mlp_sizes
        mlp_colors = "b" if mlp_colors is None else mlp_colors
        self_attn_sizes = np.ones(len(self_attn_values)) * 2 if self_attn_sizes is None else self_attn_sizes
        self_attn_colors = "r" if self_attn_colors is None else self_attn_colors

        self.mlp_scatter._offsets3d = (mlp_values[:, 0], mlp_values[:, 1], mlp_values[:, 2])
        self.mlp_scatter.set_sizes(mlp_sizes)
        self.mlp_scatter.set_facecolors(mlp_colors)

        self.self_attn_scatter._offsets3d = (self_attn_values[:, 0], self_attn_values[:, 1], self_attn_values[:, 2])
        self.self_attn_scatter.set_sizes(self_attn_sizes)
        self.self_attn_scatter.set_facecolors(self_attn_colors)

        self.fig.canvas.draw_idle()


class NewVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Interactive Visualization: {MODEL_NAME}")
        self.setGeometry(100, 100, 1500, 1200)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

    def visualize(self, mlp_projections: torch.Tensor, self_attn_projections: torch.Tensor):
        self.is_activation_visualization = False
        initial_layer = 0

        self.mlp_values = mlp_projections.float().cpu().numpy()
        self.self_attn_values = self_attn_projections.float().cpu().numpy()
        n_layers = self.mlp_values.shape[0]

        self.mpl_canvas = MplCanvas(
            mlp_values=self._demean(self.mlp_values[initial_layer]),
            self_attn_values=self._demean(self.self_attn_values[initial_layer]),
        )
        self.layout.addWidget(self.mpl_canvas)

        self.slider_layout = QHBoxLayout()

        self.layer_label = QLabel("Layer: 0")

        self.layer_slider = QSlider(Qt.Orientation.Horizontal)
        self.layer_slider.setMinimum(0)
        self.layer_slider.setMaximum(n_layers - 1)
        self.layer_slider.setValue(initial_layer)
        self.layer_slider.valueChanged.connect(self._on_slider_change)

        self.slider_layout.addWidget(self.layer_label)
        self.slider_layout.addWidget(self.layer_slider)
        self.layout.addLayout(self.slider_layout)

    def visualize_activations(self, mlp_projections: torch.Tensor, self_attn_projections: torch.Tensor, activations: Dict[str, torch.Tensor], tokenizer, output_tokens):
        self.is_activation_visualization = True
        initial_layer, initial_token = 0, 0
        n_layers = mlp_projections.shape[0]

        self.mlp_values = mlp_projections.float().cpu().numpy()
        self.self_attn_values = self_attn_projections.float().cpu().numpy()
        self.mlp_activation_tensor, self.self_attn_activation_tensor = self._collect_activations_into_tensors(activations)

        self.tokenizer = tokenizer
        self.output_tokens = output_tokens

        mlp_activations = self.mlp_activation_tensor[initial_layer]
        self_attn_activations = self.self_attn_activation_tensor[initial_layer]

        self.help_text = QLabel(
            "For the 3D scatter plots, size of the point is based on absolute value of the activation, and color is based on sign. For example, values near zero will be small points with neutral color, large values will be large green points, and large negative values will be large red points."
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

        self.layer_slider_layout = QHBoxLayout()
        self.layer_label = QLabel("Layer: 0")
        self.layer_slider = QSlider(Qt.Orientation.Horizontal)
        self.layer_slider.setMinimum(0)
        self.layer_slider.setMaximum(n_layers - 1)
        self.layer_slider.setValue(initial_layer)
        self.layer_slider_layout.addWidget(self.layer_label)
        self.layer_slider_layout.addWidget(self.layer_slider)
        self.layout.addLayout(self.layer_slider_layout)

        self.activation_slider_layout = QHBoxLayout()
        self.activation_label = QLabel("Activation: 0")
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

    def _on_slider_change(self):
        if self.is_activation_visualization:
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

            # Highlight the current token in the text box
            self._highlight_current_token(activation_index)
        else:
            layer_index = self.layer_slider.value()
            self.layer_label.setText(f"Layer: {layer_index}")
            self.mpl_canvas.update_data(
                mlp_values=self._demean(self.mlp_values[layer_index]),
                self_attn_values=self._demean(self.self_attn_values[layer_index]),
            )

    def _highlight_current_token(self, token_index):
        # Get the text and tokenizer from the visualization
        text = self.output_text_area.toPlainText()

        if not hasattr(self, "tokenized_text_indices"):
            # First time - we need to compute the token boundaries
            self._compute_token_boundaries()

        # Reset any previous formatting
        cursor = self.output_text_area.textCursor()
        cursor.select(cursor.SelectionType.Document)
        format = cursor.charFormat()
        format.setFontUnderline(False)
        format.setBackground(Qt.GlobalColor.transparent)
        cursor.setCharFormat(format)
        cursor.clearSelection()

        # Apply new highlight if we have valid indices
        if token_index < len(self.tokenized_text_indices) and token_index >= 0:
            start_pos, end_pos = self.tokenized_text_indices[token_index]

            cursor.setPosition(start_pos)
            cursor.setPosition(end_pos, cursor.MoveMode.KeepAnchor)

            highlight_format = cursor.charFormat()
            highlight_format.setFontUnderline(True)
            highlight_format.setBackground(Qt.GlobalColor.yellow)
            cursor.setCharFormat(highlight_format)

    # Add a method to compute token boundaries in the text
    def _compute_token_boundaries(self):
        text = self.output_text_area.toPlainText()

        # We need to determine where each token starts and ends in the text
        # This is a simplified approach, you'll need to adapt it based on how your
        # tokenizer actually works
        self.tokenized_text_indices = []

        # Assuming output_tokens is available and contains the token IDs
        token_strings = []
        for token_id in self.output_tokens:
            token_string = self.tokenizer.decode([token_id], skip_special_tokens=True)
            token_strings.append(token_string)

        # Find positions of each token in the full text
        pos = 0
        for token_string in token_strings:
            if token_string:  # Skip empty tokens
                # Find the token in the text starting from the current position
                start_pos = text.find(token_string, pos)
                if start_pos != -1:
                    end_pos = start_pos + len(token_string)
                    self.tokenized_text_indices.append((start_pos, end_pos))
                    pos = end_pos
                else:
                    # If token not found, use the previous end as start and add estimated length
                    # This is a fallback and not ideal
                    estimated_start = pos
                    estimated_end = estimated_start + max(1, len(token_string))
                    self.tokenized_text_indices.append((estimated_start, estimated_end))
                    pos = estimated_end

    def _convert_to_sizes(self, x: np.array) -> np.array:
        tmp = np.abs(x)
        if np.min(tmp) == np.max(tmp):
            return np.zeros_like(tmp, dtype=np.float32)
        else:
            return 20 * (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp)) + 1.5

    def _convert_to_color(self, x: np.array) -> np.array:
        if np.max(x) == np.min(x):
            return np.array([[0.5, 0.5, 0.5, 1.0]] * len(x))

        normalized_x = (x - np.min(x)) / (np.max(x) - np.min(x))
        colormap = matplotlib.colormaps.get_cmap("PiYG")
        return colormap(normalized_x)

    def _demean(self, x: np.array) -> np.array:
        return x - x.mean(axis=0)

    def _collect_activations_into_tensors(self, activations: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
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

        return mlp_tensors.float().cpu().numpy(), self_attn_tensors.float().cpu().numpy()

    @staticmethod
    def _register_activation_hooks(model: DeepReorderModel, activations: Dict[str, List]) -> List[torch.utils.hooks.RemovableHandle]:
        """Register hooks to collect activations during forward pass."""

        def _generate_hook(key):
            def _hook(_module, _input, output):
                if isinstance(output, torch.Tensor):
                    activations[key].append(output)
                elif isinstance(output, tuple):
                    activations[key].append(output[0])

            return _hook

        hooks = []
        for layer_idx, layer in enumerate(model.hf_model.model.layers):
            for component in ["mlp", "self_attn"]:
                hook = layer.__getattr__(component).register_forward_hook(_generate_hook(f"{layer_idx}_{component}"))
                hooks.append(hook)
        return hooks


class MessageConstructor:
    """Handles construction of prompts and messages for the model."""

    @staticmethod
    def construct_messages(instruction: str, input_text: str = "", system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> List[List[Dict[str, str]]]:
        """Construct a message for model inference."""
        return MessageConstructor.construct_messages_batch({"instruction": [instruction], "input": [input_text]}, system_prompt)

    @staticmethod
    def construct_messages_batch(batch: Dict[str, List[str]], system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> List[List[Dict[str, str]]]:
        """Prepare batch of messages for model inference."""
        batch_messages = []
        instructions = batch["instruction"]
        inputs = batch["input"]

        for i in range(len(instructions)):
            instruction = instructions[i]
            input_text = inputs[i]

            messages = [{"role": "system", "content": system_prompt}]
            if input_text:
                messages.append({"role": "user", "content": f"Instruction: {instruction}\nInput: {input_text}"})
            else:
                messages.append({"role": "user", "content": f"Instruction: {instruction}"})

            batch_messages.append(messages)

        return batch_messages


class DeepReorderApp(QWidget):
    """Main application for DeepReorder visualizations."""

    def __init__(self):
        """Initialize the application."""
        super().__init__()

        self.setWindowTitle("DeepReorder")
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
        self.inference_widget.setVisible(False)  # Initially hidden
        self.inference_layout = QVBoxLayout(self.inference_widget)
        self.inference_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Add heading
        self.inference_layout.addWidget(QLabel("Inference Inputs:", alignment=Qt.AlignmentFlag.AlignTop))

        # Add system prompt section
        self._add_system_prompt_section()

        # Add instruction section
        self._add_instruction_section()

        # Add content section
        self._add_content_section()

        # Add run button
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
        self.loaded_model = DeepReorderModel(model_name, DeepReorderModelParams())
        self.loaded_model.load_state(safetensors_path)
        self.loaded_model.freeze()
        self.loaded_model.hf_model = torch.compile(self.loaded_model.hf_model, backend="eager")
        self.loaded_model.hf_model = self.accelerator.prepare(self.loaded_model.hf_model)

        self.loaded_tokenizer = AutoTokenizer.from_pretrained(model_name)

    def visualize_model(self) -> None:
        """Visualize the loaded model."""
        if self.loaded_model is None:
            QMessageBox.warning(self, "Warning", "No model loaded. Please load a model first before visualizing.")
            return

        try:
            n_layers = len(self.loaded_model.hf_model.model.layers)
            mlp_projections = torch.stack([self.loaded_model.hf_model.model.layers[layer_idx].mlp.neuron_projections for layer_idx in range(n_layers)])
            self_attn_projections = torch.stack([self.loaded_model.hf_model.model.layers[layer_idx].self_attn.neuron_projections for layer_idx in range(n_layers)])
            self.visualizer = NewVisualizer()
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
        hooks = NewVisualizer._register_activation_hooks(self.loaded_model, activations)

        with torch.no_grad():
            output_tokens = self.loaded_model.hf_model.generate(
                tokenized_inputs.input_ids,
                attention_mask=tokenized_inputs.attention_mask,
                do_sample=True,
                pad_token_id=self.loaded_tokenizer.pad_token_id,
                eos_token_id=self.loaded_tokenizer.eos_token_id,
                max_new_tokens=750,
                temperature=0.7,
                top_p=0.9,
            )

        for hook in hooks:
            hook.remove()

        for key in activations.keys():
            activations[key] = torch.cat([tensor.squeeze(0) for tensor in activations[key]], dim=0)

        n_layers = len(self.loaded_model.hf_model.model.layers)
        mlp_projections = torch.stack([self.loaded_model.hf_model.model.layers[layer_idx].mlp.neuron_projections for layer_idx in range(n_layers)])
        self_attn_projections = torch.stack([self.loaded_model.hf_model.model.layers[layer_idx].self_attn.neuron_projections for layer_idx in range(n_layers)])

        input_token_length = tokenized_inputs.input_ids.shape[1]
        generated_output_tokens = output_tokens[0][input_token_length:]

        # Visualize with inference
        self.interactive_visualization = NewVisualizer()
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
    window = DeepReorderApp()
    window.show()
    sys.exit(app.exec())
