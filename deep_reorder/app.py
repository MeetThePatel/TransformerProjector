import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt

import accelerate
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib

matplotlib.use("QtAgg")
from matplotlib.widgets import Slider
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from scipy.stats import zscore

from deep_reorder import DeepReorderModel, DeepReorderModelParams

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_SYSTEM_PROMPT = "Below is an instruction that describes a task. Write a response that appropriately completes the request."


class Visualizer:
    """Class to handle visualization of model neuron projections."""

    @staticmethod
    def visualize_model(model: DeepReorderModel) -> None:
        """Visualize a model's neuron projections without activations."""
        initial_layer = 0

        fig = plt.figure(figsize=(16, 8))
        ax_mlp = fig.add_subplot(121, projection="3d")
        ax_mlp.set_title("MLP Projection")
        ax_self_attn = fig.add_subplot(122, projection="3d")
        ax_self_attn.set_title("Self Attention Projection")

        # Setup initial layer visualization
        mlp_data, scatter_mlp = Visualizer._setup_projection_plot(model.hf_model.model.layers[initial_layer].mlp.neuron_projections, ax_mlp, "blue")
        self_attn_data, scatter_self_attn = Visualizer._setup_projection_plot(model.hf_model.model.layers[initial_layer].self_attn.neuron_projections, ax_self_attn, "red")

        # Set consistent limits for axes
        Visualizer._set_axis_limits([ax_mlp, ax_self_attn])

        # Add layer slider
        plt.subplots_adjust(bottom=0.25)
        ax_layer = plt.axes((0.25, 0.1, 0.65, 0.03))
        ax_layer.set_zorder(10)
        layer_slider = Slider(ax=ax_layer, label="Layer", valmin=0, valmax=len(model.hf_model.model.layers) - 1, valinit=initial_layer, valstep=1, valfmt="%d")

        # Define update function for slider
        def update(_):
            layer_index = int(layer_slider.val)

            update_mlp_data = model.hf_model.model.layers[layer_index].mlp.neuron_projections.cpu().numpy()
            update_mlp_data = update_mlp_data - update_mlp_data.mean(axis=0)
            scatter_mlp._offsets3d = (update_mlp_data[:, 0], update_mlp_data[:, 1], update_mlp_data[:, 2])

            update_self_attn_data = model.hf_model.model.layers[layer_index].self_attn.neuron_projections.cpu().numpy()
            update_self_attn_data = update_self_attn_data - update_self_attn_data.mean(axis=0)
            scatter_self_attn._offsets3d = (update_self_attn_data[:, 0], update_self_attn_data[:, 1], update_self_attn_data[:, 2])

            fig.canvas.draw()

        layer_slider.on_changed(update)
        plt.show()

    @staticmethod
    def visualize_inference(model: DeepReorderModel, tokenizer: AutoTokenizer, tokens: BatchEncoding) -> None:
        """Visualize model neuron projections with activation data during inference."""
        # Move tokens to the model's device
        tokens.to(model.hf_model.device)

        activations = defaultdict(list)
        hooks = Visualizer._register_activation_hooks(model, activations)

        with torch.no_grad():
            output_tokens = model.hf_model.generate(
                tokens.input_ids,
                attention_mask=tokens.attention_mask,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=1000,
                temperature=0.7,
                top_p=0.9,
            )

        for hook in hooks:
            hook.remove()

        input_token_length = tokens.input_ids.shape[1]
        generated_output_tokens = output_tokens[0][input_token_length:]
        generated_text = tokenizer.decode(generated_output_tokens, skip_special_tokens=True)

        # Concatenate activation tensors for each layer and component
        for key in activations.keys():
            activations[key] = torch.cat([tensor.squeeze(0) for tensor in activations[key]], dim=0)

        # Set up visualization
        initial_layer, initial_token = 0, 0

        # Create figure with adjusted size and layout
        fig = plt.figure(figsize=(16, 10))

        # Adjust the overall layout to leave space for controls and text
        plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.20)

        # Create subplots
        ax_mlp = fig.add_subplot(121, projection="3d")
        ax_mlp.set_title("MLP Projection")
        ax_self_attn = fig.add_subplot(122, projection="3d")
        ax_self_attn.set_title("Self Attention Projection")

        # Setup initial visualizations with activations
        scatter_mlp = Visualizer._setup_activation_plot(model.hf_model.model.layers[initial_layer].mlp.neuron_projections, activations[f"{initial_layer}_mlp"][initial_token], ax_mlp)

        scatter_self_attn = Visualizer._setup_activation_plot(
            model.hf_model.model.layers[initial_layer].self_attn.neuron_projections, activations[f"{initial_layer}_self_attn"][initial_token], ax_self_attn
        )

        # Set consistent limits for axes
        Visualizer._set_axis_limits([ax_mlp, ax_self_attn])

        # Create more compact slider controls
        slider_height = 0.02
        slider_spacing = 0.03
        slider_width = 0.70
        slider_left = 0.15
        slider_bottom_start = 0.10

        # Layer slider - positioned higher
        ax_layer = plt.axes((slider_left, slider_bottom_start + slider_spacing, slider_width, slider_height))
        layer_slider = Slider(ax=ax_layer, label="Layer", valmin=0, valmax=len(model.hf_model.model.layers) - 1, valinit=initial_layer, valstep=1, valfmt="%d")

        # Token slider - positioned just below layer slider
        ax_token = plt.axes((slider_left, slider_bottom_start, slider_width, slider_height))
        token_slider = Slider(ax=ax_token, label="Token", valmin=0, valmax=generated_output_tokens.shape[0] - 1, valinit=initial_token, valstep=1, valfmt="%d")

        # Add text display for the output - position at bottom left with proper size
        text_output_ax = plt.axes((0.05, 0.01, 0.90, 0.15))
        text_output_ax.axis("off")

        full_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # Create text display with appropriate wrapping and alignment
        text_output = text_output_ax.text(
            0,
            1.0,  # Position at top-left of the text area
            full_text,
            ha="left",
            va="top",
            wrap=True,
            fontsize=9,
            transform=text_output_ax.transAxes,
        )

        # Define update function for sliders
        def update(_):
            layer_index = int(layer_slider.val)
            token_index = int(token_slider.val)

            # Update MLP visualization
            Visualizer._update_activation_plot(scatter_mlp, model.hf_model.model.layers[layer_index].mlp.neuron_projections, activations[f"{layer_index}_mlp"][input_token_length + token_index - 1])

            # Update Self-Attention visualization
            Visualizer._update_activation_plot(
                scatter_self_attn, model.hf_model.model.layers[layer_index].self_attn.neuron_projections, activations[f"{layer_index}_self_attn"][input_token_length + token_index - 1]
            )

            # Get current token info
            all_tokens = tokenizer.convert_ids_to_tokens(generated_output_tokens)
            current_token = all_tokens[token_index] if token_index < len(all_tokens) else ""

            # Update text with clear formatting and no overlap
            formatted_text = f"Current token: '{current_token}'\n\nFull text: {full_text}"
            text_output.set_text(formatted_text)
            text_output.set_wrap(True)

            fig.canvas.draw_idle()

        # Connect sliders to update function
        layer_slider.on_changed(update)
        token_slider.on_changed(update)

        plt.show()

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

    @staticmethod
    def _setup_projection_plot(projections: torch.Tensor, ax: plt.Axes, color: str) -> Tuple[np.ndarray, Any]:
        """Setup a 3D scatter plot of neuron projections."""
        projection_data = projections.cpu().numpy()
        projection_data = projection_data - projection_data.mean(axis=0)
        scatter = ax.scatter(projection_data[:, 0], projection_data[:, 1], projection_data[:, 2], s=10, c=color)
        return projection_data, scatter

    @staticmethod
    def _setup_activation_plot(projections: torch.Tensor, activations: torch.Tensor, ax: plt.Axes) -> Any:
        """Setup a 3D scatter plot with activation data."""
        projection_data = projections.float().cpu().numpy()
        projection_data = projection_data - projection_data.mean(axis=0)

        # Get sizes and colors based on activations
        sizes, activation_values = Visualizer._generate_sizes_and_colors(activations.float().cpu().numpy())
        colors = plt.cm.coolwarm(activation_values)

        scatter = ax.scatter(projection_data[:, 0], projection_data[:, 1], projection_data[:, 2], s=sizes, c=colors)
        return scatter

    @staticmethod
    def _update_activation_plot(scatter: Any, projections: torch.Tensor, activations: torch.Tensor) -> None:
        """Update an existing 3D scatter plot with new data."""
        projection_data = projections.float().cpu().numpy()
        projection_data = projection_data - projection_data.mean(axis=0)
        scatter._offsets3d = (projection_data[:, 0], projection_data[:, 1], projection_data[:, 2])

        # Update sizes and colors based on activations
        sizes, activation_values = Visualizer._generate_sizes_and_colors(activations.float().cpu().numpy())
        colors = plt.cm.coolwarm(activation_values)
        scatter.set_sizes(sizes)
        scatter.set_facecolors(colors)

    @staticmethod
    def _generate_sizes_and_colors(activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate point sizes and color values based on activations."""
        mean_activation = np.mean(activations)
        std_activation = np.std(activations)

        if std_activation == 0:
            normalized_activations = np.zeros_like(activations)
        else:
            normalized_activations = (activations - mean_activation) / std_activation

        sizes = np.abs(normalized_activations) + 0.1
        return sizes, normalized_activations

    @staticmethod
    def _set_axis_limits(axes: List[plt.Axes], limit: float = 1.5) -> None:
        """Set consistent limits for 3D axes."""
        for ax in axes:
            ax.set_xlim([-limit, limit])
            ax.set_ylim([-limit, limit])
            ax.set_zlim([-limit, limit])


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
            Visualizer.visualize_model(self.loaded_model)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to visualize model:\n{e}")

    def run_inference(self) -> None:
        """Run inference with the loaded model."""
        if self.loaded_model is None:
            QMessageBox.warning(self, "Warning", "Model not loaded. Please load a model first.")
            return

        # Get input values
        system_prompt = self.system_prompt_text_box.text()
        instruction = self.instruction_text_box.toPlainText()
        content = self.content_text_box.toPlainText()

        if not instruction:
            QMessageBox.warning(self, "Warning", "Instruction cannot be empty.")
            return

        # Prepare messages and tokenize
        constructed_message = MessageConstructor.construct_messages(instruction, content, system_prompt)

        tokenized_inputs = self._tokenize_input_messages(constructed_message)

        # Visualize with inference
        Visualizer.visualize_inference(self.loaded_model, self.loaded_tokenizer, tokenized_inputs)

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
