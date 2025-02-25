import sys
from collections import defaultdict

from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt

import accelerate
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib
from matplotlib.widgets import Slider
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from scipy.stats import zscore

from deep_reorder import DeepReorderModel, DeepReorderModelParams

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_SYSTEM_PROMPT = "Below is an instruction that describes a task. Write a response that appropriately completes the request."


def visualize(model: DeepReorderModel):
    """Visualize a model's neuron projections."""
    initial_layer = 0

    fig = plt.figure(figsize=(16, 8))
    ax_mlp = fig.add_subplot(121, projection="3d")
    ax_mlp.set_title("MLP Projection")
    ax_self_attn = fig.add_subplot(122, projection="3d")
    ax_self_attn.set_title("Self Attention Projection")

    mlp_data = model.hf_model.model.layers[initial_layer].mlp.neuron_projections.cpu().numpy()
    mlp_data = mlp_data - mlp_data.mean(axis=0)
    scatter_mlp = ax_mlp.scatter(mlp_data[:, 0], mlp_data[:, 1], mlp_data[:, 2], s=10, c="blue")

    self_attn_data = model.hf_model.model.layers[initial_layer].self_attn.neuron_projections.cpu().numpy()
    self_attn_data = self_attn_data - self_attn_data.mean(axis=0)
    scatter_self_attn = ax_self_attn.scatter(self_attn_data[:, 0], self_attn_data[:, 1], self_attn_data[:, 2], s=10, c="red")

    ax_mlp.set_xlim([-1.5, 1.5])
    ax_mlp.set_ylim([-1.5, 1.5])
    ax_mlp.set_zlim([-1.5, 1.5])
    ax_self_attn.set_xlim([-1.5, 1.5])
    ax_self_attn.set_ylim([-1.5, 1.5])
    ax_self_attn.set_zlim([-1.5, 1.5])

    plt.subplots_adjust(bottom=0.25)

    ax_layer = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_layer.set_zorder(10)
    layer_slider = Slider(ax=ax_layer, label="Layer", valmin=0, valmax=len(model.hf_model.model.layers) - 1, valinit=initial_layer, valstep=1, valfmt="%d")

    def update(_):
        layer_index = int(layer_slider.val)

        mlp_data = model.hf_model.model.layers[layer_index].mlp.neuron_projections.cpu().numpy()
        mlp_data = mlp_data - mlp_data.mean(axis=0)
        scatter_mlp._offsets3d = (mlp_data[:, 0], mlp_data[:, 1], mlp_data[:, 2])

        self_attn_data = model.hf_model.model.layers[layer_index].self_attn.neuron_projections.cpu().numpy()
        self_attn_data = self_attn_data - self_attn_data.mean(axis=0)
        scatter_self_attn._offsets3d = (self_attn_data[:, 0], self_attn_data[:, 1], self_attn_data[:, 2])

        fig.canvas.draw()

    layer_slider.on_changed(update)
    plt.show()


def _generate_sizes_and_colors(activations):
    activations_np = activations
    mean_activation = np.mean(activations_np)
    std_activation = np.std(activations_np)
    if std_activation == 0:
        normalized_activations = np.zeros_like(activations_np)
    else:
        normalized_activations = (activations_np - mean_activation) / std_activation
    sizes = np.abs(normalized_activations) + 0.1
    return sizes, normalized_activations


def visualize_inference(model: DeepReorderModel, tokenizer: AutoTokenizer, tokens: BatchEncoding):
    tokens.to(model.hf_model.device)

    hooks = []
    activations = defaultdict(list)

    def _generate_hook(key):
        def _hook(module, input, output):
            if isinstance(output, torch.Tensor):
                if output.shape[1] == 1:
                    activations[key].append(output)
            elif isinstance(output, tuple):
                if output[0].shape[1] == 1:
                    activations[key].append(output[0])

        return _hook

    for layer_idx, layer in enumerate(model.hf_model.model.layers):
        for component in ["mlp", "self_attn"]:
            hook = layer.__getattr__(component).register_forward_hook(_generate_hook(f"{layer_idx}_{component}"))
            hooks.append(hook)

    with torch.no_grad():
        output_tokens = model.hf_model.generate(
            tokens.input_ids,
            attention_mask=tokens.attention_mask,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
        )
        for hook in hooks:
            hook.remove()

        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        print("\n" * 10)

        for key in activations.keys():
            activations[key] = torch.cat([tensor.squeeze(0) for tensor in activations[key]], dim=0)
            print(f"{key}: {activations[key].shape}")

    initial_layer, initial_token = 0, 0
    fig = plt.figure(figsize=(16, 8))
    ax_mlp = fig.add_subplot(121, projection="3d")
    ax_mlp.set_title("MLP Projection")
    ax_self_attn = fig.add_subplot(122, projection="3d")
    ax_self_attn.set_title("Self Attention Projection")

    mlp_data = model.hf_model.model.layers[initial_layer].mlp.neuron_projections.cpu().numpy()
    mlp_data = mlp_data - mlp_data.mean(axis=0)
    mlp_sizes, mlp_activation_values = _generate_sizes_and_colors(activations[f"{initial_layer}_mlp"][initial_token].float().cpu().numpy())
    mlp_colors = plt.cm.coolwarm(mlp_activation_values)
    scatter_mlp = ax_mlp.scatter(mlp_data[:, 0], mlp_data[:, 1], mlp_data[:, 2], s=mlp_sizes, c=mlp_colors)

    self_attn_data = model.hf_model.model.layers[initial_layer].self_attn.neuron_projections.cpu().numpy()
    self_attn_data = self_attn_data - self_attn_data.mean(axis=0)
    self_attn_sizes, self_attn_activation_values = _generate_sizes_and_colors(activations[f"{initial_layer}_self_attn"][initial_token].float().cpu().numpy())
    self_attn_colors = plt.cm.coolwarm(self_attn_activation_values)
    scatter_self_attn = ax_self_attn.scatter(self_attn_data[:, 0], self_attn_data[:, 1], self_attn_data[:, 2], s=self_attn_sizes, c=self_attn_colors)

    ax_mlp.set_xlim([-1.5, 1.5])
    ax_mlp.set_ylim([-1.5, 1.5])
    ax_mlp.set_zlim([-1.5, 1.5])
    ax_self_attn.set_xlim([-1.5, 1.5])
    ax_self_attn.set_ylim([-1.5, 1.5])
    ax_self_attn.set_zlim([-1.5, 1.5])

    plt.subplots_adjust(bottom=0.25)

    ax_layer = plt.axes([0.25, 0.2, 0.65, 0.03])
    ax_layer.set_zorder(10)
    layer_slider = Slider(ax=ax_layer, label="Layer", valmin=0, valmax=len(model.hf_model.model.layers) - 1, valinit=initial_layer, valstep=1, valfmt="%d")

    ax_token = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_token.set_zorder(10)
    token_slider = Slider(ax=ax_token, label="Token", valmin=0, valmax=tokens.input_ids.shape[1] - 1, valinit=initial_token, valstep=1, valfmt="%d")  # Set valmax based on input tokens

    def update(_):
        layer_index = int(layer_slider.val)
        token_index = int(token_slider.val)

        mlp_data = model.hf_model.model.layers[layer_index].mlp.neuron_projections.float().cpu().numpy()
        mlp_data = mlp_data - mlp_data.mean(axis=0)
        scatter_mlp._offsets3d = (mlp_data[:, 0], mlp_data[:, 1], mlp_data[:, 2])
        activation_input = activations[f"{layer_index}_mlp"][token_index].float().cpu().numpy()
        mlp_sizes, mlp_activation_values = _generate_sizes_and_colors(activation_input)
        mlp_colors = plt.cm.coolwarm(mlp_activation_values)
        scatter_mlp.set_sizes(mlp_sizes)
        scatter_mlp.set_facecolors(mlp_colors)

        self_attn_data = model.hf_model.model.layers[layer_index].self_attn.neuron_projections.float().cpu().numpy()
        self_attn_data = self_attn_data - self_attn_data.mean(axis=0)
        scatter_self_attn._offsets3d = (self_attn_data[:, 0], self_attn_data[:, 1], self_attn_data[:, 2])
        self_attn_sizes, self_attn_activation_values = _generate_sizes_and_colors(activations[f"{layer_index}_self_attn"][token_index].float().cpu().numpy())
        self_attn_colors = plt.cm.coolwarm(self_attn_activation_values)
        scatter_self_attn.set_sizes(self_attn_sizes)
        scatter_self_attn.set_facecolors(self_attn_colors)

        fig.canvas.draw_idle()

    layer_slider.on_changed(update)
    token_slider.on_changed(update)
    plt.show()


class DeepReorderApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("DeepReorder")
        self.setGeometry(100, 100, 800, 600)

        self.loaded_model = None
        self.loaded_tokenizer = None

        self.accelerator = accelerate.Accelerator()
        self.device = self.accelerator.device

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Model Selection
        hf_model_layout = QHBoxLayout()
        hf_model_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.hf_model_label = QLabel("HF Model:")
        hf_model_layout.addWidget(self.hf_model_label)
        hf_model_layout.setAlignment(self.hf_model_label, Qt.AlignmentFlag.AlignTop)
        self.hf_model_text_box = QLineEdit()
        hf_model_layout.addWidget(self.hf_model_text_box)
        hf_model_layout.setAlignment(self.hf_model_text_box, Qt.AlignmentFlag.AlignTop)
        self.hf_model_text_box.setText(MODEL_NAME)
        main_layout.addLayout(hf_model_layout)

        # Load Tensors
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
        main_layout.addLayout(file_selector_layout)

        # Load and Visualize Buttons in a Horizontal Layout
        load_visualize_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_and_visualize_model)  # Connect Load button
        load_visualize_layout.addWidget(self.load_button)
        self.visualize_button = QPushButton("Visualize")
        self.visualize_button.clicked.connect(self.visualize_button_clicked)  # Connect Visualize button
        load_visualize_layout.addWidget(self.visualize_button)
        main_layout.addLayout(load_visualize_layout)
        load_visualize_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Horizontal Separator Line
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(separator)

        # Inference Widget
        self.inference_widget = QWidget()
        self.inference_widget.setVisible(False)  # Initially hide inference widget
        self.inference_layout = QVBoxLayout(self.inference_widget)
        self.inference_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.inference_layout.addWidget(QLabel("Inference Inputs:", alignment=Qt.AlignmentFlag.AlignTop))

        # System Prompt
        self.system_prompt_label = QLabel("System Prompt:")
        self.inference_layout.addWidget(self.system_prompt_label, alignment=Qt.AlignmentFlag.AlignTop)
        self.system_prompt_text_box = QLineEdit()
        self.system_prompt_text_box.setText(DEFAULT_SYSTEM_PROMPT)
        self.inference_layout.addWidget(self.system_prompt_text_box, alignment=Qt.AlignmentFlag.AlignTop)

        # Instruction
        self.instruction_label = QLabel("Instruction:")
        self.inference_layout.addWidget(self.instruction_label, alignment=Qt.AlignmentFlag.AlignTop)
        self.instruction_text_box = QTextEdit()
        self.inference_layout.addWidget(self.instruction_text_box, alignment=Qt.AlignmentFlag.AlignTop)

        # Content
        self.content_label = QLabel("Content:")
        self.inference_layout.addWidget(self.content_label, alignment=Qt.AlignmentFlag.AlignTop)
        self.content_text_box = QTextEdit()
        self.inference_layout.addWidget(self.content_text_box, alignment=Qt.AlignmentFlag.AlignTop)

        # Run Button
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_inference)
        self.inference_layout.addWidget(self.run_button, alignment=Qt.AlignmentFlag.AlignTop)

        main_layout.addWidget(self.inference_widget)
        main_layout.addStretch()

        self.setLayout(main_layout)

    def open_file_dialog(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select SafeTensors File")
        if file_path:
            self.file_path_line_edit.setText(file_path)

    def load_and_visualize_model(self):
        model_name = self.hf_model_text_box.text()
        safetensors_path = self.file_path_line_edit.text()

        if not safetensors_path:
            QMessageBox.warning(self, "Warning", "Please select a SafeTensors file path.")
            return

        try:
            self.loaded_model = DeepReorderModel(model_name, DeepReorderModelParams())  # Store loaded model
            self.loaded_model.load_state(safetensors_path)
            self.loaded_model.freeze()
            self.loaded_model.hf_model = torch.compile(self.loaded_model.hf_model, backend="eager")
            self.loaded_model.hf_model = self.accelerator.prepare(self.loaded_model.hf_model)

            self.loaded_tokenizer = AutoTokenizer.from_pretrained(model_name)

            self.inference_widget.setVisible(True)  # Show inference widget after successful load
            QMessageBox.information(self, "Success", "Model loaded successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")
            self.inference_widget.setVisible(False)  # Ensure inference widget is hidden on error
            self.loaded_model = None  # Reset loaded model on error

    def visualize_button_clicked(self):
        if self.loaded_model is None:
            QMessageBox.warning(self, "Warning", "No model loaded. Please load a model first before visualizing.")
            return
        try:
            visualize(self.loaded_model)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to visualize model:\n{e}")

    def run_inference(self):
        if self.loaded_model is None:
            QMessageBox.warning(self, "Warning", "Model not loaded. Please load a model first.")
            return

        system_prompt = self.system_prompt_text_box.text()
        instruction = self.instruction_text_box.toPlainText()
        content = self.content_text_box.toPlainText()

        if not instruction:
            QMessageBox.warning(self, "Warning", "Instruction cannot be empty.")
            return

        constructed_message = self._construct_messages(instruction, content, system_prompt)
        tokenized_inputs = self.loaded_tokenizer(
            [self.loaded_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in constructed_message],
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=self.loaded_model.hf_model.config.max_position_embeddings,
            return_tensors="pt",
        )
        visualize_inference(self.loaded_model, self.loaded_tokenizer, tokenized_inputs)
        # Example inference (you'll need to adapt this based on how you use DeepReorderModel for inference)
        # input_text = f"{system_prompt}\nInstruction: {instruction}\nContent: {content}"
        # response = self.loaded_model.generate_response(input_text) # Assuming a generate_response method exists in DeepReorderModel
        # print("Model Response:", response)
        # --- END REPLACE ---

    def _construct_messages_batch(self, batch, system_prompt="Below is an instruction that describes a task. Write a response that appropriately completes the request."):
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

            batch_messages.append(messages)

        return batch_messages

    def _construct_messages(self, instruction, input="", system_prompt="Below is an instruction that describes a task. Write a response that appropriately completes the request."):
        return self._construct_messages_batch({"instruction": [instruction], "input": [input]}, system_prompt)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeepReorderApp()
    window.show()
    sys.exit(app.exec())
