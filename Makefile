# Grab weights from HuggingFace.
download:
	git clone https://huggingface.co/Qwen/Qwen2.5-3B build/qwen2_5-3B/

move_tokenizer:
	mv build/qwen2_5-3B/vocab.json build/vocab.json
	mv build/qwen2_5-3B/tokenizer.json build/tokenizer.json
	mv build/qwen2_5-3B/merges.txt build/merges.txt

# Convert from HuggingFace convention to TorchTune convention.
preprocess_weights:
	python scripts/merge_safetensors.py -i build/qwen2_5-3B/model-00001-of-00002.safetensors build/qwen2_5-3B/model-00002-of-00002.safetensors -o build/merged_weights.safetensors
	python scripts/rename_weights.py -i build/merged_weights.safetensors -o build/Qwen2_5-3B.safetensors

# Remove large unnecessary files. This will retain the processed weights.
clean:
	rm -f build/qwen2_5-3B/model-00001-of-00002.safetensors
	rm -f build/qwen2_5-3B/model-00002-of-00002.safetensors
	rm -f build/merged_weights.safetensors
	rm -rf build/qwen2_5-3B/

# Revert to just code from the repository.
clean_all:
	rm -rf build/