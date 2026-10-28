"""Download the base model from the Hugging Face Hub into LYM/models/ and check that it loads.

Usage:  python LYM/src/fetch_model.py
"""
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

from common import MODEL_DIR, MODEL_ID, end_of_turn_id


def main() -> None:
    print(f"Downloading {MODEL_ID} to {MODEL_DIR} ...")
    # Only the files needed to run the model: configs, tokenizer files and the safetensors weights (~3 GB).
    # Re-running skips files that are already complete, so an interrupted download can simply be restarted.
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE"],
    )

    # Sanity check: config and tokenizer load from the local folder, and the tokenizer has the token that ends a turn.
    config = AutoConfig.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    end_of_turn_id(tokenizer)
    print(f"OK: {config.model_type}, {config.num_hidden_layers} layers, vocabulary of {len(tokenizer)} tokens")


if __name__ == "__main__":
    main()
