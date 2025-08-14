"""
Script to save a trained PyTorch or HuggingFace model.
Usage: import and call save_model(model, output_dir)
"""

import os


def save_model(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # For HuggingFace/transformers models
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
    # For plain PyTorch models
    elif hasattr(model, "state_dict"):
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        print(f"PyTorch model state_dict saved to {output_dir}/pytorch_model.bin")
    else:
        raise ValueError("Model type not supported for saving.")
