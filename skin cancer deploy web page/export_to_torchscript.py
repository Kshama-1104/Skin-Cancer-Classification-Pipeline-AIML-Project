"""
Export a PyTorch model checkpoint (state_dict) to TorchScript for deployment.

Option A: Use built-in presets for common torchvision models (no code edits).
Option B: Implement build_model() yourself to match your custom architecture.

Usage (PowerShell):
  .\.venv\Scripts\Activate.ps1
  python export_to_torchscript.py --checkpoint model.pt --out model_scripted.pt --image-size 224
"""

import argparse
from typing import Tuple

import torch
import torchvision.models as models


def build_model(num_classes: int) -> torch.nn.Module:
    """Customize this if you didn't use a preset.

    Example:
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    """
    raise NotImplementedError("Provide a model or use --preset.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to state_dict .pt or checkpoint")
    parser.add_argument("--out", default="model_scripted.pt", help="Output TorchScript file path")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of output classes")
    parser.add_argument("--image-size", type=int, default=224, help="Square input size (H=W)")
    parser.add_argument(
        "--preset",
        choices=[
            "resnet18", "resnet34", "resnet50",
            "mobilenet_v3_small", "mobilenet_v3_large",
            "efficientnet_b0", "efficientnet_b1",
            "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"
        ],
        help="Optional: build a common torchvision backbone automatically",
    )
    args = parser.parse_args()

    if args.preset:
        if args.preset.startswith("resnet"):
            backbone = getattr(models, args.preset)(weights=None)
            in_features = backbone.fc.in_features
            backbone.fc = torch.nn.Linear(in_features, args.num_classes)
            model = backbone
        elif args.preset.startswith("mobilenet_v3"):
            backbone = getattr(models, args.preset)(weights=None)
            in_features = backbone.classifier[-1].in_features
            backbone.classifier[-1] = torch.nn.Linear(in_features, args.num_classes)
            model = backbone
        elif args.preset.startswith("efficientnet_b"):
            backbone = getattr(models, args.preset)(weights=None)
            in_features = backbone.classifier[1].in_features
            backbone.classifier[1] = torch.nn.Linear(in_features, args.num_classes)
            model = backbone
        elif args.preset.startswith("vit_"):
            # Vision Transformer presets in torchvision
            backbone = getattr(models, args.preset)(weights=None)
            # torchvision ViT heads are under .heads.head
            if hasattr(backbone, "heads") and hasattr(backbone.heads, "head"):
                in_features = backbone.heads.head.in_features
                backbone.heads.head = torch.nn.Linear(in_features, args.num_classes)
            else:
                # Fallback common naming
                in_features = backbone.classifier.in_features  # type: ignore[attr-defined]
                backbone.classifier = torch.nn.Linear(in_features, args.num_classes)  # type: ignore[attr-defined]
            model = backbone
        else:
            raise ValueError(f"Unsupported preset: {args.preset}")
    else:
        model = build_model(num_classes=args.num_classes)
    state = torch.load(args.checkpoint, map_location=torch.device("cpu"))
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    dummy = torch.zeros(1, 3, args.image_size, args.image_size)
    scripted = torch.jit.trace(model, dummy)
    scripted.save(args.out)
    print(f"Saved TorchScript to {args.out}")


if __name__ == "__main__":
    main()


