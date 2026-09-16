"""Test environment script for checking packages, GPUs, and foundation models."""
import torch
import torchvision

print("=== Environment Verification ===")
print("PyTorch Version:", torch.__version__)
print("Torchvision Version:", torchvision.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("Device Count:", torch.cuda.device_count())

print("\n=== Testing Package Imports ===")
packages = ["timm", "open_clip", "transformers", "pywt", "scipy", "sklearn", "huggingface_hub"]
for pkg in packages:
    try:
        mod = __import__(pkg)
        print(f"  [AVAILABLE] {pkg} (version: {getattr(mod, '__version__', 'unknown')})")
    except ImportError as e:
        print(f"  [MISSING]   {pkg} ({e})")

print("\n=== Testing DINOv2 / Model Loading ===")
try:
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    print("  [SUCCESS] DINOv2-small loaded successfully!")
    x = torch.randn(1, 3, 224, 224)
    out = dinov2(x)
    print("  DINOv2 output shape:", out.shape)
except Exception as e:
    print(f"  [FAILED] DINOv2 load failed: {e}")

try:
    dinov2_base = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    print("  [SUCCESS] DINOv2-base loaded successfully!")
except Exception as e:
    print(f"  [FAILED] DINOv2-base load failed: {e}")

try:
    import torchvision.models as m
    swin = m.swin_t(weights=m.Swin_T_Weights.DEFAULT)
    print("  [SUCCESS] Torchvision Swin-T loaded successfully!")
except Exception as e:
    print(f"  [FAILED] Swin-T load failed: {e}")

try:
    vit = m.vit_b_16(weights=m.ViT_B_16_Weights.DEFAULT)
    print("  [SUCCESS] Torchvision ViT-B/16 loaded successfully!")
except Exception as e:
    print(f"  [FAILED] ViT-B/16 load failed: {e}")
