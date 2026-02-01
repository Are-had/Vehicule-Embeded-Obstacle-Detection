import torch
import torch.onnx
from models.model_stages import BiSeNet  # We need this just once for export

# --- CONFIG ---
MODEL_PATH = 'pths/stdc813m_maxmiou_4.pth'
OUTPUT_ONNX = 'onnxs/stdc813m_maxmiou_4.onnx'
INPUT_SHAPE = (1, 3, 512, 1024)  # (Batch, Channels, Height, Width) - Fixed size is best for NPUs

# 1. Load the PyTorch Model as usual
print("Loading PyTorch model...")
model = BiSeNet(backbone='STDCNet813', n_classes=2, use_boundary_8=True)
state_dict = torch.load(MODEL_PATH, map_location='cpu')

# Clean keys (same as your inference script)
new_state_dict = {}
for k, v in state_dict.items():
    k = k.replace("module.", "").replace(".bn.bn.", ".bn.")
    new_state_dict[k] = v
model.load_state_dict(new_state_dict, strict=False)
model.eval()

# 2. Create a dummy input
# This helps ONNX "trace" the execution flow to understand the architecture
dummy_input = torch.randn(*INPUT_SHAPE)

# 3. Export to ONNX
print(f"Exporting to {OUTPUT_ONNX}...")
torch.onnx.export(
    model,
    dummy_input,
    OUTPUT_ONNX,
    opset_version=11,          # Opset 11 is widely supported by AI HATs
    input_names=['input'],     # Name of the input layer
    output_names=['output'],   # Name of the output layer
    do_constant_folding=True   # Optimizes constants
)

print("✅ Export success! You can now delete the 'models' folder for deployment.")