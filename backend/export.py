import torch
from model import LeNet

# Load trained weights
# weights_only=True suppresses the FutureWarning about pickle security
model = LeNet()
model.load_state_dict(torch.load("saved_model/lenet.pth", map_location="cpu", weights_only=True))
model.eval()   # Sets BatchNorm to use running stats (not batch stats)

dummy_input = torch.randn(1, 1, 28, 28)

class ExportModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output, activations = self.model(x)
        return (
            output,
            activations["conv1"],
            activations["conv2"],
            activations["fc1"],
            activations["fc2"],
        )

export_model = ExportModel(model)
export_model.eval()

torch.onnx.export(
    export_model,
    dummy_input,
    "saved_model/lenet.onnx",
    input_names=["input"],
    output_names=["output", "conv1", "conv2", "fc1", "fc2"],
    opset_version=11,
    do_constant_folding=True,
)

print("ONNX exported → saved_model/lenet.onnx")

# ── Quick sanity check with onnxruntime ──────────────────────
try:
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession("saved_model/lenet.onnx",
                                providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 1, 28, 28).astype(np.float32)
    outs  = sess.run(None, {"input": dummy})
    print("Output shapes :", [o.shape for o in outs])
    print("Test prediction:", outs[0].argmax(), " (random input — just checking no crash)")
    print("\nAll good — copy saved_model/lenet.onnx to frontend/model/lenet.onnx")
except ImportError:
    print("(onnxruntime not installed — skipping sanity check)")
    print("Copy saved_model/lenet.onnx to frontend/model/lenet.onnx")