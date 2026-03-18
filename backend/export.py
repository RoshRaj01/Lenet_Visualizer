import torch
from model import LeNet

model = LeNet()
model.load_state_dict(torch.load("saved_model/lenet.pth"))
model.eval()

dummy_input = torch.randn(1, 1, 28, 28)

# Wrapper for ONNX export
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
            activations["fc2"]
        )

export_model = ExportModel(model)

torch.onnx.export(
    export_model,
    dummy_input,
    "saved_model/lenet.onnx",
    input_names=["input"],
    output_names=["output", "conv1", "conv2", "fc1", "fc2"]
)

print("ONNX exported with layers")