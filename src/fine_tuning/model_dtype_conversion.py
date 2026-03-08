
"""
Python snippet will convert a model file's data type into float32. 

it assumes that the loaded model (m in this script) is a nn.Module object. Most are, because this
is typically the characteristic object of a model in PyTorch. 

input_model_path is the directory of the model file, and output_model_path is 
the created file that the new model is written to. 

"""


import torch
from pathlib import Path

input_model_path = Path("assets/models/mace/mace-mpa-0-medium.model")
output_model_path = Path("assets/models/mace/mace-mpa-0-medium-f32.model")

m = torch.load(input_model_path, map_location="cpu", weights_only=False)
m = m.to(torch.float32)
torch.save(m, output_model_path)

print("wrote:", output_model_path)

