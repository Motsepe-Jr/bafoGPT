import torch
import sys

def analyze_model(model_path):
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    total_params = 0
    total_bytes = 0
    layer_types = {}

    for key, tensor in state_dict.items():
        params = tensor.numel()
        bytes = tensor.element_size() * params
        
        total_params += params
        total_bytes += bytes
        
        layer_type = key.split('.')[0]
        if layer_type not in layer_types:
            layer_types[layer_type] = {'count': 0, 'params': 0, 'bytes': 0}
        layer_types[layer_type]['count'] += 1
        layer_types[layer_type]['params'] += params
        layer_types[layer_type]['bytes'] += bytes

    print(f"Model: {model_path}")
    print(state_dict.keys)
    print(f"Total Parameters: {total_params}")
    print(f"Total Size: {total_bytes / (1024**3):.2f} GB")
    print("\nLayer Type Analysis:")
    for layer_type, info in layer_types.items():
        print(f"  {layer_type}:")
        print(f"    Count: {info['count']}")
        print(f"    Parameters: {info['params']}")
        print(f"    Size: {info['bytes'] / (1024**3):.2f} GB")
    print("\n")

# Analyze both models
analyze_model("convert_lit_model/model.pth")
analyze_model("convert_pretrained_model/lit_model.pth") # the LitGPT model (lit_model.pth) has fewer layers - it combines Q, K, and V into a single layer.


# Model: pytorch_model.bin
# Total Parameters: 3352801536
# Total Size: 12.49 GB

# Layer Type Analysis:
#   lm_head:
#     Count: 1
#     Parameters: 664141824
#     Size: 2.47 GB
#   model:
#     Count: 288
#     Parameters: 2688659712
#     Size: 10.02 GB


# Model: lit_model.pth
# Total Parameters: 3352801536
# Total Size: 12.49 GB

# Layer Type Analysis:
#   lm_head:
#     Count: 1
#     Parameters: 664141824
#     Size: 2.47 GB
#   transformer:
#     Count: 236
#     Parameters: 2688659712
#     Size: 10.02 GB