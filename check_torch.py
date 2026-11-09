import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

    props = torch.cuda.get_device_properties(0)
    print("GPU Memory:", round(props.total_memory / (1024**3), 2), "GB")
else:
    print("Running on CPU")
