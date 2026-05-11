import torch

if __name__ == "__main__":
    print(torch.__version__)
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    x = torch.rand(5, 3)
    print(x)
