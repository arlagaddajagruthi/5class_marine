import torch

def check_cuda():
    print("="*50)
    print("PyTorch and CUDA Environment Check")
    print("="*50)
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"  Memory Cached:    {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    else:
        print("CUDA is NOT available. Models will run on CPU, which will be significantly slower.")
    print("="*50)

if __name__ == "__main__":
    check_cuda()
