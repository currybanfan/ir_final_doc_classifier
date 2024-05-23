import torch

def print_cuda_memory_info(device_id=0):
  device = torch.device(f'cuda:{device_id}')
  total_mem = torch.cuda.get_device_properties(device).total_memory
  reserved_mem = torch.cuda.memory_reserved(device)
  allocated_mem = torch.cuda.memory_allocated(device)
  free_mem = reserved_mem - allocated_mem
  
  print(f"Device: {torch.cuda.get_device_name(device)}")
  print(f"Total Memory: {total_mem / 1e9:.2f} GB")
  print(f"Reserved Memory: {reserved_mem / 1e9:.2f} GB")
  print(f"Allocated Memory: {allocated_mem / 1e9:.2f} GB")
  print(f"Free Memory: {free_mem / 1e9:.2f} GB")


def clear_cache():
  torch.cuda.empty_cache()
