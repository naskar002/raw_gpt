# README: GPU Constraints and Code Modifications

## 🚀 GPU Information

This project runs on a **NVIDIA GTX 1650** GPU. Due to hardware limitations, the following constraints should be considered:

### ❌ Unsupported Features

- **TFLOAT16 & BFLOAT16 are not supported**: The GTX 1650 does not support **TensorFloat-16 (TFLOAT16)** or **BFloat16 (BFLOAT16)**, meaning mixed precision training may not work efficiently.
- **No Multi-GPU Parallel Processing**: Since only a single GPU is available, we cannot use **Data Parallelism (**``**)** or **Model Parallelism**.

---

## ⚡ Code Modifications for Performance Optimization

To optimize performance on a **GTX 1650**, certain hyperparameters and configurations need adjustments.

### 🔧 Required Changes

Modify the **batch size** and other settings in your script to prevent memory overflows and improve speed:

```bash
# Reduce batch size to fit GPU memory constraints
# Example: Change from
B = 256  # Too large for GTX 1650

# To a more optimized value
B = 64   # Adjusted for better performance
```

Other possible optimizations:

- Reduce model size (fewer layers or parameters)
- Use **float32** instead of unsupported **tfloat16/bfloat16**
- Optimize data loading with `num_workers` and `pin_memory=True`
- Enable **cudnn benchmark** for performance tuning:
  ```python
  import torch
  torch.backends.cudnn.benchmark = True
  ```

---

## 🛠️ Future Improvements

If upgrading hardware in the future, consider:

- A **RTX-series GPU** (e.g., RTX 3060 or higher) to support **TFLOAT16/BFLOAT16**
- Adding more **VRAM** for handling larger batch sizes and more complex models

---

### 🔗 Additional References

- NVIDIA CUDA Compute Capability: [GTX 1650 Specs](https://developer.nvidia.com/cuda-gpus)
- PyTorch Performance Guide: [PyTorch Docs](https://pytorch.org/docs/stable/notes/cuda.html)

💡 **Tip**: Always monitor GPU usage with `nvidia-smi` to avoid crashes!

