# 🚀 Distributed AI Model Training using MPI and GPU Acceleration

This project demonstrates how High-Performance Computing (HPC) techniques can drastically accelerate deep learning training using *MPI-based distributed computing, **multi-GPU acceleration, and **PyTorch Distributed Data Parallel (DDP)*.

A *MobileNetV2* image classification model is trained on the *CIFAR-10* dataset under three configurations:

- CPU Serial Training
- Single GPU Training
- Multi-GPU Distributed Training (MPI + DDP)

The project also includes a *Gradio web interface* for real-time inference.

---

## 🧠 Objective

Deep learning models are computationally intensive. Training on a single device becomes slow and inefficient as model complexity increases.

This project explores how much training time can be reduced using *MPI + Multi-GPU parallelism* without compromising model accuracy.

---

## 🏗️ System Overview

- *Model:* MobileNetV2 (ImageNet pretrained, fine-tuned for CIFAR-10)
- *Dataset:* CIFAR-10 (60,000 images, 10 classes)
- *Input Resolution:* 32×32 → *224×224*
- *Data Augmentation:* RandomResizedCrop, Horizontal Flip, ColorJitter, Rotation
- *Training Modes:*
  - Serial CPU
  - Serial GPU
  - Distributed Multi-GPU (MPI + PyTorch DDP)
- *Deployment:* Gradio interface for inference

---

## ⚙️ Technologies Used

- Python
- PyTorch, torchvision
- MPI (mpi4py), OpenMPI
- PyTorch Distributed Data Parallel (DDP)
- CUDA, cuDNN
- SLURM (job scheduling)
- Gradio
- PIL, NumPy

---

## 🖥️ Hardware Configuration (HPC Environment)

- Multi-core CPU nodes
- NVIDIA Tesla V100 GPUs (16GB)
- CUDA-enabled environment
- SLURM + OpenMPI setup for distributed execution

---

## 🔁 Training Pipeline

1. Resize CIFAR-10 images to 224×224
2. Apply strong data augmentation
3. Load pretrained MobileNetV2
4. Replace classifier layer for 10 classes
5. Train using:
   - Serial baseline
   - MPI + DDP distributed setup
6. Synchronize gradients across GPUs using All-Reduce
7. Save best model checkpoints
8. Deploy model via Gradio for inference

---

## 📊 Performance Comparison

| Execution Mode | Hardware | Training Time (min) | Best Test Accuracy |
|----------------|----------|---------------------|--------------------|
| CPU Serial | Single CPU | *515.92* | 96.17% |
| GPU Serial | 1 GPU | *178.30* | 96.03% |
| Distributed (MPI+DDP) | 2 GPUs | *87.01* | 95.58% |

### ⚡ Speedup
- GPU vs CPU → *2.9× faster*
- Distributed vs CPU → *~6× faster*

---

## 🌐 Inference Interface (Gradio)

Users can upload an image and get *Top-3 predictions* with confidence scores through a simple web UI.
