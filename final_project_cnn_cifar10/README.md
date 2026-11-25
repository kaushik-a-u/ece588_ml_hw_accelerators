# Final Project: VGG-style CNN for CIFAR-10, Prepared for FPGA

Group project with teammate Sathish. The goal is an image classifier for CIFAR-10 that is accurate, fast and small enough to run on the PYNQ-Z2 FPGA.

## Progress so far

- **Model:** a VGG-style CNN with four blocks of two 3×3 convolutions, batch normalization and pooling (about 1.05M parameters).
- **Training:** 30 epochs with a one-cycle learning-rate schedule. 98.75% training accuracy and 92.24% validation accuracy.
- **Test accuracy:**

| Model | Test accuracy |
|---|---|
| Original LeNet-5 | 55.68% |
| Modern LeNet-5 | 65.76% |
| VGG-style (FP32) | 92.40% |

- **Quantization for the FPGA:**

| Method | Matches FP32 prediction | Accuracy |
|---|---|---|
| INT16 per-layer | 100.00% | 92.40% |
| INT16 per-channel | 100.00% | 92.40% |
| INT8 per-layer | 99.80% | 92.46% |
| INT8 per-channel | 99.86% | 92.41% |

8-bit weights keep the FP32 accuracy and need 4x less storage (about 5.7 MB instead of 23 MB), which suits the 8-bit DSP blocks on the PYNQ-Z2.

**Next:** build the network in Vitis HLS, deploy it on the board, and compare FPGA and GPU accuracy, latency and resource use.

## Files

| File | Contents |
|---|---|
| `cnn_cifar10.ipynb` | Training, evaluation and quantization notebook |
| `initial_report.pdf` | Initial report comparing LeNet-5 variants and the VGG-style model |
| `progress_report.pdf` | Progress report with accuracy, latency and quantization results |
| `figures/` | Accuracy, loss and confusion-matrix plots for each model |
