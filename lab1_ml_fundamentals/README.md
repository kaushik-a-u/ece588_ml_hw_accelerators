# Lab 1: ML Fundamentals

The groundwork for the rest of the course: how neural networks compute, before building them in hardware.

## What I did

- **Part 1, perceptrons as logic gates:** set weights and biases by hand to make AND, OR, NAND and NOT gates, then combined them into XOR and a 2-to-1 MUX. All truth tables matched.
- **Part 2, MLP:** trained a small MLP to learn XOR. The loss fell from about 0.72 to 0.0001.
- **Part 3, CNN on MNIST:**
  - MLP (784 → 256 → 128 → 10): 97.37% test accuracy after 3 epochs
  - AlexNet-style CNN: 99.02% test accuracy after 3 epochs
  - Wrote a convolution by hand and checked it against PyTorch's `conv2d`
- **Roofline study:** measured MobileNetV2, ResNet50 and EfficientNetB0 on CPU and GPU. The GPU cut latency by about 7 to 20x (ResNet50: 143.37 ms → 7.28 ms per image). Both devices used under 15% of their peak compute.

## Files

| File | Contents |
|---|---|
| `part1_perceptron_logic_gates.ipynb` | Perceptron gates, XOR and MUX |
| `part2_mlp.ipynb` | MLP trained on XOR |
| `part3_cnn_mnist.ipynb` | MLP and CNN on MNIST, manual convolution, roofline |
| `report.pdf` | Full lab report |
