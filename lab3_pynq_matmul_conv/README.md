# Lab 3: Running HLS Accelerators on the PYNQ-Z2 Board

This lab takes HLS designs all the way onto real hardware: Vitis HLS → IP export → Vivado block design with the Zynq processor → bitstream → run from a Jupyter notebook on the PYNQ-Z2 board.

## Part 1: Matrix multiply (64×64×64, int16)

| | Unoptimized | Optimized |
|---|---|---|
| HLS latency estimate | 3,050,162 cycles | 90,253 cycles |
| BRAM / DSP | 80 / 7 | 182 / 154 |
| On-board time (FPGA) | 0.463 ms | 0.402 ms |
| On-board time (ARM CPU) | 3.309 ms | 4.047 ms |
| Speedup over CPU | 7.15x | 10.08x |

Both designs matched the CPU result exactly.

## Part 2: AlexNet first convolution layer

The layer takes a 3×224×224 input and produces a 64×55×55 output, using 64 filters of 11×11 with stride 4.

| | Unoptimized | Optimized |
|---|---|---|
| HLS latency estimate | 6,173,528,582 cycles (61.7 s) | 39,393,879 cycles (0.394 s) |
| On-board time (FPGA) | 98.778 s | 0.630 s |

The optimized version keeps the weights and input rows in on-chip BRAM, partitions the arrays and pipelines the 11-tap multiply-accumulate. That cut on-board time by 157x. Both versions matched the reference output (MSE 2.48e-13 for the unoptimized design).

## Files

| Folder | Contents |
|---|---|
| `src/` | HLS C++ for the unoptimized and optimized matmul and convolution |
| `notebooks/` | PYNQ notebooks that load each design, time it and check results |
| `bitstreams/` | `.bit` and `.hwh` files for each design, ready to load on a PYNQ-Z2 |
| `report.pdf` | Full lab report |
