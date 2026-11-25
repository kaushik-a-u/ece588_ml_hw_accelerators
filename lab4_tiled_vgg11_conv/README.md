# Lab 4: Tiled VGG-11 Convolution on FPGA

I built the first convolution layer of VGG-11 (3×224×224 input, 64 filters of 3×3) as an FPGA accelerator. The full image doesn't fit in on-chip memory, so the design splits it into tiles. Each tile is loaded into BRAM, convolved and written back.

## Steps and results

- **Part 1, baseline tiled convolution:** 7×7 tiles. Matches the reference output (MSE 4.60, limit 5).
- **Part 2, design space exploration:** swept tile size (4×4 to 28×28) and filter-group size. I picked 14×14 tiles with 16 filters per group as the best balance of speed and memory.
- **Part 3, HLS pragmas:** array partition and pipeline brought latency from 10,613,505 to 7,499,521 cycles, 29% fewer.
- **Part 4, fixed point:** tested six `ap_fixed` formats from 8 to 32 bits. All passed the accuracy limit. I chose `ap_fixed<8,4>`: 7,384,065 cycles using only 29 DSPs, with MSE 3.62.
- **Part 5, on the board:**

| Measure | Result |
|---|---|
| HLS estimate | 73.84 ms |
| Measured on the FPGA | 93.18 ms |
| Same tiled 8-bit convolution on the ARM CPU (Python) | 1,147.75 s |
| Speedup | 12,318x |
| Output check | FPGA output matches CPU output 100% |

## Files

| Folder | Contents |
|---|---|
| `part1/` to `part5/` | My HLS code for each step (`tiled_conv.cpp`, `tiled_conv.hpp`, `utils.cpp`) |
| `part5/pynq_tiled_conv.ipynb` | Notebook that runs the design on the PYNQ-Z2 and checks the output |
| `report.pdf` | Full lab report with every synthesis run |

The testbench, `utils.hpp` and the layer weights were provided by the course and are not included.
