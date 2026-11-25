# Lab 2: HLS Optimization of Matrix Multiply and Sobel Filter

I wrote a matrix multiply (100×150 by 150×200) and a Sobel edge filter in C++ and synthesized them with Vitis HLS for the Zynq xc7z020 FPGA. Then I tried each HLS pragma on its own and in combination to see how it changes latency and resource use.

## Matrix multiply results

| Version | Latency (cycles) | II | LUT | FF | BRAM | DSP |
|---|---|---|---|---|---|---|
| Baseline (no pragmas) | 3,310,041 | - | 3,433 | 1,912 | 80 | 6 |
| Unroll factor 4 | 2,650,041 | 3 | 3,523 | 1,914 | 80 | 13 |
| Array partition | 3,290,040 | - | 15,863 | 6,578 | 332 | 4 |
| Pipeline | 250,055 | 9 | 13,704 | 6,686 | 438 | 154 |
| **Array partition + pipeline** | **90,052** | **1** | 15,472 | 2,578 | 332 | 153 |

Partition and pipeline together gave the best result, about 37x faster than the baseline. Pipelining alone was stuck at II = 9 because each BRAM has only two ports and can't feed the 150 multipliers fast enough. Partitioning the arrays removed that limit, so II reached 1.

## Sobel filter results

The baseline took 3,670,427 cycles. Array reshape (cyclic, factor 4) with unroll factor 4 brought it down to 697,179 cycles, about 5.3x faster.

## Files

| File | Contents |
|---|---|
| `synthesis_reports/` | Vitis HLS report for each optimization (matmul `*_p1.txt`, Sobel `sobel_*.txt`) |
| `Makefile` | Build script for C simulation and synthesis |
| `report.pdf` | Full lab report with screenshots and discussion |
