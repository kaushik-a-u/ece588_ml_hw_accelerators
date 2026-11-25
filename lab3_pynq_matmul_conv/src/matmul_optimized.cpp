#include "matmul.hpp"

extern "C" void matmul(
    bit16_t Matrix_A_DRAM[M][N],
    bit16_t Matrix_B_DRAM[N][K],
    bit16_t Matrix_C_DRAM[M][K])
{
    // ---------- Interfaces: match your exact style ----------
    #pragma HLS interface mode = m_axi depth = 1 port = Matrix_A_DRAM offset = slave bundle = mem1
    #pragma HLS interface mode = m_axi depth = 1 port = Matrix_B_DRAM offset = slave bundle = mem2
    #pragma HLS interface mode = m_axi depth = 1 port = Matrix_C_DRAM offset = slave bundle = mem1
    #pragma HLS interface mode = s_axilite port = Matrix_A_DRAM bundle = control
    #pragma HLS interface mode = s_axilite port = Matrix_B_DRAM bundle = control
    #pragma HLS interface mode = s_axilite port = Matrix_C_DRAM bundle = control
    #pragma HLS interface mode = s_axilite port = return        bundle = control

    // ---------- On-chip buffers ----------
    bit16_t Matrix_A_BRAM[M][N];
    bit16_t Matrix_B_BRAM[N][K];
    bit16_t Matrix_C_BRAM[M][K];

    // Optional: mild partitioning (tune factors later)
    #pragma HLS ARRAY_PARTITION variable=Matrix_A_BRAM cyclic factor=12 dim=2
    #pragma HLS ARRAY_PARTITION variable=Matrix_B_BRAM cyclic factor=12 dim=1
    #pragma HLS ARRAY_PARTITION variable=Matrix_C_BRAM cyclic factor=12 dim=2

    // ---------- Read A ----------
    read_A:
    for (int r = 0; r < M; r++) {
        #pragma HLS PIPELINE II=1
        for (int c = 0; c < N; c++) {
            Matrix_A_BRAM[r][c] = Matrix_A_DRAM[r][c];
        }
    }

    // ---------- Read B ----------
    read_B:
    for (int r = 0; r < N; r++) {
        #pragma HLS PIPELINE II=1
        for (int c = 0; c < K; c++) {
            Matrix_B_BRAM[r][c] = Matrix_B_DRAM[r][c];
        }
    }

    // ---------- Zero C ----------
    zero_C:
    for (int r = 0; r < M; r++) {
        #pragma HLS PIPELINE II=1
        for (int c = 0; c < K; c++) {
            Matrix_C_BRAM[r][c] = 0;
        }
    }

    // ---------- Compute C = A * B ----------
    compute_C:
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < K; c++) {
            #pragma HLS PIPELINE II=1
            for (int dot = 0; dot < N; dot++) {
                #pragma HLS UNROLL factor=1
                // ap_int arithmetic widens the product to 32 bits, then truncates into 16 on +=
                Matrix_C_BRAM[r][c] += (bit16_t)((ap_int<32>)Matrix_A_BRAM[r][dot] * (ap_int<32>)Matrix_B_BRAM[dot][c]);
            }
        }
    }

    // ---------- Write back ----------
    write_C:
    for (int r = 0; r < M; r++) {
        #pragma HLS PIPELINE II=1
        for (int c = 0; c < K; c++) {
            Matrix_C_DRAM[r][c] = Matrix_C_BRAM[r][c];
        }
    }
}

