#include "conv.hpp"

#define S 4 // stride

void conv(
    fm_t input_fm[M][I][I],
    wt_t weights[F][M][K][K],
    wt_t biases[F],
    fm_t output_fm[N][O][O])
{
#pragma HLS interface mode = m_axi depth = 1 port = input_fm offset = slave bundle = mem1
#pragma HLS interface mode = m_axi depth = 1 port = output_fm offset = slave bundle = mem1
#pragma HLS interface mode = m_axi depth = 1 port = weights offset = slave bundle = mem2
#pragma HLS interface mode = m_axi depth = 1 port = biases offset = slave bundle = mem2
#pragma HLS interface mode = s_axilite port = input_fm bundle = control
#pragma HLS interface mode = s_axilite port = weights bundle = control
#pragma HLS interface mode = s_axilite port = biases bundle = control
#pragma HLS interface mode = s_axilite port = output_fm bundle = control
#pragma HLS interface mode = s_axilite port = return bundle = control

    // ---- On-chip buffers (BRAM) ----
    wt_t w_buf[F][M][K][K];   // all kernel weights, loaded once
    wt_t b_buf[F];            // biases
    fm_t in_buf[M][K][I];     // the 11 input rows needed for one output row
    fm_t out_line[O];         // one output row of one filter

    // kc is unrolled in the MAC loop, so give each kc its own memory bank
#pragma HLS ARRAY_PARTITION variable = w_buf dim = 4 complete
    // input column index is c*4 + kc, so (index mod 4) == (kc mod 4) for every c
#pragma HLS ARRAY_PARTITION variable = in_buf dim = 3 cyclic factor = 4

    // ---- Load weights and biases once (burst reads from DRAM) ----
LOAD_W_N:
    for (int n = 0; n < F; n++)
    {
    LOAD_W_M:
        for (int m = 0; m < M; m++)
        {
        LOAD_W_KR:
            for (int kr = 0; kr < K; kr++)
            {
            LOAD_W_KC:
                for (int kc = 0; kc < K; kc++)
                {
#pragma HLS PIPELINE II = 1
                    w_buf[n][m][kr][kc] = weights[n][m][kr][kc];
                }
            }
        }
    }

LOAD_B:
    for (int n = 0; n < F; n++)
    {
#pragma HLS PIPELINE II = 1
        b_buf[n] = biases[n];
    }

    // ---- Main loop: one output row at a time ----
ROW:
    for (int r = 0; r < O; r++)
    {
        // Burst-load the 11 input rows (3 channels) used by this output row
    LOAD_IN_M:
        for (int m = 0; m < M; m++)
        {
        LOAD_IN_I:
            for (int i = 0; i < K; i++)
            {
            LOAD_IN_J:
                for (int j = 0; j < I; j++)
                {
#pragma HLS PIPELINE II = 1
                    in_buf[m][i][j] = input_fm[m][r * S + i][j];
                }
            }
        }

    FILTER:
        for (int n = 0; n < F; n++)
        {
            const fm_t bias = b_buf[n];

        COL:
            for (int c = 0; c < O; c++)
            {
                fm_t acc = bias;
                int m = 0;
                int kr = 0;

                // 33 steps (3 channels x 11 kernel rows); each step does 11 MACs in parallel
            MAC:
                for (int t = 0; t < M * K; t++)
                {
#pragma HLS PIPELINE
                    fm_t p[K];
#pragma HLS ARRAY_PARTITION variable = p complete
                MAC_KC:
                    for (int kc = 0; kc < K; kc++)
                    {
#pragma HLS UNROLL
                        p[kc] = in_buf[m][kr][c * S + kc] * w_buf[n][m][kr][kc];
                    }

                    // balanced adder tree over the 11 products
                    fm_t s0 = p[0] + p[1];
                    fm_t s1 = p[2] + p[3];
                    fm_t s2 = p[4] + p[5];
                    fm_t s3 = p[6] + p[7];
                    fm_t s4 = p[8] + p[9];
                    fm_t t0 = s0 + s1;
                    fm_t t1 = s2 + s3;
                    fm_t t2 = s4 + p[10];
                    fm_t part = (t0 + t1) + t2;

                    acc += part;

                    if (kr == K - 1)
                    {
                        kr = 0;
                        m++;
                    }
                    else
                    {
                        kr++;
                    }
                }

                out_line[c] = (acc > 0) ? acc : 0; // ReLU
            }

            // Burst-write one output row of this filter
        WRITE:
            for (int c = 0; c < O; c++)
            {
#pragma HLS PIPELINE II = 1
                output_fm[n][r][c] = out_line[c];
            }
        }
    }
}
