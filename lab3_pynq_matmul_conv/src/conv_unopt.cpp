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

    // Perform 11X11 Convolution (stride 4) + bias + ReLU
    for (int f = 0; f < F; f++)
    {
        for (int r = 0; r < O; r++)
        {
            for (int c = 0; c < O; c++)
            {
                fm_t sum = biases[f];
                for (int m = 0; m < M; m++)
                {
                    for (int kr = 0; kr < K; kr++)
                    {
                        for (int kc = 0; kc < K; kc++)
                        {
                            sum += input_fm[m][r * S + kr][c * S + kc] * weights[f][m][kr][kc];
                        }
                    }
                }
                output_fm[f][r][c] = (sum > 0) ? sum : 0;
            }
        }
    }
}
