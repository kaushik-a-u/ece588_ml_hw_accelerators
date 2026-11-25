#include <iostream>
#include <fstream>
#include <cmath>
#include "utils.hpp"

using namespace std;

void tiled_conv(
    fm_t input_fm[M][I][I],
    wt_t weights[N][M][K][K],
    wt_t biases[N],
    fm_t output_fm[N][O][O])
{
#pragma HLS INTERFACE m_axi depth = 1 port = input_fm bundle = fm
#pragma HLS INTERFACE m_axi depth = 1 port = weights bundle = wt
#pragma HLS INTERFACE m_axi depth = 1 port = biases bundle = wt
#pragma HLS INTERFACE m_axi depth = 1 port = output_fm bundle = fm

#pragma HLS INTERFACE s_axilite register port = return

    fm_t conv_in_buf[IN_BUF_DEPTH][IN_BUF_HW][IN_BUF_HW];
    wt_t conv_wt_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][K][K];
    wt_t conv_bias_buf[OUT_BUF_DEPTH];
    fm_t conv_out_buf[OUT_BUF_DEPTH][OUT_BUF_HW][OUT_BUF_HW];

    // ARRAY_PARTITION: split the buffers so the convolution can read all the
    // values it needs for one output pixel in the same clock cycle.
    //
    // Input tile: one bank per channel (dim 1), and rows/columns interleaved
    // in groups of 3 (dims 2 and 3). The 3x3 window always touches 3 neighbouring
    // rows and 3 neighbouring columns, so each of its 9 values per channel falls
    // in a different bank.
#pragma HLS ARRAY_PARTITION variable = conv_in_buf complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_in_buf cyclic factor = 3 dim = 2
#pragma HLS ARRAY_PARTITION variable = conv_in_buf cyclic factor = 3 dim = 3
    // Weights: one pixel needs every channel and every kernel position of ONE
    // filter, so split dims 2, 3 and 4 completely (the filter index stays a
    // normal memory address).
#pragma HLS ARRAY_PARTITION variable = conv_wt_buf complete dim = 2
#pragma HLS ARRAY_PARTITION variable = conv_wt_buf complete dim = 3
#pragma HLS ARRAY_PARTITION variable = conv_wt_buf complete dim = 4
    // Biases are tiny, so make them plain registers.
#pragma HLS ARRAY_PARTITION variable = conv_bias_buf complete dim = 1

    for (int tr = 0; tr < NUM_TILE_RC; tr++)
    {
        for (int tc = 0; tc < NUM_TILE_RC; tc++)
        {
            std::cout << "Processing Tile " << tr * NUM_TILE_RC + tc + 1;
            std::cout << "/" << NUM_TILE_RC * NUM_TILE_RC << std::endl;

            // Load the input tile (with padding margin) once; reuse for every kernel group
            load_input_tile_from_DRAM(input_fm, conv_in_buf, tr, tc);

            for (int kg = 0; kg < NUM_KERNEL_GROUPS; kg++)
            {
                load_layer_params_from_DRAM(weights, biases, conv_wt_buf, conv_bias_buf, kg);
                conv_3x3(conv_in_buf, conv_wt_buf, conv_bias_buf, conv_out_buf);
                store_output_tile_to_DRAM(conv_out_buf, output_fm, tr, tc, kg);
            }
        }
    }
}
