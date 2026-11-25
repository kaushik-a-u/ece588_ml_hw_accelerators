/**
 * @file utils.cpp
 * @brief Implementation of utility functions for tiled convolution operations.
 *
 * This file contains the implementations of functions that handle loading input tiles,
 * loading layer parameters, performing 3x3 convolution, and storing output tiles.
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include "utils.hpp"

using namespace std;

void load_input_tile_from_DRAM(
    fm_t in_fm[M][I][I],
    fm_t in_fm_buf[IN_BUF_DEPTH][IN_BUF_HW][IN_BUF_HW],
    int tr,
    int tc)
{
    // Top-left corner of the buffer in image coordinates (shifted by padding)
    const int row0 = tr * TILE_HW - P;
    const int col0 = tc * TILE_HW - P;

    for (int m = 0; m < IN_BUF_DEPTH; m++)
    {
        for (int i = 0; i < IN_BUF_HW; i++)
        {
            for (int j = 0; j < IN_BUF_HW; j++)
            {
                int r = row0 + i;
                int c = col0 + j;
                // Outside the image = zero padding
                if (r < 0 || r >= I || c < 0 || c >= I)
                    in_fm_buf[m][i][j] = 0;
                else
                    in_fm_buf[m][i][j] = in_fm[m][r][c];
            }
        }
    }
}

void load_layer_params_from_DRAM(
    wt_t weights[N][M][K][K],
    wt_t biases[N],
    wt_t weights_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][K][K],
    wt_t biases_buf[OUT_BUF_DEPTH],
    int kernel_group)
{
    const int n0 = kernel_group * OUT_BUF_DEPTH;

    for (int n = 0; n < OUT_BUF_DEPTH; n++)
    {
        biases_buf[n] = biases[n0 + n];
        for (int m = 0; m < IN_BUF_DEPTH; m++)
        {
            for (int kr = 0; kr < K; kr++)
            {
                for (int kc = 0; kc < K; kc++)
                {
                    weights_buf[n][m][kr][kc] = weights[n0 + n][m][kr][kc];
                }
            }
        }
    }
}

void conv_3x3(
    fm_t X_buf[IN_BUF_DEPTH][IN_BUF_HW][IN_BUF_HW],
    wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][K][K],
    wt_t B_buf[OUT_BUF_DEPTH],
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HW][OUT_BUF_HW])
{
    for (int n = 0; n < OUT_BUF_DEPTH; n++)
    {
        for (int r = 0; r < OUT_BUF_HW; r++)
        {
            for (int c = 0; c < OUT_BUF_HW; c++)
            {
                fm_t sum = B_buf[n];
                for (int m = 0; m < IN_BUF_DEPTH; m++)
                {
                    for (int kr = 0; kr < K; kr++)
                    {
                        for (int kc = 0; kc < K; kc++)
                        {
                            // stride S = 1; padding already built into X_buf
                            sum += W_buf[n][m][kr][kc] * X_buf[m][r * S + kr][c * S + kc];
                        }
                    }
                }
		if(sum<0) sum=0;
                Y_buf[n][r][c] = sum;
            }
        }
    }
}

void store_output_tile_to_DRAM(
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HW][OUT_BUF_HW],
    fm_t out_fm[N][O][O],
    int tr,
    int tc,
    int kernel_group)
{
    const int n0 = kernel_group * OUT_BUF_DEPTH;
    const int r0 = tr * TILE_HW;
    const int c0 = tc * TILE_HW;

    for (int n = 0; n < OUT_BUF_DEPTH; n++)
    {
        for (int r = 0; r < OUT_BUF_HW; r++)
        {
            for (int c = 0; c < OUT_BUF_HW; c++)
            {
                out_fm[n0 + n][r0 + r][c0 + c] = out_fm_buf[n][r][c];
            }
        }
    }
}
