/**
 * @file tiled_conv.hpp
 * @brief Configuration and type definitions for tiled convolution operations (Part 4, fixed point).
 */
#ifndef __TILED_CONV_HPP__
#define __TILED_CONV_HPP__

#include <ap_fixed.h>

//--------------------------------------------------------------------------
// FIXED POINT SETTINGS: change only these two lines for the Part 4 experiments
// Baseline : ap_fixed<16, 4>
//   FM_W = total number of bits
//   FM_I = number of integer bits (the remaining bits are fractional)
//--------------------------------------------------------------------------
#define FM_W 8
#define FM_I 4
typedef ap_fixed<FM_W, FM_I> fm_t, wt_t;

// Wider type used only to add up the 27 products of one output pixel, so the
// sum does not overflow or lose precision before the final conversion to fm_t.
// (A product of two <W,I> numbers is <2W,2I>; 5 extra bits leave room for 28 adds.)
typedef ap_fixed<2 * FM_W + 5, 2 * FM_I + 5> acc_t;

//--------------------------------------------------------------------------
// DO NOT CHANGE THE VGG11 First Convolution LAYER VALUES BELOW
//--------------------------------------------------------------------------
// IN_FM_DEPTH
#define M 3
// IN_FM_HEIGHT/WIDTH
#define I 224
// KERNEL_HEIGHT/WIDTH
#define K 3
// STRIDE
#define S 1
// PADDING
#define P 1
// OUT_FM_DEPTH
#define N 64
// OUT_FM_HEIGHT/WIDTH
#define O 224
/////////////////////////////////////////////

//--------------------------------------------------------------------------
// TILING BLOCKS HARDWARE (golden design from Part 3)
//  - Tile Size : 14x14
//  - Kernel groups : 4 groups of 16 kernels
//--------------------------------------------------------------------------
// TILE_HEIGHT/WIDTH
#define TILE_HW 14
// NUM_TILES_ROW/COL (224 / 14 = 16)
#define NUM_TILE_RC (O / TILE_HW)
// MARGIN
#define MARGIN (K - 1)
// IN_BUF_DEPTH
#define IN_BUF_DEPTH M
// IN_BUF_HEIGHT/WIDTH (14 + 2 = 16)
#define IN_BUF_HW (TILE_HW + MARGIN)
// OUT_BUF_DEPTH (kernels per group)
#define OUT_BUF_DEPTH 16
// OUT_BUF_HEIGHT/WIDTH
#define OUT_BUF_HW TILE_HW
// NUM_KERNEL_GROUPS (64 / 16 = 4)
#define NUM_KERNEL_GROUPS (N / OUT_BUF_DEPTH)

/**
 * @brief Perform tiled convolution on the input feature maps.
 *
 * @param input_fm Input feature maps.
 * @param weights Weights for the convolutional layer.
 * @param biases Biases for the convolutional layer.
 * @param output_fm Output feature maps.
 */
void tiled_conv(
    fm_t input_fm[M][I][I],
    wt_t weights[N][M][K][K],
    wt_t biases[N],
    fm_t output_fm[N][O][O]);

#endif
