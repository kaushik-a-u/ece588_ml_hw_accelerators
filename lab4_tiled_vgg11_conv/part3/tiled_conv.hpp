/**
 * @file tiled_conv.hpp
 * @brief Configuration and type definitions for tiled convolution operations.
 *
 * This file contains type definitions and configuration macros for the VGG-11
 * convolutional neural network and its tiled convolution operations.
 */
#ifndef __TILED_CONV_HPP__
#define __TILED_CONV_HPP__

typedef float fm_t, wt_t;
typedef float acc_t;
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
// TODO: ASSIGN VALUES FOR YOUR TILING BLOCKS HARDWARE
// Baseline values :
//                  Tile Size : 14x14
//                  Number of kernel groups : 4
//-------------------------------------------------------------------------- 
// TILE_HEIGHT/WIDTH
#define TILE_HW 14
// NUM_TILES_ROW/COL (224 / 7 = 32)
#define NUM_TILE_RC (O / TILE_HW)
// MARGIN
#define MARGIN (K - 1)
// IN_BUF_DEPTH
#define IN_BUF_DEPTH M
// IN_BUF_HEIGHT/WIDTH
#define IN_BUF_HW (TILE_HW + MARGIN)
// OUT_BUF_DEPTH
#define OUT_BUF_DEPTH 16
// OUT_BUF_HEIGHT/WIDTH
#define OUT_BUF_HW TILE_HW
// NUM_KERNEL_GROUPS (64 / 4 = 16)
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
