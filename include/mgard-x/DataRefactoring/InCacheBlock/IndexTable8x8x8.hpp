/*
 * Copyright 2023, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: Jan. 15, 2023
 */

#ifndef MGARD_X_MULTI_DIMENSION_8x8x8_TABLE_TEMPLATE
#define MGARD_X_MULTI_DIMENSION_8x8x8_TABLE_TEMPLATE

namespace mgard_x {
// clang-format off

MGARDX_EXEC constexpr int offset8x8x8(SIZE z, SIZE y, SIZE x) {
  return z * 8 * 8 + y * 8 + x;
}

MGARDX_EXEC constexpr int offset8x8x8(SIZE z, SIZE y, SIZE x, SIZE ld1, SIZE ld2) {
  return z * ld1 * ld2 + y * ld1 + x;
}

static constexpr int c1d_x_8x8x8[75][3] = { {0, 0, 1}, {0, 0, 3}, {0, 0, 5}, // X
                                            {0, 2, 1}, {0, 2, 3}, {0, 2, 5},
                                            {0, 4, 1}, {0, 4, 3}, {0, 4, 5},
                                            {0, 6, 1}, {0, 6, 3}, {0, 6, 5},
                                            {0, 7, 1}, {0, 7, 3}, {0, 7, 5},
                                            {2, 0, 1}, {2, 0, 3}, {2, 0, 5},
                                            {2, 2, 1}, {2, 2, 3}, {2, 2, 5},
                                            {2, 4, 1}, {2, 4, 3}, {2, 4, 5},
                                            {2, 6, 1}, {2, 6, 3}, {2, 6, 5},
                                            {2, 7, 1}, {2, 7, 3}, {2, 7, 5},
                                            {4, 0, 1}, {4, 0, 3}, {4, 0, 5},
                                            {4, 2, 1}, {4, 2, 3}, {4, 2, 5},
                                            {4, 4, 1}, {4, 4, 3}, {4, 4, 5},
                                            {4, 6, 1}, {4, 6, 3}, {4, 6, 5},
                                            {4, 7, 1}, {4, 7, 3}, {4, 7, 5},
                                            {6, 0, 1}, {6, 0, 3}, {6, 0, 5},
                                            {6, 2, 1}, {6, 2, 3}, {6, 2, 5},
                                            {6, 4, 1}, {6, 4, 3}, {6, 4, 5},
                                            {6, 6, 1}, {6, 6, 3}, {6, 6, 5},
                                            {6, 7, 1}, {6, 7, 3}, {6, 7, 5},
                                            {7, 0, 1}, {7, 0, 3}, {7, 0, 5},
                                            {7, 2, 1}, {7, 2, 3}, {7, 2, 5},
                                            {7, 4, 1}, {7, 4, 3}, {7, 4, 5},
                                            {7, 6, 1}, {7, 6, 3}, {7, 6, 5},
                                            {7, 7, 1}, {7, 7, 3}, {7, 7, 5}};

static constexpr int c1d_y_8x8x8[75][3] = { {0, 1, 0}, {0, 3, 0}, {0, 5, 0}, // Y
                                            {0, 1, 2}, {0, 3, 2}, {0, 5, 2},
                                            {0, 1, 4}, {0, 3, 4}, {0, 5, 4},
                                            {0, 1, 6}, {0, 3, 6}, {0, 5, 6},
                                            {0, 1, 7}, {0, 3, 7}, {0, 5, 7},
                                            {2, 1, 0}, {2, 3, 0}, {2, 5, 0},
                                            {2, 1, 2}, {2, 3, 2}, {2, 5, 2},
                                            {2, 1, 4}, {2, 3, 4}, {2, 5, 4},
                                            {2, 1, 6}, {2, 3, 6}, {2, 5, 6},
                                            {2, 1, 7}, {2, 3, 7}, {2, 5, 7},
                                            {4, 1, 0}, {4, 3, 0}, {4, 5, 0},
                                            {4, 1, 2}, {4, 3, 2}, {4, 5, 2},
                                            {4, 1, 4}, {4, 3, 4}, {4, 5, 4},
                                            {4, 1, 6}, {4, 3, 6}, {4, 5, 6},
                                            {4, 1, 7}, {4, 3, 7}, {4, 5, 7},
                                            {6, 1, 0}, {6, 3, 0}, {6, 5, 0},
                                            {6, 1, 2}, {6, 3, 2}, {6, 5, 2},
                                            {6, 1, 4}, {6, 3, 4}, {6, 5, 4},
                                            {6, 1, 6}, {6, 3, 6}, {6, 5, 6},
                                            {6, 1, 7}, {6, 3, 7}, {6, 5, 7},
                                            {7, 1, 0}, {7, 3, 0}, {7, 5, 0},
                                            {7, 1, 2}, {7, 3, 2}, {7, 5, 2},
                                            {7, 1, 4}, {7, 3, 4}, {7, 5, 4},
                                            {7, 1, 6}, {7, 3, 6}, {7, 5, 6},
                                            {7, 1, 7}, {7, 3, 7}, {7, 5, 7}};

static constexpr int c1d_z_8x8x8[75][3] = { {1, 0, 0}, {3, 0, 0}, {5, 0, 0}, // Z
                                            {1, 0, 2}, {3, 0, 2}, {5, 0, 2},
                                            {1, 0, 4}, {3, 0, 4}, {5, 0, 4},
                                            {1, 0, 6}, {3, 0, 6}, {5, 0, 6},
                                            {1, 0, 7}, {3, 0, 7}, {5, 0, 7},
                                            {1, 2, 0}, {3, 2, 0}, {5, 2, 0},
                                            {1, 2, 2}, {3, 2, 2}, {5, 2, 2},
                                            {1, 2, 4}, {3, 2, 4}, {5, 2, 4},
                                            {1, 2, 6}, {3, 2, 6}, {5, 2, 6},
                                            {1, 2, 7}, {3, 2, 7}, {5, 2, 7},
                                            {1, 4, 0}, {3, 4, 0}, {5, 4, 0},
                                            {1, 4, 2}, {3, 4, 2}, {5, 4, 2},
                                            {1, 4, 4}, {3, 4, 4}, {5, 4, 4},
                                            {1, 4, 6}, {3, 4, 6}, {5, 4, 6},
                                            {1, 4, 7}, {3, 4, 7}, {5, 4, 7},
                                            {1, 6, 0}, {3, 6, 0}, {5, 6, 0},
                                            {1, 6, 2}, {3, 6, 2}, {5, 6, 2},
                                            {1, 6, 4}, {3, 6, 4}, {5, 6, 4},
                                            {1, 6, 6}, {3, 6, 6}, {5, 6, 6},
                                            {1, 6, 7}, {3, 6, 7}, {5, 6, 7},
                                            {1, 7, 0}, {3, 7, 0}, {5, 7, 0},
                                            {1, 7, 2}, {3, 7, 2}, {5, 7, 2},
                                            {1, 7, 4}, {3, 7, 4}, {5, 7, 4},
                                            {1, 7, 6}, {3, 7, 6}, {5, 7, 6},
                                            {1, 7, 7}, {3, 7, 7}, {5, 7, 7}};

static constexpr int c2d_xy_8x8x8[45][3] = {{0, 1, 1}, {0, 1, 3}, {0, 1, 5}, // XY
                                        {0, 3, 1}, {0, 3, 3}, {0, 3, 5},
                                        {0, 5, 1}, {0, 5, 3}, {0, 5, 5},
                                        {2, 1, 1}, {2, 1, 3}, {2, 1, 5},
                                        {2, 3, 1}, {2, 3, 3}, {2, 3, 5},
                                        {2, 5, 1}, {2, 5, 3}, {2, 5, 5},
                                        {4, 1, 1}, {4, 1, 3}, {4, 1, 5},
                                        {4, 3, 1}, {4, 3, 3}, {4, 3, 5},
                                        {4, 5, 1}, {4, 5, 3}, {4, 5, 5},
                                        {6, 1, 1}, {6, 1, 3}, {6, 1, 5},
                                        {6, 3, 1}, {6, 3, 3}, {6, 3, 5},
                                        {6, 5, 1}, {6, 5, 3}, {6, 5, 5},
                                        {7, 1, 1}, {7, 1, 3}, {7, 1, 5},
                                        {7, 3, 1}, {7, 3, 3}, {7, 3, 5},
                                        {7, 5, 1}, {7, 5, 3}, {7, 5, 5}};

static constexpr int c2d_yz_8x8x8[45][3] = {
                                        {1, 1, 0}, {1, 3, 0}, {1, 5, 0}, // YZ
                                        {1, 1, 2}, {1, 3, 2}, {1, 5, 2},
                                        {1, 1, 4}, {1, 3, 4}, {1, 5, 4},
                                        {1, 1, 6}, {1, 3, 6}, {1, 5, 6},
                                        {1, 1, 7}, {1, 3, 7}, {1, 5, 7},
                                        {3, 1, 0}, {3, 3, 0}, {3, 5, 0},
                                        {3, 1, 2}, {3, 3, 2}, {3, 5, 2},
                                        {3, 1, 4}, {3, 3, 4}, {3, 5, 4},
                                        {3, 1, 6}, {3, 3, 6}, {3, 5, 6},
                                        {3, 1, 7}, {3, 3, 7}, {3, 5, 7},
                                        {5, 1, 0}, {5, 3, 0}, {5, 5, 0},
                                        {5, 1, 2}, {5, 3, 2}, {5, 5, 2},
                                        {5, 1, 4}, {5, 3, 4}, {5, 5, 4},
                                        {5, 1, 6}, {5, 3, 6}, {5, 5, 6},
                                        {5, 1, 7}, {5, 3, 7}, {5, 5, 7}};

static constexpr int c2d_xz_8x8x8[45][3] = {
                                        {1, 0, 1}, {3, 0, 1}, {5, 0, 1}, // XZ
                                        {1, 0, 3}, {3, 0, 3}, {5, 0, 3},
                                        {1, 0, 5}, {3, 0, 5}, {5, 0, 5},
                                        {1, 2, 1}, {3, 2, 1}, {5, 2, 1},
                                        {1, 2, 3}, {3, 2, 3}, {5, 2, 3},
                                        {1, 2, 5}, {3, 2, 5}, {5, 2, 5},
                                        {1, 4, 1}, {3, 4, 1}, {5, 4, 1},
                                        {1, 4, 3}, {3, 4, 3}, {5, 4, 3},
                                        {1, 4, 5}, {3, 4, 5}, {5, 4, 5},
                                        {1, 6, 1}, {3, 6, 1}, {5, 6, 1},
                                        {1, 6, 3}, {3, 6, 3}, {5, 6, 3},
                                        {1, 6, 5}, {3, 6, 5}, {5, 6, 5},
                                        {1, 7, 1}, {3, 7, 1}, {5, 7, 1},
                                        {1, 7, 3}, {3, 7, 3}, {5, 7, 3},
                                        {1, 7, 5}, {3, 7, 5}, {5, 7, 5}};

static constexpr int c3d_xyz_8x8x8[27][3] = {
                                          {1, 1, 1},
                                          {1, 1, 3},
                                          {1, 1, 5},
                                          {1, 3, 1},
                                          {1, 3, 3},
                                          {1, 3, 5},
                                          {1, 5, 1},
                                          {1, 5, 3},
                                          {1, 5, 5},
                                          {3, 1, 1},
                                          {3, 1, 3},
                                          {3, 1, 5},
                                          {3, 3, 1},
                                          {3, 3, 3},
                                          {3, 3, 5},
                                          {3, 5, 1},
                                          {3, 5, 3},
                                          {3, 5, 5},
                                          {5, 1, 1},
                                          {5, 1, 3},
                                          {5, 1, 5},
                                          {5, 3, 1},
                                          {5, 3, 3},
                                          {5, 3, 5},
                                          {5, 5, 1},
                                          {5, 5, 3},
                                          {5, 5, 5}};

static constexpr int coarse_x_8x8x8[5] = {0, 2, 4, 6, 7};

static constexpr int coarse_y_8x8x8[5] = {0, 2, 4, 6, 7};

static constexpr int coarse_z_8x8x8[5] = {0, 2, 4, 6, 7};

MGARDX_EXEC int Coeff1D_M_Offset_8x8x8(SIZE i) {
  static constexpr int offset[225] = {offset8x8x8(c1d_x_8x8x8[0][0], c1d_x_8x8x8[0][1], c1d_x_8x8x8[0][2]), // X
                                      offset8x8x8(c1d_x_8x8x8[1][0], c1d_x_8x8x8[1][1], c1d_x_8x8x8[1][2]),
                                      offset8x8x8(c1d_x_8x8x8[2][0], c1d_x_8x8x8[2][1], c1d_x_8x8x8[2][2]),
                                      offset8x8x8(c1d_x_8x8x8[3][0], c1d_x_8x8x8[3][1], c1d_x_8x8x8[3][2]),
                                      offset8x8x8(c1d_x_8x8x8[4][0], c1d_x_8x8x8[4][1], c1d_x_8x8x8[4][2]),
                                      offset8x8x8(c1d_x_8x8x8[5][0], c1d_x_8x8x8[5][1], c1d_x_8x8x8[5][2]),
                                      offset8x8x8(c1d_x_8x8x8[6][0], c1d_x_8x8x8[6][1], c1d_x_8x8x8[6][2]),
                                      offset8x8x8(c1d_x_8x8x8[7][0], c1d_x_8x8x8[7][1], c1d_x_8x8x8[7][2]),
                                      offset8x8x8(c1d_x_8x8x8[8][0], c1d_x_8x8x8[8][1], c1d_x_8x8x8[8][2]),
                                      offset8x8x8(c1d_x_8x8x8[9][0], c1d_x_8x8x8[9][1], c1d_x_8x8x8[9][2]),
                                      offset8x8x8(c1d_x_8x8x8[10][0], c1d_x_8x8x8[10][1], c1d_x_8x8x8[10][2]),
                                      offset8x8x8(c1d_x_8x8x8[11][0], c1d_x_8x8x8[11][1], c1d_x_8x8x8[11][2]),
                                      offset8x8x8(c1d_x_8x8x8[12][0], c1d_x_8x8x8[12][1], c1d_x_8x8x8[12][2]),
                                      offset8x8x8(c1d_x_8x8x8[13][0], c1d_x_8x8x8[13][1], c1d_x_8x8x8[13][2]),
                                      offset8x8x8(c1d_x_8x8x8[14][0], c1d_x_8x8x8[14][1], c1d_x_8x8x8[14][2]),
                                      offset8x8x8(c1d_x_8x8x8[15][0], c1d_x_8x8x8[15][1], c1d_x_8x8x8[15][2]),
                                      offset8x8x8(c1d_x_8x8x8[16][0], c1d_x_8x8x8[16][1], c1d_x_8x8x8[16][2]),
                                      offset8x8x8(c1d_x_8x8x8[17][0], c1d_x_8x8x8[17][1], c1d_x_8x8x8[17][2]),
                                      offset8x8x8(c1d_x_8x8x8[18][0], c1d_x_8x8x8[18][1], c1d_x_8x8x8[18][2]),
                                      offset8x8x8(c1d_x_8x8x8[19][0], c1d_x_8x8x8[19][1], c1d_x_8x8x8[19][2]),
                                      offset8x8x8(c1d_x_8x8x8[20][0], c1d_x_8x8x8[20][1], c1d_x_8x8x8[20][2]),
                                      offset8x8x8(c1d_x_8x8x8[21][0], c1d_x_8x8x8[21][1], c1d_x_8x8x8[21][2]),
                                      offset8x8x8(c1d_x_8x8x8[22][0], c1d_x_8x8x8[22][1], c1d_x_8x8x8[22][2]),
                                      offset8x8x8(c1d_x_8x8x8[23][0], c1d_x_8x8x8[23][1], c1d_x_8x8x8[23][2]),
                                      offset8x8x8(c1d_x_8x8x8[24][0], c1d_x_8x8x8[24][1], c1d_x_8x8x8[24][2]),
                                      offset8x8x8(c1d_x_8x8x8[25][0], c1d_x_8x8x8[25][1], c1d_x_8x8x8[25][2]),
                                      offset8x8x8(c1d_x_8x8x8[26][0], c1d_x_8x8x8[26][1], c1d_x_8x8x8[26][2]),
                                      offset8x8x8(c1d_x_8x8x8[27][0], c1d_x_8x8x8[27][1], c1d_x_8x8x8[27][2]),
                                      offset8x8x8(c1d_x_8x8x8[28][0], c1d_x_8x8x8[28][1], c1d_x_8x8x8[28][2]),
                                      offset8x8x8(c1d_x_8x8x8[29][0], c1d_x_8x8x8[29][1], c1d_x_8x8x8[29][2]),
                                      offset8x8x8(c1d_x_8x8x8[30][0], c1d_x_8x8x8[30][1], c1d_x_8x8x8[30][2]),
                                      offset8x8x8(c1d_x_8x8x8[31][0], c1d_x_8x8x8[31][1], c1d_x_8x8x8[31][2]),
                                      offset8x8x8(c1d_x_8x8x8[32][0], c1d_x_8x8x8[32][1], c1d_x_8x8x8[32][2]),
                                      offset8x8x8(c1d_x_8x8x8[33][0], c1d_x_8x8x8[33][1], c1d_x_8x8x8[33][2]),
                                      offset8x8x8(c1d_x_8x8x8[34][0], c1d_x_8x8x8[34][1], c1d_x_8x8x8[34][2]),
                                      offset8x8x8(c1d_x_8x8x8[35][0], c1d_x_8x8x8[35][1], c1d_x_8x8x8[35][2]),
                                      offset8x8x8(c1d_x_8x8x8[36][0], c1d_x_8x8x8[36][1], c1d_x_8x8x8[36][2]),
                                      offset8x8x8(c1d_x_8x8x8[37][0], c1d_x_8x8x8[37][1], c1d_x_8x8x8[37][2]),
                                      offset8x8x8(c1d_x_8x8x8[38][0], c1d_x_8x8x8[38][1], c1d_x_8x8x8[38][2]),
                                      offset8x8x8(c1d_x_8x8x8[39][0], c1d_x_8x8x8[39][1], c1d_x_8x8x8[39][2]),
                                      offset8x8x8(c1d_x_8x8x8[40][0], c1d_x_8x8x8[40][1], c1d_x_8x8x8[40][2]),
                                      offset8x8x8(c1d_x_8x8x8[41][0], c1d_x_8x8x8[41][1], c1d_x_8x8x8[41][2]),
                                      offset8x8x8(c1d_x_8x8x8[42][0], c1d_x_8x8x8[42][1], c1d_x_8x8x8[42][2]),
                                      offset8x8x8(c1d_x_8x8x8[43][0], c1d_x_8x8x8[43][1], c1d_x_8x8x8[43][2]),
                                      offset8x8x8(c1d_x_8x8x8[44][0], c1d_x_8x8x8[44][1], c1d_x_8x8x8[44][2]),
                                      offset8x8x8(c1d_x_8x8x8[45][0], c1d_x_8x8x8[45][1], c1d_x_8x8x8[45][2]),
                                      offset8x8x8(c1d_x_8x8x8[46][0], c1d_x_8x8x8[46][1], c1d_x_8x8x8[46][2]),
                                      offset8x8x8(c1d_x_8x8x8[47][0], c1d_x_8x8x8[47][1], c1d_x_8x8x8[47][2]),
                                      offset8x8x8(c1d_x_8x8x8[48][0], c1d_x_8x8x8[48][1], c1d_x_8x8x8[48][2]),
                                      offset8x8x8(c1d_x_8x8x8[49][0], c1d_x_8x8x8[49][1], c1d_x_8x8x8[49][2]),
                                      offset8x8x8(c1d_x_8x8x8[50][0], c1d_x_8x8x8[50][1], c1d_x_8x8x8[50][2]),
                                      offset8x8x8(c1d_x_8x8x8[51][0], c1d_x_8x8x8[51][1], c1d_x_8x8x8[51][2]),
                                      offset8x8x8(c1d_x_8x8x8[52][0], c1d_x_8x8x8[52][1], c1d_x_8x8x8[52][2]),
                                      offset8x8x8(c1d_x_8x8x8[53][0], c1d_x_8x8x8[53][1], c1d_x_8x8x8[53][2]),
                                      offset8x8x8(c1d_x_8x8x8[54][0], c1d_x_8x8x8[54][1], c1d_x_8x8x8[54][2]),
                                      offset8x8x8(c1d_x_8x8x8[55][0], c1d_x_8x8x8[55][1], c1d_x_8x8x8[55][2]),
                                      offset8x8x8(c1d_x_8x8x8[56][0], c1d_x_8x8x8[56][1], c1d_x_8x8x8[56][2]),
                                      offset8x8x8(c1d_x_8x8x8[57][0], c1d_x_8x8x8[57][1], c1d_x_8x8x8[57][2]),
                                      offset8x8x8(c1d_x_8x8x8[58][0], c1d_x_8x8x8[58][1], c1d_x_8x8x8[58][2]),
                                      offset8x8x8(c1d_x_8x8x8[59][0], c1d_x_8x8x8[59][1], c1d_x_8x8x8[59][2]),
                                      offset8x8x8(c1d_x_8x8x8[60][0], c1d_x_8x8x8[60][1], c1d_x_8x8x8[60][2]),
                                      offset8x8x8(c1d_x_8x8x8[61][0], c1d_x_8x8x8[61][1], c1d_x_8x8x8[61][2]),
                                      offset8x8x8(c1d_x_8x8x8[62][0], c1d_x_8x8x8[62][1], c1d_x_8x8x8[62][2]),
                                      offset8x8x8(c1d_x_8x8x8[63][0], c1d_x_8x8x8[63][1], c1d_x_8x8x8[63][2]),
                                      offset8x8x8(c1d_x_8x8x8[64][0], c1d_x_8x8x8[64][1], c1d_x_8x8x8[64][2]),
                                      offset8x8x8(c1d_x_8x8x8[65][0], c1d_x_8x8x8[65][1], c1d_x_8x8x8[65][2]),
                                      offset8x8x8(c1d_x_8x8x8[66][0], c1d_x_8x8x8[66][1], c1d_x_8x8x8[66][2]),
                                      offset8x8x8(c1d_x_8x8x8[67][0], c1d_x_8x8x8[67][1], c1d_x_8x8x8[67][2]),
                                      offset8x8x8(c1d_x_8x8x8[68][0], c1d_x_8x8x8[68][1], c1d_x_8x8x8[68][2]),
                                      offset8x8x8(c1d_x_8x8x8[69][0], c1d_x_8x8x8[69][1], c1d_x_8x8x8[69][2]),
                                      offset8x8x8(c1d_x_8x8x8[70][0], c1d_x_8x8x8[70][1], c1d_x_8x8x8[70][2]),
                                      offset8x8x8(c1d_x_8x8x8[71][0], c1d_x_8x8x8[71][1], c1d_x_8x8x8[71][2]),
                                      offset8x8x8(c1d_x_8x8x8[72][0], c1d_x_8x8x8[72][1], c1d_x_8x8x8[72][2]),
                                      offset8x8x8(c1d_x_8x8x8[73][0], c1d_x_8x8x8[73][1], c1d_x_8x8x8[73][2]),
                                      offset8x8x8(c1d_x_8x8x8[74][0], c1d_x_8x8x8[74][1], c1d_x_8x8x8[74][2]),

                                      offset8x8x8(c1d_y_8x8x8[0][0], c1d_y_8x8x8[0][1], c1d_y_8x8x8[0][2]), // Y
                                      offset8x8x8(c1d_y_8x8x8[1][0], c1d_y_8x8x8[1][1], c1d_y_8x8x8[1][2]),
                                      offset8x8x8(c1d_y_8x8x8[2][0], c1d_y_8x8x8[2][1], c1d_y_8x8x8[2][2]),
                                      offset8x8x8(c1d_y_8x8x8[3][0], c1d_y_8x8x8[3][1], c1d_y_8x8x8[3][2]),
                                      offset8x8x8(c1d_y_8x8x8[4][0], c1d_y_8x8x8[4][1], c1d_y_8x8x8[4][2]),
                                      offset8x8x8(c1d_y_8x8x8[5][0], c1d_y_8x8x8[5][1], c1d_y_8x8x8[5][2]),
                                      offset8x8x8(c1d_y_8x8x8[6][0], c1d_y_8x8x8[6][1], c1d_y_8x8x8[6][2]),
                                      offset8x8x8(c1d_y_8x8x8[7][0], c1d_y_8x8x8[7][1], c1d_y_8x8x8[7][2]),
                                      offset8x8x8(c1d_y_8x8x8[8][0], c1d_y_8x8x8[8][1], c1d_y_8x8x8[8][2]),
                                      offset8x8x8(c1d_y_8x8x8[9][0], c1d_y_8x8x8[9][1], c1d_y_8x8x8[9][2]),
                                      offset8x8x8(c1d_y_8x8x8[10][0], c1d_y_8x8x8[10][1], c1d_y_8x8x8[10][2]),
                                      offset8x8x8(c1d_y_8x8x8[11][0], c1d_y_8x8x8[11][1], c1d_y_8x8x8[11][2]),
                                      offset8x8x8(c1d_y_8x8x8[12][0], c1d_y_8x8x8[12][1], c1d_y_8x8x8[12][2]),
                                      offset8x8x8(c1d_y_8x8x8[13][0], c1d_y_8x8x8[13][1], c1d_y_8x8x8[13][2]),
                                      offset8x8x8(c1d_y_8x8x8[14][0], c1d_y_8x8x8[14][1], c1d_y_8x8x8[14][2]),
                                      offset8x8x8(c1d_y_8x8x8[15][0], c1d_y_8x8x8[15][1], c1d_y_8x8x8[15][2]),
                                      offset8x8x8(c1d_y_8x8x8[16][0], c1d_y_8x8x8[16][1], c1d_y_8x8x8[16][2]),
                                      offset8x8x8(c1d_y_8x8x8[17][0], c1d_y_8x8x8[17][1], c1d_y_8x8x8[17][2]),
                                      offset8x8x8(c1d_y_8x8x8[18][0], c1d_y_8x8x8[18][1], c1d_y_8x8x8[18][2]),
                                      offset8x8x8(c1d_y_8x8x8[19][0], c1d_y_8x8x8[19][1], c1d_y_8x8x8[19][2]),
                                      offset8x8x8(c1d_y_8x8x8[20][0], c1d_y_8x8x8[20][1], c1d_y_8x8x8[20][2]),
                                      offset8x8x8(c1d_y_8x8x8[21][0], c1d_y_8x8x8[21][1], c1d_y_8x8x8[21][2]),
                                      offset8x8x8(c1d_y_8x8x8[22][0], c1d_y_8x8x8[22][1], c1d_y_8x8x8[22][2]),
                                      offset8x8x8(c1d_y_8x8x8[23][0], c1d_y_8x8x8[23][1], c1d_y_8x8x8[23][2]),
                                      offset8x8x8(c1d_y_8x8x8[24][0], c1d_y_8x8x8[24][1], c1d_y_8x8x8[24][2]),
                                      offset8x8x8(c1d_y_8x8x8[25][0], c1d_y_8x8x8[25][1], c1d_y_8x8x8[25][2]),
                                      offset8x8x8(c1d_y_8x8x8[26][0], c1d_y_8x8x8[26][1], c1d_y_8x8x8[26][2]),
                                      offset8x8x8(c1d_y_8x8x8[27][0], c1d_y_8x8x8[27][1], c1d_y_8x8x8[27][2]),
                                      offset8x8x8(c1d_y_8x8x8[28][0], c1d_y_8x8x8[28][1], c1d_y_8x8x8[28][2]),
                                      offset8x8x8(c1d_y_8x8x8[29][0], c1d_y_8x8x8[29][1], c1d_y_8x8x8[29][2]),
                                      offset8x8x8(c1d_y_8x8x8[30][0], c1d_y_8x8x8[30][1], c1d_y_8x8x8[30][2]),
                                      offset8x8x8(c1d_y_8x8x8[31][0], c1d_y_8x8x8[31][1], c1d_y_8x8x8[31][2]),
                                      offset8x8x8(c1d_y_8x8x8[32][0], c1d_y_8x8x8[32][1], c1d_y_8x8x8[32][2]),
                                      offset8x8x8(c1d_y_8x8x8[33][0], c1d_y_8x8x8[33][1], c1d_y_8x8x8[33][2]),
                                      offset8x8x8(c1d_y_8x8x8[34][0], c1d_y_8x8x8[34][1], c1d_y_8x8x8[34][2]),
                                      offset8x8x8(c1d_y_8x8x8[35][0], c1d_y_8x8x8[35][1], c1d_y_8x8x8[35][2]),
                                      offset8x8x8(c1d_y_8x8x8[36][0], c1d_y_8x8x8[36][1], c1d_y_8x8x8[36][2]),
                                      offset8x8x8(c1d_y_8x8x8[37][0], c1d_y_8x8x8[37][1], c1d_y_8x8x8[37][2]),
                                      offset8x8x8(c1d_y_8x8x8[38][0], c1d_y_8x8x8[38][1], c1d_y_8x8x8[38][2]),
                                      offset8x8x8(c1d_y_8x8x8[39][0], c1d_y_8x8x8[39][1], c1d_y_8x8x8[39][2]),
                                      offset8x8x8(c1d_y_8x8x8[40][0], c1d_y_8x8x8[40][1], c1d_y_8x8x8[40][2]),
                                      offset8x8x8(c1d_y_8x8x8[41][0], c1d_y_8x8x8[41][1], c1d_y_8x8x8[41][2]),
                                      offset8x8x8(c1d_y_8x8x8[42][0], c1d_y_8x8x8[42][1], c1d_y_8x8x8[42][2]),
                                      offset8x8x8(c1d_y_8x8x8[43][0], c1d_y_8x8x8[43][1], c1d_y_8x8x8[43][2]),
                                      offset8x8x8(c1d_y_8x8x8[44][0], c1d_y_8x8x8[44][1], c1d_y_8x8x8[44][2]),
                                      offset8x8x8(c1d_y_8x8x8[45][0], c1d_y_8x8x8[45][1], c1d_y_8x8x8[45][2]),
                                      offset8x8x8(c1d_y_8x8x8[46][0], c1d_y_8x8x8[46][1], c1d_y_8x8x8[46][2]),
                                      offset8x8x8(c1d_y_8x8x8[47][0], c1d_y_8x8x8[47][1], c1d_y_8x8x8[47][2]),
                                      offset8x8x8(c1d_y_8x8x8[48][0], c1d_y_8x8x8[48][1], c1d_y_8x8x8[48][2]),
                                      offset8x8x8(c1d_y_8x8x8[49][0], c1d_y_8x8x8[49][1], c1d_y_8x8x8[49][2]),
                                      offset8x8x8(c1d_y_8x8x8[50][0], c1d_y_8x8x8[50][1], c1d_y_8x8x8[50][2]),
                                      offset8x8x8(c1d_y_8x8x8[51][0], c1d_y_8x8x8[51][1], c1d_y_8x8x8[51][2]),
                                      offset8x8x8(c1d_y_8x8x8[52][0], c1d_y_8x8x8[52][1], c1d_y_8x8x8[52][2]),
                                      offset8x8x8(c1d_y_8x8x8[53][0], c1d_y_8x8x8[53][1], c1d_y_8x8x8[53][2]),
                                      offset8x8x8(c1d_y_8x8x8[54][0], c1d_y_8x8x8[54][1], c1d_y_8x8x8[54][2]),
                                      offset8x8x8(c1d_y_8x8x8[55][0], c1d_y_8x8x8[55][1], c1d_y_8x8x8[55][2]),
                                      offset8x8x8(c1d_y_8x8x8[56][0], c1d_y_8x8x8[56][1], c1d_y_8x8x8[56][2]),
                                      offset8x8x8(c1d_y_8x8x8[57][0], c1d_y_8x8x8[57][1], c1d_y_8x8x8[57][2]),
                                      offset8x8x8(c1d_y_8x8x8[58][0], c1d_y_8x8x8[58][1], c1d_y_8x8x8[58][2]),
                                      offset8x8x8(c1d_y_8x8x8[59][0], c1d_y_8x8x8[59][1], c1d_y_8x8x8[59][2]),
                                      offset8x8x8(c1d_y_8x8x8[60][0], c1d_y_8x8x8[60][1], c1d_y_8x8x8[60][2]),
                                      offset8x8x8(c1d_y_8x8x8[61][0], c1d_y_8x8x8[61][1], c1d_y_8x8x8[61][2]),
                                      offset8x8x8(c1d_y_8x8x8[62][0], c1d_y_8x8x8[62][1], c1d_y_8x8x8[62][2]),
                                      offset8x8x8(c1d_y_8x8x8[63][0], c1d_y_8x8x8[63][1], c1d_y_8x8x8[63][2]),
                                      offset8x8x8(c1d_y_8x8x8[64][0], c1d_y_8x8x8[64][1], c1d_y_8x8x8[64][2]),
                                      offset8x8x8(c1d_y_8x8x8[65][0], c1d_y_8x8x8[65][1], c1d_y_8x8x8[65][2]),
                                      offset8x8x8(c1d_y_8x8x8[66][0], c1d_y_8x8x8[66][1], c1d_y_8x8x8[66][2]),
                                      offset8x8x8(c1d_y_8x8x8[67][0], c1d_y_8x8x8[67][1], c1d_y_8x8x8[67][2]),
                                      offset8x8x8(c1d_y_8x8x8[68][0], c1d_y_8x8x8[68][1], c1d_y_8x8x8[68][2]),
                                      offset8x8x8(c1d_y_8x8x8[69][0], c1d_y_8x8x8[69][1], c1d_y_8x8x8[69][2]),
                                      offset8x8x8(c1d_y_8x8x8[70][0], c1d_y_8x8x8[70][1], c1d_y_8x8x8[70][2]),
                                      offset8x8x8(c1d_y_8x8x8[71][0], c1d_y_8x8x8[71][1], c1d_y_8x8x8[71][2]),
                                      offset8x8x8(c1d_y_8x8x8[72][0], c1d_y_8x8x8[72][1], c1d_y_8x8x8[72][2]),
                                      offset8x8x8(c1d_y_8x8x8[73][0], c1d_y_8x8x8[73][1], c1d_y_8x8x8[73][2]),
                                      offset8x8x8(c1d_y_8x8x8[74][0], c1d_y_8x8x8[74][1], c1d_y_8x8x8[74][2]),

                                      offset8x8x8(c1d_z_8x8x8[0][0], c1d_z_8x8x8[0][1], c1d_z_8x8x8[0][2]), // Z
                                      offset8x8x8(c1d_z_8x8x8[1][0], c1d_z_8x8x8[1][1], c1d_z_8x8x8[1][2]),
                                      offset8x8x8(c1d_z_8x8x8[2][0], c1d_z_8x8x8[2][1], c1d_z_8x8x8[2][2]),
                                      offset8x8x8(c1d_z_8x8x8[3][0], c1d_z_8x8x8[3][1], c1d_z_8x8x8[3][2]),
                                      offset8x8x8(c1d_z_8x8x8[4][0], c1d_z_8x8x8[4][1], c1d_z_8x8x8[4][2]),
                                      offset8x8x8(c1d_z_8x8x8[5][0], c1d_z_8x8x8[5][1], c1d_z_8x8x8[5][2]),
                                      offset8x8x8(c1d_z_8x8x8[6][0], c1d_z_8x8x8[6][1], c1d_z_8x8x8[6][2]),
                                      offset8x8x8(c1d_z_8x8x8[7][0], c1d_z_8x8x8[7][1], c1d_z_8x8x8[7][2]),
                                      offset8x8x8(c1d_z_8x8x8[8][0], c1d_z_8x8x8[8][1], c1d_z_8x8x8[8][2]),
                                      offset8x8x8(c1d_z_8x8x8[9][0], c1d_z_8x8x8[9][1], c1d_z_8x8x8[9][2]),
                                      offset8x8x8(c1d_z_8x8x8[10][0], c1d_z_8x8x8[10][1], c1d_z_8x8x8[10][2]),
                                      offset8x8x8(c1d_z_8x8x8[11][0], c1d_z_8x8x8[11][1], c1d_z_8x8x8[11][2]),
                                      offset8x8x8(c1d_z_8x8x8[12][0], c1d_z_8x8x8[12][1], c1d_z_8x8x8[12][2]),
                                      offset8x8x8(c1d_z_8x8x8[13][0], c1d_z_8x8x8[13][1], c1d_z_8x8x8[13][2]),
                                      offset8x8x8(c1d_z_8x8x8[14][0], c1d_z_8x8x8[14][1], c1d_z_8x8x8[14][2]),
                                      offset8x8x8(c1d_z_8x8x8[15][0], c1d_z_8x8x8[15][1], c1d_z_8x8x8[15][2]),
                                      offset8x8x8(c1d_z_8x8x8[16][0], c1d_z_8x8x8[16][1], c1d_z_8x8x8[16][2]),
                                      offset8x8x8(c1d_z_8x8x8[17][0], c1d_z_8x8x8[17][1], c1d_z_8x8x8[17][2]),
                                      offset8x8x8(c1d_z_8x8x8[18][0], c1d_z_8x8x8[18][1], c1d_z_8x8x8[18][2]),
                                      offset8x8x8(c1d_z_8x8x8[19][0], c1d_z_8x8x8[19][1], c1d_z_8x8x8[19][2]),
                                      offset8x8x8(c1d_z_8x8x8[20][0], c1d_z_8x8x8[20][1], c1d_z_8x8x8[20][2]),
                                      offset8x8x8(c1d_z_8x8x8[21][0], c1d_z_8x8x8[21][1], c1d_z_8x8x8[21][2]),
                                      offset8x8x8(c1d_z_8x8x8[22][0], c1d_z_8x8x8[22][1], c1d_z_8x8x8[22][2]),
                                      offset8x8x8(c1d_z_8x8x8[23][0], c1d_z_8x8x8[23][1], c1d_z_8x8x8[23][2]),
                                      offset8x8x8(c1d_z_8x8x8[24][0], c1d_z_8x8x8[24][1], c1d_z_8x8x8[24][2]),
                                      offset8x8x8(c1d_z_8x8x8[25][0], c1d_z_8x8x8[25][1], c1d_z_8x8x8[25][2]),
                                      offset8x8x8(c1d_z_8x8x8[26][0], c1d_z_8x8x8[26][1], c1d_z_8x8x8[26][2]),
                                      offset8x8x8(c1d_z_8x8x8[27][0], c1d_z_8x8x8[27][1], c1d_z_8x8x8[27][2]),
                                      offset8x8x8(c1d_z_8x8x8[28][0], c1d_z_8x8x8[28][1], c1d_z_8x8x8[28][2]),
                                      offset8x8x8(c1d_z_8x8x8[29][0], c1d_z_8x8x8[29][1], c1d_z_8x8x8[29][2]),
                                      offset8x8x8(c1d_z_8x8x8[30][0], c1d_z_8x8x8[30][1], c1d_z_8x8x8[30][2]),
                                      offset8x8x8(c1d_z_8x8x8[31][0], c1d_z_8x8x8[31][1], c1d_z_8x8x8[31][2]),
                                      offset8x8x8(c1d_z_8x8x8[32][0], c1d_z_8x8x8[32][1], c1d_z_8x8x8[32][2]),
                                      offset8x8x8(c1d_z_8x8x8[33][0], c1d_z_8x8x8[33][1], c1d_z_8x8x8[33][2]),
                                      offset8x8x8(c1d_z_8x8x8[34][0], c1d_z_8x8x8[34][1], c1d_z_8x8x8[34][2]),
                                      offset8x8x8(c1d_z_8x8x8[35][0], c1d_z_8x8x8[35][1], c1d_z_8x8x8[35][2]),
                                      offset8x8x8(c1d_z_8x8x8[36][0], c1d_z_8x8x8[36][1], c1d_z_8x8x8[36][2]),
                                      offset8x8x8(c1d_z_8x8x8[37][0], c1d_z_8x8x8[37][1], c1d_z_8x8x8[37][2]),
                                      offset8x8x8(c1d_z_8x8x8[38][0], c1d_z_8x8x8[38][1], c1d_z_8x8x8[38][2]),
                                      offset8x8x8(c1d_z_8x8x8[39][0], c1d_z_8x8x8[39][1], c1d_z_8x8x8[39][2]),
                                      offset8x8x8(c1d_z_8x8x8[40][0], c1d_z_8x8x8[40][1], c1d_z_8x8x8[40][2]),
                                      offset8x8x8(c1d_z_8x8x8[41][0], c1d_z_8x8x8[41][1], c1d_z_8x8x8[41][2]),
                                      offset8x8x8(c1d_z_8x8x8[42][0], c1d_z_8x8x8[42][1], c1d_z_8x8x8[42][2]),
                                      offset8x8x8(c1d_z_8x8x8[43][0], c1d_z_8x8x8[43][1], c1d_z_8x8x8[43][2]),
                                      offset8x8x8(c1d_z_8x8x8[44][0], c1d_z_8x8x8[44][1], c1d_z_8x8x8[44][2]),
                                      offset8x8x8(c1d_z_8x8x8[45][0], c1d_z_8x8x8[45][1], c1d_z_8x8x8[45][2]),
                                      offset8x8x8(c1d_z_8x8x8[46][0], c1d_z_8x8x8[46][1], c1d_z_8x8x8[46][2]),
                                      offset8x8x8(c1d_z_8x8x8[47][0], c1d_z_8x8x8[47][1], c1d_z_8x8x8[47][2]),
                                      offset8x8x8(c1d_z_8x8x8[48][0], c1d_z_8x8x8[48][1], c1d_z_8x8x8[48][2]),
                                      offset8x8x8(c1d_z_8x8x8[49][0], c1d_z_8x8x8[49][1], c1d_z_8x8x8[49][2]),
                                      offset8x8x8(c1d_z_8x8x8[50][0], c1d_z_8x8x8[50][1], c1d_z_8x8x8[50][2]),
                                      offset8x8x8(c1d_z_8x8x8[51][0], c1d_z_8x8x8[51][1], c1d_z_8x8x8[51][2]),
                                      offset8x8x8(c1d_z_8x8x8[52][0], c1d_z_8x8x8[52][1], c1d_z_8x8x8[52][2]),
                                      offset8x8x8(c1d_z_8x8x8[53][0], c1d_z_8x8x8[53][1], c1d_z_8x8x8[53][2]),
                                      offset8x8x8(c1d_z_8x8x8[54][0], c1d_z_8x8x8[54][1], c1d_z_8x8x8[54][2]),
                                      offset8x8x8(c1d_z_8x8x8[55][0], c1d_z_8x8x8[55][1], c1d_z_8x8x8[55][2]),
                                      offset8x8x8(c1d_z_8x8x8[56][0], c1d_z_8x8x8[56][1], c1d_z_8x8x8[56][2]),
                                      offset8x8x8(c1d_z_8x8x8[57][0], c1d_z_8x8x8[57][1], c1d_z_8x8x8[57][2]),
                                      offset8x8x8(c1d_z_8x8x8[58][0], c1d_z_8x8x8[58][1], c1d_z_8x8x8[58][2]),
                                      offset8x8x8(c1d_z_8x8x8[59][0], c1d_z_8x8x8[59][1], c1d_z_8x8x8[59][2]),
                                      offset8x8x8(c1d_z_8x8x8[60][0], c1d_z_8x8x8[60][1], c1d_z_8x8x8[60][2]),
                                      offset8x8x8(c1d_z_8x8x8[61][0], c1d_z_8x8x8[61][1], c1d_z_8x8x8[61][2]),
                                      offset8x8x8(c1d_z_8x8x8[62][0], c1d_z_8x8x8[62][1], c1d_z_8x8x8[62][2]),
                                      offset8x8x8(c1d_z_8x8x8[63][0], c1d_z_8x8x8[63][1], c1d_z_8x8x8[63][2]),
                                      offset8x8x8(c1d_z_8x8x8[64][0], c1d_z_8x8x8[64][1], c1d_z_8x8x8[64][2]),
                                      offset8x8x8(c1d_z_8x8x8[65][0], c1d_z_8x8x8[65][1], c1d_z_8x8x8[65][2]),
                                      offset8x8x8(c1d_z_8x8x8[66][0], c1d_z_8x8x8[66][1], c1d_z_8x8x8[66][2]),
                                      offset8x8x8(c1d_z_8x8x8[67][0], c1d_z_8x8x8[67][1], c1d_z_8x8x8[67][2]),
                                      offset8x8x8(c1d_z_8x8x8[68][0], c1d_z_8x8x8[68][1], c1d_z_8x8x8[68][2]),
                                      offset8x8x8(c1d_z_8x8x8[69][0], c1d_z_8x8x8[69][1], c1d_z_8x8x8[69][2]),
                                      offset8x8x8(c1d_z_8x8x8[70][0], c1d_z_8x8x8[70][1], c1d_z_8x8x8[70][2]),
                                      offset8x8x8(c1d_z_8x8x8[71][0], c1d_z_8x8x8[71][1], c1d_z_8x8x8[71][2]),
                                      offset8x8x8(c1d_z_8x8x8[72][0], c1d_z_8x8x8[72][1], c1d_z_8x8x8[72][2]),
                                      offset8x8x8(c1d_z_8x8x8[73][0], c1d_z_8x8x8[73][1], c1d_z_8x8x8[73][2]),
                                      offset8x8x8(c1d_z_8x8x8[74][0], c1d_z_8x8x8[74][1], c1d_z_8x8x8[74][2])};

  return offset[i];
}

MGARDX_EXEC int Coeff1D_L_Offset_8x8x8(SIZE i) {
  static constexpr int offset[225] = {offset8x8x8(c1d_x_8x8x8[0][0], c1d_x_8x8x8[0][1], c1d_x_8x8x8[0][2]-1), // X
                                      offset8x8x8(c1d_x_8x8x8[1][0], c1d_x_8x8x8[1][1], c1d_x_8x8x8[1][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[2][0], c1d_x_8x8x8[2][1], c1d_x_8x8x8[2][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[3][0], c1d_x_8x8x8[3][1], c1d_x_8x8x8[3][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[4][0], c1d_x_8x8x8[4][1], c1d_x_8x8x8[4][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[5][0], c1d_x_8x8x8[5][1], c1d_x_8x8x8[5][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[6][0], c1d_x_8x8x8[6][1], c1d_x_8x8x8[6][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[7][0], c1d_x_8x8x8[7][1], c1d_x_8x8x8[7][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[8][0], c1d_x_8x8x8[8][1], c1d_x_8x8x8[8][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[9][0], c1d_x_8x8x8[9][1], c1d_x_8x8x8[9][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[10][0], c1d_x_8x8x8[10][1], c1d_x_8x8x8[10][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[11][0], c1d_x_8x8x8[11][1], c1d_x_8x8x8[11][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[12][0], c1d_x_8x8x8[12][1], c1d_x_8x8x8[12][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[13][0], c1d_x_8x8x8[13][1], c1d_x_8x8x8[13][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[14][0], c1d_x_8x8x8[14][1], c1d_x_8x8x8[14][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[15][0], c1d_x_8x8x8[15][1], c1d_x_8x8x8[15][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[16][0], c1d_x_8x8x8[16][1], c1d_x_8x8x8[16][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[17][0], c1d_x_8x8x8[17][1], c1d_x_8x8x8[17][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[18][0], c1d_x_8x8x8[18][1], c1d_x_8x8x8[18][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[19][0], c1d_x_8x8x8[19][1], c1d_x_8x8x8[19][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[20][0], c1d_x_8x8x8[20][1], c1d_x_8x8x8[20][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[21][0], c1d_x_8x8x8[21][1], c1d_x_8x8x8[21][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[22][0], c1d_x_8x8x8[22][1], c1d_x_8x8x8[22][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[23][0], c1d_x_8x8x8[23][1], c1d_x_8x8x8[23][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[24][0], c1d_x_8x8x8[24][1], c1d_x_8x8x8[24][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[25][0], c1d_x_8x8x8[25][1], c1d_x_8x8x8[25][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[26][0], c1d_x_8x8x8[26][1], c1d_x_8x8x8[26][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[27][0], c1d_x_8x8x8[27][1], c1d_x_8x8x8[27][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[28][0], c1d_x_8x8x8[28][1], c1d_x_8x8x8[28][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[29][0], c1d_x_8x8x8[29][1], c1d_x_8x8x8[29][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[30][0], c1d_x_8x8x8[30][1], c1d_x_8x8x8[30][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[31][0], c1d_x_8x8x8[31][1], c1d_x_8x8x8[31][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[32][0], c1d_x_8x8x8[32][1], c1d_x_8x8x8[32][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[33][0], c1d_x_8x8x8[33][1], c1d_x_8x8x8[33][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[34][0], c1d_x_8x8x8[34][1], c1d_x_8x8x8[34][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[35][0], c1d_x_8x8x8[35][1], c1d_x_8x8x8[35][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[36][0], c1d_x_8x8x8[36][1], c1d_x_8x8x8[36][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[37][0], c1d_x_8x8x8[37][1], c1d_x_8x8x8[37][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[38][0], c1d_x_8x8x8[38][1], c1d_x_8x8x8[38][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[39][0], c1d_x_8x8x8[39][1], c1d_x_8x8x8[39][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[40][0], c1d_x_8x8x8[40][1], c1d_x_8x8x8[40][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[41][0], c1d_x_8x8x8[41][1], c1d_x_8x8x8[41][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[42][0], c1d_x_8x8x8[42][1], c1d_x_8x8x8[42][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[43][0], c1d_x_8x8x8[43][1], c1d_x_8x8x8[43][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[44][0], c1d_x_8x8x8[44][1], c1d_x_8x8x8[44][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[45][0], c1d_x_8x8x8[45][1], c1d_x_8x8x8[45][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[46][0], c1d_x_8x8x8[46][1], c1d_x_8x8x8[46][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[47][0], c1d_x_8x8x8[47][1], c1d_x_8x8x8[47][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[48][0], c1d_x_8x8x8[48][1], c1d_x_8x8x8[48][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[49][0], c1d_x_8x8x8[49][1], c1d_x_8x8x8[49][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[50][0], c1d_x_8x8x8[50][1], c1d_x_8x8x8[50][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[51][0], c1d_x_8x8x8[51][1], c1d_x_8x8x8[51][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[52][0], c1d_x_8x8x8[52][1], c1d_x_8x8x8[52][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[53][0], c1d_x_8x8x8[53][1], c1d_x_8x8x8[53][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[54][0], c1d_x_8x8x8[54][1], c1d_x_8x8x8[54][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[55][0], c1d_x_8x8x8[55][1], c1d_x_8x8x8[55][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[56][0], c1d_x_8x8x8[56][1], c1d_x_8x8x8[56][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[57][0], c1d_x_8x8x8[57][1], c1d_x_8x8x8[57][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[58][0], c1d_x_8x8x8[58][1], c1d_x_8x8x8[58][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[59][0], c1d_x_8x8x8[59][1], c1d_x_8x8x8[59][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[60][0], c1d_x_8x8x8[60][1], c1d_x_8x8x8[60][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[61][0], c1d_x_8x8x8[61][1], c1d_x_8x8x8[61][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[62][0], c1d_x_8x8x8[62][1], c1d_x_8x8x8[62][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[63][0], c1d_x_8x8x8[63][1], c1d_x_8x8x8[63][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[64][0], c1d_x_8x8x8[64][1], c1d_x_8x8x8[64][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[65][0], c1d_x_8x8x8[65][1], c1d_x_8x8x8[65][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[66][0], c1d_x_8x8x8[66][1], c1d_x_8x8x8[66][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[67][0], c1d_x_8x8x8[67][1], c1d_x_8x8x8[67][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[68][0], c1d_x_8x8x8[68][1], c1d_x_8x8x8[68][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[69][0], c1d_x_8x8x8[69][1], c1d_x_8x8x8[69][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[70][0], c1d_x_8x8x8[70][1], c1d_x_8x8x8[70][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[71][0], c1d_x_8x8x8[71][1], c1d_x_8x8x8[71][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[72][0], c1d_x_8x8x8[72][1], c1d_x_8x8x8[72][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[73][0], c1d_x_8x8x8[73][1], c1d_x_8x8x8[73][2]-1),
                                      offset8x8x8(c1d_x_8x8x8[74][0], c1d_x_8x8x8[74][1], c1d_x_8x8x8[74][2]-1),

                                      offset8x8x8(c1d_y_8x8x8[0][0], c1d_y_8x8x8[0][1]-1, c1d_y_8x8x8[0][2]), // Y
                                      offset8x8x8(c1d_y_8x8x8[1][0], c1d_y_8x8x8[1][1]-1, c1d_y_8x8x8[1][2]),
                                      offset8x8x8(c1d_y_8x8x8[2][0], c1d_y_8x8x8[2][1]-1, c1d_y_8x8x8[2][2]),
                                      offset8x8x8(c1d_y_8x8x8[3][0], c1d_y_8x8x8[3][1]-1, c1d_y_8x8x8[3][2]),
                                      offset8x8x8(c1d_y_8x8x8[4][0], c1d_y_8x8x8[4][1]-1, c1d_y_8x8x8[4][2]),
                                      offset8x8x8(c1d_y_8x8x8[5][0], c1d_y_8x8x8[5][1]-1, c1d_y_8x8x8[5][2]),
                                      offset8x8x8(c1d_y_8x8x8[6][0], c1d_y_8x8x8[6][1]-1, c1d_y_8x8x8[6][2]),
                                      offset8x8x8(c1d_y_8x8x8[7][0], c1d_y_8x8x8[7][1]-1, c1d_y_8x8x8[7][2]),
                                      offset8x8x8(c1d_y_8x8x8[8][0], c1d_y_8x8x8[8][1]-1, c1d_y_8x8x8[8][2]),
                                      offset8x8x8(c1d_y_8x8x8[9][0], c1d_y_8x8x8[9][1]-1, c1d_y_8x8x8[9][2]),
                                      offset8x8x8(c1d_y_8x8x8[10][0], c1d_y_8x8x8[10][1]-1, c1d_y_8x8x8[10][2]),
                                      offset8x8x8(c1d_y_8x8x8[11][0], c1d_y_8x8x8[11][1]-1, c1d_y_8x8x8[11][2]),
                                      offset8x8x8(c1d_y_8x8x8[12][0], c1d_y_8x8x8[12][1]-1, c1d_y_8x8x8[12][2]),
                                      offset8x8x8(c1d_y_8x8x8[13][0], c1d_y_8x8x8[13][1]-1, c1d_y_8x8x8[13][2]),
                                      offset8x8x8(c1d_y_8x8x8[14][0], c1d_y_8x8x8[14][1]-1, c1d_y_8x8x8[14][2]),
                                      offset8x8x8(c1d_y_8x8x8[15][0], c1d_y_8x8x8[15][1]-1, c1d_y_8x8x8[15][2]),
                                      offset8x8x8(c1d_y_8x8x8[16][0], c1d_y_8x8x8[16][1]-1, c1d_y_8x8x8[16][2]),
                                      offset8x8x8(c1d_y_8x8x8[17][0], c1d_y_8x8x8[17][1]-1, c1d_y_8x8x8[17][2]),
                                      offset8x8x8(c1d_y_8x8x8[18][0], c1d_y_8x8x8[18][1]-1, c1d_y_8x8x8[18][2]),
                                      offset8x8x8(c1d_y_8x8x8[19][0], c1d_y_8x8x8[19][1]-1, c1d_y_8x8x8[19][2]),
                                      offset8x8x8(c1d_y_8x8x8[20][0], c1d_y_8x8x8[20][1]-1, c1d_y_8x8x8[20][2]),
                                      offset8x8x8(c1d_y_8x8x8[21][0], c1d_y_8x8x8[21][1]-1, c1d_y_8x8x8[21][2]),
                                      offset8x8x8(c1d_y_8x8x8[22][0], c1d_y_8x8x8[22][1]-1, c1d_y_8x8x8[22][2]),
                                      offset8x8x8(c1d_y_8x8x8[23][0], c1d_y_8x8x8[23][1]-1, c1d_y_8x8x8[23][2]),
                                      offset8x8x8(c1d_y_8x8x8[24][0], c1d_y_8x8x8[24][1]-1, c1d_y_8x8x8[24][2]),
                                      offset8x8x8(c1d_y_8x8x8[25][0], c1d_y_8x8x8[25][1]-1, c1d_y_8x8x8[25][2]),
                                      offset8x8x8(c1d_y_8x8x8[26][0], c1d_y_8x8x8[26][1]-1, c1d_y_8x8x8[26][2]),
                                      offset8x8x8(c1d_y_8x8x8[27][0], c1d_y_8x8x8[27][1]-1, c1d_y_8x8x8[27][2]),
                                      offset8x8x8(c1d_y_8x8x8[28][0], c1d_y_8x8x8[28][1]-1, c1d_y_8x8x8[28][2]),
                                      offset8x8x8(c1d_y_8x8x8[29][0], c1d_y_8x8x8[29][1]-1, c1d_y_8x8x8[29][2]),
                                      offset8x8x8(c1d_y_8x8x8[30][0], c1d_y_8x8x8[30][1]-1, c1d_y_8x8x8[30][2]),
                                      offset8x8x8(c1d_y_8x8x8[31][0], c1d_y_8x8x8[31][1]-1, c1d_y_8x8x8[31][2]),
                                      offset8x8x8(c1d_y_8x8x8[32][0], c1d_y_8x8x8[32][1]-1, c1d_y_8x8x8[32][2]),
                                      offset8x8x8(c1d_y_8x8x8[33][0], c1d_y_8x8x8[33][1]-1, c1d_y_8x8x8[33][2]),
                                      offset8x8x8(c1d_y_8x8x8[34][0], c1d_y_8x8x8[34][1]-1, c1d_y_8x8x8[34][2]),
                                      offset8x8x8(c1d_y_8x8x8[35][0], c1d_y_8x8x8[35][1]-1, c1d_y_8x8x8[35][2]),
                                      offset8x8x8(c1d_y_8x8x8[36][0], c1d_y_8x8x8[36][1]-1, c1d_y_8x8x8[36][2]),
                                      offset8x8x8(c1d_y_8x8x8[37][0], c1d_y_8x8x8[37][1]-1, c1d_y_8x8x8[37][2]),
                                      offset8x8x8(c1d_y_8x8x8[38][0], c1d_y_8x8x8[38][1]-1, c1d_y_8x8x8[38][2]),
                                      offset8x8x8(c1d_y_8x8x8[39][0], c1d_y_8x8x8[39][1]-1, c1d_y_8x8x8[39][2]),
                                      offset8x8x8(c1d_y_8x8x8[40][0], c1d_y_8x8x8[40][1]-1, c1d_y_8x8x8[40][2]),
                                      offset8x8x8(c1d_y_8x8x8[41][0], c1d_y_8x8x8[41][1]-1, c1d_y_8x8x8[41][2]),
                                      offset8x8x8(c1d_y_8x8x8[42][0], c1d_y_8x8x8[42][1]-1, c1d_y_8x8x8[42][2]),
                                      offset8x8x8(c1d_y_8x8x8[43][0], c1d_y_8x8x8[43][1]-1, c1d_y_8x8x8[43][2]),
                                      offset8x8x8(c1d_y_8x8x8[44][0], c1d_y_8x8x8[44][1]-1, c1d_y_8x8x8[44][2]),
                                      offset8x8x8(c1d_y_8x8x8[45][0], c1d_y_8x8x8[45][1]-1, c1d_y_8x8x8[45][2]),
                                      offset8x8x8(c1d_y_8x8x8[46][0], c1d_y_8x8x8[46][1]-1, c1d_y_8x8x8[46][2]),
                                      offset8x8x8(c1d_y_8x8x8[47][0], c1d_y_8x8x8[47][1]-1, c1d_y_8x8x8[47][2]),
                                      offset8x8x8(c1d_y_8x8x8[48][0], c1d_y_8x8x8[48][1]-1, c1d_y_8x8x8[48][2]),
                                      offset8x8x8(c1d_y_8x8x8[49][0], c1d_y_8x8x8[49][1]-1, c1d_y_8x8x8[49][2]),
                                      offset8x8x8(c1d_y_8x8x8[50][0], c1d_y_8x8x8[50][1]-1, c1d_y_8x8x8[50][2]),
                                      offset8x8x8(c1d_y_8x8x8[51][0], c1d_y_8x8x8[51][1]-1, c1d_y_8x8x8[51][2]),
                                      offset8x8x8(c1d_y_8x8x8[52][0], c1d_y_8x8x8[52][1]-1, c1d_y_8x8x8[52][2]),
                                      offset8x8x8(c1d_y_8x8x8[53][0], c1d_y_8x8x8[53][1]-1, c1d_y_8x8x8[53][2]),
                                      offset8x8x8(c1d_y_8x8x8[54][0], c1d_y_8x8x8[54][1]-1, c1d_y_8x8x8[54][2]),
                                      offset8x8x8(c1d_y_8x8x8[55][0], c1d_y_8x8x8[55][1]-1, c1d_y_8x8x8[55][2]),
                                      offset8x8x8(c1d_y_8x8x8[56][0], c1d_y_8x8x8[56][1]-1, c1d_y_8x8x8[56][2]),
                                      offset8x8x8(c1d_y_8x8x8[57][0], c1d_y_8x8x8[57][1]-1, c1d_y_8x8x8[57][2]),
                                      offset8x8x8(c1d_y_8x8x8[58][0], c1d_y_8x8x8[58][1]-1, c1d_y_8x8x8[58][2]),
                                      offset8x8x8(c1d_y_8x8x8[59][0], c1d_y_8x8x8[59][1]-1, c1d_y_8x8x8[59][2]),
                                      offset8x8x8(c1d_y_8x8x8[60][0], c1d_y_8x8x8[60][1]-1, c1d_y_8x8x8[60][2]),
                                      offset8x8x8(c1d_y_8x8x8[61][0], c1d_y_8x8x8[61][1]-1, c1d_y_8x8x8[61][2]),
                                      offset8x8x8(c1d_y_8x8x8[62][0], c1d_y_8x8x8[62][1]-1, c1d_y_8x8x8[62][2]),
                                      offset8x8x8(c1d_y_8x8x8[63][0], c1d_y_8x8x8[63][1]-1, c1d_y_8x8x8[63][2]),
                                      offset8x8x8(c1d_y_8x8x8[64][0], c1d_y_8x8x8[64][1]-1, c1d_y_8x8x8[64][2]),
                                      offset8x8x8(c1d_y_8x8x8[65][0], c1d_y_8x8x8[65][1]-1, c1d_y_8x8x8[65][2]),
                                      offset8x8x8(c1d_y_8x8x8[66][0], c1d_y_8x8x8[66][1]-1, c1d_y_8x8x8[66][2]),
                                      offset8x8x8(c1d_y_8x8x8[67][0], c1d_y_8x8x8[67][1]-1, c1d_y_8x8x8[67][2]),
                                      offset8x8x8(c1d_y_8x8x8[68][0], c1d_y_8x8x8[68][1]-1, c1d_y_8x8x8[68][2]),
                                      offset8x8x8(c1d_y_8x8x8[69][0], c1d_y_8x8x8[69][1]-1, c1d_y_8x8x8[69][2]),
                                      offset8x8x8(c1d_y_8x8x8[70][0], c1d_y_8x8x8[70][1]-1, c1d_y_8x8x8[70][2]),
                                      offset8x8x8(c1d_y_8x8x8[71][0], c1d_y_8x8x8[71][1]-1, c1d_y_8x8x8[71][2]),
                                      offset8x8x8(c1d_y_8x8x8[72][0], c1d_y_8x8x8[72][1]-1, c1d_y_8x8x8[72][2]),
                                      offset8x8x8(c1d_y_8x8x8[73][0], c1d_y_8x8x8[73][1]-1, c1d_y_8x8x8[73][2]),
                                      offset8x8x8(c1d_y_8x8x8[74][0], c1d_y_8x8x8[74][1]-1, c1d_y_8x8x8[74][2]),

                                      offset8x8x8(c1d_z_8x8x8[0][0]-1, c1d_z_8x8x8[0][1], c1d_z_8x8x8[0][2]), // Z
                                      offset8x8x8(c1d_z_8x8x8[1][0]-1, c1d_z_8x8x8[1][1], c1d_z_8x8x8[1][2]),
                                      offset8x8x8(c1d_z_8x8x8[2][0]-1, c1d_z_8x8x8[2][1], c1d_z_8x8x8[2][2]),
                                      offset8x8x8(c1d_z_8x8x8[3][0]-1, c1d_z_8x8x8[3][1], c1d_z_8x8x8[3][2]),
                                      offset8x8x8(c1d_z_8x8x8[4][0]-1, c1d_z_8x8x8[4][1], c1d_z_8x8x8[4][2]),
                                      offset8x8x8(c1d_z_8x8x8[5][0]-1, c1d_z_8x8x8[5][1], c1d_z_8x8x8[5][2]),
                                      offset8x8x8(c1d_z_8x8x8[6][0]-1, c1d_z_8x8x8[6][1], c1d_z_8x8x8[6][2]),
                                      offset8x8x8(c1d_z_8x8x8[7][0]-1, c1d_z_8x8x8[7][1], c1d_z_8x8x8[7][2]),
                                      offset8x8x8(c1d_z_8x8x8[8][0]-1, c1d_z_8x8x8[8][1], c1d_z_8x8x8[8][2]),
                                      offset8x8x8(c1d_z_8x8x8[9][0]-1, c1d_z_8x8x8[9][1], c1d_z_8x8x8[9][2]),
                                      offset8x8x8(c1d_z_8x8x8[10][0]-1, c1d_z_8x8x8[10][1], c1d_z_8x8x8[10][2]),
                                      offset8x8x8(c1d_z_8x8x8[11][0]-1, c1d_z_8x8x8[11][1], c1d_z_8x8x8[11][2]),
                                      offset8x8x8(c1d_z_8x8x8[12][0]-1, c1d_z_8x8x8[12][1], c1d_z_8x8x8[12][2]),
                                      offset8x8x8(c1d_z_8x8x8[13][0]-1, c1d_z_8x8x8[13][1], c1d_z_8x8x8[13][2]),
                                      offset8x8x8(c1d_z_8x8x8[14][0]-1, c1d_z_8x8x8[14][1], c1d_z_8x8x8[14][2]),
                                      offset8x8x8(c1d_z_8x8x8[15][0]-1, c1d_z_8x8x8[15][1], c1d_z_8x8x8[15][2]),
                                      offset8x8x8(c1d_z_8x8x8[16][0]-1, c1d_z_8x8x8[16][1], c1d_z_8x8x8[16][2]),
                                      offset8x8x8(c1d_z_8x8x8[17][0]-1, c1d_z_8x8x8[17][1], c1d_z_8x8x8[17][2]),
                                      offset8x8x8(c1d_z_8x8x8[18][0]-1, c1d_z_8x8x8[18][1], c1d_z_8x8x8[18][2]),
                                      offset8x8x8(c1d_z_8x8x8[19][0]-1, c1d_z_8x8x8[19][1], c1d_z_8x8x8[19][2]),
                                      offset8x8x8(c1d_z_8x8x8[20][0]-1, c1d_z_8x8x8[20][1], c1d_z_8x8x8[20][2]),
                                      offset8x8x8(c1d_z_8x8x8[21][0]-1, c1d_z_8x8x8[21][1], c1d_z_8x8x8[21][2]),
                                      offset8x8x8(c1d_z_8x8x8[22][0]-1, c1d_z_8x8x8[22][1], c1d_z_8x8x8[22][2]),
                                      offset8x8x8(c1d_z_8x8x8[23][0]-1, c1d_z_8x8x8[23][1], c1d_z_8x8x8[23][2]),
                                      offset8x8x8(c1d_z_8x8x8[24][0]-1, c1d_z_8x8x8[24][1], c1d_z_8x8x8[24][2]),
                                      offset8x8x8(c1d_z_8x8x8[25][0]-1, c1d_z_8x8x8[25][1], c1d_z_8x8x8[25][2]),
                                      offset8x8x8(c1d_z_8x8x8[26][0]-1, c1d_z_8x8x8[26][1], c1d_z_8x8x8[26][2]),
                                      offset8x8x8(c1d_z_8x8x8[27][0]-1, c1d_z_8x8x8[27][1], c1d_z_8x8x8[27][2]),
                                      offset8x8x8(c1d_z_8x8x8[28][0]-1, c1d_z_8x8x8[28][1], c1d_z_8x8x8[28][2]),
                                      offset8x8x8(c1d_z_8x8x8[29][0]-1, c1d_z_8x8x8[29][1], c1d_z_8x8x8[29][2]),
                                      offset8x8x8(c1d_z_8x8x8[30][0]-1, c1d_z_8x8x8[30][1], c1d_z_8x8x8[30][2]),
                                      offset8x8x8(c1d_z_8x8x8[31][0]-1, c1d_z_8x8x8[31][1], c1d_z_8x8x8[31][2]),
                                      offset8x8x8(c1d_z_8x8x8[32][0]-1, c1d_z_8x8x8[32][1], c1d_z_8x8x8[32][2]),
                                      offset8x8x8(c1d_z_8x8x8[33][0]-1, c1d_z_8x8x8[33][1], c1d_z_8x8x8[33][2]),
                                      offset8x8x8(c1d_z_8x8x8[34][0]-1, c1d_z_8x8x8[34][1], c1d_z_8x8x8[34][2]),
                                      offset8x8x8(c1d_z_8x8x8[35][0]-1, c1d_z_8x8x8[35][1], c1d_z_8x8x8[35][2]),
                                      offset8x8x8(c1d_z_8x8x8[36][0]-1, c1d_z_8x8x8[36][1], c1d_z_8x8x8[36][2]),
                                      offset8x8x8(c1d_z_8x8x8[37][0]-1, c1d_z_8x8x8[37][1], c1d_z_8x8x8[37][2]),
                                      offset8x8x8(c1d_z_8x8x8[38][0]-1, c1d_z_8x8x8[38][1], c1d_z_8x8x8[38][2]),
                                      offset8x8x8(c1d_z_8x8x8[39][0]-1, c1d_z_8x8x8[39][1], c1d_z_8x8x8[39][2]),
                                      offset8x8x8(c1d_z_8x8x8[40][0]-1, c1d_z_8x8x8[40][1], c1d_z_8x8x8[40][2]),
                                      offset8x8x8(c1d_z_8x8x8[41][0]-1, c1d_z_8x8x8[41][1], c1d_z_8x8x8[41][2]),
                                      offset8x8x8(c1d_z_8x8x8[42][0]-1, c1d_z_8x8x8[42][1], c1d_z_8x8x8[42][2]),
                                      offset8x8x8(c1d_z_8x8x8[43][0]-1, c1d_z_8x8x8[43][1], c1d_z_8x8x8[43][2]),
                                      offset8x8x8(c1d_z_8x8x8[44][0]-1, c1d_z_8x8x8[44][1], c1d_z_8x8x8[44][2]),
                                      offset8x8x8(c1d_z_8x8x8[45][0]-1, c1d_z_8x8x8[45][1], c1d_z_8x8x8[45][2]),
                                      offset8x8x8(c1d_z_8x8x8[46][0]-1, c1d_z_8x8x8[46][1], c1d_z_8x8x8[46][2]),
                                      offset8x8x8(c1d_z_8x8x8[47][0]-1, c1d_z_8x8x8[47][1], c1d_z_8x8x8[47][2]),
                                      offset8x8x8(c1d_z_8x8x8[48][0]-1, c1d_z_8x8x8[48][1], c1d_z_8x8x8[48][2]),
                                      offset8x8x8(c1d_z_8x8x8[49][0]-1, c1d_z_8x8x8[49][1], c1d_z_8x8x8[49][2]),
                                      offset8x8x8(c1d_z_8x8x8[50][0]-1, c1d_z_8x8x8[50][1], c1d_z_8x8x8[50][2]),
                                      offset8x8x8(c1d_z_8x8x8[51][0]-1, c1d_z_8x8x8[51][1], c1d_z_8x8x8[51][2]),
                                      offset8x8x8(c1d_z_8x8x8[52][0]-1, c1d_z_8x8x8[52][1], c1d_z_8x8x8[52][2]),
                                      offset8x8x8(c1d_z_8x8x8[53][0]-1, c1d_z_8x8x8[53][1], c1d_z_8x8x8[53][2]),
                                      offset8x8x8(c1d_z_8x8x8[54][0]-1, c1d_z_8x8x8[54][1], c1d_z_8x8x8[54][2]),
                                      offset8x8x8(c1d_z_8x8x8[55][0]-1, c1d_z_8x8x8[55][1], c1d_z_8x8x8[55][2]),
                                      offset8x8x8(c1d_z_8x8x8[56][0]-1, c1d_z_8x8x8[56][1], c1d_z_8x8x8[56][2]),
                                      offset8x8x8(c1d_z_8x8x8[57][0]-1, c1d_z_8x8x8[57][1], c1d_z_8x8x8[57][2]),
                                      offset8x8x8(c1d_z_8x8x8[58][0]-1, c1d_z_8x8x8[58][1], c1d_z_8x8x8[58][2]),
                                      offset8x8x8(c1d_z_8x8x8[59][0]-1, c1d_z_8x8x8[59][1], c1d_z_8x8x8[59][2]),
                                      offset8x8x8(c1d_z_8x8x8[60][0]-1, c1d_z_8x8x8[60][1], c1d_z_8x8x8[60][2]),
                                      offset8x8x8(c1d_z_8x8x8[61][0]-1, c1d_z_8x8x8[61][1], c1d_z_8x8x8[61][2]),
                                      offset8x8x8(c1d_z_8x8x8[62][0]-1, c1d_z_8x8x8[62][1], c1d_z_8x8x8[62][2]),
                                      offset8x8x8(c1d_z_8x8x8[63][0]-1, c1d_z_8x8x8[63][1], c1d_z_8x8x8[63][2]),
                                      offset8x8x8(c1d_z_8x8x8[64][0]-1, c1d_z_8x8x8[64][1], c1d_z_8x8x8[64][2]),
                                      offset8x8x8(c1d_z_8x8x8[65][0]-1, c1d_z_8x8x8[65][1], c1d_z_8x8x8[65][2]),
                                      offset8x8x8(c1d_z_8x8x8[66][0]-1, c1d_z_8x8x8[66][1], c1d_z_8x8x8[66][2]),
                                      offset8x8x8(c1d_z_8x8x8[67][0]-1, c1d_z_8x8x8[67][1], c1d_z_8x8x8[67][2]),
                                      offset8x8x8(c1d_z_8x8x8[68][0]-1, c1d_z_8x8x8[68][1], c1d_z_8x8x8[68][2]),
                                      offset8x8x8(c1d_z_8x8x8[69][0]-1, c1d_z_8x8x8[69][1], c1d_z_8x8x8[69][2]),
                                      offset8x8x8(c1d_z_8x8x8[70][0]-1, c1d_z_8x8x8[70][1], c1d_z_8x8x8[70][2]),
                                      offset8x8x8(c1d_z_8x8x8[71][0]-1, c1d_z_8x8x8[71][1], c1d_z_8x8x8[71][2]),
                                      offset8x8x8(c1d_z_8x8x8[72][0]-1, c1d_z_8x8x8[72][1], c1d_z_8x8x8[72][2]),
                                      offset8x8x8(c1d_z_8x8x8[73][0]-1, c1d_z_8x8x8[73][1], c1d_z_8x8x8[73][2]),
                                      offset8x8x8(c1d_z_8x8x8[74][0]-1, c1d_z_8x8x8[74][1], c1d_z_8x8x8[74][2])};

  return offset[i];
}

MGARDX_EXEC int Coeff1D_R_Offset_8x8x8(SIZE i) {
  static constexpr int offset[225] = {offset8x8x8(c1d_x_8x8x8[0][0], c1d_x_8x8x8[0][1], c1d_x_8x8x8[0][2]+1), // X
                                      offset8x8x8(c1d_x_8x8x8[1][0], c1d_x_8x8x8[1][1], c1d_x_8x8x8[1][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[2][0], c1d_x_8x8x8[2][1], c1d_x_8x8x8[2][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[3][0], c1d_x_8x8x8[3][1], c1d_x_8x8x8[3][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[4][0], c1d_x_8x8x8[4][1], c1d_x_8x8x8[4][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[5][0], c1d_x_8x8x8[5][1], c1d_x_8x8x8[5][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[6][0], c1d_x_8x8x8[6][1], c1d_x_8x8x8[6][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[7][0], c1d_x_8x8x8[7][1], c1d_x_8x8x8[7][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[8][0], c1d_x_8x8x8[8][1], c1d_x_8x8x8[8][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[9][0], c1d_x_8x8x8[9][1], c1d_x_8x8x8[9][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[10][0], c1d_x_8x8x8[10][1], c1d_x_8x8x8[10][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[11][0], c1d_x_8x8x8[11][1], c1d_x_8x8x8[11][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[12][0], c1d_x_8x8x8[12][1], c1d_x_8x8x8[12][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[13][0], c1d_x_8x8x8[13][1], c1d_x_8x8x8[13][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[14][0], c1d_x_8x8x8[14][1], c1d_x_8x8x8[14][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[15][0], c1d_x_8x8x8[15][1], c1d_x_8x8x8[15][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[16][0], c1d_x_8x8x8[16][1], c1d_x_8x8x8[16][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[17][0], c1d_x_8x8x8[17][1], c1d_x_8x8x8[17][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[18][0], c1d_x_8x8x8[18][1], c1d_x_8x8x8[18][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[19][0], c1d_x_8x8x8[19][1], c1d_x_8x8x8[19][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[20][0], c1d_x_8x8x8[20][1], c1d_x_8x8x8[20][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[21][0], c1d_x_8x8x8[21][1], c1d_x_8x8x8[21][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[22][0], c1d_x_8x8x8[22][1], c1d_x_8x8x8[22][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[23][0], c1d_x_8x8x8[23][1], c1d_x_8x8x8[23][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[24][0], c1d_x_8x8x8[24][1], c1d_x_8x8x8[24][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[25][0], c1d_x_8x8x8[25][1], c1d_x_8x8x8[25][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[26][0], c1d_x_8x8x8[26][1], c1d_x_8x8x8[26][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[27][0], c1d_x_8x8x8[27][1], c1d_x_8x8x8[27][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[28][0], c1d_x_8x8x8[28][1], c1d_x_8x8x8[28][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[29][0], c1d_x_8x8x8[29][1], c1d_x_8x8x8[29][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[30][0], c1d_x_8x8x8[30][1], c1d_x_8x8x8[30][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[31][0], c1d_x_8x8x8[31][1], c1d_x_8x8x8[31][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[32][0], c1d_x_8x8x8[32][1], c1d_x_8x8x8[32][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[33][0], c1d_x_8x8x8[33][1], c1d_x_8x8x8[33][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[34][0], c1d_x_8x8x8[34][1], c1d_x_8x8x8[34][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[35][0], c1d_x_8x8x8[35][1], c1d_x_8x8x8[35][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[36][0], c1d_x_8x8x8[36][1], c1d_x_8x8x8[36][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[37][0], c1d_x_8x8x8[37][1], c1d_x_8x8x8[37][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[38][0], c1d_x_8x8x8[38][1], c1d_x_8x8x8[38][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[39][0], c1d_x_8x8x8[39][1], c1d_x_8x8x8[39][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[40][0], c1d_x_8x8x8[40][1], c1d_x_8x8x8[40][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[41][0], c1d_x_8x8x8[41][1], c1d_x_8x8x8[41][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[42][0], c1d_x_8x8x8[42][1], c1d_x_8x8x8[42][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[43][0], c1d_x_8x8x8[43][1], c1d_x_8x8x8[43][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[44][0], c1d_x_8x8x8[44][1], c1d_x_8x8x8[44][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[45][0], c1d_x_8x8x8[45][1], c1d_x_8x8x8[45][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[46][0], c1d_x_8x8x8[46][1], c1d_x_8x8x8[46][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[47][0], c1d_x_8x8x8[47][1], c1d_x_8x8x8[47][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[48][0], c1d_x_8x8x8[48][1], c1d_x_8x8x8[48][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[49][0], c1d_x_8x8x8[49][1], c1d_x_8x8x8[49][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[50][0], c1d_x_8x8x8[50][1], c1d_x_8x8x8[50][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[51][0], c1d_x_8x8x8[51][1], c1d_x_8x8x8[51][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[52][0], c1d_x_8x8x8[52][1], c1d_x_8x8x8[52][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[53][0], c1d_x_8x8x8[53][1], c1d_x_8x8x8[53][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[54][0], c1d_x_8x8x8[54][1], c1d_x_8x8x8[54][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[55][0], c1d_x_8x8x8[55][1], c1d_x_8x8x8[55][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[56][0], c1d_x_8x8x8[56][1], c1d_x_8x8x8[56][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[57][0], c1d_x_8x8x8[57][1], c1d_x_8x8x8[57][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[58][0], c1d_x_8x8x8[58][1], c1d_x_8x8x8[58][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[59][0], c1d_x_8x8x8[59][1], c1d_x_8x8x8[59][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[60][0], c1d_x_8x8x8[60][1], c1d_x_8x8x8[60][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[61][0], c1d_x_8x8x8[61][1], c1d_x_8x8x8[61][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[62][0], c1d_x_8x8x8[62][1], c1d_x_8x8x8[62][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[63][0], c1d_x_8x8x8[63][1], c1d_x_8x8x8[63][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[64][0], c1d_x_8x8x8[64][1], c1d_x_8x8x8[64][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[65][0], c1d_x_8x8x8[65][1], c1d_x_8x8x8[65][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[66][0], c1d_x_8x8x8[66][1], c1d_x_8x8x8[66][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[67][0], c1d_x_8x8x8[67][1], c1d_x_8x8x8[67][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[68][0], c1d_x_8x8x8[68][1], c1d_x_8x8x8[68][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[69][0], c1d_x_8x8x8[69][1], c1d_x_8x8x8[69][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[70][0], c1d_x_8x8x8[70][1], c1d_x_8x8x8[70][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[71][0], c1d_x_8x8x8[71][1], c1d_x_8x8x8[71][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[72][0], c1d_x_8x8x8[72][1], c1d_x_8x8x8[72][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[73][0], c1d_x_8x8x8[73][1], c1d_x_8x8x8[73][2]+1),
                                      offset8x8x8(c1d_x_8x8x8[74][0], c1d_x_8x8x8[74][1], c1d_x_8x8x8[74][2]+1),

                                      offset8x8x8(c1d_y_8x8x8[0][0], c1d_y_8x8x8[0][1]+1, c1d_y_8x8x8[0][2]), // Y
                                      offset8x8x8(c1d_y_8x8x8[1][0], c1d_y_8x8x8[1][1]+1, c1d_y_8x8x8[1][2]),
                                      offset8x8x8(c1d_y_8x8x8[2][0], c1d_y_8x8x8[2][1]+1, c1d_y_8x8x8[2][2]),
                                      offset8x8x8(c1d_y_8x8x8[3][0], c1d_y_8x8x8[3][1]+1, c1d_y_8x8x8[3][2]),
                                      offset8x8x8(c1d_y_8x8x8[4][0], c1d_y_8x8x8[4][1]+1, c1d_y_8x8x8[4][2]),
                                      offset8x8x8(c1d_y_8x8x8[5][0], c1d_y_8x8x8[5][1]+1, c1d_y_8x8x8[5][2]),
                                      offset8x8x8(c1d_y_8x8x8[6][0], c1d_y_8x8x8[6][1]+1, c1d_y_8x8x8[6][2]),
                                      offset8x8x8(c1d_y_8x8x8[7][0], c1d_y_8x8x8[7][1]+1, c1d_y_8x8x8[7][2]),
                                      offset8x8x8(c1d_y_8x8x8[8][0], c1d_y_8x8x8[8][1]+1, c1d_y_8x8x8[8][2]),
                                      offset8x8x8(c1d_y_8x8x8[9][0], c1d_y_8x8x8[9][1]+1, c1d_y_8x8x8[9][2]),
                                      offset8x8x8(c1d_y_8x8x8[10][0], c1d_y_8x8x8[10][1]+1, c1d_y_8x8x8[10][2]),
                                      offset8x8x8(c1d_y_8x8x8[11][0], c1d_y_8x8x8[11][1]+1, c1d_y_8x8x8[11][2]),
                                      offset8x8x8(c1d_y_8x8x8[12][0], c1d_y_8x8x8[12][1]+1, c1d_y_8x8x8[12][2]),
                                      offset8x8x8(c1d_y_8x8x8[13][0], c1d_y_8x8x8[13][1]+1, c1d_y_8x8x8[13][2]),
                                      offset8x8x8(c1d_y_8x8x8[14][0], c1d_y_8x8x8[14][1]+1, c1d_y_8x8x8[14][2]),
                                      offset8x8x8(c1d_y_8x8x8[15][0], c1d_y_8x8x8[15][1]+1, c1d_y_8x8x8[15][2]),
                                      offset8x8x8(c1d_y_8x8x8[16][0], c1d_y_8x8x8[16][1]+1, c1d_y_8x8x8[16][2]),
                                      offset8x8x8(c1d_y_8x8x8[17][0], c1d_y_8x8x8[17][1]+1, c1d_y_8x8x8[17][2]),
                                      offset8x8x8(c1d_y_8x8x8[18][0], c1d_y_8x8x8[18][1]+1, c1d_y_8x8x8[18][2]),
                                      offset8x8x8(c1d_y_8x8x8[19][0], c1d_y_8x8x8[19][1]+1, c1d_y_8x8x8[19][2]),
                                      offset8x8x8(c1d_y_8x8x8[20][0], c1d_y_8x8x8[20][1]+1, c1d_y_8x8x8[20][2]),
                                      offset8x8x8(c1d_y_8x8x8[21][0], c1d_y_8x8x8[21][1]+1, c1d_y_8x8x8[21][2]),
                                      offset8x8x8(c1d_y_8x8x8[22][0], c1d_y_8x8x8[22][1]+1, c1d_y_8x8x8[22][2]),
                                      offset8x8x8(c1d_y_8x8x8[23][0], c1d_y_8x8x8[23][1]+1, c1d_y_8x8x8[23][2]),
                                      offset8x8x8(c1d_y_8x8x8[24][0], c1d_y_8x8x8[24][1]+1, c1d_y_8x8x8[24][2]),
                                      offset8x8x8(c1d_y_8x8x8[25][0], c1d_y_8x8x8[25][1]+1, c1d_y_8x8x8[25][2]),
                                      offset8x8x8(c1d_y_8x8x8[26][0], c1d_y_8x8x8[26][1]+1, c1d_y_8x8x8[26][2]),
                                      offset8x8x8(c1d_y_8x8x8[27][0], c1d_y_8x8x8[27][1]+1, c1d_y_8x8x8[27][2]),
                                      offset8x8x8(c1d_y_8x8x8[28][0], c1d_y_8x8x8[28][1]+1, c1d_y_8x8x8[28][2]),
                                      offset8x8x8(c1d_y_8x8x8[29][0], c1d_y_8x8x8[29][1]+1, c1d_y_8x8x8[29][2]),
                                      offset8x8x8(c1d_y_8x8x8[30][0], c1d_y_8x8x8[30][1]+1, c1d_y_8x8x8[30][2]),
                                      offset8x8x8(c1d_y_8x8x8[31][0], c1d_y_8x8x8[31][1]+1, c1d_y_8x8x8[31][2]),
                                      offset8x8x8(c1d_y_8x8x8[32][0], c1d_y_8x8x8[32][1]+1, c1d_y_8x8x8[32][2]),
                                      offset8x8x8(c1d_y_8x8x8[33][0], c1d_y_8x8x8[33][1]+1, c1d_y_8x8x8[33][2]),
                                      offset8x8x8(c1d_y_8x8x8[34][0], c1d_y_8x8x8[34][1]+1, c1d_y_8x8x8[34][2]),
                                      offset8x8x8(c1d_y_8x8x8[35][0], c1d_y_8x8x8[35][1]+1, c1d_y_8x8x8[35][2]),
                                      offset8x8x8(c1d_y_8x8x8[36][0], c1d_y_8x8x8[36][1]+1, c1d_y_8x8x8[36][2]),
                                      offset8x8x8(c1d_y_8x8x8[37][0], c1d_y_8x8x8[37][1]+1, c1d_y_8x8x8[37][2]),
                                      offset8x8x8(c1d_y_8x8x8[38][0], c1d_y_8x8x8[38][1]+1, c1d_y_8x8x8[38][2]),
                                      offset8x8x8(c1d_y_8x8x8[39][0], c1d_y_8x8x8[39][1]+1, c1d_y_8x8x8[39][2]),
                                      offset8x8x8(c1d_y_8x8x8[40][0], c1d_y_8x8x8[40][1]+1, c1d_y_8x8x8[40][2]),
                                      offset8x8x8(c1d_y_8x8x8[41][0], c1d_y_8x8x8[41][1]+1, c1d_y_8x8x8[41][2]),
                                      offset8x8x8(c1d_y_8x8x8[42][0], c1d_y_8x8x8[42][1]+1, c1d_y_8x8x8[42][2]),
                                      offset8x8x8(c1d_y_8x8x8[43][0], c1d_y_8x8x8[43][1]+1, c1d_y_8x8x8[43][2]),
                                      offset8x8x8(c1d_y_8x8x8[44][0], c1d_y_8x8x8[44][1]+1, c1d_y_8x8x8[44][2]),
                                      offset8x8x8(c1d_y_8x8x8[45][0], c1d_y_8x8x8[45][1]+1, c1d_y_8x8x8[45][2]),
                                      offset8x8x8(c1d_y_8x8x8[46][0], c1d_y_8x8x8[46][1]+1, c1d_y_8x8x8[46][2]),
                                      offset8x8x8(c1d_y_8x8x8[47][0], c1d_y_8x8x8[47][1]+1, c1d_y_8x8x8[47][2]),
                                      offset8x8x8(c1d_y_8x8x8[48][0], c1d_y_8x8x8[48][1]+1, c1d_y_8x8x8[48][2]),
                                      offset8x8x8(c1d_y_8x8x8[49][0], c1d_y_8x8x8[49][1]+1, c1d_y_8x8x8[49][2]),
                                      offset8x8x8(c1d_y_8x8x8[50][0], c1d_y_8x8x8[50][1]+1, c1d_y_8x8x8[50][2]),
                                      offset8x8x8(c1d_y_8x8x8[51][0], c1d_y_8x8x8[51][1]+1, c1d_y_8x8x8[51][2]),
                                      offset8x8x8(c1d_y_8x8x8[52][0], c1d_y_8x8x8[52][1]+1, c1d_y_8x8x8[52][2]),
                                      offset8x8x8(c1d_y_8x8x8[53][0], c1d_y_8x8x8[53][1]+1, c1d_y_8x8x8[53][2]),
                                      offset8x8x8(c1d_y_8x8x8[54][0], c1d_y_8x8x8[54][1]+1, c1d_y_8x8x8[54][2]),
                                      offset8x8x8(c1d_y_8x8x8[55][0], c1d_y_8x8x8[55][1]+1, c1d_y_8x8x8[55][2]),
                                      offset8x8x8(c1d_y_8x8x8[56][0], c1d_y_8x8x8[56][1]+1, c1d_y_8x8x8[56][2]),
                                      offset8x8x8(c1d_y_8x8x8[57][0], c1d_y_8x8x8[57][1]+1, c1d_y_8x8x8[57][2]),
                                      offset8x8x8(c1d_y_8x8x8[58][0], c1d_y_8x8x8[58][1]+1, c1d_y_8x8x8[58][2]),
                                      offset8x8x8(c1d_y_8x8x8[59][0], c1d_y_8x8x8[59][1]+1, c1d_y_8x8x8[59][2]),
                                      offset8x8x8(c1d_y_8x8x8[60][0], c1d_y_8x8x8[60][1]+1, c1d_y_8x8x8[60][2]),
                                      offset8x8x8(c1d_y_8x8x8[61][0], c1d_y_8x8x8[61][1]+1, c1d_y_8x8x8[61][2]),
                                      offset8x8x8(c1d_y_8x8x8[62][0], c1d_y_8x8x8[62][1]+1, c1d_y_8x8x8[62][2]),
                                      offset8x8x8(c1d_y_8x8x8[63][0], c1d_y_8x8x8[63][1]+1, c1d_y_8x8x8[63][2]),
                                      offset8x8x8(c1d_y_8x8x8[64][0], c1d_y_8x8x8[64][1]+1, c1d_y_8x8x8[64][2]),
                                      offset8x8x8(c1d_y_8x8x8[65][0], c1d_y_8x8x8[65][1]+1, c1d_y_8x8x8[65][2]),
                                      offset8x8x8(c1d_y_8x8x8[66][0], c1d_y_8x8x8[66][1]+1, c1d_y_8x8x8[66][2]),
                                      offset8x8x8(c1d_y_8x8x8[67][0], c1d_y_8x8x8[67][1]+1, c1d_y_8x8x8[67][2]),
                                      offset8x8x8(c1d_y_8x8x8[68][0], c1d_y_8x8x8[68][1]+1, c1d_y_8x8x8[68][2]),
                                      offset8x8x8(c1d_y_8x8x8[69][0], c1d_y_8x8x8[69][1]+1, c1d_y_8x8x8[69][2]),
                                      offset8x8x8(c1d_y_8x8x8[70][0], c1d_y_8x8x8[70][1]+1, c1d_y_8x8x8[70][2]),
                                      offset8x8x8(c1d_y_8x8x8[71][0], c1d_y_8x8x8[71][1]+1, c1d_y_8x8x8[71][2]),
                                      offset8x8x8(c1d_y_8x8x8[72][0], c1d_y_8x8x8[72][1]+1, c1d_y_8x8x8[72][2]),
                                      offset8x8x8(c1d_y_8x8x8[73][0], c1d_y_8x8x8[73][1]+1, c1d_y_8x8x8[73][2]),
                                      offset8x8x8(c1d_y_8x8x8[74][0], c1d_y_8x8x8[74][1]+1, c1d_y_8x8x8[74][2]),

                                      offset8x8x8(c1d_z_8x8x8[0][0]+1, c1d_z_8x8x8[0][1], c1d_z_8x8x8[0][2]), // Z
                                      offset8x8x8(c1d_z_8x8x8[1][0]+1, c1d_z_8x8x8[1][1], c1d_z_8x8x8[1][2]),
                                      offset8x8x8(c1d_z_8x8x8[2][0]+1, c1d_z_8x8x8[2][1], c1d_z_8x8x8[2][2]),
                                      offset8x8x8(c1d_z_8x8x8[3][0]+1, c1d_z_8x8x8[3][1], c1d_z_8x8x8[3][2]),
                                      offset8x8x8(c1d_z_8x8x8[4][0]+1, c1d_z_8x8x8[4][1], c1d_z_8x8x8[4][2]),
                                      offset8x8x8(c1d_z_8x8x8[5][0]+1, c1d_z_8x8x8[5][1], c1d_z_8x8x8[5][2]),
                                      offset8x8x8(c1d_z_8x8x8[6][0]+1, c1d_z_8x8x8[6][1], c1d_z_8x8x8[6][2]),
                                      offset8x8x8(c1d_z_8x8x8[7][0]+1, c1d_z_8x8x8[7][1], c1d_z_8x8x8[7][2]),
                                      offset8x8x8(c1d_z_8x8x8[8][0]+1, c1d_z_8x8x8[8][1], c1d_z_8x8x8[8][2]),
                                      offset8x8x8(c1d_z_8x8x8[9][0]+1, c1d_z_8x8x8[9][1], c1d_z_8x8x8[9][2]),
                                      offset8x8x8(c1d_z_8x8x8[10][0]+1, c1d_z_8x8x8[10][1], c1d_z_8x8x8[10][2]),
                                      offset8x8x8(c1d_z_8x8x8[11][0]+1, c1d_z_8x8x8[11][1], c1d_z_8x8x8[11][2]),
                                      offset8x8x8(c1d_z_8x8x8[12][0]+1, c1d_z_8x8x8[12][1], c1d_z_8x8x8[12][2]),
                                      offset8x8x8(c1d_z_8x8x8[13][0]+1, c1d_z_8x8x8[13][1], c1d_z_8x8x8[13][2]),
                                      offset8x8x8(c1d_z_8x8x8[14][0]+1, c1d_z_8x8x8[14][1], c1d_z_8x8x8[14][2]),
                                      offset8x8x8(c1d_z_8x8x8[15][0]+1, c1d_z_8x8x8[15][1], c1d_z_8x8x8[15][2]),
                                      offset8x8x8(c1d_z_8x8x8[16][0]+1, c1d_z_8x8x8[16][1], c1d_z_8x8x8[16][2]),
                                      offset8x8x8(c1d_z_8x8x8[17][0]+1, c1d_z_8x8x8[17][1], c1d_z_8x8x8[17][2]),
                                      offset8x8x8(c1d_z_8x8x8[18][0]+1, c1d_z_8x8x8[18][1], c1d_z_8x8x8[18][2]),
                                      offset8x8x8(c1d_z_8x8x8[19][0]+1, c1d_z_8x8x8[19][1], c1d_z_8x8x8[19][2]),
                                      offset8x8x8(c1d_z_8x8x8[20][0]+1, c1d_z_8x8x8[20][1], c1d_z_8x8x8[20][2]),
                                      offset8x8x8(c1d_z_8x8x8[21][0]+1, c1d_z_8x8x8[21][1], c1d_z_8x8x8[21][2]),
                                      offset8x8x8(c1d_z_8x8x8[22][0]+1, c1d_z_8x8x8[22][1], c1d_z_8x8x8[22][2]),
                                      offset8x8x8(c1d_z_8x8x8[23][0]+1, c1d_z_8x8x8[23][1], c1d_z_8x8x8[23][2]),
                                      offset8x8x8(c1d_z_8x8x8[24][0]+1, c1d_z_8x8x8[24][1], c1d_z_8x8x8[24][2]),
                                      offset8x8x8(c1d_z_8x8x8[25][0]+1, c1d_z_8x8x8[25][1], c1d_z_8x8x8[25][2]),
                                      offset8x8x8(c1d_z_8x8x8[26][0]+1, c1d_z_8x8x8[26][1], c1d_z_8x8x8[26][2]),
                                      offset8x8x8(c1d_z_8x8x8[27][0]+1, c1d_z_8x8x8[27][1], c1d_z_8x8x8[27][2]),
                                      offset8x8x8(c1d_z_8x8x8[28][0]+1, c1d_z_8x8x8[28][1], c1d_z_8x8x8[28][2]),
                                      offset8x8x8(c1d_z_8x8x8[29][0]+1, c1d_z_8x8x8[29][1], c1d_z_8x8x8[29][2]),
                                      offset8x8x8(c1d_z_8x8x8[30][0]+1, c1d_z_8x8x8[30][1], c1d_z_8x8x8[30][2]),
                                      offset8x8x8(c1d_z_8x8x8[31][0]+1, c1d_z_8x8x8[31][1], c1d_z_8x8x8[31][2]),
                                      offset8x8x8(c1d_z_8x8x8[32][0]+1, c1d_z_8x8x8[32][1], c1d_z_8x8x8[32][2]),
                                      offset8x8x8(c1d_z_8x8x8[33][0]+1, c1d_z_8x8x8[33][1], c1d_z_8x8x8[33][2]),
                                      offset8x8x8(c1d_z_8x8x8[34][0]+1, c1d_z_8x8x8[34][1], c1d_z_8x8x8[34][2]),
                                      offset8x8x8(c1d_z_8x8x8[35][0]+1, c1d_z_8x8x8[35][1], c1d_z_8x8x8[35][2]),
                                      offset8x8x8(c1d_z_8x8x8[36][0]+1, c1d_z_8x8x8[36][1], c1d_z_8x8x8[36][2]),
                                      offset8x8x8(c1d_z_8x8x8[37][0]+1, c1d_z_8x8x8[37][1], c1d_z_8x8x8[37][2]),
                                      offset8x8x8(c1d_z_8x8x8[38][0]+1, c1d_z_8x8x8[38][1], c1d_z_8x8x8[38][2]),
                                      offset8x8x8(c1d_z_8x8x8[39][0]+1, c1d_z_8x8x8[39][1], c1d_z_8x8x8[39][2]),
                                      offset8x8x8(c1d_z_8x8x8[40][0]+1, c1d_z_8x8x8[40][1], c1d_z_8x8x8[40][2]),
                                      offset8x8x8(c1d_z_8x8x8[41][0]+1, c1d_z_8x8x8[41][1], c1d_z_8x8x8[41][2]),
                                      offset8x8x8(c1d_z_8x8x8[42][0]+1, c1d_z_8x8x8[42][1], c1d_z_8x8x8[42][2]),
                                      offset8x8x8(c1d_z_8x8x8[43][0]+1, c1d_z_8x8x8[43][1], c1d_z_8x8x8[43][2]),
                                      offset8x8x8(c1d_z_8x8x8[44][0]+1, c1d_z_8x8x8[44][1], c1d_z_8x8x8[44][2]),
                                      offset8x8x8(c1d_z_8x8x8[45][0]+1, c1d_z_8x8x8[45][1], c1d_z_8x8x8[45][2]),
                                      offset8x8x8(c1d_z_8x8x8[46][0]+1, c1d_z_8x8x8[46][1], c1d_z_8x8x8[46][2]),
                                      offset8x8x8(c1d_z_8x8x8[47][0]+1, c1d_z_8x8x8[47][1], c1d_z_8x8x8[47][2]),
                                      offset8x8x8(c1d_z_8x8x8[48][0]+1, c1d_z_8x8x8[48][1], c1d_z_8x8x8[48][2]),
                                      offset8x8x8(c1d_z_8x8x8[49][0]+1, c1d_z_8x8x8[49][1], c1d_z_8x8x8[49][2]),
                                      offset8x8x8(c1d_z_8x8x8[50][0]+1, c1d_z_8x8x8[50][1], c1d_z_8x8x8[50][2]),
                                      offset8x8x8(c1d_z_8x8x8[51][0]+1, c1d_z_8x8x8[51][1], c1d_z_8x8x8[51][2]),
                                      offset8x8x8(c1d_z_8x8x8[52][0]+1, c1d_z_8x8x8[52][1], c1d_z_8x8x8[52][2]),
                                      offset8x8x8(c1d_z_8x8x8[53][0]+1, c1d_z_8x8x8[53][1], c1d_z_8x8x8[53][2]),
                                      offset8x8x8(c1d_z_8x8x8[54][0]+1, c1d_z_8x8x8[54][1], c1d_z_8x8x8[54][2]),
                                      offset8x8x8(c1d_z_8x8x8[55][0]+1, c1d_z_8x8x8[55][1], c1d_z_8x8x8[55][2]),
                                      offset8x8x8(c1d_z_8x8x8[56][0]+1, c1d_z_8x8x8[56][1], c1d_z_8x8x8[56][2]),
                                      offset8x8x8(c1d_z_8x8x8[57][0]+1, c1d_z_8x8x8[57][1], c1d_z_8x8x8[57][2]),
                                      offset8x8x8(c1d_z_8x8x8[58][0]+1, c1d_z_8x8x8[58][1], c1d_z_8x8x8[58][2]),
                                      offset8x8x8(c1d_z_8x8x8[59][0]+1, c1d_z_8x8x8[59][1], c1d_z_8x8x8[59][2]),
                                      offset8x8x8(c1d_z_8x8x8[60][0]+1, c1d_z_8x8x8[60][1], c1d_z_8x8x8[60][2]),
                                      offset8x8x8(c1d_z_8x8x8[61][0]+1, c1d_z_8x8x8[61][1], c1d_z_8x8x8[61][2]),
                                      offset8x8x8(c1d_z_8x8x8[62][0]+1, c1d_z_8x8x8[62][1], c1d_z_8x8x8[62][2]),
                                      offset8x8x8(c1d_z_8x8x8[63][0]+1, c1d_z_8x8x8[63][1], c1d_z_8x8x8[63][2]),
                                      offset8x8x8(c1d_z_8x8x8[64][0]+1, c1d_z_8x8x8[64][1], c1d_z_8x8x8[64][2]),
                                      offset8x8x8(c1d_z_8x8x8[65][0]+1, c1d_z_8x8x8[65][1], c1d_z_8x8x8[65][2]),
                                      offset8x8x8(c1d_z_8x8x8[66][0]+1, c1d_z_8x8x8[66][1], c1d_z_8x8x8[66][2]),
                                      offset8x8x8(c1d_z_8x8x8[67][0]+1, c1d_z_8x8x8[67][1], c1d_z_8x8x8[67][2]),
                                      offset8x8x8(c1d_z_8x8x8[68][0]+1, c1d_z_8x8x8[68][1], c1d_z_8x8x8[68][2]),
                                      offset8x8x8(c1d_z_8x8x8[69][0]+1, c1d_z_8x8x8[69][1], c1d_z_8x8x8[69][2]),
                                      offset8x8x8(c1d_z_8x8x8[70][0]+1, c1d_z_8x8x8[70][1], c1d_z_8x8x8[70][2]),
                                      offset8x8x8(c1d_z_8x8x8[71][0]+1, c1d_z_8x8x8[71][1], c1d_z_8x8x8[71][2]),
                                      offset8x8x8(c1d_z_8x8x8[72][0]+1, c1d_z_8x8x8[72][1], c1d_z_8x8x8[72][2]),
                                      offset8x8x8(c1d_z_8x8x8[73][0]+1, c1d_z_8x8x8[73][1], c1d_z_8x8x8[73][2]),
                                      offset8x8x8(c1d_z_8x8x8[74][0]+1, c1d_z_8x8x8[74][1], c1d_z_8x8x8[74][2])};

  return offset[i];
}

MGARDX_EXEC int Coeff2D_MM_Offset_8x8x8(SIZE i) {
  static constexpr int offset[135] = {offset8x8x8(c2d_xy_8x8x8[0][0], c2d_xy_8x8x8[0][1], c2d_xy_8x8x8[0][2]), // XY
                                      offset8x8x8(c2d_xy_8x8x8[1][0], c2d_xy_8x8x8[1][1], c2d_xy_8x8x8[1][2]),
                                      offset8x8x8(c2d_xy_8x8x8[2][0], c2d_xy_8x8x8[2][1], c2d_xy_8x8x8[2][2]),
                                      offset8x8x8(c2d_xy_8x8x8[3][0], c2d_xy_8x8x8[3][1], c2d_xy_8x8x8[3][2]),
                                      offset8x8x8(c2d_xy_8x8x8[4][0], c2d_xy_8x8x8[4][1], c2d_xy_8x8x8[4][2]),
                                      offset8x8x8(c2d_xy_8x8x8[5][0], c2d_xy_8x8x8[5][1], c2d_xy_8x8x8[5][2]),
                                      offset8x8x8(c2d_xy_8x8x8[6][0], c2d_xy_8x8x8[6][1], c2d_xy_8x8x8[6][2]),
                                      offset8x8x8(c2d_xy_8x8x8[7][0], c2d_xy_8x8x8[7][1], c2d_xy_8x8x8[7][2]),
                                      offset8x8x8(c2d_xy_8x8x8[8][0], c2d_xy_8x8x8[8][1], c2d_xy_8x8x8[8][2]),
                                      offset8x8x8(c2d_xy_8x8x8[9][0], c2d_xy_8x8x8[9][1], c2d_xy_8x8x8[9][2]),
                                      offset8x8x8(c2d_xy_8x8x8[10][0], c2d_xy_8x8x8[10][1], c2d_xy_8x8x8[10][2]),
                                      offset8x8x8(c2d_xy_8x8x8[11][0], c2d_xy_8x8x8[11][1], c2d_xy_8x8x8[11][2]),
                                      offset8x8x8(c2d_xy_8x8x8[12][0], c2d_xy_8x8x8[12][1], c2d_xy_8x8x8[12][2]),
                                      offset8x8x8(c2d_xy_8x8x8[13][0], c2d_xy_8x8x8[13][1], c2d_xy_8x8x8[13][2]),
                                      offset8x8x8(c2d_xy_8x8x8[14][0], c2d_xy_8x8x8[14][1], c2d_xy_8x8x8[14][2]),
                                      offset8x8x8(c2d_xy_8x8x8[15][0], c2d_xy_8x8x8[15][1], c2d_xy_8x8x8[15][2]),
                                      offset8x8x8(c2d_xy_8x8x8[16][0], c2d_xy_8x8x8[16][1], c2d_xy_8x8x8[16][2]),
                                      offset8x8x8(c2d_xy_8x8x8[17][0], c2d_xy_8x8x8[17][1], c2d_xy_8x8x8[17][2]),
                                      offset8x8x8(c2d_xy_8x8x8[18][0], c2d_xy_8x8x8[18][1], c2d_xy_8x8x8[18][2]),
                                      offset8x8x8(c2d_xy_8x8x8[19][0], c2d_xy_8x8x8[19][1], c2d_xy_8x8x8[19][2]),
                                      offset8x8x8(c2d_xy_8x8x8[20][0], c2d_xy_8x8x8[20][1], c2d_xy_8x8x8[20][2]),
                                      offset8x8x8(c2d_xy_8x8x8[21][0], c2d_xy_8x8x8[21][1], c2d_xy_8x8x8[21][2]),
                                      offset8x8x8(c2d_xy_8x8x8[22][0], c2d_xy_8x8x8[22][1], c2d_xy_8x8x8[22][2]),
                                      offset8x8x8(c2d_xy_8x8x8[23][0], c2d_xy_8x8x8[23][1], c2d_xy_8x8x8[23][2]),
                                      offset8x8x8(c2d_xy_8x8x8[24][0], c2d_xy_8x8x8[24][1], c2d_xy_8x8x8[24][2]),
                                      offset8x8x8(c2d_xy_8x8x8[25][0], c2d_xy_8x8x8[25][1], c2d_xy_8x8x8[25][2]),
                                      offset8x8x8(c2d_xy_8x8x8[26][0], c2d_xy_8x8x8[26][1], c2d_xy_8x8x8[26][2]),
                                      offset8x8x8(c2d_xy_8x8x8[27][0], c2d_xy_8x8x8[27][1], c2d_xy_8x8x8[27][2]),
                                      offset8x8x8(c2d_xy_8x8x8[28][0], c2d_xy_8x8x8[28][1], c2d_xy_8x8x8[28][2]),
                                      offset8x8x8(c2d_xy_8x8x8[29][0], c2d_xy_8x8x8[29][1], c2d_xy_8x8x8[29][2]),
                                      offset8x8x8(c2d_xy_8x8x8[30][0], c2d_xy_8x8x8[30][1], c2d_xy_8x8x8[30][2]),
                                      offset8x8x8(c2d_xy_8x8x8[31][0], c2d_xy_8x8x8[31][1], c2d_xy_8x8x8[31][2]),
                                      offset8x8x8(c2d_xy_8x8x8[32][0], c2d_xy_8x8x8[32][1], c2d_xy_8x8x8[32][2]),
                                      offset8x8x8(c2d_xy_8x8x8[33][0], c2d_xy_8x8x8[33][1], c2d_xy_8x8x8[33][2]),
                                      offset8x8x8(c2d_xy_8x8x8[34][0], c2d_xy_8x8x8[34][1], c2d_xy_8x8x8[34][2]),
                                      offset8x8x8(c2d_xy_8x8x8[35][0], c2d_xy_8x8x8[35][1], c2d_xy_8x8x8[35][2]),
                                      offset8x8x8(c2d_xy_8x8x8[36][0], c2d_xy_8x8x8[36][1], c2d_xy_8x8x8[36][2]),
                                      offset8x8x8(c2d_xy_8x8x8[37][0], c2d_xy_8x8x8[37][1], c2d_xy_8x8x8[37][2]),
                                      offset8x8x8(c2d_xy_8x8x8[38][0], c2d_xy_8x8x8[38][1], c2d_xy_8x8x8[38][2]),
                                      offset8x8x8(c2d_xy_8x8x8[39][0], c2d_xy_8x8x8[39][1], c2d_xy_8x8x8[39][2]),
                                      offset8x8x8(c2d_xy_8x8x8[40][0], c2d_xy_8x8x8[40][1], c2d_xy_8x8x8[40][2]),
                                      offset8x8x8(c2d_xy_8x8x8[41][0], c2d_xy_8x8x8[41][1], c2d_xy_8x8x8[41][2]),
                                      offset8x8x8(c2d_xy_8x8x8[42][0], c2d_xy_8x8x8[42][1], c2d_xy_8x8x8[42][2]),
                                      offset8x8x8(c2d_xy_8x8x8[43][0], c2d_xy_8x8x8[43][1], c2d_xy_8x8x8[43][2]),
                                      offset8x8x8(c2d_xy_8x8x8[44][0], c2d_xy_8x8x8[44][1], c2d_xy_8x8x8[44][2]),

                                      offset8x8x8(c2d_xz_8x8x8[0][0], c2d_xz_8x8x8[0][1], c2d_xz_8x8x8[0][2]), // XZ
                                      offset8x8x8(c2d_xz_8x8x8[1][0], c2d_xz_8x8x8[1][1], c2d_xz_8x8x8[1][2]),
                                      offset8x8x8(c2d_xz_8x8x8[2][0], c2d_xz_8x8x8[2][1], c2d_xz_8x8x8[2][2]),
                                      offset8x8x8(c2d_xz_8x8x8[3][0], c2d_xz_8x8x8[3][1], c2d_xz_8x8x8[3][2]),
                                      offset8x8x8(c2d_xz_8x8x8[4][0], c2d_xz_8x8x8[4][1], c2d_xz_8x8x8[4][2]),
                                      offset8x8x8(c2d_xz_8x8x8[5][0], c2d_xz_8x8x8[5][1], c2d_xz_8x8x8[5][2]),
                                      offset8x8x8(c2d_xz_8x8x8[6][0], c2d_xz_8x8x8[6][1], c2d_xz_8x8x8[6][2]),
                                      offset8x8x8(c2d_xz_8x8x8[7][0], c2d_xz_8x8x8[7][1], c2d_xz_8x8x8[7][2]),
                                      offset8x8x8(c2d_xz_8x8x8[8][0], c2d_xz_8x8x8[8][1], c2d_xz_8x8x8[8][2]),
                                      offset8x8x8(c2d_xz_8x8x8[9][0], c2d_xz_8x8x8[9][1], c2d_xz_8x8x8[9][2]),
                                      offset8x8x8(c2d_xz_8x8x8[10][0], c2d_xz_8x8x8[10][1], c2d_xz_8x8x8[10][2]),
                                      offset8x8x8(c2d_xz_8x8x8[11][0], c2d_xz_8x8x8[11][1], c2d_xz_8x8x8[11][2]),
                                      offset8x8x8(c2d_xz_8x8x8[12][0], c2d_xz_8x8x8[12][1], c2d_xz_8x8x8[12][2]),
                                      offset8x8x8(c2d_xz_8x8x8[13][0], c2d_xz_8x8x8[13][1], c2d_xz_8x8x8[13][2]),
                                      offset8x8x8(c2d_xz_8x8x8[14][0], c2d_xz_8x8x8[14][1], c2d_xz_8x8x8[14][2]),
                                      offset8x8x8(c2d_xz_8x8x8[15][0], c2d_xz_8x8x8[15][1], c2d_xz_8x8x8[15][2]),
                                      offset8x8x8(c2d_xz_8x8x8[16][0], c2d_xz_8x8x8[16][1], c2d_xz_8x8x8[16][2]),
                                      offset8x8x8(c2d_xz_8x8x8[17][0], c2d_xz_8x8x8[17][1], c2d_xz_8x8x8[17][2]),
                                      offset8x8x8(c2d_xz_8x8x8[18][0], c2d_xz_8x8x8[18][1], c2d_xz_8x8x8[18][2]),
                                      offset8x8x8(c2d_xz_8x8x8[19][0], c2d_xz_8x8x8[19][1], c2d_xz_8x8x8[19][2]),
                                      offset8x8x8(c2d_xz_8x8x8[20][0], c2d_xz_8x8x8[20][1], c2d_xz_8x8x8[20][2]),
                                      offset8x8x8(c2d_xz_8x8x8[21][0], c2d_xz_8x8x8[21][1], c2d_xz_8x8x8[21][2]),
                                      offset8x8x8(c2d_xz_8x8x8[22][0], c2d_xz_8x8x8[22][1], c2d_xz_8x8x8[22][2]),
                                      offset8x8x8(c2d_xz_8x8x8[23][0], c2d_xz_8x8x8[23][1], c2d_xz_8x8x8[23][2]),
                                      offset8x8x8(c2d_xz_8x8x8[24][0], c2d_xz_8x8x8[24][1], c2d_xz_8x8x8[24][2]),
                                      offset8x8x8(c2d_xz_8x8x8[25][0], c2d_xz_8x8x8[25][1], c2d_xz_8x8x8[25][2]),
                                      offset8x8x8(c2d_xz_8x8x8[26][0], c2d_xz_8x8x8[26][1], c2d_xz_8x8x8[26][2]),
                                      offset8x8x8(c2d_xz_8x8x8[27][0], c2d_xz_8x8x8[27][1], c2d_xz_8x8x8[27][2]),
                                      offset8x8x8(c2d_xz_8x8x8[28][0], c2d_xz_8x8x8[28][1], c2d_xz_8x8x8[28][2]),
                                      offset8x8x8(c2d_xz_8x8x8[29][0], c2d_xz_8x8x8[29][1], c2d_xz_8x8x8[29][2]),
                                      offset8x8x8(c2d_xz_8x8x8[30][0], c2d_xz_8x8x8[30][1], c2d_xz_8x8x8[30][2]),
                                      offset8x8x8(c2d_xz_8x8x8[31][0], c2d_xz_8x8x8[31][1], c2d_xz_8x8x8[31][2]),
                                      offset8x8x8(c2d_xz_8x8x8[32][0], c2d_xz_8x8x8[32][1], c2d_xz_8x8x8[32][2]),
                                      offset8x8x8(c2d_xz_8x8x8[33][0], c2d_xz_8x8x8[33][1], c2d_xz_8x8x8[33][2]),
                                      offset8x8x8(c2d_xz_8x8x8[34][0], c2d_xz_8x8x8[34][1], c2d_xz_8x8x8[34][2]),
                                      offset8x8x8(c2d_xz_8x8x8[35][0], c2d_xz_8x8x8[35][1], c2d_xz_8x8x8[35][2]),
                                      offset8x8x8(c2d_xz_8x8x8[36][0], c2d_xz_8x8x8[36][1], c2d_xz_8x8x8[36][2]),
                                      offset8x8x8(c2d_xz_8x8x8[37][0], c2d_xz_8x8x8[37][1], c2d_xz_8x8x8[37][2]),
                                      offset8x8x8(c2d_xz_8x8x8[38][0], c2d_xz_8x8x8[38][1], c2d_xz_8x8x8[38][2]),
                                      offset8x8x8(c2d_xz_8x8x8[39][0], c2d_xz_8x8x8[39][1], c2d_xz_8x8x8[39][2]),
                                      offset8x8x8(c2d_xz_8x8x8[40][0], c2d_xz_8x8x8[40][1], c2d_xz_8x8x8[40][2]),
                                      offset8x8x8(c2d_xz_8x8x8[41][0], c2d_xz_8x8x8[41][1], c2d_xz_8x8x8[41][2]),
                                      offset8x8x8(c2d_xz_8x8x8[42][0], c2d_xz_8x8x8[42][1], c2d_xz_8x8x8[42][2]),
                                      offset8x8x8(c2d_xz_8x8x8[43][0], c2d_xz_8x8x8[43][1], c2d_xz_8x8x8[43][2]),
                                      offset8x8x8(c2d_xz_8x8x8[44][0], c2d_xz_8x8x8[44][1], c2d_xz_8x8x8[44][2]),

                                      offset8x8x8(c2d_yz_8x8x8[0][0], c2d_yz_8x8x8[0][1], c2d_yz_8x8x8[0][2]), // YZ
                                      offset8x8x8(c2d_yz_8x8x8[1][0], c2d_yz_8x8x8[1][1], c2d_yz_8x8x8[1][2]),
                                      offset8x8x8(c2d_yz_8x8x8[2][0], c2d_yz_8x8x8[2][1], c2d_yz_8x8x8[2][2]),
                                      offset8x8x8(c2d_yz_8x8x8[3][0], c2d_yz_8x8x8[3][1], c2d_yz_8x8x8[3][2]),
                                      offset8x8x8(c2d_yz_8x8x8[4][0], c2d_yz_8x8x8[4][1], c2d_yz_8x8x8[4][2]),
                                      offset8x8x8(c2d_yz_8x8x8[5][0], c2d_yz_8x8x8[5][1], c2d_yz_8x8x8[5][2]),
                                      offset8x8x8(c2d_yz_8x8x8[6][0], c2d_yz_8x8x8[6][1], c2d_yz_8x8x8[6][2]),
                                      offset8x8x8(c2d_yz_8x8x8[7][0], c2d_yz_8x8x8[7][1], c2d_yz_8x8x8[7][2]),
                                      offset8x8x8(c2d_yz_8x8x8[8][0], c2d_yz_8x8x8[8][1], c2d_yz_8x8x8[8][2]),
                                      offset8x8x8(c2d_yz_8x8x8[9][0], c2d_yz_8x8x8[9][1], c2d_yz_8x8x8[9][2]),
                                      offset8x8x8(c2d_yz_8x8x8[10][0], c2d_yz_8x8x8[10][1], c2d_yz_8x8x8[10][2]),
                                      offset8x8x8(c2d_yz_8x8x8[11][0], c2d_yz_8x8x8[11][1], c2d_yz_8x8x8[11][2]),
                                      offset8x8x8(c2d_yz_8x8x8[12][0], c2d_yz_8x8x8[12][1], c2d_yz_8x8x8[12][2]),
                                      offset8x8x8(c2d_yz_8x8x8[13][0], c2d_yz_8x8x8[13][1], c2d_yz_8x8x8[13][2]),
                                      offset8x8x8(c2d_yz_8x8x8[14][0], c2d_yz_8x8x8[14][1], c2d_yz_8x8x8[14][2]),
                                      offset8x8x8(c2d_yz_8x8x8[15][0], c2d_yz_8x8x8[15][1], c2d_yz_8x8x8[15][2]),
                                      offset8x8x8(c2d_yz_8x8x8[16][0], c2d_yz_8x8x8[16][1], c2d_yz_8x8x8[16][2]),
                                      offset8x8x8(c2d_yz_8x8x8[17][0], c2d_yz_8x8x8[17][1], c2d_yz_8x8x8[17][2]),
                                      offset8x8x8(c2d_yz_8x8x8[18][0], c2d_yz_8x8x8[18][1], c2d_yz_8x8x8[18][2]),
                                      offset8x8x8(c2d_yz_8x8x8[19][0], c2d_yz_8x8x8[19][1], c2d_yz_8x8x8[19][2]),
                                      offset8x8x8(c2d_yz_8x8x8[20][0], c2d_yz_8x8x8[20][1], c2d_yz_8x8x8[20][2]),
                                      offset8x8x8(c2d_yz_8x8x8[21][0], c2d_yz_8x8x8[21][1], c2d_yz_8x8x8[21][2]),
                                      offset8x8x8(c2d_yz_8x8x8[22][0], c2d_yz_8x8x8[22][1], c2d_yz_8x8x8[22][2]),
                                      offset8x8x8(c2d_yz_8x8x8[23][0], c2d_yz_8x8x8[23][1], c2d_yz_8x8x8[23][2]),
                                      offset8x8x8(c2d_yz_8x8x8[24][0], c2d_yz_8x8x8[24][1], c2d_yz_8x8x8[24][2]),
                                      offset8x8x8(c2d_yz_8x8x8[25][0], c2d_yz_8x8x8[25][1], c2d_yz_8x8x8[25][2]),
                                      offset8x8x8(c2d_yz_8x8x8[26][0], c2d_yz_8x8x8[26][1], c2d_yz_8x8x8[26][2]),
                                      offset8x8x8(c2d_yz_8x8x8[27][0], c2d_yz_8x8x8[27][1], c2d_yz_8x8x8[27][2]),
                                      offset8x8x8(c2d_yz_8x8x8[28][0], c2d_yz_8x8x8[28][1], c2d_yz_8x8x8[28][2]),
                                      offset8x8x8(c2d_yz_8x8x8[29][0], c2d_yz_8x8x8[29][1], c2d_yz_8x8x8[29][2]),
                                      offset8x8x8(c2d_yz_8x8x8[30][0], c2d_yz_8x8x8[30][1], c2d_yz_8x8x8[30][2]),
                                      offset8x8x8(c2d_yz_8x8x8[31][0], c2d_yz_8x8x8[31][1], c2d_yz_8x8x8[31][2]),
                                      offset8x8x8(c2d_yz_8x8x8[32][0], c2d_yz_8x8x8[32][1], c2d_yz_8x8x8[32][2]),
                                      offset8x8x8(c2d_yz_8x8x8[33][0], c2d_yz_8x8x8[33][1], c2d_yz_8x8x8[33][2]),
                                      offset8x8x8(c2d_yz_8x8x8[34][0], c2d_yz_8x8x8[34][1], c2d_yz_8x8x8[34][2]),
                                      offset8x8x8(c2d_yz_8x8x8[35][0], c2d_yz_8x8x8[35][1], c2d_yz_8x8x8[35][2]),
                                      offset8x8x8(c2d_yz_8x8x8[36][0], c2d_yz_8x8x8[36][1], c2d_yz_8x8x8[36][2]),
                                      offset8x8x8(c2d_yz_8x8x8[37][0], c2d_yz_8x8x8[37][1], c2d_yz_8x8x8[37][2]),
                                      offset8x8x8(c2d_yz_8x8x8[38][0], c2d_yz_8x8x8[38][1], c2d_yz_8x8x8[38][2]),
                                      offset8x8x8(c2d_yz_8x8x8[39][0], c2d_yz_8x8x8[39][1], c2d_yz_8x8x8[39][2]),
                                      offset8x8x8(c2d_yz_8x8x8[40][0], c2d_yz_8x8x8[40][1], c2d_yz_8x8x8[40][2]),
                                      offset8x8x8(c2d_yz_8x8x8[41][0], c2d_yz_8x8x8[41][1], c2d_yz_8x8x8[41][2]),
                                      offset8x8x8(c2d_yz_8x8x8[42][0], c2d_yz_8x8x8[42][1], c2d_yz_8x8x8[42][2]),
                                      offset8x8x8(c2d_yz_8x8x8[43][0], c2d_yz_8x8x8[43][1], c2d_yz_8x8x8[43][2]),
                                      offset8x8x8(c2d_yz_8x8x8[44][0], c2d_yz_8x8x8[44][1], c2d_yz_8x8x8[44][2])};
  return offset[i];
}

MGARDX_EXEC int Coeff2D_RL_Offset_8x8x8(SIZE i) {
  static constexpr int offset[135] = {offset8x8x8(c2d_xy_8x8x8[0][0], c2d_xy_8x8x8[0][1]+1, c2d_xy_8x8x8[0][2]-1), // XY
                                      offset8x8x8(c2d_xy_8x8x8[1][0], c2d_xy_8x8x8[1][1]+1, c2d_xy_8x8x8[1][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[2][0], c2d_xy_8x8x8[2][1]+1, c2d_xy_8x8x8[2][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[3][0], c2d_xy_8x8x8[3][1]+1, c2d_xy_8x8x8[3][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[4][0], c2d_xy_8x8x8[4][1]+1, c2d_xy_8x8x8[4][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[5][0], c2d_xy_8x8x8[5][1]+1, c2d_xy_8x8x8[5][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[6][0], c2d_xy_8x8x8[6][1]+1, c2d_xy_8x8x8[6][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[7][0], c2d_xy_8x8x8[7][1]+1, c2d_xy_8x8x8[7][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[8][0], c2d_xy_8x8x8[8][1]+1, c2d_xy_8x8x8[8][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[9][0], c2d_xy_8x8x8[9][1]+1, c2d_xy_8x8x8[9][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[10][0], c2d_xy_8x8x8[10][1]+1, c2d_xy_8x8x8[10][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[11][0], c2d_xy_8x8x8[11][1]+1, c2d_xy_8x8x8[11][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[12][0], c2d_xy_8x8x8[12][1]+1, c2d_xy_8x8x8[12][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[13][0], c2d_xy_8x8x8[13][1]+1, c2d_xy_8x8x8[13][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[14][0], c2d_xy_8x8x8[14][1]+1, c2d_xy_8x8x8[14][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[15][0], c2d_xy_8x8x8[15][1]+1, c2d_xy_8x8x8[15][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[16][0], c2d_xy_8x8x8[16][1]+1, c2d_xy_8x8x8[16][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[17][0], c2d_xy_8x8x8[17][1]+1, c2d_xy_8x8x8[17][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[18][0], c2d_xy_8x8x8[18][1]+1, c2d_xy_8x8x8[18][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[19][0], c2d_xy_8x8x8[19][1]+1, c2d_xy_8x8x8[19][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[20][0], c2d_xy_8x8x8[20][1]+1, c2d_xy_8x8x8[20][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[21][0], c2d_xy_8x8x8[21][1]+1, c2d_xy_8x8x8[21][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[22][0], c2d_xy_8x8x8[22][1]+1, c2d_xy_8x8x8[22][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[23][0], c2d_xy_8x8x8[23][1]+1, c2d_xy_8x8x8[23][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[24][0], c2d_xy_8x8x8[24][1]+1, c2d_xy_8x8x8[24][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[25][0], c2d_xy_8x8x8[25][1]+1, c2d_xy_8x8x8[25][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[26][0], c2d_xy_8x8x8[26][1]+1, c2d_xy_8x8x8[26][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[27][0], c2d_xy_8x8x8[27][1]+1, c2d_xy_8x8x8[27][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[28][0], c2d_xy_8x8x8[28][1]+1, c2d_xy_8x8x8[28][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[29][0], c2d_xy_8x8x8[29][1]+1, c2d_xy_8x8x8[29][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[30][0], c2d_xy_8x8x8[30][1]+1, c2d_xy_8x8x8[30][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[31][0], c2d_xy_8x8x8[31][1]+1, c2d_xy_8x8x8[31][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[32][0], c2d_xy_8x8x8[32][1]+1, c2d_xy_8x8x8[32][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[33][0], c2d_xy_8x8x8[33][1]+1, c2d_xy_8x8x8[33][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[34][0], c2d_xy_8x8x8[34][1]+1, c2d_xy_8x8x8[34][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[35][0], c2d_xy_8x8x8[35][1]+1, c2d_xy_8x8x8[35][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[36][0], c2d_xy_8x8x8[36][1]+1, c2d_xy_8x8x8[36][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[37][0], c2d_xy_8x8x8[37][1]+1, c2d_xy_8x8x8[37][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[38][0], c2d_xy_8x8x8[38][1]+1, c2d_xy_8x8x8[38][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[39][0], c2d_xy_8x8x8[39][1]+1, c2d_xy_8x8x8[39][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[40][0], c2d_xy_8x8x8[40][1]+1, c2d_xy_8x8x8[40][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[41][0], c2d_xy_8x8x8[41][1]+1, c2d_xy_8x8x8[41][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[42][0], c2d_xy_8x8x8[42][1]+1, c2d_xy_8x8x8[42][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[43][0], c2d_xy_8x8x8[43][1]+1, c2d_xy_8x8x8[43][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[44][0], c2d_xy_8x8x8[44][1]+1, c2d_xy_8x8x8[44][2]-1),

                                      offset8x8x8(c2d_xz_8x8x8[0][0]+1, c2d_xz_8x8x8[0][1], c2d_xz_8x8x8[0][2]-1), // XZ
                                      offset8x8x8(c2d_xz_8x8x8[1][0]+1, c2d_xz_8x8x8[1][1], c2d_xz_8x8x8[1][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[2][0]+1, c2d_xz_8x8x8[2][1], c2d_xz_8x8x8[2][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[3][0]+1, c2d_xz_8x8x8[3][1], c2d_xz_8x8x8[3][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[4][0]+1, c2d_xz_8x8x8[4][1], c2d_xz_8x8x8[4][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[5][0]+1, c2d_xz_8x8x8[5][1], c2d_xz_8x8x8[5][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[6][0]+1, c2d_xz_8x8x8[6][1], c2d_xz_8x8x8[6][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[7][0]+1, c2d_xz_8x8x8[7][1], c2d_xz_8x8x8[7][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[8][0]+1, c2d_xz_8x8x8[8][1], c2d_xz_8x8x8[8][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[9][0]+1, c2d_xz_8x8x8[9][1], c2d_xz_8x8x8[9][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[10][0]+1, c2d_xz_8x8x8[10][1], c2d_xz_8x8x8[10][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[11][0]+1, c2d_xz_8x8x8[11][1], c2d_xz_8x8x8[11][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[12][0]+1, c2d_xz_8x8x8[12][1], c2d_xz_8x8x8[12][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[13][0]+1, c2d_xz_8x8x8[13][1], c2d_xz_8x8x8[13][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[14][0]+1, c2d_xz_8x8x8[14][1], c2d_xz_8x8x8[14][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[15][0]+1, c2d_xz_8x8x8[15][1], c2d_xz_8x8x8[15][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[16][0]+1, c2d_xz_8x8x8[16][1], c2d_xz_8x8x8[16][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[17][0]+1, c2d_xz_8x8x8[17][1], c2d_xz_8x8x8[17][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[18][0]+1, c2d_xz_8x8x8[18][1], c2d_xz_8x8x8[18][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[19][0]+1, c2d_xz_8x8x8[19][1], c2d_xz_8x8x8[19][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[20][0]+1, c2d_xz_8x8x8[20][1], c2d_xz_8x8x8[20][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[21][0]+1, c2d_xz_8x8x8[21][1], c2d_xz_8x8x8[21][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[22][0]+1, c2d_xz_8x8x8[22][1], c2d_xz_8x8x8[22][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[23][0]+1, c2d_xz_8x8x8[23][1], c2d_xz_8x8x8[23][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[24][0]+1, c2d_xz_8x8x8[24][1], c2d_xz_8x8x8[24][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[25][0]+1, c2d_xz_8x8x8[25][1], c2d_xz_8x8x8[25][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[26][0]+1, c2d_xz_8x8x8[26][1], c2d_xz_8x8x8[26][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[27][0]+1, c2d_xz_8x8x8[27][1], c2d_xz_8x8x8[27][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[28][0]+1, c2d_xz_8x8x8[28][1], c2d_xz_8x8x8[28][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[29][0]+1, c2d_xz_8x8x8[29][1], c2d_xz_8x8x8[29][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[30][0]+1, c2d_xz_8x8x8[30][1], c2d_xz_8x8x8[30][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[31][0]+1, c2d_xz_8x8x8[31][1], c2d_xz_8x8x8[31][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[32][0]+1, c2d_xz_8x8x8[32][1], c2d_xz_8x8x8[32][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[33][0]+1, c2d_xz_8x8x8[33][1], c2d_xz_8x8x8[33][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[34][0]+1, c2d_xz_8x8x8[34][1], c2d_xz_8x8x8[34][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[35][0]+1, c2d_xz_8x8x8[35][1], c2d_xz_8x8x8[35][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[36][0]+1, c2d_xz_8x8x8[36][1], c2d_xz_8x8x8[36][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[37][0]+1, c2d_xz_8x8x8[37][1], c2d_xz_8x8x8[37][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[38][0]+1, c2d_xz_8x8x8[38][1], c2d_xz_8x8x8[38][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[39][0]+1, c2d_xz_8x8x8[39][1], c2d_xz_8x8x8[39][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[40][0]+1, c2d_xz_8x8x8[40][1], c2d_xz_8x8x8[40][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[41][0]+1, c2d_xz_8x8x8[41][1], c2d_xz_8x8x8[41][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[42][0]+1, c2d_xz_8x8x8[42][1], c2d_xz_8x8x8[42][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[43][0]+1, c2d_xz_8x8x8[43][1], c2d_xz_8x8x8[43][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[44][0]+1, c2d_xz_8x8x8[44][1], c2d_xz_8x8x8[44][2]-1),

                                      offset8x8x8(c2d_yz_8x8x8[0][0]+1, c2d_yz_8x8x8[0][1]-1, c2d_yz_8x8x8[0][2]), // YZ
                                      offset8x8x8(c2d_yz_8x8x8[1][0]+1, c2d_yz_8x8x8[1][1]-1, c2d_yz_8x8x8[1][2]),
                                      offset8x8x8(c2d_yz_8x8x8[2][0]+1, c2d_yz_8x8x8[2][1]-1, c2d_yz_8x8x8[2][2]),
                                      offset8x8x8(c2d_yz_8x8x8[3][0]+1, c2d_yz_8x8x8[3][1]-1, c2d_yz_8x8x8[3][2]),
                                      offset8x8x8(c2d_yz_8x8x8[4][0]+1, c2d_yz_8x8x8[4][1]-1, c2d_yz_8x8x8[4][2]),
                                      offset8x8x8(c2d_yz_8x8x8[5][0]+1, c2d_yz_8x8x8[5][1]-1, c2d_yz_8x8x8[5][2]),
                                      offset8x8x8(c2d_yz_8x8x8[6][0]+1, c2d_yz_8x8x8[6][1]-1, c2d_yz_8x8x8[6][2]),
                                      offset8x8x8(c2d_yz_8x8x8[7][0]+1, c2d_yz_8x8x8[7][1]-1, c2d_yz_8x8x8[7][2]),
                                      offset8x8x8(c2d_yz_8x8x8[8][0]+1, c2d_yz_8x8x8[8][1]-1, c2d_yz_8x8x8[8][2]),
                                      offset8x8x8(c2d_yz_8x8x8[9][0]+1, c2d_yz_8x8x8[9][1]-1, c2d_yz_8x8x8[9][2]),
                                      offset8x8x8(c2d_yz_8x8x8[10][0]+1, c2d_yz_8x8x8[10][1]-1, c2d_yz_8x8x8[10][2]),
                                      offset8x8x8(c2d_yz_8x8x8[11][0]+1, c2d_yz_8x8x8[11][1]-1, c2d_yz_8x8x8[11][2]),
                                      offset8x8x8(c2d_yz_8x8x8[12][0]+1, c2d_yz_8x8x8[12][1]-1, c2d_yz_8x8x8[12][2]),
                                      offset8x8x8(c2d_yz_8x8x8[13][0]+1, c2d_yz_8x8x8[13][1]-1, c2d_yz_8x8x8[13][2]),
                                      offset8x8x8(c2d_yz_8x8x8[14][0]+1, c2d_yz_8x8x8[14][1]-1, c2d_yz_8x8x8[14][2]),
                                      offset8x8x8(c2d_yz_8x8x8[15][0]+1, c2d_yz_8x8x8[15][1]-1, c2d_yz_8x8x8[15][2]),
                                      offset8x8x8(c2d_yz_8x8x8[16][0]+1, c2d_yz_8x8x8[16][1]-1, c2d_yz_8x8x8[16][2]),
                                      offset8x8x8(c2d_yz_8x8x8[17][0]+1, c2d_yz_8x8x8[17][1]-1, c2d_yz_8x8x8[17][2]),
                                      offset8x8x8(c2d_yz_8x8x8[18][0]+1, c2d_yz_8x8x8[18][1]-1, c2d_yz_8x8x8[18][2]),
                                      offset8x8x8(c2d_yz_8x8x8[19][0]+1, c2d_yz_8x8x8[19][1]-1, c2d_yz_8x8x8[19][2]),
                                      offset8x8x8(c2d_yz_8x8x8[20][0]+1, c2d_yz_8x8x8[20][1]-1, c2d_yz_8x8x8[20][2]),
                                      offset8x8x8(c2d_yz_8x8x8[21][0]+1, c2d_yz_8x8x8[21][1]-1, c2d_yz_8x8x8[21][2]),
                                      offset8x8x8(c2d_yz_8x8x8[22][0]+1, c2d_yz_8x8x8[22][1]-1, c2d_yz_8x8x8[22][2]),
                                      offset8x8x8(c2d_yz_8x8x8[23][0]+1, c2d_yz_8x8x8[23][1]-1, c2d_yz_8x8x8[23][2]),
                                      offset8x8x8(c2d_yz_8x8x8[24][0]+1, c2d_yz_8x8x8[24][1]-1, c2d_yz_8x8x8[24][2]),
                                      offset8x8x8(c2d_yz_8x8x8[25][0]+1, c2d_yz_8x8x8[25][1]-1, c2d_yz_8x8x8[25][2]),
                                      offset8x8x8(c2d_yz_8x8x8[26][0]+1, c2d_yz_8x8x8[26][1]-1, c2d_yz_8x8x8[26][2]),
                                      offset8x8x8(c2d_yz_8x8x8[27][0]+1, c2d_yz_8x8x8[27][1]-1, c2d_yz_8x8x8[27][2]),
                                      offset8x8x8(c2d_yz_8x8x8[28][0]+1, c2d_yz_8x8x8[28][1]-1, c2d_yz_8x8x8[28][2]),
                                      offset8x8x8(c2d_yz_8x8x8[29][0]+1, c2d_yz_8x8x8[29][1]-1, c2d_yz_8x8x8[29][2]),
                                      offset8x8x8(c2d_yz_8x8x8[30][0]+1, c2d_yz_8x8x8[30][1]-1, c2d_yz_8x8x8[30][2]),
                                      offset8x8x8(c2d_yz_8x8x8[31][0]+1, c2d_yz_8x8x8[31][1]-1, c2d_yz_8x8x8[31][2]),
                                      offset8x8x8(c2d_yz_8x8x8[32][0]+1, c2d_yz_8x8x8[32][1]-1, c2d_yz_8x8x8[32][2]),
                                      offset8x8x8(c2d_yz_8x8x8[33][0]+1, c2d_yz_8x8x8[33][1]-1, c2d_yz_8x8x8[33][2]),
                                      offset8x8x8(c2d_yz_8x8x8[34][0]+1, c2d_yz_8x8x8[34][1]-1, c2d_yz_8x8x8[34][2]),
                                      offset8x8x8(c2d_yz_8x8x8[35][0]+1, c2d_yz_8x8x8[35][1]-1, c2d_yz_8x8x8[35][2]),
                                      offset8x8x8(c2d_yz_8x8x8[36][0]+1, c2d_yz_8x8x8[36][1]-1, c2d_yz_8x8x8[36][2]),
                                      offset8x8x8(c2d_yz_8x8x8[37][0]+1, c2d_yz_8x8x8[37][1]-1, c2d_yz_8x8x8[37][2]),
                                      offset8x8x8(c2d_yz_8x8x8[38][0]+1, c2d_yz_8x8x8[38][1]-1, c2d_yz_8x8x8[38][2]),
                                      offset8x8x8(c2d_yz_8x8x8[39][0]+1, c2d_yz_8x8x8[39][1]-1, c2d_yz_8x8x8[39][2]),
                                      offset8x8x8(c2d_yz_8x8x8[40][0]+1, c2d_yz_8x8x8[40][1]-1, c2d_yz_8x8x8[40][2]),
                                      offset8x8x8(c2d_yz_8x8x8[41][0]+1, c2d_yz_8x8x8[41][1]-1, c2d_yz_8x8x8[41][2]),
                                      offset8x8x8(c2d_yz_8x8x8[42][0]+1, c2d_yz_8x8x8[42][1]-1, c2d_yz_8x8x8[42][2]),
                                      offset8x8x8(c2d_yz_8x8x8[43][0]+1, c2d_yz_8x8x8[43][1]-1, c2d_yz_8x8x8[43][2]),
                                      offset8x8x8(c2d_yz_8x8x8[44][0]+1, c2d_yz_8x8x8[44][1]-1, c2d_yz_8x8x8[44][2])};
  return offset[i];
}

MGARDX_EXEC int Coeff2D_LR_Offset_8x8x8(SIZE i) {
  static constexpr int offset[135] = {offset8x8x8(c2d_xy_8x8x8[0][0], c2d_xy_8x8x8[0][1]-1, c2d_xy_8x8x8[0][2]+1), // XY
                                      offset8x8x8(c2d_xy_8x8x8[1][0], c2d_xy_8x8x8[1][1]-1, c2d_xy_8x8x8[1][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[2][0], c2d_xy_8x8x8[2][1]-1, c2d_xy_8x8x8[2][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[3][0], c2d_xy_8x8x8[3][1]-1, c2d_xy_8x8x8[3][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[4][0], c2d_xy_8x8x8[4][1]-1, c2d_xy_8x8x8[4][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[5][0], c2d_xy_8x8x8[5][1]-1, c2d_xy_8x8x8[5][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[6][0], c2d_xy_8x8x8[6][1]-1, c2d_xy_8x8x8[6][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[7][0], c2d_xy_8x8x8[7][1]-1, c2d_xy_8x8x8[7][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[8][0], c2d_xy_8x8x8[8][1]-1, c2d_xy_8x8x8[8][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[9][0], c2d_xy_8x8x8[9][1]-1, c2d_xy_8x8x8[9][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[10][0], c2d_xy_8x8x8[10][1]-1, c2d_xy_8x8x8[10][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[11][0], c2d_xy_8x8x8[11][1]-1, c2d_xy_8x8x8[11][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[12][0], c2d_xy_8x8x8[12][1]-1, c2d_xy_8x8x8[12][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[13][0], c2d_xy_8x8x8[13][1]-1, c2d_xy_8x8x8[13][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[14][0], c2d_xy_8x8x8[14][1]-1, c2d_xy_8x8x8[14][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[15][0], c2d_xy_8x8x8[15][1]-1, c2d_xy_8x8x8[15][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[16][0], c2d_xy_8x8x8[16][1]-1, c2d_xy_8x8x8[16][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[17][0], c2d_xy_8x8x8[17][1]-1, c2d_xy_8x8x8[17][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[18][0], c2d_xy_8x8x8[18][1]-1, c2d_xy_8x8x8[18][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[19][0], c2d_xy_8x8x8[19][1]-1, c2d_xy_8x8x8[19][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[20][0], c2d_xy_8x8x8[20][1]-1, c2d_xy_8x8x8[20][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[21][0], c2d_xy_8x8x8[21][1]-1, c2d_xy_8x8x8[21][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[22][0], c2d_xy_8x8x8[22][1]-1, c2d_xy_8x8x8[22][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[23][0], c2d_xy_8x8x8[23][1]-1, c2d_xy_8x8x8[23][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[24][0], c2d_xy_8x8x8[24][1]-1, c2d_xy_8x8x8[24][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[25][0], c2d_xy_8x8x8[25][1]-1, c2d_xy_8x8x8[25][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[26][0], c2d_xy_8x8x8[26][1]-1, c2d_xy_8x8x8[26][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[27][0], c2d_xy_8x8x8[27][1]-1, c2d_xy_8x8x8[27][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[28][0], c2d_xy_8x8x8[28][1]-1, c2d_xy_8x8x8[28][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[29][0], c2d_xy_8x8x8[29][1]-1, c2d_xy_8x8x8[29][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[30][0], c2d_xy_8x8x8[30][1]-1, c2d_xy_8x8x8[30][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[31][0], c2d_xy_8x8x8[31][1]-1, c2d_xy_8x8x8[31][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[32][0], c2d_xy_8x8x8[32][1]-1, c2d_xy_8x8x8[32][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[33][0], c2d_xy_8x8x8[33][1]-1, c2d_xy_8x8x8[33][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[34][0], c2d_xy_8x8x8[34][1]-1, c2d_xy_8x8x8[34][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[35][0], c2d_xy_8x8x8[35][1]-1, c2d_xy_8x8x8[35][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[36][0], c2d_xy_8x8x8[36][1]-1, c2d_xy_8x8x8[36][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[37][0], c2d_xy_8x8x8[37][1]-1, c2d_xy_8x8x8[37][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[38][0], c2d_xy_8x8x8[38][1]-1, c2d_xy_8x8x8[38][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[39][0], c2d_xy_8x8x8[39][1]-1, c2d_xy_8x8x8[39][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[40][0], c2d_xy_8x8x8[40][1]-1, c2d_xy_8x8x8[40][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[41][0], c2d_xy_8x8x8[41][1]-1, c2d_xy_8x8x8[41][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[42][0], c2d_xy_8x8x8[42][1]-1, c2d_xy_8x8x8[42][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[43][0], c2d_xy_8x8x8[43][1]-1, c2d_xy_8x8x8[43][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[44][0], c2d_xy_8x8x8[44][1]-1, c2d_xy_8x8x8[44][2]+1),

                                      offset8x8x8(c2d_xz_8x8x8[0][0]-1, c2d_xz_8x8x8[0][1], c2d_xz_8x8x8[0][2]+1), // XZ
                                      offset8x8x8(c2d_xz_8x8x8[1][0]-1, c2d_xz_8x8x8[1][1], c2d_xz_8x8x8[1][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[2][0]-1, c2d_xz_8x8x8[2][1], c2d_xz_8x8x8[2][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[3][0]-1, c2d_xz_8x8x8[3][1], c2d_xz_8x8x8[3][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[4][0]-1, c2d_xz_8x8x8[4][1], c2d_xz_8x8x8[4][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[5][0]-1, c2d_xz_8x8x8[5][1], c2d_xz_8x8x8[5][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[6][0]-1, c2d_xz_8x8x8[6][1], c2d_xz_8x8x8[6][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[7][0]-1, c2d_xz_8x8x8[7][1], c2d_xz_8x8x8[7][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[8][0]-1, c2d_xz_8x8x8[8][1], c2d_xz_8x8x8[8][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[9][0]-1, c2d_xz_8x8x8[9][1], c2d_xz_8x8x8[9][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[10][0]-1, c2d_xz_8x8x8[10][1], c2d_xz_8x8x8[10][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[11][0]-1, c2d_xz_8x8x8[11][1], c2d_xz_8x8x8[11][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[12][0]-1, c2d_xz_8x8x8[12][1], c2d_xz_8x8x8[12][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[13][0]-1, c2d_xz_8x8x8[13][1], c2d_xz_8x8x8[13][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[14][0]-1, c2d_xz_8x8x8[14][1], c2d_xz_8x8x8[14][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[15][0]-1, c2d_xz_8x8x8[15][1], c2d_xz_8x8x8[15][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[16][0]-1, c2d_xz_8x8x8[16][1], c2d_xz_8x8x8[16][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[17][0]-1, c2d_xz_8x8x8[17][1], c2d_xz_8x8x8[17][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[18][0]-1, c2d_xz_8x8x8[18][1], c2d_xz_8x8x8[18][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[19][0]-1, c2d_xz_8x8x8[19][1], c2d_xz_8x8x8[19][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[20][0]-1, c2d_xz_8x8x8[20][1], c2d_xz_8x8x8[20][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[21][0]-1, c2d_xz_8x8x8[21][1], c2d_xz_8x8x8[21][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[22][0]-1, c2d_xz_8x8x8[22][1], c2d_xz_8x8x8[22][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[23][0]-1, c2d_xz_8x8x8[23][1], c2d_xz_8x8x8[23][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[24][0]-1, c2d_xz_8x8x8[24][1], c2d_xz_8x8x8[24][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[25][0]-1, c2d_xz_8x8x8[25][1], c2d_xz_8x8x8[25][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[26][0]-1, c2d_xz_8x8x8[26][1], c2d_xz_8x8x8[26][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[27][0]-1, c2d_xz_8x8x8[27][1], c2d_xz_8x8x8[27][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[28][0]-1, c2d_xz_8x8x8[28][1], c2d_xz_8x8x8[28][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[29][0]-1, c2d_xz_8x8x8[29][1], c2d_xz_8x8x8[29][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[30][0]-1, c2d_xz_8x8x8[30][1], c2d_xz_8x8x8[30][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[31][0]-1, c2d_xz_8x8x8[31][1], c2d_xz_8x8x8[31][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[32][0]-1, c2d_xz_8x8x8[32][1], c2d_xz_8x8x8[32][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[33][0]-1, c2d_xz_8x8x8[33][1], c2d_xz_8x8x8[33][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[34][0]-1, c2d_xz_8x8x8[34][1], c2d_xz_8x8x8[34][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[35][0]-1, c2d_xz_8x8x8[35][1], c2d_xz_8x8x8[35][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[36][0]-1, c2d_xz_8x8x8[36][1], c2d_xz_8x8x8[36][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[37][0]-1, c2d_xz_8x8x8[37][1], c2d_xz_8x8x8[37][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[38][0]-1, c2d_xz_8x8x8[38][1], c2d_xz_8x8x8[38][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[39][0]-1, c2d_xz_8x8x8[39][1], c2d_xz_8x8x8[39][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[40][0]-1, c2d_xz_8x8x8[40][1], c2d_xz_8x8x8[40][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[41][0]-1, c2d_xz_8x8x8[41][1], c2d_xz_8x8x8[41][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[42][0]-1, c2d_xz_8x8x8[42][1], c2d_xz_8x8x8[42][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[43][0]-1, c2d_xz_8x8x8[43][1], c2d_xz_8x8x8[43][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[44][0]-1, c2d_xz_8x8x8[44][1], c2d_xz_8x8x8[44][2]+1),

                                      offset8x8x8(c2d_yz_8x8x8[0][0]-1, c2d_yz_8x8x8[0][1]+1, c2d_yz_8x8x8[0][2]), // YZ
                                      offset8x8x8(c2d_yz_8x8x8[1][0]-1, c2d_yz_8x8x8[1][1]+1, c2d_yz_8x8x8[1][2]),
                                      offset8x8x8(c2d_yz_8x8x8[2][0]-1, c2d_yz_8x8x8[2][1]+1, c2d_yz_8x8x8[2][2]),
                                      offset8x8x8(c2d_yz_8x8x8[3][0]-1, c2d_yz_8x8x8[3][1]+1, c2d_yz_8x8x8[3][2]),
                                      offset8x8x8(c2d_yz_8x8x8[4][0]-1, c2d_yz_8x8x8[4][1]+1, c2d_yz_8x8x8[4][2]),
                                      offset8x8x8(c2d_yz_8x8x8[5][0]-1, c2d_yz_8x8x8[5][1]+1, c2d_yz_8x8x8[5][2]),
                                      offset8x8x8(c2d_yz_8x8x8[6][0]-1, c2d_yz_8x8x8[6][1]+1, c2d_yz_8x8x8[6][2]),
                                      offset8x8x8(c2d_yz_8x8x8[7][0]-1, c2d_yz_8x8x8[7][1]+1, c2d_yz_8x8x8[7][2]),
                                      offset8x8x8(c2d_yz_8x8x8[8][0]-1, c2d_yz_8x8x8[8][1]+1, c2d_yz_8x8x8[8][2]),
                                      offset8x8x8(c2d_yz_8x8x8[9][0]-1, c2d_yz_8x8x8[9][1]+1, c2d_yz_8x8x8[9][2]),
                                      offset8x8x8(c2d_yz_8x8x8[10][0]-1, c2d_yz_8x8x8[10][1]+1, c2d_yz_8x8x8[10][2]),
                                      offset8x8x8(c2d_yz_8x8x8[11][0]-1, c2d_yz_8x8x8[11][1]+1, c2d_yz_8x8x8[11][2]),
                                      offset8x8x8(c2d_yz_8x8x8[12][0]-1, c2d_yz_8x8x8[12][1]+1, c2d_yz_8x8x8[12][2]),
                                      offset8x8x8(c2d_yz_8x8x8[13][0]-1, c2d_yz_8x8x8[13][1]+1, c2d_yz_8x8x8[13][2]),
                                      offset8x8x8(c2d_yz_8x8x8[14][0]-1, c2d_yz_8x8x8[14][1]+1, c2d_yz_8x8x8[14][2]),
                                      offset8x8x8(c2d_yz_8x8x8[15][0]-1, c2d_yz_8x8x8[15][1]+1, c2d_yz_8x8x8[15][2]),
                                      offset8x8x8(c2d_yz_8x8x8[16][0]-1, c2d_yz_8x8x8[16][1]+1, c2d_yz_8x8x8[16][2]),
                                      offset8x8x8(c2d_yz_8x8x8[17][0]-1, c2d_yz_8x8x8[17][1]+1, c2d_yz_8x8x8[17][2]),
                                      offset8x8x8(c2d_yz_8x8x8[18][0]-1, c2d_yz_8x8x8[18][1]+1, c2d_yz_8x8x8[18][2]),
                                      offset8x8x8(c2d_yz_8x8x8[19][0]-1, c2d_yz_8x8x8[19][1]+1, c2d_yz_8x8x8[19][2]),
                                      offset8x8x8(c2d_yz_8x8x8[20][0]-1, c2d_yz_8x8x8[20][1]+1, c2d_yz_8x8x8[20][2]),
                                      offset8x8x8(c2d_yz_8x8x8[21][0]-1, c2d_yz_8x8x8[21][1]+1, c2d_yz_8x8x8[21][2]),
                                      offset8x8x8(c2d_yz_8x8x8[22][0]-1, c2d_yz_8x8x8[22][1]+1, c2d_yz_8x8x8[22][2]),
                                      offset8x8x8(c2d_yz_8x8x8[23][0]-1, c2d_yz_8x8x8[23][1]+1, c2d_yz_8x8x8[23][2]),
                                      offset8x8x8(c2d_yz_8x8x8[24][0]-1, c2d_yz_8x8x8[24][1]+1, c2d_yz_8x8x8[24][2]),
                                      offset8x8x8(c2d_yz_8x8x8[25][0]-1, c2d_yz_8x8x8[25][1]+1, c2d_yz_8x8x8[25][2]),
                                      offset8x8x8(c2d_yz_8x8x8[26][0]-1, c2d_yz_8x8x8[26][1]+1, c2d_yz_8x8x8[26][2]),
                                      offset8x8x8(c2d_yz_8x8x8[27][0]-1, c2d_yz_8x8x8[27][1]+1, c2d_yz_8x8x8[27][2]),
                                      offset8x8x8(c2d_yz_8x8x8[28][0]-1, c2d_yz_8x8x8[28][1]+1, c2d_yz_8x8x8[28][2]),
                                      offset8x8x8(c2d_yz_8x8x8[29][0]-1, c2d_yz_8x8x8[29][1]+1, c2d_yz_8x8x8[29][2]),
                                      offset8x8x8(c2d_yz_8x8x8[30][0]-1, c2d_yz_8x8x8[30][1]+1, c2d_yz_8x8x8[30][2]),
                                      offset8x8x8(c2d_yz_8x8x8[31][0]-1, c2d_yz_8x8x8[31][1]+1, c2d_yz_8x8x8[31][2]),
                                      offset8x8x8(c2d_yz_8x8x8[32][0]-1, c2d_yz_8x8x8[32][1]+1, c2d_yz_8x8x8[32][2]),
                                      offset8x8x8(c2d_yz_8x8x8[33][0]-1, c2d_yz_8x8x8[33][1]+1, c2d_yz_8x8x8[33][2]),
                                      offset8x8x8(c2d_yz_8x8x8[34][0]-1, c2d_yz_8x8x8[34][1]+1, c2d_yz_8x8x8[34][2]),
                                      offset8x8x8(c2d_yz_8x8x8[35][0]-1, c2d_yz_8x8x8[35][1]+1, c2d_yz_8x8x8[35][2]),
                                      offset8x8x8(c2d_yz_8x8x8[36][0]-1, c2d_yz_8x8x8[36][1]+1, c2d_yz_8x8x8[36][2]),
                                      offset8x8x8(c2d_yz_8x8x8[37][0]-1, c2d_yz_8x8x8[37][1]+1, c2d_yz_8x8x8[37][2]),
                                      offset8x8x8(c2d_yz_8x8x8[38][0]-1, c2d_yz_8x8x8[38][1]+1, c2d_yz_8x8x8[38][2]),
                                      offset8x8x8(c2d_yz_8x8x8[39][0]-1, c2d_yz_8x8x8[39][1]+1, c2d_yz_8x8x8[39][2]),
                                      offset8x8x8(c2d_yz_8x8x8[40][0]-1, c2d_yz_8x8x8[40][1]+1, c2d_yz_8x8x8[40][2]),
                                      offset8x8x8(c2d_yz_8x8x8[41][0]-1, c2d_yz_8x8x8[41][1]+1, c2d_yz_8x8x8[41][2]),
                                      offset8x8x8(c2d_yz_8x8x8[42][0]-1, c2d_yz_8x8x8[42][1]+1, c2d_yz_8x8x8[42][2]),
                                      offset8x8x8(c2d_yz_8x8x8[43][0]-1, c2d_yz_8x8x8[43][1]+1, c2d_yz_8x8x8[43][2]),
                                      offset8x8x8(c2d_yz_8x8x8[44][0]-1, c2d_yz_8x8x8[44][1]+1, c2d_yz_8x8x8[44][2])};
  return offset[i];
}

MGARDX_EXEC int Coeff2D_RR_Offset_8x8x8(SIZE i) {
  static constexpr int offset[135] = {offset8x8x8(c2d_xy_8x8x8[0][0], c2d_xy_8x8x8[0][1]+1, c2d_xy_8x8x8[0][2]+1), // XY
                                      offset8x8x8(c2d_xy_8x8x8[1][0], c2d_xy_8x8x8[1][1]+1, c2d_xy_8x8x8[1][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[2][0], c2d_xy_8x8x8[2][1]+1, c2d_xy_8x8x8[2][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[3][0], c2d_xy_8x8x8[3][1]+1, c2d_xy_8x8x8[3][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[4][0], c2d_xy_8x8x8[4][1]+1, c2d_xy_8x8x8[4][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[5][0], c2d_xy_8x8x8[5][1]+1, c2d_xy_8x8x8[5][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[6][0], c2d_xy_8x8x8[6][1]+1, c2d_xy_8x8x8[6][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[7][0], c2d_xy_8x8x8[7][1]+1, c2d_xy_8x8x8[7][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[8][0], c2d_xy_8x8x8[8][1]+1, c2d_xy_8x8x8[8][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[9][0], c2d_xy_8x8x8[9][1]+1, c2d_xy_8x8x8[9][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[10][0], c2d_xy_8x8x8[10][1]+1, c2d_xy_8x8x8[10][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[11][0], c2d_xy_8x8x8[11][1]+1, c2d_xy_8x8x8[11][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[12][0], c2d_xy_8x8x8[12][1]+1, c2d_xy_8x8x8[12][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[13][0], c2d_xy_8x8x8[13][1]+1, c2d_xy_8x8x8[13][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[14][0], c2d_xy_8x8x8[14][1]+1, c2d_xy_8x8x8[14][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[15][0], c2d_xy_8x8x8[15][1]+1, c2d_xy_8x8x8[15][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[16][0], c2d_xy_8x8x8[16][1]+1, c2d_xy_8x8x8[16][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[17][0], c2d_xy_8x8x8[17][1]+1, c2d_xy_8x8x8[17][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[18][0], c2d_xy_8x8x8[18][1]+1, c2d_xy_8x8x8[18][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[19][0], c2d_xy_8x8x8[19][1]+1, c2d_xy_8x8x8[19][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[20][0], c2d_xy_8x8x8[20][1]+1, c2d_xy_8x8x8[20][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[21][0], c2d_xy_8x8x8[21][1]+1, c2d_xy_8x8x8[21][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[22][0], c2d_xy_8x8x8[22][1]+1, c2d_xy_8x8x8[22][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[23][0], c2d_xy_8x8x8[23][1]+1, c2d_xy_8x8x8[23][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[24][0], c2d_xy_8x8x8[24][1]+1, c2d_xy_8x8x8[24][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[25][0], c2d_xy_8x8x8[25][1]+1, c2d_xy_8x8x8[25][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[26][0], c2d_xy_8x8x8[26][1]+1, c2d_xy_8x8x8[26][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[27][0], c2d_xy_8x8x8[27][1]+1, c2d_xy_8x8x8[27][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[28][0], c2d_xy_8x8x8[28][1]+1, c2d_xy_8x8x8[28][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[29][0], c2d_xy_8x8x8[29][1]+1, c2d_xy_8x8x8[29][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[30][0], c2d_xy_8x8x8[30][1]+1, c2d_xy_8x8x8[30][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[31][0], c2d_xy_8x8x8[31][1]+1, c2d_xy_8x8x8[31][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[32][0], c2d_xy_8x8x8[32][1]+1, c2d_xy_8x8x8[32][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[33][0], c2d_xy_8x8x8[33][1]+1, c2d_xy_8x8x8[33][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[34][0], c2d_xy_8x8x8[34][1]+1, c2d_xy_8x8x8[34][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[35][0], c2d_xy_8x8x8[35][1]+1, c2d_xy_8x8x8[35][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[36][0], c2d_xy_8x8x8[36][1]+1, c2d_xy_8x8x8[36][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[37][0], c2d_xy_8x8x8[37][1]+1, c2d_xy_8x8x8[37][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[38][0], c2d_xy_8x8x8[38][1]+1, c2d_xy_8x8x8[38][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[39][0], c2d_xy_8x8x8[39][1]+1, c2d_xy_8x8x8[39][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[40][0], c2d_xy_8x8x8[40][1]+1, c2d_xy_8x8x8[40][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[41][0], c2d_xy_8x8x8[41][1]+1, c2d_xy_8x8x8[41][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[42][0], c2d_xy_8x8x8[42][1]+1, c2d_xy_8x8x8[42][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[43][0], c2d_xy_8x8x8[43][1]+1, c2d_xy_8x8x8[43][2]+1),
                                      offset8x8x8(c2d_xy_8x8x8[44][0], c2d_xy_8x8x8[44][1]+1, c2d_xy_8x8x8[44][2]+1),

                                      offset8x8x8(c2d_xz_8x8x8[0][0]+1, c2d_xz_8x8x8[0][1], c2d_xz_8x8x8[0][2]+1), // XZ
                                      offset8x8x8(c2d_xz_8x8x8[1][0]+1, c2d_xz_8x8x8[1][1], c2d_xz_8x8x8[1][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[2][0]+1, c2d_xz_8x8x8[2][1], c2d_xz_8x8x8[2][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[3][0]+1, c2d_xz_8x8x8[3][1], c2d_xz_8x8x8[3][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[4][0]+1, c2d_xz_8x8x8[4][1], c2d_xz_8x8x8[4][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[5][0]+1, c2d_xz_8x8x8[5][1], c2d_xz_8x8x8[5][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[6][0]+1, c2d_xz_8x8x8[6][1], c2d_xz_8x8x8[6][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[7][0]+1, c2d_xz_8x8x8[7][1], c2d_xz_8x8x8[7][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[8][0]+1, c2d_xz_8x8x8[8][1], c2d_xz_8x8x8[8][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[9][0]+1, c2d_xz_8x8x8[9][1], c2d_xz_8x8x8[9][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[10][0]+1, c2d_xz_8x8x8[10][1], c2d_xz_8x8x8[10][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[11][0]+1, c2d_xz_8x8x8[11][1], c2d_xz_8x8x8[11][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[12][0]+1, c2d_xz_8x8x8[12][1], c2d_xz_8x8x8[12][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[13][0]+1, c2d_xz_8x8x8[13][1], c2d_xz_8x8x8[13][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[14][0]+1, c2d_xz_8x8x8[14][1], c2d_xz_8x8x8[14][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[15][0]+1, c2d_xz_8x8x8[15][1], c2d_xz_8x8x8[15][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[16][0]+1, c2d_xz_8x8x8[16][1], c2d_xz_8x8x8[16][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[17][0]+1, c2d_xz_8x8x8[17][1], c2d_xz_8x8x8[17][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[18][0]+1, c2d_xz_8x8x8[18][1], c2d_xz_8x8x8[18][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[19][0]+1, c2d_xz_8x8x8[19][1], c2d_xz_8x8x8[19][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[20][0]+1, c2d_xz_8x8x8[20][1], c2d_xz_8x8x8[20][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[21][0]+1, c2d_xz_8x8x8[21][1], c2d_xz_8x8x8[21][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[22][0]+1, c2d_xz_8x8x8[22][1], c2d_xz_8x8x8[22][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[23][0]+1, c2d_xz_8x8x8[23][1], c2d_xz_8x8x8[23][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[24][0]+1, c2d_xz_8x8x8[24][1], c2d_xz_8x8x8[24][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[25][0]+1, c2d_xz_8x8x8[25][1], c2d_xz_8x8x8[25][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[26][0]+1, c2d_xz_8x8x8[26][1], c2d_xz_8x8x8[26][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[27][0]+1, c2d_xz_8x8x8[27][1], c2d_xz_8x8x8[27][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[28][0]+1, c2d_xz_8x8x8[28][1], c2d_xz_8x8x8[28][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[29][0]+1, c2d_xz_8x8x8[29][1], c2d_xz_8x8x8[29][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[30][0]+1, c2d_xz_8x8x8[30][1], c2d_xz_8x8x8[30][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[31][0]+1, c2d_xz_8x8x8[31][1], c2d_xz_8x8x8[31][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[32][0]+1, c2d_xz_8x8x8[32][1], c2d_xz_8x8x8[32][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[33][0]+1, c2d_xz_8x8x8[33][1], c2d_xz_8x8x8[33][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[34][0]+1, c2d_xz_8x8x8[34][1], c2d_xz_8x8x8[34][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[35][0]+1, c2d_xz_8x8x8[35][1], c2d_xz_8x8x8[35][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[36][0]+1, c2d_xz_8x8x8[36][1], c2d_xz_8x8x8[36][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[37][0]+1, c2d_xz_8x8x8[37][1], c2d_xz_8x8x8[37][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[38][0]+1, c2d_xz_8x8x8[38][1], c2d_xz_8x8x8[38][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[39][0]+1, c2d_xz_8x8x8[39][1], c2d_xz_8x8x8[39][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[40][0]+1, c2d_xz_8x8x8[40][1], c2d_xz_8x8x8[40][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[41][0]+1, c2d_xz_8x8x8[41][1], c2d_xz_8x8x8[41][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[42][0]+1, c2d_xz_8x8x8[42][1], c2d_xz_8x8x8[42][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[43][0]+1, c2d_xz_8x8x8[43][1], c2d_xz_8x8x8[43][2]+1),
                                      offset8x8x8(c2d_xz_8x8x8[44][0]+1, c2d_xz_8x8x8[44][1], c2d_xz_8x8x8[44][2]+1),

                                      offset8x8x8(c2d_yz_8x8x8[0][0]+1, c2d_yz_8x8x8[0][1]+1, c2d_yz_8x8x8[0][2]), // YZ
                                      offset8x8x8(c2d_yz_8x8x8[1][0]+1, c2d_yz_8x8x8[1][1]+1, c2d_yz_8x8x8[1][2]),
                                      offset8x8x8(c2d_yz_8x8x8[2][0]+1, c2d_yz_8x8x8[2][1]+1, c2d_yz_8x8x8[2][2]),
                                      offset8x8x8(c2d_yz_8x8x8[3][0]+1, c2d_yz_8x8x8[3][1]+1, c2d_yz_8x8x8[3][2]),
                                      offset8x8x8(c2d_yz_8x8x8[4][0]+1, c2d_yz_8x8x8[4][1]+1, c2d_yz_8x8x8[4][2]),
                                      offset8x8x8(c2d_yz_8x8x8[5][0]+1, c2d_yz_8x8x8[5][1]+1, c2d_yz_8x8x8[5][2]),
                                      offset8x8x8(c2d_yz_8x8x8[6][0]+1, c2d_yz_8x8x8[6][1]+1, c2d_yz_8x8x8[6][2]),
                                      offset8x8x8(c2d_yz_8x8x8[7][0]+1, c2d_yz_8x8x8[7][1]+1, c2d_yz_8x8x8[7][2]),
                                      offset8x8x8(c2d_yz_8x8x8[8][0]+1, c2d_yz_8x8x8[8][1]+1, c2d_yz_8x8x8[8][2]),
                                      offset8x8x8(c2d_yz_8x8x8[9][0]+1, c2d_yz_8x8x8[9][1]+1, c2d_yz_8x8x8[9][2]),
                                      offset8x8x8(c2d_yz_8x8x8[10][0]+1, c2d_yz_8x8x8[10][1]+1, c2d_yz_8x8x8[10][2]),
                                      offset8x8x8(c2d_yz_8x8x8[11][0]+1, c2d_yz_8x8x8[11][1]+1, c2d_yz_8x8x8[11][2]),
                                      offset8x8x8(c2d_yz_8x8x8[12][0]+1, c2d_yz_8x8x8[12][1]+1, c2d_yz_8x8x8[12][2]),
                                      offset8x8x8(c2d_yz_8x8x8[13][0]+1, c2d_yz_8x8x8[13][1]+1, c2d_yz_8x8x8[13][2]),
                                      offset8x8x8(c2d_yz_8x8x8[14][0]+1, c2d_yz_8x8x8[14][1]+1, c2d_yz_8x8x8[14][2]),
                                      offset8x8x8(c2d_yz_8x8x8[15][0]+1, c2d_yz_8x8x8[15][1]+1, c2d_yz_8x8x8[15][2]),
                                      offset8x8x8(c2d_yz_8x8x8[16][0]+1, c2d_yz_8x8x8[16][1]+1, c2d_yz_8x8x8[16][2]),
                                      offset8x8x8(c2d_yz_8x8x8[17][0]+1, c2d_yz_8x8x8[17][1]+1, c2d_yz_8x8x8[17][2]),
                                      offset8x8x8(c2d_yz_8x8x8[18][0]+1, c2d_yz_8x8x8[18][1]+1, c2d_yz_8x8x8[18][2]),
                                      offset8x8x8(c2d_yz_8x8x8[19][0]+1, c2d_yz_8x8x8[19][1]+1, c2d_yz_8x8x8[19][2]),
                                      offset8x8x8(c2d_yz_8x8x8[20][0]+1, c2d_yz_8x8x8[20][1]+1, c2d_yz_8x8x8[20][2]),
                                      offset8x8x8(c2d_yz_8x8x8[21][0]+1, c2d_yz_8x8x8[21][1]+1, c2d_yz_8x8x8[21][2]),
                                      offset8x8x8(c2d_yz_8x8x8[22][0]+1, c2d_yz_8x8x8[22][1]+1, c2d_yz_8x8x8[22][2]),
                                      offset8x8x8(c2d_yz_8x8x8[23][0]+1, c2d_yz_8x8x8[23][1]+1, c2d_yz_8x8x8[23][2]),
                                      offset8x8x8(c2d_yz_8x8x8[24][0]+1, c2d_yz_8x8x8[24][1]+1, c2d_yz_8x8x8[24][2]),
                                      offset8x8x8(c2d_yz_8x8x8[25][0]+1, c2d_yz_8x8x8[25][1]+1, c2d_yz_8x8x8[25][2]),
                                      offset8x8x8(c2d_yz_8x8x8[26][0]+1, c2d_yz_8x8x8[26][1]+1, c2d_yz_8x8x8[26][2]),
                                      offset8x8x8(c2d_yz_8x8x8[27][0]+1, c2d_yz_8x8x8[27][1]+1, c2d_yz_8x8x8[27][2]),
                                      offset8x8x8(c2d_yz_8x8x8[28][0]+1, c2d_yz_8x8x8[28][1]+1, c2d_yz_8x8x8[28][2]),
                                      offset8x8x8(c2d_yz_8x8x8[29][0]+1, c2d_yz_8x8x8[29][1]+1, c2d_yz_8x8x8[29][2]),
                                      offset8x8x8(c2d_yz_8x8x8[30][0]+1, c2d_yz_8x8x8[30][1]+1, c2d_yz_8x8x8[30][2]),
                                      offset8x8x8(c2d_yz_8x8x8[31][0]+1, c2d_yz_8x8x8[31][1]+1, c2d_yz_8x8x8[31][2]),
                                      offset8x8x8(c2d_yz_8x8x8[32][0]+1, c2d_yz_8x8x8[32][1]+1, c2d_yz_8x8x8[32][2]),
                                      offset8x8x8(c2d_yz_8x8x8[33][0]+1, c2d_yz_8x8x8[33][1]+1, c2d_yz_8x8x8[33][2]),
                                      offset8x8x8(c2d_yz_8x8x8[34][0]+1, c2d_yz_8x8x8[34][1]+1, c2d_yz_8x8x8[34][2]),
                                      offset8x8x8(c2d_yz_8x8x8[35][0]+1, c2d_yz_8x8x8[35][1]+1, c2d_yz_8x8x8[35][2]),
                                      offset8x8x8(c2d_yz_8x8x8[36][0]+1, c2d_yz_8x8x8[36][1]+1, c2d_yz_8x8x8[36][2]),
                                      offset8x8x8(c2d_yz_8x8x8[37][0]+1, c2d_yz_8x8x8[37][1]+1, c2d_yz_8x8x8[37][2]),
                                      offset8x8x8(c2d_yz_8x8x8[38][0]+1, c2d_yz_8x8x8[38][1]+1, c2d_yz_8x8x8[38][2]),
                                      offset8x8x8(c2d_yz_8x8x8[39][0]+1, c2d_yz_8x8x8[39][1]+1, c2d_yz_8x8x8[39][2]),
                                      offset8x8x8(c2d_yz_8x8x8[40][0]+1, c2d_yz_8x8x8[40][1]+1, c2d_yz_8x8x8[40][2]),
                                      offset8x8x8(c2d_yz_8x8x8[41][0]+1, c2d_yz_8x8x8[41][1]+1, c2d_yz_8x8x8[41][2]),
                                      offset8x8x8(c2d_yz_8x8x8[42][0]+1, c2d_yz_8x8x8[42][1]+1, c2d_yz_8x8x8[42][2]),
                                      offset8x8x8(c2d_yz_8x8x8[43][0]+1, c2d_yz_8x8x8[43][1]+1, c2d_yz_8x8x8[43][2]),
                                      offset8x8x8(c2d_yz_8x8x8[44][0]+1, c2d_yz_8x8x8[44][1]+1, c2d_yz_8x8x8[44][2])};
  return offset[i];
}

MGARDX_EXEC int Coeff2D_LL_Offset_8x8x8(SIZE i) {
  static constexpr int offset[135] = {offset8x8x8(c2d_xy_8x8x8[0][0], c2d_xy_8x8x8[0][1]-1, c2d_xy_8x8x8[0][2]-1), // XY
                                      offset8x8x8(c2d_xy_8x8x8[1][0], c2d_xy_8x8x8[1][1]-1, c2d_xy_8x8x8[1][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[2][0], c2d_xy_8x8x8[2][1]-1, c2d_xy_8x8x8[2][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[3][0], c2d_xy_8x8x8[3][1]-1, c2d_xy_8x8x8[3][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[4][0], c2d_xy_8x8x8[4][1]-1, c2d_xy_8x8x8[4][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[5][0], c2d_xy_8x8x8[5][1]-1, c2d_xy_8x8x8[5][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[6][0], c2d_xy_8x8x8[6][1]-1, c2d_xy_8x8x8[6][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[7][0], c2d_xy_8x8x8[7][1]-1, c2d_xy_8x8x8[7][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[8][0], c2d_xy_8x8x8[8][1]-1, c2d_xy_8x8x8[8][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[9][0], c2d_xy_8x8x8[9][1]-1, c2d_xy_8x8x8[9][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[10][0], c2d_xy_8x8x8[10][1]-1, c2d_xy_8x8x8[10][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[11][0], c2d_xy_8x8x8[11][1]-1, c2d_xy_8x8x8[11][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[12][0], c2d_xy_8x8x8[12][1]-1, c2d_xy_8x8x8[12][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[13][0], c2d_xy_8x8x8[13][1]-1, c2d_xy_8x8x8[13][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[14][0], c2d_xy_8x8x8[14][1]-1, c2d_xy_8x8x8[14][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[15][0], c2d_xy_8x8x8[15][1]-1, c2d_xy_8x8x8[15][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[16][0], c2d_xy_8x8x8[16][1]-1, c2d_xy_8x8x8[16][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[17][0], c2d_xy_8x8x8[17][1]-1, c2d_xy_8x8x8[17][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[18][0], c2d_xy_8x8x8[18][1]-1, c2d_xy_8x8x8[18][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[19][0], c2d_xy_8x8x8[19][1]-1, c2d_xy_8x8x8[19][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[20][0], c2d_xy_8x8x8[20][1]-1, c2d_xy_8x8x8[20][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[21][0], c2d_xy_8x8x8[21][1]-1, c2d_xy_8x8x8[21][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[22][0], c2d_xy_8x8x8[22][1]-1, c2d_xy_8x8x8[22][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[23][0], c2d_xy_8x8x8[23][1]-1, c2d_xy_8x8x8[23][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[24][0], c2d_xy_8x8x8[24][1]-1, c2d_xy_8x8x8[24][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[25][0], c2d_xy_8x8x8[25][1]-1, c2d_xy_8x8x8[25][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[26][0], c2d_xy_8x8x8[26][1]-1, c2d_xy_8x8x8[26][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[27][0], c2d_xy_8x8x8[27][1]-1, c2d_xy_8x8x8[27][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[28][0], c2d_xy_8x8x8[28][1]-1, c2d_xy_8x8x8[28][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[29][0], c2d_xy_8x8x8[29][1]-1, c2d_xy_8x8x8[29][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[30][0], c2d_xy_8x8x8[30][1]-1, c2d_xy_8x8x8[30][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[31][0], c2d_xy_8x8x8[31][1]-1, c2d_xy_8x8x8[31][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[32][0], c2d_xy_8x8x8[32][1]-1, c2d_xy_8x8x8[32][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[33][0], c2d_xy_8x8x8[33][1]-1, c2d_xy_8x8x8[33][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[34][0], c2d_xy_8x8x8[34][1]-1, c2d_xy_8x8x8[34][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[35][0], c2d_xy_8x8x8[35][1]-1, c2d_xy_8x8x8[35][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[36][0], c2d_xy_8x8x8[36][1]-1, c2d_xy_8x8x8[36][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[37][0], c2d_xy_8x8x8[37][1]-1, c2d_xy_8x8x8[37][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[38][0], c2d_xy_8x8x8[38][1]-1, c2d_xy_8x8x8[38][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[39][0], c2d_xy_8x8x8[39][1]-1, c2d_xy_8x8x8[39][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[40][0], c2d_xy_8x8x8[40][1]-1, c2d_xy_8x8x8[40][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[41][0], c2d_xy_8x8x8[41][1]-1, c2d_xy_8x8x8[41][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[42][0], c2d_xy_8x8x8[42][1]-1, c2d_xy_8x8x8[42][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[43][0], c2d_xy_8x8x8[43][1]-1, c2d_xy_8x8x8[43][2]-1),
                                      offset8x8x8(c2d_xy_8x8x8[44][0], c2d_xy_8x8x8[44][1]-1, c2d_xy_8x8x8[44][2]-1),

                                      offset8x8x8(c2d_xz_8x8x8[0][0]-1, c2d_xz_8x8x8[0][1], c2d_xz_8x8x8[0][2]-1), // XZ
                                      offset8x8x8(c2d_xz_8x8x8[1][0]-1, c2d_xz_8x8x8[1][1], c2d_xz_8x8x8[1][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[2][0]-1, c2d_xz_8x8x8[2][1], c2d_xz_8x8x8[2][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[3][0]-1, c2d_xz_8x8x8[3][1], c2d_xz_8x8x8[3][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[4][0]-1, c2d_xz_8x8x8[4][1], c2d_xz_8x8x8[4][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[5][0]-1, c2d_xz_8x8x8[5][1], c2d_xz_8x8x8[5][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[6][0]-1, c2d_xz_8x8x8[6][1], c2d_xz_8x8x8[6][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[7][0]-1, c2d_xz_8x8x8[7][1], c2d_xz_8x8x8[7][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[8][0]-1, c2d_xz_8x8x8[8][1], c2d_xz_8x8x8[8][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[9][0]-1, c2d_xz_8x8x8[9][1], c2d_xz_8x8x8[9][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[10][0]-1, c2d_xz_8x8x8[10][1], c2d_xz_8x8x8[10][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[11][0]-1, c2d_xz_8x8x8[11][1], c2d_xz_8x8x8[11][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[12][0]-1, c2d_xz_8x8x8[12][1], c2d_xz_8x8x8[12][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[13][0]-1, c2d_xz_8x8x8[13][1], c2d_xz_8x8x8[13][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[14][0]-1, c2d_xz_8x8x8[14][1], c2d_xz_8x8x8[14][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[15][0]-1, c2d_xz_8x8x8[15][1], c2d_xz_8x8x8[15][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[16][0]-1, c2d_xz_8x8x8[16][1], c2d_xz_8x8x8[16][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[17][0]-1, c2d_xz_8x8x8[17][1], c2d_xz_8x8x8[17][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[18][0]-1, c2d_xz_8x8x8[18][1], c2d_xz_8x8x8[18][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[19][0]-1, c2d_xz_8x8x8[19][1], c2d_xz_8x8x8[19][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[20][0]-1, c2d_xz_8x8x8[20][1], c2d_xz_8x8x8[20][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[21][0]-1, c2d_xz_8x8x8[21][1], c2d_xz_8x8x8[21][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[22][0]-1, c2d_xz_8x8x8[22][1], c2d_xz_8x8x8[22][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[23][0]-1, c2d_xz_8x8x8[23][1], c2d_xz_8x8x8[23][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[24][0]-1, c2d_xz_8x8x8[24][1], c2d_xz_8x8x8[24][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[25][0]-1, c2d_xz_8x8x8[25][1], c2d_xz_8x8x8[25][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[26][0]-1, c2d_xz_8x8x8[26][1], c2d_xz_8x8x8[26][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[27][0]-1, c2d_xz_8x8x8[27][1], c2d_xz_8x8x8[27][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[28][0]-1, c2d_xz_8x8x8[28][1], c2d_xz_8x8x8[28][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[29][0]-1, c2d_xz_8x8x8[29][1], c2d_xz_8x8x8[29][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[30][0]-1, c2d_xz_8x8x8[30][1], c2d_xz_8x8x8[30][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[31][0]-1, c2d_xz_8x8x8[31][1], c2d_xz_8x8x8[31][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[32][0]-1, c2d_xz_8x8x8[32][1], c2d_xz_8x8x8[32][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[33][0]-1, c2d_xz_8x8x8[33][1], c2d_xz_8x8x8[33][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[34][0]-1, c2d_xz_8x8x8[34][1], c2d_xz_8x8x8[34][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[35][0]-1, c2d_xz_8x8x8[35][1], c2d_xz_8x8x8[35][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[36][0]-1, c2d_xz_8x8x8[36][1], c2d_xz_8x8x8[36][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[37][0]-1, c2d_xz_8x8x8[37][1], c2d_xz_8x8x8[37][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[38][0]-1, c2d_xz_8x8x8[38][1], c2d_xz_8x8x8[38][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[39][0]-1, c2d_xz_8x8x8[39][1], c2d_xz_8x8x8[39][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[40][0]-1, c2d_xz_8x8x8[40][1], c2d_xz_8x8x8[40][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[41][0]-1, c2d_xz_8x8x8[41][1], c2d_xz_8x8x8[41][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[42][0]-1, c2d_xz_8x8x8[42][1], c2d_xz_8x8x8[42][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[43][0]-1, c2d_xz_8x8x8[43][1], c2d_xz_8x8x8[43][2]-1),
                                      offset8x8x8(c2d_xz_8x8x8[44][0]-1, c2d_xz_8x8x8[44][1], c2d_xz_8x8x8[44][2]-1),

                                      offset8x8x8(c2d_yz_8x8x8[0][0]-1, c2d_yz_8x8x8[0][1]-1, c2d_yz_8x8x8[0][2]), // YZ
                                      offset8x8x8(c2d_yz_8x8x8[1][0]-1, c2d_yz_8x8x8[1][1]-1, c2d_yz_8x8x8[1][2]),
                                      offset8x8x8(c2d_yz_8x8x8[2][0]-1, c2d_yz_8x8x8[2][1]-1, c2d_yz_8x8x8[2][2]),
                                      offset8x8x8(c2d_yz_8x8x8[3][0]-1, c2d_yz_8x8x8[3][1]-1, c2d_yz_8x8x8[3][2]),
                                      offset8x8x8(c2d_yz_8x8x8[4][0]-1, c2d_yz_8x8x8[4][1]-1, c2d_yz_8x8x8[4][2]),
                                      offset8x8x8(c2d_yz_8x8x8[5][0]-1, c2d_yz_8x8x8[5][1]-1, c2d_yz_8x8x8[5][2]),
                                      offset8x8x8(c2d_yz_8x8x8[6][0]-1, c2d_yz_8x8x8[6][1]-1, c2d_yz_8x8x8[6][2]),
                                      offset8x8x8(c2d_yz_8x8x8[7][0]-1, c2d_yz_8x8x8[7][1]-1, c2d_yz_8x8x8[7][2]),
                                      offset8x8x8(c2d_yz_8x8x8[8][0]-1, c2d_yz_8x8x8[8][1]-1, c2d_yz_8x8x8[8][2]),
                                      offset8x8x8(c2d_yz_8x8x8[9][0]-1, c2d_yz_8x8x8[9][1]-1, c2d_yz_8x8x8[9][2]),
                                      offset8x8x8(c2d_yz_8x8x8[10][0]-1, c2d_yz_8x8x8[10][1]-1, c2d_yz_8x8x8[10][2]),
                                      offset8x8x8(c2d_yz_8x8x8[11][0]-1, c2d_yz_8x8x8[11][1]-1, c2d_yz_8x8x8[11][2]),
                                      offset8x8x8(c2d_yz_8x8x8[12][0]-1, c2d_yz_8x8x8[12][1]-1, c2d_yz_8x8x8[12][2]),
                                      offset8x8x8(c2d_yz_8x8x8[13][0]-1, c2d_yz_8x8x8[13][1]-1, c2d_yz_8x8x8[13][2]),
                                      offset8x8x8(c2d_yz_8x8x8[14][0]-1, c2d_yz_8x8x8[14][1]-1, c2d_yz_8x8x8[14][2]),
                                      offset8x8x8(c2d_yz_8x8x8[15][0]-1, c2d_yz_8x8x8[15][1]-1, c2d_yz_8x8x8[15][2]),
                                      offset8x8x8(c2d_yz_8x8x8[16][0]-1, c2d_yz_8x8x8[16][1]-1, c2d_yz_8x8x8[16][2]),
                                      offset8x8x8(c2d_yz_8x8x8[17][0]-1, c2d_yz_8x8x8[17][1]-1, c2d_yz_8x8x8[17][2]),
                                      offset8x8x8(c2d_yz_8x8x8[18][0]-1, c2d_yz_8x8x8[18][1]-1, c2d_yz_8x8x8[18][2]),
                                      offset8x8x8(c2d_yz_8x8x8[19][0]-1, c2d_yz_8x8x8[19][1]-1, c2d_yz_8x8x8[19][2]),
                                      offset8x8x8(c2d_yz_8x8x8[20][0]-1, c2d_yz_8x8x8[20][1]-1, c2d_yz_8x8x8[20][2]),
                                      offset8x8x8(c2d_yz_8x8x8[21][0]-1, c2d_yz_8x8x8[21][1]-1, c2d_yz_8x8x8[21][2]),
                                      offset8x8x8(c2d_yz_8x8x8[22][0]-1, c2d_yz_8x8x8[22][1]-1, c2d_yz_8x8x8[22][2]),
                                      offset8x8x8(c2d_yz_8x8x8[23][0]-1, c2d_yz_8x8x8[23][1]-1, c2d_yz_8x8x8[23][2]),
                                      offset8x8x8(c2d_yz_8x8x8[24][0]-1, c2d_yz_8x8x8[24][1]-1, c2d_yz_8x8x8[24][2]),
                                      offset8x8x8(c2d_yz_8x8x8[25][0]-1, c2d_yz_8x8x8[25][1]-1, c2d_yz_8x8x8[25][2]),
                                      offset8x8x8(c2d_yz_8x8x8[26][0]-1, c2d_yz_8x8x8[26][1]-1, c2d_yz_8x8x8[26][2]),
                                      offset8x8x8(c2d_yz_8x8x8[27][0]-1, c2d_yz_8x8x8[27][1]-1, c2d_yz_8x8x8[27][2]),
                                      offset8x8x8(c2d_yz_8x8x8[28][0]-1, c2d_yz_8x8x8[28][1]-1, c2d_yz_8x8x8[28][2]),
                                      offset8x8x8(c2d_yz_8x8x8[29][0]-1, c2d_yz_8x8x8[29][1]-1, c2d_yz_8x8x8[29][2]),
                                      offset8x8x8(c2d_yz_8x8x8[30][0]-1, c2d_yz_8x8x8[30][1]-1, c2d_yz_8x8x8[30][2]),
                                      offset8x8x8(c2d_yz_8x8x8[31][0]-1, c2d_yz_8x8x8[31][1]-1, c2d_yz_8x8x8[31][2]),
                                      offset8x8x8(c2d_yz_8x8x8[32][0]-1, c2d_yz_8x8x8[32][1]-1, c2d_yz_8x8x8[32][2]),
                                      offset8x8x8(c2d_yz_8x8x8[33][0]-1, c2d_yz_8x8x8[33][1]-1, c2d_yz_8x8x8[33][2]),
                                      offset8x8x8(c2d_yz_8x8x8[34][0]-1, c2d_yz_8x8x8[34][1]-1, c2d_yz_8x8x8[34][2]),
                                      offset8x8x8(c2d_yz_8x8x8[35][0]-1, c2d_yz_8x8x8[35][1]-1, c2d_yz_8x8x8[35][2]),
                                      offset8x8x8(c2d_yz_8x8x8[36][0]-1, c2d_yz_8x8x8[36][1]-1, c2d_yz_8x8x8[36][2]),
                                      offset8x8x8(c2d_yz_8x8x8[37][0]-1, c2d_yz_8x8x8[37][1]-1, c2d_yz_8x8x8[37][2]),
                                      offset8x8x8(c2d_yz_8x8x8[38][0]-1, c2d_yz_8x8x8[38][1]-1, c2d_yz_8x8x8[38][2]),
                                      offset8x8x8(c2d_yz_8x8x8[39][0]-1, c2d_yz_8x8x8[39][1]-1, c2d_yz_8x8x8[39][2]),
                                      offset8x8x8(c2d_yz_8x8x8[40][0]-1, c2d_yz_8x8x8[40][1]-1, c2d_yz_8x8x8[40][2]),
                                      offset8x8x8(c2d_yz_8x8x8[41][0]-1, c2d_yz_8x8x8[41][1]-1, c2d_yz_8x8x8[41][2]),
                                      offset8x8x8(c2d_yz_8x8x8[42][0]-1, c2d_yz_8x8x8[42][1]-1, c2d_yz_8x8x8[42][2]),
                                      offset8x8x8(c2d_yz_8x8x8[43][0]-1, c2d_yz_8x8x8[43][1]-1, c2d_yz_8x8x8[43][2]),
                                      offset8x8x8(c2d_yz_8x8x8[44][0]-1, c2d_yz_8x8x8[44][1]-1, c2d_yz_8x8x8[44][2])};
  return offset[i];
}

MGARDX_EXEC int Coeff3D_MMM_Offset_8x8x8(SIZE i) {
  static constexpr int offset[27] = {offset8x8x8(c3d_xyz_8x8x8[0][0], c3d_xyz_8x8x8[0][1], c3d_xyz_8x8x8[0][2]), // XY
                                      offset8x8x8(c3d_xyz_8x8x8[1][0], c3d_xyz_8x8x8[1][1], c3d_xyz_8x8x8[1][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[2][0], c3d_xyz_8x8x8[2][1], c3d_xyz_8x8x8[2][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[3][0], c3d_xyz_8x8x8[3][1], c3d_xyz_8x8x8[3][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[4][0], c3d_xyz_8x8x8[4][1], c3d_xyz_8x8x8[4][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[5][0], c3d_xyz_8x8x8[5][1], c3d_xyz_8x8x8[5][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[6][0], c3d_xyz_8x8x8[6][1], c3d_xyz_8x8x8[6][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[7][0], c3d_xyz_8x8x8[7][1], c3d_xyz_8x8x8[7][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[8][0], c3d_xyz_8x8x8[8][1], c3d_xyz_8x8x8[8][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[9][0], c3d_xyz_8x8x8[9][1], c3d_xyz_8x8x8[9][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[10][0], c3d_xyz_8x8x8[10][1], c3d_xyz_8x8x8[10][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[11][0], c3d_xyz_8x8x8[11][1], c3d_xyz_8x8x8[11][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[12][0], c3d_xyz_8x8x8[12][1], c3d_xyz_8x8x8[12][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[13][0], c3d_xyz_8x8x8[13][1], c3d_xyz_8x8x8[13][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[14][0], c3d_xyz_8x8x8[14][1], c3d_xyz_8x8x8[14][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[15][0], c3d_xyz_8x8x8[15][1], c3d_xyz_8x8x8[15][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[16][0], c3d_xyz_8x8x8[16][1], c3d_xyz_8x8x8[16][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[17][0], c3d_xyz_8x8x8[17][1], c3d_xyz_8x8x8[17][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[18][0], c3d_xyz_8x8x8[18][1], c3d_xyz_8x8x8[18][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[19][0], c3d_xyz_8x8x8[19][1], c3d_xyz_8x8x8[19][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[20][0], c3d_xyz_8x8x8[20][1], c3d_xyz_8x8x8[20][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[21][0], c3d_xyz_8x8x8[21][1], c3d_xyz_8x8x8[21][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[22][0], c3d_xyz_8x8x8[22][1], c3d_xyz_8x8x8[22][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[23][0], c3d_xyz_8x8x8[23][1], c3d_xyz_8x8x8[23][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[24][0], c3d_xyz_8x8x8[24][1], c3d_xyz_8x8x8[24][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[25][0], c3d_xyz_8x8x8[25][1], c3d_xyz_8x8x8[25][2]),
                                      offset8x8x8(c3d_xyz_8x8x8[26][0], c3d_xyz_8x8x8[26][1], c3d_xyz_8x8x8[26][2])};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_LLL_Offset_8x8x8(SIZE i) {
  static constexpr int offset[27] = { offset8x8x8(c3d_xyz_8x8x8[0][0]-1, c3d_xyz_8x8x8[0][1]-1, c3d_xyz_8x8x8[0][2]-1), // XY
                                      offset8x8x8(c3d_xyz_8x8x8[1][0]-1, c3d_xyz_8x8x8[1][1]-1, c3d_xyz_8x8x8[1][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[2][0]-1, c3d_xyz_8x8x8[2][1]-1, c3d_xyz_8x8x8[2][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[3][0]-1, c3d_xyz_8x8x8[3][1]-1, c3d_xyz_8x8x8[3][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[4][0]-1, c3d_xyz_8x8x8[4][1]-1, c3d_xyz_8x8x8[4][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[5][0]-1, c3d_xyz_8x8x8[5][1]-1, c3d_xyz_8x8x8[5][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[6][0]-1, c3d_xyz_8x8x8[6][1]-1, c3d_xyz_8x8x8[6][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[7][0]-1, c3d_xyz_8x8x8[7][1]-1, c3d_xyz_8x8x8[7][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[8][0]-1, c3d_xyz_8x8x8[8][1]-1, c3d_xyz_8x8x8[8][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[9][0]-1, c3d_xyz_8x8x8[9][1]-1, c3d_xyz_8x8x8[9][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[10][0]-1, c3d_xyz_8x8x8[10][1]-1, c3d_xyz_8x8x8[10][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[11][0]-1, c3d_xyz_8x8x8[11][1]-1, c3d_xyz_8x8x8[11][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[12][0]-1, c3d_xyz_8x8x8[12][1]-1, c3d_xyz_8x8x8[12][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[13][0]-1, c3d_xyz_8x8x8[13][1]-1, c3d_xyz_8x8x8[13][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[14][0]-1, c3d_xyz_8x8x8[14][1]-1, c3d_xyz_8x8x8[14][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[15][0]-1, c3d_xyz_8x8x8[15][1]-1, c3d_xyz_8x8x8[15][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[16][0]-1, c3d_xyz_8x8x8[16][1]-1, c3d_xyz_8x8x8[16][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[17][0]-1, c3d_xyz_8x8x8[17][1]-1, c3d_xyz_8x8x8[17][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[18][0]-1, c3d_xyz_8x8x8[18][1]-1, c3d_xyz_8x8x8[18][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[19][0]-1, c3d_xyz_8x8x8[19][1]-1, c3d_xyz_8x8x8[19][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[20][0]-1, c3d_xyz_8x8x8[20][1]-1, c3d_xyz_8x8x8[20][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[21][0]-1, c3d_xyz_8x8x8[21][1]-1, c3d_xyz_8x8x8[21][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[22][0]-1, c3d_xyz_8x8x8[22][1]-1, c3d_xyz_8x8x8[22][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[23][0]-1, c3d_xyz_8x8x8[23][1]-1, c3d_xyz_8x8x8[23][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[24][0]-1, c3d_xyz_8x8x8[24][1]-1, c3d_xyz_8x8x8[24][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[25][0]-1, c3d_xyz_8x8x8[25][1]-1, c3d_xyz_8x8x8[25][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[26][0]-1, c3d_xyz_8x8x8[26][1]-1, c3d_xyz_8x8x8[26][2]-1)};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_LLR_Offset_8x8x8(SIZE i) {
  static constexpr int offset[27] = { offset8x8x8(c3d_xyz_8x8x8[0][0]-1, c3d_xyz_8x8x8[0][1]-1, c3d_xyz_8x8x8[0][2]+1), // XY
                                      offset8x8x8(c3d_xyz_8x8x8[1][0]-1, c3d_xyz_8x8x8[1][1]-1, c3d_xyz_8x8x8[1][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[2][0]-1, c3d_xyz_8x8x8[2][1]-1, c3d_xyz_8x8x8[2][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[3][0]-1, c3d_xyz_8x8x8[3][1]-1, c3d_xyz_8x8x8[3][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[4][0]-1, c3d_xyz_8x8x8[4][1]-1, c3d_xyz_8x8x8[4][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[5][0]-1, c3d_xyz_8x8x8[5][1]-1, c3d_xyz_8x8x8[5][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[6][0]-1, c3d_xyz_8x8x8[6][1]-1, c3d_xyz_8x8x8[6][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[7][0]-1, c3d_xyz_8x8x8[7][1]-1, c3d_xyz_8x8x8[7][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[8][0]-1, c3d_xyz_8x8x8[8][1]-1, c3d_xyz_8x8x8[8][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[9][0]-1, c3d_xyz_8x8x8[9][1]-1, c3d_xyz_8x8x8[9][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[10][0]-1, c3d_xyz_8x8x8[10][1]-1, c3d_xyz_8x8x8[10][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[11][0]-1, c3d_xyz_8x8x8[11][1]-1, c3d_xyz_8x8x8[11][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[12][0]-1, c3d_xyz_8x8x8[12][1]-1, c3d_xyz_8x8x8[12][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[13][0]-1, c3d_xyz_8x8x8[13][1]-1, c3d_xyz_8x8x8[13][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[14][0]-1, c3d_xyz_8x8x8[14][1]-1, c3d_xyz_8x8x8[14][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[15][0]-1, c3d_xyz_8x8x8[15][1]-1, c3d_xyz_8x8x8[15][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[16][0]-1, c3d_xyz_8x8x8[16][1]-1, c3d_xyz_8x8x8[16][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[17][0]-1, c3d_xyz_8x8x8[17][1]-1, c3d_xyz_8x8x8[17][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[18][0]-1, c3d_xyz_8x8x8[18][1]-1, c3d_xyz_8x8x8[18][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[19][0]-1, c3d_xyz_8x8x8[19][1]-1, c3d_xyz_8x8x8[19][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[20][0]-1, c3d_xyz_8x8x8[20][1]-1, c3d_xyz_8x8x8[20][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[21][0]-1, c3d_xyz_8x8x8[21][1]-1, c3d_xyz_8x8x8[21][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[22][0]-1, c3d_xyz_8x8x8[22][1]-1, c3d_xyz_8x8x8[22][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[23][0]-1, c3d_xyz_8x8x8[23][1]-1, c3d_xyz_8x8x8[23][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[24][0]-1, c3d_xyz_8x8x8[24][1]-1, c3d_xyz_8x8x8[24][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[25][0]-1, c3d_xyz_8x8x8[25][1]-1, c3d_xyz_8x8x8[25][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[26][0]-1, c3d_xyz_8x8x8[26][1]-1, c3d_xyz_8x8x8[26][2]+1)};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_LRL_Offset_8x8x8(SIZE i) {
  static constexpr int offset[27] = { offset8x8x8(c3d_xyz_8x8x8[0][0]-1, c3d_xyz_8x8x8[0][1]+1, c3d_xyz_8x8x8[0][2]-1), // XY
                                      offset8x8x8(c3d_xyz_8x8x8[1][0]-1, c3d_xyz_8x8x8[1][1]+1, c3d_xyz_8x8x8[1][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[2][0]-1, c3d_xyz_8x8x8[2][1]+1, c3d_xyz_8x8x8[2][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[3][0]-1, c3d_xyz_8x8x8[3][1]+1, c3d_xyz_8x8x8[3][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[4][0]-1, c3d_xyz_8x8x8[4][1]+1, c3d_xyz_8x8x8[4][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[5][0]-1, c3d_xyz_8x8x8[5][1]+1, c3d_xyz_8x8x8[5][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[6][0]-1, c3d_xyz_8x8x8[6][1]+1, c3d_xyz_8x8x8[6][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[7][0]-1, c3d_xyz_8x8x8[7][1]+1, c3d_xyz_8x8x8[7][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[8][0]-1, c3d_xyz_8x8x8[8][1]+1, c3d_xyz_8x8x8[8][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[9][0]-1, c3d_xyz_8x8x8[9][1]+1, c3d_xyz_8x8x8[9][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[10][0]-1, c3d_xyz_8x8x8[10][1]+1, c3d_xyz_8x8x8[10][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[11][0]-1, c3d_xyz_8x8x8[11][1]+1, c3d_xyz_8x8x8[11][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[12][0]-1, c3d_xyz_8x8x8[12][1]+1, c3d_xyz_8x8x8[12][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[13][0]-1, c3d_xyz_8x8x8[13][1]+1, c3d_xyz_8x8x8[13][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[14][0]-1, c3d_xyz_8x8x8[14][1]+1, c3d_xyz_8x8x8[14][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[15][0]-1, c3d_xyz_8x8x8[15][1]+1, c3d_xyz_8x8x8[15][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[16][0]-1, c3d_xyz_8x8x8[16][1]+1, c3d_xyz_8x8x8[16][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[17][0]-1, c3d_xyz_8x8x8[17][1]+1, c3d_xyz_8x8x8[17][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[18][0]-1, c3d_xyz_8x8x8[18][1]+1, c3d_xyz_8x8x8[18][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[19][0]-1, c3d_xyz_8x8x8[19][1]+1, c3d_xyz_8x8x8[19][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[20][0]-1, c3d_xyz_8x8x8[20][1]+1, c3d_xyz_8x8x8[20][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[21][0]-1, c3d_xyz_8x8x8[21][1]+1, c3d_xyz_8x8x8[21][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[22][0]-1, c3d_xyz_8x8x8[22][1]+1, c3d_xyz_8x8x8[22][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[23][0]-1, c3d_xyz_8x8x8[23][1]+1, c3d_xyz_8x8x8[23][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[24][0]-1, c3d_xyz_8x8x8[24][1]+1, c3d_xyz_8x8x8[24][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[25][0]-1, c3d_xyz_8x8x8[25][1]+1, c3d_xyz_8x8x8[25][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[26][0]-1, c3d_xyz_8x8x8[26][1]+1, c3d_xyz_8x8x8[26][2]-1)};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_LRR_Offset_8x8x8(SIZE i) {
  static constexpr int offset[27] = { offset8x8x8(c3d_xyz_8x8x8[0][0]-1, c3d_xyz_8x8x8[0][1]+1, c3d_xyz_8x8x8[0][2]+1), // XY
                                      offset8x8x8(c3d_xyz_8x8x8[1][0]-1, c3d_xyz_8x8x8[1][1]+1, c3d_xyz_8x8x8[1][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[2][0]-1, c3d_xyz_8x8x8[2][1]+1, c3d_xyz_8x8x8[2][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[3][0]-1, c3d_xyz_8x8x8[3][1]+1, c3d_xyz_8x8x8[3][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[4][0]-1, c3d_xyz_8x8x8[4][1]+1, c3d_xyz_8x8x8[4][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[5][0]-1, c3d_xyz_8x8x8[5][1]+1, c3d_xyz_8x8x8[5][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[6][0]-1, c3d_xyz_8x8x8[6][1]+1, c3d_xyz_8x8x8[6][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[7][0]-1, c3d_xyz_8x8x8[7][1]+1, c3d_xyz_8x8x8[7][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[8][0]-1, c3d_xyz_8x8x8[8][1]+1, c3d_xyz_8x8x8[8][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[9][0]-1, c3d_xyz_8x8x8[9][1]+1, c3d_xyz_8x8x8[9][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[10][0]-1, c3d_xyz_8x8x8[10][1]+1, c3d_xyz_8x8x8[10][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[11][0]-1, c3d_xyz_8x8x8[11][1]+1, c3d_xyz_8x8x8[11][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[12][0]-1, c3d_xyz_8x8x8[12][1]+1, c3d_xyz_8x8x8[12][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[13][0]-1, c3d_xyz_8x8x8[13][1]+1, c3d_xyz_8x8x8[13][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[14][0]-1, c3d_xyz_8x8x8[14][1]+1, c3d_xyz_8x8x8[14][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[15][0]-1, c3d_xyz_8x8x8[15][1]+1, c3d_xyz_8x8x8[15][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[16][0]-1, c3d_xyz_8x8x8[16][1]+1, c3d_xyz_8x8x8[16][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[17][0]-1, c3d_xyz_8x8x8[17][1]+1, c3d_xyz_8x8x8[17][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[18][0]-1, c3d_xyz_8x8x8[18][1]+1, c3d_xyz_8x8x8[18][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[19][0]-1, c3d_xyz_8x8x8[19][1]+1, c3d_xyz_8x8x8[19][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[20][0]-1, c3d_xyz_8x8x8[20][1]+1, c3d_xyz_8x8x8[20][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[21][0]-1, c3d_xyz_8x8x8[21][1]+1, c3d_xyz_8x8x8[21][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[22][0]-1, c3d_xyz_8x8x8[22][1]+1, c3d_xyz_8x8x8[22][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[23][0]-1, c3d_xyz_8x8x8[23][1]+1, c3d_xyz_8x8x8[23][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[24][0]-1, c3d_xyz_8x8x8[24][1]+1, c3d_xyz_8x8x8[24][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[25][0]-1, c3d_xyz_8x8x8[25][1]+1, c3d_xyz_8x8x8[25][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[26][0]-1, c3d_xyz_8x8x8[26][1]+1, c3d_xyz_8x8x8[26][2]+1)};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_RLL_Offset_8x8x8(SIZE i) {
  static constexpr int offset[27] = { offset8x8x8(c3d_xyz_8x8x8[0][0]+1, c3d_xyz_8x8x8[0][1]-1, c3d_xyz_8x8x8[0][2]-1), // XY
                                      offset8x8x8(c3d_xyz_8x8x8[1][0]+1, c3d_xyz_8x8x8[1][1]-1, c3d_xyz_8x8x8[1][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[2][0]+1, c3d_xyz_8x8x8[2][1]-1, c3d_xyz_8x8x8[2][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[3][0]+1, c3d_xyz_8x8x8[3][1]-1, c3d_xyz_8x8x8[3][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[4][0]+1, c3d_xyz_8x8x8[4][1]-1, c3d_xyz_8x8x8[4][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[5][0]+1, c3d_xyz_8x8x8[5][1]-1, c3d_xyz_8x8x8[5][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[6][0]+1, c3d_xyz_8x8x8[6][1]-1, c3d_xyz_8x8x8[6][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[7][0]+1, c3d_xyz_8x8x8[7][1]-1, c3d_xyz_8x8x8[7][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[8][0]+1, c3d_xyz_8x8x8[8][1]-1, c3d_xyz_8x8x8[8][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[9][0]+1, c3d_xyz_8x8x8[9][1]-1, c3d_xyz_8x8x8[9][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[10][0]+1, c3d_xyz_8x8x8[10][1]-1, c3d_xyz_8x8x8[10][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[11][0]+1, c3d_xyz_8x8x8[11][1]-1, c3d_xyz_8x8x8[11][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[12][0]+1, c3d_xyz_8x8x8[12][1]-1, c3d_xyz_8x8x8[12][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[13][0]+1, c3d_xyz_8x8x8[13][1]-1, c3d_xyz_8x8x8[13][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[14][0]+1, c3d_xyz_8x8x8[14][1]-1, c3d_xyz_8x8x8[14][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[15][0]+1, c3d_xyz_8x8x8[15][1]-1, c3d_xyz_8x8x8[15][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[16][0]+1, c3d_xyz_8x8x8[16][1]-1, c3d_xyz_8x8x8[16][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[17][0]+1, c3d_xyz_8x8x8[17][1]-1, c3d_xyz_8x8x8[17][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[18][0]+1, c3d_xyz_8x8x8[18][1]-1, c3d_xyz_8x8x8[18][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[19][0]+1, c3d_xyz_8x8x8[19][1]-1, c3d_xyz_8x8x8[19][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[20][0]+1, c3d_xyz_8x8x8[20][1]-1, c3d_xyz_8x8x8[20][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[21][0]+1, c3d_xyz_8x8x8[21][1]-1, c3d_xyz_8x8x8[21][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[22][0]+1, c3d_xyz_8x8x8[22][1]-1, c3d_xyz_8x8x8[22][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[23][0]+1, c3d_xyz_8x8x8[23][1]-1, c3d_xyz_8x8x8[23][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[24][0]+1, c3d_xyz_8x8x8[24][1]-1, c3d_xyz_8x8x8[24][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[25][0]+1, c3d_xyz_8x8x8[25][1]-1, c3d_xyz_8x8x8[25][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[26][0]+1, c3d_xyz_8x8x8[26][1]-1, c3d_xyz_8x8x8[26][2]-1)};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_RLR_Offset_8x8x8(SIZE i) {
  static constexpr int offset[27] = { offset8x8x8(c3d_xyz_8x8x8[0][0]+1, c3d_xyz_8x8x8[0][1]-1, c3d_xyz_8x8x8[0][2]+1), // XY
                                      offset8x8x8(c3d_xyz_8x8x8[1][0]+1, c3d_xyz_8x8x8[1][1]-1, c3d_xyz_8x8x8[1][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[2][0]+1, c3d_xyz_8x8x8[2][1]-1, c3d_xyz_8x8x8[2][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[3][0]+1, c3d_xyz_8x8x8[3][1]-1, c3d_xyz_8x8x8[3][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[4][0]+1, c3d_xyz_8x8x8[4][1]-1, c3d_xyz_8x8x8[4][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[5][0]+1, c3d_xyz_8x8x8[5][1]-1, c3d_xyz_8x8x8[5][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[6][0]+1, c3d_xyz_8x8x8[6][1]-1, c3d_xyz_8x8x8[6][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[7][0]+1, c3d_xyz_8x8x8[7][1]-1, c3d_xyz_8x8x8[7][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[8][0]+1, c3d_xyz_8x8x8[8][1]-1, c3d_xyz_8x8x8[8][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[9][0]+1, c3d_xyz_8x8x8[9][1]-1, c3d_xyz_8x8x8[9][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[10][0]+1, c3d_xyz_8x8x8[10][1]-1, c3d_xyz_8x8x8[10][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[11][0]+1, c3d_xyz_8x8x8[11][1]-1, c3d_xyz_8x8x8[11][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[12][0]+1, c3d_xyz_8x8x8[12][1]-1, c3d_xyz_8x8x8[12][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[13][0]+1, c3d_xyz_8x8x8[13][1]-1, c3d_xyz_8x8x8[13][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[14][0]+1, c3d_xyz_8x8x8[14][1]-1, c3d_xyz_8x8x8[14][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[15][0]+1, c3d_xyz_8x8x8[15][1]-1, c3d_xyz_8x8x8[15][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[16][0]+1, c3d_xyz_8x8x8[16][1]-1, c3d_xyz_8x8x8[16][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[17][0]+1, c3d_xyz_8x8x8[17][1]-1, c3d_xyz_8x8x8[17][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[18][0]+1, c3d_xyz_8x8x8[18][1]-1, c3d_xyz_8x8x8[18][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[19][0]+1, c3d_xyz_8x8x8[19][1]-1, c3d_xyz_8x8x8[19][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[20][0]+1, c3d_xyz_8x8x8[20][1]-1, c3d_xyz_8x8x8[20][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[21][0]+1, c3d_xyz_8x8x8[21][1]-1, c3d_xyz_8x8x8[21][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[22][0]+1, c3d_xyz_8x8x8[22][1]-1, c3d_xyz_8x8x8[22][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[23][0]+1, c3d_xyz_8x8x8[23][1]-1, c3d_xyz_8x8x8[23][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[24][0]+1, c3d_xyz_8x8x8[24][1]-1, c3d_xyz_8x8x8[24][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[25][0]+1, c3d_xyz_8x8x8[25][1]-1, c3d_xyz_8x8x8[25][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[26][0]+1, c3d_xyz_8x8x8[26][1]-1, c3d_xyz_8x8x8[26][2]+1)};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_RRL_Offset_8x8x8(SIZE i) {
  static constexpr int offset[27] = { offset8x8x8(c3d_xyz_8x8x8[0][0]+1, c3d_xyz_8x8x8[0][1]+1, c3d_xyz_8x8x8[0][2]-1), // XY
                                      offset8x8x8(c3d_xyz_8x8x8[1][0]+1, c3d_xyz_8x8x8[1][1]+1, c3d_xyz_8x8x8[1][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[2][0]+1, c3d_xyz_8x8x8[2][1]+1, c3d_xyz_8x8x8[2][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[3][0]+1, c3d_xyz_8x8x8[3][1]+1, c3d_xyz_8x8x8[3][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[4][0]+1, c3d_xyz_8x8x8[4][1]+1, c3d_xyz_8x8x8[4][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[5][0]+1, c3d_xyz_8x8x8[5][1]+1, c3d_xyz_8x8x8[5][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[6][0]+1, c3d_xyz_8x8x8[6][1]+1, c3d_xyz_8x8x8[6][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[7][0]+1, c3d_xyz_8x8x8[7][1]+1, c3d_xyz_8x8x8[7][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[8][0]+1, c3d_xyz_8x8x8[8][1]+1, c3d_xyz_8x8x8[8][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[9][0]+1, c3d_xyz_8x8x8[9][1]+1, c3d_xyz_8x8x8[9][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[10][0]+1, c3d_xyz_8x8x8[10][1]+1, c3d_xyz_8x8x8[10][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[11][0]+1, c3d_xyz_8x8x8[11][1]+1, c3d_xyz_8x8x8[11][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[12][0]+1, c3d_xyz_8x8x8[12][1]+1, c3d_xyz_8x8x8[12][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[13][0]+1, c3d_xyz_8x8x8[13][1]+1, c3d_xyz_8x8x8[13][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[14][0]+1, c3d_xyz_8x8x8[14][1]+1, c3d_xyz_8x8x8[14][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[15][0]+1, c3d_xyz_8x8x8[15][1]+1, c3d_xyz_8x8x8[15][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[16][0]+1, c3d_xyz_8x8x8[16][1]+1, c3d_xyz_8x8x8[16][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[17][0]+1, c3d_xyz_8x8x8[17][1]+1, c3d_xyz_8x8x8[17][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[18][0]+1, c3d_xyz_8x8x8[18][1]+1, c3d_xyz_8x8x8[18][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[19][0]+1, c3d_xyz_8x8x8[19][1]+1, c3d_xyz_8x8x8[19][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[20][0]+1, c3d_xyz_8x8x8[20][1]+1, c3d_xyz_8x8x8[20][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[21][0]+1, c3d_xyz_8x8x8[21][1]+1, c3d_xyz_8x8x8[21][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[22][0]+1, c3d_xyz_8x8x8[22][1]+1, c3d_xyz_8x8x8[22][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[23][0]+1, c3d_xyz_8x8x8[23][1]+1, c3d_xyz_8x8x8[23][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[24][0]+1, c3d_xyz_8x8x8[24][1]+1, c3d_xyz_8x8x8[24][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[25][0]+1, c3d_xyz_8x8x8[25][1]+1, c3d_xyz_8x8x8[25][2]-1),
                                      offset8x8x8(c3d_xyz_8x8x8[26][0]+1, c3d_xyz_8x8x8[26][1]+1, c3d_xyz_8x8x8[26][2]-1)};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_RRR_Offset_8x8x8(SIZE i) {
  static constexpr int offset[27] = { offset8x8x8(c3d_xyz_8x8x8[0][0]+1, c3d_xyz_8x8x8[0][1]+1, c3d_xyz_8x8x8[0][2]+1), // XY
                                      offset8x8x8(c3d_xyz_8x8x8[1][0]+1, c3d_xyz_8x8x8[1][1]+1, c3d_xyz_8x8x8[1][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[2][0]+1, c3d_xyz_8x8x8[2][1]+1, c3d_xyz_8x8x8[2][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[3][0]+1, c3d_xyz_8x8x8[3][1]+1, c3d_xyz_8x8x8[3][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[4][0]+1, c3d_xyz_8x8x8[4][1]+1, c3d_xyz_8x8x8[4][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[5][0]+1, c3d_xyz_8x8x8[5][1]+1, c3d_xyz_8x8x8[5][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[6][0]+1, c3d_xyz_8x8x8[6][1]+1, c3d_xyz_8x8x8[6][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[7][0]+1, c3d_xyz_8x8x8[7][1]+1, c3d_xyz_8x8x8[7][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[8][0]+1, c3d_xyz_8x8x8[8][1]+1, c3d_xyz_8x8x8[8][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[9][0]+1, c3d_xyz_8x8x8[9][1]+1, c3d_xyz_8x8x8[9][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[10][0]+1, c3d_xyz_8x8x8[10][1]+1, c3d_xyz_8x8x8[10][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[11][0]+1, c3d_xyz_8x8x8[11][1]+1, c3d_xyz_8x8x8[11][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[12][0]+1, c3d_xyz_8x8x8[12][1]+1, c3d_xyz_8x8x8[12][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[13][0]+1, c3d_xyz_8x8x8[13][1]+1, c3d_xyz_8x8x8[13][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[14][0]+1, c3d_xyz_8x8x8[14][1]+1, c3d_xyz_8x8x8[14][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[15][0]+1, c3d_xyz_8x8x8[15][1]+1, c3d_xyz_8x8x8[15][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[16][0]+1, c3d_xyz_8x8x8[16][1]+1, c3d_xyz_8x8x8[16][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[17][0]+1, c3d_xyz_8x8x8[17][1]+1, c3d_xyz_8x8x8[17][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[18][0]+1, c3d_xyz_8x8x8[18][1]+1, c3d_xyz_8x8x8[18][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[19][0]+1, c3d_xyz_8x8x8[19][1]+1, c3d_xyz_8x8x8[19][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[20][0]+1, c3d_xyz_8x8x8[20][1]+1, c3d_xyz_8x8x8[20][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[21][0]+1, c3d_xyz_8x8x8[21][1]+1, c3d_xyz_8x8x8[21][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[22][0]+1, c3d_xyz_8x8x8[22][1]+1, c3d_xyz_8x8x8[22][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[23][0]+1, c3d_xyz_8x8x8[23][1]+1, c3d_xyz_8x8x8[23][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[24][0]+1, c3d_xyz_8x8x8[24][1]+1, c3d_xyz_8x8x8[24][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[25][0]+1, c3d_xyz_8x8x8[25][1]+1, c3d_xyz_8x8x8[25][2]+1),
                                      offset8x8x8(c3d_xyz_8x8x8[26][0]+1, c3d_xyz_8x8x8[26][1]+1, c3d_xyz_8x8x8[26][2]+1)};

  return offset[i];
}

MGARDX_EXEC int MassTrans_X_DistCase_8x8x8(SIZE i) {
  #define DIST(Z, Y) 0, 1, 1, 2, 3

  static constexpr int offset[320] = {
    DIST(0, 0), DIST(0, 1), DIST(0, 2), DIST(0, 3), DIST(0, 4), DIST(0, 5), DIST(0, 6), DIST(0, 7),
    DIST(1, 0), DIST(1, 1), DIST(1, 2), DIST(1, 3), DIST(1, 4), DIST(1, 5), DIST(1, 6), DIST(1, 7),
    DIST(2, 0), DIST(2, 1), DIST(2, 2), DIST(2, 3), DIST(2, 4), DIST(2, 5), DIST(2, 6), DIST(2, 7),
    DIST(3, 0), DIST(3, 1), DIST(3, 2), DIST(3, 3), DIST(3, 4), DIST(3, 5), DIST(3, 6), DIST(3, 7),
    DIST(4, 0), DIST(4, 1), DIST(4, 2), DIST(4, 3), DIST(4, 4), DIST(4, 5), DIST(4, 6), DIST(4, 7),
    DIST(5, 0), DIST(5, 1), DIST(5, 2), DIST(5, 3), DIST(5, 4), DIST(5, 5), DIST(5, 6), DIST(5, 7),
    DIST(6, 0), DIST(6, 1), DIST(6, 2), DIST(6, 3), DIST(6, 4), DIST(6, 5), DIST(6, 6), DIST(6, 7),
    DIST(7, 0), DIST(7, 1), DIST(7, 2), DIST(7, 3), DIST(7, 4), DIST(7, 5), DIST(7, 6), DIST(7, 7)
  };
  #undef DIST
  return offset[i];
}


template<typename T>
MGARDX_EXEC constexpr T wa(T h1, T h2, T h3, T h4) {
  T r1 = 0.0;
  if (h1 + h2 != 0) {
    r1 = h1 / (h1 + h2);
  }
  return (h1 / 6) * r1;
}

template<typename T>
MGARDX_EXEC constexpr T wb(T h1, T h2, T h3, T h4) {
  T r1 = 0.0;
  if (h1 + h2 != 0) {
    r1 = h1 / (h1 + h2);
  }
  return ((h1 + h2) / 3) * r1 + (h2 / 6);
}

template<typename T>
MGARDX_EXEC constexpr T wc(T h1, T h2, T h3, T h4) {
  T r1 = 0.0;
  if (h1 + h2 != 0) {
    r1 = h1 / (h1 + h2);
  }
  T r4 = 0.0;
  if (h3 + h4 != 0) {
    r4 = h4 / (h3 + h4);
  }
  return ((h2 + h3) / 3) + (h2 / 6) * r1 + (h3 / 6) * r4;
}

template<typename T>
MGARDX_EXEC constexpr T wd(T h1, T h2, T h3, T h4) {
  T r4 = 0.0;
  if (h3 + h4 != 0) {
    r4 = h4 / (h3 + h4);
  }
  return ((h3 + h4) / 3) * r4 + (h3 / 6);
}

template<typename T>
MGARDX_EXEC constexpr T we(T h1, T h2, T h3, T h4) {
  T r4 = 0.0;
  if (h3 + h4 != 0) {
    r4 = h4 / (h3 + h4);
  }
  return (h4 / 6) * r4;
}

template<typename T>
MGARDX_EXEC T const *MassTrans_Weights_8x8x8(SIZE i) {
  
  static constexpr T offset[5][5] = {
    {wa<T>(0.0, 0.0, 1.0, 1.0), wb<T>(0.0, 0.0, 1.0, 1.0), wc<T>(0.0, 0.0, 1.0, 1.0), wd<T>(0.0, 0.0, 1.0, 1.0), we<T>(0.0, 0.0, 1.0, 1.0)},
    {wa<T>(1.0, 1.0, 1.0, 1.0), wb<T>(1.0, 1.0, 1.0, 1.0), wc<T>(1.0, 1.0, 1.0, 1.0), wd<T>(1.0, 1.0, 1.0, 1.0), we<T>(1.0, 1.0, 1.0, 1.0)},
    {wa<T>(1.0, 1.0, 1.0, 1.0), wb<T>(1.0, 1.0, 1.0, 1.0), wc<T>(1.0, 1.0, 1.0, 1.0), wd<T>(1.0, 1.0, 1.0, 1.0), we<T>(1.0, 1.0, 1.0, 1.0)},
    {wa<T>(1.0, 1.0, 0.5, 0.5), wb<T>(1.0, 1.0, 0.5, 0.5), wc<T>(1.0, 1.0, 0.5, 0.5), wd<T>(1.0, 1.0, 0.5, 0.5), we<T>(1.0, 1.0, 0.5, 0.5)},
    {wa<T>(0.5, 0.5, 0.0, 0.0), wb<T>(0.5, 0.5, 0.0, 0.0), wc<T>(0.5, 0.5, 0.0, 0.0), wd<T>(0.5, 0.5, 0.0, 0.0), we<T>(0.5, 0.5, 0.0, 0.0)}
  };
  return offset[i];
}

MGARDX_EXEC int const *MassTrans_X_Offset_8x8x8(SIZE i) {
  static constexpr int zero_offset = 8*8*8 + 8*8*5 + 8*5*5 + 5*5*5;
  #define OFFSET1(Z, Y)                            \
  {                                                \
    zero_offset,                                   \
    zero_offset,                                   \
    zero_offset,                                   \
    offset8x8x8(Z, Y, coarse_x_8x8x8[0]+1, 8, 8),  \
    zero_offset,                                   \
    offset8x8x8(Z, Y, 0,                   5, 8),  \
    0                                              \
  },                                               \
  {                                                \
    zero_offset,                                   \
    offset8x8x8(Z, Y, coarse_x_8x8x8[1]-1, 8, 8),  \
    zero_offset,                                   \
    offset8x8x8(Z, Y, coarse_x_8x8x8[1]+1, 8, 8),  \
    zero_offset,                                   \
    offset8x8x8(Z, Y, 1,                   5, 8),  \
    1                                              \
  },                                               \
  {                                                \
    zero_offset,                                   \
    offset8x8x8(Z, Y, coarse_x_8x8x8[2]-1, 8, 8),  \
    zero_offset,                                   \
    offset8x8x8(Z, Y, coarse_x_8x8x8[2]+1, 8, 8),  \
    zero_offset,                                   \
    offset8x8x8(Z, Y, 2,                   5, 8),  \
    2                                              \
  },                                               \
  {                                                \
    zero_offset,                                   \
    offset8x8x8(Z, Y, coarse_x_8x8x8[3]-1, 8, 8),  \
    zero_offset,                                   \
    zero_offset,                                   \
    zero_offset,                                   \
    offset8x8x8(Z, Y, 3,                   5, 8),  \
    3                                              \
  },                                               \
  {                                                \
    zero_offset,                                   \
    zero_offset,                                   \
    zero_offset,                                   \
    zero_offset,                                   \
    zero_offset,                                   \
    offset8x8x8(Z, Y, 4,                   5, 8),  \
    4                                              \
  }

  #define OFFSET2(Z, Y)                            \
  {                                                \
    zero_offset,                                   \
    zero_offset,                                   \
    offset8x8x8(Z, Y, coarse_x_8x8x8[0],   8, 8),  \
    offset8x8x8(Z, Y, coarse_x_8x8x8[0]+1, 8, 8),  \
    offset8x8x8(Z, Y, coarse_x_8x8x8[0]+2, 8, 8),  \
    offset8x8x8(Z, Y, 0,                   5, 8),  \
    0                                              \
  },                                               \
  {                                                \
    offset8x8x8(Z, Y, coarse_x_8x8x8[1]-2, 8, 8),  \
    offset8x8x8(Z, Y, coarse_x_8x8x8[1]-1, 8, 8),  \
    offset8x8x8(Z, Y, coarse_x_8x8x8[1],   8, 8),  \
    offset8x8x8(Z, Y, coarse_x_8x8x8[1]+1, 8, 8),  \
    offset8x8x8(Z, Y, coarse_x_8x8x8[1]+2, 8, 8),  \
    offset8x8x8(Z, Y, 1,                   5, 8),  \
    1                                              \
  },                                               \
  {                                                \
    offset8x8x8(Z, Y, coarse_x_8x8x8[2]-2, 8, 8),  \
    offset8x8x8(Z, Y, coarse_x_8x8x8[2]-1, 8, 8),  \
    offset8x8x8(Z, Y, coarse_x_8x8x8[2],   8, 8),  \
    offset8x8x8(Z, Y, coarse_x_8x8x8[2]+1, 8, 8),  \
    offset8x8x8(Z, Y, coarse_x_8x8x8[2]+2, 8, 8),  \
    offset8x8x8(Z, Y, 2,                   5, 8),  \
    2                                              \
  },                                               \
  {                                                \
    offset8x8x8(Z, Y, coarse_x_8x8x8[3]-2, 8, 8),  \
    offset8x8x8(Z, Y, coarse_x_8x8x8[3]-1, 8, 8),  \
    offset8x8x8(Z, Y, coarse_x_8x8x8[3],   8, 8),  \
    zero_offset,                                   \
    offset8x8x8(Z, Y, coarse_x_8x8x8[3]+1, 8, 8),  \
    offset8x8x8(Z, Y, 3,                   5, 8),  \
    3                                              \
  },                                               \
  {                                                \
    offset8x8x8(Z, Y, coarse_x_8x8x8[4]-1, 8, 8),  \
    zero_offset,                                   \
    offset8x8x8(Z, Y, coarse_x_8x8x8[4],   8, 8),  \
    zero_offset,                                   \
    zero_offset,                                   \
    offset8x8x8(Z, Y, 4,                   5, 8),  \
    4                                              \
  }

  static constexpr int offset[320][7] = {
    OFFSET1(0, 0), OFFSET2(0, 1), OFFSET1(0, 2), OFFSET2(0, 3), OFFSET1(0, 4), OFFSET2(0, 5), OFFSET1(0, 6), OFFSET1(0, 7),
    OFFSET2(1, 0), OFFSET2(1, 1), OFFSET2(1, 2), OFFSET2(1, 3), OFFSET2(1, 4), OFFSET2(1, 5), OFFSET2(1, 6), OFFSET2(1, 7),
    OFFSET1(2, 0), OFFSET2(2, 1), OFFSET1(2, 2), OFFSET2(2, 3), OFFSET1(2, 4), OFFSET2(2, 5), OFFSET1(2, 6), OFFSET1(2, 7),
    OFFSET2(3, 0), OFFSET2(3, 1), OFFSET2(3, 2), OFFSET2(3, 3), OFFSET2(3, 4), OFFSET2(3, 5), OFFSET2(3, 6), OFFSET2(3, 7),
    OFFSET1(4, 0), OFFSET2(4, 1), OFFSET1(4, 2), OFFSET2(4, 3), OFFSET1(4, 4), OFFSET2(4, 5), OFFSET1(4, 6), OFFSET1(4, 7),
    OFFSET2(5, 0), OFFSET2(5, 1), OFFSET2(5, 2), OFFSET2(5, 3), OFFSET2(5, 4), OFFSET2(5, 5), OFFSET2(5, 6), OFFSET2(5, 7),
    OFFSET1(6, 0), OFFSET2(6, 1), OFFSET1(6, 2), OFFSET2(6, 3), OFFSET1(6, 4), OFFSET2(6, 5), OFFSET1(6, 6), OFFSET1(6, 7),
    OFFSET1(7, 0), OFFSET2(7, 1), OFFSET1(7, 2), OFFSET2(7, 3), OFFSET1(7, 4), OFFSET2(7, 5), OFFSET1(7, 6), OFFSET1(7, 7)
  };
  #undef OFFSET1
  #undef OFFSET2
  return offset[i];
}


MGARDX_EXEC int const *MassTrans_Y_Offset_8x8x8(SIZE i) {
  static constexpr int zero_offset = 8*8*5 + 8*5*5 + 5*5*5;
  #define OFFSET(Z, X)                               \
  {                                                  \
      zero_offset,                                   \
      zero_offset,                                   \
      offset8x8x8(Z, coarse_y_8x8x8[0],   X, 5, 8),  \
      offset8x8x8(Z, coarse_y_8x8x8[0]+1, X, 5, 8),  \
      offset8x8x8(Z, coarse_y_8x8x8[0]+2, X, 5, 8),  \
      offset8x8x8(Z, 0,                   X, 5, 5),  \
      0                                              \
    },                                               \
    {                                                \
      offset8x8x8(Z, coarse_y_8x8x8[1]-2, X, 5, 8),  \
      offset8x8x8(Z, coarse_y_8x8x8[1]-1, X, 5, 8),  \
      offset8x8x8(Z, coarse_y_8x8x8[1],   X, 5, 8),  \
      offset8x8x8(Z, coarse_y_8x8x8[1]+1, X, 5, 8),  \
      offset8x8x8(Z, coarse_y_8x8x8[1]+2, X, 5, 8),  \
      offset8x8x8(Z, 1,                   X, 5, 5),  \
      1                                              \
    },                                               \
    {                                                \
      offset8x8x8(Z, coarse_y_8x8x8[2]-2, X, 5, 8),  \
      offset8x8x8(Z, coarse_y_8x8x8[2]-1, X, 5, 8),  \
      offset8x8x8(Z, coarse_y_8x8x8[2],   X, 5, 8),  \
      offset8x8x8(Z, coarse_y_8x8x8[2]+1, X, 5, 8),  \
      offset8x8x8(Z, coarse_y_8x8x8[2]+2, X, 5, 8),  \
      offset8x8x8(Z, 2,                   X, 5, 5),  \
      2                                              \
    },                                               \
    {                                                \
      offset8x8x8(Z, coarse_y_8x8x8[3]-2, X, 5, 8),  \
      offset8x8x8(Z, coarse_y_8x8x8[3]-1, X, 5, 8),  \
      offset8x8x8(Z, coarse_y_8x8x8[3],   X, 5, 8),  \
      zero_offset,                                   \
      offset8x8x8(Z, coarse_y_8x8x8[3]+1, X, 5, 8),  \
      offset8x8x8(Z, 3,                   X, 5, 5),  \
      3                                              \
    },                                               \
    {                                                \
      offset8x8x8(Z, coarse_y_8x8x8[4]-1, X, 5, 8),  \
      zero_offset,                                   \
      offset8x8x8(Z, coarse_y_8x8x8[4],   X, 5, 8),  \
      zero_offset,                                   \
      zero_offset,                                   \
      offset8x8x8(Z, 4,                   X, 5, 5),  \
      4                                              \
    }

  static constexpr int offset[200][7] = {
    OFFSET(0, 0), OFFSET(0, 1), OFFSET(0, 2), OFFSET(0, 3), OFFSET(0, 4),
    OFFSET(1, 0), OFFSET(1, 1), OFFSET(1, 2), OFFSET(1, 3), OFFSET(1, 4),
    OFFSET(2, 0), OFFSET(2, 1), OFFSET(2, 2), OFFSET(2, 3), OFFSET(2, 4),
    OFFSET(3, 0), OFFSET(3, 1), OFFSET(3, 2), OFFSET(3, 3), OFFSET(3, 4),
    OFFSET(4, 0), OFFSET(4, 1), OFFSET(4, 2), OFFSET(4, 3), OFFSET(4, 4),
    OFFSET(5, 0), OFFSET(5, 1), OFFSET(5, 2), OFFSET(5, 3), OFFSET(5, 4),
    OFFSET(6, 0), OFFSET(6, 1), OFFSET(6, 2), OFFSET(6, 3), OFFSET(6, 4),
    OFFSET(7, 0), OFFSET(7, 1), OFFSET(7, 2), OFFSET(7, 3), OFFSET(7, 4)
  };
  #undef OFFSET
  return offset[i];
}

MGARDX_EXEC int const *MassTrans_Z_Offset_8x8x8(SIZE i) {
  static constexpr int zero_offset = 8*5*5 + 5*5*5;
  #define OFFSET(Y, X)                               \
  {                                                  \
      zero_offset,                                   \
      zero_offset,                                   \
      offset8x8x8(coarse_z_8x8x8[0],   Y, X, 5, 5),  \
      offset8x8x8(coarse_z_8x8x8[0]+1, Y, X, 5, 5),  \
      offset8x8x8(coarse_z_8x8x8[0]+2, Y, X, 5, 5),  \
      offset8x8x8(0,                   Y, X, 5, 5),  \
      0                                              \
    },                                               \
    {                                                \
      offset8x8x8(coarse_z_8x8x8[1]-2, Y, X, 5, 5),  \
      offset8x8x8(coarse_z_8x8x8[1]-1, Y, X, 5, 5),  \
      offset8x8x8(coarse_z_8x8x8[1],   Y, X, 5, 5),  \
      offset8x8x8(coarse_z_8x8x8[1]+1, Y, X, 5, 5),  \
      offset8x8x8(coarse_z_8x8x8[1]+2, Y, X, 5, 5),  \
      offset8x8x8(1,                   Y, X, 5, 5),  \
      1                                              \
    },                                               \
    {                                                \
      offset8x8x8(coarse_z_8x8x8[2]-2, Y, X, 5, 5),  \
      offset8x8x8(coarse_z_8x8x8[2]-1, Y, X, 5, 5),  \
      offset8x8x8(coarse_z_8x8x8[2],   Y, X, 5, 5),  \
      offset8x8x8(coarse_z_8x8x8[2]+1, Y, X, 5, 5),  \
      offset8x8x8(coarse_z_8x8x8[2]+2, Y, X, 5, 5),  \
      offset8x8x8(2                ,   Y, X, 5, 5),  \
      2                                              \
    },                                               \
    {                                                \
      offset8x8x8(coarse_z_8x8x8[3]-2, Y, X, 5, 5),  \
      offset8x8x8(coarse_z_8x8x8[3]-1, Y, X, 5, 5),  \
      offset8x8x8(coarse_z_8x8x8[3],   Y, X, 5, 5),  \
      zero_offset,                                   \
      offset8x8x8(coarse_z_8x8x8[3]+1, Y, X, 5, 5),  \
      offset8x8x8(3,                   Y, X, 5, 5),  \
      3                                              \
    },                                               \
    {                                                \
      offset8x8x8(coarse_z_8x8x8[4]-1, Y, X, 5, 5),  \
      zero_offset,                                   \
      offset8x8x8(coarse_z_8x8x8[4],   Y, X, 5, 5),  \
      zero_offset,                                   \
      zero_offset,                                   \
      offset8x8x8(4,                   Y, X, 5, 5),  \
      4                                              \
    }

  static constexpr int offset[125][7] = {
    OFFSET(0, 0), OFFSET(0, 1), OFFSET(0, 2), OFFSET(0, 3), OFFSET(0, 4),
    OFFSET(1, 0), OFFSET(1, 1), OFFSET(1, 2), OFFSET(1, 3), OFFSET(1, 4),
    OFFSET(2, 0), OFFSET(2, 1), OFFSET(2, 2), OFFSET(2, 3), OFFSET(2, 4),
    OFFSET(3, 0), OFFSET(3, 1), OFFSET(3, 2), OFFSET(3, 3), OFFSET(3, 4),
    OFFSET(4, 0), OFFSET(4, 1), OFFSET(4, 2), OFFSET(4, 3), OFFSET(4, 4)
  };
  #undef OFFSET
  return offset[i];
}

template <typename T>
constexpr T am_8x8x8(int index) {
  T h_dist[4] = {2.0, 2.0, 2.0, 1.0};
  T am[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  T bm[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  bm[0] = 2 * h_dist[0] / 6;
  am[0] = 0.0;
  SIZE dof = 5;
  for (int i = 1; i < dof - 1; i++) {
    T a_j = h_dist[i - 1] / 6;
    T w = a_j / bm[i - 1];
    bm[i] = 2 * (h_dist[i - 1] + h_dist[i]) / 6 - w * a_j;
    am[i] = a_j;
  }
  T a_j = h_dist[dof - 2] / 6;
  T w = a_j / bm[dof - 2];
  bm[dof - 1] = 2 * h_dist[dof - 2] / 6 - w * a_j;
  am[dof - 1] = a_j;

  am[dof] = 0.0;
  for (int i = dof; i >= 1; i--) {
    bm[i] = bm[i-1];
  }
  bm[0] = 1.0;
  return am[index] * -1;
}

template <typename T>
constexpr T bm_8x8x8(int index) {
  T h_dist[4] = {2.0, 2.0, 2.0, 1.0};
  T am[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  T bm[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  bm[0] = 2 * h_dist[0] / 6;
  am[0] = 0.0;
  SIZE dof = 5;
  for (int i = 1; i < dof - 1; i++) {
    T a_j = h_dist[i - 1] / 6;
    T w = a_j / bm[i - 1];
    bm[i] = 2 * (h_dist[i - 1] + h_dist[i]) / 6 - w * a_j;
    am[i] = a_j;
  }
  T a_j = h_dist[dof - 2] / 6;
  T w = a_j / bm[dof - 2];
  bm[dof - 1] = 2 * h_dist[dof - 2] / 6 - w * a_j;
  am[dof - 1] = a_j;

  am[dof] = 0.0;
  for (int i = dof; i >= 1; i--) {
    bm[i] = bm[i-1];
  }
  bm[0] = 1.0;
  return 1.0 / bm[index];
}

template <typename T>
constexpr T amxbm_8x8x8(int index) {
  return am_8x8x8<T>(index) * bm_8x8x8<T>(index);
}


MGARDX_EXEC int const *TriDiag_X_Offset_8x8x8(SIZE i) {
  static constexpr int offset[25][5] = {
    {offset8x8x8(0, 0, 0, 5, 5), offset8x8x8(0, 0, 1, 5, 5), offset8x8x8(0, 0, 2, 5, 5), offset8x8x8(0, 0, 3, 5, 5), offset8x8x8(0, 0, 4, 5, 5)},
    {offset8x8x8(0, 1, 0, 5, 5), offset8x8x8(0, 1, 1, 5, 5), offset8x8x8(0, 1, 2, 5, 5), offset8x8x8(0, 1, 3, 5, 5), offset8x8x8(0, 1, 4, 5, 5)},
    {offset8x8x8(0, 2, 0, 5, 5), offset8x8x8(0, 2, 1, 5, 5), offset8x8x8(0, 2, 2, 5, 5), offset8x8x8(0, 2, 3, 5, 5), offset8x8x8(0, 2, 4, 5, 5)},
    {offset8x8x8(0, 3, 0, 5, 5), offset8x8x8(0, 3, 1, 5, 5), offset8x8x8(0, 3, 2, 5, 5), offset8x8x8(0, 3, 3, 5, 5), offset8x8x8(0, 3, 4, 5, 5)},
    {offset8x8x8(0, 4, 0, 5, 5), offset8x8x8(0, 4, 1, 5, 5), offset8x8x8(0, 4, 2, 5, 5), offset8x8x8(0, 4, 3, 5, 5), offset8x8x8(0, 4, 4, 5, 5)},

    {offset8x8x8(1, 0, 0, 5, 5), offset8x8x8(1, 0, 1, 5, 5), offset8x8x8(1, 0, 2, 5, 5), offset8x8x8(1, 0, 3, 5, 5), offset8x8x8(1, 0, 4, 5, 5)},
    {offset8x8x8(1, 1, 0, 5, 5), offset8x8x8(1, 1, 1, 5, 5), offset8x8x8(1, 1, 2, 5, 5), offset8x8x8(1, 1, 3, 5, 5), offset8x8x8(1, 1, 4, 5, 5)},
    {offset8x8x8(1, 2, 0, 5, 5), offset8x8x8(1, 2, 1, 5, 5), offset8x8x8(1, 2, 2, 5, 5), offset8x8x8(1, 2, 3, 5, 5), offset8x8x8(1, 2, 4, 5, 5)},
    {offset8x8x8(1, 3, 0, 5, 5), offset8x8x8(1, 3, 1, 5, 5), offset8x8x8(1, 3, 2, 5, 5), offset8x8x8(1, 3, 3, 5, 5), offset8x8x8(1, 3, 4, 5, 5)},
    {offset8x8x8(1, 4, 0, 5, 5), offset8x8x8(1, 4, 1, 5, 5), offset8x8x8(1, 4, 2, 5, 5), offset8x8x8(1, 4, 3, 5, 5), offset8x8x8(1, 4, 4, 5, 5)},

    {offset8x8x8(2, 0, 0, 5, 5), offset8x8x8(2, 0, 1, 5, 5), offset8x8x8(2, 0, 2, 5, 5), offset8x8x8(2, 0, 3, 5, 5), offset8x8x8(2, 0, 4, 5, 5)},
    {offset8x8x8(2, 1, 0, 5, 5), offset8x8x8(2, 1, 1, 5, 5), offset8x8x8(2, 1, 2, 5, 5), offset8x8x8(2, 1, 3, 5, 5), offset8x8x8(2, 1, 4, 5, 5)},
    {offset8x8x8(2, 2, 0, 5, 5), offset8x8x8(2, 2, 1, 5, 5), offset8x8x8(2, 2, 2, 5, 5), offset8x8x8(2, 2, 3, 5, 5), offset8x8x8(2, 2, 4, 5, 5)},
    {offset8x8x8(2, 3, 0, 5, 5), offset8x8x8(2, 3, 1, 5, 5), offset8x8x8(2, 3, 2, 5, 5), offset8x8x8(2, 3, 3, 5, 5), offset8x8x8(2, 3, 4, 5, 5)},
    {offset8x8x8(2, 4, 0, 5, 5), offset8x8x8(2, 4, 1, 5, 5), offset8x8x8(2, 4, 2, 5, 5), offset8x8x8(2, 4, 3, 5, 5), offset8x8x8(2, 4, 4, 5, 5)},

    {offset8x8x8(3, 0, 0, 5, 5), offset8x8x8(3, 0, 1, 5, 5), offset8x8x8(3, 0, 2, 5, 5), offset8x8x8(3, 0, 3, 5, 5), offset8x8x8(3, 0, 4, 5, 5)},
    {offset8x8x8(3, 1, 0, 5, 5), offset8x8x8(3, 1, 1, 5, 5), offset8x8x8(3, 1, 2, 5, 5), offset8x8x8(3, 1, 3, 5, 5), offset8x8x8(3, 1, 4, 5, 5)},
    {offset8x8x8(3, 2, 0, 5, 5), offset8x8x8(3, 2, 1, 5, 5), offset8x8x8(3, 2, 2, 5, 5), offset8x8x8(3, 2, 3, 5, 5), offset8x8x8(3, 2, 4, 5, 5)},
    {offset8x8x8(3, 3, 0, 5, 5), offset8x8x8(3, 3, 1, 5, 5), offset8x8x8(3, 3, 2, 5, 5), offset8x8x8(3, 3, 3, 5, 5), offset8x8x8(3, 3, 4, 5, 5)},
    {offset8x8x8(3, 4, 0, 5, 5), offset8x8x8(3, 4, 1, 5, 5), offset8x8x8(3, 4, 2, 5, 5), offset8x8x8(3, 4, 3, 5, 5), offset8x8x8(3, 4, 4, 5, 5)},

    {offset8x8x8(4, 0, 0, 5, 5), offset8x8x8(4, 0, 1, 5, 5), offset8x8x8(4, 0, 2, 5, 5), offset8x8x8(4, 0, 3, 5, 5), offset8x8x8(4, 0, 4, 5, 5)},
    {offset8x8x8(4, 1, 0, 5, 5), offset8x8x8(4, 1, 1, 5, 5), offset8x8x8(4, 1, 2, 5, 5), offset8x8x8(4, 1, 3, 5, 5), offset8x8x8(4, 1, 4, 5, 5)},
    {offset8x8x8(4, 2, 0, 5, 5), offset8x8x8(4, 2, 1, 5, 5), offset8x8x8(4, 2, 2, 5, 5), offset8x8x8(4, 2, 3, 5, 5), offset8x8x8(4, 2, 4, 5, 5)},
    {offset8x8x8(4, 3, 0, 5, 5), offset8x8x8(4, 3, 1, 5, 5), offset8x8x8(4, 3, 2, 5, 5), offset8x8x8(4, 3, 3, 5, 5), offset8x8x8(4, 3, 4, 5, 5)},
    {offset8x8x8(4, 4, 0, 5, 5), offset8x8x8(4, 4, 1, 5, 5), offset8x8x8(4, 4, 2, 5, 5), offset8x8x8(4, 4, 3, 5, 5), offset8x8x8(4, 4, 4, 5, 5)}
  };
  return offset[i];
}

MGARDX_EXEC int const *TriDiag_Y_Offset_8x8x8(SIZE i) {
  static constexpr int offset[25][5] = {
    {offset8x8x8(0, 0, 0, 5, 5), offset8x8x8(0, 1, 0, 5, 5), offset8x8x8(0, 2, 0, 5, 5), offset8x8x8(0, 3, 0, 5, 5), offset8x8x8(0, 4, 0, 5, 5)},
    {offset8x8x8(0, 0, 1, 5, 5), offset8x8x8(0, 1, 1, 5, 5), offset8x8x8(0, 2, 1, 5, 5), offset8x8x8(0, 3, 1, 5, 5), offset8x8x8(0, 4, 1, 5, 5)},
    {offset8x8x8(0, 0, 2, 5, 5), offset8x8x8(0, 1, 2, 5, 5), offset8x8x8(0, 2, 2, 5, 5), offset8x8x8(0, 3, 2, 5, 5), offset8x8x8(0, 4, 2, 5, 5)},
    {offset8x8x8(0, 0, 3, 5, 5), offset8x8x8(0, 1, 3, 5, 5), offset8x8x8(0, 2, 3, 5, 5), offset8x8x8(0, 3, 3, 5, 5), offset8x8x8(0, 4, 3, 5, 5)},
    {offset8x8x8(0, 0, 4, 5, 5), offset8x8x8(0, 1, 4, 5, 5), offset8x8x8(0, 2, 4, 5, 5), offset8x8x8(0, 3, 4, 5, 5), offset8x8x8(0, 4, 4, 5, 5)},

    {offset8x8x8(1, 0, 0, 5, 5), offset8x8x8(1, 1, 0, 5, 5), offset8x8x8(1, 2, 0, 5, 5), offset8x8x8(1, 3, 0, 5, 5), offset8x8x8(1, 4, 0, 5, 5)},
    {offset8x8x8(1, 0, 1, 5, 5), offset8x8x8(1, 1, 1, 5, 5), offset8x8x8(1, 2, 1, 5, 5), offset8x8x8(1, 3, 1, 5, 5), offset8x8x8(1, 4, 1, 5, 5)},
    {offset8x8x8(1, 0, 2, 5, 5), offset8x8x8(1, 1, 2, 5, 5), offset8x8x8(1, 2, 2, 5, 5), offset8x8x8(1, 3, 2, 5, 5), offset8x8x8(1, 4, 2, 5, 5)},
    {offset8x8x8(1, 0, 3, 5, 5), offset8x8x8(1, 1, 3, 5, 5), offset8x8x8(1, 2, 3, 5, 5), offset8x8x8(1, 3, 3, 5, 5), offset8x8x8(1, 4, 3, 5, 5)},
    {offset8x8x8(1, 0, 4, 5, 5), offset8x8x8(1, 1, 4, 5, 5), offset8x8x8(1, 2, 4, 5, 5), offset8x8x8(1, 3, 4, 5, 5), offset8x8x8(1, 4, 4, 5, 5)},

    {offset8x8x8(2, 0, 0, 5, 5), offset8x8x8(2, 1, 0, 5, 5), offset8x8x8(2, 2, 0, 5, 5), offset8x8x8(2, 3, 0, 5, 5), offset8x8x8(2, 4, 0, 5, 5)},
    {offset8x8x8(2, 0, 1, 5, 5), offset8x8x8(2, 1, 1, 5, 5), offset8x8x8(2, 2, 1, 5, 5), offset8x8x8(2, 3, 1, 5, 5), offset8x8x8(2, 4, 1, 5, 5)},
    {offset8x8x8(2, 0, 2, 5, 5), offset8x8x8(2, 1, 2, 5, 5), offset8x8x8(2, 2, 2, 5, 5), offset8x8x8(2, 3, 2, 5, 5), offset8x8x8(2, 4, 2, 5, 5)},
    {offset8x8x8(2, 0, 3, 5, 5), offset8x8x8(2, 1, 3, 5, 5), offset8x8x8(2, 2, 3, 5, 5), offset8x8x8(2, 3, 3, 5, 5), offset8x8x8(2, 4, 3, 5, 5)},
    {offset8x8x8(2, 0, 4, 5, 5), offset8x8x8(2, 1, 4, 5, 5), offset8x8x8(2, 2, 4, 5, 5), offset8x8x8(2, 3, 4, 5, 5), offset8x8x8(2, 4, 4, 5, 5)},

    {offset8x8x8(3, 0, 0, 5, 5), offset8x8x8(3, 1, 0, 5, 5), offset8x8x8(3, 2, 0, 5, 5), offset8x8x8(3, 3, 0, 5, 5), offset8x8x8(3, 4, 0, 5, 5)},
    {offset8x8x8(3, 0, 1, 5, 5), offset8x8x8(3, 1, 1, 5, 5), offset8x8x8(3, 2, 1, 5, 5), offset8x8x8(3, 3, 1, 5, 5), offset8x8x8(3, 4, 1, 5, 5)},
    {offset8x8x8(3, 0, 2, 5, 5), offset8x8x8(3, 1, 2, 5, 5), offset8x8x8(3, 2, 2, 5, 5), offset8x8x8(3, 3, 2, 5, 5), offset8x8x8(3, 4, 2, 5, 5)},
    {offset8x8x8(3, 0, 3, 5, 5), offset8x8x8(3, 1, 3, 5, 5), offset8x8x8(3, 2, 3, 5, 5), offset8x8x8(3, 3, 3, 5, 5), offset8x8x8(3, 4, 3, 5, 5)},
    {offset8x8x8(3, 0, 4, 5, 5), offset8x8x8(3, 1, 4, 5, 5), offset8x8x8(3, 2, 4, 5, 5), offset8x8x8(3, 3, 4, 5, 5), offset8x8x8(3, 4, 4, 5, 5)},

    {offset8x8x8(4, 0, 0, 5, 5), offset8x8x8(4, 1, 0, 5, 5), offset8x8x8(4, 2, 0, 5, 5), offset8x8x8(4, 3, 0, 5, 5), offset8x8x8(4, 4, 0, 5, 5)},
    {offset8x8x8(4, 0, 1, 5, 5), offset8x8x8(4, 1, 1, 5, 5), offset8x8x8(4, 2, 1, 5, 5), offset8x8x8(4, 3, 1, 5, 5), offset8x8x8(4, 4, 1, 5, 5)},
    {offset8x8x8(4, 0, 2, 5, 5), offset8x8x8(4, 1, 2, 5, 5), offset8x8x8(4, 2, 2, 5, 5), offset8x8x8(4, 3, 2, 5, 5), offset8x8x8(4, 4, 2, 5, 5)},
    {offset8x8x8(4, 0, 3, 5, 5), offset8x8x8(4, 1, 3, 5, 5), offset8x8x8(4, 2, 3, 5, 5), offset8x8x8(4, 3, 3, 5, 5), offset8x8x8(4, 4, 3, 5, 5)},
    {offset8x8x8(4, 0, 4, 5, 5), offset8x8x8(4, 1, 4, 5, 5), offset8x8x8(4, 2, 4, 5, 5), offset8x8x8(4, 3, 4, 5, 5), offset8x8x8(4, 4, 4, 5, 5)}
  };
  return offset[i];
}

MGARDX_EXEC int const *TriDiag_Z_Offset_8x8x8(SIZE i) {
  static constexpr int offset[25][5] = {
    {offset8x8x8(0, 0, 0, 5, 5), offset8x8x8(1, 0, 0, 5, 5), offset8x8x8(2, 0, 0, 5, 5), offset8x8x8(3, 0, 0, 5, 5), offset8x8x8(4, 0, 0, 5, 5)},
    {offset8x8x8(0, 0, 1, 5, 5), offset8x8x8(1, 0, 1, 5, 5), offset8x8x8(2, 0, 1, 5, 5), offset8x8x8(3, 0, 1, 5, 5), offset8x8x8(4, 0, 1, 5, 5)},
    {offset8x8x8(0, 0, 2, 5, 5), offset8x8x8(1, 0, 2, 5, 5), offset8x8x8(2, 0, 2, 5, 5), offset8x8x8(3, 0, 2, 5, 5), offset8x8x8(4, 0, 2, 5, 5)},
    {offset8x8x8(0, 0, 3, 5, 5), offset8x8x8(1, 0, 3, 5, 5), offset8x8x8(2, 0, 3, 5, 5), offset8x8x8(3, 0, 3, 5, 5), offset8x8x8(4, 0, 3, 5, 5)},
    {offset8x8x8(0, 0, 4, 5, 5), offset8x8x8(1, 0, 4, 5, 5), offset8x8x8(2, 0, 4, 5, 5), offset8x8x8(3, 0, 4, 5, 5), offset8x8x8(4, 0, 4, 5, 5)},

    {offset8x8x8(0, 1, 0, 5, 5), offset8x8x8(1, 1, 0, 5, 5), offset8x8x8(2, 1, 0, 5, 5), offset8x8x8(3, 1, 0, 5, 5), offset8x8x8(4, 1, 0, 5, 5)},
    {offset8x8x8(0, 1, 1, 5, 5), offset8x8x8(1, 1, 1, 5, 5), offset8x8x8(2, 1, 1, 5, 5), offset8x8x8(3, 1, 1, 5, 5), offset8x8x8(4, 1, 1, 5, 5)},
    {offset8x8x8(0, 1, 2, 5, 5), offset8x8x8(1, 1, 2, 5, 5), offset8x8x8(2, 1, 2, 5, 5), offset8x8x8(3, 1, 2, 5, 5), offset8x8x8(4, 1, 2, 5, 5)},
    {offset8x8x8(0, 1, 3, 5, 5), offset8x8x8(1, 1, 3, 5, 5), offset8x8x8(2, 1, 3, 5, 5), offset8x8x8(3, 1, 3, 5, 5), offset8x8x8(4, 1, 3, 5, 5)},
    {offset8x8x8(0, 1, 4, 5, 5), offset8x8x8(1, 1, 4, 5, 5), offset8x8x8(2, 1, 4, 5, 5), offset8x8x8(3, 1, 4, 5, 5), offset8x8x8(4, 1, 4, 5, 5)},

    {offset8x8x8(0, 2, 0, 5, 5), offset8x8x8(1, 2, 0, 5, 5), offset8x8x8(2, 2, 0, 5, 5), offset8x8x8(3, 2, 0, 5, 5), offset8x8x8(4, 2, 0, 5, 5)},
    {offset8x8x8(0, 2, 1, 5, 5), offset8x8x8(1, 2, 1, 5, 5), offset8x8x8(2, 2, 1, 5, 5), offset8x8x8(3, 2, 1, 5, 5), offset8x8x8(4, 2, 1, 5, 5)},
    {offset8x8x8(0, 2, 2, 5, 5), offset8x8x8(1, 2, 2, 5, 5), offset8x8x8(2, 2, 2, 5, 5), offset8x8x8(3, 2, 2, 5, 5), offset8x8x8(4, 2, 2, 5, 5)},
    {offset8x8x8(0, 2, 3, 5, 5), offset8x8x8(1, 2, 3, 5, 5), offset8x8x8(2, 2, 3, 5, 5), offset8x8x8(3, 2, 3, 5, 5), offset8x8x8(4, 2, 3, 5, 5)},
    {offset8x8x8(0, 2, 4, 5, 5), offset8x8x8(1, 2, 4, 5, 5), offset8x8x8(2, 2, 4, 5, 5), offset8x8x8(3, 2, 4, 5, 5), offset8x8x8(4, 2, 4, 5, 5)},

    {offset8x8x8(0, 3, 0, 5, 5), offset8x8x8(1, 3, 0, 5, 5), offset8x8x8(2, 3, 0, 5, 5), offset8x8x8(3, 3, 0, 5, 5), offset8x8x8(4, 3, 0, 5, 5)},
    {offset8x8x8(0, 3, 1, 5, 5), offset8x8x8(1, 3, 1, 5, 5), offset8x8x8(2, 3, 1, 5, 5), offset8x8x8(3, 3, 1, 5, 5), offset8x8x8(4, 3, 1, 5, 5)},
    {offset8x8x8(0, 3, 2, 5, 5), offset8x8x8(1, 3, 2, 5, 5), offset8x8x8(2, 3, 2, 5, 5), offset8x8x8(3, 3, 2, 5, 5), offset8x8x8(4, 3, 2, 5, 5)},
    {offset8x8x8(0, 3, 3, 5, 5), offset8x8x8(1, 3, 3, 5, 5), offset8x8x8(2, 3, 3, 5, 5), offset8x8x8(3, 3, 3, 5, 5), offset8x8x8(4, 3, 3, 5, 5)},
    {offset8x8x8(0, 3, 4, 5, 5), offset8x8x8(1, 3, 4, 5, 5), offset8x8x8(2, 3, 4, 5, 5), offset8x8x8(3, 3, 4, 5, 5), offset8x8x8(4, 3, 4, 5, 5)},

    {offset8x8x8(0, 4, 0, 5, 5), offset8x8x8(1, 4, 0, 5, 5), offset8x8x8(2, 4, 0, 5, 5), offset8x8x8(3, 4, 0, 5, 5), offset8x8x8(4, 4, 0, 5, 5)},
    {offset8x8x8(0, 4, 1, 5, 5), offset8x8x8(1, 4, 1, 5, 5), offset8x8x8(2, 4, 1, 5, 5), offset8x8x8(3, 4, 1, 5, 5), offset8x8x8(4, 4, 1, 5, 5)},
    {offset8x8x8(0, 4, 2, 5, 5), offset8x8x8(1, 4, 2, 5, 5), offset8x8x8(2, 4, 2, 5, 5), offset8x8x8(3, 4, 2, 5, 5), offset8x8x8(4, 4, 2, 5, 5)},
    {offset8x8x8(0, 4, 3, 5, 5), offset8x8x8(1, 4, 3, 5, 5), offset8x8x8(2, 4, 3, 5, 5), offset8x8x8(3, 4, 3, 5, 5), offset8x8x8(4, 4, 3, 5, 5)},
    {offset8x8x8(0, 4, 4, 5, 5), offset8x8x8(1, 4, 4, 5, 5), offset8x8x8(2, 4, 4, 5, 5), offset8x8x8(3, 4, 4, 5, 5), offset8x8x8(4, 4, 4, 5, 5)}
  };
  return offset[i];
}


MGARDX_EXEC int Coarse_Offset_8x8x8(SIZE i) {
  static constexpr int offset[125] = {
    offset8x8x8(0, 0, 0, 8, 8), offset8x8x8(0, 0, 2, 8, 8), offset8x8x8(0, 0, 4, 8, 8), offset8x8x8(0, 0, 6, 8, 8), offset8x8x8(0, 0, 7, 8, 8),
    offset8x8x8(0, 2, 0, 8, 8), offset8x8x8(0, 2, 2, 8, 8), offset8x8x8(0, 2, 4, 8, 8), offset8x8x8(0, 2, 6, 8, 8), offset8x8x8(0, 2, 7, 8, 8),
    offset8x8x8(0, 4, 0, 8, 8), offset8x8x8(0, 4, 2, 8, 8), offset8x8x8(0, 4, 4, 8, 8), offset8x8x8(0, 4, 6, 8, 8), offset8x8x8(0, 4, 7, 8, 8),
    offset8x8x8(0, 6, 0, 8, 8), offset8x8x8(0, 6, 2, 8, 8), offset8x8x8(0, 6, 4, 8, 8), offset8x8x8(0, 6, 6, 8, 8), offset8x8x8(0, 6, 7, 8, 8),
    offset8x8x8(0, 7, 0, 8, 8), offset8x8x8(0, 7, 2, 8, 8), offset8x8x8(0, 7, 4, 8, 8), offset8x8x8(0, 7, 6, 8, 8), offset8x8x8(0, 7, 7, 8, 8),

    offset8x8x8(2, 0, 0, 8, 8), offset8x8x8(2, 0, 2, 8, 8), offset8x8x8(2, 0, 4, 8, 8), offset8x8x8(2, 0, 6, 8, 8), offset8x8x8(2, 0, 7, 8, 8),
    offset8x8x8(2, 2, 0, 8, 8), offset8x8x8(2, 2, 2, 8, 8), offset8x8x8(2, 2, 4, 8, 8), offset8x8x8(2, 2, 6, 8, 8), offset8x8x8(2, 2, 7, 8, 8),
    offset8x8x8(2, 4, 0, 8, 8), offset8x8x8(2, 4, 2, 8, 8), offset8x8x8(2, 4, 4, 8, 8), offset8x8x8(2, 4, 6, 8, 8), offset8x8x8(2, 4, 7, 8, 8),
    offset8x8x8(2, 6, 0, 8, 8), offset8x8x8(2, 6, 2, 8, 8), offset8x8x8(2, 6, 4, 8, 8), offset8x8x8(2, 6, 6, 8, 8), offset8x8x8(2, 6, 7, 8, 8),
    offset8x8x8(2, 7, 0, 8, 8), offset8x8x8(2, 7, 2, 8, 8), offset8x8x8(2, 7, 4, 8, 8), offset8x8x8(2, 7, 6, 8, 8), offset8x8x8(2, 7, 7, 8, 8),

    offset8x8x8(4, 0, 0, 8, 8), offset8x8x8(4, 0, 2, 8, 8), offset8x8x8(4, 0, 4, 8, 8), offset8x8x8(4, 0, 6, 8, 8), offset8x8x8(4, 0, 7, 8, 8),
    offset8x8x8(4, 2, 0, 8, 8), offset8x8x8(4, 2, 2, 8, 8), offset8x8x8(4, 2, 4, 8, 8), offset8x8x8(4, 2, 6, 8, 8), offset8x8x8(4, 2, 7, 8, 8),
    offset8x8x8(4, 4, 0, 8, 8), offset8x8x8(4, 4, 2, 8, 8), offset8x8x8(4, 4, 4, 8, 8), offset8x8x8(4, 4, 6, 8, 8), offset8x8x8(4, 4, 7, 8, 8),
    offset8x8x8(4, 6, 0, 8, 8), offset8x8x8(4, 6, 2, 8, 8), offset8x8x8(4, 6, 4, 8, 8), offset8x8x8(4, 6, 6, 8, 8), offset8x8x8(4, 6, 7, 8, 8),
    offset8x8x8(4, 7, 0, 8, 8), offset8x8x8(4, 7, 2, 8, 8), offset8x8x8(4, 7, 4, 8, 8), offset8x8x8(4, 7, 6, 8, 8), offset8x8x8(4, 7, 7, 8, 8),

    offset8x8x8(6, 0, 0, 8, 8), offset8x8x8(6, 0, 2, 8, 8), offset8x8x8(6, 0, 4, 8, 8), offset8x8x8(6, 0, 6, 8, 8), offset8x8x8(6, 0, 7, 8, 8),
    offset8x8x8(6, 2, 0, 8, 8), offset8x8x8(6, 2, 2, 8, 8), offset8x8x8(6, 2, 4, 8, 8), offset8x8x8(6, 2, 6, 8, 8), offset8x8x8(6, 2, 7, 8, 8),
    offset8x8x8(6, 4, 0, 8, 8), offset8x8x8(6, 4, 2, 8, 8), offset8x8x8(6, 4, 4, 8, 8), offset8x8x8(6, 4, 6, 8, 8), offset8x8x8(6, 4, 7, 8, 8),
    offset8x8x8(6, 6, 0, 8, 8), offset8x8x8(6, 6, 2, 8, 8), offset8x8x8(6, 6, 4, 8, 8), offset8x8x8(6, 6, 6, 8, 8), offset8x8x8(6, 6, 7, 8, 8),
    offset8x8x8(6, 7, 0, 8, 8), offset8x8x8(6, 7, 2, 8, 8), offset8x8x8(6, 7, 4, 8, 8), offset8x8x8(6, 7, 6, 8, 8), offset8x8x8(6, 7, 7, 8, 8),

    offset8x8x8(7, 0, 0, 8, 8), offset8x8x8(7, 0, 2, 8, 8), offset8x8x8(7, 0, 4, 8, 8), offset8x8x8(7, 0, 6, 8, 8), offset8x8x8(7, 0, 7, 8, 8),
    offset8x8x8(7, 2, 0, 8, 8), offset8x8x8(7, 2, 2, 8, 8), offset8x8x8(7, 2, 4, 8, 8), offset8x8x8(7, 2, 6, 8, 8), offset8x8x8(7, 2, 7, 8, 8),
    offset8x8x8(7, 4, 0, 8, 8), offset8x8x8(7, 4, 2, 8, 8), offset8x8x8(7, 4, 4, 8, 8), offset8x8x8(7, 4, 6, 8, 8), offset8x8x8(7, 4, 7, 8, 8),
    offset8x8x8(7, 6, 0, 8, 8), offset8x8x8(7, 6, 2, 8, 8), offset8x8x8(7, 6, 4, 8, 8), offset8x8x8(7, 6, 6, 8, 8), offset8x8x8(7, 6, 7, 8, 8),
    offset8x8x8(7, 7, 0, 8, 8), offset8x8x8(7, 7, 2, 8, 8), offset8x8x8(7, 7, 4, 8, 8), offset8x8x8(7, 7, 6, 8, 8), offset8x8x8(7, 7, 7, 8, 8)
  };
  return offset[i];
}

MGARDX_EXEC int const * Coarse_Reorder_8x8x8(SIZE i) {
  static constexpr int offset[125][3] = {
    {0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {0, 0, 3}, {0, 0, 4},
    {0, 1, 0}, {0, 1, 1}, {0, 1, 2}, {0, 1, 3}, {0, 1, 4},
    {0, 2, 0}, {0, 2, 1}, {0, 2, 2}, {0, 2, 3}, {0, 2, 4},
    {0, 3, 0}, {0, 3, 1}, {0, 3, 2}, {0, 3, 3}, {0, 3, 4},
    {0, 4, 0}, {0, 4, 1}, {0, 4, 2}, {0, 4, 3}, {0, 4, 4},

    {1, 0, 0}, {1, 0, 1}, {1, 0, 2}, {1, 0, 3}, {1, 0, 4},
    {1, 1, 0}, {1, 1, 1}, {1, 1, 2}, {1, 1, 3}, {1, 1, 4},
    {1, 2, 0}, {1, 2, 1}, {1, 2, 2}, {1, 2, 3}, {1, 2, 4},
    {1, 3, 0}, {1, 3, 1}, {1, 3, 2}, {1, 3, 3}, {1, 3, 4},
    {1, 4, 0}, {1, 4, 1}, {1, 4, 2}, {1, 4, 3}, {1, 4, 4},

    {2, 0, 0}, {2, 0, 1}, {2, 0, 2}, {2, 0, 3}, {2, 0, 4},
    {2, 1, 0}, {2, 1, 1}, {2, 1, 2}, {2, 1, 3}, {2, 1, 4},
    {2, 2, 0}, {2, 2, 1}, {2, 2, 2}, {2, 2, 3}, {2, 2, 4},
    {2, 3, 0}, {2, 3, 1}, {2, 3, 2}, {2, 3, 3}, {2, 3, 4},
    {2, 4, 0}, {2, 4, 1}, {2, 4, 2}, {2, 4, 3}, {2, 4, 4},

    {3, 0, 0}, {3, 0, 1}, {3, 0, 2}, {3, 0, 3}, {3, 0, 4},
    {3, 1, 0}, {3, 1, 1}, {3, 1, 2}, {3, 1, 3}, {3, 1, 4},
    {3, 2, 0}, {3, 2, 1}, {3, 2, 2}, {3, 2, 3}, {3, 2, 4},
    {3, 3, 0}, {3, 3, 1}, {3, 3, 2}, {3, 3, 3}, {3, 3, 4},
    {3, 4, 0}, {3, 4, 1}, {3, 4, 2}, {3, 4, 3}, {3, 4, 4},

    {4, 0, 0}, {4, 0, 1}, {4, 0, 2}, {4, 0, 3}, {4, 0, 4},
    {4, 1, 0}, {4, 1, 1}, {4, 1, 2}, {4, 1, 3}, {4, 1, 4},
    {4, 2, 0}, {4, 2, 1}, {4, 2, 2}, {4, 2, 3}, {4, 2, 4},
    {4, 3, 0}, {4, 3, 1}, {4, 3, 2}, {4, 3, 3}, {4, 3, 4},
    {4, 4, 0}, {4, 4, 1}, {4, 4, 2}, {4, 4, 3}, {4, 4, 4}
  };
  return offset[i];
}


MGARDX_EXEC int Coeff_Offset_8x8x8(SIZE i) {
  static constexpr int offset[387] = {
    offset8x8x8(0, 0, 1, 8, 8), offset8x8x8(0, 0, 3, 8, 8), offset8x8x8(0, 0, 5, 8, 8), 
    offset8x8x8(0, 1, 0, 8, 8), offset8x8x8(0, 1, 1, 8, 8), offset8x8x8(0, 1, 2, 8, 8), offset8x8x8(0, 1, 3, 8, 8), offset8x8x8(0, 1, 4, 8, 8), offset8x8x8(0, 1, 5, 8, 8), offset8x8x8(0, 1, 6, 8, 8), offset8x8x8(0, 1, 7, 8, 8),
    offset8x8x8(0, 2, 1, 8, 8), offset8x8x8(0, 2, 3, 8, 8), offset8x8x8(0, 2, 5, 8, 8), 
    offset8x8x8(0, 3, 0, 8, 8), offset8x8x8(0, 3, 1, 8, 8), offset8x8x8(0, 3, 2, 8, 8), offset8x8x8(0, 3, 3, 8, 8), offset8x8x8(0, 3, 4, 8, 8), offset8x8x8(0, 3, 5, 8, 8), offset8x8x8(0, 3, 6, 8, 8), offset8x8x8(0, 3, 7, 8, 8),
    offset8x8x8(0, 4, 1, 8, 8), offset8x8x8(0, 4, 3, 8, 8), offset8x8x8(0, 4, 5, 8, 8), 
    offset8x8x8(0, 5, 0, 8, 8), offset8x8x8(0, 5, 1, 8, 8), offset8x8x8(0, 5, 2, 8, 8), offset8x8x8(0, 5, 3, 8, 8), offset8x8x8(0, 5, 4, 8, 8), offset8x8x8(0, 5, 5, 8, 8), offset8x8x8(0, 5, 6, 8, 8), offset8x8x8(0, 5, 7, 8, 8),
    offset8x8x8(0, 6, 1, 8, 8), offset8x8x8(0, 6, 3, 8, 8), offset8x8x8(0, 6, 5, 8, 8), 
    offset8x8x8(0, 7, 1, 8, 8), offset8x8x8(0, 7, 3, 8, 8), offset8x8x8(0, 7, 5, 8, 8), 

    offset8x8x8(1, 0, 0, 8, 8), offset8x8x8(1, 0, 1, 8, 8), offset8x8x8(1, 0, 2, 8, 8), offset8x8x8(1, 0, 3, 8, 8), offset8x8x8(1, 0, 4, 8, 8), offset8x8x8(1, 0, 5, 8, 8), offset8x8x8(1, 0, 6, 8, 8), offset8x8x8(1, 0, 7, 8, 8),
    offset8x8x8(1, 1, 0, 8, 8), offset8x8x8(1, 1, 1, 8, 8), offset8x8x8(1, 1, 2, 8, 8), offset8x8x8(1, 1, 3, 8, 8), offset8x8x8(1, 1, 4, 8, 8), offset8x8x8(1, 1, 5, 8, 8), offset8x8x8(1, 1, 6, 8, 8), offset8x8x8(1, 1, 7, 8, 8),
    offset8x8x8(1, 2, 0, 8, 8), offset8x8x8(1, 2, 1, 8, 8), offset8x8x8(1, 2, 2, 8, 8), offset8x8x8(1, 2, 3, 8, 8), offset8x8x8(1, 2, 4, 8, 8), offset8x8x8(1, 2, 5, 8, 8), offset8x8x8(1, 2, 6, 8, 8), offset8x8x8(1, 2, 7, 8, 8),
    offset8x8x8(1, 3, 0, 8, 8), offset8x8x8(1, 3, 1, 8, 8), offset8x8x8(1, 3, 2, 8, 8), offset8x8x8(1, 3, 3, 8, 8), offset8x8x8(1, 3, 4, 8, 8), offset8x8x8(1, 3, 5, 8, 8), offset8x8x8(1, 3, 6, 8, 8), offset8x8x8(1, 3, 7, 8, 8),
    offset8x8x8(1, 4, 0, 8, 8), offset8x8x8(1, 4, 1, 8, 8), offset8x8x8(1, 4, 2, 8, 8), offset8x8x8(1, 4, 3, 8, 8), offset8x8x8(1, 4, 4, 8, 8), offset8x8x8(1, 4, 5, 8, 8), offset8x8x8(1, 4, 6, 8, 8), offset8x8x8(1, 4, 7, 8, 8),
    offset8x8x8(1, 5, 0, 8, 8), offset8x8x8(1, 5, 1, 8, 8), offset8x8x8(1, 5, 2, 8, 8), offset8x8x8(1, 5, 3, 8, 8), offset8x8x8(1, 5, 4, 8, 8), offset8x8x8(1, 5, 5, 8, 8), offset8x8x8(1, 5, 6, 8, 8), offset8x8x8(1, 5, 7, 8, 8),
    offset8x8x8(1, 6, 0, 8, 8), offset8x8x8(1, 6, 1, 8, 8), offset8x8x8(1, 6, 2, 8, 8), offset8x8x8(1, 6, 3, 8, 8), offset8x8x8(1, 6, 4, 8, 8), offset8x8x8(1, 6, 5, 8, 8), offset8x8x8(1, 6, 6, 8, 8), offset8x8x8(1, 6, 7, 8, 8),
    offset8x8x8(1, 7, 0, 8, 8), offset8x8x8(1, 7, 1, 8, 8), offset8x8x8(1, 7, 2, 8, 8), offset8x8x8(1, 7, 3, 8, 8), offset8x8x8(1, 7, 4, 8, 8), offset8x8x8(1, 7, 5, 8, 8), offset8x8x8(1, 7, 6, 8, 8), offset8x8x8(1, 7, 7, 8, 8),

    offset8x8x8(2, 0, 1, 8, 8), offset8x8x8(2, 0, 3, 8, 8), offset8x8x8(2, 0, 5, 8, 8), 
    offset8x8x8(2, 1, 0, 8, 8), offset8x8x8(2, 1, 1, 8, 8), offset8x8x8(2, 1, 2, 8, 8), offset8x8x8(2, 1, 3, 8, 8), offset8x8x8(2, 1, 4, 8, 8), offset8x8x8(2, 1, 5, 8, 8), offset8x8x8(2, 1, 6, 8, 8), offset8x8x8(2, 1, 7, 8, 8),
    offset8x8x8(2, 2, 1, 8, 8), offset8x8x8(2, 2, 3, 8, 8), offset8x8x8(2, 2, 5, 8, 8), 
    offset8x8x8(2, 3, 0, 8, 8), offset8x8x8(2, 3, 1, 8, 8), offset8x8x8(2, 3, 2, 8, 8), offset8x8x8(2, 3, 3, 8, 8), offset8x8x8(2, 3, 4, 8, 8), offset8x8x8(2, 3, 5, 8, 8), offset8x8x8(2, 3, 6, 8, 8), offset8x8x8(2, 3, 7, 8, 8),
    offset8x8x8(2, 4, 1, 8, 8), offset8x8x8(2, 4, 3, 8, 8), offset8x8x8(2, 4, 5, 8, 8), 
    offset8x8x8(2, 5, 0, 8, 8), offset8x8x8(2, 5, 1, 8, 8), offset8x8x8(2, 5, 2, 8, 8), offset8x8x8(2, 5, 3, 8, 8), offset8x8x8(2, 5, 4, 8, 8), offset8x8x8(2, 5, 5, 8, 8), offset8x8x8(2, 5, 6, 8, 8), offset8x8x8(2, 5, 7, 8, 8),
    offset8x8x8(2, 6, 1, 8, 8), offset8x8x8(2, 6, 3, 8, 8), offset8x8x8(2, 6, 5, 8, 8), 
    offset8x8x8(2, 7, 1, 8, 8), offset8x8x8(2, 7, 3, 8, 8), offset8x8x8(2, 7, 5, 8, 8), 

    offset8x8x8(3, 0, 0, 8, 8), offset8x8x8(3, 0, 1, 8, 8), offset8x8x8(3, 0, 2, 8, 8), offset8x8x8(3, 0, 3, 8, 8), offset8x8x8(3, 0, 4, 8, 8), offset8x8x8(3, 0, 5, 8, 8), offset8x8x8(3, 0, 6, 8, 8), offset8x8x8(3, 0, 7, 8, 8),
    offset8x8x8(3, 1, 0, 8, 8), offset8x8x8(3, 1, 1, 8, 8), offset8x8x8(3, 1, 2, 8, 8), offset8x8x8(3, 1, 3, 8, 8), offset8x8x8(3, 1, 4, 8, 8), offset8x8x8(3, 1, 5, 8, 8), offset8x8x8(3, 1, 6, 8, 8), offset8x8x8(3, 1, 7, 8, 8),
    offset8x8x8(3, 2, 0, 8, 8), offset8x8x8(3, 2, 1, 8, 8), offset8x8x8(3, 2, 2, 8, 8), offset8x8x8(3, 2, 3, 8, 8), offset8x8x8(3, 2, 4, 8, 8), offset8x8x8(3, 2, 5, 8, 8), offset8x8x8(3, 2, 6, 8, 8), offset8x8x8(3, 2, 7, 8, 8),
    offset8x8x8(3, 3, 0, 8, 8), offset8x8x8(3, 3, 1, 8, 8), offset8x8x8(3, 3, 2, 8, 8), offset8x8x8(3, 3, 3, 8, 8), offset8x8x8(3, 3, 4, 8, 8), offset8x8x8(3, 3, 5, 8, 8), offset8x8x8(3, 3, 6, 8, 8), offset8x8x8(3, 3, 7, 8, 8),
    offset8x8x8(3, 4, 0, 8, 8), offset8x8x8(3, 4, 1, 8, 8), offset8x8x8(3, 4, 2, 8, 8), offset8x8x8(3, 4, 3, 8, 8), offset8x8x8(3, 4, 4, 8, 8), offset8x8x8(3, 4, 5, 8, 8), offset8x8x8(3, 4, 6, 8, 8), offset8x8x8(3, 4, 7, 8, 8),
    offset8x8x8(3, 5, 0, 8, 8), offset8x8x8(3, 5, 1, 8, 8), offset8x8x8(3, 5, 2, 8, 8), offset8x8x8(3, 5, 3, 8, 8), offset8x8x8(3, 5, 4, 8, 8), offset8x8x8(3, 5, 5, 8, 8), offset8x8x8(3, 5, 6, 8, 8), offset8x8x8(3, 5, 7, 8, 8),
    offset8x8x8(3, 6, 0, 8, 8), offset8x8x8(3, 6, 1, 8, 8), offset8x8x8(3, 6, 2, 8, 8), offset8x8x8(3, 6, 3, 8, 8), offset8x8x8(3, 6, 4, 8, 8), offset8x8x8(3, 6, 5, 8, 8), offset8x8x8(3, 6, 6, 8, 8), offset8x8x8(3, 6, 7, 8, 8),
    offset8x8x8(3, 7, 0, 8, 8), offset8x8x8(3, 7, 1, 8, 8), offset8x8x8(3, 7, 2, 8, 8), offset8x8x8(3, 7, 3, 8, 8), offset8x8x8(3, 7, 4, 8, 8), offset8x8x8(3, 7, 5, 8, 8), offset8x8x8(3, 7, 6, 8, 8), offset8x8x8(3, 7, 7, 8, 8),

    offset8x8x8(4, 0, 1, 8, 8), offset8x8x8(4, 0, 3, 8, 8), offset8x8x8(4, 0, 5, 8, 8), 
    offset8x8x8(4, 1, 0, 8, 8), offset8x8x8(4, 1, 1, 8, 8), offset8x8x8(4, 1, 2, 8, 8), offset8x8x8(4, 1, 3, 8, 8), offset8x8x8(4, 1, 4, 8, 8), offset8x8x8(4, 1, 5, 8, 8), offset8x8x8(4, 1, 6, 8, 8), offset8x8x8(4, 1, 7, 8, 8),
    offset8x8x8(4, 2, 1, 8, 8), offset8x8x8(4, 2, 3, 8, 8), offset8x8x8(4, 2, 5, 8, 8), 
    offset8x8x8(4, 3, 0, 8, 8), offset8x8x8(4, 3, 1, 8, 8), offset8x8x8(4, 3, 2, 8, 8), offset8x8x8(4, 3, 3, 8, 8), offset8x8x8(4, 3, 4, 8, 8), offset8x8x8(4, 3, 5, 8, 8), offset8x8x8(4, 3, 6, 8, 8), offset8x8x8(4, 3, 7, 8, 8),
    offset8x8x8(4, 4, 1, 8, 8), offset8x8x8(4, 4, 3, 8, 8), offset8x8x8(4, 4, 5, 8, 8), 
    offset8x8x8(4, 5, 0, 8, 8), offset8x8x8(4, 5, 1, 8, 8), offset8x8x8(4, 5, 2, 8, 8), offset8x8x8(4, 5, 3, 8, 8), offset8x8x8(4, 5, 4, 8, 8), offset8x8x8(4, 5, 5, 8, 8), offset8x8x8(4, 5, 6, 8, 8), offset8x8x8(4, 5, 7, 8, 8),
    offset8x8x8(4, 6, 1, 8, 8), offset8x8x8(4, 6, 3, 8, 8), offset8x8x8(4, 6, 5, 8, 8), 
    offset8x8x8(4, 7, 1, 8, 8), offset8x8x8(4, 7, 3, 8, 8), offset8x8x8(4, 7, 5, 8, 8), 

    offset8x8x8(5, 0, 0, 8, 8), offset8x8x8(5, 0, 1, 8, 8), offset8x8x8(5, 0, 2, 8, 8), offset8x8x8(5, 0, 3, 8, 8), offset8x8x8(5, 0, 4, 8, 8), offset8x8x8(5, 0, 5, 8, 8), offset8x8x8(5, 0, 6, 8, 8), offset8x8x8(5, 0, 7, 8, 8),
    offset8x8x8(5, 1, 0, 8, 8), offset8x8x8(5, 1, 1, 8, 8), offset8x8x8(5, 1, 2, 8, 8), offset8x8x8(5, 1, 3, 8, 8), offset8x8x8(5, 1, 4, 8, 8), offset8x8x8(5, 1, 5, 8, 8), offset8x8x8(5, 1, 6, 8, 8), offset8x8x8(5, 1, 7, 8, 8),
    offset8x8x8(5, 2, 0, 8, 8), offset8x8x8(5, 2, 1, 8, 8), offset8x8x8(5, 2, 2, 8, 8), offset8x8x8(5, 2, 3, 8, 8), offset8x8x8(5, 2, 4, 8, 8), offset8x8x8(5, 2, 5, 8, 8), offset8x8x8(5, 2, 6, 8, 8), offset8x8x8(5, 2, 7, 8, 8),
    offset8x8x8(5, 3, 0, 8, 8), offset8x8x8(5, 3, 1, 8, 8), offset8x8x8(5, 3, 2, 8, 8), offset8x8x8(5, 3, 3, 8, 8), offset8x8x8(5, 3, 4, 8, 8), offset8x8x8(5, 3, 5, 8, 8), offset8x8x8(5, 3, 6, 8, 8), offset8x8x8(5, 3, 7, 8, 8),
    offset8x8x8(5, 4, 0, 8, 8), offset8x8x8(5, 4, 1, 8, 8), offset8x8x8(5, 4, 2, 8, 8), offset8x8x8(5, 4, 3, 8, 8), offset8x8x8(5, 4, 4, 8, 8), offset8x8x8(5, 4, 5, 8, 8), offset8x8x8(5, 4, 6, 8, 8), offset8x8x8(5, 4, 7, 8, 8),
    offset8x8x8(5, 5, 0, 8, 8), offset8x8x8(5, 5, 1, 8, 8), offset8x8x8(5, 5, 2, 8, 8), offset8x8x8(5, 5, 3, 8, 8), offset8x8x8(5, 5, 4, 8, 8), offset8x8x8(5, 5, 5, 8, 8), offset8x8x8(5, 5, 6, 8, 8), offset8x8x8(5, 5, 7, 8, 8),
    offset8x8x8(5, 6, 0, 8, 8), offset8x8x8(5, 6, 1, 8, 8), offset8x8x8(5, 6, 2, 8, 8), offset8x8x8(5, 6, 3, 8, 8), offset8x8x8(5, 6, 4, 8, 8), offset8x8x8(5, 6, 5, 8, 8), offset8x8x8(5, 6, 6, 8, 8), offset8x8x8(5, 6, 7, 8, 8),
    offset8x8x8(5, 7, 0, 8, 8), offset8x8x8(5, 7, 1, 8, 8), offset8x8x8(5, 7, 2, 8, 8), offset8x8x8(5, 7, 3, 8, 8), offset8x8x8(5, 7, 4, 8, 8), offset8x8x8(5, 7, 5, 8, 8), offset8x8x8(5, 7, 6, 8, 8), offset8x8x8(5, 7, 7, 8, 8),

    offset8x8x8(6, 0, 1, 8, 8), offset8x8x8(6, 0, 3, 8, 8), offset8x8x8(6, 0, 5, 8, 8), 
    offset8x8x8(6, 1, 0, 8, 8), offset8x8x8(6, 1, 1, 8, 8), offset8x8x8(6, 1, 2, 8, 8), offset8x8x8(6, 1, 3, 8, 8), offset8x8x8(6, 1, 4, 8, 8), offset8x8x8(6, 1, 5, 8, 8), offset8x8x8(6, 1, 6, 8, 8), offset8x8x8(6, 1, 7, 8, 8),
    offset8x8x8(6, 2, 1, 8, 8), offset8x8x8(6, 2, 3, 8, 8), offset8x8x8(6, 2, 5, 8, 8), 
    offset8x8x8(6, 3, 0, 8, 8), offset8x8x8(6, 3, 1, 8, 8), offset8x8x8(6, 3, 2, 8, 8), offset8x8x8(6, 3, 3, 8, 8), offset8x8x8(6, 3, 4, 8, 8), offset8x8x8(6, 3, 5, 8, 8), offset8x8x8(6, 3, 6, 8, 8), offset8x8x8(6, 3, 7, 8, 8),
    offset8x8x8(6, 4, 1, 8, 8), offset8x8x8(6, 4, 3, 8, 8), offset8x8x8(6, 4, 5, 8, 8), 
    offset8x8x8(6, 5, 0, 8, 8), offset8x8x8(6, 5, 1, 8, 8), offset8x8x8(6, 5, 2, 8, 8), offset8x8x8(6, 5, 3, 8, 8), offset8x8x8(6, 5, 4, 8, 8), offset8x8x8(6, 5, 5, 8, 8), offset8x8x8(6, 5, 6, 8, 8), offset8x8x8(6, 5, 7, 8, 8),
    offset8x8x8(6, 6, 1, 8, 8), offset8x8x8(6, 6, 3, 8, 8), offset8x8x8(6, 6, 5, 8, 8), 
    offset8x8x8(6, 7, 1, 8, 8), offset8x8x8(6, 7, 3, 8, 8), offset8x8x8(6, 7, 5, 8, 8), 

    offset8x8x8(7, 0, 1, 8, 8), offset8x8x8(7, 0, 3, 8, 8), offset8x8x8(7, 0, 5, 8, 8), 
    offset8x8x8(7, 1, 0, 8, 8), offset8x8x8(7, 1, 1, 8, 8), offset8x8x8(7, 1, 2, 8, 8), offset8x8x8(7, 1, 3, 8, 8), offset8x8x8(7, 1, 4, 8, 8), offset8x8x8(7, 1, 5, 8, 8), offset8x8x8(7, 1, 6, 8, 8), offset8x8x8(7, 1, 7, 8, 8),
    offset8x8x8(7, 2, 1, 8, 8), offset8x8x8(7, 2, 3, 8, 8), offset8x8x8(7, 2, 5, 8, 8), 
    offset8x8x8(7, 3, 0, 8, 8), offset8x8x8(7, 3, 1, 8, 8), offset8x8x8(7, 3, 2, 8, 8), offset8x8x8(7, 3, 3, 8, 8), offset8x8x8(7, 3, 4, 8, 8), offset8x8x8(7, 3, 5, 8, 8), offset8x8x8(7, 3, 6, 8, 8), offset8x8x8(7, 3, 7, 8, 8),
    offset8x8x8(7, 4, 1, 8, 8), offset8x8x8(7, 4, 3, 8, 8), offset8x8x8(7, 4, 5, 8, 8), 
    offset8x8x8(7, 5, 0, 8, 8), offset8x8x8(7, 5, 1, 8, 8), offset8x8x8(7, 5, 2, 8, 8), offset8x8x8(7, 5, 3, 8, 8), offset8x8x8(7, 5, 4, 8, 8), offset8x8x8(7, 5, 5, 8, 8), offset8x8x8(7, 5, 6, 8, 8), offset8x8x8(7, 5, 7, 8, 8),
    offset8x8x8(7, 6, 1, 8, 8), offset8x8x8(7, 6, 3, 8, 8), offset8x8x8(7, 6, 5, 8, 8), 
    offset8x8x8(7, 7, 1, 8, 8), offset8x8x8(7, 7, 3, 8, 8), offset8x8x8(7, 7, 5, 8, 8)
  };
  return offset[i];
}

// clang-format on
} // namespace mgard_x

#endif