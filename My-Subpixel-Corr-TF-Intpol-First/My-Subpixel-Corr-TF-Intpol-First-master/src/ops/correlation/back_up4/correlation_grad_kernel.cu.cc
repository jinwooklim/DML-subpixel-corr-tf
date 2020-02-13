#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#define ROUND_OFF 100000 // 50000

#include <stdio.h>
#include <iostream>

#include "correlation_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
  typedef Eigen::GpuDevice GPUDevice;

   __global__ void CorrelateDataBackward0(const int    nthreads,
   int          item,
   int          out_width,
   int          out_height,
   int          out_channels,
   int          max_displacement,
   int          neighborhood_grid_radius,
   int          neighborhood_grid_width,
   int          kernel_radius,
   int          stride_1,
   int          stride_2,
   int          in_width,
   int          in_height,
   int          padded_in_width,
   int          padded_in_height,
   int          in_channels,
   int          in_count_per_sample,
   int          pad_size,
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//   int          interpolated_width,
//   int          interpolated_height,
//   int          padded_interpolated_width,
//   int          padded_interpolated_height,
//   int          in_count_per_sample_b,
//   float        rate,
//   int          neighborhood_grid_radius_a,
//   int          neighborhood_grid_width_a,
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   float       *output_a_gradient,
   const float *input_b,
   const float *gradient)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // index == interpolated_width * interpolated_height * in_channels;
        int k = index % in_channels;                                     // channels
        int x = (index / in_channels) % in_width + pad_size;             // w-pos
        int y = (index / in_channels / in_width) % in_height + pad_size; // h-pos

        // Get X,Y ranges and clamp
        // round_off is a trick to enable integer division with ceil, even for
        // negative numbers
        // We use a large offset, for the inner part not to become negative.
        const int round_off    = ROUND_OFF;
        const int round_off_s1 = stride_1 * round_off;

        // We add round_off before_s1 the int division and subtract round_off after
        // it, to ensure the formula matches ceil behavior:
        int xmin = (x - 2 * kernel_radius - max_displacement + round_off_s1 - 1) / stride_1 + 1 - round_off;
        int ymin = (y - 2 * kernel_radius - max_displacement + round_off_s1 - 1) / stride_1 + 1 - round_off;
        int xmax = (x - max_displacement + round_off_s1) / stride_1 - round_off;
        int ymax = (y - max_displacement + round_off_s1) / stride_1 - round_off;

        float sum = 0;

        if ((xmax >= 0) && (ymax >= 0) && (xmin <= out_width - 1) && (ymin <= out_height - 1)) {
            xmin = max(0, xmin);
            xmax = min(out_width - 1, xmax);
            ymin = max(0, ymin);
            ymax = min(out_height - 1, ymax);

            for (int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) {
                for (int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) {
                    int s2o         = stride_2 * o/2;
                    int s2p         = stride_2 * p/2;

                    // Version # 1
                    float input_b_tmp;
                    if((o%2 == 0) && (p%2 == 0)){
                        int idx_input_b_01 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o)) * in_channels + k;
                        input_b_tmp = input_b[idx_input_b_01];
                    }
                    else if((o%2 == 1) && (p%2 == 0)) {
                        int idx_input_b_01, idx_input_b_02;
                        if(o >= 0) {
                            idx_input_b_01 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_b_02 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o + 1)) * in_channels + k;
                        }
                        else {
                            idx_input_b_01 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_b_02 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o - 1)) * in_channels + k;
                        }
                        input_b_tmp = (input_b[idx_input_b_01] + input_b[idx_input_b_02]) / 2.0;
                    }
                    else if((o%2 == 0) && (p%2 == 1)){
                        int idx_input_b_01, idx_input_b_03;
                        if(p >= 0) {
                            idx_input_b_01 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_b_03 = ((item * padded_in_height + (y + s2p + 1)) * padded_in_width + (x + s2o)) * in_channels + k;
                        }
                        else {
                            idx_input_b_01 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_b_03 = ((item * padded_in_height + (y + s2p - 1)) * padded_in_width + (x + s2o)) * in_channels + k;
                        }
                        input_b_tmp = (input_b[idx_input_b_01] + input_b[idx_input_b_03]) / 2.0;
                    }
                    else if((o%2 == 1) && (p%2 == 1)){
                        int idx_input_b_01, idx_input_b_02, idx_input_b_03, idx_input_b_04;;
                        if(p >= 0 && o >= 0) {
                            idx_input_b_01 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_b_02 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o + 1)) * in_channels + k;
                            idx_input_b_03 = ((item * padded_in_height + (y + s2p + 1)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_b_04 = ((item * padded_in_height + (y + s2p + 1)) * padded_in_width + (x + s2o + 1)) * in_channels + k;
                        }
                        else if(p >= 0 && o < 0) {
                            idx_input_b_01 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_b_02 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o - 1)) * in_channels + k;
                            idx_input_b_03 = ((item * padded_in_height + (y + s2p + 1)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_b_04 = ((item * padded_in_height + (y + s2p + 1)) * padded_in_width + (x + s2o - 1)) * in_channels + k;
                        }
                        else if(p < 0 && o >= 0) {
                            idx_input_b_01 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_b_02 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o + 1)) * in_channels + k;
                            idx_input_b_03 = ((item * padded_in_height + (y + s2p - 1)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_b_04 = ((item * padded_in_height + (y + s2p - 1)) * padded_in_width + (x + s2o + 1)) * in_channels + k;
                        }
                        else if(p < 0 && o < 0) {
                            idx_input_b_01 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_b_02 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o - 1)) * in_channels + k;
                            idx_input_b_03 = ((item * padded_in_height + (y + s2p - 1)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_b_04 = ((item * padded_in_height + (y + s2p - 1)) * padded_in_width + (x + s2o - 1)) * in_channels + k;
                        }
                        input_b_tmp = (input_b[idx_input_b_01] + input_b[idx_input_b_02] + input_b[idx_input_b_03] + input_b[idx_input_b_04]) / 4.0;
                    }

                    ////////////////////////////////////////////////////////////////////////////////////////////////////
                    ////////////////////////////////////////////////////////////////////////////////////////////////////
                    // Version # 2
//                    float input_b_tmp;
//                    if((o%2 == 0) && (p%2 == 0)){
//                        int idx_input_b_01 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o)) * in_channels + k;
//                        input_b_tmp = input_b[idx_input_b_01];
//                    }
//                    else if((o%2 == 1) && (p%2 == 0)){
//                        int idx_input_b_01, idx_input_b_02;
//                        if(o >= 0) {
//                            idx_input_b_01 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_b_02 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o - 1)) * in_channels + k;
//                        }
//                        else {
//                            idx_input_b_01 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_b_02 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o + 1)) * in_channels + k;
//                        }
//                        input_b_tmp = (input_b[idx_input_b_01] + input_b[idx_input_b_02]) / 2.0;
//                    }
//                    else if((o%2 == 0) && (p%2 == 1)){
//                        int idx_input_b_01, idx_input_b_03;
//                        if(p >= 0) {
//                            idx_input_b_01 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_b_03 = ((item * padded_in_height + (y - s2p - 1)) * padded_in_width + (x - s2o)) * in_channels + k;
//                        }
//                        else {
//                            idx_input_b_01 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_b_03 = ((item * padded_in_height + (y - s2p + 1)) * padded_in_width + (x - s2o)) * in_channels + k;
//                        }
//                        input_b_tmp = (input_b[idx_input_b_01] + input_b[idx_input_b_03]) / 2.0;
//                    }
//                    else if((o%2 == 1) && (p%2 == 1)){
//                        int idx_input_b_01, idx_input_b_02, idx_input_b_03, idx_input_b_04;;
//                        if(p >= 0 && o >= 0) {
//                            idx_input_b_01 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_b_02 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o - 1)) * in_channels + k;
//                            idx_input_b_03 = ((item * padded_in_height + (y - s2p - 1)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_b_04 = ((item * padded_in_height + (y - s2p - 1)) * padded_in_width + (x - s2o - 1)) * in_channels + k;
//                        }
//                        else if(p >= 0 && o < 0) {
//                            idx_input_b_01 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_b_02 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o + 1)) * in_channels + k;
//                            idx_input_b_03 = ((item * padded_in_height + (y - s2p - 1)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_b_04 = ((item * padded_in_height + (y - s2p - 1)) * padded_in_width + (x - s2o + 1)) * in_channels + k;
//                        }
//                        else if(p < 0 && o >= 0) {
//                            idx_input_b_01 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_b_02 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o - 1)) * in_channels + k;
//                            idx_input_b_03 = ((item * padded_in_height + (y - s2p + 1)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_b_04 = ((item * padded_in_height + (y - s2p + 1)) * padded_in_width + (x - s2o - 1)) * in_channels + k;
//                        }
//                        else if(p < 0 && o < 0) {
//                            idx_input_b_01 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_b_02 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o + 1)) * in_channels + k;
//                            idx_input_b_03 = ((item * padded_in_height + (y - s2p + 1)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_b_04 = ((item * padded_in_height + (y - s2p + 1)) * padded_in_width + (x - s2o + 1)) * in_channels + k;
//                        }
//                        input_b_tmp = (input_b[idx_input_b_01] + input_b[idx_input_b_02] + input_b[idx_input_b_03] + input_b[idx_input_b_04]) / 4.0;
//                    }



                    // Index offset for gradient in following loops:
                    int op = (p + neighborhood_grid_radius) * neighborhood_grid_width + (o + neighborhood_grid_radius);

                    for (int yg = ymin; yg <= ymax; yg++) {
                        for (int xg = xmin; xg <= xmax; xg++) {
                            int idx_gradient = ((item * out_height + yg) * out_width + xg) * out_channels + op;
                            sum += gradient[idx_gradient] * input_b_tmp;
                        }
                    }
                } // o
            } // p
        } // if
        const int sumelems    = (kernel_radius * 2 + 1) * (kernel_radius * 2 + 1) * in_channels;
        const int input_a_idx = ((y - pad_size) * in_width + (x - pad_size)) * in_channels + k;
        output_a_gradient[input_a_idx + item * in_count_per_sample] = sum / (float)sumelems;
    }
} // end of CorrelateDataBackward0





__global__ void CorrelateDataBackward1(const int    nthreads,
 int          item,
 int          out_width,
 int          out_height,
 int          out_channels,
 int          max_displacement,
 int          neighborhood_grid_radius,
 int          neighborhood_grid_width,
 int          kernel_radius,
 int          stride_1,
 int          stride_2,
 int          in_width,
 int          in_height,
 int          padded_in_width,
 int          padded_in_height,
 int          in_channels,
 int          in_count_per_sample,
 int          pad_size,
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// int          interpolated_width,
// int          interpolated_height,
// int          padded_interpolated_width,
// int          padded_interpolated_height,
// int          in_count_per_sample_b,
// float        rate,
// int          neighborhood_grid_radius_a,
// int          neighborhood_grid_width_a,
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 float       *output_b_gradient,
 const float *input_a,
 const float *gradient)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        /*
            CorrelateDataBackward1
        */
        int k = index % in_channels;                                     // channels
        int x = (index / in_channels) % in_width + pad_size;             // w-pos
        int y = (index / in_channels / in_width) % in_height + pad_size; // h-pos

        // round_off is a trick to enable integer division with ceil, even for
        // negative numbers
        // We use a large offset, for the inner part not to become negative.
        const int round_off    = ROUND_OFF;
        const int round_off_s1 = stride_1 * round_off;

        float sum = 0;

        // Height (y)
        for (int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) {
            // Width (x)
            for (int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) {
//                int s2o = stride_2 * o;
//                int s2p = stride_2 * p;
                int s2o = stride_2 * o/2;
                int s2p = stride_2 * p/2;

                // Get X,Y ranges and clamp
                // We add round_off before_s1 the int division and subtract round_off
                // after it, to ensure the formula matches ceil behavior:
                int xmin = (x - 2 * kernel_radius - max_displacement - s2o + round_off_s1 - 1) / stride_1 + 1 - round_off;
                int ymin = (y - 2 * kernel_radius - max_displacement - s2p + round_off_s1 - 1) / stride_1 + 1 - round_off;

                // Caffe, NKHW: ((n * K + k) * H + h) * W + w at point (n, k, h, w)
                // TF, NHWK: ((n * H + h) * W + w) * K + k at point (n, h, w, k)

                // Same here:
                int xmax = (x - max_displacement - s2o + round_off_s1) / stride_1 - round_off;
                int ymax = (y - max_displacement - s2p + round_off_s1) / stride_1 - round_off;

                if ((xmax >= 0) && (ymax >= 0) && (xmin <= out_width - 1) && (ymin <= out_height - 1)) {
                    xmin = max(0, xmin);
                    xmax = min(out_width - 1, xmax);

                    ymin = max(0, ymin);
                    ymax = min(out_height - 1, ymax);

                    // Get input_a data:
//                    int idx_input_a = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o)) * in_channels + k;
//                    float input_a_tmp = input_a[idx_input_a];

                    // Version # 1
                    float input_a_tmp;
                    if((o%2 == 0) && (p%2 == 0)){
                        int idx_input_a_01 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o)) * in_channels + k;
                        input_a_tmp = input_a[idx_input_a_01];
                    }
                    else if((o%2 == 1) && (p%2 == 0)) {
                        int idx_input_a_01, idx_input_a_02;
                        if(o >= 0) {
                            idx_input_a_01 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_a_02 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o + 1)) * in_channels + k;
                        }
                        else {
                            idx_input_a_01 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_a_02 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o - 1)) * in_channels + k;
                        }
                        input_a_tmp = (input_a[idx_input_a_01] + input_a[idx_input_a_02]) / 2.0;
                    }
                    else if((o%2 == 0) && (p%2 == 1)){
                        int idx_input_a_01, idx_input_a_03;
                        if(p >= 0) {
                            idx_input_a_01 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_a_03 = ((item * padded_in_height + (y + s2p + 1)) * padded_in_width + (x + s2o)) * in_channels + k;
                        }
                        else {
                            idx_input_a_01 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_a_03 = ((item * padded_in_height + (y + s2p - 1)) * padded_in_width + (x + s2o)) * in_channels + k;
                        }
                        input_a_tmp = (input_a[idx_input_a_01] + input_a[idx_input_a_03]) / 2.0;
                    }
                    else if((o%2 == 1) && (p%2 == 1)){
                        int idx_input_a_01, idx_input_a_02, idx_input_a_03, idx_input_a_04;;
                        if(p >= 0 && o >= 0) {
                            idx_input_a_01 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_a_02 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o + 1)) * in_channels + k;
                            idx_input_a_03 = ((item * padded_in_height + (y + s2p + 1)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_a_04 = ((item * padded_in_height + (y + s2p + 1)) * padded_in_width + (x + s2o + 1)) * in_channels + k;
                        }
                        else if(p >= 0 && o < 0) {
                            idx_input_a_01 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_a_02 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o - 1)) * in_channels + k;
                            idx_input_a_03 = ((item * padded_in_height + (y + s2p + 1)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_a_04 = ((item * padded_in_height + (y + s2p + 1)) * padded_in_width + (x + s2o - 1)) * in_channels + k;
                        }
                        else if(p < 0 && o >= 0) {
                            idx_input_a_01 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_a_02 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o + 1)) * in_channels + k;
                            idx_input_a_03 = ((item * padded_in_height + (y + s2p - 1)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_a_04 = ((item * padded_in_height + (y + s2p - 1)) * padded_in_width + (x + s2o + 1)) * in_channels + k;
                        }
                        else if(p < 0 && o < 0) {
                            idx_input_a_01 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_a_02 = ((item * padded_in_height + (y + s2p)) * padded_in_width + (x + s2o - 1)) * in_channels + k;
                            idx_input_a_03 = ((item * padded_in_height + (y + s2p - 1)) * padded_in_width + (x + s2o)) * in_channels + k;
                            idx_input_a_04 = ((item * padded_in_height + (y + s2p - 1)) * padded_in_width + (x + s2o - 1)) * in_channels + k;
                        }
                        input_a_tmp = (input_a[idx_input_a_01] + input_a[idx_input_a_02] + input_a[idx_input_a_03] + input_a[idx_input_a_04]) / 4.0;
                    }

                    ////////////////////////////////////////////////////////////////////////////////////////////////////
                    ////////////////////////////////////////////////////////////////////////////////////////////////////
                    // Version # 2
//                    float input_a_tmp;
//                    if((o%2 == 0) && (p%2 == 0)){
//                        int idx_input_a_01 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o)) * in_channels + k;
//                        input_a_tmp = input_a[idx_input_a_01];
//                    }
//                    else if((o%2 == 1) && (p%2 == 0)){
//                        int idx_input_a_01, idx_input_a_02;
//                        if(o >= 0) {
//                            idx_input_a_01 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_a_02 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o - 1)) * in_channels + k;
//                        }
//                        else {
//                            idx_input_a_01 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_a_02 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o + 1)) * in_channels + k;
//                        }
//                        input_a_tmp = (input_a[idx_input_a_01] + input_a[idx_input_a_02]) / 2.0;
//                    }
//                    else if((o%2 == 0) && (p%2 == 1)){
//                        int idx_input_a_01, idx_input_a_03;
//                        if(p >= 0) {
//                            idx_input_a_01 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_a_03 = ((item * padded_in_height + (y - s2p - 1)) * padded_in_width + (x - s2o)) * in_channels + k;
//                        }
//                        else {
//                            idx_input_a_01 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_a_03 = ((item * padded_in_height + (y - s2p + 1)) * padded_in_width + (x - s2o)) * in_channels + k;
//                        }
//                        input_a_tmp = (input_a[idx_input_a_01] + input_a[idx_input_a_03]) / 2.0;
//                    }
//                    else if((o%2 == 1) && (p%2 == 1)){
//                        int idx_input_a_01, idx_input_a_02, idx_input_a_03, idx_input_a_04;;
//                        if(p >= 0 && o >= 0) {
//                            idx_input_a_01 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_a_02 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o - 1)) * in_channels + k;
//                            idx_input_a_03 = ((item * padded_in_height + (y - s2p - 1)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_a_04 = ((item * padded_in_height + (y - s2p - 1)) * padded_in_width + (x - s2o - 1)) * in_channels + k;
//                        }
//                        else if(p >= 0 && o < 0) {
//                            idx_input_a_01 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_a_02 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o + 1)) * in_channels + k;
//                            idx_input_a_03 = ((item * padded_in_height + (y - s2p - 1)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_a_04 = ((item * padded_in_height + (y - s2p - 1)) * padded_in_width + (x - s2o + 1)) * in_channels + k;
//                        }
//                        else if(p < 0 && o >= 0) {
//                            idx_input_a_01 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_a_02 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o - 1)) * in_channels + k;
//                            idx_input_a_03 = ((item * padded_in_height + (y - s2p + 1)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_a_04 = ((item * padded_in_height + (y - s2p + 1)) * padded_in_width + (x - s2o - 1)) * in_channels + k;
//                        }
//                        else if(p < 0 && o < 0) {
//                            idx_input_a_01 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_a_02 = ((item * padded_in_height + (y - s2p)) * padded_in_width + (x - s2o + 1)) * in_channels + k;
//                            idx_input_a_03 = ((item * padded_in_height + (y - s2p + 1)) * padded_in_width + (x - s2o)) * in_channels + k;
//                            idx_input_a_04 = ((item * padded_in_height + (y - s2p + 1)) * padded_in_width + (x - s2o + 1)) * in_channels + k;
//                        }
//                        input_a_tmp = (input_a[idx_input_a_01] + input_a[idx_input_a_02] + input_a[idx_input_a_03] + input_a[idx_input_a_04]) / 4.0;
//                    }


                    // Index offset for gradient in following loops:
                    int op = (p + neighborhood_grid_radius) * neighborhood_grid_width + (o + neighborhood_grid_radius); // index [o,p]

                    for (int y = ymin; y <= ymax; y++) {
                        for (int x = xmin; x <= xmax; x++) {
                            int idx_gradient = ((item * out_height + y) * out_width + x) * out_channels + op;
                            sum += gradient[idx_gradient] * input_a_tmp;
                        }
                    }
                }
            }
        }
        const int sumelems    = (kernel_radius * 2 + 1) * (kernel_radius * 2 + 1) * in_channels;
        const int input_b_idx = ((y - pad_size) * in_width + (x - pad_size)) * in_channels + k;
        output_b_gradient[input_b_idx + item * in_count_per_sample] = sum / (float)sumelems;
    } // CorrelateDataBackward1
}





void CorrelationGradA(const GPUDevice& device,
  const int        batch_size,
  const int        out_width,
  const int        out_height,
  const int        out_channels,
  const int        max_displacement,
  const int        neighborhood_grid_radius,
  const int        neighborhood_grid_width,
  const int        kernel_radius,
  const int        stride_1,
  const int        stride_2,
  const int        in_width,
  const int        in_height,
  const int        padded_in_width,
  const int        padded_in_height,
  const int        in_channels,
  const int        in_count_per_sample, // h * w * ch
  const int        pad,
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  const int        interpolated_width,
//  const int        interpolated_height,
//  const int        padded_interpolated_width,
//  const int        padded_interpolated_height,
//  const int        in_count_per_sample_b,
//  const float      rate,
//  const int        neighborhood_grid_radius_a,
//  const int        neighborhood_grid_width_a,
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  const float     *input_b,
  const float     *gradient,
  float           *output_a_gradient) {
  CudaLaunchConfig config = GetCudaLaunchConfig(in_count_per_sample, device);

  for (int n = 0; n < batch_size; n++) {
    CorrelateDataBackward0 <<< config.block_count, config.thread_per_block, 0, device.stream() >>> (
      in_count_per_sample,
      n, out_width, out_height, out_channels,
      max_displacement, neighborhood_grid_radius, neighborhood_grid_width, kernel_radius,
      stride_1, stride_2,
      in_width, in_height, padded_in_width, padded_in_height, in_channels, in_count_per_sample, pad,
      ////
//      interpolated_width, interpolated_height, padded_interpolated_width, padded_interpolated_height, in_count_per_sample_b, rate, neighborhood_grid_radius_a, neighborhood_grid_width_a,
      ////
      output_a_gradient, input_b, gradient);
  }
}

void CorrelationGradB(const GPUDevice& device,
  const int        batch_size,
  const int        out_width,
  const int        out_height,
  const int        out_channels,
  const int        max_displacement,
  const int        neighborhood_grid_radius,
  const int        neighborhood_grid_width,
  const int        kernel_radius,
  const int        stride_1,
  const int        stride_2,
  const int        in_width,
  const int        in_height,
  const int        padded_in_width,
  const int        padded_in_height,
  const int        in_channels,
  const int        in_count_per_sample,
  const int        pad,
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  const int        interpolated_width,
//  const int        interpolated_height,
//  const int        padded_interpolated_width,
//  const int        padded_interpolated_height,
//  const int        in_count_per_sample_b,
//  const float      rate,
//  const int        neighborhood_grid_radius_a,
//  const int        neighborhood_grid_width_a,
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  const float     *input_a,
  const float     *gradient,
  float           *output_b_gradient) {
  CudaLaunchConfig config = GetCudaLaunchConfig(in_count_per_sample, device);

  for (int n = 0; n < batch_size; n++) {
    CorrelateDataBackward1 << < config.block_count, config.thread_per_block, 0, device.stream() >> > (
      in_count_per_sample,
      n, out_width, out_height, out_channels,
      max_displacement, neighborhood_grid_radius, neighborhood_grid_width, kernel_radius,
      stride_1, stride_2,
      in_width, in_height, padded_in_width, padded_in_height, in_channels, in_count_per_sample, pad,
      ////
//      interpolated_width, interpolated_height, padded_interpolated_width, padded_interpolated_height, in_count_per_sample_b, rate, neighborhood_grid_radius_a, neighborhood_grid_width_a,
      ////
      output_b_gradient, input_a, gradient);
  }
}
} // end namespace tensorflow

#endif  // GOOGLE_CUDA
