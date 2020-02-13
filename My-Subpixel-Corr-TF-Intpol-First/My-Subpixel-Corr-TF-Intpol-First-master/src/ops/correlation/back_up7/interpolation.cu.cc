#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <iostream>

#include "interpolation.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

__global__ void InterpolationData(
  const float *in,
  int out_widthheight,
  int          in_width,
  int          in_height,
  int          interpolated_width,
  int          interpolated_height,
  int          channels,
  float rate,
  float       *interpolated_output) {
  int xy = blockIdx.x * blockDim.x + threadIdx.x;

  // UpSampling == Bilinear method
  // if(rate >= 1.0) {
    int x  = xy % interpolated_width;
    int y  = xy / interpolated_height;
    int ch = blockIdx.y;
    int n  = blockIdx.z;

    int px = x / (int)rate;
    int py = y / (int)rate;

    float fx1 = ((float)x)/rate - (float)px;
    float fx2 = 1.0 - fx1;
    float fy1 = ((float)y)/rate - (float)py;
    float fy2 = 1.0 - fy1;

    float w1 = fx2 * fy2;
    float w2 = fx1 * fy2;
    float w3 = fx2 * fy1;
    float w4 = fx1 * fy1;

    float p1 = in[((n * in_height + py) * in_width + px) * channels + ch];
    float p2 = in[((n * in_height + py) * in_width + px + 1) * channels + ch];
    float p3 = in[((n * in_height + py + 1) * in_width + px) * channels + ch];
    float p4 = in[((n * in_height + py + 1) * in_width + px + 1) * channels + ch];
  // }

  // DownSampling == Average method
  // else {
  // }

  float value = w1*p1 + w2*p2 + w3*p3 + w4*p4;

  __syncthreads();

  interpolated_output[((n * interpolated_height + y) * interpolated_width + x) * channels + ch] = value;

}

void Interpolation(const GPUDevice& device,
         const float     *input,
         int              batch_size,
         int              input_height,
         int              input_width,
         int              input_channels,
         int              interpolated_height,
         int              interpolated_width,
         float rate,
         float           *interpolated_output) {
//  int  in_widthheight    = input_width * input_height;
  int  out_widthheight = interpolated_width * interpolated_height;
  int  threads_per_block = 16;
  dim3 totalBlocks((out_widthheight - 1) / threads_per_block + 1, input_channels, batch_size);

  cudaMemset(interpolated_output, 0, batch_size * interpolated_height * interpolated_width * input_channels * sizeof(float));

  // LAUNCH KERNEL
  InterpolationData << < totalBlocks, threads_per_block, 0, device.stream() >> > (
    input,
    out_widthheight,
    input_width,
    input_height,
    interpolated_width,
    interpolated_height,
    input_channels,
    rate,
    interpolated_output);
}
}
#endif // if GOOGLE_CUDA

