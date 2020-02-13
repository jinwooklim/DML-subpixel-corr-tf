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

//#define CUDART_NAN_F            __int_as_float(0x7fffffff)

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
   int          interpolated_width,
   int          interpolated_height,
   int          padded_interpolated_width,
   int          padded_interpolated_height,
   int          in_count_per_sample_i,
   float        rate,
   int          neighborhood_grid_radius_a,
   int          neighborhood_grid_width_a,
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   float       *output_a_gradient,
   const float *input_b,
   const float *gradient)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int k = index % in_channels;                                     // channels
        int x = (index / in_channels) % interpolated_width + pad_size;             // w-pos
        int y = (index / in_channels / interpolated_width) % interpolated_height + pad_size; // h-pos

        const int round_off    = ROUND_OFF;
        const int round_off_s1 = stride_1 * round_off;

        int xmin = (x - 2 * kernel_radius - max_displacement + round_off_s1 - 1) / stride_1 + 1 - round_off;
        int ymin = (y - 2 * kernel_radius - max_displacement + round_off_s1 - 1) / stride_1 + 1 - round_off;
        int xmax = (x - max_displacement + round_off_s1) / stride_1 - round_off;
        int ymax = (y - max_displacement + round_off_s1) / stride_1 - round_off;

        float sum = 0;

        if ((xmax >= 0) && (ymax >= 0) && (xmin <= interpolated_width - 1) && (ymin <= interpolated_height - 1)) {
            xmin = max(0, xmin);
            xmax = min(interpolated_width - 1, xmax);
            ymin = max(0, ymin);
            ymax = min(interpolated_height - 1, ymax);

            for (int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) {
                for (int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) {
                    int s2o         = stride_2 * o;
                    int s2p         = stride_2 * p;

                    int idx_input_b = ((item * padded_interpolated_height + (y + s2p)) * padded_interpolated_width + (x + s2o)) * in_channels + k;
                    float input_b_tmp = input_b[idx_input_b];

                    // Index offset for gradient in following loops:
                    int op = (p + neighborhood_grid_radius) * neighborhood_grid_width + (o + neighborhood_grid_radius);

                    for (int y = ymin; y <= ymax; y++) {
                        for (int x = xmin; x <= xmax; x++) {
                            int idx_gradient = ((item * interpolated_height + y) * interpolated_width + x) * out_channels + op;
                            sum += gradient[idx_gradient] * input_b_tmp;
                        }
                    }
                } // o
            } // p
        } // if
        const int sumelems    = (kernel_radius * 2 + 1) * (kernel_radius * 2 + 1) * in_channels;
        const int input_a_idx = ((y - pad_size) * interpolated_width + (x - pad_size)) * in_channels + k;
        output_a_gradient[input_a_idx + item * in_count_per_sample_i] = sum / (float)sumelems;
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
 int          interpolated_width,
 int          interpolated_height,
 int          padded_interpolated_width,
 int          padded_interpolated_height,
 int          in_count_per_sample_i,
 float        rate,
 int          neighborhood_grid_radius_a,
 int          neighborhood_grid_width_a,
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 float       *output_b_gradient,
 const float *input_a,
 const float *gradient)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int k = index % in_channels;                                     // channels
        int x = (index / in_channels) % interpolated_width + pad_size;             // w-pos
        int y = (index / in_channels / interpolated_width) % interpolated_height + pad_size; // h-pos

        const int round_off    = ROUND_OFF;
        const int round_off_s1 = stride_1 * round_off;

        float sum = 0;

        for (int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) {
            for (int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) {
                int s2o = stride_2 * o;
                int s2p = stride_2 * p;

                int xmin = (x - 2 * kernel_radius - max_displacement - s2o + round_off_s1 - 1) / stride_1 + 1 - round_off;
                int ymin = (y - 2 * kernel_radius - max_displacement - s2p + round_off_s1 - 1) / stride_1 + 1 - round_off;
                int xmax = (x - max_displacement - s2o + round_off_s1) / stride_1 - round_off;
                int ymax = (y - max_displacement - s2p + round_off_s1) / stride_1 - round_off;

                if ((xmax >= 0) && (ymax >= 0) && (xmin <= interpolated_width - 1) && (ymin <= interpolated_height - 1)) {
                    xmin = max(0, xmin);
                    xmax = min(interpolated_width - 1, xmax);

                    ymin = max(0, ymin);
                    ymax = min(interpolated_height - 1, ymax);

                    // Get input_a data:
                    int idx_input_a = ((item * padded_interpolated_height + (y - s2p)) * padded_interpolated_width + (x - s2o)) * in_channels + k;
                    float input_a_tmp = input_a[idx_input_a];

                    // Index offset for gradient in following loops:
                    int op = (p + neighborhood_grid_radius) * neighborhood_grid_width + (o + neighborhood_grid_radius); // index [o,p]

                    for (int y = ymin; y <= ymax; y++) {
                        for (int x = xmin; x <= xmax; x++) {
                            int idx_gradient = ((item * interpolated_height + y) * interpolated_width + x) * out_channels + op;
                            sum += gradient[idx_gradient] * input_a_tmp;
                        } // xg
                    } // yg
                } // if
//                printf("%d  %d \n", x, y);
            } // o
        } // p
        const int sumelems    = (kernel_radius * 2 + 1) * (kernel_radius * 2 + 1) * in_channels;
        const int input_b_idx = ((y - pad_size) * interpolated_width + (x - pad_size)) * in_channels + k;
        output_b_gradient[input_b_idx + item * in_count_per_sample_i] = sum / (float)sumelems;
    } // CorrelateDataBackward1
}





//__global__ void DownsampleKernel(
//    const int32 nthreads,
//    const float* input_ptr,
//    float* output_ptr,
//    const int in_width,
//    const int in_height,
//    const int out_width,
//    const int out_height,
//    const int channels,
//    const float width_scale,
//    const float height_scale,
//    const int wradius,
//    const int hradius) {
//        CUDA_1D_KERNEL_LOOP(index, nthreads) {
//            const int c = index % channels;
//            const int destx = (index / channels) % out_width;
//            const int desty = (index / channels / out_width) % out_height;
//            const int n = (index / channels / out_width) / out_height;
//
//            const float srcx = ((float)destx / (float)(out_width - 1)) * (float)(in_width - 1);
//            const float srcy = ((float)desty / (float)(out_height - 1)) * (float)(in_height - 1);
//
//            const int isrcx = round(srcx);
//            const int isrcy = round(srcy);
//
//            float accum_value = 0;
//            float accum_weight = 0;
//            float accum_nan = 0;
//
//            for (int dy = -hradius; dy <= hradius; dy++) {
//                int yoff = isrcy + dy;
//                //
//                for (int dx = -wradius; dx <= wradius; dx++) {
//                    int xoff = isrcx + dx;
//
//                    if (xoff >= 0 && yoff >= 0 && xoff < in_width && yoff < in_height) {
//                        int idx = ((n * in_height + yoff) * in_width + xoff) * channels + c;
//                        float sample = input_ptr[idx];
//                        float weight = fmaxf(0.0f, 1.0f - (fabsf((float)xoff - srcx) / width_scale))
//                                       * fmaxf(0.0f, 1.0f - (fabsf((float)yoff - srcy) / height_scale));
//                        if (sample != sample) { // isnan
//                            accum_nan += weight;
//                            sample = 0;
//                            weight = 0;
//                        }
//                        accum_value += sample * weight;
//                        accum_weight += weight;
//                    }
//                }
//            }
//
//            if (accum_nan / accum_weight > 0.5) {
//                output_ptr[index] = CUDART_NAN_F;
//            } else {
//                output_ptr[index] = accum_value / accum_weight;
//            }
//        }
//}
//
//bool Downsample(const GPUDevice& device,
//                typename TTypes<float, 4>::ConstTensor input,
//                typename TTypes<float, 4>::Tensor output
////                 const float     *input,
////                 const float     *output,
//                ) {
//    const int batch_size = output.dimension(0);
//    const int out_height = output.dimension(1);
//    const int out_width = output.dimension(2);
//    const int out_channels = output.dimension(3);
//    const int total_count = batch_size * out_height * out_width * out_channels;
//
//    const int in_height = input.dimension(1);
//    const int in_width = input.dimension(2);
//
//    const float width_scale = (float)(in_width - 1) / (float)(out_width - 1);
//    const float height_scale = (float)(in_height - 1) / (float)(out_height - 1);
//
//    const int wradius = ceil(width_scale);
//    const int hradius = ceil(height_scale);
//
//    CudaLaunchConfig config = GetCudaLaunchConfig(total_count, device);
//    DownsampleKernel<<<config.block_count, config.thread_per_block, 0,
//                        device.stream()>>>(total_count, input.data(), output.data(),
//                        in_width, in_height, out_width, out_height, out_channels,
//                        width_scale, height_scale, wradius, hradius);
//    return device.ok();
//}











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
  const int        interpolated_width,
  const int        interpolated_height,
  const int        padded_interpolated_width,
  const int        padded_interpolated_height,
  const int        in_count_per_sample_i,
  const float      rate,
  const int        neighborhood_grid_radius_a,
  const int        neighborhood_grid_width_a,
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  const float     *input_b,
  const float     *gradient,
  float           *output_a_gradient) {
CudaLaunchConfig config = GetCudaLaunchConfig(in_count_per_sample_i, device);

  for (int n = 0; n < batch_size; n++) {
    CorrelateDataBackward0 <<< config.block_count, config.thread_per_block, 0, device.stream() >>> (
      in_count_per_sample_i,
      n, out_width, out_height, out_channels,
      max_displacement, neighborhood_grid_radius, neighborhood_grid_width, kernel_radius,
      stride_1, stride_2,
      in_width, in_height, padded_in_width, padded_in_height, in_channels, in_count_per_sample, pad,
      ////
      interpolated_width, interpolated_height, padded_interpolated_width, padded_interpolated_height, in_count_per_sample_i, rate, neighborhood_grid_radius_a, neighborhood_grid_width_a,
      ////
      output_a_gradient, input_b, gradient);
//      printf("Finish - Backwoard 000 \n");
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
  const int        interpolated_width,
  const int        interpolated_height,
  const int        padded_interpolated_width,
  const int        padded_interpolated_height,
  const int        in_count_per_sample_i,
  const float      rate,
  const int        neighborhood_grid_radius_a,
  const int        neighborhood_grid_width_a,
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  const float     *input_a,
  const float     *gradient,
  float           *output_b_gradient) {
CudaLaunchConfig config = GetCudaLaunchConfig(in_count_per_sample_i, device);

  for (int n = 0; n < batch_size; n++) {
    CorrelateDataBackward1 << < config.block_count, config.thread_per_block, 0, device.stream() >> > (
      in_count_per_sample_i,
      n, out_width, out_height, out_channels,
      max_displacement, neighborhood_grid_radius, neighborhood_grid_width, kernel_radius,
      stride_1, stride_2,
      in_width, in_height, padded_in_width, padded_in_height, in_channels, in_count_per_sample, pad,
      ////
      interpolated_width, interpolated_height, padded_interpolated_width, padded_interpolated_height, in_count_per_sample_i, rate, neighborhood_grid_radius_a, neighborhood_grid_width_a,
      ////
      output_b_gradient, input_a, gradient);
//      printf("Finish - Backwoard 111 \n");
  }
//  printf("#################################################################\n");
}
} // end namespace tensorflow

#endif  // GOOGLE_CUDA
