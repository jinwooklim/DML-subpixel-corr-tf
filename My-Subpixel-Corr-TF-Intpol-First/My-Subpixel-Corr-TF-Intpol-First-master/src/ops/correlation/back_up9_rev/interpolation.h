#ifndef FLOWNET_INTERPOLATION_H_
#define FLOWNET_INTERPOLATION_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

void Interpolation(const GPUDevice& device,
         const float     *input,
         int              batch_size,
         int              input_height,
         int              input_width,
         int              input_channels,
         int              interpolated_height,
         int              interpolated_width,
         float rate,
         float           *interpolated_input);
} // end namespace tensorflow

#endif // FLOWNET_INTERPOLATION_H_
