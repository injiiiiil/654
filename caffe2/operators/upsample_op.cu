#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/upsample_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

inline __device__ int idx(
    const int n,
    const int num_channels,
    const int c,
    const int height,
    const int width,
    const int y,
    const int x) {
  return ((n * num_channels + c) * height + y) * width + x;
}

// input is X, output is Y
__global__ void UpsampleBilinearKernel(
    const int output_size,
    const int num_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const float height_scale,
    const float width_scale,
    const float* X,
    float* Y) {
  CUDA_1D_KERNEL_LOOP(index, output_size) {
    int indexTemp = index;
    const int out_x = indexTemp % output_width;
    indexTemp /= output_width;
    const int out_y = indexTemp % output_height;
    indexTemp /= output_height;
    const int c = indexTemp % num_channels;
    indexTemp /= num_channels;
    const int n = indexTemp;

    const int in_y = fminf(out_y / height_scale, input_height - 1);
    const int in_x = fminf(out_x / width_scale, input_width - 1);

    const float rheight =
        output_height > 1 ? (input_height - 1.f) / (output_height - 1.f) : 0.f;
    const float rwidth =
        output_width > 1 ? (input_width - 1.f) / (output_width - 1.f) : 0.f;

    // Compute Y axis lambdas
    const float h1r = rheight * out_y;
    const int h1 = (int)h1r;
    const int h1p = (h1 < input_height - 1) ? 1 : 0;
    const float h1lambda = h1r - h1;
    const float h0lambda = 1.f - h1lambda;

    // Compute X axis lambdas
    const float w1r = rwidth * out_x;
    const int w1 = (int)w1r;
    const int w1p = (w1 < input_width - 1) ? 1 : 0;
    const float w1lambda = w1r - w1;
    const float w0lambda = 1.f - w1lambda;

    Y[index] =
        (h0lambda *
             (w0lambda *
                  X[idx(
                      n, num_channels, c, input_height, input_width, h1, w1)] +
              w1lambda *
                  X[idx(
                      n,
                      num_channels,
                      c,
                      input_height,
                      input_width,
                      h1,
                      w1 + w1p)]) +
         h1lambda *
             (w0lambda *
                  X[idx(
                      n,
                      num_channels,
                      c,
                      input_height,
                      input_width,
                      h1 + h1p,
                      w1)] +
              w1lambda *
                  X[idx(
                      n,
                      num_channels,
                      c,
                      input_height,
                      input_width,
                      h1 + h1p,
                      w1 + w1p)]));
  }
}

// input is dY, output is dX
__global__ void UpsampleBilinearGradientKernel(
    const int input_size,
    const int num_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const float height_scale,
    const float width_scale,
    const float* dY,
    float* dX) {
  CUDA_1D_KERNEL_LOOP(index, input_size) {
    int indexTemp = index;
    const int in_x = indexTemp % input_width;
    indexTemp /= input_width;
    const int in_y = indexTemp % input_height;
    indexTemp /= input_height;
    const int c = indexTemp % num_channels;
    indexTemp /= num_channels;
    const int n = indexTemp;

    const int out_y = fminf(in_y / height_scale, output_height - 1);
    const int out_x = fminf(in_x / width_scale, output_width - 1);

    const float rheight =
        output_height > 1 ? (output_height - 1.f) / (input_height - 1.f) : 0.f;
    const float rwidth =
        output_width > 1 ? (output_width - 1.f) / (input_width - 1.f) : 0.f;

    // Compute Y axis lambdas
    const float h1r = rheight * in_y;
    const int h1 = (int)h1r;
    const int h1p = (h1 < output_height - 1) ? 1 : 0;
    const float h1lambda = h1r - h1;
    const float h0lambda = 1.f - h1lambda;

    // Compute X axis lambdas
    const float w1r = rwidth * in_x;
    const int w1 = (int)w1r;
    const int w1p = (w1 < output_width - 1) ? 1 : 0;
    const float w1lambda = w1r - w1;
    const float w0lambda = 1.f - w1lambda;

#if __CUDA_ARCH__ >= 350
    const float dYi = __ldg(&dY[index]);
#else
    const float dYi = dY[index];
#endif

    atomicAdd(
        &dX[idx(n, num_channels, c, output_height, output_width, h1, w1)],
        h0lambda * w0lambda * dYi);
    atomicAdd(
        &dX[idx(n, num_channels, c, output_height, output_width, h1, w1 + w1p)],
        h0lambda * w1lambda * dYi);
    atomicAdd(
        &dX[idx(n, num_channels, c, output_height, output_width, h1 + h1p, w1)],
        h1lambda * w0lambda * dYi);
    atomicAdd(
        &dX[idx(
            n,
            num_channels,
            c,
            output_height,
            output_width,
            h1 + h1p,
            w1 + w1p)],
        h1lambda * w1lambda * dYi);
  }
}

} // namespace

template <>
bool UpsampleBilinearOp<float, CUDAContext>::RunOnDevice() {
  const auto& X = Input(0);

  const auto inputDims = X.dims();
  CAFFE_ENFORCE_EQ(4, inputDims.size());
  const int batch_size = X.dim32(0), num_channels = X.dim32(1),
            input_height = X.dim32(2), input_width = X.dim32(3);
  if (InputSize() == 2) {
    const auto& scales = Input(1);
    CAFFE_ENFORCE_EQ(scales.ndim(), 1);
    CAFFE_ENFORCE_EQ(scales.size(), 2);
    float scales_data[2];
    context_.CopyToCPU<float>(2, scales.data<float>(), scales_data);
    height_scale_ = scales_data[0];
    width_scale_ = scales_data[1];
  }
  int output_width = input_width * width_scale_;
  int output_height = input_height * height_scale_;
  auto* Y = Output(
      0,
      {batch_size, num_channels, output_height, output_width},
      at::dtype<float>());

  const auto size = Y->size();
  UpsampleBilinearKernel<<<
      CAFFE_GET_BLOCKS(size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      size,
      num_channels,
      input_height,
      input_width,
      output_height,
      output_width,
      height_scale_,
      width_scale_,
      X.data<float>(),
      Y->template mutable_data<float>());

  return true;
}

template <>
bool UpsampleBilinearGradientOp<float, CUDAContext>::RunOnDevice() {
  const auto& dY = Input(0);
  const auto& X = Input(1);

  const auto inputDims = dY.dims();
  CAFFE_ENFORCE_EQ(4, inputDims.size());
  const int batch_size = dY.dim32(0);
  const int num_channels = dY.dim32(1);
  const int input_height = dY.dim32(2);
  const int input_width = dY.dim32(3);
  const int output_height = X.dim32(2);
  const int output_width = X.dim32(3);
  if (InputSize() == 3) {
    const auto& scales = Input(2);
    CAFFE_ENFORCE_EQ(scales.ndim(), 1);
    CAFFE_ENFORCE_EQ(scales.size(), 2);
    float scales_data[2];
    context_.CopyToCPU<float>(2, scales.data<float>(), scales_data);
    height_scale_ = scales_data[0];
    width_scale_ = scales_data[1];
  }
  auto* dX = Output(
      0,
      {batch_size, num_channels, output_height, output_width},
      at::dtype<float>());
  math::Set<float, CUDAContext>(
      dX->size(), 0.0f, dX->mutable_data<float>(), &context_);

  const auto size = dY.size();
  UpsampleBilinearGradientKernel<<<
      CAFFE_GET_BLOCKS(size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      size,
      num_channels,
      input_height,
      input_width,
      output_height,
      output_width,
      height_scale_,
      width_scale_,
      dY.data<float>(),
      dX->template mutable_data<float>());

  return true;
}

REGISTER_CUDA_OPERATOR(
    UpsampleBilinear,
    UpsampleBilinearOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    UpsampleBilinearGradient,
    UpsampleBilinearGradientOp<float, CUDAContext>);
} // namespace caffe2
