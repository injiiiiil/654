#pragma once

#include "THC/THC.h"

#include "Descriptors.h"

#include <ATen/ATen.h>

#include "cudnn-wrapper.h"
#include <vector>

namespace torch { namespace cudnn {

struct ConvolutionParams
{
  cudnnDataType_t dataType;
  int input_size[5];
  int input_stride[5];
  int weight_size[5];
  int pad[3];
  int stride[3];
  int dilation[3];
  int groups;
};

struct Convolution
{
  ConvolutionParams params;
  TensorDescriptor idesc;
  TensorDescriptor odesc;
  TensorDescriptor odesc_bias;
  TensorDescriptor bdesc;
  FilterDescriptor wdesc;
  ConvolutionDescriptor cdesc;
  int groups;
  bool transposed;

  // WARNING: if transposed == true, then idesc and odesc are swapped!
  // WARNING2: WARNING does not apply to odesc_bias :)
  // This allows for reusing the function code (with a small exception in
  // backward_filter)

  Convolution(
      cudnnDataType_t dataType, const at::Tensor& input, const at::Tensor& weight,
      const at::Tensor& bias, const at::Tensor& output, std::vector<int> pad,
      std::vector<int> stride, std::vector<int> dilation, int groups,
      bool transposed);
};

void cudnn_convolution_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& output,
    Convolution* info, bool benchmark, bool deterministic);

void cudnn_convolution_add_bias(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    const at::Tensor& bias, const at::Tensor& output, Convolution* info);

void cudnn_convolution_backward_data(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    const at::Tensor& gradOutput, const at::Tensor& gradInput, const at::Tensor& weight,
    Convolution* info, bool benchmark, bool deterministic);

void cudnn_convolution_backward_filter(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    const at::Tensor& gradOutput, const at::Tensor& input, const at::Tensor& gradWeight,
    Convolution* info, bool benchmark, bool deterministic);

void cudnn_convolution_backward_bias(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    const at::Tensor& gradOutput, const at::Tensor& gradBias, Convolution* info);

// Helpers that allow to queue initialization, conv kernel and bias addition
// without reacquiring GIL in between.
Convolution* cudnn_convolution_full_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& output, std::vector<int> pad, std::vector<int> stride,
    std::vector<int> dilation, int groups, bool benchmark, bool deterministic);

Convolution* cudnn_convolution_transpose_full_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& output, std::vector<int> pad, std::vector<int> stride,
    std::vector<int> dilation, int groups, bool benchmark, bool deterministic);

}}  // namespace torch::cudnn
