#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/native/Pool.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/cuda/LaunchUtils.h>
#include <THC/THCNumerics.cuh>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace {

#define CUDA_MAX_THREADS 1024
#define BLOCK_STRIDE 2

__device__ inline int min(int a, int b) {
  return a <= b ? a : b;
}

__device__ inline int max(int a, int b) {
  return a >= b ? a : b;
}
  
static __device__ inline int p_start(int size, int pad, int kernel, int dilation, int stride) {
  return (size + pad < ((kernel - 1) * dilation + 1)) ? 0 : (size + pad - ((kernel - 1) * dilation + 1)) / stride + 1;
}

static __device__ inline int p_end(int size, int pad, int pooled_size, int stride) {
  return min((size + pad) / stride + 1, pooled_size);
}

template <typename scalar_t, typename accscalar_t>
__global__ void avg_pool2d_out_cuda_frame(const int nthreads,
    const scalar_t* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    scalar_t* const top_data, const int divisor_override,
    const bool count_include_pad, const bool use_divisor) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    accscalar_t aveval = accscalar_t(0);
    const scalar_t* const bottom_slice = bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    int divide_factor;
    if (use_divisor) {
      divide_factor = divisor_override;
    } else {
      if(count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (hend - hstart) * (wend - wstart);
      }
    }
    top_data[index] = ScalarConvert<accscalar_t, scalar_t>::to(aveval / divide_factor);
  }
}

template <typename scalar_t, typename accscalar_t>
__global__ void avg_pool2d_out_cuda_frame_nhwc(const int nthreads,
    const scalar_t* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    scalar_t* const top_data, const int divisor_override,
    const bool count_include_pad, const bool use_divisor) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = index % channels;
    const int pw = (index / channels) % pooled_width;
    const int ph = (index / channels / pooled_width) % pooled_height;
    const int n = index / channels / pooled_width / pooled_height;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    accscalar_t aveval = accscalar_t(0);
    const scalar_t* const bottom_slice = bottom_data + n * channels * height * width + c;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[(h * width + w) * channels];
      }
    }
    int divide_factor;
    if (use_divisor) {
      divide_factor = divisor_override;
    } else {
      if(count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (hend - hstart) * (wend - wstart);
      }
    }
    top_data[index] = ScalarConvert<accscalar_t, scalar_t>::to(aveval / divide_factor);
  }
}

template <typename scalar_t, typename accscalar_t>
__global__ void avg_pool2d_backward_out_cuda_frame(const int nthreads, const scalar_t* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    scalar_t* const bottom_diff, const int divisor_override, 
    bool count_include_pad, bool use_divisor) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    accscalar_t gradient = accscalar_t(0);
    const scalar_t* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, height);
        wend = min(wend, width);
        int divide_factor;
        if (use_divisor) {
          divide_factor = divisor_override;
        } else {
          if(count_include_pad) {
            divide_factor = pool_size;
          } else {
            divide_factor = (hend - hstart) * (wend - wstart);
          }
        }
        gradient += top_diff_slice[ph * pooled_width + pw] / divide_factor;
      }
    }
    bottom_diff[index] = ScalarConvert<accscalar_t, scalar_t>::to(gradient);
  }
}

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(CUDA_MAX_THREADS)
__global__ void avg_pool2d_backward_out_cuda_frame_nhwc_old(
    const int nthreads, const scalar_t* top_diff,
    const int nbatch, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int out_stride_c, const int out_stride_h, const int out_stride_w,
    const int in_stride_n, const int in_stride_c,
    const int in_stride_h, const int in_stride_w,
    const int kernel_stride_C, const int kernel_size_C,
    scalar_t* bottom_diff,
    const int divisor_override, const bool count_include_pad, const bool use_divisor) {
  // reserved for future use
  const int dilation_h = 1;
  const int dilation_w = 1;

  extern __shared__ int smem[];
  accscalar_t *out_cached = reinterpret_cast<accscalar_t*>(smem);

  int thread_id = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  int block_size = blockDim.x * blockDim.y * blockDim.z;

  for (int i = thread_id; i < kernel_size_C*blockDim.x*blockDim.y*blockDim.z; i+= block_size) {
    out_cached[i] = accscalar_t(0.0);
  }

  __syncthreads();

  int batch_id = blockIdx.x % nbatch;
  int channel_id = blockIdx.x / nbatch;
  int channel_offset = threadIdx.x + channel_id * blockDim.x;

  bottom_diff = bottom_diff + batch_id * height * width * channels;
  top_diff = top_diff + batch_id * pooled_height * pooled_width * channels;

  out_cached = &out_cached[(threadIdx.z * blockDim.y + threadIdx.y) * kernel_size_C*blockDim.x];

  int iH = (height + gridDim.z-1) / gridDim.z;
  int iW = (width + gridDim.y-1) / gridDim.y;
  int istartH = threadIdx.z + blockIdx.z*iH;
  int iendH = ::min(istartH+iH, height);
  int istartW = threadIdx.y + blockIdx.y*iW;
  int iendW = ::min(istartW+iW, width);

  for (int ih = istartH; ih < iendH; ih+=blockDim.z) {
    int phstart = p_start(ih, pad_h, kernel_h, dilation_h, stride_h);
    int phend = p_end(ih, pad_h, pooled_height, stride_h);
    for (int iw = istartW; iw < iendW; iw+=blockDim.y) {
      int pwstart = p_start(iw, pad_w, kernel_w, dilation_w, stride_w);
      int pwend = p_end(iw, pad_w, pooled_width, stride_w);

      int index_shift = ih * width + iw;
      for(int oh = phstart; oh < phend; ++oh) {
        int hstart = oh * stride_h - pad_h;
        int hend = min(hstart + kernel_h, height + pad_h);
        for(int ow = pwstart; ow < pwend; ++ow) {
          int wstart = ow * stride_w - pad_w;
          int wend = min(wstart + kernel_w, width + pad_w);

          // pool_size if count_include_pad
          int pool_size = (hend - hstart) * (wend - wstart);

          while (hstart < 0) hstart += dilation_h;
          while (wstart < 0) wstart += dilation_w;
          hend = min(hend, height);
          wend = min(wend, width);

          int divide_factor;
          if (use_divisor) {
            divide_factor = divisor_override;
          } else {
            if(count_include_pad) {
              divide_factor = pool_size;
            } else {
              divide_factor = (hend - hstart) * (wend - wstart);
            }
          }
          // avoid division in loops
          accscalar_t mul_factor = 1.0 / divide_factor;

          const scalar_t* ptr_top_diff = top_diff + oh*out_stride_h + ow*out_stride_w;
          int cached_index = threadIdx.x;
          for (int c = channel_offset; c < channels; c += blockDim.x*kernel_stride_C) {
            out_cached[cached_index] += ptr_top_diff[c*out_stride_c] * mul_factor;
            cached_index += blockDim.x;
          }
        }
      }
      scalar_t *ptr_bottom_diff = bottom_diff + index_shift * channels;
      int cached_index = threadIdx.x;
      for (int c = channel_offset; c < channels; c += blockDim.x*kernel_stride_C) {
        ptr_bottom_diff[c] = scalar_cast<scalar_t>(out_cached[cached_index]);
        out_cached[cached_index] = accscalar_t(0.0);
        cached_index += blockDim.x;
      }
    }
  }
}

template <typename scalar_t, typename accscalar_t>
__global__ void avg_pool2d_backward_out_cuda_frame_nhwc(const int nthreads, 
    const scalar_t* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    scalar_t* const bottom_diff, const int divisor_override, 
    bool count_include_pad, bool use_divisor) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = index % channels; 
    const int w = (index / channels) % width;
    const int h = (index / channels / width) % height; 
    const int n = index / channels / width / height; 

    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    accscalar_t gradient = accscalar_t(0);
    const scalar_t* const top_diff_slice = top_diff + n * channels * pooled_height * pooled_width + c;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, height);
        wend = min(wend, width);
        int divide_factor;
        if (use_divisor) {
          divide_factor = divisor_override;
        } else {
          if(count_include_pad) {
            divide_factor = pool_size;
          } else {
            divide_factor = (hend - hstart) * (wend - wstart);
          }
        }
        gradient += top_diff_slice[(ph * pooled_width + pw) * channels] / divide_factor;
      }
    }
    bottom_diff[index] = ScalarConvert<accscalar_t, scalar_t>::to(gradient);
  }
}

void avg_pool2d_out_cuda_template(
  Tensor& output,
  const Tensor& input_,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  c10::optional<int64_t> divisor_override)
{
  TensorArg output_arg{ output, "output", 1 };
  TensorArg input_arg{ input_, "input_", 2 };

  checkAllSameGPU("avg_pool2d_out_cuda", {output_arg, input_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints");
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
    "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "avg_pool2d: padding must either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  const auto memory_format = input_.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast){
    TORCH_CHECK(input_.ndimension() == 4,
      "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
  } else {
    TORCH_CHECK((input_.ndimension() == 3 || input_.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
  }

  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0,
    "divisor must be not zero");

  const int64_t nbatch = input_.ndimension() == 4 ? input_.size(-4) : 1;
  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);
  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);

  pool2d_shape_check(
    input_,
    kH, kW, dH, dW, padH, padW, 1, 1,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight, outputWidth);

  Tensor input = input_.contiguous(memory_format);

  output.resize_({nbatch, nInputPlane, outputHeight, outputWidth});

  output.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);

  const int32_t count = safe_downcast<int32_t, int64_t>(output.numel());
  const uint32_t num_threads = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);
  const uint32_t num_blocks = cuda::ATenCeilDiv<uint32_t>(count, num_threads);

  bool use_divisor = divisor_override.has_value();
  const auto divisor_override_value = use_divisor ? divisor_override.value() : 0; 

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
    "avg_pool2d_out_cuda_frame",
    [&] {
      AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "avg_pool2d_out_cuda_frame", [&] {
        using accscalar_t = acc_type<scalar_t, true>;

        scalar_t *output_data = output.data_ptr<scalar_t>();
        scalar_t *input_data = input.data_ptr<scalar_t>();

        switch (memory_format){
          case MemoryFormat::ChannelsLast: {
             avg_pool2d_out_cuda_frame_nhwc<scalar_t, accscalar_t>
               <<<num_blocks, num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                 count,
                 input_data,
                 nbatch,
                 nInputPlane,
                 inputHeight, inputWidth,
                 outputHeight, outputWidth,
                 kH, kW,
                 dH, dW,
                 padH, padW,
                 output_data,
                 divisor_override_value,
                 count_include_pad, use_divisor);
            break;
          }
          case MemoryFormat::Contiguous: {
            avg_pool2d_out_cuda_frame<scalar_t, accscalar_t>
              <<<num_blocks, num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                count,
                input_data,
                nbatch,
                nInputPlane,
                inputHeight, inputWidth,
                outputHeight, outputWidth,
                kH, kW,
                dH, dW,
                padH, padW,
                output_data,
                divisor_override_value,
                count_include_pad, use_divisor);
            break; 
          }
          default: TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous"); 
        }
      });
    }
  );

  AT_CUDA_CHECK(cudaGetLastError());

  if (input.ndimension() == 3) {
    output.resize_({nInputPlane, outputHeight, outputWidth});
  }
}

Tensor& avg_pool2d_backward_out_cuda_template(
  Tensor& gradInput,
  const Tensor& gradOutput_,
  const Tensor& input_,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  c10::optional<int64_t> divisor_override)
{
  TensorArg gradInput_arg{ gradInput, "gradInput", 1 };
  TensorArg gradOutput_arg{ gradOutput_, "gradOutput_", 2 };
  TensorArg input_arg{ input_, "input_", 3 };

  checkAllSameGPU("avg_pool2d_backward_out_cuda",
                  {gradInput_arg, gradOutput_arg, input_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints");
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
    "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "avg_pool2d: padding must either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0,
    "divisor must be not zero");

  const auto memory_format = input_.suggest_memory_format(); 
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(input_.ndimension() == 4,
      "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
  } else {
    TORCH_CHECK((input_.ndimension() == 3 || input_.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
  }

  const Tensor input = input_.contiguous(memory_format);
  const Tensor gradOutput = gradOutput_.contiguous(memory_format);

  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);
  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);

  avg_pool2d_backward_shape_check(
    input_,
    gradOutput_,
    nbatch,
    kH, kW, dH, dW, padH, padW,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight, outputWidth);

  gradInput.resize_as_(input);
  gradInput.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);

  const int32_t count = safe_downcast<int32_t, int64_t>(input.numel());
  const uint32_t num_threads = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);
  const uint32_t num_blocks = cuda::ATenCeilDiv<uint32_t>(count, num_threads);

  bool use_divisor = divisor_override.has_value();
  const auto divisor_override_value = use_divisor ? divisor_override.value() : 0;

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
    "avg_pool2d_backward_out_cuda_frame",
    [&] {
      AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "avg_pool2d_backward_out_cuda_frame", [&] {
        using accscalar_t = acc_type<scalar_t, true>;

        scalar_t *gradOutput_data = gradOutput.data_ptr<scalar_t>();
        scalar_t *gradInput_data = gradInput.data_ptr<scalar_t>();

        switch (memory_format) {
          case MemoryFormat::ChannelsLast: {
            avg_pool2d_backward_out_cuda_frame_nhwc<scalar_t, accscalar_t>
              <<<num_blocks, num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                count,
                gradOutput_data,
                nbatch,
                nInputPlane,
                inputHeight, inputWidth,
                outputHeight, outputWidth,
                kH, kW,
                dH, dW,
                padH, padW,
                gradInput_data,
                divisor_override_value, 
                count_include_pad, use_divisor);
            break;
          }
          case MemoryFormat::Contiguous: {
            avg_pool2d_backward_out_cuda_frame<scalar_t, accscalar_t>
              <<<num_blocks, num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                count,
                gradOutput_data,
                nbatch,
                nInputPlane,
                inputHeight, inputWidth,
                outputHeight, outputWidth,
                kH, kW,
                dH, dW,
                padH, padW,
                gradInput_data,
                divisor_override_value, 
                count_include_pad, use_divisor);
            break;
          }
          default: TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
        }
      });
    }
  );

  AT_CUDA_CHECK(cudaGetLastError());

  return gradInput;
}

} // namespace

Tensor& avg_pool2d_out_cuda(
  Tensor& output,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  c10::optional<int64_t> divisor_override)
{
  avg_pool2d_out_cuda_template(
   output,
   input,
   kernel_size,
   stride,
   padding,
   ceil_mode,
   count_include_pad,
   divisor_override);
  return output;
}

Tensor avg_pool2d_cuda(
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  c10::optional<int64_t> divisor_override)
{
  Tensor output = at::empty({0}, input.options());
  avg_pool2d_out_cuda_template(
    output,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override);
  return output;
}

Tensor& avg_pool2d_backward_out_cuda(
  Tensor& gradInput,
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  c10::optional<int64_t> divisor_override)
{
  avg_pool2d_backward_out_cuda_template(
    gradInput,
    gradOutput_,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override);
  return gradInput;
}

Tensor avg_pool2d_backward_cuda(
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  c10::optional<int64_t> divisor_override)
{
  auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  avg_pool2d_backward_out_cuda_template(
    gradInput,
    gradOutput_,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override);
  return gradInput;
}

} // at::native
} // at
