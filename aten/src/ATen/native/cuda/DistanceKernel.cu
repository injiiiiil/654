#include "ATen/ATen.h"
#include <THC/THCTensorMathReduce.cuh>

#include "DistanceKernel.cuh"


namespace at { namespace native {

namespace {

static const int warp_size = 32;
static const int forward_threads = 8 * warp_size;
static const int backward_threads = 8 * warp_size;

template <typename scalar_t>
struct dists {

  static __forceinline__ __device__ scalar_t sign(scalar_t val) {
    return (0 < val) - (val < 0);
  }

  // Zero norm
  struct zero {
    static __forceinline__ __device__ void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) { agg += diff != 0.0; }
    static __forceinline__ __device__ scalar_t finish(const scalar_t agg, const scalar_t p) { return agg; }
    static __forceinline__ __device__ void agg(scalar_t& update, const scalar_t other) { update += other; }
  };

  // One norm
  struct one {
    static __forceinline__ __device__ void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) { agg += diff; }
    static __forceinline__ __device__ scalar_t finish(const scalar_t agg, const scalar_t p) { return agg; }
    static __forceinline__ __device__ void agg(scalar_t& update, const scalar_t other) { update += other; }
    static __forceinline__ __device__ scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const scalar_t p) { return grad * sign(diff); }
  };

  // Special case backward when p is less than two
  struct lt_two {
    static __forceinline__ __device__ scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const scalar_t p) { return dist == 0.0 ? 0 : sign(diff) * std::pow(std::abs(diff), p - 1) * grad / std::pow(dist, p - 1); }
  };

  // Two norm
  struct two {
    static __forceinline__ __device__ void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) { agg += diff * diff; }
    static __forceinline__ __device__ scalar_t finish(const scalar_t agg, const scalar_t p) { return std::sqrt(agg); }
    static __forceinline__ __device__ void agg(scalar_t& update, const scalar_t other) { update += other; }
    static __forceinline__ __device__ scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const scalar_t p) { return dist == 0.0 ? 0 : grad * diff / dist; }
  };

  // General p norm
  struct p {
    static __forceinline__ __device__ void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) { agg += std::pow(diff, p); }
    static __forceinline__ __device__ scalar_t finish(const scalar_t agg, const scalar_t p) { return std::pow(agg, 1.0 / p); }
    static __forceinline__ __device__ void agg(scalar_t& update, const scalar_t other) { update += other; }
    static __forceinline__ __device__ scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const scalar_t p) { return dist == 0.0 ? 0 : diff * std::pow(std::abs(diff), p - 2) * grad / std::pow(dist, p - 1); }
  };

  // Inf norm
  struct inf {
    static __forceinline__ __device__ void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) { if (diff > agg) { agg = diff; } }
    static __forceinline__ __device__ scalar_t finish(const scalar_t agg, const scalar_t p) { return agg; }
    static __forceinline__ __device__ void agg(scalar_t& update, const scalar_t other) { if (other > update) { update = other; } }
    static __forceinline__ __device__ scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const scalar_t p) { return grad * sign(diff) * (std::abs(diff) == dist); }
  };

};

template <typename scalar_t, typename F>
__global__ static void pdist_kernel_cuda_impl(scalar_t * result, const scalar_t * self, const int64_t n, const int64_t m, const scalar_t p) {
  const int k = blockIdx.x;
  const int stride = blockDim.x;

  float n2 = n - .5;
  // The -1 accounts for floating point truncation issues
  int64_t i = static_cast<int64_t>((n2 - std::sqrt(n2 * n2 - 2 * k - 1)));
  int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;

  const scalar_t * const start = self + i * m;
  const scalar_t * const end = start + m;
  const scalar_t * a = start + threadIdx.x;
  const scalar_t * b = self + j * m + threadIdx.x;
  scalar_t agg = 0.0;
  for (; a < end; a += stride, b += stride) {
    F::inc(agg, std::abs(*a - *b), p);
  }
  
  // Reduce warps
  for (int offset = warp_size / 2; offset > 0; offset /= 2) {
    F::agg(agg, WARP_SHFL_DOWN(agg, offset));
  }

  // Reduce block
  static __shared__ scalar_t shared[forward_threads / warp_size];
  int lane = threadIdx.x % warp_size;
  int warp_id = threadIdx.x / warp_size;
  if (lane == 0) {
    shared[warp_id] = agg;
  }
  __syncthreads();
  agg = (threadIdx.x < blockDim.x / warp_size) ? shared[lane] : 0.0;
  if (warp_id == 0) {
    // Only reduce theads with nonzero data
    for (int offset = blockDim.x / warp_size / 2; offset > 0; offset /= 2) {
      F::agg(agg, WARP_SHFL_DOWN(agg, offset));
    }
  }
  if (threadIdx.x == 0) {
    result[k] = F::finish(agg, p);
  }
}

template <typename scalar_t, typename F>
__global__ static void pdist_backward_kernel_cuda_impl(scalar_t * buffer, const scalar_t * grad, const scalar_t * self, const scalar_t * dist, int64_t gs, const int64_t n, const int64_t m, const scalar_t p) {
  const int k = blockIdx.y;
  const int init = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  float n2 = n - .5;
  // The -1 accounts for floating point truncation issues
  int64_t i = static_cast<int64_t>((n2 - std::sqrt(n2 * n2 - 2 * k - 1)));
  int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;
  int64_t ib = j - i - 1;
  int64_t jb = n - 2 - i;

  const scalar_t grad_k = grad[k * gs];
  const scalar_t dist_k = dist[k];

  const scalar_t * const start = self + i * m;
  const scalar_t * const end = start + m;
  const scalar_t * self_i = start + init;
  const scalar_t * self_j = self + j * m + init;
  scalar_t * buff_i = buffer + (ib * n + i) * m + init;
  scalar_t * buff_j = buffer + (jb * n + j) * m + init;
  for (; self_i < end; self_i += stride, self_j += stride, buff_i += stride, buff_j += stride) {
    const scalar_t res = F::backward(*self_i - *self_j, grad_k, dist_k, p);
    *buff_i = res;
    *buff_j = -res;
  }
}

} // anonymous namespace

void pdist_kernel_cuda(Tensor& result, const Tensor& self, double p) {
  int64_t n = self.size(0);
  int64_t m = self.size(1);

  AT_DISPATCH_FLOATING_TYPES(self.type(), "pdist_cuda", [&] {
    if (p == 0.0) {
      pdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::zero><<<result.numel(), forward_threads>>>(result.data<scalar_t>(), self.data<scalar_t>(), n, m, p);
    } else if (p == 1.0) {
      pdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::one><<<result.numel(), forward_threads>>>(result.data<scalar_t>(), self.data<scalar_t>(), n, m, p);
    } else if (p == 2.0) {
      pdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::two><<<result.numel(), forward_threads>>>(result.data<scalar_t>(), self.data<scalar_t>(), n, m, p);
    } else if (std::isinf(p)) {
      pdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::inf><<<result.numel(), forward_threads>>>(result.data<scalar_t>(), self.data<scalar_t>(), n, m, p);
    } else {
      pdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::p><<<result.numel(), forward_threads>>>(result.data<scalar_t>(), self.data<scalar_t>(), n, m, p);
    }
  });
}

void pdist_backward_kernel_cuda(Tensor& result, const Tensor& grad, const Tensor& self, const double p, const Tensor& dist) {
  if (p == 0.0 || grad.numel() == 0 || self.numel() == 0) {
    result.fill_(0);
    return;
  }

  const int64_t n = result.size(0);
  int64_t m = self.size(1);
  const int horiz_blocks = (m + backward_threads * 8 - 1) / (backward_threads * 8);

  Tensor buffer = result.type().tensor({n - 1, result.size(0), result.size(1)});
  AT_DISPATCH_FLOATING_TYPES(self.type(), "pdist_cuda_backward", [&] {
    if (p == 1.0) {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::one><<<dim3(horiz_blocks, dist.numel()), backward_threads>>>(buffer.data<scalar_t>(), grad.data<scalar_t>(), self.data<scalar_t>(), dist.data<scalar_t>(), grad.stride(0), n, m, p);
    } else if (p < 2.0) {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::lt_two><<<dim3(horiz_blocks, dist.numel()), backward_threads>>>(buffer.data<scalar_t>(), grad.data<scalar_t>(), self.data<scalar_t>(), dist.data<scalar_t>(), grad.stride(0), n, m, p);
    } else if (p == 2.0) {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::two><<<dim3(horiz_blocks, dist.numel()), backward_threads>>>(buffer.data<scalar_t>(), grad.data<scalar_t>(), self.data<scalar_t>(), dist.data<scalar_t>(), grad.stride(0), n, m, p);
    } else if (std::isinf(p)) {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::inf><<<dim3(horiz_blocks, dist.numel()), backward_threads>>>(buffer.data<scalar_t>(), grad.data<scalar_t>(), self.data<scalar_t>(), dist.data<scalar_t>(), grad.stride(0), n, m, p);
    } else {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::p><<<dim3(horiz_blocks, dist.numel()), backward_threads>>>(buffer.data<scalar_t>(), grad.data<scalar_t>(), self.data<scalar_t>(), dist.data<scalar_t>(), grad.stride(0), n, m, p);
    }
  });

  at::sum_out(result, buffer, 0);
}

}} // at::native
