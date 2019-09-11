/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/common.h>
#include <qnnpack/params.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PYTORCH_DECLARE_Q8GEMM_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(            \
      size_t mr,                                 \
      size_t nr,                                 \
      size_t k,                                  \
      const uint8_t* a,                          \
      size_t a_stride,                           \
      const void* w,                             \
      uint8_t* c,                                \
      size_t c_stride,                           \
      const union pytorch_qnnp_conv_quantization_params* quantization_params);

PYTORCH_DECLARE_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_3x3c8__neon)
PYTORCH_DECLARE_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_2x4c8__neon)
PYTORCH_DECLARE_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_4x8__neon)
PYTORCH_DECLARE_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_6x4__neon)
PYTORCH_DECLARE_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_8x8__neon)

PYTORCH_DECLARE_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_4x8__aarch32_neon)

PYTORCH_DECLARE_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_8x8__aarch64_neon)

PYTORCH_DECLARE_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_2x4c8__sse2)
PYTORCH_DECLARE_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_4x4c2__sse2)

#define PYTORCH_DECLARE_Q8GEMM_XZP_UKERNEL_FUNCTION(fn_name)      \
  PYTORCH_QNNP_INTERNAL void fn_name(                     \
      size_t mr,                                          \
      size_t nr,                                          \
      size_t k,                                           \
      const uint8_t* a,                                   \
      size_t a_stride,                                    \
      const int32_t* a_sum,                               \
      const void* w,                                      \
      uint8_t* c,                                         \
      size_t c_stride,                                    \
      const union pytorch_qnnp_q31_requantization_params* \
          requantization_params);
PYTORCH_DECLARE_Q8GEMM_XZP_UKERNEL_FUNCTION(pytorch_q8gemm_xzp_ukernel_4x8c2__neon)
PYTORCH_DECLARE_Q8GEMM_XZP_UKERNEL_FUNCTION(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon)

PYTORCH_QNNP_INTERNAL void pytorch_q8sumrows_ukernel_4x__neon(
    const uint8_t* a,
    size_t m,
    size_t k,
    size_t stride,
    const int32_t multiplier,
    int32_t* row_sum);

#ifdef __cplusplus
} /* extern "C" */
#endif
