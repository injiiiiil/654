#pragma once
#ifdef CPU_CAPABILITY_AVX2

#include <ATen/core/Tensor.h>
#include <ATen/cpu/vec/intrinsics.h>
#include <c10/util/irange.h>

namespace {

static __m128i inline mm_cvtepu8_epi32(void* ptr) {
  return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)ptr));
}

void unpack_rgb(
    uint8_t* unpacked, // OUT
    const at::Tensor& packed_tensor, // IN
    bool is_channels_last) {
  const uint8_t* packed = (const uint8_t*)packed_tensor.data_ptr<uint8_t>();
  auto num_pixels = packed_tensor.size(1) * packed_tensor.size(2);
  auto num_channels = packed_tensor.size(0);
  auto packed_stride = is_channels_last ? 1 : num_pixels;
  auto packed_increment = is_channels_last ? num_channels : 1;

  for (const auto i : c10::irange(num_pixels)) {
    for (const auto j : c10::irange(num_channels)) {
      unpacked[j] = packed[j * packed_stride];
    }
    for (const auto j : c10::irange(num_channels, 4)) {
      unpacked[j] = 255;
    }
    unpacked += 4;
    packed += packed_increment;
  }
}

void pack_rgb(
    const uint8_t* unpacked, // IN
    const at::Tensor& packed_tensor, // OUT
    bool is_channels_last) {
  uint8_t* packed = (uint8_t*)packed_tensor.data_ptr<uint8_t>();
  auto num_pixels = packed_tensor.size(1) * packed_tensor.size(2);
  auto num_channels = packed_tensor.size(0);
  auto packed_stride = is_channels_last ? 1 : num_pixels;
  auto packed_increment = is_channels_last ? num_channels : 1;
  for (const auto i : c10::irange(num_pixels)) {
    for (const auto j : c10::irange(num_channels)) {
      packed[j * packed_stride] = unpacked[j];
    }
    unpacked += 4;
    packed += packed_increment;
  }
}

void ImagingResampleHorizontalConvolution8u4x(
    uint32_t* lineOut0,
    uint32_t* lineOut1,
    uint32_t* lineOut2,
    uint32_t* lineOut3,
    uint32_t* lineIn0,
    uint32_t* lineIn1,
    uint32_t* lineIn2,
    uint32_t* lineIn3,
    int xsize,
    int* xbounds,
    int16_t* kk,
    int kmax,
    int coefs_precision);
void ImagingResampleHorizontalConvolution8u(
    uint32_t* lineOut,
    uint32_t* lineIn,
    int xsize,
    int* xbounds,
    int16_t* kk,
    int kmax,
    int coefs_precision);
void ImagingResampleVerticalConvolution8u(
    uint32_t* lineOut,
    uint32_t* imIn,
    int xmin,
    int xmax,
    int16_t* k,
    int coefs_precision,
    int xin);

void ImagingResampleHorizontal_8bpc(
    uint32_t* unpacked_output_p,
    uint32_t* unpacked_input_p,
    int ksize,
    std::vector<at::Tensor> horiz_indices_weights,
    unsigned int horiz_weights_precision,
    int xout,
    int yout,
    int xin) {
  int yy;

  int16_t* kk = (int16_t*)(horiz_indices_weights[3].data_ptr<double>());

  std::vector<int> bounds_vec(2 * xout, 0);
  int* bounds = bounds_vec.data();

  int64_t* idx_ptr_xmin = horiz_indices_weights[0].data_ptr<int64_t>();
  int64_t* idx_ptr_size = horiz_indices_weights[1].data_ptr<int64_t>();
  for (int i = 0; i < xout; i++) {
    bounds[2 * i + 0] = idx_ptr_xmin[i];
    bounds[2 * i + 1] = idx_ptr_size[i];
  }

  yy = 0;
  for (; yy < yout - 3; yy += 4) {
    ImagingResampleHorizontalConvolution8u4x(
        unpacked_output_p + yy * xout,
        unpacked_output_p + (yy + 1) * xout,
        unpacked_output_p + (yy + 2) * xout,
        unpacked_output_p + (yy + 3) * xout,
        unpacked_input_p + yy * xin,
        unpacked_input_p + (yy + 1) * xin,
        unpacked_input_p + (yy + 2) * xin,
        unpacked_input_p + (yy + 3) * xin,
        xout,
        bounds,
        kk,
        ksize,
        (int)horiz_weights_precision);
  }
  for (; yy < yout; yy++) {
    ImagingResampleHorizontalConvolution8u(
        unpacked_output_p + yy * xout,
        unpacked_input_p + yy * xin,
        xout,
        bounds,
        kk,
        ksize,
        (int)horiz_weights_precision);
  }
}

void ImagingResampleVertical_8bpc(
    uint32_t* unpacked_output_p,
    uint32_t* unpacked_input_p,
    int ksize,
    std::vector<at::Tensor> vert_indices_weights,
    unsigned int vert_weights_precision,
    int xout,
    int yout) {
  int ymin, ymax;
  int16_t* k = nullptr;
  int16_t* kk = (int16_t*)(vert_indices_weights[3].data_ptr<double>());

  int64_t* idx_ptr_xmin = vert_indices_weights[0].data_ptr<int64_t>();
  int64_t* idx_ptr_size = vert_indices_weights[1].data_ptr<int64_t>();
  for (const auto yy : c10::irange(yout)) {
    k = &kk[yy * ksize];

    ymin = idx_ptr_xmin[yy];
    ymax = idx_ptr_size[yy];
    ImagingResampleVerticalConvolution8u(
        unpacked_output_p + yy * xout,
        unpacked_input_p,
        ymin,
        ymax,
        k,
        (int)vert_weights_precision,
        xout);
  }
}

template <typename scale_type, class F>
uint32_t* ImagingResampleInner(
    uint32_t* unpacked_input_p,
    int xin,
    int yin,
    int xout,
    int yout,
    bool align_corners,
    const scale_type& scales,
    bool antialias) {
  int need_horizontal, need_vertical;

  uint32_t* unpacked_output_p = nullptr;
  uint32_t* unpacked_output_temp_p = nullptr;

  need_horizontal = xout != xin;
  need_vertical = yout != yin;

  if (need_horizontal) {
    int interp_dim = 3;
    auto [horiz_indices_weights, ksize_horiz, horiz_weights_precision] =
        F::compute_indices_int16_weights_aa(
            /*input_size=*/xin,
            /*output_size=*/xout,
            /*stride=*/1,
            /*ndims=*/4,
            /*reshape_dim=*/interp_dim,
            /*align_corners=*/align_corners,
            /*opt_scale=*/scales[interp_dim - 2],
            /*antialias=*/antialias);

    unpacked_output_temp_p = (uint32_t*)malloc(sizeof(uint32_t) * xout * yin);
    ImagingResampleHorizontal_8bpc(
        unpacked_output_temp_p,
        unpacked_input_p,
        ksize_horiz,
        horiz_indices_weights,
        horiz_weights_precision,
        xout,
        yin,
        xin);
    unpacked_output_p = unpacked_input_p = unpacked_output_temp_p;
  }
  if (need_vertical) {
    int interp_dim = 2;
    auto [vert_indices_weights, ksize_vert, vert_weights_precision] =
        F::compute_indices_int16_weights_aa(
            /*input_size=*/yin,
            /*output_size=*/yout,
            /*stride=*/1,
            /*ndims=*/4,
            /*reshape_dim=*/interp_dim,
            /*align_corners=*/align_corners,
            /*opt_scale=*/scales[interp_dim - 2],
            /*antialias=*/antialias);

    unpacked_output_p = (uint32_t*)malloc(sizeof(uint32_t) * xout * yout);
    ImagingResampleVertical_8bpc(
        unpacked_output_p,
        unpacked_input_p,
        ksize_vert,
        vert_indices_weights,
        vert_weights_precision,
        xout,
        yout);
    if (unpacked_output_temp_p != nullptr) {
      free(unpacked_output_temp_p);
    }
  }

  if (unpacked_output_p == nullptr) {
    // Happens if need_horizontal and need_vertical are both False.
    // But this should never be hit because we check for that in
    // upsample_avx_bilinear_or_bicubic()
    unpacked_output_p = unpacked_input_p;
  }

  return unpacked_output_p;
}

template <typename scale_type, class F>
void upsample_avx_bilinear_or_bicubic(
    const at::Tensor& input,
    const at::Tensor& output,
    bool align_corners,
    const scale_type& scales,
    bool antialias) {
  auto batch_size = input.size(0);
  auto xin = input.size(3);
  auto yin = input.size(2);
  auto xout = output.size(3);
  auto yout = output.size(2);
  auto num_pixels_input = xin * yin;

  uint32_t* unpacked_input_p =
      (uint32_t*)malloc(sizeof(uint32_t) * num_pixels_input);
  uint32_t* unpacked_output_p = nullptr;

  if (xin == xout && yin == yout) {
    output.copy_(input);
    return;
  }

  for (const auto i : c10::irange(batch_size)) {
    unpack_rgb(
        (uint8_t*)unpacked_input_p,
        input[i],
        input.is_contiguous(at::MemoryFormat::ChannelsLast));

    unpacked_output_p = ImagingResampleInner<scale_type, F>(
        unpacked_input_p,
        xin,
        yin,
        xout,
        yout,
        align_corners,
        scales,
        antialias);

    pack_rgb(
        (const uint8_t*)unpacked_output_p,
        output[i],
        output.is_contiguous(at::MemoryFormat::ChannelsLast));
    if (unpacked_output_p != nullptr && unpacked_output_p != unpacked_input_p) {
      free(unpacked_output_p);
    }
  }
  free(unpacked_input_p);
}

void ImagingResampleHorizontalConvolution8u4x(
    uint32_t* lineOut0,
    uint32_t* lineOut1,
    uint32_t* lineOut2,
    uint32_t* lineOut3,
    uint32_t* lineIn0,
    uint32_t* lineIn1,
    uint32_t* lineIn2,
    uint32_t* lineIn3,
    int xsize,
    int* xbounds,
    int16_t* kk,
    int kmax,
    int coefs_precision) {
  int xmin, xmax, x;
  int16_t* k;

  for (const auto xx : c10::irange(xsize)) {
    xmin = xbounds[xx * 2 + 0];
    xmax = xbounds[xx * 2 + 1];
    k = &kk[xx * kmax];
    x = 0;

    __m256i sss0, sss1;
    __m256i zero = _mm256_setzero_si256();
    __m256i initial = _mm256_set1_epi32(1 << (coefs_precision - 1));
    sss0 = initial;
    sss1 = initial;

    for (; x < xmax - 3; x += 4) {
      __m256i pix, mmk0, mmk1, source;

      mmk0 = _mm256_set1_epi32(*(int32_t*)&k[x]);
      mmk1 = _mm256_set1_epi32(*(int32_t*)&k[x + 2]);

      source = _mm256_inserti128_si256(
          _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)&lineIn0[x + xmin])),
          _mm_loadu_si128((__m128i*)&lineIn1[x + xmin]),
          1);
      // clang-format off
      pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
        -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0,
        -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk0));
      pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
        -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8,
        -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8));
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk1));

      source = _mm256_inserti128_si256(
          _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)&lineIn2[x + xmin])),
          _mm_loadu_si128((__m128i*)&lineIn3[x + xmin]),
          1);
      pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
        -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0,
        -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk0));
      pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
        -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8,
        -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8));
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk1));
    }

    for (; x < xmax - 1; x += 2) {
      __m256i pix, mmk;

      mmk = _mm256_set1_epi32(*(int32_t*)&k[x]);

      pix = _mm256_inserti128_si256(
          _mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)&lineIn0[x + xmin])),
          _mm_loadl_epi64((__m128i*)&lineIn1[x + xmin]),
          1);
      pix = _mm256_shuffle_epi8(pix, _mm256_set_epi8(
        -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0,
        -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));

      pix = _mm256_inserti128_si256(
          _mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)&lineIn2[x + xmin])),
          _mm_loadl_epi64((__m128i*)&lineIn3[x + xmin]),
          1);
      pix = _mm256_shuffle_epi8(pix, _mm256_set_epi8(
        -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0,
        -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));
      // clang-format on
    }

    for (; x < xmax; x++) {
      __m256i pix, mmk;

      // [16] xx k0 xx k0 xx k0 xx k0 xx k0 xx k0 xx k0 xx k0
      mmk = _mm256_set1_epi32(k[x]);

      // [16] xx a0 xx b0 xx g0 xx r0 xx a0 xx b0 xx g0 xx r0
      pix = _mm256_inserti128_si256(
          _mm256_castsi128_si256(mm_cvtepu8_epi32(&lineIn0[x + xmin])),
          mm_cvtepu8_epi32(&lineIn1[x + xmin]),
          1);
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));

      pix = _mm256_inserti128_si256(
          _mm256_castsi128_si256(mm_cvtepu8_epi32(&lineIn2[x + xmin])),
          mm_cvtepu8_epi32(&lineIn3[x + xmin]),
          1);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));
    }

    sss0 = _mm256_srai_epi32(sss0, coefs_precision);
    sss1 = _mm256_srai_epi32(sss1, coefs_precision);
    sss0 = _mm256_packs_epi32(sss0, zero);
    sss1 = _mm256_packs_epi32(sss1, zero);
    sss0 = _mm256_packus_epi16(sss0, zero);
    sss1 = _mm256_packus_epi16(sss1, zero);
    lineOut0[xx] = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss0, 0));
    lineOut1[xx] = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss0, 1));
    lineOut2[xx] = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss1, 0));
    lineOut3[xx] = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss1, 1));
  }
}

void ImagingResampleHorizontalConvolution8u(
    uint32_t* lineOut,
    uint32_t* lineIn,
    int xsize,
    int* xbounds,
    int16_t* kk,
    int kmax,
    int coefs_precision) {
  int xmin, xmax, x;
  int16_t* k;

  for (const auto xx : c10::irange(xsize)) {
    __m128i sss;
    xmin = xbounds[xx * 2 + 0];
    xmax = xbounds[xx * 2 + 1];
    k = &kk[xx * kmax];
    x = 0;

    if (xmax < 8) {
      sss = _mm_set1_epi32(1 << (coefs_precision - 1));
    } else {
      // Lower part will be added to higher, use only half of the error
      __m256i sss256 = _mm256_set1_epi32(1 << (coefs_precision - 2));

      for (; x < xmax - 7; x += 8) {
        __m256i pix, mmk, source;
        __m128i tmp = _mm_loadu_si128((__m128i*)&k[x]);
        __m256i ksource =
            _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

        // clang-format off
        source = _mm256_loadu_si256((__m256i*)&lineIn[x + xmin]);
        pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
          -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0,
          -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
        mmk = _mm256_shuffle_epi8(ksource, _mm256_set_epi8(
          11,10, 9,8, 11,10, 9,8, 11,10, 9,8, 11,10, 9,8,
          3,2, 1,0, 3,2, 1,0, 3,2, 1,0, 3,2, 1,0));
        sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix, mmk));

        pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
          -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8,
          -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8));
        mmk = _mm256_shuffle_epi8(ksource, _mm256_set_epi8(
          15,14, 13,12, 15,14, 13,12, 15,14, 13,12, 15,14, 13,12,
          7,6, 5,4, 7,6, 5,4, 7,6, 5,4, 7,6, 5,4));
        sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix, mmk));
        // clang-format on
      }

      for (; x < xmax - 3; x += 4) {
        __m256i pix, mmk, source;
        __m128i tmp = _mm_loadl_epi64((__m128i*)&k[x]);
        __m256i ksource =
            _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

        tmp = _mm_loadu_si128((__m128i*)&lineIn[x + xmin]);
        source = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

        // clang-format off
        pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
          -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8,
          -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
        mmk = _mm256_shuffle_epi8(ksource, _mm256_set_epi8(
          7,6, 5,4, 7,6, 5,4, 7,6, 5,4, 7,6, 5,4,
          3,2, 1,0, 3,2, 1,0, 3,2, 1,0, 3,2, 1,0));
        sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix, mmk));
        // clang-format on
      }

      sss = _mm_add_epi32(
          _mm256_extracti128_si256(sss256, 0),
          _mm256_extracti128_si256(sss256, 1));
    }

    for (; x < xmax - 1; x += 2) {
      __m128i mmk = _mm_set1_epi32(*(int32_t*)&k[x]);
      __m128i source = _mm_loadl_epi64((__m128i*)&lineIn[x + xmin]);
      __m128i pix = _mm_shuffle_epi8(
          source,
          _mm_set_epi8(-1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0));
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    for (; x < xmax; x++) {
      __m128i pix = mm_cvtepu8_epi32(&lineIn[x + xmin]);
      __m128i mmk = _mm_set1_epi32(k[x]);
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }
    sss = _mm_srai_epi32(sss, coefs_precision);
    sss = _mm_packs_epi32(sss, sss);
    lineOut[xx] = _mm_cvtsi128_si32(_mm_packus_epi16(sss, sss));
  }
}

void ImagingResampleVerticalConvolution8u(
    uint32_t* lineOut,
    uint32_t* imIn,
    int xmin,
    int xmax,
    int16_t* k,
    int coefs_precision,
    int xin) {
  int x;
  int xx = 0;
  int xsize = xin;

  __m128i initial = _mm_set1_epi32(1 << (coefs_precision - 1));
  __m256i initial_256 = _mm256_set1_epi32(1 << (coefs_precision - 1));

  for (; xx < xsize - 7; xx += 8) {
    __m256i sss0 = initial_256;
    __m256i sss1 = initial_256;
    __m256i sss2 = initial_256;
    __m256i sss3 = initial_256;
    x = 0;
    for (; x < xmax - 1; x += 2) {
      __m256i source, source1, source2;
      __m256i pix, mmk;

      // Load two coefficients at once
      mmk = _mm256_set1_epi32(*(int32_t*)&k[x]);

      // Load 2 lines
      //                           (__m256i *) &imIn->image32[x + xmin][xx]
      source1 = _mm256_loadu_si256((__m256i*)(imIn + (x + xmin) * xin + xx));
      //                           (__m256i *) &imIn->image32[x + 1 + xmin][xx]
      source2 =
          _mm256_loadu_si256((__m256i*)(imIn + (x + 1 + xmin) * xin + xx));

      source = _mm256_unpacklo_epi8(source1, source2);
      pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));
      pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));

      source = _mm256_unpackhi_epi8(source1, source2);
      pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
      sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix, mmk));
      pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
      sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix, mmk));
    }
    for (; x < xmax; x += 1) {
      __m256i source, source1, pix, mmk;
      mmk = _mm256_set1_epi32(k[x]);

      //                           (__m256i *) &imIn->image32[x + xmin][xx])
      source1 = _mm256_loadu_si256((__m256i*)(imIn + (x + xmin) * xin + xx));

      source = _mm256_unpacklo_epi8(source1, _mm256_setzero_si256());
      pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));
      pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));

      source = _mm256_unpackhi_epi8(source1, _mm256_setzero_si256());
      pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
      sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix, mmk));
      pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
      sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix, mmk));
    }
    sss0 = _mm256_srai_epi32(sss0, coefs_precision);
    sss1 = _mm256_srai_epi32(sss1, coefs_precision);
    sss2 = _mm256_srai_epi32(sss2, coefs_precision);
    sss3 = _mm256_srai_epi32(sss3, coefs_precision);

    sss0 = _mm256_packs_epi32(sss0, sss1);
    sss2 = _mm256_packs_epi32(sss2, sss3);
    sss0 = _mm256_packus_epi16(sss0, sss2);
    _mm256_storeu_si256((__m256i*)&lineOut[xx], sss0);
  }

  for (; xx < xsize - 1; xx += 2) {
    __m128i sss0 = initial; // left row
    __m128i sss1 = initial; // right row
    x = 0;
    for (; x < xmax - 1; x += 2) {
      __m128i source, source1, source2;
      __m128i pix, mmk;

      // Load two coefficients at once
      mmk = _mm_set1_epi32(*(int32_t*)&k[x]);

      // Load 2 lines
      //                        (__m128i *) &imIn->image32[x + xmin][xx])
      source1 = _mm_loadl_epi64((__m128i*)(imIn + (x + xmin) * xin + xx));
      //                        (__m128i *) &imIn->image32[x + 1 + xmin][xx]
      source2 = _mm_loadl_epi64((__m128i*)(imIn + (x + 1 + xmin) * xin + xx));

      source = _mm_unpacklo_epi8(source1, source2);
      pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
      sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
      pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
      sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));
    }
    for (; x < xmax; x += 1) {
      __m128i source, source1, pix, mmk;
      mmk = _mm_set1_epi32(k[x]);

      //                        (__m128i *) &imIn->image32[x + xmin][xx]);
      source1 = _mm_loadl_epi64((__m128i*)(imIn + (x + xmin) * xin + xx));

      source = _mm_unpacklo_epi8(source1, _mm_setzero_si128());
      pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
      sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
      pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
      sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));
    }
    sss0 = _mm_srai_epi32(sss0, coefs_precision);
    sss1 = _mm_srai_epi32(sss1, coefs_precision);

    sss0 = _mm_packs_epi32(sss0, sss1);
    sss0 = _mm_packus_epi16(sss0, sss0);
    _mm_storel_epi64((__m128i*)&lineOut[xx], sss0);
  }

  for (; xx < xsize; xx++) {
    __m128i sss = initial;
    x = 0;
    for (; x < xmax - 1; x += 2) {
      __m128i source, source1, source2;
      __m128i pix, mmk;

      // Load two coefficients at once
      mmk = _mm_set1_epi32(*(int32_t*)&k[x]);

      // Load 2 lines
      //                           *(int *) &imIn->image32[x + xmin][xx]
      source1 = _mm_cvtsi32_si128(*(int*)(imIn + (x + xmin) * xin + xx));
      //                          *(int *) &imIn->image32[x + 1 + xmin][xx]
      source2 = _mm_cvtsi32_si128(*(int*)(imIn + (x + 1 + xmin) * xin + xx));

      source = _mm_unpacklo_epi8(source1, source2);
      pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    for (; x < xmax; x++) {
      //                             &imIn->image32[x + xmin][xx]
      __m128i pix = mm_cvtepu8_epi32(imIn + (x + xmin) * xin + xx);
      __m128i mmk = _mm_set1_epi32(k[x]);
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }
    sss = _mm_srai_epi32(sss, coefs_precision);
    sss = _mm_packs_epi32(sss, sss);
    lineOut[xx] = _mm_cvtsi128_si32(_mm_packus_epi16(sss, sss));
  }
}

} // anonymous namespace
#endif // CPU_CAPABILITY_AVX2
