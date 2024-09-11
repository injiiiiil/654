#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/native/quantized/AffineQuantizerBase.h>
#include <c10/util/qint32.h>
#include <array>

#ifndef vl
#define  vl (2*__riscv_v_min_vlen/32)
#endif

namespace at {
namespace vec {
inline namespace CPU_CAPABILITY {

template <>
struct Vectorized<c10::qint32> {
 private:
    __at_align__ int32_t vals[vl];

 public:
    static constexpr int size() {
        return 8;
    }

    static constexpr int float_num_vecs() {
        return 1;
    }
    static constexpr int int_num_vecs() {
        return 1;
    }

    using float_vec_return_type = std::array<Vectorized<float>, 1>;
    using int_vec_return_type = std::array<Vectorized<c10::qint32>, 1>;
    using value_type = typename c10::qint32::underlying;

    Vectorized() : vals{0} {}
    Vectorized(vint32m2_t v) {
        __riscv_vse32_v_i32m2(vals, v, vl);
    }
    // Broadcast constructor
    Vectorized(const c10::qint32& val) {
        vint32m2_t vec_val = __riscv_vmv_v_x_i32m2(val.val_, vl);
        __riscv_vse32_v_i32m2(vals, vec_val, vl);
    }

    operator vint32m2_t() const {
        return __riscv_vle32_v_i32m2(vals, vl);
    }

    void store(void* ptr, int count = size()) const {
        std::memcpy(ptr, vals, count * sizeof(value_type));
    }

    static Vectorized<c10::qint32> loadu(const void* ptr, int count = size()) {
        Vectorized<c10::qint32> res;
        std::memcpy(
            res.vals,
            reinterpret_cast<const value_type*>(ptr),
            count * sizeof(value_type));
        return res;
    }

    float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> /*zero_point*/,
      Vectorized<float> scale_zp_premul) const {
        vint32m2_t i32_vec = __riscv_vle32_v_i32m2(vals, vl);
        vfloat32m2_t float_vals = __riscv_vfcvt_f_x_v_f32m2(i32_vec, vl);
        return {vec::fmadd(scale, Vectorized<float>(float_vals), scale_zp_premul)};
    }

    float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point) const {
        vint32m2_t i32_vec = __riscv_vle32_v_i32m2(vals, vl);
        vfloat32m2_t float_vals = __riscv_vfcvt_f_x_v_f32m2(i32_vec, vl);
        return {(Vectorized<float>(float_vals) - zero_point) * scale};
    }

    static Vectorized<c10::qint32> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
        Vectorized<c10::qint32> retval;
        auto rhs_data = (vfloat32m2_t)rhs[0];
        at::native::quantize_vec<c10::qint32, /*precision=*/32>(
            scale, zero_point, (float*)&rhs_data, (c10::qint32*)&retval.vals, 8);
        return retval;
    }

    Vectorized<c10::qint32> maximum(Vectorized<c10::qint32> b) const {
        return __riscv_vmax_vv_i32m2(
            __riscv_vle32_v_i32m2(vals, vl),
            __riscv_vle32_v_i32m2(b.vals, vl),
            vl);
    }

    Vectorized<c10::qint32> minimum(Vectorized<c10::qint32> b) const {
        return __riscv_vmin_vv_i32m2(
            __riscv_vle32_v_i32m2(vals, vl),
            __riscv_vle32_v_i32m2(b.vals, vl),
            vl);
    }

    Vectorized<c10::qint32> relu(Vectorized<c10::qint32> zero_point) const {
        return maximum(zero_point);
    }

    Vectorized<c10::qint32> relu6(
      Vectorized<c10::qint32> zero_point,
      Vectorized<c10::qint32> q_six) {
        return __riscv_vmin_vv_i32m2(
            __riscv_vmax_vv_i32m2(__riscv_vle32_v_i32m2(vals, vl), __riscv_vle32_v_i32m2(zero_point.vals, vl), vl),
            __riscv_vle32_v_i32m2(q_six.vals, vl),
            vl);
    }

    int_vec_return_type widening_subtract(Vectorized<c10::qint32> b) const {
        return {__riscv_vsub_vv_i32m2(__riscv_vle32_v_i32m2(vals, vl), b, vl)};
    }

    static Vectorized<c10::qint32> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
        vfloat32m2_t multiplier_v = __riscv_vfmv_v_f_f32m2(multiplier, vl);
        vint32m2_t zero_point_v = __riscv_vmv_v_x_i32m2(zero_point, vl);
        vfloat32m2_t scaled = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(inp[0], vl), multiplier_v, vl);
        vint32m2_t rounded = __riscv_vfcvt_x_f_v_i32m2(scaled, vl);
        return __riscv_vadd_vv_i32m2(rounded, zero_point_v, vl);
    }
};

template <>
Vectorized<c10::qint32> inline maximum(const Vectorized<c10::qint32>& a, const Vectorized<c10::qint32>& b) {
    return a.maximum(b);
}

template <>
Vectorized<c10::qint32> inline operator*(
    const Vectorized<c10::qint32>& a,
    const Vectorized<c10::qint32>& b) {
    return __riscv_vmul_vv_i32m2(a, b, vl);
}

template <>
Vectorized<c10::qint32> inline operator+(
    const Vectorized<c10::qint32>& a,
    const Vectorized<c10::qint32>& b) {
    return __riscv_vadd_vv_i32m2(a, b, vl);
}

}
}
}

#undef vl
