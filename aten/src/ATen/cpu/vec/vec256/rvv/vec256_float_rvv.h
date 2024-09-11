#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/irange.h>

#include <sleef.h>

#ifndef vl
#define  vl (2*__riscv_v_min_vlen/32)
#endif

namespace at::vec {
inline namespace CPU_CAPABILITY {

template <> class Vectorized<float> {
private:
  __at_align__ float values[vl];
public:
  using value_type = float;
  using size_type = int;
  static constexpr size_type size() {
    return 8;
  }
  Vectorized() : values{0.f} {}
  Vectorized(vfloat32m2_t v) {
    __riscv_vse32_v_f32m2(values, v, vl);
  }
  Vectorized(float val) {
    vfloat32m2_t vec_val = __riscv_vfmv_v_f_f32m2(val, vl);
    __riscv_vse32_v_f32m2(values, vec_val, vl);
  }
  Vectorized(float val0, float val1, float val2, float val3,
    float val4, float val5, float val6, float val7) :
    values{val0, val1, val2, val3, val4, val5, val6, val7} {}

  operator vfloat32m2_t() const {
    return __riscv_vle32_v_f32m2(values, vl);
  }

  template <int64_t mask>
  static Vectorized<float> blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    Vectorized<float> vec;
    vfloat32m2_t a_vec = __riscv_vle32_v_f32m2(a.values, vl);
    vfloat32m2_t b_vec = __riscv_vle32_v_f32m2(b.values, vl);
    vint64m1_t  mask_vec = __riscv_vmv_v_x_i64m1(mask, 1);
    vbool16_t bool_vec  = __riscv_vreinterpret_v_i64m1_b16(mask_vec);
    vfloat32m2_t res = __riscv_vmerge_vvm_f32m2(a_vec, b_vec, bool_vec, vl);
    __riscv_vse32_v_f32m2(vec.values, res, vl);
    return vec;
  }

  static Vectorized<float> blendv(const Vectorized<float>& a, const Vectorized<float>& b,
                              const Vectorized<float>& mask) {
    Vectorized<float> vec;
    vuint32m2_t mask_u32 = __riscv_vreinterpret_v_f32m2_u32m2(__riscv_vle32_v_f32m2(mask.values, vl));
    vuint32m2_t and_u32 = __riscv_vand_vx_u32m2(mask_u32, 0x01, vl);
    vbool16_t  bool_vec = __riscv_vmseq_vx_u32m2_b16(and_u32, 0x01, vl);
    vfloat32m2_t res = __riscv_vmerge_vvm_f32m2(
      __riscv_vle32_v_f32m2(a.values, vl),
      __riscv_vle32_v_f32m2(b.values, vl),
      bool_vec,
      vl);
    __riscv_vse32_v_f32m2(vec.values, res, vl);
    return vec;
  }

  template<typename step_t>
  static Vectorized<float> arange(float base = 0.f, step_t step = static_cast<step_t>(1)) {
    const Vectorized<float> base_vec(base);
    const Vectorized<float> step_vec(step);
    const Vectorized<float> step_sizes(0, 1, 2, 3, 4, 5, 6, 7);
    return fmadd(step_sizes, step_vec, base_vec);
  }

  static Vectorized<float> set(
      const Vectorized<float>& a,
      const Vectorized<float>& b,
      size_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<1>(a, b);
      case 2:
        return blend<3>(a, b);
      case 3:
        return blend<7>(a, b);
      case 4:
        return blend<15>(a, b);
      case 5:
        return blend<31>(a, b);
      case 6:
        return blend<63>(a, b);
      case 7:
        return blend<127>(a, b);
    }

    return b;
  }

  static Vectorized<float> loadu(const void* ptr, int64_t count = size()) {
    Vectorized<float> res;
    std::memcpy(
        res.values,
        reinterpret_cast<const float*>(ptr),
        count * sizeof(float));
    return res;
  }

  void store(void* ptr, int64_t count = size()) const {
    std::memcpy(ptr, values, count * sizeof(float));
  }

  const float& operator[](int idx) const = delete;
  float& operator[](int idx) = delete;

  int zero_mask() const {
    __at_align__ float tmp[size()];
    store(tmp);
    int mask = 0;
    for (int i = 0; i < size(); ++ i) {
      if (tmp[i] == 0.f) {
        mask |= (1 << i);
      }
    }
    return mask;
  }

  Vectorized<float> isnan() const {
    Vectorized<float> res;
    vfloat32m2_t tmp = __riscv_vle32_v_f32m2(values, vl);
    vuint32m2_t classify= __riscv_vfclass_v_u32m2(tmp, vl);
    vbool16_t isSNaN = __riscv_vmseq_vx_u32m2_b16(classify, 0x100, vl);
    vbool16_t isQNaN = __riscv_vmseq_vx_u32m2_b16(classify, 0x200, vl);
    vbool16_t isNaN = __riscv_vmor_mm_b16(isSNaN, isQNaN, vl);
    vuint32m2_t zero_vec = __riscv_vmv_v_x_u32m2(0, vl);
    vuint32m2_t vec_u32 = __riscv_vmerge_vxm_u32m2(zero_vec, 0xFFFFFFFF, isNaN, vl);
    vfloat32m2_t vec_f32 = __riscv_vreinterpret_v_u32m2_f32m2(vec_u32);
    __riscv_vse32_v_f32m2(res.values, vec_f32, vl);
    return res;
  }

  bool has_inf_nan() const {
    __at_align__ float tmp[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      if(_isnan(tmp[i]) || _isinf(tmp[i])) {
        return true;
      }
    }
    return false;
  }

  Vectorized<float> map(float (*const f)(float)) const {
    __at_align__ float tmp[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }

  Vectorized<float> abs() const {
    vfloat32m2_t vec = __riscv_vle32_v_f32m2(values, vl);
    return Vectorized<float>(__riscv_vfabs_v_f32m2(vec, vl));
  }
  Vectorized<float> angle() const {
    auto zero = Vectorized<float>(0);
    auto pi = Vectorized<float>(c10::pi<float>);
    auto tmp = blendv(zero, pi, *this < zero);
    return blendv(tmp, *this, isnan());
  }
  Vectorized<float> real() const {
    return *this;
  }
  Vectorized<float> imag() const {
    return Vectorized<float>(0.f);
  }
  Vectorized<float> conj() const {
    return *this;
  }
  Vectorized<float> acos() const {
    return Vectorized<float>(Sleef_acosfx_u10rvvm2(__riscv_vle32_v_f32m2(values, vl)));
  }
  Vectorized<float> acosh() const {
    return Vectorized<float>(Sleef_acoshfx_u10rvvm2(__riscv_vle32_v_f32m2(values, vl)));
  }
  Vectorized<float> asin() const {
    return Vectorized<float>(Sleef_asinfx_u10rvvm2(__riscv_vle32_v_f32m2(values, vl)));
  }
  Vectorized<float> atan() const {
    return Vectorized<float>(Sleef_atanfx_u10rvvm2(__riscv_vle32_v_f32m2(values, vl)));
  }
  Vectorized<float> atanh() const {
    return Vectorized<float>(Sleef_atanhfx_u10rvvm2(__riscv_vle32_v_f32m2(values, vl)));
  }
  Vectorized<float> atan2(const Vectorized<float> &exp) const {
    return Vectorized<float>(Sleef_atan2fx_u10rvvm2(__riscv_vle32_v_f32m2(values, vl), __riscv_vle32_v_f32m2(exp.values, vl)));
  }
  Vectorized<float> copysign(const Vectorized<float> &sign) const {
    return Vectorized<float>(Sleef_copysignfx_rvvm2(__riscv_vle32_v_f32m2(values, vl), __riscv_vle32_v_f32m2(sign.values, vl)));
  }
  Vectorized<float> erf() const {
    return Vectorized<float>(Sleef_erffx_u10rvvm2(__riscv_vle32_v_f32m2(values, vl)));
  }
  Vectorized<float> erfc() const {
    return Vectorized<float>(Sleef_erfcfx_u15rvvm2(__riscv_vle32_v_f32m2(values, vl)));
  }
  Vectorized<float> erfinv() const {
    return map(calc_erfinv);
  }
  Vectorized<float> exp() const {
    return Vectorized<float>(Sleef_expfx_u10rvvm2(__riscv_vle32_v_f32m2(values, vl)));
  }
  Vectorized<float> exp2() const {
    return Vectorized<float>(Sleef_exp2fx_u10rvvm2(__riscv_vle32_v_f32m2(values, vl)));
  }
  Vectorized<float> expm1() const {
    return Vectorized<float>(Sleef_expm1fx_u10rvvm2(__riscv_vle32_v_f32m2(values, vl)));
  }
  Vectorized<float> exp_u20() const {
    return exp();
  }
  Vectorized<float> fmod(const Vectorized<float>& q) const {
    return Vectorized<float>(Sleef_fmodfx_rvvm2(__riscv_vle32_v_f32m2(values, vl), __riscv_vle32_v_f32m2(q.values, vl)));
  }
  Vectorized<float> hypot(const Vectorized<float> &b) const {
    return Vectorized<float>(Sleef_hypotfx_u05rvvm2(__riscv_vle32_v_f32m2(values, vl), __riscv_vle32_v_f32m2(b.values, vl)));
  }
  Vectorized<float> i0() const {
    return map(calc_i0);
  }
  Vectorized<float> i0e() const {
    return map(calc_i0e);
  }
  Vectorized<float> digamma() const {
    return map(calc_digamma);
  }
  Vectorized<float> igamma(const Vectorized<float> &x) const {
    __at_align__ float tmp[size()];
    __at_align__ float tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (const auto i : c10::irange(size())) {
      tmp[i] = calc_igamma(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<float> igammac(const Vectorized<float> &x) const {
    __at_align__ float tmp[size()];
    __at_align__ float tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (const auto i : c10::irange(size())) {
      tmp[i] = calc_igammac(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<float> log() const {
    return Vectorized<float>(Sleef_logfx_u10rvvm2(__riscv_vle32_v_f32m2(values, vl)));
  }
  Vectorized<float> log10() const {
    return Vectorized<float>(Sleef_log10fx_u10rvvm2(__riscv_vle32_v_f32m2(values, vl)));
  }
  Vectorized<float> log1p() const {
    return Vectorized<float>(Sleef_log1pfx_u10rvvm2(__riscv_vle32_v_f32m2(values, vl)));
  }
  Vectorized<float> log2() const {
    return Vectorized<float>(Sleef_log2fx_u10rvvm2(__riscv_vle32_v_f32m2(values, vl)));
  }
  Vectorized<float> nextafter(const Vectorized<float> &b) const {
    return Vectorized<float>(Sleef_nextafterfx_rvvm2(__riscv_vle32_v_f32m2(values, vl), __riscv_vle32_v_f32m2(b.values, vl)));
  }
  Vectorized<float> frac() const;
  Vectorized<float> sin() const {
    return Vectorized<float>(Sleef_sinfx_u10rvvm2(__riscv_vle32_v_f32m2(values, vl)));
  }
  Vectorized<float> sinh() const {
    return Vectorized<float>(Sleef_sinhfx_u10rvvm2(__riscv_vle32_v_f32m2(values, vl)));
  }
  Vectorized<float> cos() const {
    return Vectorized<float>(Sleef_cosfx_u10rvvm2(__riscv_vle32_v_f32m2(values, vl)));
  }
  Vectorized<float> cosh() const {
    return Vectorized<float>(Sleef_coshfx_u10rvvm2(__riscv_vle32_v_f32m2(values, vl)));
  }
  Vectorized<float> ceil() const {
    return map(at::native::ceil_impl);
  }
  Vectorized<float> floor() const {
    return map(at::native::floor_impl);
  }
  Vectorized<float> neg() const {
    return Vectorized<float>(__riscv_vfneg_v_f32m2(__riscv_vle32_v_f32m2(values, vl), vl));
  }
  Vectorized<float> round() const {
    return map(at::native::round_impl);
  }
  Vectorized<float> tan() const {
    return Vectorized<float>(Sleef_tanfx_u10rvvm2(__riscv_vle32_v_f32m2(values, vl)));
  }
  Vectorized<float> tanh() const {
    return Vectorized<float>(Sleef_tanhfx_u10rvvm2(__riscv_vle32_v_f32m2(values, vl)));
  }
  Vectorized<float> trunc() const {
    return map(at::native::trunc_impl);
  }
  Vectorized<float> lgamma() const {
    return Vectorized<float>(Sleef_lgammafx_u10rvvm2(__riscv_vle32_v_f32m2(values, vl)));
  }
  Vectorized<float> sqrt() const {
    return Vectorized<float>(
         __riscv_vfsqrt_v_f32m2(__riscv_vle32_v_f32m2(values, vl), vl));
  }
  Vectorized<float> reciprocal() const {
    vfloat32m2_t res = __riscv_vfdiv_vv_f32m2(
      __riscv_vfmv_v_f_f32m2(1.0f, vl),
      __riscv_vle32_v_f32m2(values, vl),
      vl);
    return Vectorized<float>(res);
  }
  Vectorized<float> rsqrt() const {
    return this->sqrt().reciprocal();
  }
  Vectorized<float> pow(const Vectorized<float> &exp) const {
    return Vectorized<float>(Sleef_powfx_u10rvvm2(__riscv_vle32_v_f32m2(values, vl), __riscv_vle32_v_f32m2(exp.values, vl)));
  }

  Vectorized<float> operator==(const Vectorized<float>& other) const {
    vbool16_t cmp_res = __riscv_vmfeq_vv_f32m2_b16(
      __riscv_vle32_v_f32m2(values, vl),
      __riscv_vle32_v_f32m2(other.values, vl),
      vl);
    vuint32m2_t merge_res = __riscv_vmerge_vvm_u32m2(
      __riscv_vmv_v_x_u32m2(0x0, vl),
      __riscv_vmv_v_x_u32m2(UINT32_MAX, vl),
      cmp_res,
      vl);
    vfloat32m2_t res = __riscv_vreinterpret_v_u32m2_f32m2(merge_res);
    return Vectorized<float>(res);
  }

  Vectorized<float> operator!=(const Vectorized<float>& other) const {
    vbool16_t cmp_res = __riscv_vmfeq_vv_f32m2_b16(
      __riscv_vle32_v_f32m2(values, vl),
      __riscv_vle32_v_f32m2(other.values, vl),
      vl);
    vuint32m2_t merge_res = __riscv_vmerge_vvm_u32m2(
      __riscv_vmv_v_x_u32m2(0x0, vl),
      __riscv_vmv_v_x_u32m2(UINT32_MAX, vl),
      cmp_res,
      vl);
    vfloat32m2_t res = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vnot_v_u32m2(merge_res, vl));
    return Vectorized<float>(res);
  }

  Vectorized<float> operator<(const Vectorized<float>& other) const {
    vbool16_t cmp_res = __riscv_vmflt_vv_f32m2_b16(
      __riscv_vle32_v_f32m2(values, vl),
      __riscv_vle32_v_f32m2(other.values, vl),
      vl);
    vuint32m2_t merge_res = __riscv_vmerge_vvm_u32m2(
      __riscv_vmv_v_x_u32m2(0x0, vl),
      __riscv_vmv_v_x_u32m2(UINT32_MAX, vl),
      cmp_res,
      vl);
    vfloat32m2_t res = __riscv_vreinterpret_v_u32m2_f32m2(merge_res);
    return Vectorized<float>(res);
  }

  Vectorized<float> operator<=(const Vectorized<float>& other) const {
    vbool16_t cmp_res = __riscv_vmfle_vv_f32m2_b16(
      __riscv_vle32_v_f32m2(values, vl),
      __riscv_vle32_v_f32m2(other.values, vl),
      vl);
    vuint32m2_t merge_res = __riscv_vmerge_vvm_u32m2(
      __riscv_vmv_v_x_u32m2(0x0, vl),
      __riscv_vmv_v_x_u32m2(UINT32_MAX, vl),
      cmp_res,
      vl);
    vfloat32m2_t res = __riscv_vreinterpret_v_u32m2_f32m2(merge_res);
    return Vectorized<float>(res);
  }

  Vectorized<float> operator>(const Vectorized<float>& other) const {
    vbool16_t cmp_res = __riscv_vmfgt_vv_f32m2_b16(
      __riscv_vle32_v_f32m2(values, vl),
      __riscv_vle32_v_f32m2(other.values, vl),
      vl);
    vuint32m2_t merge_res = __riscv_vmerge_vvm_u32m2(
      __riscv_vmv_v_x_u32m2(0x0, vl),
      __riscv_vmv_v_x_u32m2(UINT32_MAX, vl),
      cmp_res,
      vl);
    vfloat32m2_t res = __riscv_vreinterpret_v_u32m2_f32m2(merge_res);
    return Vectorized<float>(res);
  }

  Vectorized<float> operator>=(const Vectorized<float>& other) const {
    vbool16_t cmp_res = __riscv_vmfge_vv_f32m2_b16(
      __riscv_vle32_v_f32m2(values, vl),
      __riscv_vle32_v_f32m2(other.values, vl),
      vl);
    vuint32m2_t merge_res = __riscv_vmerge_vvm_u32m2(
      __riscv_vmv_v_x_u32m2(0x0, vl),
      __riscv_vmv_v_x_u32m2(UINT32_MAX, vl),
      cmp_res,
      vl);
    vfloat32m2_t res = __riscv_vreinterpret_v_u32m2_f32m2(merge_res);
    return Vectorized<float>(res);
  }

  Vectorized<float> eq(const Vectorized<float>& other) const;
  Vectorized<float> ne(const Vectorized<float>& other) const;
  Vectorized<float> gt(const Vectorized<float>& other) const;
  Vectorized<float> ge(const Vectorized<float>& other) const;
  Vectorized<float> lt(const Vectorized<float>& other) const;
  Vectorized<float> le(const Vectorized<float>& other) const;
};

template <>
Vectorized<float> inline operator+(const Vectorized<float>& a, const Vectorized<float>& b) {
  return __riscv_vfadd_vv_f32m2(a, b, vl);
}

template <>
Vectorized<float> inline operator-(const Vectorized<float>& a, const Vectorized<float>& b) {
  return __riscv_vfsub_vv_f32m2(a, b, vl);
}

template <>
Vectorized<float> inline operator*(const Vectorized<float>& a, const Vectorized<float>& b) {
  return __riscv_vfmul_vv_f32m2(a, b, vl);
}

template <>
Vectorized<float> inline operator/(const Vectorized<float>& a, const Vectorized<float>& b) {
  return __riscv_vfdiv_vv_f32m2(a, b, vl);
}

inline Vectorized<float> Vectorized<float>::frac() const {
  return *this - this->trunc();
}

template <>
Vectorized<float> inline maximum(const Vectorized<float>& a, const Vectorized<float>& b) {
  vbool16_t mask = __riscv_vmand_mm_b16(
    __riscv_vmfeq_vv_f32m2_b16(a, a, vl),
    __riscv_vmfeq_vv_f32m2_b16(b, b, vl),
    vl);
  vfloat32m2_t max_res = __riscv_vfmax_vv_f32m2(a, b, vl);
  vfloat32m2_t res = __riscv_vmerge_vvm_f32m2(__riscv_vfmv_v_f_f32m2(NAN, vl), max_res, mask, vl);
  return Vectorized<float>(res);
}

template <>
Vectorized<float> inline minimum(const Vectorized<float>& a, const Vectorized<float>& b) {
  vbool16_t mask = __riscv_vmand_mm_b16(
    __riscv_vmfeq_vv_f32m2_b16(a, a, vl),
    __riscv_vmfeq_vv_f32m2_b16(b, b, vl),
    vl);
  vfloat32m2_t min_res = __riscv_vfmin_vv_f32m2(a, b, vl);
  vfloat32m2_t res = __riscv_vmerge_vvm_f32m2(__riscv_vfmv_v_f_f32m2(NAN, vl), min_res, mask, vl);
  return Vectorized<float>(res);
}

template <>
Vectorized<float> inline clamp(const Vectorized<float>& a, const Vectorized<float>& min, const Vectorized<float>& max) {
  return minimum(max, maximum(min, a));
}

template <>
Vectorized<float> inline clamp_max(const Vectorized<float>& a, const Vectorized<float>& max) {
  return minimum(max, a);
}

template <>
Vectorized<float> inline clamp_min(const Vectorized<float>& a, const Vectorized<float>& min) {
  return maximum(min, a);
}

template <>
Vectorized<float> inline operator&(const Vectorized<float>& a, const Vectorized<float>& b) {
  vfloat32m2_t res = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vand_vv_u32m2(
    __riscv_vreinterpret_v_f32m2_u32m2(a),
    __riscv_vreinterpret_v_f32m2_u32m2(b),
    vl));
  return Vectorized<float>(res);
}

template <>
Vectorized<float> inline operator|(const Vectorized<float>& a, const Vectorized<float>& b) {
  vfloat32m2_t res = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vor_vv_u32m2(
    __riscv_vreinterpret_v_f32m2_u32m2(a),
    __riscv_vreinterpret_v_f32m2_u32m2(b),
    vl));
  return Vectorized<float>(res);
}

template <>
Vectorized<float> inline operator^(const Vectorized<float>& a, const Vectorized<float>& b) {
  vfloat32m2_t res = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vxor_vv_u32m2(
    __riscv_vreinterpret_v_f32m2_u32m2(a),
    __riscv_vreinterpret_v_f32m2_u32m2(b),
    vl));
  return Vectorized<float>(res);
}

inline Vectorized<float> Vectorized<float>::eq(const Vectorized<float>& other) const {
  return (*this == other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::ne(const Vectorized<float>& other) const {
  return (*this != other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::gt(const Vectorized<float>& other) const {
  return (*this > other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::ge(const Vectorized<float>& other) const {
  return (*this >= other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::lt(const Vectorized<float>& other) const {
  return (*this < other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::le(const Vectorized<float>& other) const {
  return (*this <= other) & Vectorized<float>(1.0f);
}

template <>
inline void convert(const float* src, int32_t* dst, int64_t n) {
  int64_t i;
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<float>::size()); i += Vectorized<float>::size()) {
    __riscv_vse32_v_i32m2(dst + i, __riscv_vfcvt_rtz_x_f_v_i32m2(__riscv_vle32_v_f32m2(src + i, vl), vl), vl);
  }
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (; i < n; i++) {
    dst[i] = static_cast<int32_t>(src[i]);
  }
}

template <>
inline void convert(const int32_t* src, float* dst, int64_t n) {
  int64_t i;
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<float>::size()); i += Vectorized<float>::size()) {
    __riscv_vse32_v_f32m2(dst + i, __riscv_vfcvt_f_x_v_f32m2(__riscv_vle32_v_i32m2(src + i, vl), vl), vl);
  }
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (; i < n; i++) {
    dst[i] = static_cast<float>(src[i]);
  }
}

template <>
Vectorized<float> inline fmadd(const Vectorized<float>& a, const Vectorized<float>& b, const Vectorized<float>& c) {
  return __riscv_vfmacc_vv_f32m2(c, a, b, vl);
}

template <>
Vectorized<float> inline fmsub(const Vectorized<float>& a, const Vectorized<float>& b, const Vectorized<float>& c) {
  return __riscv_vfnmsac_vv_f32m2(c, a, b, vl);
}

}} // namespace at::vec::CPU_CAPABILITY

#undef vl
