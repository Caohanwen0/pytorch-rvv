#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec256/rvv/rvv_helper.h>
#include <c10/util/irange.h>

#include <sleef.h>
#include <cstdint>

namespace at::vec {
inline namespace CPU_CAPABILITY {

template <> class Vectorized<double> {
private:
  fixed_vfloat64m2_t values;
public:
  using value_type = double;
  using size_type = int;
  static constexpr size_type size() {
    return 4;  // m2寄存器可以存储4个double值
  }
  Vectorized() {}
  Vectorized(vfloat64m2_t v) : values(v) {}
  Vectorized(double val) {
    values = __riscv_vfmv_v_f_f64m2(val, VFLOAT64_VL);
  }
  Vectorized(double val0, double val1, double val2, double val3) {
    double v[] = {val0, val1, val2, val3};
    values = __riscv_vle64_v_f64m2(reinterpret_cast<double*>(v), VFLOAT64_VL);
  }

  operator vfloat64m2_t() const {
    return values;
  }

  template <int64_t mask>
  static Vectorized<double> blend(const Vectorized<double>& a, const Vectorized<double>& b) {
    vint64m1_t  mask_vec = __riscv_vmv_v_x_i64m1(mask, 1);
    vbool32_t bool_vec  = __riscv_vreinterpret_v_i64m1_b32(mask_vec);
    return __riscv_vmerge_vvm_f64m2(a.values, b.values, bool_vec, VFLOAT64_VL);
  }

  static Vectorized<double> blendv(const Vectorized<double>& a, const Vectorized<double>& b,
                              const Vectorized<double>& mask) {
    vuint64m2_t mask_u64 = __riscv_vreinterpret_v_f64m2_u64m2(mask.values);
    vuint64m2_t and_u64 = __riscv_vand_vx_u64m2(mask_u64, 0x01, VFLOAT64_VL);
    vbool32_t  bool_vec = __riscv_vmseq_vx_u64m2_b32(and_u64, 0x01, VFLOAT64_VL);
    return __riscv_vmerge_vvm_f64m2(a.values, b.values, bool_vec, VFLOAT64_VL);
  }

  template<typename step_t>
  static Vectorized<double> arange(double base = 0., step_t step = static_cast<step_t>(1)) {
    const Vectorized<double> base_vec(base);
    const Vectorized<double> step_vec(step);
    const Vectorized<double> step_sizes(0, 1, 2, 3);
    return fmadd(step_sizes, step_vec, base_vec);
  }

  static Vectorized<double> set(
      const Vectorized<double>& a,
      const Vectorized<double>& b,
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
    }
    return b;
  }

  static Vectorized<double> loadu(const void* ptr, int64_t count = size()) {
#ifdef RVV_SUPPORT_UNALIGN
    if (count == size()) {
      return __riscv_vle64_v_f64m2(reinterpret_cast<const double*>(ptr), VFLOAT64_VL);
    }
    else {
      vfloat64m2_t zero_vec = __riscv_vfmv_v_f_f64m2(0., VFLOAT64_VL);
      return __riscv_vle64_v_f64m2_tu(zero_vec, reinterpret_cast<const double*>(ptr), count);
    }
#else
    // If the address of ptr is not aligned, the performance will be very slow.
    if (reinterpret_cast<uintptr_t>(ptr) & 0x7) {
      __at_align__ double tmp_values[size()];
      for (const auto i : c10::irange(size())) {
        tmp_values[i] = 0.0;
      }
      std::memcpy(
          tmp_values, reinterpret_cast<const double*>(ptr), count * sizeof(double));
      return __riscv_vle64_v_f64m2(tmp_values, VFLOAT64_VL);
    } else {
      if (count == size()) {
        return __riscv_vle64_v_f64m2(reinterpret_cast<const double*>(ptr), VFLOAT64_VL);
      }
      else {
        vfloat64m2_t zero_vec = __riscv_vfmv_v_f_f64m2(0., VFLOAT64_VL);
        return __riscv_vle64_v_f64m2_tu(zero_vec, reinterpret_cast<const double*>(ptr), count);
      }
    }
#endif
  }

  void store(void* ptr, int64_t count = size()) const {
#ifdef RVV_SUPPORT_UNALIGN
    __riscv_vse64_v_f64m2(reinterpret_cast<double*>(ptr), values, count);
#else
    // If the address of ptr is not aligned, the performance will be very slow.
    if (reinterpret_cast<uintptr_t>(ptr) & 0x7) {
      double tmp_values[size()];
      __riscv_vse64_v_f64m2(reinterpret_cast<double*>(tmp_values), values, VFLOAT64_VL);
      std::memcpy(ptr, tmp_values, count * sizeof(double));
    } else {
      __riscv_vse64_v_f64m2(reinterpret_cast<double*>(ptr), values, count);
    }
#endif
  }

  const double& operator[](int idx) const = delete;
  double& operator[](int idx) = delete;

  int zero_mask() const {
    __at_align__ double tmp[size()];
    store(tmp);
    int mask = 0;
    for (int i = 0; i < size(); ++ i) {
      if (tmp[i] == 0.) {
        mask |= (1 << i);
      }
    }
    return mask;
  }

  Vectorized<double> isnan() const {
    vuint64m2_t classify= __riscv_vfclass_v_u64m2(values, VFLOAT64_VL);
    vbool32_t isSNaN = __riscv_vmseq_vx_u64m2_b32(classify, 0x100, VFLOAT64_VL);
    vbool32_t isQNaN = __riscv_vmseq_vx_u64m2_b32(classify, 0x200, VFLOAT64_VL);
    vbool32_t isNaN = __riscv_vmor_mm_b32(isSNaN, isQNaN, VFLOAT64_VL);
    vuint64m2_t zero_vec = __riscv_vmv_v_x_u64m2(0, VFLOAT64_VL);
    vuint64m2_t vec_u64 = __riscv_vmerge_vxm_u64m2(zero_vec, 0xFFFFFFFFFFFFFFFF, isNaN, VFLOAT64_VL);
    return __riscv_vreinterpret_v_u64m2_f64m2(vec_u64);
  }

  bool has_inf_nan() const {
    __at_align__ double tmp[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      if(_isnan(tmp[i]) || _isinf(tmp[i])) {
        return true;
      }
    }
    return false;
  }

  Vectorized<double> map(double (*const f)(double)) const {
    __at_align__ double tmp[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }

  Vectorized<double> abs() const {
    return Vectorized<double>(__riscv_vfabs_v_f64m2(values, VFLOAT64_VL));
  }
  Vectorized<double> angle() const {
    auto zero = Vectorized<double>(0);
    auto pi = Vectorized<double>(c10::pi<double>);
    auto tmp = blendv(zero, pi, *this < zero);
    return blendv(tmp, *this, isnan());
  }
  Vectorized<double> real() const {
    return *this;
  }
  Vectorized<double> imag() const {
    return Vectorized<double>(0.);
  }
  Vectorized<double> conj() const {
    return *this;
  }
  Vectorized<double> acos() const {
    return Vectorized<double>(Sleef_acosdx_u10rvvm2(values));
  }
  Vectorized<double> acosh() const {
    return Vectorized<double>(Sleef_acoshdx_u10rvvm2(values));
  }
  Vectorized<double> asin() const {
    return Vectorized<double>(Sleef_asindx_u10rvvm2(values));
  }
  Vectorized<double> atan() const {
    return Vectorized<double>(Sleef_atandx_u10rvvm2(values));
  }
  Vectorized<double> atanh() const {
    return Vectorized<double>(Sleef_atanhdx_u10rvvm2(values));
  }
  Vectorized<double> atan2(const Vectorized<double> &exp) const {
    return Vectorized<double>(Sleef_atan2dx_u10rvvm2(values, exp.values));
  }
  Vectorized<double> copysign(const Vectorized<double> &sign) const {
    return Vectorized<double>(Sleef_copysigndx_rvvm2(values, sign.values));
  }
  Vectorized<double> erf() const {
    return Vectorized<double>(Sleef_erfdx_u10rvvm2(values));
  }
  Vectorized<double> erfc() const {
    return Vectorized<double>(Sleef_erfcdx_u15rvvm2(values));
  }
  Vectorized<double> erfinv() const {
    return map(calc_erfinv);
  }
  Vectorized<double> exp() const {
    return Vectorized<double>(Sleef_expdx_u10rvvm2(values));
  }
  Vectorized<double> exp2() const {
    return Vectorized<double>(Sleef_exp2dx_u10rvvm2(values));
  }
  Vectorized<double> expm1() const {
    return Vectorized<double>(Sleef_expm1dx_u10rvvm2(values));
  }
  Vectorized<double> exp_u20() const {
    return exp();
  }
  Vectorized<double> fmod(const Vectorized<double>& q) const {
    return Vectorized<double>(Sleef_fmoddx_rvvm2(values, q.values));
  }
  Vectorized<double> hypot(const Vectorized<double> &b) const {
    return Vectorized<double>(Sleef_hypotdx_u05rvvm2(values, b.values));
  }
  Vectorized<double> i0() const {
    return map(calc_i0);
  }
  Vectorized<double> i0e() const {
    return map(calc_i0e);
  }
  Vectorized<double> digamma() const {
    return map(calc_digamma);
  }
  Vectorized<double> igamma(const Vectorized<double> &x) const {
    __at_align__ double tmp[size()];
    __at_align__ double tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (const auto i : c10::irange(size())) {
      tmp[i] = calc_igamma(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<double> igammac(const Vectorized<double> &x) const {
    __at_align__ double tmp[size()];
    __at_align__ double tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (const auto i : c10::irange(size())) {
      tmp[i] = calc_igammac(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<double> log() const {
    return Vectorized<double>(Sleef_logdx_u10rvvm2(values));
  }
  Vectorized<double> log10() const {
    return Vectorized<double>(Sleef_log10dx_u10rvvm2(values));
  }
  Vectorized<double> log1p() const {
    return Vectorized<double>(Sleef_log1pdx_u10rvvm2(values));
  }
  Vectorized<double> log2() const {
    return Vectorized<double>(Sleef_log2dx_u10rvvm2(values));
  }
  Vectorized<double> nextafter(const Vectorized<double> &b) const {
    return Vectorized<double>(Sleef_nextafterdx_rvvm2(values, b.values));
  }
  Vectorized<double> frac() const;
  Vectorized<double> sin() const {
    return Vectorized<double>(Sleef_sindx_u10rvvm2(values));
  }
  Vectorized<double> sinh() const {
    return Vectorized<double>(Sleef_sinhdx_u10rvvm2(values));
  }
  Vectorized<double> cos() const {
    return Vectorized<double>(Sleef_cosdx_u10rvvm2(values));
  }
  Vectorized<double> cosh() const {
    return Vectorized<double>(Sleef_coshdx_u10rvvm2(values));
  }
  Vectorized<double> ceil() const {
    return map(at::native::ceil_impl);
  }
  Vectorized<double> floor() const {
    return map(at::native::floor_impl);
  }
  Vectorized<double> neg() const {
    return Vectorized<double>(__riscv_vfneg_v_f64m2(values, VFLOAT64_VL));
  }
  Vectorized<double> round() const {
    return map(at::native::round_impl);
  }
  Vectorized<double> tan() const {
    return Vectorized<double>(Sleef_tandx_u10rvvm2(values));
  }
  Vectorized<double> tanh() const {
    return Vectorized<double>(Sleef_tanhdx_u10rvvm2(values));
  }
  Vectorized<double> trunc() const {
    return map(at::native::trunc_impl);
  }
  Vectorized<double> lgamma() const {
    return Vectorized<double>(Sleef_lgammadx_u10rvvm2(values));
  }
  Vectorized<double> sqrt() const {
    return Vectorized<double>(
         __riscv_vfsqrt_v_f64m2(values, VFLOAT64_VL));
  }
  Vectorized<double> reciprocal() const {
    vfloat64m2_t res = __riscv_vfdiv_vv_f64m2(
      __riscv_vfmv_v_f_f64m2(1.0, VFLOAT64_VL),
      values,
      VFLOAT64_VL);
    return Vectorized<double>(res);
  }
  Vectorized<double> rsqrt() const {
    return this->sqrt().reciprocal();
  }
  Vectorized<double> pow(const Vectorized<double> &exp) const {
    return Vectorized<double>(Sleef_powdx_u10rvvm2(values, exp.values));
  }

  Vectorized<double> operator==(const Vectorized<double>& other) const {
    vbool32_t cmp_res = __riscv_vmfeq_vv_f64m2_b32(values, other.values, VFLOAT64_VL);
    vuint64m2_t merge_res = __riscv_vmerge_vvm_u64m2(
      __riscv_vmv_v_x_u64m2(0x0, VFLOAT64_VL),
      __riscv_vmv_v_x_u64m2(UINT64_MAX, VFLOAT64_VL),
      cmp_res,
      VFLOAT64_VL);
    vfloat64m2_t res = __riscv_vreinterpret_v_u64m2_f64m2(merge_res);
    return Vectorized<double>(res);
  }

  Vectorized<double> operator!=(const Vectorized<double>& other) const {
    vbool32_t cmp_res = __riscv_vmfeq_vv_f64m2_b32(values, other.values, VFLOAT64_VL);
    vuint64m2_t merge_res = __riscv_vmerge_vvm_u64m2(
      __riscv_vmv_v_x_u64m2(0x0, VFLOAT64_VL),
      __riscv_vmv_v_x_u64m2(UINT64_MAX, VFLOAT64_VL),
      cmp_res,
      VFLOAT64_VL);
    vfloat64m2_t res = __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vnot_v_u64m2(merge_res, VFLOAT64_VL));
    return Vectorized<double>(res);
  }

  Vectorized<double> operator<(const Vectorized<double>& other) const {
    vbool32_t cmp_res = __riscv_vmflt_vv_f64m2_b32(values, other.values, VFLOAT64_VL);
    vuint64m2_t merge_res = __riscv_vmerge_vvm_u64m2(
      __riscv_vmv_v_x_u64m2(0x0, VFLOAT64_VL),
      __riscv_vmv_v_x_u64m2(UINT64_MAX, VFLOAT64_VL),
      cmp_res,
      VFLOAT64_VL);
    vfloat64m2_t res = __riscv_vreinterpret_v_u64m2_f64m2(merge_res);
    return Vectorized<double>(res);
  }

  Vectorized<double> operator<=(const Vectorized<double>& other) const {
    vbool32_t cmp_res = __riscv_vmfle_vv_f64m2_b32(values, other.values, VFLOAT64_VL);
    vuint64m2_t merge_res = __riscv_vmerge_vvm_u64m2(
      __riscv_vmv_v_x_u64m2(0x0, VFLOAT64_VL),
      __riscv_vmv_v_x_u64m2(UINT64_MAX, VFLOAT64_VL),
      cmp_res,
      VFLOAT64_VL);
    vfloat64m2_t res = __riscv_vreinterpret_v_u64m2_f64m2(merge_res);
    return Vectorized<double>(res);
  }

  Vectorized<double> operator>(const Vectorized<double>& other) const {
    vbool32_t cmp_res = __riscv_vmfgt_vv_f64m2_b32(values, other.values, VFLOAT64_VL);
    vuint64m2_t merge_res = __riscv_vmerge_vvm_u64m2(
      __riscv_vmv_v_x_u64m2(0x0, VFLOAT64_VL),
      __riscv_vmv_v_x_u64m2(UINT64_MAX, VFLOAT64_VL),
      cmp_res,
      VFLOAT64_VL);
    vfloat64m2_t res = __riscv_vreinterpret_v_u64m2_f64m2(merge_res);
    return Vectorized<double>(res);
  }

  Vectorized<double> operator>=(const Vectorized<double>& other) const {
    vbool32_t cmp_res = __riscv_vmfge_vv_f64m2_b32(values, other.values, VFLOAT64_VL);
    vuint64m2_t merge_res = __riscv_vmerge_vvm_u64m2(
      __riscv_vmv_v_x_u64m2(0x0, VFLOAT64_VL),
      __riscv_vmv_v_x_u64m2(UINT64_MAX, VFLOAT64_VL),
      cmp_res,
      VFLOAT64_VL);
    vfloat64m2_t res = __riscv_vreinterpret_v_u64m2_f64m2(merge_res);
    return Vectorized<double>(res);
  }

  Vectorized<double> eq(const Vectorized<double>& other) const;
  Vectorized<double> ne(const Vectorized<double>& other) const;
  Vectorized<double> gt(const Vectorized<double>& other) const;
  Vectorized<double> ge(const Vectorized<double>& other) const;
  Vectorized<double> lt(const Vectorized<double>& other) const;
  Vectorized<double> le(const Vectorized<double>& other) const;
};

template <>
Vectorized<double> inline operator+(const Vectorized<double>& a, const Vectorized<double>& b) {
  return __riscv_vfadd_vv_f64m2(a, b, VFLOAT64_VL);
}

template <>
Vectorized<double> inline operator-(const Vectorized<double>& a, const Vectorized<double>& b) {
  return __riscv_vfsub_vv_f64m2(a, b, VFLOAT64_VL);
}

template <>
Vectorized<double> inline operator*(const Vectorized<double>& a, const Vectorized<double>& b) {
  return __riscv_vfmul_vv_f64m2(a, b, VFLOAT64_VL);
}

template <>
Vectorized<double> inline operator/(const Vectorized<double>& a, const Vectorized<double>& b) {
  return __riscv_vfdiv_vv_f64m2(a, b, VFLOAT64_VL);
}

inline Vectorized<double> Vectorized<double>::frac() const {
  return *this - this->trunc();
}

template <>
Vectorized<double> inline maximum(const Vectorized<double>& a, const Vectorized<double>& b) {
  vbool32_t mask = __riscv_vmand_mm_b32(
    __riscv_vmfeq_vv_f64m2_b32(a, a, VFLOAT64_VL),
    __riscv_vmfeq_vv_f64m2_b32(b, b, VFLOAT64_VL),
    VFLOAT64_VL);
  vfloat64m2_t max_res = __riscv_vfmax_vv_f64m2(a, b, VFLOAT64_VL);
  vfloat64m2_t res = __riscv_vmerge_vvm_f64m2(__riscv_vfmv_v_f_f64m2(NAN, VFLOAT64_VL), max_res, mask, VFLOAT64_VL);
  return Vectorized<double>(res);
}

template <>
Vectorized<double> inline minimum(const Vectorized<double>& a, const Vectorized<double>& b) {
  vbool32_t mask = __riscv_vmand_mm_b32(
    __riscv_vmfeq_vv_f64m2_b32(a, a, VFLOAT64_VL),
    __riscv_vmfeq_vv_f64m2_b32(b, b, VFLOAT64_VL),
    VFLOAT64_VL);
  vfloat64m2_t min_res = __riscv_vfmin_vv_f64m2(a, b, VFLOAT64_VL);
  vfloat64m2_t res = __riscv_vmerge_vvm_f64m2(__riscv_vfmv_v_f_f64m2(NAN, VFLOAT64_VL), min_res, mask, VFLOAT64_VL);
  return Vectorized<double>(res);
}

template <>
Vectorized<double> inline clamp(const Vectorized<double>& a, const Vectorized<double>& min, const Vectorized<double>& max) {
  return minimum(max, maximum(min, a));
}

template <>
Vectorized<double> inline clamp_max(const Vectorized<double>& a, const Vectorized<double>& max) {
  return minimum(max, a);
}

template <>
Vectorized<double> inline clamp_min(const Vectorized<double>& a, const Vectorized<double>& min) {
  return maximum(min, a);
}

template <>
Vectorized<double> inline operator&(const Vectorized<double>& a, const Vectorized<double>& b) {
  vfloat64m2_t res = __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vand_vv_u64m2(
    __riscv_vreinterpret_v_f64m2_u64m2(a),
    __riscv_vreinterpret_v_f64m2_u64m2(b),
    VFLOAT64_VL));
  return Vectorized<double>(res);
}

template <>
Vectorized<double> inline operator|(const Vectorized<double>& a, const Vectorized<double>& b) {
  vfloat64m2_t res = __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vor_vv_u64m2(
    __riscv_vreinterpret_v_f64m2_u64m2(a),
    __riscv_vreinterpret_v_f64m2_u64m2(b),
    VFLOAT64_VL));
  return Vectorized<double>(res);
}

template <>
Vectorized<double> inline operator^(const Vectorized<double>& a, const Vectorized<double>& b) {
  vfloat64m2_t res = __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vxor_vv_u64m2(
    __riscv_vreinterpret_v_f64m2_u64m2(a),
    __riscv_vreinterpret_v_f64m2_u64m2(b),
    VFLOAT64_VL));
  return Vectorized<double>(res);
}


inline Vectorized<double> Vectorized<double>::eq(const Vectorized<double>& other) const {
  return (*this == other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::ne(const Vectorized<double>& other) const {
  return (*this != other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::gt(const Vectorized<double>& other) const {
  return (*this > other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::ge(const Vectorized<double>& other) const {
  return (*this >= other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::lt(const Vectorized<double>& other) const {
    return (*this < other) & Vectorized<double>(1.0);
  }
  
inline Vectorized<double> Vectorized<double>::le(const Vectorized<double>& other) const {
    return (*this <= other) & Vectorized<double>(1.0);
}
  
template <>
inline void convert(const double* src, int64_t* dst, int64_t n) {
int64_t i;
#ifndef __msvc_cl__
#pragma unroll
#endif
for (i = 0; i <= (n - Vectorized<double>::size()); i += Vectorized<double>::size()) {
    __riscv_vse64_v_i64m2(dst + i, __riscv_vfcvt_rtz_x_f_v_i64m2(__riscv_vle64_v_f64m2(src + i, VDOUBLE64_VL), VDOUBLE64_VL), VDOUBLE64_VL);
}
#ifndef __msvc_cl__
#pragma unroll
#endif
for (; i < n; i++) {
    dst[i] = static_cast<int64_t>(src[i]);
}
}
  
template <>
inline void convert(const int64_t* src, double* dst, int64_t n) {
  int64_t i;
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<double>::size()); i += Vectorized<double>::size()) {
    __riscv_vse64_v_f64m2(dst + i, __riscv_vfcvt_f_x_v_f64m2(__riscv_vle64_v_i64m2(src + i, VDOUBLE64_VL), VDOUBLE64_VL), VDOUBLE64_VL);
}
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (; i < n; i++) {
    dst[i] = static_cast<double>(src[i]);
  }
}

template <>
Vectorized<double> inline fmadd(const Vectorized<double>& a, const Vectorized<double>& b, const Vectorized<double>& c) {
  return __riscv_vfmacc_vv_f64m2(c, a, b, VDOUBLE64_VL);
}

template <>
Vectorized<double> inline fmsub(const Vectorized<double>& a, const Vectorized<double>& b, const Vectorized<double>& c) {
  return __riscv_vfnmsac_vv_f64m2(c, a, b, VDOUBLE64_VL);
}
  
Vectorized<double> inline maximum(const Vectorized<double>& a, const Vectorized<double>& b) {
  return __riscv_vfmax_vv_f64m2(a, b, VDOUBLE64_VL);
}
  
Vectorized<double> inline minimum(const Vectorized<double>& a, const Vectorized<double>& b) {
  return __riscv_vfmin_vv_f64m2(a, b, VDOUBLE64_VL);
}
  
}} // namespace at::vec::CPU_CAPABILITY