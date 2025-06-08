#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>

#if defined(CPU_CAPABILITY_RVV)

// vector-length specific
typedef vint8m2_t fixed_vint8m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 2)));
typedef vint16m2_t fixed_vint16m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 2)));
typedef vint32m2_t fixed_vint32m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 2)));
typedef vint64m2_t fixed_vint64m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 2)));
typedef vuint8m2_t fixed_vuint8m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 2)));
typedef vuint16m2_t fixed_vuint16m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 2)));
typedef vuint32m2_t fixed_vuint32m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 2)));
typedef vuint64m2_t fixed_vuint64m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 2)));
typedef vfloat32m2_t fixed_vfloat32m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 2)));
typedef vfloat64m2_t fixed_vfloat64m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 2)));
typedef vfloat64m4_t fixed_vfloat64m4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 4))); 
// VL control uses only 256 bits
#define VFLOAT32_VL  (2*__riscv_v_min_vlen/32)
#define VQINT8_VL  (2*__riscv_v_min_vlen/8)
#define VQUINT8_VL  (2*__riscv_v_min_vlen/8)
#define VQINT32_VL  (2*__riscv_v_min_vlen/32)

#ifdef USE_RVV_M4
constexpr size_t VFLOAT64_VL = __riscv_vsetvlmax_e64m4();
#else
constexpr size_t VFLOAT64_VL = __riscv_vsetvlmax_e64m2();
#endif

// 添加fixed类型定义
#ifdef USE_RVV_M4
using fixed_vfloat64m4_t = vfloat64m4_t;
#else
using fixed_vfloat64m2_t = vfloat64m2_t;
#endif


#endif // defined(CPU_CAPABILITY_RVV)


