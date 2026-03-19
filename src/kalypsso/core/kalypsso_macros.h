// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file kalypsso_macros.h
 */
#ifndef KALYPSSO_CORE_KALYPSSO_MACROS_H_
#define KALYPSSO_CORE_KALYPSSO_MACROS_H_

#include <cassert>

#ifndef assertm
#  define assertm(exp, msg) assert(((void)msg, exp))
#endif

// clang-format off
#if defined(__GNUC__) || defined(__clang__) || defined(__NVCC__)
#define KALYPSSO_DISABLE_CONVERSION_WARNINGS_PUSH() \
  _Pragma("GCC diagnostic push") \
  _Pragma("GCC diagnostic ignored \"-Wconversion\"") \
  _Pragma("GCC diagnostic ignored \"-Wsign-conversion\"")
#define KALYPSSO_DISABLE_CONVERSION_WARNINGS_POP() \
  _Pragma("GCC diagnostic pop")
#else
#define KALYPSSO_DISABLE_CONVERSION_WARNINGS_PUSH()
#define KALYPSSO_DISABLE_CONVERSION_WARNINGS_POP()
#endif

//
// Disable -Wstringop-overflow (need for testing cpptrace)
//
// clang-format off
#if defined(__GNUC__) || defined(__clang__) || defined(__NVCC__)
#define KALYPSSO_DISABLE_STRINGOP_OVERFLOW_WARNINGS_PUSH() \
  _Pragma("GCC diagnostic push") \
  _Pragma("GCC diagnostic ignored \"-Wstringop-overflow\"")
#define KALYPSSO_DISABLE_STRINGOP_OVERFLOW_WARNINGS_POP() \
  _Pragma("GCC diagnostic pop")
#else
#define KALYPSSO_DISABLE_STRINGOP_OVERFLOW_WARNINGS_PUSH()
#define KALYPSSO_DISABLE_STRINGOP_OVERFLOW_WARNINGS_POP()
#endif

//
// macro to disable some warning check that we think are not relevant at some locations
//
#if defined(__NVCC__)
#define KALYPSSO_DISABLE_NVCC_WARNINGS_PUSH() \
  _Pragma("GCC diagnostic push") \
  _Pragma("GCC diagnostic ignored \"-Wold-style-cast\"")
#define KALYPSSO_DISABLE_NVCC_WARNINGS_POP() \
  _Pragma("GCC diagnostic pop")
#else
#define KALYPSSO_DISABLE_NVCC_WARNINGS_PUSH()
#define KALYPSSO_DISABLE_NVCC_WARNINGS_POP()
#endif

// clang-format on

#endif // KALYPSSO_CORE_KALYPSSO_MACROS_H_
