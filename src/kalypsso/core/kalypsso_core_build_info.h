// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file kalypsso_core_build_info.h
 */
#ifndef KALYPSSO_CORE_KALYPSSO_CORE_BUILD_INFO_H_
#define KALYPSSO_CORE_KALYPSSO_CORE_BUILD_INFO_H_

#include <kalypsso_core_version.h>
#include <kalypsso/core/kalypsso_core_config.h>

#include <string>

namespace kalypsso
{

struct BuildInfo
{
  static std::string
  system_processor();

  static std::string
  system_name();

  static std::string
  build_type();

  static std::string
  compiler_id();

  static std::string
  compiler_version();

  static std::string
  compile_date();
  static std::string
  compile_time();

  static std::string
  mpi_runtime_config();

  static std::string
  mpi_runtime_cuda_support();
  static std::string
  mpi_runtime_hip_support();
  static std::string
  mpi_runtime_ze_support();

  static bool
  hdf5_enabled();
  static std::string
  hdf5_version();

  static void
  print();
};

} // namespace kalypsso

#endif // KALYPSSO_CORE_KALYPSSO_CORE_BUILD_INFO_H_
