// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/core/kalypsso_core_config.h> // for KALYPSSO_CORE_USE_MPI
#include <kalypsso/core/kalypsso_core_build_info.h>
#include <kalypsso/core/kalypsso_core_config.h>

#include <iostream>

#ifdef KALYPSSO_CORE_USE_MPI
#  include <mpi.h>
#  if KALYPSSO_CORE_USE_MPI_EXT
#    include <mpi-ext.h>
#  endif
#endif

namespace kalypsso
{

std::string
BuildInfo::system_processor()
{
  return KALYPSSO_CORE_SYSTEM_PROCESSOR;
}

std::string
BuildInfo::system_name()
{
  return KALYPSSO_CORE_SYSTEM_NAME;
}

std::string
BuildInfo::build_type()
{
  return KALYPSSO_CORE_BUILD_TYPE;
}

std::string
BuildInfo::compiler_id()
{
  return KALYPSSO_CORE_BUILD_COMPILER_ID;
}

std::string
BuildInfo::compiler_version()
{
  return KALYPSSO_CORE_BUILD_COMPILER_VERSION;
}

std::string
BuildInfo::compile_date()
{
  return KALYPSSO_CORE_COMPILE_DATE;
}

std::string
BuildInfo::compile_time()
{
  return KALYPSSO_CORE_COMPILE_TIME;
}

std::string
BuildInfo::mpi_runtime_config()
{
#ifdef KALYPSSO_CORE_USE_MPI
  return []() {
    int  length;
    char mpi_version[MPI_MAX_LIBRARY_VERSION_STRING];
    MPI_Get_library_version(mpi_version, &length);
    return std::string(mpi_version);
  }();
#else
  return "none";
#endif // KALYPSSO_CORE_USE_MPI
}

std::string
BuildInfo::mpi_runtime_cuda_support()
{
#ifdef KALYPSSO_CORE_USE_MPI
#  if KALYPSSO_CORE_MPI_HAS_QUERY_CUDA_SUPPORT
  int result = MPIX_Query_cuda_support();
  return result == 1 ? std::string("yes") : std::string("no");
#  else
  return "unknown - not queryable";
#  endif
#else
  return "unknown - MPI not used";
#endif
}

std::string
BuildInfo::mpi_runtime_hip_support()
{
#ifdef KALYPSSO_CORE_USE_MPI
#  if KALYPSSO_CORE_MPI_HAS_QUERY_HIP_SUPPORT
  int result = MPIX_Query_hip_support();
  return result == 1 ? std::string("yes") : std::string("no");
#  else
  return "unknown - not queryable";
#  endif
#else
  return "unknown - MPI not used";
#endif
}

std::string
BuildInfo::mpi_runtime_ze_support()
{
#ifdef KALYPSSO_CORE_USE_MPI
#  if KALYPSSO_CORE_MPI_HAS_QUERY_ZE_SUPPORT
  int result = MPIX_Query_ze_support();
  return result == 1 ? std::string("yes") : std::string("no");
#  else
  return "unknown - not queryable";
#  endif
#else
  return "unknown - MPI not used";
#endif
}

bool
BuildInfo::hdf5_enabled()
{
#ifdef KALYPSSO_CORE_USE_HDF5
  return true;
#else
  return false;
#endif
}

std::string
BuildInfo::hdf5_version()
{
#ifdef KALYPSSO_CORE_USE_HDF5
  return KALYPSSO_CORE_USE_HDF5_VERSION;
#else
  return "unknown";
#endif
}

void
BuildInfo::print()
{
  std::cout << "##############################################\n";
  std::cout << "kalypsso-core - build info\n";
  std::cout << "system name      : " << system_name() << "\n";
  std::cout << "system processor : " << system_processor() << "\n";
  std::cout << "compile date     : " << compile_date() << "\n";
  std::cout << "compile time     : " << compile_time() << "\n";
  std::cout << "build type       : " << build_type() << "\n";
  std::cout << "compiler id      : " << compiler_id() << "\n";
  std::cout << "compiler version : " << compiler_version() << "\n";
  std::cout << "MPI runtime config : " << mpi_runtime_config() << "\n";
  std::cout << "MPI runtime cuda support : " << mpi_runtime_cuda_support() << "\n";
  std::cout << "MPI runtime hip  support : " << mpi_runtime_hip_support() << "\n";
  std::cout << "MPI runtime ze   support : " << mpi_runtime_ze_support() << "\n";
  if (hdf5_enabled())
    std::cout << "HDF5 version             : " << hdf5_version() << "\n";
  std::cout << "##############################################\n";
}

} // namespace kalypsso
