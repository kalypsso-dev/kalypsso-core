// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file memory_utils.h
 * \brief Define some helper routines for monitoring free/occupied memory on device.
 */
#ifndef KALYPSSO_UTILS_MONITORING_MEMORYUTILS_H_
#define KALYPSSO_UTILS_MONITORING_MEMORYUTILS_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/utils/log/kalypsso_log.h>

#if (defined(__APPLE__))
#  define SYS_MEMORYINFO_MAC
#  include <mach/vm_statistics.h>
#  include <mach/mach_types.h>
#  include <mach/mach_init.h>
#  include <mach/mach_host.h>
#  include <mach/mach.h>
#elif (defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || \
       defined(_WIN64))
#  define SYS_MEMORYINFO_WINDOWS
#  include <windows.h>
#  include <psapi.h>
#  undef min
#  undef max
#else
#  define SYS_MEMORYINFO_LINUX
#  include <unistd.h>
#  include <sys/types.h>
#  include <sys/sysinfo.h>
#endif

namespace kalypsso
{

/**
 * \struct MemInfo
 */
struct MemInfo
{
  //! amount of free physical memory on device in GBytes
  double free;

  //! amount of used physical memory on device in GBytes
  double used;

  //! total amount of physical memory on device in GBytes
  double total;
};

// ===============================================================
// ===============================================================
template <typename device_t>
MemInfo
get_device_mem_info()
{
  // using ExecSpace = typename device_t::execution_space;
  using MemorySpace = typename device_t::memory_space;

  MemInfo mem_info;
  mem_info.free = 0.0;
  mem_info.used = 0.0;
  mem_info.total = 0.0;

  if constexpr (std::is_same<MemorySpace, Kokkos::HostSpace>::value)
  {
#ifdef SYS_MEMORYINFO_LINUX
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    mem_info.free = static_cast<double>(memInfo.freeram) * memInfo.mem_unit * 1e-9;
    mem_info.used =
      static_cast<double>(memInfo.totalram - memInfo.freeram) * memInfo.mem_unit * 1e-9;
    mem_info.total = static_cast<double>(memInfo.totalram) * memInfo.mem_unit * 1e-9;
#elif defined(SYS_MEMORYINFO_WINDOWS)
    // TODO
#elif defined(SYS_MEMORYINFO_MAC)
    // TODO
#endif
  }
#if defined(KOKKOS_ENABLE_CUDA)
  else if constexpr (std::is_same_v<MemorySpace, Kokkos::CudaSpace>)
  {
    size_t free_t, total_t;

    cudaMemGetInfo(&free_t, &total_t);

    mem_info.free = static_cast<double>(free_t) * 1e-9;
    mem_info.used = static_cast<double>(total_t - free_t) * 1e-9;
    mem_info.total = static_cast<double>(total_t) * 1e-9;
  }
#endif
#if defined(KOKKOS_ENABLE_HIP)
  else if constexpr (std::is_same_v<MemorySpace, Kokkos::HIPSpace>)
  {
    size_t free_t, total_t;

    hipMemGetInfo(&free_t, &total_t);

    mem_info.free = static_cast<double>(free_t) * 1e-9;
    mem_info.used = static_cast<double>(total_t - free_t) * 1e-9;
    mem_info.total = static_cast<double>(total_t) * 1e-9;
  }
#endif

  return mem_info;

} // get_device_mem_info

// ===============================================================
// ===============================================================
/**
 * Print memory information on master process.
 */
template <typename device_t>
void
print_device_mem_info()
{
  const auto mem_info = get_device_mem_info<device_t>();

  KALYPSSO_INFO("Total memory: {:.3f} GBytes", mem_info.total);
  KALYPSSO_INFO("Used  memory: {:.3f} GBytes", mem_info.used);
  KALYPSSO_INFO("Free  memory: {:.3f} GBytes", mem_info.free);
}

} // namespace kalypsso

#endif // KALYPSSO_UTILS_MONITORING_MEMORYUTILS_H_
