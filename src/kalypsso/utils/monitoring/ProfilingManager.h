// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ProfilingManager.h
 * \brief Define a simple class to handle monitoring profiling region.
 *
 * By profiling we mean:
 * - use (hardware aware) timer
 * - use nvtx annotations (requires to have cuda drive installed, but don't require to have Kokkos
 * built with Kokkos::Cuda backend enabled.)
 *
 * All monitoring instrumentation can be turned off at compile time.
 * For nvtx annotation, we just need to define symbol NVTX_DISABLE in cmake.
 */
#ifndef KALYPSSO_UTILS_MONITORING_PROFILINGMANAGER_H_
#define KALYPSSO_UTILS_MONITORING_PROFILINGMANAGER_H_

#include <map>
#include <string>
#include <cassert>
#include <optional>

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include "HostTimer.h"

#ifdef KOKKOS_ENABLE_CUDA
#  include "CudaAsyncTimer.h"
#endif

#ifdef KOKKOS_ENABLE_HIP
// #  include "HIPAsyncTimer.h"
#endif

#ifdef KALYPSSO_CORE_NVTX_ANNOTATION_ENABLED
#  include <nvtx3/nvToolsExt.h>
// #  include <nvtx3/nvToolsExtCuda.h>
#endif // KALYPSSO_CORE_NVTX_ANNOTATION_ENABLED

#include "NvtxProfiling.h" // for class Color

// #include <kalypsso/core/kalypsso_core_base.h> // for assertm
#ifndef assertm
#  define assertm(exp, msg) assert(((void)msg, exp))
#endif

namespace kalypsso
{

//! a type alias to map profile region names to a color
using ColorMap_t = std::map<const std::string, const Color_t>;


// ==============================================================================================
// ==============================================================================================
// ==============================================================================================
struct ProfilingData
{
  static const std::string whole_kalypsso_region_name;
  static const Color_t     whole_kalypsso_region_color;

  //! flag indicating when all timings can be reported
  //! default is false
  //! become true when stopping the whole application timer
  static bool timings_valid;

}; // struct ProfilingData

// ==============================================================================================
// ==============================================================================================
// ==============================================================================================
class ProfilingRegion
{

private:
  //! a timer should be idle when calling start, and running when calling stop.
  enum State_t : uint32_t
  {
    IDLE,
    RUNNING
  };

#if defined(KOKKOS_ENABLE_CUDA)
  using DeviceTimer = CudaAsyncTimer;
#elif defined(KOKKOS_ENABLE_HIP)
  // using DeviceTimer = HIPAsyncTimer;
#else
  using DeviceTimer = HostTimer;
#endif

public:
  //! preferred timer location
  enum timer_location_t
  {
    TIMER_HOST,
    TIMER_DEVICE
  };

  ProfilingRegion(ParallelEnv const & par_env,
                  std::string const & name,
                  timer_location_t    timer_location,
                  Color_t             color);

  ~ProfilingRegion() = default;

  //! start timing
  void
  start();

  //! stop timing
  void
  stop();

  //! reset timer
  void
  reset();

#ifdef KALYPSSO_CORE_TIMING_ENABLED
  // =====================================================
  // =====================================================
  HostTimer &
  host_timer()
  {
    return m_host_timer;
  }

  // =====================================================
  // =====================================================
  DeviceTimer &
  device_timer()
  {
    return m_device_timer;
  }

  //! elapsed time in seconds
  double
  elapsed()
  {
    return m_timer_location == TIMER_HOST ? m_host_timer.elapsed() : m_device_timer.elapsed();
  }

#endif

private:
  ParallelEnv const & m_par_env;
  const std::string   m_name;
  timer_location_t    m_timer_location; //<! preferred timer
  const Color_t       m_color;
  State_t             m_state;

#ifdef KALYPSSO_CORE_NVTX_ANNOTATION_ENABLED
  nvtxEventAttributes_t m_eventAttrib;
#endif // KALYPSSO_CORE_NVTX_ANNOTATION_ENABLED

#ifdef KALYPSSO_CORE_TIMING_ENABLED
  HostTimer   m_host_timer;
  DeviceTimer m_device_timer;
#endif

}; // ProfilingRegion

// ==============================================================================================
// ==============================================================================================
// ==============================================================================================
/**
 * \class ProfilingManager
 *
 * Handle time measurement reporting and nvtx annotations for collecting runtime tracing information
 * with nsys.
 */
class ProfilingManager : public ProfilingData
{

public:
  ProfilingManager(ParallelEnv const & par_env);

  ~ProfilingManager() = default;

  /**
   * Get or create a profiling region.
   */
  ProfilingRegion &
  get_region(std::string const &               name,
             ProfilingRegion::timer_location_t timer_location,
             Color_t                           color = Color_t::Default());

  ProfilingRegion &
  get_region(std::string const &               name,
             ProfilingRegion::timer_location_t timer_location,
             ColorMap_t const &                colormap);

  /**
   * Get or create a profiling region.
   */
  ProfilingRegion &
  get_whole_region()
  {
    return m_profiling_region_map.at(whole_kalypsso_region_name);
  } // get_whole_region

  /**
   * print timing report.
   *
   * TODO: we probably want to perform a MPI_Reduce (mean/max) over timings before reporting
   * currently we only use timing from MPI master rank (this is probably good enough for now)
   */
  void
  print_timings();

  //! return total time in seconds.
  std::optional<double>
  total_time();

private:
  ParallelEnv const &                    m_par_env;
  std::map<std::string, ProfilingRegion> m_profiling_region_map;

}; // class ProfilingManager

} // namespace kalypsso

#endif // KALYPSSO_UTILS_MONITORING_PROFILINGMANAGER_H_
