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
#include <utility> // for std::exchange

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
//! \brief Name and color of the whole application profiling region.
struct ProfilingData
{
  static const std::string whole_kalypsso_region_name;
  static const Color_t     whole_kalypsso_region_color;

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

  ProfilingRegion(std::string const & name, timer_location_t timer_location, Color_t color);

  ~ProfilingRegion() = default;

  //! non-copyable: the nvtx event attribute stores a pointer into m_name of this instance
  ProfilingRegion(ProfilingRegion const &) = delete;
  ProfilingRegion &
  operator=(ProfilingRegion const &) = delete;

  //! start timing (no-op if the region is already running)
  void
  start();

  //! stop timing (no-op if the region is not running)
  void
  stop();

  //! reset accumulated timings (no-op if the region is running)
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

  //! return true if the region is currently running, i.e. has been started
  //! and not stopped yet
  bool
  is_running() const
  {
    return m_state == RUNNING;
  }

  //! elapsed time in seconds
  [[nodiscard]] double
  elapsed() const
  {
    return m_timer_location == TIMER_HOST ? m_host_timer.elapsed() : m_device_timer.elapsed();
  }

#endif

private:
  const std::string m_name;
  timer_location_t  m_timer_location; //<! preferred timer
  const Color_t     m_color;
  State_t           m_state;

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
 * \class ScopedRegion
 * \brief Scoped helper that times a profiling region from construction to destruction.
 *
 * The wrapped region is started at construction and stopped at destruction, making the
 * start/stop pair robust against early returns and exceptions. The region can also be
 * stopped early by calling stop(). Ownership can be transferred by move; copying and
 * assignment are disabled.
 *
 * \note distinct from Kokkos::Profiling::ScopedRegion which only pushes/pops a Kokkos
 * profiling region; this helper also drives nvtx annotations and timers.
 */
class ScopedRegion
{

public:
  explicit ScopedRegion(ProfilingRegion & region)
    : m_region(&region)
  {
    m_region->start();
  }

  ~ScopedRegion()
  {
    if (m_region != nullptr)
      m_region->stop();
  }

  ScopedRegion(ScopedRegion && other) noexcept
    : m_region(std::exchange(other.m_region, nullptr))
  {}

  //! stop the wrapped region now and release ownership (no-op if already stopped or
  //! moved-from)
  void
  stop()
  {
    if (m_region != nullptr)
    {
      m_region->stop();
      m_region = nullptr;
    }
  }

  //! return true while this helper owns a region, i.e. until stop() is called or ownership
  //! is moved away
  bool
  active() const
  {
    return m_region != nullptr;
  }

  ScopedRegion(ScopedRegion const &) = delete;
  ScopedRegion &
  operator=(ScopedRegion const &) = delete;
  ScopedRegion &
  operator=(ScopedRegion &&) = delete;

private:
  //! wrapped profiling region (null once stopped or moved away)
  ProfilingRegion * m_region;

}; // class ScopedRegion

// ==============================================================================================
// ==============================================================================================
// ==============================================================================================
/**
 * \class ProfilingManager
 *
 * Handle time measurement reporting and nvtx annotations for collecting runtime tracing information
 * with nsys.
 */
class ProfilingManager
{

public:
  ProfilingManager(ParallelEnv const & par_env);

  ~ProfilingManager() = default;

  //! non-copyable, non-movable
  ProfilingManager(ProfilingManager const &) = delete;
  ProfilingManager &
  operator=(ProfilingManager const &) = delete;

  /**
   * \brief Get or create a profiling region.
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
   * \brief Get or create a profiling region, start it right away and return a scoped
   * helper stopping it at end of scope.
   *
   * \param[in] name Region name.
   * \param[in] timer_location Preferred timer location.
   * \param[in] color Region color used for nvtx annotation.
   */
  [[nodiscard]] ScopedRegion
  make_scoped_region(std::string const &               name,
                     ProfilingRegion::timer_location_t timer_location,
                     Color_t                           color = Color_t::Default())
  {
    return ScopedRegion(get_region(name, timer_location, color));
  } // make_scoped_region

  /**
   * \brief Get or create a profiling region.
   */
  ProfilingRegion &
  get_whole_region()
  {
    return m_profiling_region_map.at(ProfilingData::whole_kalypsso_region_name);
  } // get_whole_region

  /**
   * \brief Print timing report.
   *
   * Timings are reduced over all MPI ranks (min / mean / max) before reporting; percentages
   * are computed with respect to the whole application time of the slowest rank.
   *
   * \note collective operation: all MPI ranks must call it, with the same set of profiling
   * regions.
   */
  void
  print_timings();

  /**
   * \brief Print timing report when total time is provided by another profiling manager.
   *
   * Timings, including the caller-provided total time (a per-rank value), are reduced over
   * all MPI ranks (min / mean / max) before reporting; percentages are computed with respect
   * to the slowest rank.
   *
   * \param[in] total_time Total time (per-rank value) used to compute percentage; may be
   * empty when timing is disabled at compile time (in which case nothing is reported).
   * \param[in] header A string to customize printing.
   *
   * \note collective operation: all MPI ranks must call it, with the same set of profiling
   * regions.
   */
  void
  print_timings(std::optional<double> total_time, std::string const & header);

  //! return total time in seconds.
  [[nodiscard]] std::optional<double>
  total_time();

private:
#ifdef KALYPSSO_CORE_TIMING_ENABLED
  //! Print min / mean / max timing of each region (except the whole application region) as a
  //! fraction of total_max. Collective: all MPI ranks must call it.
  void
  report_region_timings(double total_max);
#endif // KALYPSSO_CORE_TIMING_ENABLED

  ParallelEnv const &                    m_par_env;
  std::map<std::string, ProfilingRegion> m_profiling_region_map;

}; // class ProfilingManager

} // namespace kalypsso

#endif // KALYPSSO_UTILS_MONITORING_PROFILINGMANAGER_H_
