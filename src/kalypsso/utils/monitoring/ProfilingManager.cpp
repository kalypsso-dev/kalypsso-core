// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ProfilingManager.h"

#include <kalypsso/utils/log/kalypsso_log.h>

#include <iostream> // for std::cerr

namespace kalypsso
{

const std::string ProfilingData::whole_kalypsso_region_name = "Kalypsso::ALL";
const Color_t     ProfilingData::whole_kalypsso_region_color = Color_t::Silver();

// ==============================================================================================
// ==============================================================================================
// ProfilingRegion
// ==============================================================================================
// ==============================================================================================
//
// ==============================================================================================
// ==============================================================================================
ProfilingRegion::ProfilingRegion(std::string const & name,
                                 timer_location_t    timer_location,
                                 Color_t             color)
  : m_name(name)
  , m_timer_location(timer_location)
  , m_color(color)
  , m_state(IDLE)
#ifdef KALYPSSO_CORE_TIMING_ENABLED
  , m_host_timer()
  , m_device_timer()
#endif
{
#ifdef KALYPSSO_CORE_NVTX_ANNOTATION_ENABLED
  m_eventAttrib.version = NVTX_VERSION;
  m_eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  m_eventAttrib.category = 0;
  m_eventAttrib.colorType = NVTX_COLOR_ARGB;
  m_eventAttrib.color = m_color;
  m_eventAttrib.payloadType = 0;
  m_eventAttrib.reserved0 = 0;
  m_eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  m_eventAttrib.message.ascii = m_name.c_str();
#endif // KALYPSSO_CORE_NVTX_ANNOTATION_ENABLED

} // ProfilingRegion::ProfilingRegion

// ==============================================================================================
// ==============================================================================================
void
ProfilingRegion::start()
{
  if (m_state == RUNNING)
  {
    KALYPSSO_WARN("Can't start profiling region '{}', it is already RUNNING", m_name);
    return;
  }
  m_state = RUNNING;

  // add profiling hook for kokkos tools
  Kokkos::Profiling::pushRegion(m_name);

#ifdef KALYPSSO_CORE_NVTX_ANNOTATION_ENABLED
  nvtxRangePushEx(&m_eventAttrib);
#endif // KALYPSSO_CORE_NVTX_ANNOTATION_ENABLED

#ifdef KALYPSSO_CORE_TIMING_ENABLED
  m_host_timer.start();
  m_device_timer.start();
#endif // KALYPSSO_CORE_TIMING_ENABLED

} // ProfilingRegion::start

// ==============================================================================================
// ==============================================================================================
void
ProfilingRegion::stop()
{
  if (m_state != RUNNING)
  {
    KALYPSSO_WARN("Can't stop profiling region '{}', it is not RUNNING", m_name);
    return;
  }
  m_state = IDLE;

#ifdef KALYPSSO_CORE_TIMING_ENABLED
  m_host_timer.stop();
  m_device_timer.stop();
#endif // KALYPSSO_CORE_TIMING_ENABLED

#ifdef KALYPSSO_CORE_NVTX_ANNOTATION_ENABLED
  nvtxRangePop();
#endif // KALYPSSO_CORE_NVTX_ANNOTATION_ENABLED

  // add profiling hook for kokkos tools
  Kokkos::Profiling::popRegion();

} // ProfilingRegion::stop

// ==============================================================================================
// ==============================================================================================
void
ProfilingRegion::reset()
{
  if (m_state == RUNNING)
  {
    KALYPSSO_WARN("Can't reset profiling region '{}', it is RUNNING; stop it first", m_name);
    return;
  }

#ifdef KALYPSSO_CORE_TIMING_ENABLED
  m_host_timer.reset();
  m_device_timer.reset();
#endif // KALYPSSO_CORE_TIMING_ENABLED

} // ProfilingRegion::reset

// ==============================================================================================
// ==============================================================================================
// ProfilingManager
// ==============================================================================================
// ==============================================================================================

#ifdef KALYPSSO_CORE_TIMING_ENABLED
namespace
{

//! \brief min / mean / max of a timing over MPI ranks
struct TimingStats_t
{
  double min;
  double mean;
  double max;
};

//! \brief Reduce local_time into min / mean / max over all MPI ranks (collective).
//!
//! Reduced values are only meaningful on the root rank (rank 0), the only rank allowed to
//! report timings.
TimingStats_t
reduce_min_mean_max(double local_time, ParallelEnv const & par_env)
{
#ifdef KALYPSSO_CORE_USE_MPI
  double t_min = 0.0;
  double t_sum = 0.0;
  double t_max = 0.0;
  par_env.comm().MPI_Reduce<MpiComm::MIN>(&local_time, &t_min, 1, 0);
  par_env.comm().MPI_Reduce<MpiComm::SUM>(&local_time, &t_sum, 1, 0);
  par_env.comm().MPI_Reduce<MpiComm::MAX>(&local_time, &t_max, 1, 0);
  return { t_min, t_sum / par_env.size(), t_max };
#else
  return { local_time, local_time, local_time };
#endif // KALYPSSO_CORE_USE_MPI
}

//! \brief Print one line of the timing report (to be called on the root rank only).
//!
//! Percentages are computed with respect to total_max, the whole application time of the
//! slowest rank.
void
print_timings_line(std::string const & name, TimingStats_t const & t, double total_max)
{
  printf("%-40s time : %8.3f /%8.3f /%8.3f s (%6.2f%% of max)\n",
         name.c_str(), t.min, t.mean, t.max, 100.0 * t.max / total_max);
}

} // namespace
#endif // KALYPSSO_CORE_TIMING_ENABLED

// ==============================================================================================
// ==============================================================================================
ProfilingManager::ProfilingManager(ParallelEnv const & par_env)
  : m_par_env(par_env)
{
  // create a region that contains all computing regions
  // all other region is supposed to be inside this region
  // meaning that
  // - the very first call to ProfileRegion::start
  // - the very last  call to ProfileRegion::stop
  // should be using whole_kalypsso_region_name
  //
  // Note that start/stop for the whole application region should only be called once
  // or else the timing report will erroneous values
  m_profiling_region_map.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(ProfilingData::whole_kalypsso_region_name),
                                 std::forward_as_tuple(ProfilingData::whole_kalypsso_region_name,
                                                       ProfilingRegion::TIMER_HOST,
                                                       ProfilingData::whole_kalypsso_region_color));

} // ProfilingManager::ProfilingManager

// ==============================================================================================
// ==============================================================================================
ProfilingRegion &
ProfilingManager::get_region(std::string const &               name,
                             ProfilingRegion::timer_location_t timer_location,
                             Color_t                           color)
{
  auto it = m_profiling_region_map.find(name);

  // if region does'nt exist, create it
  if (it == m_profiling_region_map.end())
    it = m_profiling_region_map
           .emplace(std::piecewise_construct,
                    std::forward_as_tuple(name),
                    std::forward_as_tuple(name, timer_location, color))
           .first;
  return it->second;

} // ProfilingManager::get_region

// ==============================================================================================
// ==============================================================================================
ProfilingRegion &
ProfilingManager::get_region(std::string const &               name,
                             ProfilingRegion::timer_location_t timer_location,
                             ColorMap_t const &                colormap)
{
  // find color or use default color
  auto color = [](std::string const & name_, ColorMap_t const & colormap_) {
    auto itc = colormap_.find(name_);
    if (itc == colormap_.end())
    {
      KALYPSSO_WARN(
        "colormap doesn't contain name {}; using color default color instead (Black). You should "
        "probably update the colormap used in your solver.",
        name_);
      return Color_t::Black();
    }
    return itc->second;
  }(name, colormap);

  return get_region(name, timer_location, color);

} // ProfilingManager::get_region

// ==============================================================================================
// ==============================================================================================
void
ProfilingManager::print_timings()
{
#ifdef KALYPSSO_CORE_TIMING_ENABLED
  if (get_whole_region().is_running())
  {
    if (m_par_env.rank() == 0)
      std::cerr << "You can't print timings while the whole application region is still open.\n";
    return;
  }

  // whole application time, reduced over all MPI ranks (collective)
  const TimingStats_t total_stats = reduce_min_mean_max(total_time().value(), m_par_env);

  if (m_par_env.rank() == 0)
  {
    printf("# ====================================\n");
    printf("# Kalypsso timings (min/mean/max over %d ranks)\n", m_par_env.size());
    printf("# ====================================\n");

    print_timings_line(ProfilingData::whole_kalypsso_region_name, total_stats, total_stats.max);
  }

  report_region_timings(total_stats.max);
#else
  KALYPSSO_INFO("Timings are disabled. Please re-build with KALYPSSO_CORE_TIMING_ENABLED=ON.");
#endif // KALYPSSO_CORE_TIMING_ENABLED
} // ProfilingManager::print_timings

// ==============================================================================================
// ==============================================================================================
void
ProfilingManager::print_timings(std::optional<double> total_time, std::string const & header)
{
#ifdef KALYPSSO_CORE_TIMING_ENABLED
  if (get_whole_region().is_running())
  {
    if (m_par_env.rank() == 0)
      std::cerr << "You can't print timings while the whole application region is still open.\n";
    return;
  }

  // the caller-provided total time is a per-rank value; reduce it over all MPI ranks
  // (collective) so that percentages refer to the slowest rank
  const TimingStats_t total_stats = reduce_min_mean_max(total_time.value(), m_par_env);

  if (m_par_env.rank() == 0)
  {
    printf("# ====================================\n");
    printf("# %s (min/mean/max over %d ranks)\n", header.c_str(), m_par_env.size());
    printf("# ====================================\n");
  }

  report_region_timings(total_stats.max);
#else
  KALYPSSO_INFO("Timings are disabled. Please re-build with KALYPSSO_CORE_TIMING_ENABLED=ON.");
#endif // KALYPSSO_CORE_TIMING_ENABLED
} // ProfilingManager::print_timings

// ==============================================================================================
// ==============================================================================================
#ifdef KALYPSSO_CORE_TIMING_ENABLED
void
ProfilingManager::report_region_timings(double total_max)
{
  // This loop is collective: all MPI ranks must iterate over the same set of regions,
  // i.e. profiling regions are expected to be created uniformly on all ranks.
  // Note that elapsed() may flush pending device events (CudaAsyncTimer).
  for (auto const & prof : m_profiling_region_map)
  {
    const std::string & name = prof.first;
    if (name == ProfilingData::whole_kalypsso_region_name)
      continue;

    const TimingStats_t stats = reduce_min_mean_max(prof.second.elapsed(), m_par_env);

    if (m_par_env.rank() == 0)
      print_timings_line(name, stats, total_max);
  }

} // ProfilingManager::report_region_timings
#endif // KALYPSSO_CORE_TIMING_ENABLED

// ==============================================================================================
// ==============================================================================================
std::optional<double>
ProfilingManager::total_time()
{
#ifdef KALYPSSO_CORE_TIMING_ENABLED
  if (get_whole_region().is_running())
  {
    if (m_par_env.rank() == 0)
      std::cerr << "You can't get timings while the whole application region is still open.\n";
    return std::nullopt;
  }

  // get whole application time
  auto & whole_app_region = get_whole_region();
  return whole_app_region.host_timer().elapsed();
#else
  // return invalid value
  return std::nullopt;
#endif // KALYPSSO_CORE_TIMING_ENABLED

} // ProfilingManager::total_time

} // namespace kalypsso
