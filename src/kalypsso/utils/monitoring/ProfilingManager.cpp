// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ProfilingManager.h"

#include <kalypsso/utils/log/kalypsso_log.h>

#include <iomanip>  // for std::setw
#include <iostream> // for std::cerr

namespace kalypsso
{

const std::string ProfilingData::whole_kalypsso_region_name = "Kalypsso::ALL";
const Color_t     ProfilingData::whole_kalypsso_region_color = Color_t::Silver();

bool ProfilingData::timings_valid = false;

// ==============================================================================================
// ==============================================================================================
// ProfilingRegion
// ==============================================================================================
// ==============================================================================================
//
// ==============================================================================================
// ==============================================================================================
ProfilingRegion::ProfilingRegion(ParallelEnv const & par_env,
                                 std::string const & name,
                                 timer_location_t    timer_location,
                                 Color_t             color)
  : m_par_env(par_env)
  , m_name(name)
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
  // add profiling hook for kokkos tools
  Kokkos::Profiling::pushRegion(m_name);

#ifdef KALYPSSO_CORE_NVTX_ANNOTATION_ENABLED
  nvtxRangePushEx(&m_eventAttrib);
#endif // KALYPSSO_CORE_NVTX_ANNOTATION_ENABLED

#ifdef KALYPSSO_CORE_TIMING_ENABLED
  if (m_state == IDLE)
  {
    m_state = RUNNING;
    m_host_timer.start();
    m_device_timer.start();
  }
  else
  {
    if (m_par_env.rank() == 0)
      printf("Can't start timer, timer is already RUNNING");
  }
#endif // KALYPSSO_CORE_TIMING_ENABLED

} // ProfilingRegion::start

// ==============================================================================================
// ==============================================================================================
void
ProfilingRegion::stop()
{
#ifdef KALYPSSO_CORE_TIMING_ENABLED
  if (m_state == RUNNING)
  {
    m_state = IDLE;
    m_host_timer.stop();
    m_device_timer.stop();
    if (m_name == ProfilingData::whole_kalypsso_region_name)
      ProfilingData::timings_valid = true;
  }
  else
  {
    if (m_par_env.rank() == 0)
      printf("Can't stop timer, timer is not RUNNING");
  }

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
                                 std::forward_as_tuple(whole_kalypsso_region_name),
                                 std::forward_as_tuple(par_env,
                                                       whole_kalypsso_region_name,
                                                       ProfilingRegion::TIMER_HOST,
                                                       whole_kalypsso_region_color));

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
                    std::forward_as_tuple(m_par_env, name, timer_location, color))
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
  if (!ProfilingData::timings_valid)
  {
    if (m_par_env.rank() == 0)
      std::cerr << "You can't print timings before the whole application region is still open.\n";
    return;
  }

  if (m_par_env.rank() == 0)
  {
    printf("# ====================================\n");
    printf("# Kalypsso timings\n");
    printf("# ====================================\n");

    // get whole application time using the host timer
    auto & whole_app_region = m_profiling_region_map.at(whole_kalypsso_region_name);
    auto   total_time_sec = whole_app_region.host_timer().elapsed();
    assertm(total_time_sec > 0, "Invalid value: total_time_sec <= 0 !");
    std::cout << std::left << std::setw(40) << whole_kalypsso_region_name;
    printf(" time : %5.3f \ts (%5.2f%%)", total_time_sec, 100.0);
    std::cout << std::endl;

    // for other timer display the fraction of time in each region with respect to
    // the whole application time
    // timer elapsed method is not const (when using CudaAsyncTimer)
    for (auto & prof : m_profiling_region_map)
    {
      const std::string & name = prof.first;
      if (name != whole_kalypsso_region_name)
      {
        double time_sec = prof.second.elapsed();
        double percent = 100 * time_sec / total_time_sec;
        std::cout << std::left << std::setw(40) << name;
        printf(" time : %5.3f \ts (%5.2f%%)", time_sec, percent);
        std::cout << std::endl;
      }
    }
  }
#else
    KALYPSSO_INFO(("Timings are disabled. Please re-build with KALYPSSO_CORE_TIMING_ENABLED=ON.");
#endif // KALYPSSO_CORE_TIMING_ENABLED
} // ProfilingManager::print_timings

// ==============================================================================================
// ==============================================================================================
std::optional<double>
ProfilingManager::total_time()
{
#ifdef KALYPSSO_CORE_TIMING_ENABLED
  if (!ProfilingData::timings_valid)
  {
    if (m_par_env.rank() == 0)
      std::cerr << "You can't get timings before the whole application region is still open.\n";
    return std::nullopt;
  }

  // get whole application time
  auto & whole_app_region = m_profiling_region_map.at(whole_kalypsso_region_name);
  return whole_app_region.host_timer().elapsed();
#else
  // return invalid value
  return std::nullopt;
#endif // KALYPSSO_CORE_TIMING_ENABLED

} // ProfilingManager::total_time

} // namespace kalypsso
