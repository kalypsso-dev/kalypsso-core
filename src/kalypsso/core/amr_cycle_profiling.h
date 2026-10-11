// SPDX-FileCopyrightText: 2026 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file amr_cycle_profiling.h
 *
 * \brief Declares the profiling zones and helpers specific to AMR cycle.
 */

#ifndef KALYPSSO_CORE_AMR_CYCLE_PROFILING_H_
#define KALYPSSO_CORE_AMR_CYCLE_PROFILING_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/utils/monitoring/ProfilingManager.h>
#include <../better-enums/enum.h>

#define ADD_CASE(enum, name, color) \
  case AmrCycleProfilingZone::enum: \
    return                          \
    {                               \
      name, Color_t::color()        \
    }

#ifndef KALYPSSO_PROFILING_REGION
#  define KALYPSSO_PROFILING_REGION_HOST(mgr, value) \
    kalypsso::core::ProfilingTimer region_##value(   \
      mgr, AmrCycleProfilingZone::value, ProfilingRegion::TIMER_HOST)
#  define KALYPSSO_PROFILING_REGION_DEVICE(mgr, value) \
    kalypsso::core::ProfilingTimer region_##value(     \
      mgr, AmrCycleProfilingZone::value, ProfilingRegion::TIMER_DEVICE)
#endif

namespace kalypsso
{
namespace core
{

/**
 * \enum AmrCycleProfilingZone
 * \brief Profiling zones of the AMR cycle.
 */
// clang-format off
BETTER_ENUM(AmrCycleProfilingZone, uint32_t,
  AMR_CYCLE,
    AMR_CYCLE_UPDATE,
    AMR_CYCLE_REBUILD,
    AMR_CYCLE_REMAP
)
// clang-format on

/**
 * \class ProfilingTimer
 * \brief Scoped helper that times a profiling zone, from construction to destruction.
 */
class ProfilingTimer
{
private:
  static constexpr std::pair<const char *, Color_t>
  get_name_color(const AmrCycleProfilingZone zone)
  {
    switch (zone)
    {
      // Kind of hacky names so that the names are aligned.
      // clang-format off
      ADD_CASE(AMR_CYCLE,                         "Kalypsso::CORE::AMR_CYCLE                              ", Blue);
      ADD_CASE(AMR_CYCLE_UPDATE,                  "Kalypsso::CORE::AMR_CYCLE::UPDATE                      ", Blue);
      ADD_CASE(AMR_CYCLE_REBUILD,                 "Kalypsso::CORE::AMR_CYCLE::REBUILD                     ", Blue);
      ADD_CASE(AMR_CYCLE_REMAP,                   "Kalypsso::CORE::AMR_CYCLE::REMAP                       ", Blue);
      // clang-format on
      default:
        return { zone._to_string(), Color_t::Default() };
    }
  }

public:
  static auto &
  get_profiling_region(ProfilingManager &                profiling_mgr,
                       const AmrCycleProfilingZone       zone,
                       ProfilingRegion::timer_location_t timer_location)
  {
    const auto [name, color] = get_name_color(zone);
    return profiling_mgr.get_region(name, timer_location, color);
  }

  ProfilingTimer(ProfilingManager & profiling_mgr, const AmrCycleProfilingZone zone)
    : m_timer(get_profiling_region(profiling_mgr, zone, ProfilingRegion::TIMER_HOST))
  {
    m_timer.start();
  }

  ProfilingTimer(ProfilingManager &                profiling_mgr,
                 const AmrCycleProfilingZone       zone,
                 ProfilingRegion::timer_location_t timer_location)
    : m_timer(get_profiling_region(profiling_mgr, zone, timer_location))
  {
    m_timer.start();
  }

  ~ProfilingTimer() { m_timer.stop(); }

private:
  ProfilingRegion & m_timer;
};

} // namespace core

} // namespace kalypsso

#undef ADD_CASE

#endif // KALYPSSO_CORE_AMR_CYCLE_PROFILING_H_
