// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file HostTimer.h
 * \brief A simple timer class for CPU time measurement.
 *
 */
#ifndef KALYPSSO_UTILS_MONITORING_HOSTTIMER_H_
#define KALYPSSO_UTILS_MONITORING_HOSTTIMER_H_

#include <chrono>

namespace kalypsso
{

using namespace std::literals::chrono_literals; // for string literals 1s = 1 second

/**
 * \brief A simple Timer class to perform time measurement on CPU.
 *
 * This class supports multiple calls to start/stop.
 * When calling stop duration between the last call of start and stop is accumulated in a internal
 * variable.
 * Accumulated time can be retrieve using elapsed method.
 * Timer can be reset (accumulated sets to zero)
 *
 */
class HostTimer
{
public:
  using timer_t = std::chrono::high_resolution_clock;
  using time_point_t = timer_t::time_point;
  using duration_ns_t = std::chrono::nanoseconds;

  //! default constructor, timing starts rightaway
  HostTimer()
    : m_start(timer_t::now())
    , m_total_time(0s)
  {}

  //! destructor
  ~HostTimer() = default;

  //! start time measure
  void
  start()
  {
    m_start = timer_t::now();
  }

  //! stop time measure and add result to total_time
  void
  stop()
  {
    m_total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(timer_t::now() - m_start);
  }

  //! return elapsed time in seconds (converted from total_time in nanoseconds)
  double
  elapsed() const
  {
    return static_cast<double>(m_total_time.count()) * 1e-9;
  }

  void
  reset()
  {
    m_total_time = 0s;
  }

protected:
  //! store start time point
  time_point_t m_start;

  //! store total accumulated durations in nanoseconds
  duration_ns_t m_total_time;

}; // class HostTimer

} // namespace kalypsso

#endif // KALYPSSO_UTILS_MONITORING_HOSTTIMER_H_
