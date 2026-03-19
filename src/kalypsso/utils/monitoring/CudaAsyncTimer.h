// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file CudaAsyncTimer.h
 * \brief A simple timer class for CUDA based on events.
 *
 */
#ifndef KALYPSSO_UTILS_MONITORING_CUDAASYNCTIMER_H_
#define KALYPSSO_UTILS_MONITORING_CUDAASYNCTIMER_H_

#include <cuda_runtime.h>

#include <cassert>

#ifndef assertm
#  define assertm(exp, msg) assert(((void)msg, exp))
#endif

namespace kalypsso
{

/**
 * \brief a simple timer for CUDA kernel.
 *
 * cudaEventSynchronize is called with some delay (asynchronously) to avoid too much synchronization
 * between CPU and GPU, and allow the CPU to perform useful stuff while the GPU is computing.
 *
 * The way to achieve this is to use an array of cuda events (start/stop), and use two indexes one
 * to write/record events, and the other to read/synchronize then and compute accumulated time.
 * These read/write indexes are always increasing; as they are stored on 64 bits integer the upper
 * limit is sufficiently high to be never reached in practice. We use the rest of division by the
 * size of the event array, to address the array.
 *
 */
class CudaAsyncTimer
{
protected:
  class CUDAEvent
  {
  public:
    CUDAEvent() { cudaEventCreate(&m_event); }

    CUDAEvent(const CUDAEvent &) = delete;

    ~CUDAEvent() { cudaEventDestroy(m_event); }

    void
    record()
    {
      auto res = cudaEventRecord(m_event, 0);
      assertm(res == cudaSuccess, "Recording cuda event failed");
    }

    bool
    query() const
    {
      return cudaEventQuery(m_event) == cudaSuccess;
    }

    void
    synchronize() const
    {
      cudaEventSynchronize(m_event);
    }
    cudaEvent_t
    event() const
    {
      return m_event;
    }

  private:
    cudaEvent_t m_event;
  };

  //! total number of pair of start/stop events
  static constexpr int NUM_EVENTS = 32;
  using idx_t = uint64_t;

  CUDAEvent startEv[NUM_EVENTS], stopEv[NUM_EVENTS];
  idx_t     readIdx, writeIdx;
  double    total_time;

public:
  // =======================================================
  // =======================================================
  CudaAsyncTimer()
  {
    readIdx = 0;
    writeIdx = 0;
    total_time = 0.0;
  }

  // =======================================================
  // =======================================================
  ~CudaAsyncTimer() {}

  // =======================================================
  // =======================================================
  /**
   * record start event and process past events that are completed.
   */
  void
  start()
  {
    [[maybe_unused]] const auto processed_events = process_events(false);

    startEv[writeIdx % NUM_EVENTS].record();
  }

  // =======================================================
  // =======================================================
  /**
   * record stop event.
   */
  void
  stop()
  {
    stopEv[writeIdx % NUM_EVENTS].record();
    assertm(writeIdx < std::numeric_limits<idx_t>::max(), "writeIdx is definitely too large");

    writeIdx++;
  }

  // =======================================================
  // =======================================================
  /**
   * reset internal state
   */
  void
  reset()
  {
    readIdx = 0;
    writeIdx = 0;
    total_time = 0.0;
  }

  // =======================================================
  // =======================================================
  /**
   * return elapsed time in seconds (as record in total_time)
   */
  double
  elapsed()
  {
    // flush events to process
    [[maybe_unused]] const auto processed_events = process_events(true);
    assertm(readIdx == writeIdx, "Wrong read/write index after processing all events.");
    return total_time;
  }

private:
  // =======================================================
  // =======================================================
  /**
   * Process a pair of start/stop event and accumulate time.
   *
   * \param[in] enforce_synchronization is true, CPU is blocked waiting for the stop event to happen
   *
   * When stop even is satisfied, total_time is updated, and the read index incremented.
   */
  void
  process_event(bool enforce_synchronize = true)
  {

    float gpuTime;

    if (enforce_synchronize)
      stopEv[readIdx % NUM_EVENTS].synchronize();
    cudaEventElapsedTime(
      &gpuTime, startEv[readIdx % NUM_EVENTS].event(), stopEv[readIdx % NUM_EVENTS].event());
    readIdx++;
    total_time += 1e-3 * static_cast<double>(gpuTime);

  } // process_event

  // =======================================================
  // =======================================================
  /**
   * Process pairs of start/stop events to accumulate time.
   *
   * \param[in] if flush is true, always call cudaEventSynchronize to make sure the stop event is
   *            completed.
   *
   * \return the number of events processed, i.e. events for which cudaEventQuery was successful.
   */
  int
  process_events(bool flush = false)
  {

    int            processed = 0;
    constexpr bool NO_SYNC = false;

    // if the event stack is almost full, enforce synchronization to process events
    // if flush is requested, enforce synchronization
    while ((writeIdx - readIdx > NUM_EVENTS - 4) or (flush and readIdx < writeIdx))
    {
      process_event();
      processed++;
    }

    // just query for stop events that are satisfied, and if so process them
    while (readIdx < writeIdx and stopEv[readIdx % NUM_EVENTS].query() == cudaSuccess)
    {
      process_event(NO_SYNC);
      processed++;
    }

    return processed;

  } // process_events

}; // class CudaAsyncTimer

} // namespace kalypsso

#endif // KALYPSSO_UTILS_MONITORING_CUDAASYNCTIMER_H_
