// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file CudaBlockingTimer.h
 * \brief A simple timer class for CUDA based on events.
 *
 */
#ifndef KALYPSSO_UTILS_MONITORING_CUDABLOCKINGTIMER_H_
#define KALYPSSO_UTILS_MONITORING_CUDABLOCKINGTIMER_H_

#include <cuda_runtime.h>

namespace kalypsso
{

/**
 * \brief a simple timer for CUDA kernel using CUDA events.
 * \sa https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
 * CUDA kernels run asynchronously from CPU,
 * so don't use CPU timing routines.
 *
 * Beware that stopping timer imply a call cudaEventSynchronize, so the CPU will wait (block)
 * for the stop event event to happen; you won't be able to overlap computing on GPU with computing
 * on CPU. This timer is ok for benchmarking the execution of a given kernel.
 */
class CudaBlockingTimer
{
protected:
  //! CUDA start and stop events
  cudaEvent_t startEv, stopEv;

  //! total accumulated duration
  double total_time;

public:
  CudaBlockingTimer()
  {
    cudaEventCreate(&startEv);
    cudaEventCreate(&stopEv);
    total_time = 0.0;
  }

  ~CudaBlockingTimer()
  {
    cudaEventDestroy(startEv);
    cudaEventDestroy(stopEv);
  }

  //! start timer, push a start even in a cuda stream
  void
  start()
  {
    cudaEventRecord(startEv, 0);
  }

  //! reset accumulated duration
  void
  reset()
  {
    total_time = 0.0;
  }

  //! stop timer and accumulate time in seconds
  void
  stop()
  {
    float gpuTime;
    cudaEventRecord(stopEv, 0);
    cudaEventSynchronize(stopEv);

    // get elapsed time in milliseconds
    cudaEventElapsedTime(&gpuTime, startEv, stopEv);

    // accumulate duration in seconds
    total_time += (double)1e-3 * gpuTime;
  }

  //! return elapsed time in seconds (as record in total_time)
  double
  elapsed() const
  {
    return total_time;
  }

}; // class CudaBlockingTimer

} // namespace kalypsso

#endif // KALYPSSO_UTILS_MONITORING_CUDABLOCKINGTIMER_H_
