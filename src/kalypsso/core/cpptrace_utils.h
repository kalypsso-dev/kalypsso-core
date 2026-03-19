// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file cpptrace_utils.h
 *
 * Basically define the signal handler needed to print stack trace in case current application
crashes.
 *
 * The code is adapted from cpptrace sources :
 * https://github.com/jeremy-rifkin/cpptrace/blob/main/test/signal_demo.cpp
 * provided by cpptrace under MIT license.
 *
 * See https://github.com/jeremy-rifkin/cpptrace/blob/main/docs/signal-safe-tracing.md
*/
#ifndef KALYPSSO_CORE_CPPTRACE_UTILS_H_
#define KALYPSSO_CORE_CPPTRACE_UTILS_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <sys/wait.h>
#include <cstring>
#include <signal.h>
#include <csignal>
#include <unistd.h>

#ifdef KALYPSSO_CORE_USE_CPPTRACE
#  include <cpptrace/cpptrace.hpp>
#endif

namespace kalypsso
{

// This is just a utility I like, it makes the pipe API more expressive.
struct pipe_t
{
  struct io_t
  {
    int read_end;
    int write_end;
  };

  union
  {
    io_t io;
    int  data[2];
  };
};

// =======================================================
// =======================================================
void
do_signal_safe_trace(cpptrace::frame_ptr * buffer, std::size_t count);

// =======================================================
// =======================================================
void
segv_handler(int signo, siginfo_t * info, void * context);

// =======================================================
// =======================================================
void
install_segv_handler();

// =======================================================
// =======================================================
void
warmup_cpptrace();

// =======================================================
// =======================================================
void
cpptrace_initialize();

} // namespace kalypsso

#endif // KALYPSSO_CORE_CPPTRACE_UTILS_H_
