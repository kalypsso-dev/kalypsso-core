// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file kalypsso_stack_printer.cpp
 *
 * The code is adapted from cpptrace sources :
 * https://github.com/jeremy-rifkin/cpptrace/blob/main/test/signal_tracer.cpp
 * provided by cpptrace under MIT license.
 *
 * See https://github.com/jeremy-rifkin/cpptrace/blob/main/docs/signal-safe-tracing.md
 *
 */

#include <unistd.h>

#include <cstdio>
#include <iostream>

#include <cpptrace/cpptrace.hpp>

int
main()
{
  cpptrace::object_trace trace;
  while (true)
  {
    cpptrace::safe_object_frame frame;
    std::size_t                 res = fread(&frame, sizeof(frame), 1, stdin);
    if (res == 0)
    {
      break;
    }
    else if (res != 1)
    {
      std::cerr << "Oops, size mismatch " << res << " " << sizeof(frame) << std::endl;
      break;
    }
    else
    {
      trace.frames.push_back(frame.resolve());
    }
  }
  trace.resolve().print();
}
