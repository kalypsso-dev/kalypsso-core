// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/utils/log/kalypsso_log.h>
#include <kalypsso/core/cmdline_utils.h>
#include <cstdio>
#include <iostream>

#include <kalypsso/utils/mpi/ParallelEnv.h>

namespace kalypsso
{

void
test_log()
{

  // log with the default logger
  SPDLOG_TRACE("Kalypsso trace log test {}", 3.141592);
  SPDLOG_DEBUG("Kalypsso debug log test {}", 3.141592);
  SPDLOG_INFO("Kalypsso info log test {}", 3.141592);
  SPDLOG_WARN("Kalypsso warn log test {}", 3.141592);
  SPDLOG_ERROR("Kalypsso error log test {}", 3.141592);
  SPDLOG_CRITICAL("Kalypsso critical log test {}", 3.141592);

} // test_log

void
test_kalypsso_log_master()
{

  // log with the default logger
  KALYPSSO_TRACE("Kalypsso trace log test {}", 3.141592);
  KALYPSSO_DEBUG("Kalypsso debug log test {}", 3.141592);
  KALYPSSO_INFO("Kalypsso info log test {}", 3.141592);
  KALYPSSO_WARN("Kalypsso warn log test {}", 3.141592);
  KALYPSSO_ERROR("Kalypsso error log test {}", 3.141592);
  KALYPSSO_CRITICAL("Kalypsso critical log test {}", 3.141592);

} // test_kalypsso_log_master

void
test_kalypsso_log_all()
{

  // log with the default logger
  KALYPSSO_TRACE_ALL("Kalypsso trace log test {}", 3.141592);
  KALYPSSO_DEBUG_ALL("Kalypsso debug log test {}", 3.141592);
  KALYPSSO_INFO_ALL("Kalypsso info log test {}", 3.141592);
  KALYPSSO_WARN_ALL("Kalypsso warn log test {}", 3.141592);
  KALYPSSO_ERROR_ALL("Kalypsso error log test {}", 3.141592);
  KALYPSSO_CRITICAL_ALL("Kalypsso critical log test {}", 3.141592);

} // test_kalypsso_log_all

} // namespace kalypsso


int
main(int argc, char * argv[])
{
  kalypsso::ParallelEnv par_env(argc, argv);

#ifdef KALYPSSO_CORE_USE_SPDLOG
  kalypsso::kalypsso_spdlog_config(argc, argv, par_env.rank(), par_env.size());
#endif

  if (kalypsso::cmdline_arg_exists(argv, argv + argc, "--help"))
  {
    if (par_env.rank() == 0)
    {
      std::cout
        << "You can change log levels by setting environment variable SPDLOG_LEVEL.\n"
        << "kalypsso predefines two loggers: \"kalypsso_mpi_master\" and \"kalypsso_mpi_all\"\n"
        << "e.g. to turn off login in logger \"kalypsso_mpi_all\":\n"
        << "   mpirun -np 3 ./test_spdlog SPDLOG_LEVEL=off,kalypsso_mpi_master=trace\n"
        << "Available levels are: trace, debug, info, warn, error and critical\n";
    }
    return 0;
  }

  // kalypsso::test_log();
  kalypsso::test_kalypsso_log_master();
  kalypsso::test_kalypsso_log_all();

  return EXIT_SUCCESS;
}
