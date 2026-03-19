// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file kalypsso_log.cpp
 */
#include <kalypsso/utils/log/kalypsso_log.h>
#include <kalypsso/core/kalypsso_core_config.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <memory> // for shared_ptr

#ifdef KALYPSSO_CORE_USE_MPI
#  include <mpi.h>
#endif // KALYPSSO_CORE_USE_MPI

namespace kalypsso
{

//! return the number of digits to represent an integer
int
numDigits(int64_t n)
{
  if (n / 10 == 0)
    return 1;
  return 1 + numDigits(n / 10);
}

// should be properly initialized in kalypsso_spdlog_config
int spdlog_mpi_rank = -1;

void
kalypsso_spdlog_config(int & argc, char **& argv, int mpi_rank, int mpi_size)
{
  kalypsso::spdlog_mpi_rank = mpi_rank;

#ifdef KALYPSSO_CORE_USE_SPDLOG
  // create kalypsso_master logger - only MPI rank 0 involved
  auto kalypsso_logger_master = spdlog::stdout_color_mt("kalypsso_mpi_master");

#  ifdef KALYPSSO_CORE_ENABLE_DEBUG
  kalypsso_logger_master->set_pattern("[kalypsso master] [%^%l%$] [%s:%#] %v");
#  else
  kalypsso_logger_master->set_pattern("[kalypsso master] [%^%l%$] %v");
#  endif

  // create kalypsso_all logger - all MPI proc involved
  // logger name is built using MPI rank
  auto        kalypsso_logger_all = spdlog::stdout_color_mt("kalypsso_mpi_all");
  std::string kalypsso_logger_all_pattern = [=]() {
    std::stringstream ss;
#  ifdef KALYPSSO_CORE_ENABLE_DEBUG
    ss << "[kalypsso " << std::setw(numDigits(mpi_size)) << mpi_rank << "] [%^%l%$] [%s:%#] %v";
#  else
    ss << "[kalypsso " << std::setw(numDigits(mpi_size)) << mpi_rank << "] [%^%l%$] %v";
#  endif
    return ss.str();
  }();
  kalypsso_logger_all->set_pattern(kalypsso_logger_all_pattern);

  // default log_level, default is info (everything above is printed)
  spdlog::cfg::load_env_levels();

  // load log levels from command line arguments:
  spdlog::cfg::load_argv_levels(argc, argv);

  if (kalypsso::spdlog_mpi_rank == 0)
  {
    printf("================================================ \n");
    printf("START SPDLOG CONFIG\n");
    printf("================================================ \n");
    spdlog::apply_all([&](const std::shared_ptr<spdlog::logger> & logger) {
      if (logger->name().empty())
      {
        return; // ignore default logger
      }
      std::cout << logger->name() << ": " << logger->level() << "\n";
    });
    printf("compile-time value of SPDLOG_ACTIVE_LEVEL is %d\n", SPDLOG_ACTIVE_LEVEL);
    printf("================================================ \n");
    printf("END SPDLOG CONFIG\n");
    printf("================================================ \n");
  }
#endif
} // kalypsso_spdlog_config

} // namespace kalypsso
