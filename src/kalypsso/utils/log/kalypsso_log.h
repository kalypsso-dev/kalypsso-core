// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file kalypsso_log.h
 *
 * the macro defined here only require that the main of application defines
 * two logger named "kalypsso_logger_master" and "kalypsso_logger_all"
 *
 * Minimal log level can be changed:
 * - at compile-time using cmake variable KALYPSSO_CORE_LOG_LEVEL (default is
 *      SPDLOG_LEVEL_TRACE)
 * - at run-time using env variable SPDLOG_LEVEL:
 *      export SPDLOG_LEVEL=trace
 */
#ifndef KALYPSSO_UTILS_LOG_KALYPSSO_LOG_H_
#define KALYPSSO_UTILS_LOG_KALYPSSO_LOG_H_

#ifdef KALYPSSO_CORE_USE_SPDLOG
#  include <spdlog/spdlog.h>
#  include <spdlog/cfg/env.h>  // for loading levels from the environment variable
#  include <spdlog/cfg/argv.h> // for loading levels from the command line
#  include <spdlog/sinks/stdout_sinks.h>
#  include <spdlog/sinks/stdout_color_sinks.h>
#endif

namespace kalypsso
{

extern int spdlog_mpi_rank;

/**
 * create and config default logger :
 * - one for logging only in MPI master process
 * - one for logging in all MPI processes
 */
void
kalypsso_spdlog_config(int & argc, char **& argv, int mpi_rank, int mpi_size);

} // namespace kalypsso


// slightly modified log macro to log only in master MPI rank (rank == 0)
#ifndef KALYPSSO_TRACE
#  if defined(KALYPSSO_CORE_USE_SPDLOG) && (SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_TRACE)
#    define KALYPSSO_LOGGER_TRACE(logger, ...)                         \
      if (spdlog_mpi_rank == 0)                                        \
      {                                                                \
        SPDLOG_LOGGER_CALL(logger, spdlog::level::trace, __VA_ARGS__); \
      }
#    define KALYPSSO_TRACE(...) \
      KALYPSSO_LOGGER_TRACE(spdlog::get("kalypsso_mpi_master"), __VA_ARGS__)
#  else
#    define KALYPSSO_LOGGER_TRACE(logger, ...) (void)0
#    define KALYPSSO_TRACE(...) (void)0
#  endif
#endif

#ifndef KALYPSSO_DEBUG
#  if defined(KALYPSSO_CORE_USE_SPDLOG) && (SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG)
#    define KALYPSSO_LOGGER_DEBUG(logger, ...)                         \
      if (spdlog_mpi_rank == 0)                                        \
      {                                                                \
        SPDLOG_LOGGER_CALL(logger, spdlog::level::debug, __VA_ARGS__); \
      }
#    define KALYPSSO_DEBUG(...) \
      KALYPSSO_LOGGER_DEBUG(spdlog::get("kalypsso_mpi_master"), __VA_ARGS__)
#  else
#    define KALYPSSO_LOGGER_DEBUG(logger, ...) (void)0
#    define KALYPSSO_DEBUG(...) (void)0
#  endif
#endif

#ifndef KALYPSSO_INFO
#  if defined(KALYPSSO_CORE_USE_SPDLOG) && (SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_INFO)
#    define KALYPSSO_LOGGER_INFO(logger, ...)                         \
      if (spdlog_mpi_rank == 0)                                       \
      {                                                               \
        SPDLOG_LOGGER_CALL(logger, spdlog::level::info, __VA_ARGS__); \
      }
#    define KALYPSSO_INFO(...) KALYPSSO_LOGGER_INFO(spdlog::get("kalypsso_mpi_master"), __VA_ARGS__)
#  else
#    define KALYPSSO_LOGGER_INFO(logger, ...) (void)0
#    define KALYPSSO_INFO(...) (void)0
#  endif
#endif

#ifndef KALYPSSO_WARN
#  if defined(KALYPSSO_CORE_USE_SPDLOG) && (SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_WARN)
#    define KALYPSSO_LOGGER_WARN(logger, ...)                         \
      if (spdlog_mpi_rank == 0)                                       \
      {                                                               \
        SPDLOG_LOGGER_CALL(logger, spdlog::level::warn, __VA_ARGS__); \
      }
#    define KALYPSSO_WARN(...) KALYPSSO_LOGGER_WARN(spdlog::get("kalypsso_mpi_master"), __VA_ARGS__)
#  else
#    define KALYPSSO_LOGGER_WARN(logger, ...) (void)0
#    define KALYPSSO_WARN(...) (void)0
#  endif
#endif

#ifndef KALYPSSO_ERROR
#  if defined(KALYPSSO_CORE_USE_SPDLOG) && (SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_ERROR)
#    define KALYPSSO_LOGGER_ERROR(logger, ...)                       \
      if (spdlog_mpi_rank == 0)                                      \
      {                                                              \
        SPDLOG_LOGGER_CALL(logger, spdlog::level::err, __VA_ARGS__); \
      }
#    define KALYPSSO_ERROR(...) \
      KALYPSSO_LOGGER_ERROR(spdlog::get("kalypsso_mpi_master"), __VA_ARGS__)
#  else
#    define KALYPSSO_LOGGER_ERROR(logger, ...) (void)0
#    define KALYPSSO_ERROR(...) (void)0
#  endif
#endif

#ifndef KALYPSSO_CRITICAL
#  if defined(KALYPSSO_CORE_USE_SPDLOG) && (SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_CRITICAL)
#    define KALYPSSO_LOGGER_CRITICAL(logger, ...)                         \
      if (spdlog_mpi_rank == 0)                                           \
      {                                                                   \
        SPDLOG_LOGGER_CALL(logger, spdlog::level::critical, __VA_ARGS__); \
      }
#    define KALYPSSO_CRITICAL(...) \
      KALYPSSO_LOGGER_CRITICAL(spdlog::get("kalypsso_mpi_master"), __VA_ARGS__)
#  else
#    define KALYPSSO_LOGGER_CRITICAL(logger, ...) (void)0
#    define KALYPSSO_CRITICAL(...) (void)0
#  endif
#endif

// slightly modified log macro : all MPI proc prints
#ifndef KALYPSSO_TRACE_ALL
#  if defined(KALYPSSO_CORE_USE_SPDLOG) && (SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_TRACE)
#    define KALYPSSO_LOGGER_TRACE_ALL(logger, ...) \
      SPDLOG_LOGGER_CALL(logger, spdlog::level::trace, __VA_ARGS__)
#    define KALYPSSO_TRACE_ALL(...) \
      KALYPSSO_LOGGER_TRACE_ALL(spdlog::get("kalypsso_mpi_all"), __VA_ARGS__)
#  else
#    define KALYPSSO_LOGGER_TRACE_ALL(logger, ...) (void)0
#    define KALYPSSO_TRACE_ALL(...) (void)0
#  endif
#endif

#ifndef KALYPSSO_DEBUG_ALL
#  if defined(KALYPSSO_CORE_USE_SPDLOG) && (SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG)
#    define KALYPSSO_LOGGER_DEBUG_ALL(logger, ...) \
      SPDLOG_LOGGER_CALL(logger, spdlog::level::debug, __VA_ARGS__)
#    define KALYPSSO_DEBUG_ALL(...) \
      KALYPSSO_LOGGER_DEBUG_ALL(spdlog::get("kalypsso_mpi_all"), __VA_ARGS__)
#  else
#    define KALYPSSO_LOGGER_DEBUG_ALL(logger, ...) (void)0
#    define KALYPSSO_DEBUG_ALL(...) (void)0
#  endif
#endif

#ifndef KALYPSSO_INFO_ALL
#  if defined(KALYPSSO_CORE_USE_SPDLOG) && (SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_INFO)
#    define KALYPSSO_LOGGER_INFO_ALL(logger, ...) \
      SPDLOG_LOGGER_CALL(logger, spdlog::level::info, __VA_ARGS__)
#    define KALYPSSO_INFO_ALL(...) \
      KALYPSSO_LOGGER_INFO_ALL(spdlog::get("kalypsso_mpi_all"), __VA_ARGS__)
#  else
#    define KALYPSSO_LOGGER_INFO_ALL(logger, ...) (void)0
#    define KALYPSSO_INFO_ALL(...) (void)0
#  endif
#endif

#ifndef KALYPSSO_WARN_ALL
#  if defined(KALYPSSO_CORE_USE_SPDLOG) && (SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_WARN)
#    define KALYPSSO_LOGGER_WARN_ALL(logger, ...) \
      SPDLOG_LOGGER_CALL(logger, spdlog::level::warn, __VA_ARGS__)
#    define KALYPSSO_WARN_ALL(...) \
      KALYPSSO_LOGGER_WARN_ALL(spdlog::get("kalypsso_mpi_all"), __VA_ARGS__)
#  else
#    define KALYPSSO_LOGGER_WARN_ALL(logger, ...) (void)0
#    define KALYPSSO_WARN_ALL(...) (void)0
#  endif
#endif

#ifndef KALYPSSO_ERROR_ALL
#  if defined(KALYPSSO_CORE_USE_SPDLOG) && (SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_ERROR)
#    define KALYPSSO_LOGGER_ERROR_ALL(logger, ...) \
      SPDLOG_LOGGER_CALL(logger, spdlog::level::err, __VA_ARGS__)
#    define KALYPSSO_ERROR_ALL(...) \
      KALYPSSO_LOGGER_ERROR_ALL(spdlog::get("kalypsso_mpi_all"), __VA_ARGS__)
#  else
#    define KALYPSSO_LOGGER_ERROR_ALL(logger, ...) (void)0
#    define KALYPSSO_ERROR_ALL(...) (void)0
#  endif
#endif

#ifndef KALYPSSO_CRITICAL_ALL
#  if defined(KALYPSSO_CORE_USE_SPDLOG) && (SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_CRITICAL)
#    define KALYPSSO_LOGGER_CRITICAL_ALL(logger, ...) \
      SPDLOG_LOGGER_CALL(logger, spdlog::level::critical, __VA_ARGS__)
#    define KALYPSSO_CRITICAL_ALL(...) \
      KALYPSSO_LOGGER_CRITICAL_ALL(spdlog::get("kalypsso_mpi_all"), __VA_ARGS__)
#  else
#    define KALYPSSO_LOGGER_CRITICAL_ALL(logger, ...) (void)0
#    define KALYPSSO_CRITICAL_ALL(...) (void)0
#  endif
#endif

#endif // KALYPSSO_UTILS_LOG_KALYPSSO_LOG_H_
