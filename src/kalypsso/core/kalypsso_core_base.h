// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file kalypsso_core_base.h
 *
 * Define utility macros inspired by p4est.
 *
 * \sa Please note that for logging we should use spdlog-based macro, defined in kalypsso_log.h.
 * spdlog-based macro should be preferred (they are more flexible (can be activated/deactivated at
 * compile-time and runtime).
 */
#ifndef KALYPSSO_CORE_KALYPSSO_CORE_BASE_H_
#define KALYPSSO_CORE_KALYPSSO_CORE_BASE_H_

#include <sc.h>
#include <assert.h>

#include <kalypsso/core/kalypsso_core_config.h>

#ifndef assertm
#  define assertm(exp, msg) assert(((void)msg, exp))
#endif

#ifdef KALYPSSO_CORE_ENABLE_DEBUG
#  define KALYPSSO_ASSERT(c) SC_CHECK_ABORT((c), "Assertion '" #c "'")
#  define KALYPSSO_EXECUTE_ASSERT_FALSE(expression)                      \
    do                                                                   \
    {                                                                    \
      int _kalypsso_i = (int)(expression);                               \
      SC_CHECK_ABORT(!_kalypsso_i, "Expected false: '" #expression "'"); \
    } while (0)
#  define KALYPSSO_EXECUTE_ASSERT_TRUE(expression)                     \
    do                                                                 \
    {                                                                  \
      int _kalypsso_i = (int)(expression);                             \
      SC_CHECK_ABORT(_kalypsso_i, "Expected true: '" #expression "'"); \
    } while (0)
#  define KALYPSSO_DEBUG_EXECUTE(expression) \
    do                                       \
    {                                        \
      (void)(expression);                    \
    } while (0)
#else
#  define KALYPSSO_ASSERT(c) SC_NOOP()
#  define KALYPSSO_EXECUTE_ASSERT_FALSE(expression) \
    do                                              \
    {                                               \
      (void)(expression);                           \
    } while (0)
#  define KALYPSSO_EXECUTE_ASSERT_TRUE(expression) \
    do                                             \
    {                                              \
      (void)(expression);                          \
    } while (0)
#  define KALYPSSO_DEBUG_EXECUTE(expression) SC_NOOP()
#endif

/* macros for memory allocation, will abort if out of memory */
/** allocate a \a t-array with \a n elements */
#define KALYPSSO_ALLOC(t, n) (t *)sc_malloc(kalypsso_package_id, (n) * sizeof(t))
/** allocate a \a t-array with \a n elements and zero */
#define KALYPSSO_ALLOC_ZERO(t, n) (t *)sc_calloc(kalypsso_package_id, (size_t)(n), sizeof(t))
/** reallocate the \a t-array \a p with \a n elements */
#define KALYPSSO_REALLOC(p, t, n) (t *)sc_realloc(kalypsso_package_id, (p), (n) * sizeof(t))
/** duplicate a string */
#define KALYPSSO_STRDUP(s) sc_strdup(kalypsso_package_id, (s))
/** free an allocated array */
#define KALYPSSO_FREE(p) sc_free(kalypsso_package_id, (p))

/* log helper macros */
#define KALYPSSO_GLOBAL_LOG(p, s) SC_GEN_LOG(kalypsso_package_id, SC_LC_GLOBAL, (p), (s))
#define KALYPSSO_LOG(p, s) SC_GEN_LOG(kalypsso_package_id, SC_LC_NORMAL, (p), (s))
void
KALYPSSO_GLOBAL_LOGF(int priority, const char * fmt, ...) __attribute__((format(printf, 2, 3)));
void
KALYPSSO_LOGF(int priority, const char * fmt, ...) __attribute__((format(printf, 2, 3)));
#ifndef __cplusplus
#  define KALYPSSO_GLOBAL_LOGF(p, f, ...) \
    SC_GEN_LOGF(kalypsso_package_id, SC_LC_GLOBAL, (p), (f), __VA_ARGS__)
#  define KALYPSSO_LOGF(p, f, ...) \
    SC_GEN_LOGF(kalypsso_package_id, SC_LC_NORMAL, (p), (f), __VA_ARGS__)
#endif

/* convenience global log macros will only print if identifier <= 0 */
#define KALYPSSO_GLOBAL_TRACE(s) KALYPSSO_GLOBAL_LOG(SC_LP_TRACE, (s))
#define KALYPSSO_GLOBAL_LDEBUG(s) KALYPSSO_GLOBAL_LOG(SC_LP_DEBUG, (s))
#define KALYPSSO_GLOBAL_VERBOSE(s) KALYPSSO_GLOBAL_LOG(SC_LP_VERBOSE, (s))
#define KALYPSSO_GLOBAL_INFO(s) KALYPSSO_GLOBAL_LOG(SC_LP_INFO, (s))
#define KALYPSSO_GLOBAL_STATISTICS(s) KALYPSSO_GLOBAL_LOG(SC_LP_STATISTICS, (s))
#define KALYPSSO_GLOBAL_PRODUCTION(s) KALYPSSO_GLOBAL_LOG(SC_LP_PRODUCTION, (s))
#define KALYPSSO_GLOBAL_ESSENTIAL(s) KALYPSSO_GLOBAL_LOG(SC_LP_ESSENTIAL, (s))
#define KALYPSSO_GLOBAL_LERROR(s) KALYPSSO_GLOBAL_LOG(SC_LP_ERROR, (s))
void
KALYPSSO_GLOBAL_TRACEF(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
void
KALYPSSO_GLOBAL_LDEBUGF(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
void
KALYPSSO_GLOBAL_VERBOSEF(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
void
KALYPSSO_GLOBAL_INFOF(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
void
KALYPSSO_GLOBAL_STATISTICSF(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
void
KALYPSSO_GLOBAL_PRODUCTIONF(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
void
KALYPSSO_GLOBAL_ESSENTIALF(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
void
KALYPSSO_GLOBAL_LERRORF(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
#ifndef __cplusplus
#  define KALYPSSO_GLOBAL_TRACEF(f, ...) KALYPSSO_GLOBAL_LOGF(SC_LP_TRACE, (f), __VA_ARGS__)
#  define KALYPSSO_GLOBAL_LDEBUGF(f, ...) KALYPSSO_GLOBAL_LOGF(SC_LP_DEBUG, (f), __VA_ARGS__)
#  define KALYPSSO_GLOBAL_VERBOSEF(f, ...) KALYPSSO_GLOBAL_LOGF(SC_LP_VERBOSE, (f), __VA_ARGS__)
#  define KALYPSSO_GLOBAL_INFOF(f, ...) KALYPSSO_GLOBAL_LOGF(SC_LP_INFO, (f), __VA_ARGS__)
#  define KALYPSSO_GLOBAL_STATISTICSF(f, ...) \
    KALYPSSO_GLOBAL_LOGF(SC_LP_STATISTICS, (f), __VA_ARGS__)
#  define KALYPSSO_GLOBAL_PRODUCTIONF(f, ...) \
    KALYPSSO_GLOBAL_LOGF(SC_LP_PRODUCTION, (f), __VA_ARGS__)
#  define KALYPSSO_GLOBAL_ESSENTIALF(f, ...) KALYPSSO_GLOBAL_LOGF(SC_LP_ESSENTIAL, (f), __VA_ARGS__)
#  define KALYPSSO_GLOBAL_LERRORF(f, ...) KALYPSSO_GLOBAL_LOGF(SC_LP_ERROR, (f), __VA_ARGS__)
#endif
#define KALYPSSO_GLOBAL_NOTICE KALYPSSO_GLOBAL_STATISTICS
#define KALYPSSO_GLOBAL_NOTICEF KALYPSSO_GLOBAL_STATISTICSF

/* convenience log macros that are active on every processor */
// #ifndef KALYPSSO_CORE_USE_SPDLOG
// #  define KALYPSSO_TRACE(s) KALYPSSO_LOG(SC_LP_TRACE, (s))
// #  define KALYPSSO_LDEBUG(s) KALYPSSO_LOG(SC_LP_DEBUG, (s))
// #  define KALYPSSO_VERBOSE(s) KALYPSSO_LOG(SC_LP_VERBOSE, (s))
// #  define KALYPSSO_INFO(s) KALYPSSO_LOG(SC_LP_INFO, (s))
// #  define KALYPSSO_STATISTICS(s) KALYPSSO_LOG(SC_LP_STATISTICS, (s))
// #  define KALYPSSO_PRODUCTION(s) KALYPSSO_LOG(SC_LP_PRODUCTION, (s))
// #  define KALYPSSO_ESSENTIAL(s) KALYPSSO_LOG(SC_LP_ESSENTIAL, (s))
// #  define KALYPSSO_LERROR(s) KALYPSSO_LOG(SC_LP_ERROR, (s))
// #endif

void
KALYPSSO_TRACEF(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
void
KALYPSSO_LDEBUGF(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
void
KALYPSSO_VERBOSEF(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
void
KALYPSSO_INFOF(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
void
KALYPSSO_STATISTICSF(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
void
KALYPSSO_PRODUCTIONF(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
void
KALYPSSO_ESSENTIALF(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
void
KALYPSSO_LERRORF(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
#ifndef __cplusplus
#  define KALYPSSO_TRACEF(f, ...) KALYPSSO_LOGF(SC_LP_TRACE, (f), __VA_ARGS__)
#  define KALYPSSO_LDEBUGF(f, ...) KALYPSSO_LOGF(SC_LP_DEBUG, (f), __VA_ARGS__)
#  define KALYPSSO_VERBOSEF(f, ...) KALYPSSO_LOGF(SC_LP_VERBOSE, (f), __VA_ARGS__)
#  define KALYPSSO_INFOF(f, ...) KALYPSSO_LOGF(SC_LP_INFO, (f), __VA_ARGS__)
#  define KALYPSSO_STATISTICSF(f, ...) KALYPSSO_LOGF(SC_LP_STATISTICS, (f), __VA_ARGS__)
#  define KALYPSSO_PRODUCTIONF(f, ...) KALYPSSO_LOGF(SC_LP_PRODUCTION, (f), __VA_ARGS__)
#  define KALYPSSO_ESSENTIALF(f, ...) KALYPSSO_LOGF(SC_LP_ESSENTIAL, (f), __VA_ARGS__)
#  define KALYPSSO_LERRORF(f, ...) KALYPSSO_LOGF(SC_LP_ERROR, (f), __VA_ARGS__)
#endif
#define KALYPSSO_NOTICE KALYPSSO_STATISTICS
#define KALYPSSO_NOTICEF KALYPSSO_STATISTICSF

/* extern declarations */
/** the libsc package id for kalypsso (set in kalypsso_init()) */
extern int kalypsso_package_id;

static inline void
kalypsso_log_indent_push()
{
  sc_log_indent_push_count(kalypsso_package_id, 1);
}

static inline void
kalypsso_log_indent_pop()
{
  sc_log_indent_pop_count(kalypsso_package_id, 1);
}

/** Registers kalypsso with the SC Library and sets the logging behavior.
 * This function is optional.
 * This function must only be called before additional threads are created.
 * If this function is not called or called with log_handler == NULL,
 * the default SC log handler will be used.
 * If this function is not called or called with log_threshold == SC_LP_DEFAULT,
 * the default SC log threshold will be used.
 * The default SC log settings can be changed with sc_set_log_defaults ().
 */
void
kalypsso_init(sc_log_handler_t log_handler, int log_threshold);

#endif // KALYPSSO_CORE_KALYPSSO_CORE_BASE_H_
