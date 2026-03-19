// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KALYPSSO_UNIT_TEST_MAIN_H_
#define KALYPSSO_UNIT_TEST_MAIN_H_

#include "gtest/gtest.h"

#include <Kokkos_Core.hpp>
#include "kalypsso/utils/mpi/ParallelEnv.h"

namespace kalypsso
{
extern ParallelEnv ** g_par_env_ptr;

extern int *    global_argc;
extern char *** global_argv;
} // namespace kalypsso

namespace
{
class GtestParEnv : public ::testing::Environment
{
public:
  void
  SetUp() override
  {
    if (m_par_env_ptr == nullptr)
    {
      // handle both MPI and Kokkos resources initialization
      m_par_env_ptr = new kalypsso::ParallelEnv(*kalypsso::global_argc, *kalypsso::global_argv);
      kalypsso::g_par_env_ptr = &m_par_env_ptr;
    }
  }
  void
  TearDown() override
  {
    if (m_par_env_ptr != nullptr)
    {
      delete m_par_env_ptr;
      kalypsso::g_par_env_ptr = nullptr;
    }
  }
  kalypsso::ParallelEnv * m_par_env_ptr;

}; // class GtestParEnv

} // namespace

#endif // KALYPSSO_UNIT_TEST_MAIN_H_
