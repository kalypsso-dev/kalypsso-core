// SPDX-FileCopyrightText: 2025 kalypsso-core authors
// © Commissariat a l'Energie Atomique et aux Energies Alternatives (CEA)
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <gtest/gtest.h>

#include "../../main_kalypsso_unittest.h"

namespace kalypsso
{

TEST(kalypsso_utils_mpi_ParallelEnv_test, num_ranks)
{
  ParallelEnv * par_env_ptr = *g_par_env_ptr;

#ifdef KALYPSSO_CORE_USE_MPI
  // here we assume this test was run with 2 MPI ranks
  EXPECT_EQ(par_env_ptr->nRanks(), 2) << "ParallelEnv rank failed";
#else
  EXPECT_EQ(par_env_ptr->nRanks(), 1) << "ParallelEnv rank failed";
#endif
}

} // namespace kalypsso
