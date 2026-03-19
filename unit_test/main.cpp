// SPDX-FileCopyrightText: 2025 kalypsso-core authors
// © Commissariat a l'Energie Atomique et aux Energies Alternatives (CEA)
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <Kokkos_Core.hpp>

#include "gtest/gtest.h"

#include <kalypsso/utils/mpi/ParallelEnv.h>

#include "main_kalypsso_unittest.h"

namespace kalypsso
{
kalypsso::ParallelEnv ** g_par_env_ptr;

int *    global_argc = nullptr;
char *** global_argv = nullptr;

} // namespace kalypsso

// entry point
int
main(int argc, char * argv[])
{
  kalypsso::global_argc = &argc;
  kalypsso::global_argv = &argv;
  kalypsso::g_par_env_ptr = nullptr;

  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new GtestParEnv());

  return RUN_ALL_TESTS();
}
