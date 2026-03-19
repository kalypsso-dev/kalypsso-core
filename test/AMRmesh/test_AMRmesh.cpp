// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file test_AMRmesh.cpp
 */
#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/AMRmesh.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/cmdline_utils.h>

#include <kalypsso/utils/p4est/p4est_wrapper.h>
#include <kalypsso/utils/p4est/connectivity.h>
#include <kalypsso/utils/p4est/geometry.h>

#include <kalypsso/utils/config/ConfigMap.h>

#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <cstdlib>

namespace kalypsso
{

/* ============================================================ */
/* ============================================================ */
/* ============================================================ */
template <int dim>
void
run_test(const ParallelEnv & par_env, [[maybe_unused]] int argc, [[maybe_unused]] char * argv[])
{
  using namespace p4est;

  ConfigMap config_map =
    broadcast_parameters(dim == 2 ? "./test_AMRmesh_2d.ini" : "./test_AMRmesh_3d.ini");

  if (par_env.rank() == 0)
    printf("================================================\n");

  AMRmesh<dim> amr_mesh(par_env, config_map);

  auto forest = amr_mesh.forest();
  // auto conn   = amr_mesh.connectivity();
  auto geom = amr_mesh.geometry();

  auto conn_name = config_map.getString("amr", "connectivity", "invalid_connectivity");
  auto geom_name = geom->name;

  if (par_env.rank() == 0)
  {
    printf("Running a %dD test with connectivity %s and geometry %s\n",
           dim,
           conn_name.c_str(),
           geom_name);
  }

  printf("Forest has %ld global octants, %d local octants on MPI proc %d.\n",
         forest->global_num_quadrants,
         forest->local_num_quadrants,
         par_env.rank());

} // run_test

} // namespace kalypsso

// ======================================================
// ======================================================
// ======================================================
int
main(int argc, char ** argv)
{


  kalypsso::ParallelEnv par_env(argc, argv);

  if (kalypsso::cmdline_arg_exists(argv, argv + argc, "--help"))
  {
    if (par_env.rank() == 0)
    {
      // clang-format off
      std::cout << "Doing nothing but creating an instance of AMRmesh<dim> class.\n";
      std::cout << "Example cmdline: \"./test_AMRmesh\"\n";
      // clang-format on
    }
    return 0;
  }
  // run a 2d test
  kalypsso::run_test<2>(par_env, argc, argv);

  // run a 3d test
  kalypsso::run_test<3>(par_env, argc, argv);

  return 0;
}
