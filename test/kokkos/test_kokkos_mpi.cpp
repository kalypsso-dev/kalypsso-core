// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * A simple MPI_Send/Recv just for cross-checking MPI implementation is cuda-aware.
 */

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>
#include <kalypsso/core/real_type.h> // choose between single and double precision

#include <utility>  // for std::make_pair
#include <iostream> // for std::cout

template <typename device_t>
void
run_test(int argc, char * argv[])
{
  using data_t = Kokkos::View<int *, device_t>;

  const int N = 100;

  // create parallel environment (p4est, MPI, kokkos, ...)
  kalypsso::ParallelEnv par_env(argc, argv);

  if (par_env.size() < 2)
  {
    if (par_env.rank() == 0)
      printf("ERROR: MPI communicator must have at least 2 MPI ranks.\n");
    return;
  }

  auto data0 = data_t("data0", N);
  auto data1 = data_t("data1", N);

  if (par_env.rank() == 0)
  {
    Kokkos::parallel_for(
      "init", Kokkos::RangePolicy<>(0, N), KOKKOS_LAMBDA(uint32_t i) { data0(i) = N - i; });
  }

  // send data0 from MPI rank0 to rank1
  if (par_env.rank() == 0)
  {
    MPI_Request req = par_env.comm().MPI_Isend(data0, 1, 911);
    par_env.comm().MPI_Waitall(1, &req);
  }
  else
  {
    MPI_Request req = par_env.comm().MPI_Irecv(data1, 0, 911);
    par_env.comm().MPI_Waitall(1, &req);
  }

  if (par_env.rank() == 1)
  {
    Kokkos::parallel_for(
      "cross-check", Kokkos::RangePolicy<>(0, N), KOKKOS_LAMBDA(uint32_t i) {
        printf("[MPI rank=1]: %d %d\n", i, data1(i));
      });
  }

} // run_test

// =======================================================================
// =======================================================================
int
main(int argc, char * argv[])
{
  run_test<kalypsso::DefaultDevice>(argc, argv);

  return EXIT_SUCCESS;
}
