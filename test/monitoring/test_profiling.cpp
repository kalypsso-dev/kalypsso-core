// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * Just checking one can use nvtx annotations regardless Kokkos backend used (OpenMP or CUDA).
 */

#include <cstdlib>

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>

#include <kalypsso/utils/mpi/ParallelEnv.h>
#include <kalypsso/utils/monitoring/ProfilingManager.h>

namespace kalypsso
{
template <typename device_t>
void
run_test(ParallelEnv const & par_env)
{
  using exec_space = typename device_t::execution_space;
  using data_t = Kokkos::View<double *, device_t>;

  ProfilingManager profiling_mgr(par_env);

  profiling_mgr.get_whole_region().start();

  auto & timer =
    profiling_mgr.get_region("Kalypsso::Alloc", ProfilingRegion::TIMER_DEVICE, Color_t::FullBlue());
  timer.start();
  const int data_size = 1000000;
  data_t    data("some data", data_size);
  timer.stop();

  auto & timer2 = profiling_mgr.get_region(
    "Kalypsso::Compute", ProfilingRegion::TIMER_DEVICE, Color_t::FullRed());
  timer2.start();
  Kokkos::parallel_for(
    "test", Kokkos::RangePolicy<exec_space>(0, data_size), KOKKOS_LAMBDA(const int i) {
      data(i) = 4. * i * i - 8. * i + Kokkos::sin(5.2 * i);
    });
  timer2.stop();

  auto & timer3 = profiling_mgr.get_region(
    "Kalypsso::Result", ProfilingRegion::TIMER_DEVICE, Color_t::FullGreen());
  timer3.start();
  auto data_host = Kokkos::create_mirror_view(data);
  printf("data(%d)=%f\n", 42, data_host(42));
  timer3.stop();

  profiling_mgr.get_whole_region().stop();

  profiling_mgr.print_timings();

} // run_test

} // namespace kalypsso

// =============================================================================
// =============================================================================
// =============================================================================
int
main(int argc, char * argv[])
{
  // initialize mpi, kokkos and p4est, use default MPI communicator : MPI_COMM_WORLD
  kalypsso::ParallelEnv par_env(argc, argv);

  {
    kalypsso::run_test<kalypsso::DefaultDevice>(par_env);
  }

  return EXIT_SUCCESS;
}
