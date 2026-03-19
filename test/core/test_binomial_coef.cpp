// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/BinomialCoef.h>

#include <iostream>
#include <cstdlib>

namespace kalypsso
{

template <int N>
void
test_binomial_coef()
{
  // print Pascal triangle
  for (int n = 1; n <= N; ++n)
  {
    for (int k = 0; k <= n; ++k)
      printf("%d ", binomial_coef(n, k));
    printf("\n");
  }

  auto coef = BinomialCoef<N>();
  for (int n = 1; n <= N; ++n)
  {
    for (int k = 0; k <= n; ++k)
      printf("%d ", coef(n, k));
    printf("\n");
  }
}

} // namespace kalypsso

// ===========================================================================
// ===========================================================================
// ===========================================================================
int
main(int argc, char * argv[])
{
  // using DefaultDevice =
  //   Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;
  //  using device = DefaultDevice;

  /*
   * Initialize kokkos (host + device)
   *
   * If CUDA is enabled, Kokkos will try to use the default GPU,
   * i.e. GPU #0 if you have multiple GPUs.
   */
  Kokkos::initialize(argc, argv);

  {
    std::cout << "##########################\n";
    std::cout << "KOKKOS CONFIG             \n";
    std::cout << "##########################\n";

    std::ostringstream msg;
    std::cout << "Kokkos configuration" << std::endl;
    if (Kokkos::hwloc::available())
    {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count() << "] x CORE["
          << Kokkos::hwloc::get_available_cores_per_numa() << "] x HT["
          << Kokkos::hwloc::get_available_threads_per_core() << "] )" << std::endl;
    }
    Kokkos::print_configuration(msg);
    std::cout << msg.str();
    std::cout << "##########################\n";
  }

  constexpr int N = 17;
  printf("Computing binomial coefficients for N=%d\n", N);
  kalypsso::test_binomial_coef<N>();

  Kokkos::finalize();

  return EXIT_SUCCESS;

} // end main
