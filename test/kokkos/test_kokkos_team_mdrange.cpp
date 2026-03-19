// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file test_kokkos_team_mdrange.cpp
 */

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>

#include <utility>  // for std::make_pair
#include <iostream> // for std::cout

// =======================================================================
// =======================================================================
template <typename device_t>
void
run_test(uint32_t nx, uint32_t ny, uint32_t nz)
{

  using exec_space = typename device_t::execution_space;

  using Data_t = Kokkos::View<int32_t ***, Kokkos::LayoutLeft, device_t>;

  /*
   * TestKokkosTeamMdrangeForFunctor
   *
   * Just playing with the Kokkos::Rank template parameters named OuterDir and InnerDir.
   * - InnerDir controls the mapping of the thread iteration ids within a tile (a tile maybe 2d, 3d,
   * ...)
   * - OuterDir controls the mapping of the tiles iteration index (the tiles set maybe 2d, 3d, ...)
   *
   * Here what we illustrate is the fact that: as Data_t is a 2d view with left layout, its best to
   * use InnerDir = Kokkos::Iterate::Left, OuterDir doesn't really matter here.
   */
  {
    std::cout << "// ======================================\n";
    std::cout << "Testing TestKokkosTeamMdrangeForFunctor with (nx,ny)= (" << nx << "," << ny << ","
              << nz << ")\n";
    std::cout << "// ======================================\n";

    // create and init test data
    auto data = Data_t("test_data", nx, ny, nz);

    using team_policy_t = Kokkos::TeamPolicy<exec_space, Kokkos::IndexType<int32_t>>;
    using thread_t = typename team_policy_t::member_type;

    Kokkos::parallel_for(
      "init_test_data - outer",
      team_policy_t(nz, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const thread_t & member) {
        uint32_t k = member.league_rank();

        Kokkos::parallel_for(
          Kokkos::TeamVectorMDRange<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                                    thread_t>(member, nx, ny),
          [=](uint32_t i, uint32_t j) { data(i, j, k) = i + j + k + 1; });
      });

    printf("Print values:\n");
    Kokkos::parallel_for(
      "init_test_data - outer",
      team_policy_t(nz, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const thread_t & member) {
        uint32_t k = member.league_rank();

        Kokkos::parallel_for(
          Kokkos::TeamVectorMDRange<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                                    thread_t>(member, nx, ny),
          [=](uint32_t i, uint32_t j) { printf("%d %d %d = %d\n", i, j, k, data(i, j, k)); });
      });

    printf("Print values of the inner 2x2x2 sub-domain:\n");
    Kokkos::parallel_for(
      "init_test_data - outer",
      team_policy_t(2, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const thread_t & member) {
        uint32_t k = member.league_rank();

        Kokkos::parallel_for(
          Kokkos::TeamVectorMDRange<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                                    thread_t>(member, 2, 2),
          [=](uint32_t i, uint32_t j) {
            printf("%d %d %d = %d\n", i + 1, j + 1, k + 1, data(i + 1, j + 1, k + 1));
          });
      });
  }
} // run_test

// =======================================================================
// =======================================================================
int
main(int argc, char * argv[])
{

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

  } // end kokkos config

  // using DefaultDevice =
  //   Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;

#ifdef KOKKOS_ENABLE_SERIAL
  using DefaultDevice = Kokkos::Serial;
  // Kokkos::Device<Kokkos::Serial::execution_space, Kokkos::Serial::memory_space>;

  uint32_t nx = 4;
  uint32_t ny = 4;
  uint32_t nz = 4;

  run_test<DefaultDevice>(nx, ny, nz);
#endif

  Kokkos::finalize();

  return EXIT_SUCCESS;
}
