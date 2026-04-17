// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file test_kokkos_mdrange.cpp
 */

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>

#include <utility>  // for std::make_pair
#include <iostream> // for std::cout

// =======================================================================
// =======================================================================
template <typename device_t>
void
run_test(uint32_t nx, uint32_t ny)
{

  using exec_space = typename device_t::execution_space;

  using Data_t = Kokkos::View<int32_t **, Kokkos::LayoutLeft, device_t>;
  // using DataHost_t = typename Data_t::host_mirror_type;

  /*
   * TestKokkosMdrangeForFunctor
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
    std::cout << "Testing TestKokkosMdrangeForFunctor with (nx,ny)= (" << nx << "," << ny << ")\n";
    std::cout << "// ======================================\n";

    // create and init test data
    auto data = Data_t("test_data", nx, ny);
    Kokkos::parallel_for(
      "init_test_data",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({ 0, 0 }, { nx, ny }),
      KOKKOS_LAMBDA(uint32_t i, uint32_t j) { data(i, j) = i + j + 1; });

    // Kokkos::fence();

    printf("print using default layouts\n");

    Kokkos::parallel_for(
      "print_results",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({ 0, 0 }, { nx, ny }),
      KOKKOS_LAMBDA(uint32_t i, uint32_t j) {
        // std::cout << i << " " << j << " " << data(i, j) << "\n";
        printf("data(%d,%d)=%d at 0x%p\n", i, j, data(i, j), &(data(i, j)));
      });

    printf("print using outer layouts=left and inner layout=default\n");
    Kokkos::parallel_for(
      "print_results",
      Kokkos::MDRangePolicy<exec_space,
                            Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Default>>(
        { 0, 0 }, { nx, ny }),
      KOKKOS_LAMBDA(uint32_t i, uint32_t j) {
        // std::cout << i << " " << j << " " << data(i, j) << "\n";
        printf("data(%d,%d)=%d at 0x%p\n", i, j, data(i, j), &(data(i, j)));
      });

    printf("print using outer layouts=left and inner layout=left\n");
    Kokkos::parallel_for(
      "print_results",
      Kokkos::MDRangePolicy<exec_space,
                            Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>(
        { 0, 0 }, { nx, ny }),
      KOKKOS_LAMBDA(uint32_t i, uint32_t j) {
        // std::cout << i << " " << j << " " << data(i, j) << "\n";
        printf("data(%d,%d)=%d at 0x%p\n", i, j, data(i, j), &(data(i, j)));
      });

    printf("print using outer layouts=left and inner layout=right\n");
    Kokkos::parallel_for(
      "print_results",
      Kokkos::MDRangePolicy<exec_space,
                            Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Right>>(
        { 0, 0 }, { nx, ny }),
      KOKKOS_LAMBDA(uint32_t i, uint32_t j) {
        // std::cout << i << " " << j << " " << data(i, j) << "\n";
        printf("data(%d,%d)=%d at 0x%p\n", i, j, data(i, j), &(data(i, j)));
      });

    printf("print using outer layouts=right and inner layout=left\n");
    Kokkos::parallel_for(
      "print_results",
      Kokkos::MDRangePolicy<exec_space,
                            Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Left>>(
        { 0, 0 }, { nx, ny }),
      KOKKOS_LAMBDA(uint32_t i, uint32_t j) {
        // std::cout << i << " " << j << " " << data(i, j) << "\n";
        printf("data(%d,%d)=%d at 0x%p\n", i, j, data(i, j), &(data(i, j)));
      });

    printf("print using outer layouts=right and inner layout=right\n");
    Kokkos::parallel_for(
      "print_results",
      Kokkos::MDRangePolicy<exec_space,
                            Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>(
        { 0, 0 }, { nx, ny }),
      KOKKOS_LAMBDA(uint32_t i, uint32_t j) {
        // std::cout << i << " " << j << " " << data(i, j) << "\n";
        printf("data(%d,%d)=%d at 0x%p\n", i, j, data(i, j), &(data(i, j)));
      });

    printf("print using all layouts=right\n");
    Kokkos::parallel_for(
      "print_results",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2, Kokkos::Iterate::Right>>({ 0, 0 },
                                                                                 { nx, ny }),
      KOKKOS_LAMBDA(uint32_t i, uint32_t j) {
        // std::cout << i << " " << j << " " << data(i, j) << "\n";
        printf("data(%d,%d)=%d at 0x%p\n", i, j, data(i, j), &(data(i, j)));
      });

    printf("print using all layouts=left\n");
    Kokkos::parallel_for(
      "print_results",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2, Kokkos::Iterate::Left>>({ 0, 0 },
                                                                                { nx, ny }),
      KOKKOS_LAMBDA(uint32_t i, uint32_t j) {
        // std::cout << i << " " << j << " " << data(i, j) << "\n";
        printf("data(%d,%d)=%d at 0x%p\n", i, j, data(i, j), &(data(i, j)));
      });

    // Kokkos::fence();
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

  uint32_t nx = 5;
  uint32_t ny = 3;

  run_test<DefaultDevice>(nx, ny);
#endif

  Kokkos::finalize();

  return EXIT_SUCCESS;
}
