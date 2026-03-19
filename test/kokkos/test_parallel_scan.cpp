// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * This executable is used to test kokkos parallel_scan
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>
#include <cstdint>

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/enums.h>

#ifdef KALYPSSO_CORE_USE_MPI
#  include <mpi.h>
#endif // KALYPSSO_CORE_USE_MPI
#include <kalypsso/utils/mpi/ParallelEnv.h>

// ================================================================================
// ================================================================================
/*
 *
 * Main test using scheme order as template parameter.
 * order is the number of solution points per direction.
 *
 */
template <typename exec_space>
void
test_parallel_scan(int N)
{
  // initialize data
  using data_t = Kokkos::View<int64_t *, exec_space>;
  auto data = data_t("some data", N);
  Kokkos::parallel_for(
    "InitializeData", Kokkos::RangePolicy<exec_space>(0, N), KOKKOS_LAMBDA(const int & i) {
      data(i) = i % 2 ? 1 : -1;
    });

  auto data_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data);

  auto scanned_data = data_t("scanned_data", N);

  int64_t result;

  Kokkos::parallel_scan(
    "Parallel_scan",
    Kokkos::RangePolicy<exec_space>(0, N),
    KOKKOS_LAMBDA(const int & i, int64_t & partial_sum, const bool is_final) {
      if (is_final)
      {
        scanned_data(i) = partial_sum;
      }

      partial_sum += data(i);
    },
    result);

  auto scanned_data_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scanned_data);

  for (int i = 0; i < N; ++i)
    printf("[%d] %ld %ld\n", i, data_host(i), scanned_data_host(i));

} // test_parallel_scan

// ================================================================================
// ================================================================================
/*
 *
 * variant test: just performing 2 interleaved scans
 *
 */
template <typename exec_space>
struct CustomScan
{
  using exec_space_t = exec_space;
  using data_t = Kokkos::View<int64_t *, exec_space>;
  data_t data_in, data_out;

  CustomScan(data_t data1, data_t data2)
    : data_in(data1)
    , data_out(data2)
  {}

  static void
  run(data_t data1, data_t data2)
  {
    auto       functor = CustomScan<exec_space>(data1, data2);
    const auto N = data1.size();

    Kokkos::parallel_scan("CustomScan", Kokkos::RangePolicy<exec_space>(0, N), functor);
  }

  struct Res
  {
    int64_t res1;
    int64_t res2;
    Res()
      : res1(0)
      , res2(0)
    {}
  };

  KOKKOS_INLINE_FUNCTION
  void
  init(Res & update) const
  {
    update.res1 = 0;
    update.res2 = 0;
  }

  KOKKOS_INLINE_FUNCTION
  void
  join(Res & update, const Res & input) const
  {
    update.res1 += input.res1;
    update.res2 += input.res2;
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const size_t i, Res & update, const bool is_final) const
  {
    if (i % 2 == 0)
    {
      if (is_final)
      {
        data_out(i) = update.res1;
      }
      update.res1 += data_in(i);
    }
    else
    {
      if (is_final)
      {
        data_out(i) = update.res2;
      }
      update.res2 += data_in(i);
    }
  }
}; // struct CustomScan

// ================================================================================
// ================================================================================
template <typename exec_space>
void
test_parallel_custom_scan(int N)
{
  // initialize data
  using data_t = Kokkos::View<int64_t *, exec_space>;
  auto data = data_t("some data", N);
  Kokkos::parallel_for(
    "InitializeData", Kokkos::RangePolicy<exec_space>(0, N), KOKKOS_LAMBDA(const int & i) {
      // data(i) = i % 2 ? 1 : -1;
      data(i) = 2 * i;
      if (i % 5 == 0)
        data(i) = 0;
    });

  auto data_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data);

  auto scanned_data = data_t("scanned_data", N);

  CustomScan<Kokkos::DefaultExecutionSpace>::run(data, scanned_data);

  auto scanned_data_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scanned_data);

  for (int i = 0; i < N; ++i)
    printf("[%d] %ld %ld\n", i, data_host(i), scanned_data_host(i));

} // test_parallel_custom_scan

// ================================================================================
// ================================================================================
/*
 * A custom scan operation where input data will be modified only if it is zero.
 * In that case, data will be replaced by the last non-zero value
 *
 */
template <typename exec_space>
struct CustomScan2
{

  using exec_space_t = exec_space;
  using data_t = Kokkos::View<int64_t *, exec_space>;
  data_t data;
  using value_t = typename data_t::value_type;

  CustomScan2(data_t data_)
    : data(data_)
  {}

  static void
  run(data_t data_)
  {
    const auto N = data_.size();

    Kokkos::parallel_scan(
      "CustomScan2", Kokkos::RangePolicy<exec_space>(0, N), CustomScan2<exec_space>(data_));
  }

  KOKKOS_INLINE_FUNCTION
  void
  join(value_t & update, const value_t & input) const
  {
    if (input > update)
      update = input;
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const size_t i, value_t & update, const bool is_final) const
  {
    value_t value = data(i);
    if (value != 0)
    {
      update = value;
    }
    else if (is_final)
    {
      data(i) = value_t(update);
    }
  }
}; // struct CustomScan2

// ================================================================================
// ================================================================================
template <typename exec_space>
void
test_parallel_custom_scan2(int N)
{
  // initialize data
  using data_t = Kokkos::View<int64_t *, exec_space>;
  auto data = data_t("some data", N);
  Kokkos::parallel_for(
    "InitializeData", Kokkos::RangePolicy<exec_space>(0, N), KOKKOS_LAMBDA(const int & i) {
      // data(i) = i % 2 ? 1 : -1;
      data(i) = N - 2 * i;
      if (i % 5 == 0)
        data(i) = 0;
    });

  auto data_host = Kokkos::create_mirror(Kokkos::HostSpace{}, data);
  Kokkos::deep_copy(data_host, data);

  CustomScan2<Kokkos::DefaultExecutionSpace>::run(data);

  auto scanned_data_host = Kokkos::create_mirror(Kokkos::HostSpace{}, data);
  Kokkos::deep_copy(scanned_data_host, data);

  for (int i = 0; i < N; ++i)
    printf("[%d] %ld %ld\n", i, data_host(i), scanned_data_host(i));

} // test_parallel_custom_scan

// ================================================================================
// ================================================================================
// ================================================================================
int
main(int argc, char * argv[])
{

  const int N = argc > 1 ? std::atoi(argv[1]) : 100;

  {
    kalypsso::ParallelEnv par_env(argc, argv);
    // instantiate tests
    std::cout << "==================================================\n";
    std::cout << "Regular scan:\n";
    test_parallel_scan<Kokkos::DefaultExecutionSpace>(N);

    std::cout << "==================================================\n";
    std::cout << "Custom scan:\n";
    test_parallel_custom_scan<Kokkos::DefaultExecutionSpace>(N);

    std::cout << "==================================================\n";
    std::cout << "Custom scan2:\n";
    test_parallel_custom_scan2<Kokkos::DefaultExecutionSpace>(N);
  }

  return EXIT_SUCCESS;
}
