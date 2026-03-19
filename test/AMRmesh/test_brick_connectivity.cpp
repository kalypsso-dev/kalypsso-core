// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file test_brick_connectivity.cpp
 */

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <kalypsso/core/brick_connectivity_utils.h>

#ifdef KALYPSSO_CORE_USE_CNPY
#  include <kalypsso/core/cnpy_io.h>
#endif // KALYPSSO_CORE_USE_CNPY

#include <cstdlib>

namespace kalypsso
{

/* ============================================================ */
/* ============================================================ */
/* ============================================================ */
template <size_t dim>
void
run_test(brick_size_t<dim> brick_sizes)
{

  BrickConnectivityData<dim> brick_conn_data(brick_sizes);

  auto num_trees = brick_conn_data.m_num_trees;
  using coord_view_t =
    Kokkos::View<int *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  coord_view_t coord_view(brick_conn_data.m_brick_coords.data(),
                          dim * static_cast<size_t>(num_trees));

  if constexpr (dim == 2)
  {
#ifdef KALYPSSO_CORE_USE_CNPY
    save_cnpy(Kokkos::subview(coord_view, std::make_pair(0 * num_trees, 1 * num_trees)), "x_2d");
    save_cnpy(Kokkos::subview(coord_view, std::make_pair(1 * num_trees, 2 * num_trees)), "y_2d");
#endif
  }
  else if constexpr (dim == 3)
  {
#ifdef KALYPSSO_CORE_USE_CNPY
    save_cnpy(Kokkos::subview(coord_view, std::make_pair(0 * num_trees, 1 * num_trees)), "x_3d");
    save_cnpy(Kokkos::subview(coord_view, std::make_pair(1 * num_trees, 2 * num_trees)), "y_3d");
    save_cnpy(Kokkos::subview(coord_view, std::make_pair(2 * num_trees, 3 * num_trees)), "z_3d");
#endif
  }

  auto treeids = brick_conn_data.m_treeIds;
  auto coords = brick_conn_data.m_brick_coords;

  if constexpr (dim == 2)
  {
    for (int i = 0; i < num_trees; ++i)
    {
      printf("i=%5d treeId=%5d x=%5d y=%5d\n", i, treeids(i), coords(i, 0), coords(i, 1));
    }
  }
  if constexpr (dim == 3)
  {
    for (int i = 0; i < num_trees; ++i)
    {
      printf("i=%5d treeId=%5d x=%5d y=%5d z=%5d\n",
             i,
             treeids(i),
             coords(i, 0),
             coords(i, 1),
             coords(i, 2));
    }
  }

} // run_test

} // namespace kalypsso

// parse command line
bool
arg_exists(char ** begin, char ** end, const std::string & arg)
{
  return std::find(begin, end, arg) != end;
}

template <size_t dim>
decltype(auto)
get_brick_sizes(char ** begin, char ** end)
{
  KALYPSSO_DISABLE_CONVERSION_WARNINGS_PUSH()
  if constexpr (dim == 2)
  {
    kalypsso::brick_size_t<2> res{ -1, -1 };
    char **                   itr = std::find(begin, end, std::string("--2d"));
    if (itr != end && (itr + 1) != end && (itr + 2) != end)
    {
      res[0] = std::atoi(*(itr + 1));
      res[1] = std::atoi(*(itr + 2));
    }
    return res;
  }
  else if constexpr (dim == 3)
  {
    kalypsso::brick_size_t<3> res{ -1, -1, -1 };
    char **                   itr = std::find(begin, end, std::string("--3d"));
    if ((itr != end) and (itr + 1) != end and (itr + 2) != end and (itr + 3) != end)
    {
      res[0] = std::atoi(*(itr + 1));
      res[1] = std::atoi(*(itr + 2));
      res[2] = std::atoi(*(itr + 3));
    }
    return res;
  }
  KALYPSSO_DISABLE_CONVERSION_WARNINGS_POP()
}

// ======================================================
// ======================================================
// ======================================================
int
main(int argc, char ** argv)
{
  // initialize mpi, kokkos and p4est, use default MPI communicator : MPI_COMM_WORLD
  kalypsso::ParallelEnv par_env(argc, argv);

  // parse command line : 2d / 3d
  bool use_2d = arg_exists(argv, argv + argc, "--2d");
  bool use_3d = arg_exists(argv, argv + argc, "--3d");

  // make sure at least 2d is enable
  if (!use_3d and !use_2d)
    use_2d = true;

  {

    if (use_3d)
    {
      auto brick_sizes = get_brick_sizes<3>(argv, argv + argc);

      if (brick_sizes[0] <= 0 or brick_sizes[1] <= 0 or brick_sizes[2] <= 0)
      {
        printf("Wrong brick sizes\nExample cmdline: \n   ./test_brick_connectivity --2d 3 2\n\n");
        return -1;
      }

      // run a 3d test
      kalypsso::run_test<3>(brick_sizes);
    }
    else if (use_2d)
    {
      auto brick_sizes = get_brick_sizes<2>(argv, argv + argc);

      if (brick_sizes[0] <= 0 or brick_sizes[1] <= 0)
      {
        printf("Wrong brick sizes\nExample cmdline: \n   ./test_brick_connectivity --2d 3 2\n\n");
        return -1;
      }

      // run a 2d test
      kalypsso::run_test<2>(brick_sizes);
    }
  }

  return 0;
}
