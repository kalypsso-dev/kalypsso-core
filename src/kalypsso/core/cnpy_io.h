// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file cnpy_io.h
 * \brief converting numpy files to/from a Kokkos::View
 */
#ifndef KALYPSSO_CORE_CNPY_IO_H_
#define KALYPSSO_CORE_CNPY_IO_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <vector>
#include <sstream>
#include <string>
#include <iomanip> // for std::setfill

#include <cnpy.h>

namespace kalypsso
{

// =============================================================================
// =============================================================================
/**
 * Save a multidimensional Kokkos::View into a file using numpy format.
 *
 * example of use in python:
 *   data = np.load('data.npy')
 *
 * \tparam View is a Kokkos::View class, we check that the view is accessible from host
 *
 * \param[in] v a View instance containing a data array to be saved
 * \param[in] prefix_name is filename prefix
 */
template <class View>
void
save_cnpy(View v, std::string prefix_name)
{
  constexpr bool view_accessible_from_host =
    Kokkos::SpaceAccessibility</*AccessSpace=*/Kokkos::HostSpace,
                               /*MemorySpace=*/typename View::memory_space>::accessible;

  if constexpr (view_accessible_from_host)
  {

    // save in numpy format
    std::vector<size_t> shape;
    if constexpr (View::rank >= 1)
      shape.push_back(v.extent(0));
    if constexpr (View::rank >= 2)
      shape.push_back(v.extent(1));
    if constexpr (View::rank >= 3)
      shape.push_back(v.extent(2));

    std::ostringstream ss;
    ss << prefix_name << ".npy";
    cnpy::npy_save(ss.str(), v.data(), shape, "w");
  }

} // save_cnpy

// =============================================================================
// =============================================================================
/**
 * Save a multidimensional Kokkos::View into a file using numpy format.
 *
 * example of use in python:
 *   data = np.load('data.npy')
 *
 * \tparam View is a Kokkos::View class, we check that the view is accessible from host
 *
 * \param[in] v a View instance containing a data array to be saved
 * \param[in] prefix_name is filename prefix
 * \param[in] par_env is the parallel environment (to access MPI_Comm)
 */
template <class View>
void
save_cnpy(View v, std::string prefix_name, const ParallelEnv & par_env)
{
  constexpr bool view_accessible_from_host =
    Kokkos::SpaceAccessibility</*AccessSpace=*/Kokkos::HostSpace,
                               /*MemorySpace=*/typename View::memory_space>::accessible;

  if constexpr (view_accessible_from_host)
  {

    // save in numpy format
    std::vector<size_t> shape;
    if constexpr (View::rank >= 1)
      shape.push_back(v.extent(0));
    if constexpr (View::rank >= 2)
      shape.push_back(v.extent(1));
    if constexpr (View::rank >= 3)
      shape.push_back(v.extent(2));

    std::ostringstream ss;
    ss << prefix_name + "_mpi_";
    ss << std::setw(5) << std::setfill('0') << par_env.rank() << ".npy";
    cnpy::npy_save(ss.str(), v.data(), shape, "w");
  }

} // save_cnpy

} // namespace kalypsso

#endif // KALYPSSO_CORE_CNPY_IO_H_
