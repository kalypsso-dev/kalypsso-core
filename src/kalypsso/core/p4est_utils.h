// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file p4est_utils.h
 *
 */
#ifndef KALYPSSO_CORE_P4EST_UTILS_H_
#define KALYPSSO_CORE_P4EST_UTILS_H_

#include <kalypsso/core/kalypsso_core_base.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/utils/p4est/p4est_wrapper.h>

namespace kalypsso
{

// =========================================================================
// =========================================================================
/**
 * Create a Kokkos::View on host containing the quadrant Id for each local tree.
 *
 * The result view has view equal to the number of trees present in current MPI process.
 * \param[in] the p4est object.
 *
 * \note this function was written at some early stage, but is currently not used. delete ?
 *
 */
template <size_t dim, typename device_t>
decltype(auto)
init_first_quadid_per_tree(forest_t<dim> * forest)
{
  using quadid_view_t = typename Kokkos::View<uint32_t *, device_t>;
  using quadid_view_host_t =
    typename Kokkos::View<uint32_t *, typename quadid_view_t::host_mirror_space>;

  uint32_t first_tree = forest->first_local_tree;
  uint32_t last_tree = forest->last_local_tree;

  quadid_view_host_t local_first_quad_id_per_tree_host(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "local_first_quad_id_per_tree_host"),
    last_tree - first_tree + 1);

  // loop over all local trees, to accumulate (scan exclusive scan)
  // the number of quadrant
  for (size_t treeid = first_tree; treeid <= last_tree; ++treeid)
  {
    tree_t<dim> * tree = p4est::Wrapper<dim>::tree_array_index(forest->trees, treeid);
    if (treeid == first_tree)
    {
      local_first_quad_id_per_tree_host(treeid - first_tree) = tree->quadrants.elem_counts;
    }
    else
    {
      local_first_quad_id_per_tree_host(treeid - first_tree) =
        local_first_quad_id_per_tree_host(treeid - first_tree - 1) + tree->quadrants.elem_counts;
    }
  }

  return local_first_quad_id_per_tree_host;

} // init_first_quadid_per_tree

} // namespace kalypsso
#endif // KALYPSSO_CORE_P4EST_UTILS_H_
