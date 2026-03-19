// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file AMRmesh_utils.h
 *
 */
#ifndef KALYPSSO_CORE_AMRMESH_UTILS_H
#define KALYPSSO_CORE_AMRMESH_UTILS_H

#include <kalypsso/utils/p4est/p4est_wrapper.h>

#include <array>

namespace kalypsso
{

// ===========================================================
// ===========================================================
/**
 * Compute coordinates in vertex space (i.e. connectivity space) of a point
 * from its logical coordinates in [0,1[^dim (given as double) relative to the
 * tree it belongs to.
 *
 * This is only valid when no geometry (in p4est sense) is involved.
 *
 * \param[in] forest p4est mesh data
 * \param[in] which_tree tree id in which octant is
 * \param[in] eta logical coordinates vector, all coordinate are in [0,1],
 *                [0,1]^Dim maps the current tree.
 *
 * \return vector of coordinates in real space
 */
template <size_t dim>
std::array<double, 3>
logical2vertex(typename p4est::Wrapper<dim>::forest_t * forest,
               p4est::topidx_t                          which_tree,
               std::array<double, 3>                    eta)
{
  using p4est_t = typename p4est::Wrapper<dim>;

  // return value : coordinates after mapping
  // from logical space coord (eta)
  // to connectivity space coord (xyz)
  std::array<double, 3> xyz;

  // number of corners of a tree (4 in 2D, 8 in 3D)
  constexpr auto NB_CHILDREN = p4est_t::NB_CHILDREN;

  // array of vertex index for all NB_CHILDREN vertices
  // (corners of current tree)
  p4est::topidx_t vt[NB_CHILDREN];

  // vertices coordinates
  const double *          v = forest->connectivity->vertices;
  const p4est::topidx_t * tree_to_vertex = forest->connectivity->tree_to_vertex;

  double & eta_x = eta[0];
  double & eta_y = eta[1];
  double & eta_z = eta[2];

  // get vertex id for current tree
  for (int k = 0; k < NB_CHILDREN; ++k)
  {
    vt[k] = tree_to_vertex[which_tree * NB_CHILDREN + k];
  }

  for (int j = 0; j < 3; ++j)
  {
    /* *INDENT-OFF* */
    // clang-format off
    xyz[static_cast<size_t>(j)] =
       (1. - eta_z) * ((1. - eta_y) * ((1. - eta_x) * v[3 * vt[0] + j] +
                                             eta_x  * v[3 * vt[1] + j]) +
                             eta_y  * ((1. - eta_x) * v[3 * vt[2] + j] +
                                             eta_x  * v[3 * vt[3] + j]));
    if constexpr (dim == 3)
    {
      xyz[static_cast<size_t>(j)] +=
             eta_z  * ((1. - eta_y) * ((1. - eta_x) * v[3 * vt[4] + j] +
                                             eta_x  * v[3 * vt[5] + j]) +
                             eta_y  * ((1. - eta_x) * v[3 * vt[6] + j] +
                                             eta_x  * v[3 * vt[7] + j]));
    }
    /* *INDENT-ON* */
    // clang-format on
  }

  return xyz;

} // logical2vertex

} // namespace kalypsso

#endif // KALYPSSO_CORE_AMRMESH_UTILS_H
