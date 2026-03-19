// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file CheckEdgeSiblingsConnectivity.cpp
 */
#include <kalypsso/core/CheckEdgeSiblingsConnectivity.h>

namespace kalypsso
{

namespace core
{

/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/
template <size_t dim, typename device_t>
CheckEdgeSiblingsConnectivity<dim, device_t>::CheckEdgeSiblingsConnectivity(
  block_size_t<dim>  bSize,
  StencilHelper_t    stencil_helper,
  orchard_key_view_t orchard_keys,
  AMRMeshInfo        amr_mesh_info,
  ConfigMap const &  config_map)
  : m_block_sizes(bSize)
  , m_stencil_helper(stencil_helper)
  , m_orchard_keys_device(orchard_keys)
  , m_amr_mesh_info(amr_mesh_info)
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_xyz_min(get_xyz_min<dim>(config_map))
  , m_edge_flat_index_offsets(compute_edge_flat_index_offsets_emf<dim>(bSize))
{} // constructor

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
CheckEdgeSiblingsConnectivity<dim, device_t>::apply(block_size_t<dim>        bSize,
                                                    amr_hashmap_t            amr_hashmap,
                                                    orchard_key_view_t       orchard_keys,
                                                    AMRMeshInfo              amr_mesh_info,
                                                    ConfigMap const &        config_map,
                                                    brick_size_t<dim>        brick_sizes,
                                                    Kokkos::Array<bool, dim> is_brick_periodic)
{

  auto stencil_helper =
    StencilHelper_t(amr_hashmap, orchard_keys, bSize, brick_sizes, is_brick_periodic);

  CheckEdgeSiblingsConnectivity<dim, device_t> functor(
    bSize, stencil_helper, orchard_keys, amr_mesh_info, config_map);

  // number of owned quadrant x total number of edges (along Z only in 2d, or along all dir in 3D)
  // be careful emf is sized upon block_size + 1 in all direction, but
  // actually it is only required in the edge transverse directions, not in the edge longitudinal
  // direction
  const auto edge_flat_index_offsets = compute_edge_flat_index_offsets_emf<dim>(bSize);
  const auto nbIterations = amr_mesh_info.local_num_quadrants() * edge_flat_index_offsets[3];

  // launch computation
  Kokkos::parallel_for("kalypsso::core::CheckEdgeSiblingsConnectivity",
                       Kokkos::RangePolicy<exec_space>(0, nbIterations),
                       functor);

} // apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION bool
CheckEdgeSiblingsConnectivity<dim, device_t>::are_location_not_near(
  Kokkos::Array<real_t, dim> const & xyz0,
  Kokkos::Array<real_t, dim> const & xyz1) const
{
  bool problem = (fabs(xyz0[IX] - xyz1[IX]) > SMALL_F) or (fabs(xyz0[IY] - xyz1[IY]) > SMALL_F);
  if constexpr (dim == 3)
  {
    problem = problem or (fabs(xyz0[IZ] - xyz1[IZ]) > SMALL_F);
  }

  return problem;
}

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
CheckEdgeSiblingsConnectivity<dim, device_t>::expect_near_xyz(
  EdgeLocation<dim> const &          edge_loc0,
  Kokkos::Array<real_t, dim> const & xyz0,
  Kokkos::Array<real_t, dim> const & xyz1) const
{
  if (are_location_not_near(xyz0, xyz1))
  {
    if constexpr (dim == 2)
    {
      KOKKOS_IF_ON_HOST((KALYPSSO_ERROR("INVALID edge sibling location found at : i={} j={} "
                                        "edge_dir={} iOct={} | xyz0={} {} | xyz1={} {}",
                                        edge_loc0.ijk[IX],
                                        edge_loc0.ijk[IY],
                                        edge_loc0.ijk[dim],
                                        edge_loc0.iOct,
                                        xyz0[IX],
                                        xyz0[IY],
                                        xyz1[IX],
                                        xyz1[IY]);))
      KOKKOS_IF_ON_DEVICE((Kokkos::printf("INVALID edge sibling location found at: i=%d j=%d "
                                          "edge_dir=%d iOct=%d | xyz0=%f %f | xyz1=%f %f\n",
                                          edge_loc0.ijk[IX],
                                          edge_loc0.ijk[IY],
                                          edge_loc0.ijk[dim],
                                          edge_loc0.iOct,
                                          xyz0[IX],
                                          xyz0[IY],
                                          xyz1[IX],
                                          xyz1[IY]);))
    }
    else if constexpr (dim == 3)
    {
      KOKKOS_IF_ON_HOST((KALYPSSO_ERROR("INVALID edge sibling location found at : i={} j={} k={} "
                                        "edge_dir={} iOct={} | xyz0={} {} {} | xyz1={} {} {}",
                                        edge_loc0.ijk[IX],
                                        edge_loc0.ijk[IY],
                                        edge_loc0.ijk[IZ],
                                        edge_loc0.ijk[dim],
                                        edge_loc0.iOct,
                                        xyz0[IX],
                                        xyz0[IY],
                                        xyz0[IZ],
                                        xyz1[IX],
                                        xyz1[IY],
                                        xyz1[IZ]);))
      KOKKOS_IF_ON_DEVICE((Kokkos::printf("INVALID edge sibling location found at: i=%d j=%d k=%d "
                                          "edge_dir=%d iOct=%d | xyz0=%f %f %f | xyz1=%f %f %f\n",
                                          edge_loc0.ijk[IX],
                                          edge_loc0.ijk[IY],
                                          edge_loc0.ijk[IZ],
                                          edge_loc0.ijk[dim],
                                          edge_loc0.iOct,
                                          xyz0[IX],
                                          xyz0[IY],
                                          xyz0[IZ],
                                          xyz1[IX],
                                          xyz1[IY],
                                          xyz1[IZ]);))
    }
  }
} // expect_near_xyz

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
CheckEdgeSiblingsConnectivity<dim, device_t>::expect_near_same_level(
  EdgeLocation<dim> const & edge_loc0,
  EdgeLocation<dim> const & edge_loc1) const
{

  const auto key0 = m_stencil_helper.key(edge_loc0.iOct);
  const auto key1 = m_stencil_helper.key(edge_loc1.iOct);

  const auto xyz0 = orchard_key_to_edgecenter_real_space<dim>(
    key0, edge_loc0.ijk, m_block_sizes, m_scaling_factor, m_xyz_min);
  const auto xyz1 = orchard_key_to_edgecenter_real_space<dim>(
    key1, edge_loc1.ijk, m_block_sizes, m_scaling_factor, m_xyz_min);

  expect_near_xyz(edge_loc0, xyz0, xyz1);

} // expect_near_same_level

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
CheckEdgeSiblingsConnectivity<dim, device_t>::expect_near_finer_level(
  EdgeLocation<dim> const & edge_loc,
  EdgeLocation<dim> const & edge_loc0) const
{
  if constexpr (dim == 3)
  {

    const auto key = m_stencil_helper.key(edge_loc.iOct);
    const auto xyz = orchard_key_to_edgecenter_real_space<dim>(
      key, edge_loc.ijk, m_block_sizes, m_scaling_factor, m_xyz_min);

    // get xyz in the 2 small edges
    const auto key0 = m_stencil_helper.key(edge_loc0.iOct);

    KOKKOS_ASSERT((edge_loc0.level() > edge_loc.level()) && "edge_loc0 must be at finer AMR level");

    const auto xyz0a = orchard_key_to_edgecenter_real_space<dim>(
      key0, edge_loc0.ijk, m_block_sizes, m_scaling_factor, m_xyz_min);

    // unit vector along edge direction
    const auto v2 = get_unit_vector<dim>(edge_loc0.ijk[dim]);

    auto edge_loc0b = edge_loc0;
    edge_loc0b.ijk[IX] += v2[IX];
    edge_loc0b.ijk[IY] += v2[IY];
    edge_loc0b.ijk[IZ] += v2[IZ]; // legal since in we are in 3d
    const auto xyz0b = orchard_key_to_edgecenter_real_space<dim>(
      key0, edge_loc0b.ijk, m_block_sizes, m_scaling_factor, m_xyz_min);
    // middle of the 2 small edges
    const auto xyz0 = 0.5 * (xyz0a + xyz0b);

    // current and and the middle of the 2 small edges should be near
    expect_near_xyz(edge_loc, xyz, xyz0);
  }
} // expect_near_finer_level

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
CheckEdgeSiblingsConnectivity<dim, device_t>::check(index_t const & edge_flatindex,
                                                    int32_t const & iOct) const
{

  auto const edge_indexes =
    edge_flat_index_unravel_emf<dim>(edge_flatindex, m_block_sizes, m_edge_flat_index_offsets);

  if constexpr (dim == 2)
  {
    KOKKOS_ASSERT(edge_indexes[dim] == IZ && "EMF can only be along Z in 2D");
  }

  // current block AMR key
  const auto key_cur = m_stencil_helper.key(iOct);

  // create an edge location for current edge's cell (just round down edge index in transverse
  // direction)
  const EdgeLocation_t edge_loc{ edge_indexes, key_cur, iOct, false };
  const auto           level = edge_loc.level();

  // get sibling edge locations
  const auto edge_loc0 = m_stencil_helper.getEdgeSiblingLoc(edge_loc, 0);
  const auto edge_loc1 = m_stencil_helper.getEdgeSiblingLoc(edge_loc, 1);
  const auto edge_loc2 = m_stencil_helper.getEdgeSiblingLoc(edge_loc, 2);

  const auto v =
    get_edge_outside_unit_vector<dim>(edge_indexes, m_block_sizes, EdgeNormalType::DIAGONAL);

  auto norm_v = v[IX] * v[IX] + v[IY] * v[IY];
  if constexpr (dim == 3)
    norm_v += v[IZ] * v[IZ];

  // determine if emf at current edge location must be corrected
  // i.e. if neighbor is at finer AMR level
  //
  // NB: an outside neighbor is always at same AMR level as current block

  const bool is_not_at_domain_border =
    !m_stencil_helper.is_edge_location_at_domain_border(edge_loc);

  if (norm_v > 0 and is_not_at_domain_border)
  {
    const bool isNotOutside0 =
      edge_loc0.iOct < (m_amr_mesh_info.local_num_quadrants() + m_amr_mesh_info.local_num_ghosts());

    if (edge_loc0.is_valid and isNotOutside0)
    {
      if (edge_loc0.level() == level)
      {
        expect_near_same_level(edge_loc, edge_loc0);
      }
      if constexpr (dim == 3)
      {
        if (edge_loc0.level() > level)
        {
          expect_near_finer_level(edge_loc, edge_loc0);
        }
      }
    }

    const bool isNotOutside1 =
      edge_loc1.iOct < (m_amr_mesh_info.local_num_quadrants() + m_amr_mesh_info.local_num_ghosts());

    if (edge_loc1.is_valid and isNotOutside1)
    {
      if (edge_loc1.level() == level)
      {
        expect_near_same_level(edge_loc, edge_loc1);
      }
      if constexpr (dim == 3)
      {
        if (edge_loc1.level() > level)
        {
          expect_near_finer_level(edge_loc, edge_loc1);
        }
      }
    }

    const bool isNotOutside2 =
      edge_loc2.iOct < (m_amr_mesh_info.local_num_quadrants() + m_amr_mesh_info.local_num_ghosts());

    if (edge_loc2.is_valid and isNotOutside2)
    {
      if (edge_loc2.level() == level)
      {
        expect_near_same_level(edge_loc, edge_loc2);
      }
      if constexpr (dim == 3)
      {
        if (edge_loc2.level() > level)
        {
          expect_near_finer_level(edge_loc, edge_loc2);
        }
      }
    }

  } // end norm_v > 0

} // check

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
CheckEdgeSiblingsConnectivity<dim, device_t>::operator()(const index_t & global_index) const
{

  const auto num_edges = m_edge_flat_index_offsets[3];

  // retrieve local octant index
  auto const iOct_local = global_index / num_edges;
  auto const edge_flatindex = global_index - iOct_local * num_edges;

  check(edge_flatindex, iOct_local);

} // operator ()

// explicit template instantiation
template class CheckEdgeSiblingsConnectivity<2, kalypsso::DefaultDevice>;
template class CheckEdgeSiblingsConnectivity<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso
