// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file CheckFaceBorderCompatibility.cpp
 */
#include <kalypsso/core/CheckFaceBorderCompatibility.h>

namespace kalypsso
{

namespace core
{

/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/
template <size_t dim, typename device_t>
CheckFaceBorderCompatibility<dim, device_t>::CheckFaceBorderCompatibility(
  FaceDataArrayBlock_t const & facedata,
  StencilHelper_t const &      stencil_helper,
  orchard_key_view_t const &   orchard_keys,
  AMRMeshInfo const &          amr_mesh_info,
  const int                    mpi_comm_rank,
  ConfigMap const &            config_map)
  : m_facedata(facedata)
  , m_stencil_helper(stencil_helper)
  , m_orchard_keys_device(orchard_keys)
  , m_amr_mesh_info(amr_mesh_info)
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_mpi_comm_rank(mpi_comm_rank)
{} // constructor

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
CheckFaceBorderCompatibility<dim, device_t>::apply(
  FaceDataArrayBlock_t const &     facedata,
  amr_hashmap_t const &            amr_hashmap,
  orchard_key_view_t const &       orchard_keys,
  AMRMeshInfo const &              amr_mesh_info,
  ConfigMap const &                config_map,
  brick_size_t<dim> const &        brick_sizes,
  Kokkos::Array<bool, dim> const & is_brick_periodic,
  ParallelEnv const &              par_env)
{

  auto stencil_helper = StencilHelper_t(
    amr_hashmap, orchard_keys, facedata.cell_block_size_inner(), brick_sizes, is_brick_periodic);

  CheckFaceBorderCompatibility<dim, device_t> functor(
    facedata, stencil_helper, orchard_keys, amr_mesh_info, par_env.rank(), config_map);

  // number of owned quadrant x number of surface cells
  const auto num_surface_cells = get_number_of_surface_cells(facedata.cell_block_size_inner());
  const auto nbIterations = amr_mesh_info.local_num_quadrants() * num_surface_cells;

  // launch computation
  Kokkos::parallel_for("kalypsso::core::CheckFaceBorderCompatibility",
                       Kokkos::RangePolicy<exec_space>(0, nbIterations),
                       functor);

} // apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
CheckFaceBorderCompatibility<dim, device_t>::check(index_t const & surface_flatindex,
                                                   int32_t const & iOct) const
{
  // auto const ijk =
  //   surface_flatindex_unravel_to_cell_ijk(surface_flatindex, m_facedata.cell_block_size_inner());

  auto const normal =
    surface_flatindex_to_normal_vector(surface_flatindex, m_facedata.cell_block_size_inner());

  auto const ijk_face =
    surface_flatindex_to_face_multiindex(surface_flatindex, m_facedata.cell_block_size_inner());

  // current block AMR key
  const auto key_cur = m_stencil_helper.key(iOct);

  // create a face location for current face
  const auto face_loc = FaceLocation_t(ijk_face, key_cur, iOct, false);

  // create a face location for neighbor
  auto         face_loc_neigh = m_stencil_helper.getNeighLoc(face_loc, normal);
  auto const & iOct_neigh = face_loc_neigh.iOct;

  // if octant id is above local_num_quadrants, it means neigh octant is not an owned quadrant, so
  // we don't check it (check for it would require applying border conditions beforehand, but we
  // can't enforce that here.)
  // if (iOct_neigh >= m_amr_mesh_info.local_num_quadrants())
  //  return;

  const auto face_dir = static_cast<size_t>(ijk_face[dim]);

  // get a face location at neighbor location
  // we only explore 2 cases:
  // - when neighbor is at same AMR level
  // - when neighbor is at finer AMR level
  // In any of these two cases, we need to "symmetrize" the face normal coordinate
  face_loc_neigh.ijk[face_dir] = m_facedata.cell_block_size_inner()[face_dir] - ijk_face[face_dir];

  // only check when neighbor is at same AMR level or at finer level
  if (face_loc_neigh.level() == face_loc.level())
  {

    if (fabs(m_facedata(ijk_face, iOct) - m_facedata(face_loc_neigh.ijk, iOct_neigh)) > SMALL_F)
    {
      if constexpr (dim == 2)
      {
        KOKKOS_IF_ON_HOST(
          (KALYPSSO_ERROR(
             "[Conformal face]: INVALID value at face location : mpi_rank={} i={} j={} "
             "var={} iOct={} iOct_neigh={} current_value={} neighbor_value={} diff={}",
             m_mpi_comm_rank,
             ijk_face[IX],
             ijk_face[IY],
             ijk_face[dim],
             iOct,
             iOct_neigh,
             m_facedata(ijk_face, iOct),
             m_facedata(face_loc_neigh.ijk, iOct_neigh),
             fabs(m_facedata(ijk_face, iOct) - m_facedata(face_loc_neigh.ijk, iOct_neigh)));))
        KOKKOS_IF_ON_DEVICE(
          (Kokkos::printf(
             "[Conformal face]: INVALID value at face location : mpi_rank=%d i=%d j=%d "
             "var=%d iOct=%d iOct_neigh=%d current_value=%f neighbor_value=%f diff=%f\n",
             m_mpi_comm_rank,
             ijk_face[IX],
             ijk_face[IY],
             ijk_face[dim],
             iOct,
             iOct_neigh,
             m_facedata(ijk_face, iOct),
             m_facedata(face_loc_neigh.ijk, iOct_neigh),
             fabs(m_facedata(ijk_face, iOct) - m_facedata(face_loc_neigh.ijk, iOct_neigh)));))
      }
      else if constexpr (dim == 3)
      {
        KOKKOS_IF_ON_HOST(
          (KALYPSSO_ERROR(
             "[Conformal face]: INVALID value at face location : mpi_rank={} i={} j={} k={} "
             "var={} iOct={} iOct_neigh={} current_value={} neighbor_value={} diff={}",
             m_mpi_comm_rank,
             ijk_face[IX],
             ijk_face[IY],
             ijk_face[IZ],
             ijk_face[dim],
             iOct,
             iOct_neigh,
             m_facedata(ijk_face, iOct),
             m_facedata(face_loc_neigh.ijk, iOct_neigh),
             fabs(m_facedata(ijk_face, iOct) - m_facedata(face_loc_neigh.ijk, iOct_neigh)));))
        KOKKOS_IF_ON_DEVICE(
          (Kokkos::printf(
             "[Conformal face]: INVALID value at face location : mpi_rank=%d i=%d j=%d k=%d "
             "var=%d iOct=%d iOct_neigh=%d current_value=%f neighbor_value=%f diff=%f\n",
             m_mpi_comm_rank,
             ijk_face[IX],
             ijk_face[IY],
             ijk_face[IZ],
             ijk_face[dim],
             iOct,
             iOct_neigh,
             m_facedata(ijk_face, iOct),
             m_facedata(face_loc_neigh.ijk, iOct_neigh),
             fabs(m_facedata(ijk_face, iOct) - m_facedata(face_loc_neigh.ijk, iOct_neigh)));))
      }
    }
  }
  else if (face_loc_neigh.level() == face_loc.level() + 1)
  {
    // neighbor is at finer scale
    const auto fine_value =
      m_stencil_helper.compute_face_siblings_average(face_loc_neigh, m_facedata);
    const auto current_value = m_facedata(ijk_face, iOct);

    if (fabs(current_value - fine_value) > SMALL_F)
    {
      if constexpr (dim == 2)
      {
        KOKKOS_IF_ON_HOST(
          (KALYPSSO_ERROR(
             "[Non-conformal face]: INVALID value at face location : mpi_rank={} i={} j={} var={} "
             "iOct={} iOct_neigh={} current_value={} fine_value={} diff={}",
             m_mpi_comm_rank,
             ijk_face[IX],
             ijk_face[IY],
             ijk_face[dim],
             iOct,
             iOct_neigh,
             current_value,
             fine_value,
             fabs(current_value - fine_value));))
        KOKKOS_IF_ON_DEVICE(
          (Kokkos::printf(
             "[Non-conformal face]: INVALID value at face location : mpi_rank=%d i=%d j=%d var=%d "
             "iOct=%d iOct_neigh=%d current_value=%f fine_value=%f diff=%f\n",
             m_mpi_comm_rank,
             ijk_face[IX],
             ijk_face[IY],
             ijk_face[dim],
             iOct,
             iOct_neigh,
             current_value,
             fine_value,
             fabs(current_value - fine_value));))
      }
      else if constexpr (dim == 3)
      {
        KOKKOS_IF_ON_HOST(
          (KALYPSSO_ERROR(
             "[Non-conformal face]: INVALID value at face location : mpi_rank={} i={} j={} k={} "
             "var={} iOct={} iOct_neigh={} current_value={} fine_value={} diff={}",
             m_mpi_comm_rank,
             ijk_face[IX],
             ijk_face[IY],
             ijk_face[IZ],
             ijk_face[dim],
             iOct,
             iOct_neigh,
             current_value,
             fine_value,
             fabs(current_value - fine_value));))
        KOKKOS_IF_ON_DEVICE(
          (Kokkos::printf(
             "[Non-conformal face]: INVALID value at face location : mpi_rank=%d i=%d j=%d k=%d "
             "var=%d iOct=%d iOct_neigh=%d current_value=%f fine_value=%f diff=%f\n",
             m_mpi_comm_rank,
             ijk_face[IX],
             ijk_face[IY],
             ijk_face[IZ],
             ijk_face[dim],
             iOct,
             iOct_neigh,
             current_value,
             fine_value,
             fabs(current_value - fine_value));))
      }
    }
  }

} // check

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
CheckFaceBorderCompatibility<dim, device_t>::operator()(const index_t & global_index) const
{

  const auto num_surface_cells = get_number_of_surface_cells(m_facedata.cell_block_size_inner());

  // retrieve local octant index
  auto const iOct_local = global_index / num_surface_cells;
  auto const surface_cell_index = global_index - iOct_local * num_surface_cells;

  check(surface_cell_index, iOct_local);

} // operator ()

// explicit template instantiation
template class CheckFaceBorderCompatibility<2, kalypsso::DefaultDevice>;
template class CheckFaceBorderCompatibility<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso
