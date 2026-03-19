// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FillBlockGhostFaces.cpp
 */
#include <kalypsso/core/FillBlockGhostFaces.h>


namespace kalypsso
{

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
FillBlockGhostFacesFunctor<dim, device_t>::apply(ConfigMap const &        config_map,
                                                 amr_hashmap_t            amr_hashmap,
                                                 orchard_key_view_t       orchard_keys,
                                                 [[maybe_unused]] int32_t local_num_octants,
                                                 int32_t                  iOct_begin,
                                                 int32_t                  num_octants,
                                                 FaceDataArrayBlock_t     facedata_in,
                                                 FaceDataArrayBlock_t     facedata_out,
                                                 brick_size_t<dim>        brick_sizes,
                                                 Kokkos::Array<bool, dim> is_brick_periodic,
                                                 AMRMeshInfo              amr_mesh_info)
{

  // make sure the range of octants to process is valid
  KOKKOS_ASSERT(iOct_begin + num_octants <= local_num_octants &&
                "Invalid range of octants to process");

  // input array must be non-ghosted
  KOKKOS_ASSERT(facedata_in.is_ghosted() == false && "Input array must be non ghosted.");

  // output array ghost size must be even
  KOKKOS_ASSERT(((facedata_out.shift()[IX] & 0x1) == 0) &&
                "Output array must have a ghost width to be an even integer.");
  KOKKOS_ASSERT(((facedata_out.shift()[IY] & 0x1) == 0) &&
                "Output array must have a ghost width to be an even integer.");
  if constexpr (dim == 3)
  {
    KOKKOS_ASSERT(((facedata_out.shift()[IZ] & 0x1) == 0) &&
                  "Output array must have a ghost width to be an even integer.");
  }

  auto stencil_helper = StencilHelper_t(
    amr_hashmap, orchard_keys, facedata_in.cell_block_size_inner(), brick_sizes, is_brick_periodic);

  FillBlockGhostFacesFunctor<dim, device_t> functor(stencil_helper,
                                                    iOct_begin,
                                                    num_octants,
                                                    facedata_in,
                                                    facedata_out,
                                                    ProlongationParam(config_map),
                                                    amr_mesh_info);

  const auto num_elements_per_octant = facedata_out.num_elements_per_octant();

  Kokkos::parallel_for("FillBlockGhostFacesFunctor - AllButInternalFaces",
                       Kokkos::RangePolicy<exec_space, TagFillAllFacesButInternal>(
                         0, num_octants * num_elements_per_octant),
                       functor);

  Kokkos::parallel_for(
    "FillBlockGhostFacesFunctor - InternalFaces",
    Kokkos::RangePolicy<exec_space, TagFillInternalFaces>(0, num_octants * num_elements_per_octant),
    functor);

} // apply

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION bool
FillBlockGhostFacesFunctor<dim, device_t>::is_face_ambiguous(
  face_multiindex_t<dim> const & face_indexes,
  block_size_t<dim> const &      block_sizes) const
{
  auto const & ivar = face_indexes[dim];

  if (ivar == IX and (face_indexes[IX] == 0 or face_indexes[IX] == block_sizes[IX]))
  {
    return true;
  }
  if (ivar == IY and (face_indexes[IY] == 0 or face_indexes[IY] == block_sizes[IY]))
  {
    return true;
  }
  if constexpr (dim == 3)
  {
    if (ivar == IZ and (face_indexes[IZ] == 0 or face_indexes[IZ] == block_sizes[IZ]))
    {
      return true;
    }
  }
  return false;
}

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION Kokkos::Array<int8_t, dim>
                       FillBlockGhostFacesFunctor<dim, device_t>::convert_face_indexes_out_to_in(
  face_multiindex_t<dim> &       face_indexes_in,
  face_multiindex_t<dim> const & face_indexes_out) const
{

  const auto & ivar = face_indexes_out[dim];
  const auto & fb = m_facedata_in.face_block_size(ivar);
  const auto & b = m_block_sizes;
  //  const auto & g = m_ghost_sizes;

  auto dir = []() {
    if constexpr (dim == 2)
      return Kokkos::Array<int8_t, 2>{ 0, 0 };
    else if constexpr (dim == 3)
      return Kokkos::Array<int8_t, 3>{ 0, 0, 0 };
  }();

  // check if output cell is inside inner block, or in neighbor block
  if (face_indexes_out[IX] < 0)
  {
    dir[IX] = -1;
    face_indexes_in[IX] = face_indexes_out[IX] + b[IX];
  }
  else if (face_indexes_out[IX] >= fb[IX])
  {
    dir[IX] = 1;
    face_indexes_in[IX] = face_indexes_out[IX] - b[IX];
  }
  else
  {
    face_indexes_in[IX] = face_indexes_out[IX];
  }

  if (face_indexes_out[IY] < 0)
  {
    dir[IY] = -1;
    face_indexes_in[IY] = face_indexes_out[IY] + b[IY];
  }
  else if (face_indexes_out[IY] >= fb[IY])
  {
    dir[IY] = 1;
    face_indexes_in[IY] = face_indexes_out[IY] - b[IY];
  }
  else
  {
    face_indexes_in[IY] = face_indexes_out[IY];
  }

  if constexpr (dim == 3)
  {
    if (face_indexes_out[IZ] < 0)
    {
      dir[IZ] = -1;
      face_indexes_in[IZ] = face_indexes_out[IZ] + b[IZ];
    }
    else if (face_indexes_out[IZ] >= fb[IZ])
    {
      dir[IZ] = 1;
      face_indexes_in[IZ] = face_indexes_out[IZ] - b[IZ];
    }
    else
    {
      face_indexes_in[IZ] = face_indexes_out[IZ];
    }
  }

  face_indexes_in[dim] = face_indexes_out[dim];

  return dir;

} // convert_face_indexes_out_to_in

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostFacesFunctor<dim, device_t>::linear_extrapolate_using_limited_slopes(
  FaceLocation<2> const & face_loc_neigh,
  FaceLocation<2> const & face_loc_out,
  int32_t const &         iOct_local) const
{
  // doing a prolongation: the neighbor must be at coarse level
  KOKKOS_ASSERT((face_loc_neigh.level() + 1 == face_loc_out.level()) &&
                "Invalid face location for doing a prolongation");

  auto const & ivar = face_loc_neigh.ijk[dim];

  // get transverse slope
  const auto face_loc_left =
    ivar == IX ? m_stencil_helper.getNeighLoc(face_loc_neigh, m_stencil_helper.unit_shift(-YDIR))
               : m_stencil_helper.getNeighLoc(face_loc_neigh, m_stencil_helper.unit_shift(-XDIR));
  const auto face_loc_right =
    ivar == IX ? m_stencil_helper.getNeighLoc(face_loc_neigh, m_stencil_helper.unit_shift(+YDIR))
               : m_stencil_helper.getNeighLoc(face_loc_neigh, m_stencil_helper.unit_shift(+XDIR));

  // determine local position of current output face inside virtual parent cell using integer
  // coordinates in -1, +1
  auto const & ii = ivar == IX ? face_loc_out.ijk[IY] : face_loc_out.ijk[IX];
  const int    i = 2 * (ii - 2 * (ii / 2)) - 1;

  // compute limited slopes
  auto const dfdx = m_stencil_helper.compute_minmod_slopes_prolongation(
    face_loc_neigh, face_loc_right, face_loc_left, m_facedata_in);

  // extrapolate
  m_facedata_out(face_loc_out.ijk[IX], face_loc_out.ijk[IY], ivar, iOct_local) =
    m_facedata_in(face_loc_neigh.ijk[IX], face_loc_neigh.ijk[IY], ivar, face_loc_neigh.iOct) +
    ONE_FOURTH_F * static_cast<real_t>(i) * dfdx;

} // linear_extrapolate_using_limited_slopes - 2d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostFacesFunctor<dim, device_t>::linear_extrapolate_using_limited_slopes(
  FaceLocation<3> const & face_loc_neigh,
  FaceLocation<3> const & face_loc_out,
  int32_t const &         iOct_local) const
{
  // doing a prolongation: the neighbor must be at coarse level
  KOKKOS_ASSERT((face_loc_neigh.level() + 1 == face_loc_out.level()) &&
                "Invalid face location for doing a prolongation");

  auto const & ivar = face_loc_neigh.ijk[dim];

  if (ivar == IX)
  {
    // get transverse slopes
    const auto face_loc_left_y =
      m_stencil_helper.getNeighLoc(face_loc_neigh, m_stencil_helper.unit_shift(-YDIR));

    const auto face_loc_right_y =
      m_stencil_helper.getNeighLoc(face_loc_neigh, m_stencil_helper.unit_shift(+YDIR));

    const auto face_loc_left_z =
      m_stencil_helper.getNeighLoc(face_loc_neigh, m_stencil_helper.unit_shift(-ZDIR));

    const auto face_loc_right_z =
      m_stencil_helper.getNeighLoc(face_loc_neigh, m_stencil_helper.unit_shift(+ZDIR));

    // determine local position of current output face inside virtual parent cell using integer
    // coordinates in -1, +1
    auto const & jj = face_loc_out.ijk[IY];
    const int    j = 2 * (jj - 2 * (jj / 2)) - 1;

    auto const & kk = face_loc_out.ijk[IZ];
    const int    k = 2 * (kk - 2 * (kk / 2)) - 1;

    // compute limited slopes
    auto const dfdy = m_stencil_helper.compute_minmod_slopes_prolongation(
      face_loc_neigh, face_loc_right_y, face_loc_left_y, m_facedata_in);

    auto const dfdz = m_stencil_helper.compute_minmod_slopes_prolongation(
      face_loc_neigh, face_loc_right_z, face_loc_left_z, m_facedata_in);

    // extrapolate
    m_facedata_out(
      face_loc_out.ijk[IX], face_loc_out.ijk[IY], face_loc_out.ijk[IZ], ivar, iOct_local) =
      m_facedata_in(face_loc_neigh.ijk[IX],
                    face_loc_neigh.ijk[IY],
                    face_loc_neigh.ijk[IZ],
                    ivar,
                    face_loc_neigh.iOct) +
      ONE_FOURTH_F * static_cast<real_t>(j) * dfdy + ONE_FOURTH_F * static_cast<real_t>(k) * dfdz;
  }
  else if (ivar == IY)
  {
    // get transverse slopes
    const auto face_loc_left_x =
      m_stencil_helper.getNeighLoc(face_loc_neigh, m_stencil_helper.unit_shift(-XDIR));

    const auto face_loc_right_x =
      m_stencil_helper.getNeighLoc(face_loc_neigh, m_stencil_helper.unit_shift(+XDIR));

    const auto face_loc_left_z =
      m_stencil_helper.getNeighLoc(face_loc_neigh, m_stencil_helper.unit_shift(-ZDIR));

    const auto face_loc_right_z =
      m_stencil_helper.getNeighLoc(face_loc_neigh, m_stencil_helper.unit_shift(+ZDIR));

    // determine local position of current output face inside virtual parent cell using integer
    // coordinates in -1, +1
    auto const & ii = face_loc_out.ijk[IX];
    const int    i = 2 * (ii - 2 * (ii / 2)) - 1;

    auto const & kk = face_loc_out.ijk[IZ];
    const int    k = 2 * (kk - 2 * (kk / 2)) - 1;

    // compute limited slopes
    auto const dfdx = m_stencil_helper.compute_minmod_slopes_prolongation(
      face_loc_neigh, face_loc_right_x, face_loc_left_x, m_facedata_in);

    auto const dfdz = m_stencil_helper.compute_minmod_slopes_prolongation(
      face_loc_neigh, face_loc_right_z, face_loc_left_z, m_facedata_in);

    // extrapolate
    m_facedata_out(
      face_loc_out.ijk[IX], face_loc_out.ijk[IY], face_loc_out.ijk[IZ], ivar, iOct_local) =
      m_facedata_in(face_loc_neigh.ijk[IX],
                    face_loc_neigh.ijk[IY],
                    face_loc_neigh.ijk[IZ],
                    ivar,
                    face_loc_neigh.iOct) +
      ONE_FOURTH_F * static_cast<real_t>(i) * dfdx + ONE_FOURTH_F * static_cast<real_t>(k) * dfdz;
  }
  else if (ivar == IZ)
  {
    // get transverse slopes
    const auto face_loc_left_x =
      m_stencil_helper.getNeighLoc(face_loc_neigh, m_stencil_helper.unit_shift(-XDIR));

    const auto face_loc_right_x =
      m_stencil_helper.getNeighLoc(face_loc_neigh, m_stencil_helper.unit_shift(+XDIR));

    const auto face_loc_left_y =
      m_stencil_helper.getNeighLoc(face_loc_neigh, m_stencil_helper.unit_shift(-YDIR));

    const auto face_loc_right_y =
      m_stencil_helper.getNeighLoc(face_loc_neigh, m_stencil_helper.unit_shift(+YDIR));

    // determine local position of current output face inside virtual parent cell using integer
    // coordinates in -1, +1
    auto const & ii = face_loc_out.ijk[IX];
    const int    i = 2 * (ii - 2 * (ii / 2)) - 1;

    auto const & jj = face_loc_out.ijk[IY];
    const int    j = 2 * (jj - 2 * (jj / 2)) - 1;

    // compute limited slopes
    auto const dfdx = m_stencil_helper.compute_minmod_slopes_prolongation(
      face_loc_neigh, face_loc_right_x, face_loc_left_x, m_facedata_in);

    auto const dfdy = m_stencil_helper.compute_minmod_slopes_prolongation(
      face_loc_neigh, face_loc_right_y, face_loc_left_y, m_facedata_in);

    // extrapolate
    m_facedata_out(
      face_loc_out.ijk[IX], face_loc_out.ijk[IY], face_loc_out.ijk[IZ], ivar, iOct_local) =
      m_facedata_in(face_loc_neigh.ijk[IX],
                    face_loc_neigh.ijk[IY],
                    face_loc_neigh.ijk[IZ],
                    ivar,
                    face_loc_neigh.iOct) +
      ONE_FOURTH_F * static_cast<real_t>(i) * dfdx + ONE_FOURTH_F * static_cast<real_t>(j) * dfdy;
  }

} // linear_extrapolate_using_limited_slopes - 3d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostFacesFunctor<dim, device_t>::internal_faces_prolongation_by_toth_and_roe(
  face_multiindex_t<dim> const & face_indexes_out,
  int32_t const &                iOct_local) const
{
  const auto & ivar = face_indexes_out[dim];

  if constexpr (dim == 2)
  {
    auto const & i_new = face_indexes_out[IX];
    auto const & j_new = face_indexes_out[IY];

    // "eldest" face : all coords even
    const auto i_e = (i_new / 2) * 2;
    const auto j_e = (j_new / 2) * 2;

    // reminder: we MUST use m_facedata_out for accessing external faces

    // note : when ivar = IZ (in 2D) there are no "internal" faces
    // in other words, ivar = IZ is already taken into account when filling external faces

    real_t Uxx = ZERO_F;
    real_t Vyy = ZERO_F;
    for (int j = -1; j <= 1; j += 2)
      for (int i = -1; i <= 1; i += 2)
      {
        // clang-format off
        Uxx += static_cast<real_t>(i * j) * m_facedata_out(i_e + (i + 1) / 2, j_e + (j + 1)    , IY, iOct_local);
        Vyy += static_cast<real_t>(i * j) * m_facedata_out(i_e + (i + 1)    , j_e + (j + 1) / 2, IX, iOct_local);
        // clang-format on
      }
    Uxx /= 4;
    Vyy /= 4;

    if (ivar == IX)
    {
      auto mid_value = HALF_F * (m_facedata_out(i_new - 1, j_new, IX, iOct_local) +
                                 m_facedata_out(i_new + 1, j_new, IX, iOct_local));

      m_facedata_out(i_new, j_new, ivar, iOct_local) = mid_value + Uxx;
    }
    else if (ivar == IY)
    {
      auto mid_value = HALF_F * (m_facedata_out(i_new, j_new - 1, IY, iOct_local) +
                                 m_facedata_out(i_new, j_new + 1, IY, iOct_local));

      m_facedata_out(i_new, j_new, ivar, iOct_local) = mid_value + Vyy;
    }
  }
  else if constexpr (dim == 3)
  {
    auto const & i_new = face_indexes_out[IX];
    auto const & j_new = face_indexes_out[IY];
    auto const & k_new = face_indexes_out[IZ];

    // "eldest" face : all coords even
    const auto i_e = (i_new / 2) * 2;
    const auto j_e = (j_new / 2) * 2;
    const auto k_e = (k_new / 2) * 2;

    // compute local face index among group of 4 co-planar faces
    const int ii = 2 * (i_new - i_e) - 1;
    const int jj = 2 * (j_new - j_e) - 1;
    const int kk = 2 * (k_new - k_e) - 1;

    real_t Uxx = ZERO_F;
    real_t Vyy = ZERO_F;
    real_t Wzz = ZERO_F;

    real_t Uxyz = ZERO_F;
    real_t Vxyz = ZERO_F;
    real_t Wxyz = ZERO_F;

    // reminder: we must use m_facedata_out for accessing external faces

    for (int k = -1; k <= 1; k += 2)
      for (int j = -1; j <= 1; j += 2)
        for (int i = -1; i <= 1; i += 2)
        {
          // clang-format off
            Uxx +=
              static_cast<real_t>(i * j) *
              m_facedata_out(i_e + (i + 1) / 2, j_e + (j + 1)    , k_e + (k + 1) / 2, IY, iOct_local);
            Uxx +=
              static_cast<real_t>(i * k) *
              m_facedata_out(i_e + (i + 1) / 2, j_e + (j + 1) / 2, k_e + (k + 1)    , IZ, iOct_local);

            Vyy +=
              static_cast<real_t>(j * k) *
              m_facedata_out(i_e + (i + 1) / 2, j_e + (j + 1) / 2, k_e + (k + 1)    , IZ, iOct_local);
            Vyy +=
              static_cast<real_t>(j * i) *
              m_facedata_out(i_e + (i + 1)    , j_e + (j + 1) / 2, k_e + (k + 1) / 2, IX, iOct_local);

            Wzz +=
              static_cast<real_t>(k * i) *
              m_facedata_out(i_e + (i + 1)    , j_e + (j + 1) / 2, k_e + (k + 1) / 2, IX, iOct_local);
            Wzz +=
              static_cast<real_t>(k * j) *
              m_facedata_out(i_e + (i + 1) / 2, j_e + (j + 1)    , k_e + (k + 1) / 2, IY, iOct_local);

            Uxyz +=
              static_cast<real_t>(i * j * k) *
              m_facedata_out(i_e + (i + 1)    , j_e + (j + 1) / 2, k_e + (k + 1) / 2, IX, iOct_local);

            Vxyz +=
              static_cast<real_t>(i * j * k) *
              m_facedata_out(i_e + (i + 1) / 2, j_e + (j + 1)    , k_e + (k + 1) / 2, IY, iOct_local);

            Wxyz +=
              static_cast<real_t>(i * j * k) *
              m_facedata_out(i_e + (i + 1) / 2, j_e + (j + 1) / 2, k_e + (k + 1)    , IZ, iOct_local);
          // clang-format on
        }
    Uxx /= 8;
    Vyy /= 8;
    Wzz /= 8;
    Uxyz /= 16; // because Delta x = Delta y = Delta z
    Vxyz /= 16; // because Delta x = Delta y = Delta z
    Wxyz /= 16; // because Delta x = Delta y = Delta z

    if (ivar == IX)
    {
      auto mid_value = HALF_F * (m_facedata_out(i_new - 1, j_new, k_new, IX, iOct_local) +
                                 m_facedata_out(i_new + 1, j_new, k_new, IX, iOct_local));

      m_facedata_out(i_new, j_new, k_new, ivar, iOct_local) =
        mid_value + Uxx + static_cast<real_t>(kk) * Vxyz + static_cast<real_t>(jj) * Wxyz;
    }
    if (ivar == IY)
    {
      auto mid_value = HALF_F * (m_facedata_out(i_new, j_new - 1, k_new, IY, iOct_local) +
                                 m_facedata_out(i_new, j_new + 1, k_new, IY, iOct_local));

      m_facedata_out(i_new, j_new, k_new, ivar, iOct_local) =
        mid_value + Vyy + static_cast<real_t>(ii) * Wxyz + static_cast<real_t>(kk) * Uxyz;
    }
    if (ivar == IZ)
    {
      auto mid_value = HALF_F * (m_facedata_out(i_new, j_new, k_new - 1, IZ, iOct_local) +
                                 m_facedata_out(i_new, j_new, k_new + 1, IZ, iOct_local));

      m_facedata_out(i_new, j_new, k_new, ivar, iOct_local) =
        mid_value + Wzz + static_cast<real_t>(jj) * Uxyz + static_cast<real_t>(ii) * Vxyz;
    }

  } // end 3D

} // internal_faces_prolongation_by_toth_and_roe

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostFacesFunctor<dim, device_t>::fill_inner(
  face_multiindex_t<dim> const & face_indexes_in,
  face_multiindex_t<dim> const & face_indexes_out,
  int32_t                        iOct_local) const
{

  // clang-format off
  if constexpr (dim == 2)
  {
    m_facedata_out(face_indexes_out[IX],
                   face_indexes_out[IY],
                   face_indexes_out[dim],
                   iOct_local) =
      m_facedata_in(face_indexes_in[IX],
                    face_indexes_in[IY],
                    face_indexes_in[dim],
                    iOct_local + m_iOct_begin);
  }
  else if constexpr (dim == 3)
  {
    m_facedata_out(face_indexes_out[IX],
                   face_indexes_out[IY],
                   face_indexes_out[IZ],
                   face_indexes_out[dim],
                   iOct_local) =
      m_facedata_in(face_indexes_in[IX],
                    face_indexes_in[IY],
                    face_indexes_in[IZ],
                    face_indexes_in[dim],
                      iOct_local + m_iOct_begin);
  }
  // clang-format on

} // fill_inner

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostFacesFunctor<dim, device_t>::fill_all_faces_but_internal(
  face_multiindex_t<dim> const & face_indexes_out,
  int32_t const &                iOct_local) const
{
  auto const & ivar = face_indexes_out[dim];

  // coordinates of source face
  face_multiindex_t<dim> face_indexes_in;

  // compute source face coordinates
  const auto dir = convert_face_indexes_out_to_in(face_indexes_in, face_indexes_out);

  auto dir_norm = dir[IX] * dir[IX] + dir[IY] * dir[IY];
  if constexpr (dim == 3)
    dir_norm += dir[IZ] * dir[IZ];

  // check if current quadrant is ghost or outside; in that case, we only fill inner faces
  // if ((dir_norm == 0) and (iOct_local >= (m_amr_mesh_info.local_num_quadrants() +
  //                                        m_amr_mesh_info.local_num_ghosts())))
  if ((dir_norm == 0) and (iOct_local >= (m_amr_mesh_info.local_num_quadrants())))
  {
    fill_inner(face_indexes_in, face_indexes_out, iOct_local);
    return;
  }

  // deal with all other quadrants (owned and ghost)
  if (dir_norm == 0)
  {
    // current faces is inside current inner block
    fill_inner(face_indexes_in, face_indexes_out, iOct_local);
  }
  else
  {
    // current face belongs to a ghost cell (thus belonging to a neighbor block)

    /*
     * fill ghost faces all around
     */

    // get orchard key of current octant
    auto key_cur = m_stencil_helper.key(iOct_local + m_iOct_begin);

    const auto & b = m_block_sizes;

    shift_t<dim> shift;
    shift[IX] = b[IX] * dir[IX];
    shift[IY] = b[IY] * dir[IY];
    if constexpr (dim == 3)
    {
      shift[IZ] = b[IZ] * dir[IZ];
    }

    // coordinates of source cell (where to read data)
    // const auto cell_coord_in = face_to_cell_coords<dim>(face_indexes_in,
    // m_block_sizes);

    // const CellLocation_t cell_loc_cur{ cell_coord_in, key_cur, iOct_local + m_iOct_begin, false
    // }; const auto           cell_loc_neigh = m_stencil_helper.getNeighLoc(cell_loc_cur, shift);

    const FaceLocation_t face_loc_cur{ face_indexes_in, key_cur, iOct_local + m_iOct_begin, false };
    auto                 face_loc_neigh = m_stencil_helper.getNeighLoc(face_loc_cur, shift);

    /*
     * Dealing with the 3 possibilities:
     * - neighbor octant is at same    AMR level : doing a simple copy
     * - neighbor octant is at finer   AMR level : doing a restriction (average values)
     * - neighbor octant is at coarser AMR level : doing a prolongation
     */

    if (face_loc_neigh.level() == face_loc_cur.level())
    {
      // doing a simple copy

      // clang-format off
        if constexpr (dim == 2)
        {
          m_facedata_out(face_indexes_out[IX],
                         face_indexes_out[IY], ivar, iOct_local) =
            m_facedata_in(face_indexes_in[IX],
                          face_indexes_in[IY], ivar, face_loc_neigh.iOct);
        }
        else if constexpr (dim == 3)
        {
          m_facedata_out(face_indexes_out[IX],
                         face_indexes_out[IY],
                         face_indexes_out[IZ], ivar, iOct_local) =
            m_facedata_in(face_indexes_in[IX],
                          face_indexes_in[IY],
                          face_indexes_in[IZ], ivar, face_loc_neigh.iOct);
        }
      // clang-format on
    }
    else if (face_loc_neigh.level() + 1 == face_loc_cur.level())
    {

      //
      // if face is ambiguous, we need to check if the symmetric neighbor must be used instead
      //
      if (is_face_ambiguous(face_indexes_out, m_block_sizes))
      {
        const auto face_loc_neigh2 = m_stencil_helper.getBorderFaceLocSymmetric(face_loc_neigh);

        if (face_loc_neigh2.level() == face_loc_cur.level())
        {
          face_loc_neigh = face_loc_neigh2;

          // we also need to adjust face coords since face_loc_neigh2 has been obtained by
          // shifting from a coarser neighbor
          if constexpr (dim == 2)
          {
            if (face_indexes_out[dim] == IX)
            {
              if (face_indexes_out[IY] & 0x1)
                face_loc_neigh.ijk[IY] += 1;
            }
            else if (face_indexes_out[dim] == IY)
            {
              if (face_indexes_out[IX] & 0x1)
                face_loc_neigh.ijk[IX] += 1;
            }
          }
          else if constexpr (dim == 3)
          {
            if (face_indexes_out[dim] == IX)
            {
              if (face_indexes_out[IY] & 0x1)
                face_loc_neigh.ijk[IY] += 1;
              if (face_indexes_out[IZ] & 0x1)
                face_loc_neigh.ijk[IZ] += 1;
            }
            else if (face_indexes_out[dim] == IY)
            {
              if (face_indexes_out[IX] & 0x1)
                face_loc_neigh.ijk[IX] += 1;
              if (face_indexes_out[IZ] & 0x1)
                face_loc_neigh.ijk[IZ] += 1;
            }
            else if (face_indexes_out[dim] == IZ)
            {
              if (face_indexes_out[IX] & 0x1)
                face_loc_neigh.ijk[IX] += 1;
              if (face_indexes_out[IY] & 0x1)
                face_loc_neigh.ijk[IY] += 1;
            }
          }
        }
      } // end ambiguous face special treatment

      //
      // doing a PROLONGATION because neighbor is coarser
      // only fill external faces (the ones that are co-localized with mother cells)
      //

      if (is_external_face<dim>(face_indexes_out))
      {
        if (m_prolongation.m_face_external == +FaceCenteredProlongationExternalType::SIMPLE_COPY)
        {
          // simple copy of the coarse value

          // clang-format off
            if constexpr (dim == 2)
            {
              m_facedata_out(face_indexes_out[IX],
                             face_indexes_out[IY], ivar, iOct_local) =
                m_facedata_in(face_loc_neigh.ijk[IX],
                              face_loc_neigh.ijk[IY], ivar, face_loc_neigh.iOct);
            }
            else if constexpr (dim == 3)
            {
              m_facedata_out(face_indexes_out[IX],
                             face_indexes_out[IY],
                             face_indexes_out[IZ], ivar, iOct_local) =
                m_facedata_in(face_loc_neigh.ijk[IX],
                              face_loc_neigh.ijk[IY],
                              face_loc_neigh.ijk[IZ], ivar, face_loc_neigh.iOct);
            }
          // clang-format on
        }
        else if (m_prolongation.m_face_external ==
                 +FaceCenteredProlongationExternalType::EXTRAPOLATE_LINEAR_MINMOD)
        {
          // not entirely sure this is a good idea to use this; we should probably stick with
          // simple copy.

          // this only valid for ivar=IX,IY not IZ in 2d

          const FaceLocation_t face_loc_out{
            face_indexes_out, key_cur, iOct_local + m_iOct_begin, false
          };

          if constexpr (dim == 2)
          {
            if (ivar == IZ)
            {
              // do a simple copy
              m_facedata_out(face_indexes_out[IX], face_indexes_out[IY], ivar, iOct_local) =
                m_facedata_in(
                  face_loc_neigh.ijk[IX], face_loc_neigh.ijk[IY], ivar, face_loc_neigh.iOct);
            }
            else
            {
              linear_extrapolate_using_limited_slopes(face_loc_neigh, face_loc_out, iOct_local);
            }
          }
          else if constexpr (dim == 3)
          {
            linear_extrapolate_using_limited_slopes(face_loc_neigh, face_loc_out, iOct_local);
          }
        }
      }
    }
    else if (face_loc_neigh.level() == face_loc_cur.level() + 1)
    {
      // doing a RESTRICTION

      // clang-format off
        if constexpr (dim == 2)
        {
          m_facedata_out(face_indexes_out[IX],
                         face_indexes_out[IY], ivar, iOct_local) =
            m_stencil_helper.compute_face_siblings_average(
              face_loc_neigh, m_facedata_in);
        }
        else if constexpr (dim == 3)
        {
          m_facedata_out(face_indexes_out[IX],
                         face_indexes_out[IY],
                         face_indexes_out[IZ], ivar, iOct_local) =
            m_stencil_helper.compute_face_siblings_average(
              face_loc_neigh, m_facedata_in);
        }
      // clang-format on
    }
    else
    {
      KOKKOS_ASSERT(false && "Logic error: neighbor octant not found (Kernel Panic !)");
    }

  } // end if (dir_norm == 0)

} // fill_all_faces_but_internal

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostFacesFunctor<dim, device_t>::fill_internal_faces_ghost(
  face_multiindex_t<dim> const & face_indexes_out,
  int32_t const &                iOct_local) const
{
  [[maybe_unused]] auto const & ivar = face_indexes_out[dim];

  // coordinates of source face
  face_multiindex_t<dim> face_indexes_in;

  // compute source face coordinates
  const auto dir = convert_face_indexes_out_to_in(face_indexes_in, face_indexes_out);

  auto dir_norm = dir[IX] * dir[IX] + dir[IY] * dir[IY];
  if constexpr (dim == 3)
    dir_norm += dir[IZ] * dir[IZ];

  // make sure we are outside inner block
  if (dir_norm > 0)
  {
    // get orchard key of current octant
    auto key_cur = m_stencil_helper.key(iOct_local + m_iOct_begin);

    const auto & b = m_block_sizes;

    shift_t<dim> shift;
    shift[IX] = b[IX] * dir[IX];
    shift[IY] = b[IY] * dir[IY];
    if constexpr (dim == 3)
    {
      shift[IZ] = b[IZ] * dir[IZ];
    }

    const FaceLocation_t face_loc_cur{ face_indexes_in, key_cur, iOct_local + m_iOct_begin, false };
    const auto           face_loc_neigh = m_stencil_helper.getNeighLoc(face_loc_cur, shift);

    if (face_loc_neigh.level() + 1 == face_loc_cur.level())
    {
      // doing a prolongation because neighbor is coarser
      // only fill internal faces (not the ones that are co-localized with mother cells)

      // beware that the following can only happen after all external faces are filled since
      // up-to-date external faces are required to fill internal faces.
      if (is_internal_face<dim>(face_indexes_out))
      {
        if (m_prolongation.m_face_internal == +FaceCenteredProlongationInternalType::TOTH_AND_ROE)
        {
          internal_faces_prolongation_by_toth_and_roe(face_indexes_out, iOct_local);
        }
      } // end internal faces
    } // end prolongation
  } // end dir_norm > 0

} // fill_internal_faces_ghost

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostFacesFunctor<dim, device_t>::operator()(TagFillAllFacesButInternal const &,
                                                      const index_t & global_index) const
{

  const auto num_elements_per_octant = m_facedata_out.num_elements_per_octant();

  const int32_t iOct_local = static_cast<int32_t>(global_index / num_elements_per_octant);
  const int32_t face_flat_index =
    static_cast<int32_t>(global_index - iOct_local * num_elements_per_octant);

  // compute ix,iy,iz,ivar of local face inside
  // block from a face flat-index
  const auto face_indexes_out = face_flat_index_unravel<dim>(
    face_flat_index, m_total_sizes, m_facedata_out.offsets(), m_facedata_out.shift());

  fill_all_faces_but_internal(face_indexes_out, iOct_local);

} // operator() - TagFillAllFacesButInternal

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostFacesFunctor<dim, device_t>::operator()(TagFillInternalFaces const &,
                                                      const index_t & global_index) const
{

  const auto num_elements_per_octant = m_facedata_out.num_elements_per_octant();

  const int32_t iOct_local = static_cast<int32_t>(global_index / num_elements_per_octant);
  const int32_t face_flat_index =
    static_cast<int32_t>(global_index - iOct_local * num_elements_per_octant);

  // compute ix,iy,iz,ivar of local face inside
  // block from a face flat-index
  const auto face_indexes_out = face_flat_index_unravel<dim>(
    face_flat_index, m_total_sizes, m_facedata_out.offsets(), m_facedata_out.shift());

  fill_internal_faces_ghost(face_indexes_out, iOct_local);

} // operator() - TagFillInternalFaces

template class FillBlockGhostFacesFunctor<2, kalypsso::DefaultDevice>;
template class FillBlockGhostFacesFunctor<3, kalypsso::DefaultDevice>;

} // namespace kalypsso
