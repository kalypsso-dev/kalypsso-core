// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ConformalFullStatus.h
 *
 */
#ifndef KALYPSSO_CORE_CONFORMALFULLSTATUS_H_
#define KALYPSSO_CORE_CONFORMALFULLSTATUS_H_

#include <kalypsso/core/BitFieldInteger.h>

#include <Kokkos_Core.hpp>                         // for Kokkos::View
#include <Kokkos_Macros.hpp>                       // for KOKKOS_ENABLE_XXX
#include <kalypsso/core/mesh_utils.h>              // for Face::XMIN, etc...
#include <kalypsso/core/ConformalNeighborStatus.h> // for conformal_neighbor_status
#include <kalypsso/core/kalypsso_macros.h>         // for macro KALYPSSO_DISABLE_NVCC_WARNINGS_PUSH

KALYPSSO_DISABLE_NVCC_WARNINGS_PUSH()

namespace kalypsso
{

/**
 * \struct conformal_full_status_t
 *
 * For each face, corner or edge (in 3d) of a tree leaf, encode the conformal status,
 * more precisely we want to store the level difference between a given AMR leaf, and
 * its neighbors.
 *
 * Encoding values (in binary, 2 bits):
 * 00 : neighbor is at same AMR level, face is conform
 * 01 : neighbor is finer than current cell (higher AMR level), face is non-conform
 * 10 : neighbor is coarser than current cell (lower AMR level), face is no-conform
 * 11 : neighbor status is not available (this may happens when visiting a MPI ghost leaf).
 *
 * Implementation note:
 * - in 2D, we only need 2 bits per face and corner, that is 16 bits in total per leaf/block.
 * - in 3D, we need 26x2bits = 52 bits => rounded-up to 64 bits
 *
 * \note this a more general version of conformal_face_status_t;
 * sometimes conformal_face_status_t is enough, sometimes not (e.g. in MHD we may need the full
 * conformal status)
 *
 */
template <size_t dim>
struct conformal_full_status_t;

// ====================================================================
// ====================================================================
// ====================================================================
template <>
struct conformal_full_status_t<2> : public BitFieldInteger<uint16_t>
{
  using status_t = uint16_t;

  conformal_full_status_t(const conformal_full_status_t &) = default;

  conformal_full_status_t(conformal_full_status_t &&) = default;

  conformal_full_status_t &
  operator=(const conformal_full_status_t &) = default;

  conformal_full_status_t &
  operator=(conformal_full_status_t &&) = default;

  using BitFieldInteger::BitFieldInteger;
  // clang-format off
  DECLARE_CASTED_FIELD(face_xmin,       0, 2, uint8_t)
  DECLARE_CASTED_FIELD(face_xmax,       2, 2, uint8_t)
  DECLARE_CASTED_FIELD(face_ymin,       4, 2, uint8_t)
  DECLARE_CASTED_FIELD(face_ymax,       6, 2, uint8_t)
  DECLARE_CASTED_FIELD(edge_xmin_ymin,  8, 2, uint8_t)
  DECLARE_CASTED_FIELD(edge_xmax_ymin, 10, 2, uint8_t)
  DECLARE_CASTED_FIELD(edge_xmin_ymax, 12, 2, uint8_t)
  DECLARE_CASTED_FIELD(edge_xmax_ymax, 14, 2, uint8_t)
  // clang-format on

  //! set a face status
  KOKKOS_INLINE_FUNCTION static void
  set_status(Face::face_t face, uint8_t face_status, status_t & conformal_status)
  {
    if (face == Face::XMIN)
      set_face_xmin(conformal_status, face_status);
    else if (face == Face::XMAX)
      set_face_xmax(conformal_status, face_status);
    else if (face == Face::YMIN)
      set_face_ymin(conformal_status, face_status);
    else if (face == Face::YMAX)
      set_face_ymax(conformal_status, face_status);
  } // set_status

  //! get a face status
  KOKKOS_INLINE_FUNCTION static uint8_t
  get_status(Face::face_t face, status_t const & conformal_status)
  {
    if (face == Face::XMIN)
      return face_xmin(conformal_status);
    else if (face == Face::XMAX)
      return face_xmax(conformal_status);
    else if (face == Face::YMIN)
      return face_ymin(conformal_status);
    else if (face == Face::YMAX)
      return face_ymax(conformal_status);
    return conformal_neighbor_status::NEIGHBOR_IS_UNAVAILABLE;
  } // get_status

  //! set an edge status
  KOKKOS_INLINE_FUNCTION static void
  set_status(Face::face_t face_x,
             Face::face_t face_y,
             uint8_t      edge_status,
             status_t &   conformal_status)
  {
    if (face_x == Face::XMIN and face_y == Face::YMIN)
      set_edge_xmin_ymin(conformal_status, edge_status);
    else if (face_x == Face::XMAX and face_y == Face::YMIN)
      set_edge_xmax_ymin(conformal_status, edge_status);
    else if (face_x == Face::XMIN and face_y == Face::YMAX)
      set_edge_xmin_ymax(conformal_status, edge_status);
    else if (face_x == Face::XMAX and face_y == Face::YMAX)
      set_edge_xmax_ymax(conformal_status, edge_status);
  } // set_status

  //! get an edge status
  KOKKOS_INLINE_FUNCTION static uint8_t
  get_status(Face::face_t face_x, Face::face_t face_y, status_t const & conformal_status)
  {
    if (face_x == Face::XMIN and face_y == Face::YMIN)
      return edge_xmin_ymin(conformal_status);
    else if (face_x == Face::XMAX and face_y == Face::YMIN)
      return edge_xmax_ymin(conformal_status);
    else if (face_x == Face::XMIN and face_y == Face::YMAX)
      return edge_xmin_ymax(conformal_status);
    else if (face_x == Face::XMAX and face_y == Face::YMAX)
      return edge_xmax_ymax(conformal_status);
    return conformal_neighbor_status::NEIGHBOR_IS_UNAVAILABLE;
  } // get_status

}; // conformal_full_status_t<2>

// ====================================================================
// ====================================================================
// ====================================================================
template <>
struct conformal_full_status_t<3> : public BitFieldInteger<uint64_t>
{
  using status_t = uint64_t;

  conformal_full_status_t(const conformal_full_status_t &) = default;

  conformal_full_status_t(conformal_full_status_t &&) = default;

  conformal_full_status_t &
  operator=(const conformal_full_status_t &) = default;

  conformal_full_status_t &
  operator=(conformal_full_status_t &&) = default;

  using BitFieldInteger::BitFieldInteger;
  // clang-format off
  DECLARE_CASTED_FIELD(face_xmin,              0, 2, uint8_t)
  DECLARE_CASTED_FIELD(face_xmax,              2, 2, uint8_t)
  DECLARE_CASTED_FIELD(face_ymin,              4, 2, uint8_t)
  DECLARE_CASTED_FIELD(face_ymax,              6, 2, uint8_t)
  DECLARE_CASTED_FIELD(face_zmin,              8, 2, uint8_t)
  DECLARE_CASTED_FIELD(face_zmax,             10, 2, uint8_t)
  DECLARE_CASTED_FIELD(edge_xmin_ymin,        12, 2, uint8_t)
  DECLARE_CASTED_FIELD(edge_xmax_ymin,        14, 2, uint8_t)
  DECLARE_CASTED_FIELD(edge_xmin_ymax,        16, 2, uint8_t)
  DECLARE_CASTED_FIELD(edge_xmax_ymax,        18, 2, uint8_t)
  DECLARE_CASTED_FIELD(edge_ymin_zmin,        20, 2, uint8_t)
  DECLARE_CASTED_FIELD(edge_ymax_zmin,        22, 2, uint8_t)
  DECLARE_CASTED_FIELD(edge_ymin_zmax,        24, 2, uint8_t)
  DECLARE_CASTED_FIELD(edge_ymax_zmax,        26, 2, uint8_t)
  DECLARE_CASTED_FIELD(edge_xmin_zmin,        28, 2, uint8_t)
  DECLARE_CASTED_FIELD(edge_xmax_zmin,        30, 2, uint8_t)
  DECLARE_CASTED_FIELD(edge_xmin_zmax,        32, 2, uint8_t)
  DECLARE_CASTED_FIELD(edge_xmax_zmax,        34, 2, uint8_t)
  DECLARE_CASTED_FIELD(corner_xmin_ymin_zmin, 36, 2, uint8_t)
  DECLARE_CASTED_FIELD(corner_xmax_ymin_zmin, 38, 2, uint8_t)
  DECLARE_CASTED_FIELD(corner_xmin_ymax_zmin, 40, 2, uint8_t)
  DECLARE_CASTED_FIELD(corner_xmax_ymax_zmin, 42, 2, uint8_t)
  DECLARE_CASTED_FIELD(corner_xmin_ymin_zmax, 44, 2, uint8_t)
  DECLARE_CASTED_FIELD(corner_xmax_ymin_zmax, 46, 2, uint8_t)
  DECLARE_CASTED_FIELD(corner_xmin_ymax_zmax, 48, 2, uint8_t)
  DECLARE_CASTED_FIELD(corner_xmax_ymax_zmax, 50, 2, uint8_t)
  // clang-format on

  //! set a face status
  KOKKOS_INLINE_FUNCTION static void
  set_status(Face::face_t face, uint8_t face_status, status_t & conformal_status)
  {
    if (face == Face::XMIN)
      set_face_xmin(conformal_status, face_status);
    else if (face == Face::XMAX)
      set_face_xmax(conformal_status, face_status);
    else if (face == Face::YMIN)
      set_face_ymin(conformal_status, face_status);
    else if (face == Face::YMAX)
      set_face_ymax(conformal_status, face_status);
    else if (face == Face::ZMIN)
      set_face_zmin(conformal_status, face_status);
    else if (face == Face::ZMAX)
      set_face_zmax(conformal_status, face_status);
  } // set_status - face

  //! get a face status
  KOKKOS_INLINE_FUNCTION static uint8_t
  get_status(Face::face_t face, status_t const & conformal_status)
  {
    if (face == Face::XMIN)
      return face_xmin(conformal_status);
    else if (face == Face::XMAX)
      return face_xmax(conformal_status);
    else if (face == Face::YMIN)
      return face_ymin(conformal_status);
    else if (face == Face::YMAX)
      return face_ymax(conformal_status);
    else if (face == Face::ZMIN)
      return face_zmin(conformal_status);
    else if (face == Face::ZMAX)
      return face_zmax(conformal_status);
    return conformal_neighbor_status::NEIGHBOR_IS_UNAVAILABLE;
  } // set_status - face

  //! set an edge status
  KOKKOS_INLINE_FUNCTION static void
  set_status(Face::face_t face_0,
             Face::face_t face_1,
             uint8_t      edge_status,
             status_t &   conformal_status)
  {
    if (face_0 == Face::XMIN and face_1 == Face::YMIN)
      set_edge_xmin_ymin(conformal_status, edge_status);
    else if (face_0 == Face::XMAX and face_1 == Face::YMIN)
      set_edge_xmax_ymin(conformal_status, edge_status);
    else if (face_0 == Face::XMIN and face_1 == Face::YMAX)
      set_edge_xmin_ymax(conformal_status, edge_status);
    else if (face_0 == Face::XMAX and face_1 == Face::YMAX)
      set_edge_xmax_ymax(conformal_status, edge_status);
    else if (face_0 == Face::YMIN and face_1 == Face::ZMIN)
      set_edge_ymin_zmin(conformal_status, edge_status);
    else if (face_0 == Face::YMAX and face_1 == Face::ZMIN)
      set_edge_ymax_zmin(conformal_status, edge_status);
    else if (face_0 == Face::YMIN and face_1 == Face::ZMAX)
      set_edge_ymin_zmax(conformal_status, edge_status);
    else if (face_0 == Face::YMAX and face_1 == Face::ZMAX)
      set_edge_ymax_zmax(conformal_status, edge_status);
    else if (face_0 == Face::XMIN and face_1 == Face::ZMIN)
      set_edge_xmin_zmin(conformal_status, edge_status);
    else if (face_0 == Face::XMAX and face_1 == Face::ZMIN)
      set_edge_xmax_zmin(conformal_status, edge_status);
    else if (face_0 == Face::XMIN and face_1 == Face::ZMAX)
      set_edge_xmin_zmax(conformal_status, edge_status);
    else if (face_0 == Face::XMAX and face_1 == Face::ZMAX)
      set_edge_xmax_zmax(conformal_status, edge_status);
  } // set_status - edge

  //! get an edge status
  KOKKOS_INLINE_FUNCTION static uint8_t
  get_status(Face::face_t face_0, Face::face_t face_1, status_t const & conformal_status)
  {
    if (face_0 == Face::XMIN and face_1 == Face::YMIN)
      return edge_xmin_ymin(conformal_status);
    else if (face_0 == Face::XMAX and face_1 == Face::YMIN)
      return edge_xmax_ymin(conformal_status);
    else if (face_0 == Face::XMIN and face_1 == Face::YMAX)
      return edge_xmin_ymax(conformal_status);
    else if (face_0 == Face::XMAX and face_1 == Face::YMAX)
      return edge_xmax_ymax(conformal_status);
    else if (face_0 == Face::YMIN and face_1 == Face::ZMIN)
      return edge_ymin_zmin(conformal_status);
    else if (face_0 == Face::YMAX and face_1 == Face::ZMIN)
      return edge_ymax_zmin(conformal_status);
    else if (face_0 == Face::YMIN and face_1 == Face::ZMAX)
      return edge_ymin_zmax(conformal_status);
    else if (face_0 == Face::YMAX and face_1 == Face::ZMAX)
      return edge_ymax_zmax(conformal_status);
    else if (face_0 == Face::XMIN and face_1 == Face::ZMIN)
      return edge_xmin_zmin(conformal_status);
    else if (face_0 == Face::XMAX and face_1 == Face::ZMIN)
      return edge_xmax_zmin(conformal_status);
    else if (face_0 == Face::XMIN and face_1 == Face::ZMAX)
      return edge_xmin_zmax(conformal_status);
    else if (face_0 == Face::XMAX and face_1 == Face::ZMAX)
      return edge_xmax_zmax(conformal_status);
    return conformal_neighbor_status::NEIGHBOR_IS_UNAVAILABLE;
  } // get_status - edge

  //! set a corner status
  KOKKOS_INLINE_FUNCTION static void
  set_status(Face::face_t face_0,
             Face::face_t face_1,
             Face::face_t face_2,
             uint8_t      corner_status,
             status_t &   conformal_status)
  {
    if (face_0 == Face::XMIN and face_1 == Face::YMIN and face_2 == Face::ZMIN)
      set_corner_xmin_ymin_zmin(conformal_status, corner_status);
    else if (face_0 == Face::XMAX and face_1 == Face::YMIN and face_2 == Face::ZMIN)
      set_corner_xmax_ymin_zmin(conformal_status, corner_status);
    else if (face_0 == Face::XMIN and face_1 == Face::YMAX and face_2 == Face::ZMIN)
      set_corner_xmin_ymax_zmin(conformal_status, corner_status);
    else if (face_0 == Face::XMAX and face_1 == Face::YMAX and face_2 == Face::ZMIN)
      set_corner_xmax_ymax_zmin(conformal_status, corner_status);
    else if (face_0 == Face::XMIN and face_1 == Face::YMIN and face_2 == Face::ZMAX)
      set_corner_xmin_ymin_zmax(conformal_status, corner_status);
    else if (face_0 == Face::XMAX and face_1 == Face::YMIN and face_2 == Face::ZMAX)
      set_corner_xmax_ymin_zmax(conformal_status, corner_status);
    else if (face_0 == Face::XMIN and face_1 == Face::YMAX and face_2 == Face::ZMAX)
      set_corner_xmin_ymax_zmax(conformal_status, corner_status);
    else if (face_0 == Face::XMAX and face_1 == Face::YMAX and face_2 == Face::ZMAX)
      set_corner_xmax_ymax_zmax(conformal_status, corner_status);
  } // set status - corner

  //! get a corner status
  KOKKOS_INLINE_FUNCTION static uint8_t
  get_status(Face::face_t     face_0,
             Face::face_t     face_1,
             Face::face_t     face_2,
             status_t const & conformal_status)
  {
    if (face_0 == Face::XMIN and face_1 == Face::YMIN and face_2 == Face::ZMIN)
      return corner_xmin_ymin_zmin(conformal_status);
    else if (face_0 == Face::XMAX and face_1 == Face::YMIN and face_2 == Face::ZMIN)
      return corner_xmax_ymin_zmin(conformal_status);
    else if (face_0 == Face::XMIN and face_1 == Face::YMAX and face_2 == Face::ZMIN)
      return corner_xmin_ymax_zmin(conformal_status);
    else if (face_0 == Face::XMAX and face_1 == Face::YMAX and face_2 == Face::ZMIN)
      return corner_xmax_ymax_zmin(conformal_status);
    else if (face_0 == Face::XMIN and face_1 == Face::YMIN and face_2 == Face::ZMAX)
      return corner_xmin_ymin_zmax(conformal_status);
    else if (face_0 == Face::XMAX and face_1 == Face::YMIN and face_2 == Face::ZMAX)
      return corner_xmax_ymin_zmax(conformal_status);
    else if (face_0 == Face::XMIN and face_1 == Face::YMAX and face_2 == Face::ZMAX)
      return corner_xmin_ymax_zmax(conformal_status);
    else if (face_0 == Face::XMAX and face_1 == Face::YMAX and face_2 == Face::ZMAX)
      return corner_xmax_ymax_zmax(conformal_status);
    return conformal_neighbor_status::NEIGHBOR_IS_UNAVAILABLE;
  } // set status - corner

}; // conformal_full_status_t<3>

//! data type alias to store full conformal status (one value by AMR leaf)
template <size_t dim>
using conformal_full_status_value_t = typename conformal_full_status_t<dim>::status_t;

//! conformal face status array type alias (device)
template <size_t dim, typename device_t>
using conformal_full_status_view_t =
  typename Kokkos::View<conformal_full_status_value_t<dim> *, device_t>;

//! conformal face status array type alias (host)
template <size_t dim, typename device_t>
using conformal_full_status_view_host_t =
  typename Kokkos::View<conformal_full_status_value_t<dim> *,
                        typename conformal_full_status_view_t<dim, device_t>::host_mirror_space>;

} // namespace kalypsso

KALYPSSO_DISABLE_NVCC_WARNINGS_POP()

#endif // KALYPSSO_CORE_CONFORMALFULLSTATUS_H_
