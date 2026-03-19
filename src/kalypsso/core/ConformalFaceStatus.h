// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ConformalFaceStatus.h
 *
 */
#ifndef KALYPSSO_CORE_CONFORMALFACESTATUS_H_
#define KALYPSSO_CORE_CONFORMALFACESTATUS_H_

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
 * \struct conformal_face_status_t
 *
 * For each face of a tree leaf, encode the conformal status, more precisely we want to store the
 * level difference between a given AMR leaf, and its neighbors.
 *
 * Encoding values (in binary, 2 bits):
 * 00 : neighbor is at same AMR level, face is conform
 * 01 : neighbor is finer than current cell (higher AMR level), face is non-conform
 * 10 : neighbor is coarser than current cell (lower AMR level), face is no-conform
 * 11 : neighbor status is not available (this may happens when visiting a MPI ghost leaf).
 *
 * Implementation note:
 * - in 2D, we only need 2 bits per face, that is 8 bits in total per leaf/block.
 * - in 3D, we need 6x2bits = 12 bits => rounded-up to 16 bits
 */
template <size_t dim>
struct conformal_face_status_t;

// ====================================================================
// ====================================================================
// ====================================================================
template <>
struct conformal_face_status_t<2> : public BitFieldInteger<uint8_t>
{
  using status_t = uint8_t;

  conformal_face_status_t(const conformal_face_status_t &) = default;

  conformal_face_status_t(conformal_face_status_t &&) = default;

  conformal_face_status_t &
  operator=(const conformal_face_status_t &) = default;

  conformal_face_status_t &
  operator=(conformal_face_status_t &&) = default;

  using BitFieldInteger::BitFieldInteger;
  DECLARE_CASTED_FIELD(face_xmin, 0, 2, uint8_t)
  DECLARE_CASTED_FIELD(face_xmax, 2, 2, uint8_t)
  DECLARE_CASTED_FIELD(face_ymin, 4, 2, uint8_t)
  DECLARE_CASTED_FIELD(face_ymax, 6, 2, uint8_t)

  KOKKOS_INLINE_FUNCTION static void
  set_status(Face::face_t const & face, uint8_t face_status, status_t & conformal_status)
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

  KOKKOS_INLINE_FUNCTION static bool
  is_face_neighbor_finer(Face::face_t const & face, status_t const & conformal_status)
  {
    if (face == Face::XMIN)
      return conformal_face_status_t<2>::face_xmin(conformal_status) ==
             conformal_neighbor_status::NEIGHBOR_IS_FINER;
    else if (face == Face::XMAX)
      return conformal_face_status_t<2>::face_xmax(conformal_status) ==
             conformal_neighbor_status::NEIGHBOR_IS_FINER;
    else if (face == Face::YMIN)
      return conformal_face_status_t<2>::face_ymin(conformal_status) ==
             conformal_neighbor_status::NEIGHBOR_IS_FINER;
    else if (face == Face::YMAX)
      return conformal_face_status_t<2>::face_ymax(conformal_status) ==
             conformal_neighbor_status::NEIGHBOR_IS_FINER;

    return false;
  } // is_face_neighbor_finer

}; // conformal_face_status_t<2>

// ====================================================================
// ====================================================================
// ====================================================================
template <>
struct conformal_face_status_t<3> : public BitFieldInteger<uint16_t>
{
  using status_t = uint16_t;

  conformal_face_status_t(const conformal_face_status_t &) = default;

  conformal_face_status_t(conformal_face_status_t &&) = default;

  conformal_face_status_t &
  operator=(const conformal_face_status_t &) = default;

  conformal_face_status_t &
  operator=(conformal_face_status_t &&) = default;

  using BitFieldInteger::BitFieldInteger;
  DECLARE_CASTED_FIELD(face_xmin, 0, 2, uint8_t)
  DECLARE_CASTED_FIELD(face_xmax, 2, 2, uint8_t)
  DECLARE_CASTED_FIELD(face_ymin, 4, 2, uint8_t)
  DECLARE_CASTED_FIELD(face_ymax, 6, 2, uint8_t)
  DECLARE_CASTED_FIELD(face_zmin, 8, 2, uint8_t)
  DECLARE_CASTED_FIELD(face_zmax, 10, 2, uint8_t)

  KOKKOS_INLINE_FUNCTION static void
  set_status(Face::face_t const & face, uint8_t face_status, status_t & conformal_status)
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
  } // set_status

  KOKKOS_INLINE_FUNCTION static bool
  is_face_neighbor_finer(Face::face_t const & face, status_t const & conformal_status)
  {
    if (face == Face::XMIN)
      return conformal_face_status_t<3>::face_xmin(conformal_status) ==
             conformal_neighbor_status::NEIGHBOR_IS_FINER;
    else if (face == Face::XMAX)
      return conformal_face_status_t<3>::face_xmax(conformal_status) ==
             conformal_neighbor_status::NEIGHBOR_IS_FINER;
    else if (face == Face::YMIN)
      return conformal_face_status_t<3>::face_ymin(conformal_status) ==
             conformal_neighbor_status::NEIGHBOR_IS_FINER;
    else if (face == Face::YMAX)
      return conformal_face_status_t<3>::face_ymax(conformal_status) ==
             conformal_neighbor_status::NEIGHBOR_IS_FINER;
    else if (face == Face::ZMIN)
      return conformal_face_status_t<3>::face_zmin(conformal_status) ==
             conformal_neighbor_status::NEIGHBOR_IS_FINER;
    else if (face == Face::ZMAX)
      return conformal_face_status_t<3>::face_zmax(conformal_status) ==
             conformal_neighbor_status::NEIGHBOR_IS_FINER;

    return false;
  } // is_face_neighbor_finer

}; // conformal_face_status_t<3>

//! data type alias to store conformal face status (one value by AMR leaf)
template <size_t dim>
using conformal_status_t = typename conformal_face_status_t<dim>::status_t;

//! conformal face status array type alias (device)
template <size_t dim, typename device_t>
using conformal_status_view_t = typename Kokkos::View<conformal_status_t<dim> *, device_t>;

//! conformal face status array type alias (host)
template <size_t dim, typename device_t>
using conformal_status_view_host_t =
  typename Kokkos::View<conformal_status_t<dim> *,
                        typename conformal_status_view_t<dim, device_t>::host_mirror_space>;


} // namespace kalypsso

KALYPSSO_DISABLE_NVCC_WARNINGS_POP()

#endif // KALYPSSO_CORE_CONFORMALFACESTATUS_H_
