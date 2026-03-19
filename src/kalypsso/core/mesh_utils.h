// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file mesh_utils.h
 */
#ifndef KALYPSSO_CORE_MESHUTILS_H_
#define KALYPSSO_CORE_MESHUTILS_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <kalypsso/core/enums.h>

#include <cstdint>
#include <array>

namespace kalypsso
{

// =======================================================================
// =======================================================================
//! \struct Face
//! Don't change the values, don't insert new values unless you know what you're doing.
//! Values defined here are also used at some location to compute direction (integer division by 2)
struct Face
{
  using face_t = uint8_t;
  static constexpr face_t XMIN = 0;
  static constexpr face_t XMAX = 1;
  static constexpr face_t YMIN = 2;
  static constexpr face_t YMAX = 3;
  static constexpr face_t ZMIN = 4;
  static constexpr face_t ZMAX = 5;
  static constexpr face_t INVALID = 16;
  static constexpr face_t MIN = 0;
  static constexpr face_t MAX = 1;
  static constexpr face_t NUM_FACES_2D = 4;
  static constexpr face_t NUM_FACES_3D = 6;

  //! return an array with all faces
  template <size_t dim>
  KOKKOS_INLINE_FUNCTION static constexpr auto
  get_all_faces() -> std::array<face_t, 2 * dim>
  {
    if constexpr (dim == 2)
      return { Face::XMIN, Face::XMAX, Face::YMIN, Face::YMAX };
    else if constexpr (dim == 3)
      return { Face::XMIN, Face::XMAX, Face::YMIN, Face::YMAX, Face::ZMIN, Face::ZMAX };
  }

  //! return number of faces
  template <size_t dim>
  KOKKOS_INLINE_FUNCTION static constexpr auto
  num_faces()
  {
    if constexpr (dim == 2)
      return NUM_FACES_2D;
    else if constexpr (dim == 3)
      return NUM_FACES_3D;
  } // num_faces

  //! return true if face is a left face
  KOKKOS_INLINE_FUNCTION static constexpr bool
  is_left_face(face_t face)
  {
    return face == Face::XMIN or face == Face::YMIN or face == Face::ZMIN;
  }

  template <size_t dim>
  KOKKOS_INLINE_FUNCTION static auto
  get_pair_of_faces(int dir) -> Kokkos::Array<face_t, 2>
  {
    if (dir == IX)
      return Kokkos::Array<face_t, 2>{ Face::XMIN, Face::XMAX };
    else if (dir == IY)
      return Kokkos::Array<face_t, 2>{ Face::YMIN, Face::YMAX };

    if constexpr (dim == 3)
    {
      if (dir == IZ)
        return Kokkos::Array<face_t, 2>{ Face::ZMIN, Face::ZMAX };
    }

    return Kokkos::Array<face_t, 2>{ Face::XMIN, Face::XMAX };
  }

}; // struct Face

// =======================================================================
// =======================================================================
//! \struct Edge.
struct Edge
{
  using edge_t = uint8_t;
  static constexpr edge_t NUM_EDGES_2D = 0;
  static constexpr edge_t NUM_EDGES_3D = 12;

  template <size_t dim>
  static constexpr uint8_t
  num_edges()
  {
    if constexpr (dim == 2)
      return NUM_EDGES_2D;
    else if constexpr (dim == 3)
      return NUM_EDGES_3D;
  }
}; // struct Edge

// =======================================================================
// =======================================================================
// \struct Corner.
struct Corner
{
  using corner_t = uint8_t;
  static constexpr corner_t CORNER_0 = 0;
  static constexpr corner_t CORNER_1 = 1;
  static constexpr corner_t CORNER_2 = 2;
  static constexpr corner_t CORNER_3 = 3;
  static constexpr corner_t CORNER_4 = 4;
  static constexpr corner_t CORNER_5 = 5;
  static constexpr corner_t CORNER_6 = 6;
  static constexpr corner_t CORNER_7 = 7;

  template <size_t dim>
  KOKKOS_INLINE_FUNCTION static constexpr uint8_t
  num_corners()
  {
    return 1 << dim;
  }
}; // struct Corner

// =======================================================================
// =======================================================================
/**
 * corner to faces - 2d.
 *
 * corners are identified by two faces.
 *
 * \param[in] corner is a corner id (from 0 to 3 in 2d)
 * \param[out] face0
 * \param[out] face1
 *
 * face0 and face1 identify a unique corner in 2d using the following mapping
 *
 * corner is encoded using 2 bits
 *
 */
KOKKOS_INLINE_FUNCTION void
corner_to_faces(uint8_t corner, Face::face_t & face0, Face::face_t & face1)
{
  KOKKOS_ASSERT(corner < 4 && "WRONG VALUE, corner must be smaller than 4.");

  face0 = corner & 0x1 ? Face::XMAX : Face::XMIN;
  face1 = corner & 0x2 ? Face::YMAX : Face::YMIN;

} // corner_to_faces

// =======================================================================
// =======================================================================
KOKKOS_INLINE_FUNCTION void
corner_to_faces(uint8_t corner, Face::face_t & face0, Face::face_t & face1, Face::face_t & face2)
{
  KOKKOS_ASSERT(corner < 8 && "WRONG VALUE, corner must be smaller than 8.");

  face0 = corner & 0x1 ? Face::XMAX : Face::XMIN;
  face1 = corner & 0x2 ? Face::YMAX : Face::YMIN;
  face2 = corner & 0x4 ? Face::ZMAX : Face::ZMIN;

} // corner_to_faces

// =======================================================================
// =======================================================================
template <size_t dim>
KOKKOS_INLINE_FUNCTION Kokkos::Array<Face::face_t, dim>
                       corner_to_faces(uint8_t corner)
{
  if constexpr (dim == 2)
  {
    KOKKOS_ASSERT(corner < 4 && "WRONG VALUE, corner must be smaller than 4.");
  }
  else if constexpr (dim == 3)
  {
    KOKKOS_ASSERT(corner < 8 && "WRONG VALUE, corner must be smaller than 8.");
  }

  Kokkos::Array<Face::face_t, dim> res;
  res[IX] = corner & 0x1 ? Face::XMAX : Face::XMIN;
  res[IY] = corner & 0x2 ? Face::YMAX : Face::YMIN;
  if constexpr (dim == 3)
    res[IZ] = corner & 0x4 ? Face::ZMAX : Face::ZMIN;

  return res;
} // corner_to_faces

// ===========================================================
// ===========================================================
/**
 * edge to faces.
 *
 * edges are identified by to faces.
 *
 * \param[in] edge is an edge id (from 0 to 11)
 * \param[out] face0
 * \param[out] face1
 *
 * face0 and face1 identify a unique edge using the following mapping
 *
 * edge is encoded using 4 bits
 *
 * we respect Morton order, with this choice, we always have dir0=face0/2 < dir1=face1/2
 */
KOKKOS_INLINE_FUNCTION void
edge_to_faces(uint8_t edge, Face::face_t & face0, Face::face_t & face1)
{
  KOKKOS_ASSERT(edge < Edge::NUM_EDGES_3D &&
                "WRONG VALUE, edge must be smaller than NUM_EDGES_3D.");

  // edge along Z
  if (edge < 4)
  {
    face0 = edge & 0x1 ? Face::XMAX : Face::XMIN;
    face1 = edge & 0x2 ? Face::YMAX : Face::YMIN;
  }
  // edge along X
  else if (edge < 8)
  {
    face0 = edge & 0x1 ? Face::YMAX : Face::YMIN;
    face1 = edge & 0x2 ? Face::ZMAX : Face::ZMIN;
  }
  // edge along Y (not a circular permutation - respect Morton order)
  else
  {
    face0 = edge & 0x1 ? Face::XMAX : Face::XMIN;
    face1 = edge & 0x2 ? Face::ZMAX : Face::ZMIN;
  }

  KOKKOS_ASSERT(face0 < face1 && "WRONG VALUE, face0,face1 is invalid.");

} // edge_to_faces

// ===========================================================
// ===========================================================
KOKKOS_INLINE_FUNCTION Kokkos::Array<Face::face_t, 2>
                       edge_to_faces(Edge::edge_t edge)
{
  KOKKOS_ASSERT(edge < Edge::NUM_EDGES_3D &&
                "WRONG VALUE, edge must be smaller than NUM_EDGES_3D.");

  Kokkos::Array<Face::face_t, 2> faces;

  // edge along Z
  if (edge < 4)
  {
    faces[0] = edge & 0x1 ? Face::XMAX : Face::XMIN;
    faces[1] = edge & 0x2 ? Face::YMAX : Face::YMIN;
  }
  // edge along X
  else if (edge < 8)
  {
    faces[0] = edge & 0x1 ? Face::YMAX : Face::YMIN;
    faces[1] = edge & 0x2 ? Face::ZMAX : Face::ZMIN;
  }
  // edge along Y (not a circular permutation - respect Morton order)
  else
  {
    faces[0] = edge & 0x1 ? Face::XMAX : Face::XMIN;
    faces[1] = edge & 0x2 ? Face::ZMAX : Face::ZMIN;
  }

  KOKKOS_ASSERT(faces[0] < faces[1] && "WRONG VALUE, faces[0],face[1] is invalid.");

  return faces;

} // edge_to_faces

// ==========================================================
// ==========================================================
/**
 * Get normal unit vector in a given direction.
 *
 * \param[in] direction specifies which component is non-zero
 *
 * \return a dim-dimensional Kokkos::Array, which is zero everywhere except one component identified
 * by direction
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION auto
get_unit_vector(int direction)
{
  Kokkos::Array<int, dim> v;
  for (int i = 0; i < static_cast<int>(dim); ++i)
  {
    v[i] = i == direction ? 1 : 0;
  }
  return v;
} // get_unit_vector

// ==========================================================
// ==========================================================
/**
 * Given an outside unit vector return the face Id.
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION Face::face_t
                       face_normal_vector_to_face_id(Kokkos::Array<int32_t, dim> const & v)
{

  {
    [[maybe_unused]] auto norm_v = v[IX] * v[IX] + v[IY] * v[IY];
    if constexpr (dim == 3)
      norm_v += v[IZ] * v[IZ];
    KOKKOS_ASSERT(norm_v == 1 && "Error a unit vector must have norm 1 !");
  }

  if constexpr (dim == 2)
  {
    if (v[IX] == -1)
      return Face::XMIN;
    else if (v[IX] == 1)
      return Face::XMAX;
    else if (v[IY] == -1)
      return Face::YMIN;
    else if (v[IY] == 1)
      return Face::YMAX;
  }
  else if constexpr (dim == 3)
  {
    if (v[IX] == -1)
      return Face::XMIN;
    else if (v[IX] == 1)
      return Face::XMAX;
    else if (v[IY] == -1)
      return Face::YMIN;
    else if (v[IY] == 1)
      return Face::YMAX;
    else if (v[IZ] == -1)
      return Face::ZMIN;
    else if (v[IZ] == 1)
      return Face::ZMAX;
  }

  return Face::INVALID;

} // face_unit_vector_to_face_id

} // namespace kalypsso

#endif // KALYPSSO_CORE_MESHUTILS_H_
