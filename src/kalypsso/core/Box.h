// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file Box.h
 */
#ifndef KALYPSSO_CORE_BOX_H_
#define KALYPSSO_CORE_BOX_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/enums.h>
#include <kalypsso/core/kalypsso_core_config.h> // for KALYPSSO_CORE_ENABLE_DEBUG
#include <kalypsso/core/utils_block.h>
#include <../better-enums/enum.h>

namespace kalypsso
{
namespace core
{

/**
 * \class Box
 *
 * Just a rectangular shaped box. Will be useful for representing a ghosted block.
 * Below is a 8x8 block of cells, surrounded by a halo of one cell.
 *
 *
 * +----+----+----+----+----+----+----+----+----+----+
 * |░░░░|░░░░|░░░░|░░░░|░░░░|░░░░|░░░░|░░░░|░░░░|░░░░|
 * |░░░░|░░░░|░░░░|░░░░|░░░░|░░░░|░░░░|░░░░|░░░░|░░░░|
 * +----+----+----+----+----+----+----+----+----+----+
 * |░░░░|    |    |    |    |    |    |    |    |░░░░|
 * |░░░░|    |    |    |    |    |    |    |    |░░░░|
 * +----+----+----+----+----+----+----+----+----+----+
 * |░░░░|    |    |    |    |    |    |    |    |░░░░|
 * |░░░░|    |    |    |    |    |    |    |    |░░░░|
 * +----+----+----+----+----+----+----+----+----+----+
 * |░░░░|    |    |    |    |    |    |    |    |░░░░|
 * |░░░░|    |    |    |    |    |    |    |    |░░░░|
 * +----+----+----+----+----+----+----+----+----+----+
 * |░░░░|    |    |    |    |    |    |    |    |░░░░|
 * |░░░░|    |    |    |    |    |    |    |    |░░░░|
 * +----+----+----+----+----+----+----+----+----+----+
 * |░░░░|    |    |    |    |    |    |    |    |░░░░|
 * |░░░░|    |    |    |    |    |    |    |    |░░░░|
 * +----+----+----+----+----+----+----+----+----+----+
 * |░░░░|    |    |    |    |    |    |    |    |░░░░|
 * |░░░░|    |    |    |    |    |    |    |    |░░░░|
 * +----+----+----+----+----+----+----+----+----+----+
 * |░░░░|    |    |    |    |    |    |    |    |░░░░|
 * |░░░░|    |    |    |    |    |    |    |    |░░░░|
 * +----+----+----+----+----+----+----+----+----+----+
 * |░░░░|    |    |    |    |    |    |    |    |░░░░|
 * |░░░░| O  |    |    |    |    |    |    |    |░░░░|
 * +----+----+----+----+----+----+----+----+----+----+
 * |░░░░|░░░░|░░░░|░░░░|░░░░|░░░░|░░░░|░░░░|░░░░|░░░░|
 * |░░░░|░░░░|░░░░|░░░░|░░░░|░░░░|░░░░|░░░░|░░░░|░░░░|
 * +----+----+----+----+----+----+----+----+----+----+
 *
 */
template <size_t dim>
class Box
{

private:
  //! origin (lower left corner coordinates)
  coord_t<dim, real_t> m_origin;

  //! rectangular box shape
  coord_t<dim, real_t> m_shape;

  //!
  KOKKOS_INLINE_FUNCTION
  void
  intersect_1d(real_t const & origin0,
               real_t const & length0,
               real_t const & origin1,
               real_t const & length1,
               real_t &       origin2,
               real_t &       length2) const
  {
    if ((origin0 + length0 < origin1) or (origin1 + length1 < origin0))
    {
      // no intersection
      origin2 = ZERO_F;
      length2 = ZERO_F;
    }
    else
    {
      origin2 = fmax(origin0, origin1);
      length2 = fmin(origin0 + length0, origin1 + length1) - origin2;
    }
  } // intersect_1d

public:
  //! constructor
  KOKKOS_INLINE_FUNCTION
  Box()
    : m_origin(make_coord<dim, real_t>(ZERO_F))
    , m_shape(make_coord<dim, real_t>(ZERO_F)) {};

  //! destructor
  KOKKOS_DEFAULTED_FUNCTION ~Box() = default;

  //! copy constructor
  KOKKOS_DEFAULTED_FUNCTION
  Box(Box const & box) = default;

  KOKKOS_INLINE_FUNCTION
  Box(coord_t<dim, real_t> const & origin, coord_t<dim, real_t> const & shape)
    : m_origin(origin)
    , m_shape(shape)
  {
    KOKKOS_ASSERT((m_shape[IX] >= 0) && "Box: shape[IX] must be non negative");
    KOKKOS_ASSERT((m_shape[IY] >= 0) && "Box: shape[IY] must be non negative");
    if constexpr (dim == 3)
      KOKKOS_ASSERT((m_shape[IZ] >= 0) && "Box: shape[IZ] must be non negative");
  };

  KOKKOS_INLINE_FUNCTION
  coord_t<dim, real_t> const &
  origin() const
  {
    return m_origin;
  }

  KOKKOS_INLINE_FUNCTION
  coord_t<dim, real_t> &
  origin()
  {
    return m_origin;
  }

  KOKKOS_INLINE_FUNCTION
  coord_t<dim, real_t> const &
  shape() const
  {
    return m_shape;
  }

  KOKKOS_INLINE_FUNCTION
  coord_t<dim, real_t> &
  shape()
  {
    return m_shape;
  }

  //! scale box shape without modifying origin
  //!
  //! \param[in] scale_factor
  KOKKOS_INLINE_FUNCTION
  void
  scale(real_t scale_factor)
  {
    KOKKOS_ASSERT((scale_factor >= 0) && "Box: scale_factor must be non negative");

    m_shape[IX] *= scale_factor;
    m_shape[IY] *= scale_factor;
    if constexpr (dim == 3)
      m_shape[IZ] *= scale_factor;
  }

  //! translate box shape without modifying shape
  //!
  //! \param[in] shift
  KOKKOS_INLINE_FUNCTION
  void
  translate(coord_t<dim, real_t> const & shift)
  {
    m_origin[IX] += shift[IX];
    m_origin[IY] += shift[IY];
    if constexpr (dim == 3)
      m_origin[IZ] += shift[IZ];
  }

  //! Create an new box s the intersection of current box with another
  //!
  //! \param[in] other another box
  KOKKOS_INLINE_FUNCTION Box<dim>
                         intersect(Box<dim> const & other)
  {
    auto b = Box<dim>();

    intersect_1d(m_origin[IX],
                 m_shape[IX],
                 other.origin()[IX],
                 other.shape()[IX],
                 b.origin()[IX],
                 b.shape()[IX]);
    intersect_1d(m_origin[IY],
                 m_shape[IY],
                 other.origin()[IY],
                 other.shape()[IY],
                 b.origin()[IY],
                 b.shape()[IY]);
    if constexpr (dim == 3)
      intersect_1d(m_origin[IZ],
                   m_shape[IZ],
                   other.origin()[IZ],
                   other.shape()[IZ],
                   b.origin()[IZ],
                   b.shape()[IZ]);

    return b;

  } // intersect

  //! Create an new box s the intersection of current box with another
  //!
  KOKKOS_INLINE_FUNCTION
  real_t
  area()
  {
    if constexpr (dim == 2)
      return m_shape[IX] * m_shape[IY];
    else if constexpr (dim == 3)
      return m_shape[IX] * m_shape[IY] * m_shape[IZ];

    return ZERO_F;
  } // area

}; // class Box

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_BOX_H_
