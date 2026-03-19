// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file UserDataRemapperImplBFC.cpp
 * \brief \copybrief UserDataRemapperImplBFC.h
 */
#include <kalypsso/core/UserDataRemapperImplBFC.h>
#include <kalypsso/core/FaceDataArrayBlock_utils.h>

namespace kalypsso
{
// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION void
UserDataRemapperImplBFC<dim, device_t>::operator()(TagComputeAllButInternalFaces const &,
                                                   const index_t & global_index) const
{
  const iOct_t  iOct = global_index / m_facedata_old.num_elements_per_octant();
  const int32_t face_flat_index =
    static_cast<int32_t>(global_index - iOct * m_facedata_old.num_elements_per_octant());
  auto const & bx = m_block_sizes[IX];

  //
  // the only reason of the following dummy code to be here, is that cuda nvcc compile doesn't
  // support capturing variables inside inside a constexpr if section of a device lambda
  //
  // strangely, nvc++ is ok and don't need that
  // TODO: remove theses lines when nvcc will have support for this.
#ifdef __NVCC__
  [[maybe_unused]] int dummy = 0;
  if (m_facedata_new.num_quadrants() == m_facedata_old.num_quadrants() or bx == 0)
    dummy++;
#endif

  // compute ix,iy,iz,ivar of local face inside
  // block from a face flat-index
  const auto face_indexes = face_flat_index_unravel<dim>(
    face_flat_index, m_block_sizes, m_facedata_old.offsets(), m_facedata_old.shift());

  const auto & ivar = face_indexes[dim];

  // get orchard key of current block/octant
  const auto key_new = m_orchard_keys_device_new(iOct);

  auto key_index = m_amr_hashmap_device_old.find(key_new);

  // first check if key exists in the old hashmap, if it exists, it means we have a
  // an octant that didn't change level
  if (m_amr_hashmap_device_old.valid_at(key_index))
  {
    // auto key_old = amr_hashmap_device_old.key_at(key_index);
    const auto iOct_old = m_amr_hashmap_device_old.value_at(key_index);

    // copy from old quadrant to new quadrant

    if constexpr (dim == 2)
    {
      auto const & i = face_indexes[IX];
      auto const & j = face_indexes[IY];
      m_facedata_new(i, j, ivar, iOct) = m_facedata_old(i, j, ivar, iOct_old);
    }
    else if constexpr (dim == 3)
    {
      auto const & i = face_indexes[IX];
      auto const & j = face_indexes[IY];
      auto const & k = face_indexes[IZ];
      m_facedata_new(i, j, k, ivar, iOct) = m_facedata_old(i, j, k, ivar, iOct_old);
    }
  } // end if old octant is at same level
  else
  {
    // check if the key correspond to a refinement, aka prolongation (increase of level)
    // to do that we search for father octant's key
    auto key_old = orchard_key_t<dim>::father(key_new);
    key_index = m_amr_hashmap_device_old.find(key_old);

    if (m_amr_hashmap_device_old.valid_at(key_index))
    {
      const auto iOct_old = m_amr_hashmap_device_old.value_at(key_index);

      // get family id of the new key, used to determine where exactly in the old block we
      // will copy data
      const auto fid = orchard_key_t<dim>::family_id(key_new);

      // prolongation of external face
      if (is_external_face<dim>(face_indexes))
      {

        if (m_prolongation.m_face_external == +FaceCenteredProlongationExternalType::SIMPLE_COPY)
        {
          external_faces_prolongation_by_copy(face_indexes, iOct, iOct_old, key_new, key_old, fid);
        }
        else if (m_prolongation.m_face_external ==
                 +FaceCenteredProlongationExternalType::EXTRAPOLATE_LINEAR_MINMOD)
        {
          // TODO : prolongation by linear extrapolation
        }
      }
    } // end if old octant is at finer level
    else
    {
      // check if the key correspond to a "coarsening" (decrease of level)
      // to do that we just need to search for the eldest child key in the old map,
      // and then average all the values from all the children (coarse graining values)

      /*
       * In principle, we should check that all the children are available in the map
       * but here, we just check for the eldest child key, and assume all the children
       * are also in the map (p4est ensure that anyway).
       */

      auto eldest_child_key = orchard_key_t<dim>::eldest_child(key_new);
      key_index = m_amr_hashmap_device_old.find(eldest_child_key);
      if (m_amr_hashmap_device_old.valid_at(key_index))
      {
        auto iOct_eldest = m_amr_hashmap_device_old.value_at(key_index);

        // for a given cell determine the child id where we will need to fetch data in the
        // old mesh
        uint32_t ichild = 0;
        if (face_indexes[IX] >= bx / 2)
          ichild |= 0x1;
        if (face_indexes[IY] >= bx / 2)
          ichild |= 0x2;
        if constexpr (dim == 3)
        {
          if (face_indexes[IZ] >= bx / 2)
          {
            ichild |= 0x4;
          }
        }

        // here we taken into account that a family of octant are stored contiguously in
        // memory
        auto iOct_old = iOct_eldest + ichild;

        // we need to average value over face siblings neighboring faces
        real_t coarsen_value = ZERO_F;
        if constexpr (dim == 2)
        {
          const auto i_old =
            2 * face_indexes[IX] >= bx ? 2 * face_indexes[IX] - bx : 2 * face_indexes[IX];
          const auto j_old =
            2 * face_indexes[IY] >= bx ? 2 * face_indexes[IY] - bx : 2 * face_indexes[IY];

          if (ivar == IX)
          {
            coarsen_value += m_facedata_old(i_old, j_old, ivar, iOct_old);
            coarsen_value += m_facedata_old(i_old, j_old + 1, ivar, iOct_old);
            coarsen_value /= 2;
          }
          else if (ivar == IY)
          {
            coarsen_value += m_facedata_old(i_old, j_old, ivar, iOct_old);
            coarsen_value += m_facedata_old(i_old + 1, j_old, ivar, iOct_old);
            coarsen_value /= 2;
          }
          else if (ivar == IZ)
          {
            // clang-format off
            coarsen_value += m_facedata_old(i_old    , j_old    , ivar, iOct_old);
            coarsen_value += m_facedata_old(i_old + 1, j_old    , ivar, iOct_old);
            coarsen_value += m_facedata_old(i_old    , j_old + 1, ivar, iOct_old);
            coarsen_value += m_facedata_old(i_old + 1, j_old + 1, ivar, iOct_old);
            coarsen_value /= 4;
            // clang-format on
          }
          m_facedata_new(face_indexes[IX], face_indexes[IY], ivar, iOct) = coarsen_value;
        }
        else if constexpr (dim == 3)
        {
          const auto i_old =
            2 * face_indexes[IX] >= bx ? 2 * face_indexes[IX] - bx : 2 * face_indexes[IX];
          const auto j_old =
            2 * face_indexes[IY] >= bx ? 2 * face_indexes[IY] - bx : 2 * face_indexes[IY];
          const auto k_old =
            2 * face_indexes[IZ] >= bx ? 2 * face_indexes[IZ] - bx : 2 * face_indexes[IZ];

          if (ivar == IX)
          {
            // clang-format off
            coarsen_value += m_facedata_old(i_old, j_old    , k_old    , ivar, iOct_old);
            coarsen_value += m_facedata_old(i_old, j_old + 1, k_old    , ivar, iOct_old);
            coarsen_value += m_facedata_old(i_old, j_old    , k_old + 1, ivar, iOct_old);
            coarsen_value += m_facedata_old(i_old, j_old + 1, k_old + 1, ivar, iOct_old);
            // clang-format on
          }
          else if (ivar == IY)
          {
            // clang-format off
            coarsen_value += m_facedata_old(i_old    , j_old, k_old    , ivar, iOct_old);
            coarsen_value += m_facedata_old(i_old + 1, j_old, k_old    , ivar, iOct_old);
            coarsen_value += m_facedata_old(i_old    , j_old, k_old + 1, ivar, iOct_old);
            coarsen_value += m_facedata_old(i_old + 1, j_old, k_old + 1, ivar, iOct_old);
            // clang-format on
          }
          else if (ivar == IZ)
          {
            // clang-format off
            coarsen_value += m_facedata_old(i_old    , j_old    , k_old, ivar, iOct_old);
            coarsen_value += m_facedata_old(i_old + 1, j_old    , k_old, ivar, iOct_old);
            coarsen_value += m_facedata_old(i_old    , j_old + 1, k_old, ivar, iOct_old);
            coarsen_value += m_facedata_old(i_old + 1, j_old + 1, k_old, ivar, iOct_old);
            // clang-format on
          }
          m_facedata_new(face_indexes[IX], face_indexes[IY], face_indexes[IZ], ivar, iOct) =
            coarsen_value / 4;
        }
      }
    } // end if new octant is at coarser level
  }

} // operator() - all faces but not internal faces when prolongating

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION void
UserDataRemapperImplBFC<dim, device_t>::operator()(TagComputeInternalFaces const &,
                                                   const index_t & global_index) const
{

  const iOct_t  iOct = global_index / m_facedata_old.num_elements_per_octant();
  const int32_t face_flat_index =
    static_cast<int32_t>(global_index - iOct * m_facedata_old.num_elements_per_octant());

  //
  // the only reason of the following dummy code to be here, is that cuda nvcc compile doesn't
  // support capturing variables inside inside a constexpr if section of a device lambda
  //
  // strangely, nvc++ is ok and don't need that
  // TODO: remove theses lines when nvcc will have support for this.
#ifdef __NVCC__
  [[maybe_unused]] int dummy = 0;
  if (m_facedata_new.num_quadrants() == m_facedata_old.num_quadrants())
    dummy++;
#endif

  // compute ix,iy,iz,ivar of local face inside
  // block from a face flat-index
  const auto face_indexes = face_flat_index_unravel<dim>(
    face_flat_index, m_block_sizes, m_facedata_old.offsets(), m_facedata_old.shift());

  // get orchard key of current block/octant
  const auto key_new = m_orchard_keys_device_new(iOct);

  // get orchard key of parent (it must exist when doing prolongation)
  const auto key_old = orchard_key_t<dim>::father(key_new);

  auto key_old_index = m_amr_hashmap_device_old.find(key_old);

  // key must be valid as we are only interested in prologation (from coarse to fine level)
  if (m_amr_hashmap_device_old.valid_at(key_old_index))
  {
    if (is_internal_face<dim>(face_indexes))
    {

      if (m_prolongation.m_face_internal == +FaceCenteredProlongationInternalType::TOTH_AND_ROE)
      {
        internal_faces_prolongation_by_toth_and_roe(face_indexes, iOct);
      }

    } // end is_internal_face
  } // end if old octant is at finer level

} // operator () - prolongation at internal faces

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
UserDataRemapperImplBFC<dim, device_t>::external_faces_prolongation_by_copy(
  face_multiindex_t<dim> const & face_indexes_new,
  iOct_t const &                 iOct_new,
  iOct_t const &                 iOct_old,
  uint64_t const &               key_new,
  uint64_t const &               key_old,
  uint32_t                       family_id) const
{
  auto const & bx = m_block_sizes[IX];
  const auto & ivar = face_indexes_new[dim];

  if constexpr (dim == 2)
  {
    auto const & i_new = face_indexes_new[IX];
    auto const & j_new = face_indexes_new[IY];

    const auto i_old = i_new / 2 + static_cast<int32_t>((family_id & 0x1) >> 0) * (bx / 2);
    const auto j_old = j_new / 2 + static_cast<int32_t>((family_id & 0x2) >> 1) * (bx / 2);

    // convert face index into cell index, i.e. in range [0, bx-1]
    const coord_t<dim> ijk_old{ i_old == bx ? i_old - 1 : i_old, j_old == bx ? j_old - 1 : j_old };

    const CellLocation_t cell_loc_old{ ijk_old, key_old, iOct_old, false };

    // the following initialization is also valid when ivar = IZ
    auto old_value = m_facedata_old(i_old, j_old, ivar, iOct_old);

    // first check if face is at block border of old octant, and if neighbor is already at fine
    // in that case we copy data from neighbor
    if (ivar == IX and i_old == 0)
    {
      const auto cell_loc_neigh =
        m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(-XDIR));

      if (cell_loc_neigh.level() == orchard_key_t<dim>::level(key_new))
      {
        old_value = m_facedata_old(bx, j_new, ivar, cell_loc_neigh.iOct);
      }
    }
    else if (ivar == IX and i_old == bx)
    {
      const auto cell_loc_neigh =
        m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(+XDIR));

      if (cell_loc_neigh.level() == orchard_key_t<dim>::level(key_new))
      {
        old_value = m_facedata_old(0, j_new, ivar, cell_loc_neigh.iOct);
      }
    }
    else if (ivar == IY and j_old == 0)
    {
      const auto cell_loc_neigh =
        m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(-YDIR));

      if (cell_loc_neigh.level() == orchard_key_t<dim>::level(key_new))
      {
        old_value = m_facedata_old(i_new, bx, ivar, cell_loc_neigh.iOct);
      }
    }
    else if (ivar == IY and j_old == bx)
    {
      const auto cell_loc_neigh =
        m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(+YDIR));

      if (cell_loc_neigh.level() == orchard_key_t<dim>::level(key_new))
      {
        old_value = m_facedata_old(i_new, 0, ivar, cell_loc_neigh.iOct);
      }
    }

    // copy data from the old mesh (at coarse level)
    m_facedata_new(i_new, j_new, ivar, iOct_new) = old_value;
  }
  else // 3D
  {
    auto const & i_new = face_indexes_new[IX];
    auto const & j_new = face_indexes_new[IY];
    auto const & k_new = face_indexes_new[IZ];

    const auto i_old = i_new / 2 + static_cast<int32_t>((family_id & 0x1) >> 0) * (bx / 2);
    const auto j_old = j_new / 2 + static_cast<int32_t>((family_id & 0x2) >> 1) * (bx / 2);
    const auto k_old = k_new / 2 + static_cast<int32_t>((family_id & 0x4) >> 2) * (bx / 2);

    // convert face index into cell index, i.e. in range [0, bx-1]
    const coord_t<dim> ijk_old{ i_old == bx ? i_old - 1 : i_old,
                                j_old == bx ? j_old - 1 : j_old,
                                k_old == bx ? k_old - 1 : k_old };

    const CellLocation_t cell_loc_old{ ijk_old, key_old, iOct_old, false };

    auto old_value = m_facedata_old(i_old, j_old, k_old, ivar, iOct_old);

    // first check if face is at block border of old octant, and if neighbor is already at fine
    // in that case we copy data from neighbor
    if (ivar == IX and i_old == 0)
    {
      const auto cell_loc_neigh =
        m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(-XDIR));

      if (cell_loc_neigh.level() == orchard_key_t<dim>::level(key_new))
      {
        old_value = m_facedata_old(bx, j_new, k_new, ivar, cell_loc_neigh.iOct);
      }
    }
    else if (ivar == IX and i_old == bx)
    {
      const auto cell_loc_neigh =
        m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(+XDIR));

      if (cell_loc_neigh.level() == orchard_key_t<dim>::level(key_new))
      {
        old_value = m_facedata_old(0, j_new, k_new, ivar, cell_loc_neigh.iOct);
      }
    }
    else if (ivar == IY and j_old == 0)
    {
      const auto cell_loc_neigh =
        m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(-YDIR));

      if (cell_loc_neigh.level() == orchard_key_t<dim>::level(key_new))
      {
        old_value = m_facedata_old(i_new, bx, k_new, ivar, cell_loc_neigh.iOct);
      };
    }
    else if (ivar == IY and j_old == bx)
    {
      const auto cell_loc_neigh =
        m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(+YDIR));

      if (cell_loc_neigh.level() == orchard_key_t<dim>::level(key_new))
      {
        old_value = m_facedata_old(i_new, 0, k_new, ivar, cell_loc_neigh.iOct);
      }
    }
    else if (ivar == IZ and k_old == 0)
    {
      const auto cell_loc_neigh =
        m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(-ZDIR));

      if (cell_loc_neigh.level() == orchard_key_t<dim>::level(key_new))
      {
        old_value = m_facedata_old(i_new, j_new, bx, ivar, cell_loc_neigh.iOct);
      };
    }
    else if (ivar == IZ and k_old == bx)
    {
      const auto cell_loc_neigh =
        m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(+ZDIR));

      if (cell_loc_neigh.level() == orchard_key_t<dim>::level(key_new))
      {
        old_value = m_facedata_old(i_new, j_new, 0, ivar, cell_loc_neigh.iOct);
      }
    }

    // copy data from the old mesh (at coarse level)
    m_facedata_new(i_new, j_new, k_new, ivar, iOct_new) = old_value;

  } // end 3D

} // external_faces_prolongation_by_copy

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
UserDataRemapperImplBFC<dim, device_t>::internal_faces_prolongation_by_toth_and_roe(
  face_multiindex_t<dim> const & face_indexes_new,
  iOct_t const &                 iOct_new) const
{
  const auto & ivar = face_indexes_new[dim];

  if constexpr (dim == 2)
  {
    auto const & i_new = face_indexes_new[IX];
    auto const & j_new = face_indexes_new[IY];

    // "eldest" face : all coords even
    const auto i_e = (i_new / 2) * 2;
    const auto j_e = (j_new / 2) * 2;

    // reminder: we must use m_facedata_new for accessing external faces

    // note : when ivar = IZ (in 2D) there are no "internal" faces
    // in other words, ivar = IZ is already taken into account when filling external faces

    real_t Uxx = ZERO_F;
    real_t Vyy = ZERO_F;
    for (int j = -1; j <= 1; j += 2)
      for (int i = -1; i <= 1; i += 2)
      {
        // clang-format off
        Uxx += static_cast<real_t>(i * j) * m_facedata_new(i_e + (i + 1) / 2, j_e + (j + 1)    , IY, iOct_new);
        Vyy += static_cast<real_t>(i * j) * m_facedata_new(i_e + (i + 1)    , j_e + (j + 1) / 2, IX, iOct_new);
        // clang-format on
      }
    Uxx /= 4;
    Vyy /= 4;

    if (ivar == IX)
    {
      auto mid_value = HALF_F * (m_facedata_new(i_new - 1, j_new, IX, iOct_new) +
                                 m_facedata_new(i_new + 1, j_new, IX, iOct_new));

      m_facedata_new(i_new, j_new, ivar, iOct_new) = mid_value + Uxx;
    }
    else if (ivar == IY)
    {
      auto mid_value = HALF_F * (m_facedata_new(i_new, j_new - 1, IY, iOct_new) +
                                 m_facedata_new(i_new, j_new + 1, IY, iOct_new));

      m_facedata_new(i_new, j_new, ivar, iOct_new) = mid_value + Vyy;
    }
  }
  else if constexpr (dim == 3)
  {
    auto const & i_new = face_indexes_new[IX];
    auto const & j_new = face_indexes_new[IY];
    auto const & k_new = face_indexes_new[IZ];

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

    // reminder: we must use m_facedata_new for accessing external faces

    for (int k = -1; k <= 1; k += 2)
      for (int j = -1; j <= 1; j += 2)
        for (int i = -1; i <= 1; i += 2)
        {
          // clang-format off
          Uxx +=
            static_cast<real_t>(i * j) *
            m_facedata_new(i_e + (i + 1) / 2, j_e + (j + 1)    , k_e + (k + 1) / 2, IY, iOct_new);
          Uxx +=
            static_cast<real_t>(i * k) *
            m_facedata_new(i_e + (i + 1) / 2, j_e + (j + 1) / 2, k_e + (k + 1)    , IZ, iOct_new);

          Vyy +=
            static_cast<real_t>(j * k) *
            m_facedata_new(i_e + (i + 1) / 2, j_e + (j + 1) / 2, k_e + (k + 1)    , IZ, iOct_new);
          Vyy +=
            static_cast<real_t>(j * i) *
            m_facedata_new(i_e + (i + 1)    , j_e + (j + 1) / 2, k_e + (k + 1) / 2, IX, iOct_new);

          Wzz +=
            static_cast<real_t>(k * i) *
            m_facedata_new(i_e + (i + 1)    , j_e + (j + 1) / 2, k_e + (k + 1) / 2, IX, iOct_new);
          Wzz +=
            static_cast<real_t>(k * j) *
            m_facedata_new(i_e + (i + 1) / 2, j_e + (j + 1)    , k_e + (k + 1) / 2, IY, iOct_new);

          Uxyz +=
            static_cast<real_t>(i * j * k) *
            m_facedata_new(i_e + (i + 1)    , j_e + (j + 1) / 2, k_e + (k + 1) / 2, IX, iOct_new);

          Vxyz +=
            static_cast<real_t>(i * j * k) *
            m_facedata_new(i_e + (i + 1) / 2, j_e + (j + 1)    , k_e + (k + 1) / 2, IY, iOct_new);

          Wxyz +=
            static_cast<real_t>(i * j * k) *
            m_facedata_new(i_e + (i + 1) / 2, j_e + (j + 1) / 2, k_e + (k + 1)    , IZ, iOct_new);
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
      auto mid_value = HALF_F * (m_facedata_new(i_new - 1, j_new, k_new, IX, iOct_new) +
                                 m_facedata_new(i_new + 1, j_new, k_new, IX, iOct_new));

      m_facedata_new(i_new, j_new, k_new, ivar, iOct_new) =
        mid_value + Uxx + static_cast<real_t>(kk) * Vxyz + static_cast<real_t>(jj) * Wxyz;
    }
    if (ivar == IY)
    {
      auto mid_value = HALF_F * (m_facedata_new(i_new, j_new - 1, k_new, IY, iOct_new) +
                                 m_facedata_new(i_new, j_new + 1, k_new, IY, iOct_new));

      m_facedata_new(i_new, j_new, k_new, ivar, iOct_new) =
        mid_value + Vyy + static_cast<real_t>(ii) * Wxyz + static_cast<real_t>(kk) * Uxyz;
    }
    if (ivar == IZ)
    {
      auto mid_value = HALF_F * (m_facedata_new(i_new, j_new, k_new - 1, IZ, iOct_new) +
                                 m_facedata_new(i_new, j_new, k_new + 1, IZ, iOct_new));

      m_facedata_new(i_new, j_new, k_new, ivar, iOct_new) =
        mid_value + Wzz + static_cast<real_t>(jj) * Uxyz + static_cast<real_t>(ii) * Vxyz;
    }

  } // end 3D

} // internal_faces_prolongation_by_toth_and_roe

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
UserDataRemapperImplBFC<dim, device_t>::linear_extrapolate_using_limited_slopes(
  CellLocation<2> const & cell_loc_old,
  coord_t<2> const &      coord_new,
  index_t const &         cellindex_new,
  int32_t const &         iOct_global) const
{
  // const real_t slope_type = 1;
  // const auto   nbvar = m_facedata_old.num_vars();

  // // get neighbor location, each of then may be at finer, same or coarser level compared
  // // cell_loc_old
  // const auto cell_loc_left_x =
  //   m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(-XDIR));
  // const auto cell_loc_right_x =
  //   m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(+XDIR));

  // const auto cell_loc_left_y =
  //   m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(-YDIR));
  // const auto cell_loc_right_y =
  //   m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(+YDIR));

  // // determine local position of current new cell inside old parent cell using integer
  // coordinates
  // // in -1, +1
  // const int ix = 2 * (coord_new[IX] - 2 * (coord_new[IX] / 2)) - 1;
  // const int iy = 2 * (coord_new[IY] - 2 * (coord_new[IY] / 2)) - 1;

  // for (int32_t ivar = 0; ivar < nbvar; ++ivar)
  // {
  //   // compute limited slopes
  //   auto const dudx = m_stencil_helper.compute_minmod_slopes_prolongation(
  //     cell_loc_old, cell_loc_right_x, cell_loc_left_x, ivar, m_facedata_old, slope_type);
  //   auto const dudy = m_stencil_helper.compute_minmod_slopes_prolongation(
  //     cell_loc_old, cell_loc_right_y, cell_loc_left_y, ivar, m_facedata_old, slope_type);

  //   // extrapolate
  //   m_facedata_new(cellindex_new, ivar, iOct_global) =
  //     m_facedata_old(cell_loc_old.cellindex(m_block_sizes), ivar, cell_loc_old.iOct) +
  //     ONE_FOURTH_F * ix * dudx + ONE_FOURTH_F * iy * dudy;
  // }
} // linear_extrapolate_using_limited_slopes - 2d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
UserDataRemapperImplBFC<dim, device_t>::linear_extrapolate_using_limited_slopes(
  CellLocation<3> const & cell_loc_old,
  coord_t<3> const &      coord_new,
  index_t const &         cellindex_new,
  int32_t const &         iOct_global) const
{
  // const real_t slope_type = 1;
  // const auto   nbvar = m_facedata_old.num_vars();

  // // get neighbor location, each of then may be at finer, same or coarser level compared
  // // cell_loc_old
  // const auto cell_loc_left_x =
  //   m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(-XDIR));
  // const auto cell_loc_right_x =
  //   m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(+XDIR));

  // const auto cell_loc_left_y =
  //   m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(-YDIR));
  // const auto cell_loc_right_y =
  //   m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(+YDIR));

  // const auto cell_loc_left_z =
  //   m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(-ZDIR));
  // const auto cell_loc_right_z =
  //   m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(+ZDIR));

  // // determine local position of current new cell inside old parent cell using integer
  // coordinates
  // // in -1, +1
  // const int ix = 2 * (coord_new[IX] - 2 * (coord_new[IX] / 2)) - 1;
  // const int iy = 2 * (coord_new[IY] - 2 * (coord_new[IY] / 2)) - 1;
  // const int iz = 2 * (coord_new[IZ] - 2 * (coord_new[IZ] / 2)) - 1;

  // for (int32_t ivar = 0; ivar < nbvar; ++ivar)
  // {
  //   // compute limited slopes
  //   auto const dudx = m_stencil_helper.compute_minmod_slopes_prolongation(
  //     cell_loc_old, cell_loc_right_x, cell_loc_left_x, ivar, m_facedata_old, slope_type);
  //   auto const dudy = m_stencil_helper.compute_minmod_slopes_prolongation(
  //     cell_loc_old, cell_loc_right_y, cell_loc_left_y, ivar, m_facedata_old, slope_type);
  //   auto const dudz = m_stencil_helper.compute_minmod_slopes_prolongation(
  //     cell_loc_old, cell_loc_right_z, cell_loc_left_z, ivar, m_facedata_old, slope_type);

  //   // extrapolate
  //   m_facedata_new(cellindex_new, ivar, iOct_global) =
  //     m_facedata_old(cell_loc_old.cellindex(m_block_sizes), ivar, cell_loc_old.iOct) +
  //     ONE_FOURTH_F * ix * dudx + ONE_FOURTH_F * iy * dudy + ONE_FOURTH_F * iz * dudz;
  // }
} // linear_extrapolate_using_limited_slopes - 3d

// explicit template instantiation
template class UserDataRemapperImplBFC<2, kalypsso::DefaultDevice>;
template class UserDataRemapperImplBFC<3, kalypsso::DefaultDevice>;

#ifdef KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE
template class UserDataRemapperImplBFC<2, HostDevice>;
template class UserDataRemapperImplBFC<3, HostDevice>;
#endif // KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE

} // namespace kalypsso
