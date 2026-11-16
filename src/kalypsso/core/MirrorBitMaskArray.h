// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MirrorQuadrantUtilsMetadata.h
 *
 * This is a container for metadata related to mirror quadrants (used in ghost exchange).
 */
#ifndef KALYPSSO_CORE_MIRRORBITMASKARRAY_H_
#define KALYPSSO_CORE_MIRRORBITMASKARRAY_H_

#include <Kokkos_Macros.hpp>
#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>
#include <kalypsso/core/utils_block.h> // for definition of type block_size_t and shift_t
#include <kalypsso/core/MeshMap.h> // for orchard_key_view_t, amr_map_t and access to mirror keys
#include <kalypsso/core/MirrorQuadrantUtilsMetadata.h>

#include <../better-enums/enum.h>

// clang-format off
/**
 * An enum type to represent all possible ghost shapes types for creating a MirrorBitMask.
 *
 * When type is
 * - CELL_CENTER : all cell in the mirror block is individually evaluated to know if is it participating the MPI communication
 * - FLUX_X : only one face participate to communication (either left or right along X axis depending on the position of the MPI neighbor)
 */
BETTER_ENUM(MirrorBitMaskArrayType, uint8_t,
            CELL_CENTER = 0,
            FLUX_X = 1,
            FLUX_Y = 2,
            FLUX_Z = 3)
// clang-format on

namespace kalypsso
{

namespace core
{

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Configuration of a mirror bit mask type.
 */
template <size_t dim>
struct MirrorBitMaskConfig
{
  MirrorBitMaskConfig(size_t                    mirror_bit_mask_id_,
                      block_size_t<dim> const & block_sizes_,
                      shift_t<dim> const &      shift_,
                      block_size_t<dim> const & ghost_sizes_,
                      MirrorBitMaskArrayType    mirror_bit_mask_type_)
    : mirror_bit_mask_id(mirror_bit_mask_id_)
    , block_sizes(block_sizes_)
    , shift(shift_)
    , ghost_sizes(ghost_sizes_)
    , mirror_bit_mask_type(mirror_bit_mask_type_)
  {}

  //! bit mask id (key to retrieve a given mask among others)
  size_t mirror_bit_mask_id;

  //! block sizes (same as in DataArrayGhostedBlock)
  block_size_t<dim> block_sizes;

  //! ghost cell pattern shift (same as in DataArrayGhostedBlock)
  shift_t<dim> shift;

  //! ghost cell pattern sizes (same as in DataArrayGhostedBlock)
  block_size_t<dim> ghost_sizes;

  //! type of bit mask array
  MirrorBitMaskArrayType mirror_bit_mask_type;

  //! number of elements (equal to the number of cells when dealing with cell-centered data) per
  //! mirror octant
  KOKKOS_INLINE_FUNCTION
  size_t
  num_elements() const
  {
    return static_cast<size_t>(Kokkos::dim_prod(block_sizes));
  }

  //! number of bytes per mirror octant
  KOKKOS_INLINE_FUNCTION
  size_t
  num_bytes_per_mirror() const
  {
    return ((num_elements() + 7) / 8) * 8;
  }

  //! bit mask size in number of elements
  KOKKOS_INLINE_FUNCTION
  size_t
  bit_mask_size(size_t num_mirror_quadrants) const
  {
    return num_elements() * num_mirror_quadrants;
  }

  //! bit mask size in bytes (for memory allocation)
  KOKKOS_INLINE_FUNCTION
  size_t
  bit_mask_size_in_bytes(size_t num_mirror_quadrants) const
  {
    return num_bytes_per_mirror() * num_mirror_quadrants;
  }

}; // struct MirrorBitMaskConfig

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * \class MirrorBitMaskArray
 *
 * \brief Compute and store an array of mirror bit mask.
 *
 * Here the purpose is to ease filling ghost cells in a DataArrayGhostedBlock. Filling ghost cell
 * data may require accessing data from a neighbor octant, that may actually be a MPI ghost octant.
 * In the legacy version of MeshGhostExchanger, the entire mirror block data are transfered by MPI
 * communications. In this class we compute precisely for each mirror octant the exact data subset
 * that need to be transfered for filling a DataArrayGhostedBlock with a given ghost shape.
 *
 * A mirror bit mask is itself an array containing one bit per cell per mirror octant
 * indicating if the cell data should be included in the MPI message sent to the neighbor MPI
 * process, which rank is associated to the given mirror octant (as defined in p4est). The precise
 * mirror bit mask is determined by a ghost stencil pattern (associated to a given
 * DataArrayGhostedBlock shape)
 *
 * A ghost stencil pattern is a structure containing:
 * - a block size array corresponding to the number of cells per block / octant for each direction
 * - a ghosted shape size as specified in DataArrayGhostedBlock (ghosted size and a shift)
 *
 * A numerical scheme may required different ghost stencil pattern (and so different MPI
 * communication buffer sizes).
 *
 * For example, in the MUSCL-Hancock finite volume scheme, at the begining of a time step, we will
 * use 2 ghost cells all arround the block of cells, so that the second order scheme can applied in
 * all owned block; at the end of the time step, right before the update, Riemann fluxes at
 * non-conforming interface must be exchanged by MPI, but here we can use a stencil containing ghost
 * only along X-faces when update with X-fluxes, and the same for all other direction (Y and Z). The
 * amount of data sent is very different, we aim at precisely determining which data should
 * participate in the MPI data exchange.
 *
 * That is why current class implement an array of mirror bit mask, one bit mask per type of MPI
 * communication.
 *
 */
template <size_t dim, typename device_t>
class MirrorBitMaskArray
{
private:
  //! raw storage for all mirror bit mask
  DataArray<uint8_t, device_t> m_storage;

  //! number of bit mask
  size_t m_num_mirror_bit_mask;

  //! array of MirrorBitMaskConfig (should be rare to have more than 8 configs)
  DataArray<MirrorBitMaskConfig<dim>, device_t> m_mirror_bit_mask_configs;

public:
  //! constructor
  MirrorBitMaskArray()
    : m_storage("MirrorBitMaskArray_storage", 0)
    , m_num_mirror_bit_mask(0)
    , m_mirror_bit_mask_configs("MirrorBitMaskConfigs", 8) {};

  //! destructor
  ~MirrorBitMaskArray() = default;

  /**
   * Register a cell-center mirror bit mask associated to a given ghost pattern.
   *
   * \param[in] block size
   * \param[in] shift ghost pattern origin (lower left corner in cell coordinates)
   * \param[in] ghost_sizes ghost pattern sizes
   */
  size_t
  register_mirror_bit_mask(block_size_t<dim> const & block_sizes,
                           shift_t<dim> const &      shift,
                           block_size_t<dim> const & ghost_sizes,
                           MirrorBitMaskArrayType    mirror_bit_mask_type);

  /**
   * Register a mirror bit mask associated to a type of mask (e.g. FLUX_X).
   *
   * \param[in] block size
   * \param[in] mirror_bit_mask_type
   *
   * \todo resize config, resize storage
   */
  size_t
  register_mirror_bit_mask(block_size_t<dim> const &      block_sizes,
                           MirrorBitMaskArrayType const & mirror_bit_mask_type);

  /**
   * Recompute a mirror bit mask (after AMR cycle, the list of mirror octant changed)
   */
  void
  update_mask(MeshMap<dim, device_t> &                           mesh_map,
              MirrorQuadrantUtilsMetadata<dim, device_t> const & mirror_quad_metadata,
              size_t                                             mirror_bit_mask_id);

  /**
   * Recompute all mirror bit mask (after AMR cycle, the list of mirror octant changed)
   */
  void
  update_all_masks(MeshMap<dim, device_t> &                           mesh_map,
                   MirrorQuadrantUtilsMetadata<dim, device_t> const & mirror_quad_metadata);

  /**
   * Return a valid mirror bit mask array
   */
  auto
  get_mirror_bit_mask(size_t mirror_bit_mask_id)
  {
    KOKKOS_ASSERT((mirror_bit_mask_id < m_num_mirror_bit_mask) &&
                  "Invalid value for mirror_bit_mask_id");
  }

}; // class MirrorBitMaskArray

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_MIRRORBITMASKARRAY_H_
