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

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>
#include <kalypsso/core/utils_block.h> // for definition of type block_size_t and shift_t
#include <kalypsso/core/MeshMap.h> // for orchard_key_view_t, amr_map_t and access to mirror keys

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
struct MirrorBitMaskConfig
{
  size_t                 mirror_bit_mask_id;
  shift_t<dim>           shift;
  block_size_t<dim>      ghost_sizes;
  MirrorBitMaskArrayType mirror_bit_mask_type;
};

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

  //! block size (number of cells per direction)
  block_size_t<dim> m_block_sizes;

  //! array of MirrorBitMaskConfig
  DataArray<MirrorBitMaskConfig *, device_t> m_mirror_bit_mask_configs;

public:
  MirrorBitMaskArray()
    : m_storage("MirrorBitMaskArray_storage", 0)
    , m_num_mirror_bit_mask(0) {};
  ~MirrorBitMaskArray() = default;

  /**
   * Register a cell-center mirror bit mask associated to a given ghost pattern.
   *
   * \param[in] mirror_bit_mask_id
   * \param[in] shift ghost pattern origin (lower left corner in cell coordinates)
   * \param[in] ghost_sizes ghost pattern sizes
   */
  void
  register_mirror_bit_mask(size_t                    mirror_bit_mask_id,
                           shift_t<dim> const &      shift,
                           block_size_t<dim> const & ghost_sizes);

  /**
   * Register a mirror bit mask associated to a type of mask (e.g. FLUX_X).
   *
   * \param[in] mirror_bit_mask_id
   * \param[in] mirror_bit_mask_type
   *
   * \todo resize config, resize storage
   */
  void
  register_mirror_bit_mask(size_t mirror_bit_mask_id, MirrorBitMaskArrayType mirror_bit_mask_type);

  /**
   * Recompute a mirror bit mask (after AMR cycle, the list of mirror octant changed)
   */
  void
  update_all_masks(MeshMap<dim, device_t> &             mesh_map,
                   MirrorQuadrantsUtilsMetadata const & mirror_quad_metadata);

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
