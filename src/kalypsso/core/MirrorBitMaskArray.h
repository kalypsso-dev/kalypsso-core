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
                      MirrorBitMaskArrayType    type_)
    : mirror_bit_mask_id(mirror_bit_mask_id_)
    , block_sizes(block_sizes_)
    , shift(shift_)
    , ghost_sizes(ghost_sizes_)
    , type(type_)
    , offset(0)
    , size(0)
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
  MirrorBitMaskArrayType type;

  //! offset within storage at which the bit mask array starts
  size_t offset;

  //! size (in bytes)
  size_t size;

  //! number of elements (equal to the number of cells when dealing with cell-centered data) per
  //! mirror octant
  KOKKOS_INLINE_FUNCTION
  size_t
  num_elements_per_mirror() const
  {
    return static_cast<size_t>(Kokkos::dim_prod(block_sizes));
  }

  //! number of bytes per mirror octant
  KOKKOS_INLINE_FUNCTION
  size_t
  num_bytes_per_mirror() const
  {
    return ((num_elements_per_mirror() + 7) / 8);
  }

  //! bit mask size in number of elements
  KOKKOS_INLINE_FUNCTION
  size_t
  bit_mask_size(size_t num_mirror_quadrants) const
  {
    return num_elements_per_mirror() * num_mirror_quadrants;
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
public:
  using exec_space = typename device_t::execution_space;

  using storage_t = uint8_t;

  using bit_mask_view_t =
    Kokkos::View<storage_t *, device_t, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

private:
  //! raw storage for all mirror bit mask
  DataArray<storage_t, device_t> m_storage;

  //! number of bit mask
  size_t m_num_mirror_bit_mask;

  //! array of MirrorBitMaskConfig (should be rare to have more than 8 configs)
  DataArray<MirrorBitMaskConfig<dim>, device_t> m_mirror_bit_mask_configs;

  //! host mirror of the bit mask configs
  DataArray<MirrorBitMaskConfig<dim>, kalypsso::HostDevice> m_mirror_bit_mask_configs_host;

  /**
   * Reset configuration and memory.
   *
   * Here we do multiple things:
   * - recompute mask's offset and mask's size for all masks, update mask configuration on host
   * - copy mask configurations on device
   * - resize storage
   */
  void
  reset(size_t num_mirrors);

  //! This struct is only defined as way to protect some public API to be called inadvertedly
  struct Lock
  {
    explicit Lock() = default;
  };

public:
  //! constructor
  MirrorBitMaskArray()
    : m_storage("MirrorBitMaskArray_storage", 0)
    , m_num_mirror_bit_mask(0)
    , m_mirror_bit_mask_configs("MirrorBitMaskConfigs", 8)
    , m_mirror_bit_mask_configs_host("MirrorBitMaskConfigs_host", 8) {};

  //! destructor
  ~MirrorBitMaskArray() = default;

  /**
   * Register a cell-center mirror bit mask associated to a given ghost pattern.
   *
   * Here we all register a bit mask array (i.e. create a new entry in array of configurations).
   * Memory allocation for storage is done in reset.
   *
   * \param[in] block size
   * \param[in] shift ghost pattern origin (lower left corner in cell coordinates)
   * \param[in] ghost_sizes ghost pattern sizes
   * \param[in] mirror_bit_mask_type
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
   */
  size_t
  register_mirror_bit_mask(block_size_t<dim> const &      block_sizes,
                           MirrorBitMaskArrayType const & mirror_bit_mask_type);

  /**
   * Recompute a mirror bit mask (after AMR cycle, the list of mirror octant changed)
   *
   * This member is intended to be called inside update_all_masks, which is responsible for resizing
   * storage. It should be a private member of this class, but nvcc compiler need to ha
   *
   * \param[in] mesh_map to get access to orchard keys (as kokkos view) and amr_mesh_info
   * \param[in] mirror_quad_metadata mirror quad neighborhood connectivity information
   * \param[in] mirror_bit_mask_id identifies the bit mask array to update
   */
  void
  update_mask(Lock,
              MeshMap<dim, device_t> &                           mesh_map,
              MirrorQuadrantUtilsMetadata<dim, device_t> const & mirror_quad_metadata,
              size_t                                             mirror_bit_mask_id);

  /**
   * Recompute all mirror bit mask (after AMR cycle, the list of mirror octant changed)
   *
   * \param[in] mesh_map to get access to orchard keys (as kokkos view) and amr_mesh_info
   * \param[in] mirror_quad_metadata mirror quad neighborhood connectivity information
   *
   */
  void
  update_all_masks(MeshMap<dim, device_t> &                           mesh_map,
                   MirrorQuadrantUtilsMetadata<dim, device_t> const & mirror_quad_metadata);

  /**
   * Return a valid mirror bit mask array (unmanaged kokkos view on device)
   *
   * \param[in] mirror_bit_mask_id identifies the bit mask array to update
   */
  auto
  get_mirror_bit_mask(size_t mirror_bit_mask_id) -> bit_mask_view_t
  {
    KOKKOS_ASSERT((mirror_bit_mask_id < m_num_mirror_bit_mask) &&
                  "Invalid value for mirror_bit_mask_id");

    return bit_mask_view_t(m_storage.data() +
                             m_mirror_bit_mask_configs_host(mirror_bit_mask_id).offset,
                           m_mirror_bit_mask_configs_host(mirror_bit_mask_id).size);
  }

}; // class MirrorBitMaskArray

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_MIRRORBITMASKARRAY_H_
