// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file NeighborLocalOffsets.h
 *
 * \see struct NeighborLocalOffsets
 */
#ifndef KALYPSSO_CORE_NEIGHBORLOCALOFFSETS_H_
#define KALYPSSO_CORE_NEIGHBORLOCALOFFSETS_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>

namespace kalypsso
{

namespace core
{

// ====================================================================================
// ====================================================================================
// ====================================================================================
/**
 * \struct NeighborLocalOffsets
 *
 * This structure is just a helper to manipulate local offsets in the array of neighbors.
 * The array of neighbors stores id's of all neighbors to eahc quadrant, ordered by sectors.
 *
 * A sector identifies one of the neighbor region associated to a direction (see drawing below).
 *
 * Sectors are enumerated in raw-major order from lower left with dir = (-1,-1) to upper right
 *  dir=(1,1)
 *
 *               |                     |
 *  dir=(-1,1)   |      dir=(0,1)      |  dir=(1,1)
 *          _____|_____________________|_______
 *               |                     |
 *               |                     |
 *               |                     |
 *               |   Current quadrant  |
 *  dir=(-1,0)   |      dir=(0,0)      |  dir=(1,0)
 *               |                     |
 *               |                     |
 *               |                     |
 *               |                     |
 *          _____|_____________________|_______
 *               |                     |
 *  dir=(-1,-1)  |     dir=(0,-1)      | dir=(1,-1)
 *               |                     |
 *
 * \note about storage
 * - in 2d, there are at most 12 different neighbors so a local offset (one per sector) can be
 *   stored using 4 bits, as we use 3x3+1=10 offsets, in total 40 bits are necessary (one 64 bit
 integer)
 * - in 3d, there are at most 56 different neighbors so a local offset (one per sector) can be
 *   stored using 6 bits, as we use 3x3x3+1=28 offsets, in total 168 bits are
 necessary (three 64 bit integers)
 *
 */
template <size_t dim>
struct NeighborLocalOffsets
{
  static constexpr size_t LOCAL_OFFSETS_NUM = (dim == 2) ? 1 : 3;

  static constexpr uint64_t LOCAL_OFFSET_MASK_2D = 0xFULL;  // last 4 bits
  static constexpr uint64_t LOCAL_OFFSET_MASK_3D = 0x3FULL; // last 6 bits

  static constexpr size_t NUM_BITS_PER_LOCAL_OFFSET_2D = 4;
  static constexpr size_t NUM_BITS_PER_LOCAL_OFFSET_3D = 6;

  static constexpr size_t NUM_SECTORS_2D = 10; // 3*3+1
  static constexpr size_t NUM_SECTORS_3D = 28; // 3*3*3+1

  static constexpr size_t LOCAL_OFFSET_MAX_2D = 12; // 4**2 - 2**2
  static constexpr size_t LOCAL_OFFSET_MAX_3D = 56; // 4**3 - 2**3

  //! in 3d local_offset contains 3 64-bits integers
  //! each integer contains 9 local offsets, except the last one
  //! which contains 10 offsets
  static constexpr uint8_t NUM_SECTOR_OFFSETS_PER_INT = 9;

  using local_offsets_t = Kokkos::Array<uint64_t, LOCAL_OFFSETS_NUM>;

  /**
   * Encode local offset neighbors to given sector.
   *
   * \param[in,out] local_offset
   * \param[in] sector from 0 to 3**dim
   * \param[in] offset number of neighbors in all preceding sectors
   */
  static KOKKOS_INLINE_FUNCTION void
  encode_local_offset(local_offsets_t & local_offsets_, uint8_t sector, uint8_t offset)
  {
    if constexpr (dim == 2)
    {
      KOKKOS_ASSERT((sector < NUM_SECTORS_2D) && "Wrong value for sector");
      KOKKOS_ASSERT((offset <= LOCAL_OFFSET_MAX_2D) && "Wrong value for offset");
      const uint64_t mask = LOCAL_OFFSET_MASK_2D << (sector * NUM_BITS_PER_LOCAL_OFFSET_2D);
      local_offsets_[0] &= ~mask;
      local_offsets_[0] |= (static_cast<uint64_t>(offset) & LOCAL_OFFSET_MASK_2D)
                           << (sector * NUM_BITS_PER_LOCAL_OFFSET_2D);
    }
    else if constexpr (dim == 3)
    {
      KOKKOS_ASSERT((sector < NUM_SECTORS_3D) && "Wrong value for sector");
      KOKKOS_ASSERT((offset <= LOCAL_OFFSET_MAX_3D) && "Wrong value for offset");
      const auto id = (sector == (NUM_SECTORS_3D - 1)) ? 2 : sector / NUM_SECTOR_OFFSETS_PER_INT;
      const auto sector2 = sector - NUM_SECTOR_OFFSETS_PER_INT * id;
      const uint64_t mask = LOCAL_OFFSET_MASK_3D << (sector2 * NUM_BITS_PER_LOCAL_OFFSET_3D);
      local_offsets_[id] &= ~mask;
      local_offsets_[id] |= (static_cast<uint64_t>(offset) & LOCAL_OFFSET_MASK_3D)
                            << (sector2 * NUM_BITS_PER_LOCAL_OFFSET_3D);
    }
  }

  /**
   * Decode local offset neighbors to given sector.
   *
   * \param[in] local_offset
   * \param[in] sector from 0 to 3**dim
   *
   */
  static KOKKOS_INLINE_FUNCTION uint8_t
  decode_local_offset(local_offsets_t const & local_offsets, uint8_t sector)
  {
    if constexpr (dim == 2)
    {
      KOKKOS_ASSERT((sector < NUM_SECTORS_2D) && "Wrong value for sector");
      return static_cast<uint8_t>((local_offsets[0] >> (sector * NUM_BITS_PER_LOCAL_OFFSET_2D)) &
                                  LOCAL_OFFSET_MASK_2D);
    }
    else if constexpr (dim == 3)
    {
      KOKKOS_ASSERT((sector < NUM_SECTORS_3D) && "Wrong value for sector");

      // last extra "sector" is processed differently
      const auto id = (sector == (NUM_SECTORS_3D - 1)) ? 2 : sector / NUM_SECTOR_OFFSETS_PER_INT;
      const auto sector2 = sector - NUM_SECTOR_OFFSETS_PER_INT * id;
      return static_cast<uint8_t>((local_offsets[id] >> (sector2 * NUM_BITS_PER_LOCAL_OFFSET_3D)) &
                                  LOCAL_OFFSET_MASK_3D);
    }
  }

}; // struct NeighborLocalOffsets

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_NEIGHBORLOCALOFFSETS_H_
