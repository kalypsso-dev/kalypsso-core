// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeMirrorGeometricalPattern.h
 *
 * Just compute L1 or L2 norm of the difference between two data array.
 */
#ifndef KALYPSSO_CORE_COMPUTEMIRRORGEOMETRICALPATTERN_H_
#define KALYPSSO_CORE_COMPUTEMIRRORGEOMETRICALPATTERN_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/BitFieldInteger.h>
#include <kalypsso/core/MeshMap.h>   // for orchard_key_view_t and amr_map_t
#include <kalypsso/core/real_type.h> // for math functions (max, min, ...)
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <../better-enums/enum.h>

namespace kalypsso
{

namespace core
{

// clang-format off
BETTER_ENUM(MirrorGeometricalPatternType, int, ALL, FACE_X, FACE_Y, FACE_Z)
// clang-format on

/**
 * A simple bit field data (stored as 32 bit integer) used to store
 * all geometrical elements (face, edge or corner) characterizing a p4est mirror
 * in an MPI communication.
 *
 * E.g. when using a finite volume MUSCL-Hancock scheme, all blocks are surrounded by a uniform
 * ghost layer. For each mirror, we encode which face, edge corner data need to be transferred.
 * In a dimensionnaly split Lagrange-Remap scheme, we only need to transfer face data, one direction
 * at a time. So this explains the different pattern types.
 *
 * This bit field integer will then be used by the mesh ghost exchanger to compute the amount of
 * data transferred by each mirror, and by cumulating over all mirrors, determine the memory
 * allocation size needed for the MPI send and receive buffers.
 */
template <size_t dim>
struct mirror_geometrical_pattern_t;


// ====================================================================
// ====================================================================
template <>
struct mirror_geometrical_pattern_t<2> : public BitFieldInteger<uint16_t>
{
  // 4 faces + 4 corners => 8 bits
  using status_t = uint8_t;
};

// ====================================================================
// ====================================================================
template <>
struct mirror_geometrical_pattern_t<3> : public BitFieldInteger<uint16_t>
{
  // 6 faces + 12 edges + 8 corners = 26 bits
  using status_t = uint32_t;
};

// ====================================================================
// ====================================================================

// ====================================================================
// ====================================================================
/**
 * \class ComputeMirrorGeometricalPattern
 */
template <size_t dim, typename device_t>
class ComputeMirrorGeometricalPattern
{
public:
  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  //! type alias for a data array at block level (see kalypsso_data_container.h)
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  using index_t = int64_t;

private:
  //! heavy data
  DataArrayBlock_t m_mirror_pattern;

public:
  // ====================================================================
  // ====================================================================
  //! constructor.
  //!
  //! \param[in] varId identify which variable to reduce (ID, IE, IU, ...)
  ComputeMirrorGeometricalPattern(MeshMap<dim, device_t> & mesh_map)
    : m_mesh_map(mesh_map)
  {}

  // ====================================================================
  // ====================================================================
  //! destructor.
  ~ComputeMirrorGeometricalPattern() = default;

  // ====================================================================
  // ====================================================================
  //! perform reduction.
  //!
  //! \return return relative error: |U-Uref| / |Uref|
  //! norm used can be either L1 or L2
  static auto
  update_pattern(MeshMap<dim, device_t> & mesh_map, MirrorGeometricalPatternType pattern_type)
  {
    ComputeMirrorGeometricalPattern<dim, device_t> functor(mesh_map);

    // Kokkos::parallel_reduce("ComputeMirrorGeometricalPattern",
    //                         Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
    //                         functor,
    //                         reducer);

  } // update_pattern

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const index_t & global_index) const
  {} // operator ()

}; // class ComputeMirrorGeometricalPattern

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_COMPUTEMIRRORGEOMETRICALPATTERN_H_
