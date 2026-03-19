// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file AMRMeshInfo.h
 *
 */
#ifndef KALYPSSO_CORE_AMRMESHINFO_H_
#define KALYPSSO_CORE_AMRMESHINFO_H_

#include <cstdint>
#include <iostream>

namespace kalypsso
{

/**
 * A minimalist struct containing information about a AMR mesh:
 *
 * - number of owned quadrants inside domain (local to current MPI process)
 * - sum of the number of owned quadrants (global, i.e. sum over all MPI process)
 * - number of MPI ghost quadrants local to current MPI process
 * - number of quadrants outside domain (local to current MPI process); may be zero
 * - number of quadrants outside domain (global, i.e. sum over all MPI process); may be zero
 *
 */
class AMRMeshInfo
{

private:
  //! local (current MPI proc) number of owned quads/octs
  int32_t m_local_num_quadrants;

  //! global accumulated (all MPI proc) number of owned quads/octs
  int64_t m_global_num_quadrants;

  //! global index to the first quadrant in current MPI process (along the Morton curve)
  int64_t m_global_first_quadrant;

  //! local number of MPI ghost quadrants
  int32_t m_local_num_ghosts;

  //! local number of MPI owned quadrants that are mirrors (to be send to fill ghost quadrant in
  //! some other MPI process)
  int32_t m_local_num_mirrors;

  //! local (current MPI proc) number of quads/octs outside domain
  int32_t m_local_num_quadrants_outside;

  //! global accumulated (all MPI proc) number of quads/octs outside domain
  int64_t m_global_num_quadrants_outside;

  //! global index to the first outside quadrant in current MPI process
  int64_t m_global_first_quadrant_outside;

  //! local (current MPI proc) number of quads/octs that are outside mirror of a ghost quadrant
  int32_t m_local_num_quadrants_outside_ghost;

  //! mpi rank (so that we can use in inside kokkos parallel region)
  int32_t m_mpi_rank;

public:
  //! constructor
  KOKKOS_INLINE_FUNCTION
  AMRMeshInfo()
    : m_local_num_quadrants(0)
    , m_global_num_quadrants(0)
    , m_global_first_quadrant(0)
    , m_local_num_ghosts(0)
    , m_local_num_mirrors(0)
    , m_local_num_quadrants_outside(0)
    , m_global_num_quadrants_outside(0)
    , m_global_first_quadrant_outside(0)
    , m_local_num_quadrants_outside_ghost(0)
    , m_mpi_rank(-1)
  {}

  KOKKOS_DEFAULTED_FUNCTION
  ~AMRMeshInfo() = default;

  KOKKOS_DEFAULTED_FUNCTION
  AMRMeshInfo(AMRMeshInfo const & amr_mesh_info) = default;

  // ================================================================
  KOKKOS_INLINE_FUNCTION
  auto
  local_num_quadrants() const
  {
    return m_local_num_quadrants;
  }

  KOKKOS_INLINE_FUNCTION
  auto &
  local_num_quadrants()
  {
    return m_local_num_quadrants;
  }

  // ================================================================
  KOKKOS_INLINE_FUNCTION
  auto
  global_num_quadrants() const
  {
    return m_global_num_quadrants;
  }

  KOKKOS_INLINE_FUNCTION
  auto &
  global_num_quadrants()
  {
    return m_global_num_quadrants;
  }

  // ================================================================
  KOKKOS_INLINE_FUNCTION
  auto
  global_first_quadrant() const
  {
    return m_global_first_quadrant;
  }

  KOKKOS_INLINE_FUNCTION
  auto &
  global_first_quadrant()
  {
    return m_global_first_quadrant;
  }

  // ================================================================
  KOKKOS_INLINE_FUNCTION
  auto
  local_num_ghosts() const
  {
    return m_local_num_ghosts;
  }

  KOKKOS_INLINE_FUNCTION
  auto &
  local_num_ghosts()
  {
    return m_local_num_ghosts;
  }

  // ================================================================
  KOKKOS_INLINE_FUNCTION
  auto
  local_num_mirrors() const
  {
    return m_local_num_mirrors;
  }

  KOKKOS_INLINE_FUNCTION
  auto &
  local_num_mirrors()
  {
    return m_local_num_mirrors;
  }

  // ================================================================
  KOKKOS_INLINE_FUNCTION
  auto
  local_num_quadrants_outside() const
  {
    return m_local_num_quadrants_outside;
  }

  KOKKOS_INLINE_FUNCTION
  auto &
  local_num_quadrants_outside()
  {
    return m_local_num_quadrants_outside;
  }

  // ================================================================
  KOKKOS_INLINE_FUNCTION
  auto
  global_num_quadrants_outside() const
  {
    return m_global_num_quadrants_outside;
  }

  KOKKOS_INLINE_FUNCTION
  auto &
  global_num_quadrants_outside()
  {
    return m_global_num_quadrants_outside;
  }

  // ================================================================
  KOKKOS_INLINE_FUNCTION
  auto
  global_first_quadrant_outside() const
  {
    return m_global_first_quadrant_outside;
  }

  KOKKOS_INLINE_FUNCTION
  auto &
  global_first_quadrant_outside()
  {
    return m_global_first_quadrant_outside;
  }

  // ================================================================
  KOKKOS_INLINE_FUNCTION
  auto
  local_num_quadrants_outside_ghost() const
  {
    return m_local_num_quadrants_outside_ghost;
  }

  KOKKOS_INLINE_FUNCTION
  auto &
  local_num_quadrants_outside_ghost()
  {
    return m_local_num_quadrants_outside_ghost;
  }

  // ================================================================
  KOKKOS_INLINE_FUNCTION
  auto
  mpi_rank() const
  {
    return m_mpi_rank;
  }

  KOKKOS_INLINE_FUNCTION
  auto &
  mpi_rank()
  {
    return m_mpi_rank;
  }

  //! return the total number of quadrants in local MPI process.
  //! Total number of quadrants means sum of number of owned, ghost, outside and outside_ghosts
  //! quadrants.
  KOKKOS_INLINE_FUNCTION
  int32_t
  local_num_quadrants_total() const
  {
    // clang-format off
    return
      m_local_num_quadrants +
      m_local_num_ghosts +
      m_local_num_quadrants_outside +
      m_local_num_quadrants_outside_ghost;
    // clang-format on
  } // local_num_quadrants_total

  //! return the local index (local = inside current MPI process) to the first outside quadrants
  KOKKOS_INLINE_FUNCTION
  int32_t
  first_outside_quad_local_id() const
  {
    return m_local_num_quadrants + m_local_num_ghosts;
  }

  //! return the total number of outside quadrants (outside and ghost_outside) in local MPI process
  KOKKOS_INLINE_FUNCTION
  int32_t
  total_local_number_of_outside_quads() const
  {
    return m_local_num_quadrants_outside + m_local_num_quadrants_outside_ghost;
  }

  //! return the total number of quadrant other than owned quadrants; i.e. ghost + outside +
  //! outside_ghost
  KOKKOS_INLINE_FUNCTION
  int32_t
  local_num_quadrants_other()
  {
    // clang-format off
    return
      m_local_num_ghosts +
      m_local_num_quadrants_outside +
      m_local_num_quadrants_outside_ghost;
    // clang-format on
  }


  //! print (host only)
  void
  print() const
  {
    // clang-format off
    std::cout << "====================================================\n";
    std::cout << "AMRMeshInfo: \n";
    std::cout << "local_num_quadrants: " << m_local_num_quadrants << "\n";
    std::cout << "global_num_quadrants: " << m_global_num_quadrants << "\n";
    std::cout << "global_first_quadrant: " << m_global_first_quadrant << "\n";
    std::cout << "local_num_ghosts: " << m_local_num_ghosts << "\n";
    std::cout << "local_num_quadrants_outside: " << m_local_num_quadrants_outside << "\n";
    std::cout << "global_num_quadrants_outside: " << m_global_num_quadrants_outside << "\n";
    std::cout << "global_first_quadrant_outside: " << m_global_first_quadrant_outside << "\n";
    std::cout << "local_num_quadrants_outside_ghost: " << m_local_num_quadrants_outside_ghost << "\n";
    std::cout << "local_num_quadrants_total: " << local_num_quadrants_total() << "\n";
    std::cout << "first_outside_quad_local_id: " << first_outside_quad_local_id() << "\n";
    std::cout << "====================================================\n";
    // clang-format on
  } // print

}; // struct AMRMeshInfo

} // namespace kalypsso

#endif // KALYPSSO_CORE_AMRMESHINFO_H_
