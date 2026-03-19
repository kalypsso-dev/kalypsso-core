// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file OutsideQuadsInfo.h
 * \brief A small utility class holding geometric information about quadrants that are outside
 *  domain.
 *
 * Just remember that if p4est brick connectivity is not periodic, we provide quads that are outside
 * domain to ease border condition implementation. We should also be able to dump these specific
 * quadrants in the hdf5 output files. Here we pre-compute MPI decomposition of these outside
 * quadrants.
 */
#ifndef KALYPSSO_CORE_OUTSIDEQUADSINFO_H_
#define KALYPSSO_CORE_OUTSIDEQUADSINFO_H_

#include <kalypsso/core/kalypsso_core_base.h> // which include kalypsso_core_config.h

#include <kalypsso/utils/mpi/ParallelEnv.h>
#include <kalypsso/utils/p4est/p4est_wrapper.h>
#include <kalypsso/utils/p4est/connectivity.h> // for CONNECTIVITY_PERIODIC_FALSE
#include <kalypsso/core/brick_connectivity_utils.h>
#include <kalypsso/core/mesh_utils.h>
#include <kalypsso/core/brick_utils.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/orchard_key_utils.h>

#include <iostream>

namespace kalypsso
{

// ==========================================================================================
// ==========================================================================================
//! \brief recompute the local number of outside quads.
//!
//! Let us remind ourselves that out side quads are additional quadrants outside physical
//! domain used to implement border condition other than periodic (which is taken into account
//! natively by p4est).
//! We need to add an outside quadrant for all border than is not periodic.
//!
//!
//! \note this free standing function is needed here, as well as in MeshMap.
template <size_t dim>
int32_t
compute_number_outside_quads(/*const*/ forest_t<dim> *            forest,
                             [[maybe_unused]] const ParallelEnv & par_env,
                             Kokkos::Array<bool, dim>             is_brick_periodic,
                             brick_size_t<dim>                    brick_sizes)
{

  // type alias to access p4est C API (2D or 3D)
  using p4est_t = typename p4est::Wrapper<dim>;

  // do we have at least one non-periodic border ?
  bool has_non_periodic_border = false;
  for (size_t idim = 0; idim < dim; ++idim)
  {
    if (!is_brick_periodic[idim])
      has_non_periodic_border = true;
  }

  // if all border are periodic, ok no outside quads
  if (!has_non_periodic_border)
    return 0;

  // tree linear index to xyz converter
  const BrickConnectivityData<dim> convert(brick_sizes);

  // init returned value
  int32_t number_outside_quads = 0;

  // get list of faces
  const auto faces = Face::get_all_faces<dim>();

  // loop over all owned (inside) quadrants, and compute how many are touching external border
  // through a non-periodic border
  for (auto treeid = forest->first_local_tree; treeid <= forest->last_local_tree; ++treeid)
  {
    // get current tree
    auto tree = p4est_t::tree_array_index(forest->trees, treeid);

    // get tree coordinate
    auto tree_xyz = convert.toXYZ(treeid);

    for (size_t qId = 0; qId < tree->quadrants.elem_count; ++qId)
    {
      auto q = p4est_t::quadrant_array_index(&(tree->quadrants), qId);

      Kokkos::Array<uint32_t, dim> octCoord;
      octCoord[0] =
        static_cast<uint32_t>(q->x >> (p4est_t::QMAXLEVEL - orchard_key_t<dim>::NUM_LEVELS + 2));
      octCoord[1] =
        static_cast<uint32_t>(q->y >> (p4est_t::QMAXLEVEL - orchard_key_t<dim>::NUM_LEVELS + 2));
      if constexpr (dim == 3)
      {
        octCoord[2] =
          static_cast<uint32_t>(q->z >> (p4est_t::QMAXLEVEL - orchard_key_t<dim>::NUM_LEVELS + 2));
      }

      auto key =
        orchard_key_t<dim>::encode_orchard(tree_xyz, octCoord, static_cast<uint8_t>(q->level));

      // look for outside quad across a face
      for (const auto face : faces)
      {
        if (orchard_key_t<dim>::is_at_domain_border(key, face, brick_sizes))
        {
          if (!is_brick_periodic[face / 2])
            number_outside_quads++;
        }
      } // end for faces

      // look for outside quad across an edge
      if constexpr (dim == 3)
      {
        Face::face_t face0, face1;
        for (uint8_t iEdge = 0; iEdge < Edge::num_edges<dim>(); ++iEdge)
        {
          edge_to_faces(iEdge, face0, face1);
          if (orchard_key_t<dim>::is_at_domain_border(key, face0, brick_sizes) and
              orchard_key_t<dim>::is_at_domain_border(key, face1, brick_sizes))
          {
            if (!is_brick_periodic[face0 / 2] or !is_brick_periodic[face1 / 2])
              number_outside_quads++;
          }
        }
      } // end edge - 3d

      // look for outside quad across a corner
      for (uint8_t iCorner = 0; iCorner < Corner::num_corners<dim>(); ++iCorner)
      {
        if constexpr (dim == 2)
        {
          Face::face_t face0, face1;
          corner_to_faces(iCorner, face0, face1);
          if (orchard_key_t<dim>::is_at_domain_border(key, face0, brick_sizes) and
              orchard_key_t<dim>::is_at_domain_border(key, face1, brick_sizes))
          {
            if (!is_brick_periodic[face0 / 2] or !is_brick_periodic[face1 / 2])
              number_outside_quads++;
          }
        }
        else if constexpr (dim == 3)
        {
          Face::face_t face0, face1, face2;
          corner_to_faces(iCorner, face0, face1, face2);
          if (orchard_key_t<dim>::is_at_domain_border(key, face0, brick_sizes) and
              orchard_key_t<dim>::is_at_domain_border(key, face1, brick_sizes) and
              orchard_key_t<dim>::is_at_domain_border(key, face2, brick_sizes))
          {
            // clang-format off
            if (!is_brick_periodic[face0 / 2] or
                !is_brick_periodic[face1 / 2] or
                !is_brick_periodic[face2 / 2])
              number_outside_quads++;
            // clang-format on
          }
        }
      } // end for iCorner

    } // end for qId

  } // end for treeid

  return number_outside_quads;

} // compute_number_outside_quads

// ==========================================================================================
// ==========================================================================================
//! recompute the local number of outside quads of ghosts.
//! this free standing function is needed here, as well as in MeshMap.
template <size_t dim>
int32_t
compute_number_outside_ghosts(/*const*/ ghost_t<dim> *             ghost,
                              [[maybe_unused]] const ParallelEnv & par_env,
                              Kokkos::Array<bool, dim>             is_brick_periodic,
                              brick_size_t<dim>                    brick_sizes)
{
  // type alias to access p4est C API (2D or 3D)
  using p4est_t = typename p4est::Wrapper<dim>;

  if (ghost == nullptr)
    return 0;

  // do we have at least one non-periodic border ?
  bool has_non_periodic_border = false;
  for (size_t idim = 0; idim < dim; ++idim)
  {
    if (!is_brick_periodic[idim])
      has_non_periodic_border = true;
  }

  // if all border are periodic, ok no quad outside (owned or ghost anyway)
  if (!has_non_periodic_border)
    return 0;

  // tree linear index to xyz converter
  const BrickConnectivityData<dim> convert(brick_sizes);

  // init returned value
  int32_t number_outside_ghosts = 0;

  // get list of faces
  const auto faces = Face::get_all_faces<dim>();

  //
  // complete array of orchard keys using (MPI) ghost quadrants
  //
  for (size_t qId = 0; qId < ghost->ghosts.elem_count; ++qId)
  {
    auto q = p4est_t::quadrant_array_index(&(ghost->ghosts), qId);

    // get current ghost's tree id
    auto treeid = q->p.which_tree;

    // get tree coordinate
    auto tree_xyz = convert.toXYZ(treeid);

    Kokkos::Array<uint32_t, dim> octCoord;
    octCoord[0] =
      static_cast<uint32_t>(q->x >> (p4est_t::QMAXLEVEL - orchard_key_t<dim>::NUM_LEVELS + 2));
    octCoord[1] =
      static_cast<uint32_t>(q->y >> (p4est_t::QMAXLEVEL - orchard_key_t<dim>::NUM_LEVELS + 2));
    if constexpr (dim == 3)
    {
      octCoord[2] =
        static_cast<uint32_t>(q->z >> (p4est_t::QMAXLEVEL - orchard_key_t<dim>::NUM_LEVELS + 2));
    }

    auto key =
      orchard_key_t<dim>::encode_orchard(tree_xyz, octCoord, static_cast<uint8_t>(q->level));

    for (const auto face : faces)
    {
      if (orchard_key_t<dim>::is_at_domain_border(key, face, brick_sizes))
      {
        if (!is_brick_periodic[face / 2])
        {
          number_outside_ghosts++;
          // if (par_env.rank() == 1)
          // {
          //   const auto xyz = orchard_key_to_vertex_coord<dim>(key, false);

          //   printf("[rank %d] KK %d %lu | %f %f\n",
          //          par_env.rank(),
          //          number_outside_ghosts,
          //          key,
          //          xyz[0],
          //          xyz[1]);
          // }
        }
      }
    } // end for faces

    // look for outside quad across an edge
    if constexpr (dim == 3)
    {
      Face::face_t face0, face1;
      for (uint8_t iEdge = 0; iEdge < Edge::num_edges<dim>(); ++iEdge)
      {
        edge_to_faces(iEdge, face0, face1);
        if (orchard_key_t<dim>::is_at_domain_border(key, face0, brick_sizes) and
            orchard_key_t<dim>::is_at_domain_border(key, face1, brick_sizes))
        {
          if (!is_brick_periodic[face0 / 2] or !is_brick_periodic[face1 / 2])
          {
            number_outside_ghosts++;
          }
        }
      }
    } // end edge - 3d

    // look for outside quad across a corner
    for (uint8_t iCorner = 0; iCorner < Corner::num_corners<dim>(); ++iCorner)
    {
      if constexpr (dim == 2)
      {
        Face::face_t face0, face1;
        corner_to_faces(iCorner, face0, face1);
        if (orchard_key_t<dim>::is_at_domain_border(key, face0, brick_sizes) and
            orchard_key_t<dim>::is_at_domain_border(key, face1, brick_sizes))
        {
          if (!is_brick_periodic[face0 / 2] or !is_brick_periodic[face1 / 2])
          {
            number_outside_ghosts++;
          }
        }
      }
      else if constexpr (dim == 3)
      {
        Face::face_t face0, face1, face2;
        corner_to_faces(iCorner, face0, face1, face2);
        if (orchard_key_t<dim>::is_at_domain_border(key, face0, brick_sizes) and
            orchard_key_t<dim>::is_at_domain_border(key, face1, brick_sizes) and
            orchard_key_t<dim>::is_at_domain_border(key, face2, brick_sizes))
        {
          if (!is_brick_periodic[face0 / 2] or !is_brick_periodic[face1 / 2] or
              !is_brick_periodic[face2 / 2])
          {
            number_outside_ghosts++;
          }
        }
      }
    } // end for iCorner

  } // end for qId

  return number_outside_ghosts;

} // compute_number_outside_ghosts

// ==========================================================================================
// ==========================================================================================
// ==========================================================================================
/**
 * Simple class to pass the number of outside quads to HDF5_Writer.
 */
struct OutsideQuadsInfo
{

  //! number of outside quadrants for current MPI process
  int32_t local_num_outside_quads;

  //! total number of outside quadrants for all MPI processes
  int32_t global_num_outside_quads;

  //! global id of the first outside quadrant of current process.
  //! this is only useful when dumping outside quadrants in a file (parallel HDF5)
  int32_t first_outside_quad_global_id;

  //! local num of MPI ghost quadrants.
  //! this is useful because outside quadrants comes after MPI ghosts quadrant,
  //! so we need this number to compute the right offset to access data in UserData arrays.
  int32_t local_num_ghost_quads;

  //! local number of outside quadrants that are associated to a ghost quadrants
  int32_t local_num_outside_ghosts;

  // =========================================================================
  // =========================================================================
  //! constructor
  OutsideQuadsInfo()
    : local_num_outside_quads(0)
    , global_num_outside_quads(0)
    , first_outside_quad_global_id(0)
    , local_num_ghost_quads(0)
    , local_num_outside_ghosts(0)
  {}


  // =========================================================================
  // =========================================================================
  void
  print()
  {
    std::cout << "local_num_outside_quads: " << local_num_outside_quads << "\n";
    std::cout << "global_num_outside_quads: " << global_num_outside_quads << "\n";
    std::cout << "first_outside_quad_global_id: " << first_outside_quad_global_id << "\n";
    std::cout << "local_num_ghost_quads: " << local_num_ghost_quads << "\n";
  } // print

  // =========================================================================
  // =========================================================================
  void
  reset()
  {
    local_num_outside_quads = 0;
    global_num_outside_quads = 0;
    first_outside_quad_global_id = 0;
    local_num_ghost_quads = 0;
    local_num_outside_ghosts = 0;
  }

  // =========================================================================
  // =========================================================================
  //! update outside quadrants information using p4est forest
  template <size_t dim>
  void
  update(/*const*/ ghost_t<dim> *             ghost,
         [[maybe_unused]] const ParallelEnv & par_env,
         int32_t                              num_outside_quads,
         int32_t                              num_outside_ghosts)
  {
    // from here we assume m_num_outside_quads and num_outside_ghosts have already been computed
    local_num_outside_quads = num_outside_quads;
    local_num_outside_ghosts = num_outside_ghosts;

    // perform a MPI reduce to compute total number of outside quads
#ifdef KALYPSSO_CORE_USE_MPI
    par_env.comm().MPI_Allreduce<MpiComm::SUM>(
      &local_num_outside_quads, &global_num_outside_quads, 1);
#else
    global_num_outside_quads = local_num_outside_quads;
#endif // KALYPSSO_CORE_USE_MPI

    // perform a MPI scan to compute global index of the first outside quads in current MPI proc
#ifdef KALYPSSO_CORE_USE_MPI
    par_env.comm().MPI_Exscan<MpiComm::SUM>(
      &local_num_outside_quads, &first_outside_quad_global_id, 1);
#else
    // it is an exclusive scan, no MPI
    first_outside_quad_global_id = 0;
#endif // KALYPSSO_CORE_USE_MPI

    local_num_ghost_quads = static_cast<int32_t>(ghost->ghosts.elem_count);

  } // update

  // =========================================================================
  // =========================================================================
  //! update outside quadrants information using p4est forest (no ghost, only for test/debug)
  template <size_t dim>
  void
  update(/*const*/ forest_t<dim> *            forest,
         /*const*/ ghost_t<dim> *             ghost,
         [[maybe_unused]] const ParallelEnv & par_env,
         Kokkos::Array<bool, dim>             is_brick_periodic,
         brick_size_t<dim>                    brick_sizes)
  {
    // from here we assume m_num_outside_quads has already been computed
    local_num_outside_quads =
      compute_number_outside_quads<dim>(forest, par_env, is_brick_periodic, brick_sizes);

    // perform a MPI reduce to compute total number of outside quads
#ifdef KALYPSSO_CORE_USE_MPI
    par_env.comm().MPI_Allreduce<MpiComm::SUM>(
      &local_num_outside_quads, &global_num_outside_quads, 1);
#else
    // it is an exclusive scan, no MPI
    global_num_outside_quads = local_num_outside_quads;
#endif // KALYPSSO_CORE_USE_MPI

    // perform a MPI scan to compute global index of the first outside quads in current MPI proc
#ifdef KALYPSSO_CORE_USE_MPI
    par_env.comm().MPI_Exscan<MpiComm::SUM>(
      &local_num_outside_quads, &first_outside_quad_global_id, 1);
#else
    first_outside_quad_global_id = 0;
#endif // KALYPSSO_CORE_USE_MPI


    if (ghost != nullptr)
      local_num_ghost_quads = ghost->ghosts.elem_count;
    else
      local_num_ghost_quads = 0;

    // compute local number of outside quadrants associated to an inside ghost
    local_num_outside_ghosts =
      compute_number_outside_ghosts<dim>(ghost, par_env, is_brick_periodic, brick_sizes);

  } // update

}; // struct OutsideQuadsInfo

} // namespace kalypsso

#endif // KALYPSSO_CORE_OUTSIDEQUADSINFO_H_
