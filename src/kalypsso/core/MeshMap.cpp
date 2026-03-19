// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MeshMap.cpp
 * \brief Container for mesh data adapted from Kokkos::UnorderedMap.
 *
 */

#include <kalypsso/core/MeshMap.h>

namespace kalypsso
{

// =========================================================================
// =========================================================================
template <size_t dim, typename device_t>
MeshMap<dim, device_t>::MeshMap(ConfigMap const & config_map, ParallelEnv const & par_env)
  : m_config_map(config_map)
  , m_par_env(par_env)
  , m_outside_quads_info()
  , m_amr_mesh_info()
{

  m_brick_sizes = []() {
    if constexpr (dim == 2)
      return brick_size_t<2>{ 1, 1 };
    else if constexpr (dim == 3)
      return brick_size_t<3>{ 1, 1, 1 };
  }();
  m_is_brick_periodic = []() {
    if constexpr (dim == 2)
      return Kokkos::Array<bool, 2>{ false, false };
    else if constexpr (dim == 3)
      return Kokkos::Array<bool, 3>{ false, false, false };
  }();

  // check if connectivity is either "unit" or "brick"
  const auto conn_name = m_config_map.getString("amr", "connectivity", "invalid_connectivity");

  if (conn_name == "brick")
  {
    m_brick_sizes = get_brick_sizes<dim>(config_map);

    m_is_brick_periodic[IX] = static_cast<bool>(
      m_config_map.getInteger("p4est_connectivity", "periodic_x", CONNECTIVITY_PERIODIC_FALSE));
    m_is_brick_periodic[IY] = static_cast<bool>(
      m_config_map.getInteger("p4est_connectivity", "periodic_y", CONNECTIVITY_PERIODIC_FALSE));
    if constexpr (dim == 3)
      m_is_brick_periodic[IZ] = static_cast<bool>(
        m_config_map.getInteger("p4est_connectivity", "periodic_z", CONNECTIVITY_PERIODIC_FALSE));
  }
} // MeshMap<dim, device_t>::MeshMap - constructor

// =========================================================================
// =========================================================================
template <size_t dim, typename device_t>
void
MeshMap<dim, device_t>::compute_orchard_keys_view_host(/*const*/ forest_t<dim> * forest,
                                                       /*const*/ ghost_t<dim> *  ghost)
{

  // type alias to access p4est C API (2D or 3D)
  using p4est_t = typename p4est::Wrapper<dim>;

  // perform update mesh info (only if required)
  update_amr_mesh_info(forest, ghost);

  [[maybe_unused]] const auto total_num_quads = m_amr_mesh_info.local_num_quadrants_total();

  assertm(m_orchard_keys_host.extent(0) == static_cast<size_t>(total_num_quads),
          "m_orchard_keys_host has wrong size !");

  // tree linear index to xyz converter
  const BrickConnectivityData<dim> convert(m_brick_sizes);

  size_t quadIndex = 0;

  //
  // fill array of orchard keys using MPI-owned quadrants
  //
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

      auto orchard_key =
        orchard_key_t<dim>::encode_orchard(tree_xyz, octCoord, static_cast<uint8_t>(q->level));

      m_orchard_keys_host(quadIndex) = orchard_key;

      quadIndex++;
    }
  } // end filling with owned quadrants

  // const auto num_owned_quads = quadIndex;

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

    auto orchard_key =
      orchard_key_t<dim>::encode_orchard(tree_xyz, octCoord, static_cast<uint8_t>(q->level));

    m_orchard_keys_host(quadIndex) = orchard_key;

    quadIndex++;

  } // end filling with ghost quadrants

  const auto num_owned_and_ghost_quads = quadIndex;

  //
  // complete array of orchard keys using quadrants outside of domain (external border)
  // - if brick connectivity is periodic, these quadrants are already counted as (MPI) ghost
  //   quadrants (see above)
  // - if brick connectivity is not periodic, we chose to add a layer of one quadrant all
  //   around the domain (i.e. outside domain)
  //
  // idea of implementation: just loop over all owned orchard keys, and check if corresponding
  // quadrant touches outside domain by a face, edge or corner, and if true, then add the mirror
  // key to the list of key

  // do we have at least one non-periodic border ?
  bool has_non_periodic_border = false;
  for (size_t idim = 0; idim < dim; ++idim)
  {
    if (!m_is_brick_periodic[idim])
      has_non_periodic_border = true;
  }

  if (has_non_periodic_border)
  {
    // when computing outside quadrant key, we always use a "virtual" key computed as the periodic
    // image of the outside quadrant; so here we need is_periodic to be array of "true"
    // constexpr auto is_periodic = get_bool_array<dim>(true);

    // get list of faces
    const auto faces = Face::get_all_faces<dim>();

    // finally loop over quadrant at border (from the inside)
    for (size_t qId = 0; qId < num_owned_and_ghost_quads; ++qId)
    {
      auto key = m_orchard_keys_host(qId);

      //
      // insert face-outside neighbors
      //
      for (const auto face : faces)
      {
        if (orchard_key_t<dim>::is_at_domain_border(key, face, m_brick_sizes))
        {
          if (!m_is_brick_periodic[face / 2])
          {
            const auto displacement = orchard_key_t<dim>::face_to_displacement(face);
            auto       mirror_key = orchard_key_t<dim>::get_neighbor_key_same_level(
              key, displacement, m_brick_sizes, m_is_brick_periodic);

            // et voila !
            m_orchard_keys_host(quadIndex) = mirror_key;

            quadIndex++;
          }
        }
      } // end for face

      // insert edge-outside neighbors (3D only)
      if constexpr (dim == 3)
      {
        Face::face_t face0, face1;
        for (uint8_t iEdge = 0; iEdge < Edge::num_edges<dim>(); ++iEdge)
        {
          edge_to_faces(iEdge, face0, face1);
          if (orchard_key_t<dim>::is_at_domain_border(key, face0, m_brick_sizes) and
              orchard_key_t<dim>::is_at_domain_border(key, face1, m_brick_sizes))
          {
            if (!m_is_brick_periodic[face0 / 2] or !m_is_brick_periodic[face1 / 2])
            {

              const auto displacement = orchard_key_t<dim>::face_to_displacement(face0, face1);
              auto       mirror_key = orchard_key_t<dim>::get_neighbor_key_same_level(
                key, displacement, m_brick_sizes, m_is_brick_periodic);

              // et voila !
              m_orchard_keys_host(quadIndex) = mirror_key;

              quadIndex++;
            }
          }
        }
      }

      //
      // insert corner-outside neighbors
      //
      for (uint8_t iCorner = 0; iCorner < Corner::num_corners<dim>(); ++iCorner)
      {
        if constexpr (dim == 2)
        {
          Face::face_t face0, face1;
          corner_to_faces(iCorner, face0, face1);
          if (orchard_key_t<dim>::is_at_domain_border(key, face0, m_brick_sizes) and
              orchard_key_t<dim>::is_at_domain_border(key, face1, m_brick_sizes))
          {
            if (!m_is_brick_periodic[face0 / 2] or !m_is_brick_periodic[face1 / 2])
            {
              const auto displacement = orchard_key_t<dim>::face_to_displacement(face0, face1);
              auto       mirror_key = orchard_key_t<dim>::get_neighbor_key_same_level(
                key, displacement, m_brick_sizes, m_is_brick_periodic);

              // et voila !
              m_orchard_keys_host(quadIndex) = mirror_key;

              quadIndex++;
            }
          }
        }
        else if constexpr (dim == 3)
        {
          Face::face_t face0, face1, face2;
          corner_to_faces(iCorner, face0, face1, face2);
          if (orchard_key_t<dim>::is_at_domain_border(key, face0, m_brick_sizes) and
              orchard_key_t<dim>::is_at_domain_border(key, face1, m_brick_sizes) and
              orchard_key_t<dim>::is_at_domain_border(key, face2, m_brick_sizes))
          {
            // clang-format off
              if (!m_is_brick_periodic[face0 / 2] or
                  !m_is_brick_periodic[face1 / 2] or
                  !m_is_brick_periodic[face2 / 2])
              {
                const auto displacement =
                  orchard_key_t<dim>::face_to_displacement(face0, face1, face2);
                auto mirror_key = orchard_key_t<dim>::get_neighbor_key_same_level(
                  key, displacement, m_brick_sizes, m_is_brick_periodic);

                // et voila !
                m_orchard_keys_host(quadIndex) = mirror_key;

                quadIndex++;
              }
            // clang-format on
          }
        }
      } // end for iCorner

    } // end filling external quadrant

  } // end has_non_periodic_border == true

} // MeshMap<dim, device_t>::compute_orchard_keys_view_host

// =========================================================================
// =========================================================================
template <size_t dim, typename device_t>
void
MeshMap<dim, device_t>::update_orchard_keys(/*const*/ forest_t<dim> * forest,
                                            /*const*/ ghost_t<dim> *  ghost)
{

  assertm(ghost != nullptr, "ghost is not valid / allocated");

  // perform update mesh info (only if required)
  update_amr_mesh_info(forest, ghost);

  [[maybe_unused]] const auto total_num_quads = m_amr_mesh_info.local_num_quadrants_total();

  // memory allocation host view
  Kokkos::resize(Kokkos::view_alloc(Kokkos::WithoutInitializing),
                 m_orchard_keys_host,
                 static_cast<size_t>(total_num_quads));

  compute_orchard_keys_view_host(forest, ghost);

  // memory allocation device view
  Kokkos::resize(Kokkos::view_alloc(exec_space(), Kokkos::WithoutInitializing),
                 m_orchard_keys,
                 static_cast<size_t>(total_num_quads));

  // upload view to device
  Kokkos::deep_copy(m_orchard_keys, m_orchard_keys_host);

} // MeshMap<dim, device_t>::update_orchard_keys

// =========================================================================
// =========================================================================
template <size_t dim, typename device_t>
void
MeshMap<dim, device_t>::update_conformal_status()
{

  Kokkos::resize(Kokkos::view_alloc(exec_space(), Kokkos::WithoutInitializing),
                 m_conformal_status_view,
                 static_cast<size_t>(m_amr_mesh_info.local_num_quadrants() +
                                     m_amr_mesh_info.local_num_ghosts()));

  ComputeConformalStatusFunctor<dim, device_t>::apply(m_amr_hashmap,
                                                      m_orchard_keys,
                                                      m_amr_mesh_info,
                                                      m_brick_sizes,
                                                      m_is_brick_periodic,
                                                      m_conformal_status_view);

} // MeshMap<dim, device_t>::update_conformal_status

// =========================================================================
// =========================================================================
template <size_t dim, typename device_t>
auto
MeshMap<dim, device_t>::update_conformal_full_status()
  -> conformal_full_status_view_t<dim, device_t>
{
  auto conformal_full_status_view = conformal_full_status_view_t<dim, device_t>(
    "conformal full status view",
    static_cast<size_t>(m_amr_mesh_info.local_num_quadrants() +
                        m_amr_mesh_info.local_num_ghosts()));

  ComputeConformalFullStatusFunctor<dim, device_t>::apply(m_amr_hashmap,
                                                          m_orchard_keys,
                                                          m_amr_mesh_info,
                                                          m_brick_sizes,
                                                          m_is_brick_periodic,
                                                          conformal_full_status_view);

  return conformal_full_status_view;

} // MeshMap<dim, device_t>::update_conformal_full_status

// =========================================================================
// =========================================================================
template <size_t dim, typename device_t>
void
MeshMap<dim, device_t>::compute_mirror_orchard_keys_view_host(/*const*/ ghost_t<dim> * ghost)
{

  // type alias to access p4est C API (2D or 3D)
  using p4est_t = typename p4est::Wrapper<dim>;

  assertm(static_cast<int32_t>(m_mirror_orchard_keys_host.extent(0)) ==
            (ghost->mirror_proc_offsets[m_par_env.size()]),
          "m_mirror_orchard_keys_host has wrong size, should be "
          "ghost->mirror_proc_offsets[par_env.size()] !");

  // tree linear index to xyz converter
  const BrickConnectivityData<dim> convert(m_brick_sizes);

  //
  // very important :
  // - mirrors in p4est ghost->mirrors array are sorted only by Morton key
  // - mirrors data used in MPI ghost exchange routine (e.g. p4est_ghost_exchange_custom_begin)
  //   must be sorted first by MPI process, then by Morton key; this sorting is done thanks to
  //   ghost->mirror_proc_mirrors (that we actually use below)
  //
  uint32_t index = 0;
  for (int iproc = 0; iproc < m_par_env.size(); ++iproc)
  {
    // retrieve num of mirror quadrants involved in sending data to process iproc
    auto num_mirrors = ghost->mirror_proc_offsets[iproc + 1] - ghost->mirror_proc_offsets[iproc];

    auto index_offset = ghost->mirror_proc_offsets[iproc];

    for (int mirrorId = 0; mirrorId < num_mirrors; ++mirrorId)
    {
      // get index (ordered by Morton order) for addressing ghost->mirrors
      auto mir_index = ghost->mirror_proc_mirrors[index_offset + mirrorId];

      // get the mirror quadrant itself
      auto q = p4est_t::quadrant_array_index(&(ghost->mirrors), static_cast<size_t>(mir_index));
      auto which_tree = q->p.which_tree;

      // get tree cartesian coordinate (with respect to brick connectivity)
      auto tree_xyz = convert.toXYZ(which_tree);

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

      auto orchard_key =
        orchard_key_t<dim>::encode_orchard(tree_xyz, octCoord, static_cast<uint8_t>(q->level));

      m_mirror_orchard_keys_host(index) = orchard_key;
      ++index;
    }

  } // end for iproc

} // MeshMap<dim, device_t>::compute_mirror_orchard_keys_view_host

// =========================================================================
// =========================================================================
template <size_t dim, typename device_t>
void
MeshMap<dim, device_t>::update_mirror_orchard_keys(/*const*/ ghost_t<dim> * ghost)
{

  assertm(ghost != nullptr, "ghost is not valid / allocated");

  // memory allocation for host view
  Kokkos::resize(Kokkos::view_alloc(Kokkos::WithoutInitializing),
                 m_mirror_orchard_keys_host,
                 static_cast<size_t>(ghost->mirror_proc_offsets[m_par_env.size()]));

  compute_mirror_orchard_keys_view_host(ghost);

  // memory allocation for device view
  Kokkos::resize(Kokkos::view_alloc(Kokkos::WithoutInitializing),
                 m_mirror_orchard_keys,
                 static_cast<size_t>(ghost->mirror_proc_offsets[m_par_env.size()]));

  // upload view to device
  Kokkos::deep_copy(m_mirror_orchard_keys, m_mirror_orchard_keys_host);

} // MeshMap<dim, device_t>::update_mirror_orchard_keys

// =========================================================================
// =========================================================================
template <size_t dim, typename device_t>
void
MeshMap<dim, device_t>::update_hashmap(bool on_device)
{
  const auto total_num_quads = m_amr_mesh_info.local_num_quadrants_total();

  if (on_device)
  {
    // avoid nvcc's warning about lambda capturing this implicitly
    // TODO see if we can avoid this (eventhough the following are shadow copies)
    auto amr_hashmap_device = m_amr_hashmap;
    auto orchard_keys_device = m_orchard_keys;

    // 2. fill the hash map on device (using our device_t)
    parallel_for(
      "MeshMap::fill_map",
      Kokkos::RangePolicy<exec_space>(0, total_num_quads),
      KOKKOS_LAMBDA(const int & index) {
        amr_hashmap_device.insert(orchard_keys_device(index), index);
      });

    //! 3. download hash map to host
    m_amr_hashmap_host.rehash(static_cast<uint32_t>(total_num_quads));
    Kokkos::deep_copy(m_amr_hashmap_host, m_amr_hashmap);
  }
  else
  {
    // avoid nvcc's warning about lambda capturing this implicitly
    // TODO see if we can avoid this (eventhough the following are shadow copies)
    auto amr_hashmap_host = m_amr_hashmap_host;
    auto orchard_keys_host = m_orchard_keys_host;

    //! 2. fill the hash map on host (using Kokkos::OpenMP)
    parallel_for(
      "MeshMap::fill_map",
      Kokkos::RangePolicy<Kokkos::OpenMP>(0, total_num_quads),
      KOKKOS_LAMBDA(const int & index) {
        amr_hashmap_host.insert(orchard_keys_host(index), index);
      });

    //! 3. upload hash map on device
    m_amr_hashmap.rehash(static_cast<uint32_t>(total_num_quads));
    Kokkos::deep_copy(m_amr_hashmap, m_amr_hashmap_host);
  }
} // MeshMap<dim, device_t>::update_hashmap

// =========================================================================
// =========================================================================
template <size_t dim, typename device_t>
void
MeshMap<dim, device_t>::update_hashmap(/* const */ forest_t<dim> * forest,
                                       /* const */ ghost_t<dim> *  ghost,
                                       bool                        on_device)
{

  // perform update mesh info (only if required)
  update_amr_mesh_info(forest, ghost);

  [[maybe_unused]] const auto total_num_quads = m_amr_mesh_info.local_num_quadrants_total();

  // update orchard keys arrays (host and device)
  update_orchard_keys(forest, ghost);

  // clear (device) hashmap
  m_amr_hashmap.clear();
  m_amr_hashmap_host.clear();

  // resize (device) hashmap to hold all quad (owned + ghosts + outside) of current MPI task
  m_amr_hashmap.rehash(static_cast<uint32_t>(total_num_quads));
  m_amr_hashmap_host.rehash(static_cast<uint32_t>(total_num_quads));

  update_hashmap(on_device);

} // MeshMap<dim, device_t>::update_hashmap

// =========================================================================
// =========================================================================
template <size_t dim, typename device_t>
void
MeshMap<dim, device_t>::update_hashmap_serial(/*const*/ forest_t<dim> * forest,
                                              /*const*/ ghost_t<dim> *  ghost)
{
  // type alias to access p4est C API (2D or 3D)
  using p4est_t = typename p4est::Wrapper<dim>;

  // tree linear index to xyz converter
  const BrickConnectivityData<dim> convert(m_brick_sizes);

  // perform update mesh info (only if required)
  update_amr_mesh_info(forest, ghost);

  [[maybe_unused]] const auto total_num_quads = m_amr_mesh_info.local_num_quadrants_total();

  // clear hashmap
  m_amr_hashmap.clear();

  // resize hashmap to hold all quad (owned + ghosts + external) of current MPI task
  m_amr_hashmap.rehash(static_cast<uint32_t>(total_num_quads));

  amr_hashmap_host_t amr_hashmap_host;
  amr_hashmap_host.rehash(static_cast<uint32_t>(total_num_quads));

  iOct_t quadIndex = 0;

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

      auto orchard_key =
        orchard_key_t<dim>::encode_orchard(tree_xyz, octCoord, static_cast<uint8_t>(q->level));

      // auto insert_result =
      amr_hashmap_host.insert(orchard_key, quadIndex);

      quadIndex++;
    }
  } // end filling hashmap with owned quadrants

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

    auto orchard_key =
      orchard_key_t<dim>::encode_orchard(tree_xyz, octCoord, static_cast<uint8_t>(q->level));

    // auto insert_result =
    amr_hashmap_host.insert(orchard_key, quadIndex);

    quadIndex++;

  } // end filling hashmap with ghost quadrants

  if (amr_hashmap_host.failed_insert() and m_par_env.rank() == 0)
    printf("[MeshMap] Something went wrong in fill the hash table)\n");

  m_amr_hashmap.rehash(static_cast<uint32_t>(total_num_quads));
  Kokkos::deep_copy(m_amr_hashmap, amr_hashmap_host);

} // MeshMap<dim, device_t>::update_hashmap_serial

// =========================================================================
// =========================================================================
template <size_t dim, typename device_t>
bool
MeshMap<dim, device_t>::is_amr_mesh_info_uptodate(/*const*/ forest_t<dim> * forest,
                                                  /*const*/ ghost_t<dim> *  ghost)
{
  // clang-format off
    return
      (m_amr_mesh_info.local_num_quadrants() == forest->local_num_quadrants) &&
      (m_amr_mesh_info.global_num_quadrants() == forest->global_num_quadrants) &&
      (m_amr_mesh_info.local_num_ghosts() == static_cast<int32_t>(ghost->ghosts.elem_count)) &&
      (m_amr_mesh_info.local_num_mirrors() == ghost->mirror_proc_offsets[m_par_env.size()]);
  // clang-format on
} // MeshMap<dim, device_t>::is_amr_mesh_info_uptodate

// =========================================================================
// =========================================================================
template <size_t dim, typename device_t>
void
MeshMap<dim, device_t>::compute_outside_quad_info(/*const*/ forest_t<dim> * forest,
                                                  /*const*/ ghost_t<dim> *  ghost)
{
  // do we have at least one non-periodic border ?
  bool has_non_periodic_border = false;
  for (size_t idim = 0; idim < dim; ++idim)
  {
    if (!m_is_brick_periodic[idim])
      has_non_periodic_border = true;
  }

  // if all border are periodic, ok no quad outside
  if (!has_non_periodic_border)
  {
    m_outside_quads_info.reset();
  }
  else
  {
    auto num_outside_quads =
      compute_number_outside_quads<dim>(forest, m_par_env, m_is_brick_periodic, m_brick_sizes);
    auto num_outside_ghosts =
      compute_number_outside_ghosts<dim>(ghost, m_par_env, m_is_brick_periodic, m_brick_sizes);

    m_outside_quads_info.update<dim>(ghost, m_par_env, num_outside_quads, num_outside_ghosts);
  }
} // MeshMap<dim, device_t>::compute_outside_quad_info

// =========================================================================
// =========================================================================
template <size_t dim, typename device_t>
void
MeshMap<dim, device_t>::update_amr_mesh_info(/*const*/ forest_t<dim> * forest,
                                             /*const*/ ghost_t<dim> *  ghost)
{

  {
    // make sure the number of outside quads is up to date
    compute_outside_quad_info(forest, ghost);

    m_amr_mesh_info.local_num_quadrants() = forest->local_num_quadrants;
    m_amr_mesh_info.global_num_quadrants() = forest->global_num_quadrants;
    m_amr_mesh_info.global_first_quadrant() = forest->global_first_quadrant[m_par_env.rank()];

    m_amr_mesh_info.local_num_ghosts() = static_cast<int32_t>(ghost->ghosts.elem_count);
    m_amr_mesh_info.local_num_mirrors() = ghost->mirror_proc_offsets[forest->mpisize];

    // clang-format off
    m_amr_mesh_info.local_num_quadrants_outside() = m_outside_quads_info.local_num_outside_quads;
    m_amr_mesh_info.global_num_quadrants_outside() = m_outside_quads_info.global_num_outside_quads;
    m_amr_mesh_info.global_first_quadrant_outside() = m_outside_quads_info.first_outside_quad_global_id;
    m_amr_mesh_info.local_num_quadrants_outside_ghost() = m_outside_quads_info.local_num_outside_ghosts;
    m_amr_mesh_info.mpi_rank() = m_par_env.rank();
    // clang-format on
  }

} // MeshMap<dim, device_t>::update_amr_mesh_info

// explicit template instantiation
template class MeshMap<2, kalypsso::DefaultDevice>;
template class MeshMap<3, kalypsso::DefaultDevice>;

// only instantiate those class when the default device is not on host
#if defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP) || \
  defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_THREADS) ||  \
  defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL)
#else
template class MeshMap<2, kalypsso::HostDevice>;
template class MeshMap<3, kalypsso::HostDevice>;
#endif // KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE

} // namespace kalypsso
