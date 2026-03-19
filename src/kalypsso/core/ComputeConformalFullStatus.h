// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeConformalFullStatus.h
 *
 * Kokkos functor which computes conformal face status for each owned + MPI quadrants.
 */
#ifndef KALYPSSO_CORE_COMPUTECONFORMALFULLSTATUS_H_
#define KALYPSSO_CORE_COMPUTECONFORMALFULLSTATUS_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>
#include <kalypsso/core/ConformalFullStatus.h>

#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/AMRMeshInfo.h>
#include <kalypsso/core/mesh_utils.h> // for definition of Face::XMIN, etc...

namespace kalypsso
{

/**
 * \class ComputeConformalFullStatusFunctor
 */
template <size_t dim, typename device_t>
class ComputeConformalFullStatusFunctor
{
public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;
  using conformal_full_status_view_type = conformal_full_status_view_t<dim, device_t>;

private:
  //! AMR unordered map which maps orchard keys to quadrant number for all key in the mesh
  //! (owned quadrants and ghost quadrants)
  amr_hashmap_t m_amr_hashmap_device;

  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! AMR mesh info (number of owned, MPI ghost, outside quadrants)
  AMRMeshInfo m_amr_mesh_info;

  //! p4est brick connectivity sizes
  const brick_size_t<dim> m_brick_sizes;

  //! is p4est connectivity periodic ?
  const Kokkos::Array<bool, dim> m_is_brick_periodic;

  //! conformal status array
  conformal_full_status_view_type m_conformal_full_status_view;

public:
  /**
   * Constructor.
   *
   * \param[in] amr_hashmap unordered map from orchard key to memory index for owned and ghost
   *            quadrants
   * \param[in] orchard_keys array of orchard key ordered by Morton order
   * \param[in] amr_mesh_info (number of owned, MPI ghosts and outside quadrants)
   * \param[in] brick_sizes is an array of p4est brick connectivity sizes (number of trees in
   *            each dimension)
   * \param[in] is_brick_periodic array of boolean value indicating if the p4est
   *            brick connectivity is periodic in the given dimension
   * \param[out] conformal_status_view
   */
  ComputeConformalFullStatusFunctor(amr_hashmap_t                   amr_hashmap,
                                    orchard_key_view_t              orchard_keys,
                                    AMRMeshInfo                     amr_mesh_info,
                                    brick_size_t<dim>               brick_sizes,
                                    Kokkos::Array<bool, dim>        is_brick_periodic,
                                    conformal_full_status_view_type conformal_full_status_view)
    : m_amr_hashmap_device(amr_hashmap)
    , m_orchard_keys_device(orchard_keys)
    , m_amr_mesh_info(amr_mesh_info)
    , m_brick_sizes(brick_sizes)
    , m_is_brick_periodic(is_brick_periodic)
    , m_conformal_full_status_view(conformal_full_status_view)
  {
    // make sure conformal_status_view has the right size
    KOKKOS_ASSERT(conformal_full_status_view.extent(0) ==
                    static_cast<size_t>(
                      (amr_mesh_info.local_num_quadrants() + amr_mesh_info.local_num_ghosts())) &&
                  "conformal status view has wrong size");
  }

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  static void
  apply(amr_hashmap_t                   amr_hashmap,
        orchard_key_view_t              orchard_keys,
        AMRMeshInfo                     amr_mesh_info,
        brick_size_t<dim>               brick_sizes,
        Kokkos::Array<bool, dim>        is_brick_periodic,
        conformal_full_status_view_type conformal_full_status_view)
  {
    ComputeConformalFullStatusFunctor<dim, device_t> functor(amr_hashmap,
                                                             orchard_keys,
                                                             amr_mesh_info,
                                                             brick_sizes,
                                                             is_brick_periodic,
                                                             conformal_full_status_view);

    const auto num_iterations =
      amr_mesh_info.local_num_quadrants() + amr_mesh_info.local_num_ghosts();

    Kokkos::parallel_for("kalypsso::core::ComputeConformalFullStatusFunctor",
                         Kokkos::RangePolicy<exec_space>(0, num_iterations),
                         functor);

  } // apply

  // ==============================================================
  // ==============================================================
  //! get outside normal direction to a face
  //!
  //! \param[in] face
  KOKKOS_INLINE_FUNCTION
  auto
  get_direction(Face::face_t face) const
  {
    if constexpr (dim == 2)
    {
      if (face == Face::XMIN)
      {
        return Kokkos::Array<int8_t, 2>{ -1, 0 };
      }
      else if (face == Face::XMAX)
      {
        return Kokkos::Array<int8_t, 2>{ 1, 0 };
      }
      else if (face == Face::YMIN)
      {
        return Kokkos::Array<int8_t, 2>{ 0, -1 };
      }
      else if (face == Face::YMAX)
      {
        return Kokkos::Array<int8_t, 2>{ 0, 1 };
      }
      else
      {
        return Kokkos::Array<int8_t, 2>{ 0, 0 };
      }
    }
    else if constexpr (dim == 3)
    {
      if (face == Face::XMIN)
      {
        return Kokkos::Array<int8_t, 3>{ -1, 0, 0 };
      }
      else if (face == Face::XMAX)
      {
        return Kokkos::Array<int8_t, 3>{ 1, 0, 0 };
      }
      else if (face == Face::YMIN)
      {
        return Kokkos::Array<int8_t, 3>{ 0, -1, 0 };
      }
      else if (face == Face::YMAX)
      {
        return Kokkos::Array<int8_t, 3>{ 0, 1, 0 };
      }
      else if (face == Face::ZMIN)
      {
        return Kokkos::Array<int8_t, 3>{ 0, 0, -1 };
      }
      else if (face == Face::ZMAX)
      {
        return Kokkos::Array<int8_t, 3>{ 0, 0, 1 };
      }
      else
      {
        return Kokkos::Array<int8_t, 3>{ 0, 0, 0 };
      }
    }
  } // get_direction - face normal

  // ==============================================================
  // ==============================================================
  //! get outside normal direction to an edge.
  //!
  //! An edge is identified by the intersection of two faces.
  //!
  //! \param[in] face0
  //! \param[in] face1
  KOKKOS_INLINE_FUNCTION
  auto
  get_direction(Face::face_t face0, Face::face_t face1) const
  {
    if constexpr (dim == 2)
    {
      if (face0 == Face::XMIN and face1 == Face::YMIN)
      {
        return Kokkos::Array<int8_t, 2>{ -1, -1 };
      }
      else if (face0 == Face::XMAX and face1 == Face::YMIN)
      {
        return Kokkos::Array<int8_t, 2>{ 1, -1 };
      }
      else if (face0 == Face::XMIN and face1 == Face::YMAX)
      {
        return Kokkos::Array<int8_t, 2>{ -1, 1 };
      }
      else if (face0 == Face::XMAX and face1 == Face::YMAX)
      {
        return Kokkos::Array<int8_t, 2>{ 1, 1 };
      }
      else
      {
        return Kokkos::Array<int8_t, 2>{ 0, 0 };
      }
    }
    else if constexpr (dim == 3)
    {
      if (face0 == Face::XMIN and face1 == Face::YMIN)
      {
        return Kokkos::Array<int8_t, 3>{ -1, -1, 0 };
      }
      else if (face0 == Face::XMAX and face1 == Face::YMIN)
      {
        return Kokkos::Array<int8_t, 3>{ 1, -1, 0 };
      }
      else if (face0 == Face::XMIN and face1 == Face::YMAX)
      {
        return Kokkos::Array<int8_t, 3>{ -1, 1, 0 };
      }
      else if (face0 == Face::XMAX and face1 == Face::YMAX)
      {
        return Kokkos::Array<int8_t, 3>{ 1, 1, 0 };
      }
      else if (face0 == Face::YMIN and face1 == Face::ZMIN)
      {
        return Kokkos::Array<int8_t, 3>{ 0, -1, -1 };
      }
      else if (face0 == Face::YMAX and face1 == Face::ZMIN)
      {
        return Kokkos::Array<int8_t, 3>{ 0, 1, -1 };
      }
      else if (face0 == Face::YMIN and face1 == Face::ZMAX)
      {
        return Kokkos::Array<int8_t, 3>{ 0, -1, 1 };
      }
      else if (face0 == Face::YMAX and face1 == Face::ZMAX)
      {
        return Kokkos::Array<int8_t, 3>{ 0, 1, 1 };
      }
      else if (face0 == Face::XMIN and face1 == Face::ZMIN)
      {
        return Kokkos::Array<int8_t, 3>{ -1, 0, -1 };
      }
      else if (face0 == Face::XMAX and face1 == Face::ZMIN)
      {
        return Kokkos::Array<int8_t, 3>{ 1, 0, -1 };
      }
      else if (face0 == Face::XMIN and face1 == Face::ZMAX)
      {
        return Kokkos::Array<int8_t, 3>{ -1, 0, 1 };
      }
      else if (face0 == Face::XMAX and face1 == Face::ZMAX)
      {
        return Kokkos::Array<int8_t, 3>{ 1, 0, 1 };
      }
      else
      {
        return Kokkos::Array<int8_t, 3>{ 0, 0, 0 };
      }
    }
  } // get_direction - edge normal

  // ==============================================================
  // ==============================================================
  //! get outside normal direction to a corner (3d).
  //!
  //! An corner in 3d is identified by the intersection of three faces.
  //!
  //! \param[in] face0
  //! \param[in] face1
  //! \param[in] face2
  KOKKOS_INLINE_FUNCTION
  auto
  get_direction(Face::face_t face0, Face::face_t face1, Face::face_t face2) const
  {
    if constexpr (dim == 3)
    {
      if (face0 == Face::XMIN and face1 == Face::YMIN and face2 == Face::ZMIN)
      {
        return Kokkos::Array<int8_t, 3>{ -1, -1, -1 };
      }
      else if (face0 == Face::XMAX and face1 == Face::YMIN and face2 == Face::ZMIN)
      {
        return Kokkos::Array<int8_t, 3>{ 1, -1, -1 };
      }
      else if (face0 == Face::XMIN and face1 == Face::YMAX and face2 == Face::ZMIN)
      {
        return Kokkos::Array<int8_t, 3>{ -1, 1, -1 };
      }
      else if (face0 == Face::XMAX and face1 == Face::YMAX and face2 == Face::ZMIN)
      {
        return Kokkos::Array<int8_t, 3>{ 1, 1, -1 };
      }
      else if (face0 == Face::XMIN and face1 == Face::YMIN and face2 == Face::ZMAX)
      {
        return Kokkos::Array<int8_t, 3>{ -1, -1, 1 };
      }
      else if (face0 == Face::XMAX and face1 == Face::YMIN and face2 == Face::ZMAX)
      {
        return Kokkos::Array<int8_t, 3>{ 1, -1, 1 };
      }
      else if (face0 == Face::XMIN and face1 == Face::YMAX and face2 == Face::ZMAX)
      {
        return Kokkos::Array<int8_t, 3>{ -1, 1, 1 };
      }
      else if (face0 == Face::XMAX and face1 == Face::YMAX and face2 == Face::ZMAX)
      {
        return Kokkos::Array<int8_t, 3>{ 1, 1, 1 };
      }
      else
      {
        return Kokkos::Array<int8_t, 3>{ 0, 0, 0 };
      }
    }
  } // get_direction - corner normal (3d only)

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION void
  set_face_status(Face::face_t                                      face,
                  key_t                                             key_cur,
                  typename conformal_full_status_t<dim>::status_t & status) const
  {
    const auto dir = get_direction(face);

    // get neighbor key at same level
    const auto key_neigh_same_level = orchard_key_t<dim>::get_neighbor_key_same_level(
      key_cur, dir, m_brick_sizes, m_is_brick_periodic);

    // check if neighbor key exists in hash map
    auto key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_same_level);

    const auto is_at_same_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

    // if key is valid, it means neighbor is actually at the same level
    if (is_at_same_level)
    {
      conformal_full_status_t<dim>::set_status(
        face, conformal_neighbor_status::NEIGHBOR_IS_AT_SAME_LEVEL, status);
    }
    else
    {
      // check if father exist (coarser level)
      const auto key_neigh_coarser = orchard_key_t<dim>::father(key_neigh_same_level);

      key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_coarser);

      const auto is_at_coarser_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

      if (is_at_coarser_level)
      {
        conformal_full_status_t<dim>::set_status(
          face, conformal_neighbor_status::NEIGHBOR_IS_COARSER, status);
      }
      else
      {
        // test if a neighbor at finer scale exists
        const auto child_id = orchard_key_t<dim>::get_face_neighbor_smallest_child_id(face);
        const auto key_neigh_finer = orchard_key_t<dim>::child(key_neigh_same_level, child_id);

        key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_finer);
        const auto is_at_finer_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

        if (is_at_finer_level)
        {
          conformal_full_status_t<dim>::set_status(
            face, conformal_neighbor_status::NEIGHBOR_IS_FINER, status);
        }
        else
        {
          // unavailable (should be in a ghost octant)
          conformal_full_status_t<dim>::set_status(
            face, conformal_neighbor_status::NEIGHBOR_IS_UNAVAILABLE, status);
        }
      }
    }

  } // set_face_status

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION void
  set_edge_status(Face::face_t                                      face0,
                  Face::face_t                                      face1,
                  key_t                                             key_cur,
                  typename conformal_full_status_t<dim>::status_t & status) const
  {
    const auto dir = get_direction(face0, face1);

    // get neighbor key at same level
    const auto key_neigh_same_level = orchard_key_t<dim>::get_neighbor_key_same_level(
      key_cur, dir, m_brick_sizes, m_is_brick_periodic);

    // check if neighbor key exists in hash map
    auto key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_same_level);

    const auto is_at_same_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

    // if key is valid, it means neighbor is actually at the same level
    if (is_at_same_level)
    {
      conformal_full_status_t<dim>::set_status(
        face0, face1, conformal_neighbor_status::NEIGHBOR_IS_AT_SAME_LEVEL, status);
    }
    else
    {
      // check if father exist (coarser level)
      const auto key_neigh_coarser = orchard_key_t<dim>::father(key_neigh_same_level);

      key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_coarser);

      const auto is_at_coarser_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

      if (is_at_coarser_level)
      {
        conformal_full_status_t<dim>::set_status(
          face0, face1, conformal_neighbor_status::NEIGHBOR_IS_COARSER, status);
      }
      else
      {
        // test if a neighbor at finer scale exists
        const auto child_id = orchard_key_t<dim>::get_edge_neighbor_smallest_child_id(face0, face1);
        const auto key_neigh_finer = orchard_key_t<dim>::child(key_neigh_same_level, child_id);

        key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_finer);
        const auto is_at_finer_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

        if (is_at_finer_level)
        {
          conformal_full_status_t<dim>::set_status(
            face0, face1, conformal_neighbor_status::NEIGHBOR_IS_FINER, status);
        }
        else
        {
          // unavailable (should be in a ghost octant)
          conformal_full_status_t<dim>::set_status(
            face0, face1, conformal_neighbor_status::NEIGHBOR_IS_UNAVAILABLE, status);
        }
      }
    }

  } // set_edge_status

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION void
  set_corner_status(Face::face_t                                      face0,
                    Face::face_t                                      face1,
                    Face::face_t                                      face2,
                    key_t                                             key_cur,
                    typename conformal_full_status_t<dim>::status_t & status) const
  {
    if constexpr (dim == 3)
    {
      const auto dir = get_direction(face0, face1, face2);

      // get neighbor key at same level
      const auto key_neigh_same_level = orchard_key_t<dim>::get_neighbor_key_same_level(
        key_cur, dir, m_brick_sizes, m_is_brick_periodic);

      // check if neighbor key exists in hash map
      auto key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_same_level);

      const auto is_at_same_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

      // if key is valid, it means neighbor is actually at the same level
      if (is_at_same_level)
      {
        conformal_full_status_t<dim>::set_status(
          face0, face1, face2, conformal_neighbor_status::NEIGHBOR_IS_AT_SAME_LEVEL, status);
      }
      else
      {
        // check if father exist (coarser level)
        const auto key_neigh_coarser = orchard_key_t<dim>::father(key_neigh_same_level);

        key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_coarser);

        const auto is_at_coarser_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

        if (is_at_coarser_level)
        {
          conformal_full_status_t<dim>::set_status(
            face0, face1, face2, conformal_neighbor_status::NEIGHBOR_IS_COARSER, status);
        }
        else
        {
          // test if a neighbor at finer scale exists
          const auto child_id =
            orchard_key_t<dim>::get_corner_neighbor_smallest_child_id(face0, face1, face2);
          const auto key_neigh_finer = orchard_key_t<dim>::child(key_neigh_same_level, child_id);

          key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_finer);
          const auto is_at_finer_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

          if (is_at_finer_level)
          {
            conformal_full_status_t<dim>::set_status(
              face0, face1, face2, conformal_neighbor_status::NEIGHBOR_IS_FINER, status);
          }
          else
          {
            // unavailable (should be in a ghost octant)
            conformal_full_status_t<dim>::set_status(
              face0, face1, face2, conformal_neighbor_status::NEIGHBOR_IS_UNAVAILABLE, status);
          }
        }
      }
    }
  } // set_corner_status

  // ==============================================================
  // ==============================================================
  /**
   * range policy functor
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(const index_t & iOct) const
  {

    // get orchard key of current octant
    auto key_cur = m_orchard_keys_device(iOct);

    typename conformal_full_status_t<dim>::status_t status = 0;

    //
    // set face status
    //
    for (Face::face_t face = 0; face < Face::num_faces<dim>(); ++face)
    {
      set_face_status(face, key_cur, status);
    } // end for face

    //
    // set edge status
    //
    set_edge_status(Face::XMIN, Face::YMIN, key_cur, status);
    set_edge_status(Face::XMAX, Face::YMIN, key_cur, status);
    set_edge_status(Face::XMIN, Face::YMAX, key_cur, status);
    set_edge_status(Face::XMAX, Face::YMAX, key_cur, status);
    if constexpr (dim == 3)
    {
      set_edge_status(Face::YMIN, Face::ZMIN, key_cur, status);
      set_edge_status(Face::YMAX, Face::ZMIN, key_cur, status);
      set_edge_status(Face::YMIN, Face::ZMAX, key_cur, status);
      set_edge_status(Face::YMAX, Face::ZMAX, key_cur, status);

      set_edge_status(Face::XMIN, Face::ZMIN, key_cur, status);
      set_edge_status(Face::XMAX, Face::ZMIN, key_cur, status);
      set_edge_status(Face::XMIN, Face::ZMAX, key_cur, status);
      set_edge_status(Face::XMAX, Face::ZMAX, key_cur, status);
    }

    if constexpr (dim == 3)
    {
      set_corner_status(Face::XMIN, Face::YMIN, Face::ZMIN, key_cur, status);
      set_corner_status(Face::XMAX, Face::YMIN, Face::ZMIN, key_cur, status);
      set_corner_status(Face::XMIN, Face::YMAX, Face::ZMIN, key_cur, status);
      set_corner_status(Face::XMAX, Face::YMAX, Face::ZMIN, key_cur, status);
      set_corner_status(Face::XMIN, Face::YMIN, Face::ZMAX, key_cur, status);
      set_corner_status(Face::XMAX, Face::YMIN, Face::ZMAX, key_cur, status);
      set_corner_status(Face::XMIN, Face::YMAX, Face::ZMAX, key_cur, status);
      set_corner_status(Face::XMAX, Face::YMAX, Face::ZMAX, key_cur, status);
    }

    // status is up to date, just write it now
    m_conformal_full_status_view(iOct) = status;

  } // operator()

}; // class ComputeConformalFullStatusFunctor

} // namespace kalypsso

#endif // KALYPSSO_CORE_COMPUTECONFORMALFULLSTATUS_H_
