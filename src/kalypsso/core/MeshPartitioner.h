// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MeshPartitioner.h
 */
#ifndef KALYPSSO_CORE_MESHPARTITIONER_H_
#define KALYPSSO_CORE_MESHPARTITIONER_H_

#include <kalypsso/core/kalypsso_core_base.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock, FaceDataArrayBlock, ...
#include <kalypsso/core/MaterialPresence.h>

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <kalypsso/utils/p4est/p4est_wrapper.h>

#include <kalypsso/core/kalypsso_comm_config.h>

#include <kalypsso/utils/log/kalypsso_log.h>

#include <vector> // for std::vector

#include "MeshPartitioner_helper.h"

namespace kalypsso
{

/**
 * \class MeshPartitioner
 *
 * Two main functions:
 *
 * - wrapper around p4est API p4est_partition for re-partitioning mesh among MPI processes
 * - propagate new partition to user data
 */
template <size_t dim, typename device_t>
class MeshPartitioner
{

private:
public:
  using exec_space = typename device_t::execution_space;

  //! type alias to access p4est C API (2D or 3D)
  using p4est_t = typename p4est::Wrapper<dim>;
  using forest_t = typename p4est_t::forest_t;
  using weight_cb_t = typename p4est_t::weight_cb_t;

  template <typename T>
  using DataArrayLeafSoA_t = DataArrayLeafSoA<T, device_t>;

  template <typename T>
  using DataArrayLeafAoS_t = DataArrayLeafAoS<T, device_t>;

  template <typename T>
  using DataArrayBlock_t = DataArrayBlock<dim, T, device_t>;

  template <typename T>
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, T, device_t>;

  template <typename T>
  using DataArrayBlockMultiVar_t = DataArrayBlockMultiVar<dim, T, device_t>;

  template <typename T>
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, T, device_t>;

  using MaterialPresenceView_t = MaterialPresenceView<device_t>;

  using Weights_t = Kokkos::View<uint32_t *, HostDevice>;
  using Hashmap_t = typename hashmap_base_t<device_t>::map_t::HostMirror;

  // =========================================================================
  // =========================================================================
  //! constructor
  MeshPartitioner(const ConfigMap & config_map, const ParallelEnv & par_env)
    : m_config_map(config_map)
    , m_par_env(par_env)
    , m_local_num_quadrants_new(0)
  {
    m_lb_gfq_old.resize(static_cast<size_t>(m_par_env.nRanks() + 1));
    m_lb_gfq_new.resize(static_cast<size_t>(m_par_env.nRanks() + 1));
  }

  // =========================================================================
  // =========================================================================
  //! destructor
  ~MeshPartitioner() = default;

  // =========================================================================
  // =========================================================================
  //! Repartition a p4est mesh (ONLY meta data, NO user data).
  //!
  //! We also use alternatively the words \"load balancing\" for designating the mesh partitioning
  //! operation.
  //!
  //! \param[in] forest, the p4est main object
  //!
  //! After this call, p4est object will change, e.g. :
  //! - local number of quadrant will change; this is the main goal of p4est_partition to have a new
  //!   mesh for which the local number of quadrants is (almost) the same on all MPI processes
  //! - ghost won't be valid anymore
  void
  partition_mesh(forest_t * forest)
  {
    Kokkos::Profiling::ScopedRegion myprof("MeshPartitioner::partition_mesh");

    // if there is only one MPI rank, nothing to do
    if (m_par_env.nRanks() < 2)
      return;

    // remainder: MPI partition is encoded into the gfq array which is an array containing the
    // global index of the first quadrant of each MPI process)

    // 1. record the current (soon to be old) partition gfq array
    {
      const size_t gfq_size = m_lb_gfq_old.size();
      for (uint32_t rank = 0; rank < gfq_size; ++rank)
      {
        m_lb_gfq_old[rank] = forest->global_first_quadrant[rank];
      }
    }

    // 2. call p4est_partition
    const int         allow_for_coarsening = 1;
    const weight_cb_t weight_cb = nullptr;

    KALYPSSO_INFO("before p4est_partition");

    p4est_t::partition(forest, allow_for_coarsening, weight_cb);

    KALYPSSO_INFO("after p4est_partition");

    // 3. record new gfq (global first quadrant index)
    {
      const auto gfq_size = m_lb_gfq_new.size();
      for (uint32_t rank = 0; rank < gfq_size; ++rank)
      {
        m_lb_gfq_new[rank] = forest->global_first_quadrant[rank];
      }
    }

    // 4. record the new current number of leaves/quadrants
    m_local_num_quadrants_new = forest->local_num_quadrants;

  } // partition_mesh

  // =========================================================================
  // =========================================================================
  //! Inner type for partition_mesh_with_weights
  struct WeightsCallback
  {
    using topidx_t = p4est::topidx_t;
    using quadrant_t = typename p4est_t::quadrant_t;

    struct Data
    {
      Weights_t                  weights;
      BrickConnectivityData<dim> convert;
      Hashmap_t                  hashmap;
    };

    static Data * data;

    static int
    callback([[maybe_unused]] forest_t * forest, topidx_t tree_id, quadrant_t * quadrant)
    {
      Kokkos::Array<uint32_t, dim> oct_coord;
      oct_coord[0] = static_cast<uint32_t>(
        quadrant->x >> (p4est_t::QMAXLEVEL - orchard_key_t<dim>::NUM_LEVELS + 2));

      oct_coord[1] = static_cast<uint32_t>(
        quadrant->y >> (p4est_t::QMAXLEVEL - orchard_key_t<dim>::NUM_LEVELS + 2));

      if constexpr (dim == 3)
        oct_coord[2] = static_cast<uint32_t>(
          quadrant->z >> (p4est_t::QMAXLEVEL - orchard_key_t<dim>::NUM_LEVELS + 2));

      const auto orchard_key = orchard_key_t<dim>::encode_orchard(
        data->convert.toXYZ(tree_id), oct_coord, static_cast<uint8_t>(quadrant->level));

      const auto i_oct_hash = data->hashmap.find(orchard_key);
      if (data->hashmap.valid_at(i_oct_hash))
      {
        const auto i_oct = data->hashmap.value_at(i_oct_hash);
        return static_cast<int>(data->weights(i_oct));
      }
      else
      {
        KOKKOS_ASSERT("Invalid orchard id in partition_mesh_with_weights callback");
        return 0;
      }
    }
  };

  // =========================================================================
  // =========================================================================
  //!
  //! \param[in] forest the p4est main object
  //! \param[in] weights a host kokkos view that contains the weight of each octant
  //! \param[in] hashmap the orchard key to octant index map
  //!
  void
  partition_mesh_with_weights(forest_t *        forest,
                              Weights_t         weights,
                              brick_size_t<dim> brick_sizes,
                              Hashmap_t         hashmap)
  {
    Kokkos::Profiling::ScopedRegion myprof("MeshPartitioner::partition_mesh_with_weights");

    // if there is only one MPI rank, nothing to do
    if (m_par_env.nRanks() < 2)
      return;

    // remainder: MPI partition is encoded into the gfq array which is an array containing the
    // global index of the first quadrant of each MPI process)

    // 1. record the current (soon to be old) partition gfq array
    {
      const size_t gfq_size = m_lb_gfq_old.size();
      for (uint32_t rank = 0; rank < gfq_size; ++rank)
      {
        m_lb_gfq_old[rank] = forest->global_first_quadrant[rank];
      }
    }

    // 2. call p4est_partition
    const int                      allow_for_coarsening = 1;
    typename WeightsCallback::Data data{ weights,
                                         BrickConnectivityData<dim>(brick_sizes),
                                         hashmap };
    WeightsCallback::data = &data;
    const weight_cb_t weight_cb = &WeightsCallback::callback;

    KALYPSSO_INFO("before p4est_partition");

    p4est_t::partition(forest, allow_for_coarsening, weight_cb);

    KALYPSSO_INFO("after p4est_partition");

    WeightsCallback::data = nullptr;

    // 3. record new gfq (global first quadrant index)
    {
      const auto gfq_size = m_lb_gfq_new.size();
      for (uint32_t rank = 0; rank < gfq_size; ++rank)
      {
        m_lb_gfq_new[rank] = forest->global_first_quadrant[rank];
      }
    }

    // 4. record the new current number of leaves/quadrants
    m_local_num_quadrants_new = forest->local_num_quadrants;

  } // partition_mesh_with_weights

  // =======================================================
  // =======================================================
  //! cell-center user data load balancing (repartitioning) with MPI data communication.
  //!
  //! MPI communication pattern is provided by
  //! p4est, and used inside API p4est_transfer_fixed
  //! \note load balancing is also called re-partitioning
  //!
  //! \param[in] userdata_old a DataArrayBlock_t (device Kokkos::View) before re-partition
  //! \param[in,out] userdata_new a DataArrayBlock_t (device Kokkos::View) after re-partition,
  //! userdata_new will be properly resized
  //!
  //! Implementation note: use old and new gfq, and p4est_transfer_fixed to re-partition user data
  template <typename T>
  void
  repartition_userdata(DataArrayBlock_t<T> userdata_old, DataArrayBlock_t<T> & userdata_new)
  {
    Kokkos::Profiling::ScopedRegion myprof(
      "MeshPartitioner::repartition_userdata - cell-center block");

    // if there is only one MPI rank, nothing to do
    if (m_par_env.nRanks() < 2)
      return;

    // resize userdata_new (actually just reallocate memory without initializing)
    // so that it can at least old data for the new partition
    // if (userdata_new.extent(2) < static_cast<size_t>(m_local_num_quadrants_new))
    {
      userdata_new.resize(static_cast<size_t>(m_local_num_quadrants_new));
    }

    // size in bytes of data per octant
    const size_t datasize_per_octant_bytes =
      static_cast<size_t>(userdata_old.num_cells() * userdata_old.num_vars()) * sizeof(T);

    // \note userdata_old is a Kokkos::View with Left memory layout, so that all data attached to
    // a octant is memory contiguous, so we don't need any auxiliary array, userdata_old can be
    // directly used in the p4est_transfer_fixed API \note here we strongly assume the MPI
    // implementation to be CUDA aware (i.e. when device_t is Kokkos::CUDA); this is a requirement
    // tested by top-level cmake
    // TODO: if the MPI implementation is not cuda-aware (we need to first transfer data on host,
    // before calling p4est_transfer_fixed); this maybe useful for debug on a platform where
    // cuda-aware MPI is not available (even though, it is always possible to recompile locally
    // OpenMPI from source with cuda-awareness)
    // p4est_t::transfer_fixed(m_lb_gfq_new.data(),
    //                         m_lb_gfq_old.data(),
    //                         m_par_env.mpi_comm(),
    //                         KALYPSSO_COMM_MESH_PARTITIONER_TAG,
    //                         userdata_new.data(), // after re-partition
    //                         userdata_old.data(), // before re-partition
    //                         datasize_per_octant_bytes);
    MeshPartitioner_helper<dim, device_t>::transfer_fixed(
      m_lb_gfq_new.data(),
      m_lb_gfq_old.data(),
      m_par_env.mpi_comm(),
      KALYPSSO_COMM_MESH_PARTITIONER_TAG,
      userdata_new.data(), // after re-partition
      userdata_old.data(), // before re-partition
      datasize_per_octant_bytes);

  } // repartition_userdata - cell-center block data

  // =======================================================
  // =======================================================
  //! face-center user data load balancing (repartitioning) with MPI data communication.
  //!
  //! MPI communication pattern is provided by
  //! p4est, and used inside API p4est_transfer_fixed
  //! \note load balancing is also called re-partitioning
  //!
  //! \param[in] userdata_old a FaceDataArrayBlock_t (device Kokkos::View) before re-partition
  //! \param[in,out] userdata_new a FaceDataArrayBlock_t (device Kokkos::View) after re-partition,
  //! userdata_new will be properly resized
  //!
  //! Implementation note: use old and new gfq, and p4est_transfer_fixed to re-partition user data
  template <typename T>
  void
  repartition_userdata(FaceDataArrayBlock_t<T> userdata_old, FaceDataArrayBlock_t<T> & userdata_new)
  {
    Kokkos::Profiling::ScopedRegion myprof(
      "MeshPartitioner::repartition_userdata - face-center block");

    // if there is only one MPI rank, nothing to do
    if (m_par_env.nRanks() < 2)
      return;

    // resize userdata_new (actually just reallocate memory without initializing)
    // so that it can at least old data for the new partition
    userdata_new.resize(static_cast<size_t>(m_local_num_quadrants_new));

    // size in bytes of data per octant
    const size_t datasize_per_octant_bytes =
      static_cast<size_t>(userdata_old.num_elements_per_octant()) * sizeof(T);

    // \note userdata_old is a FaceDataArrayBlock which uses sort of left memory layout, so that all
    // data attached to a octant is memory contiguous, so we don't need any auxiliary array,
    // userdata_old can be directly used in the p4est_transfer_fixed API.
    //
    // \note here we STRONGLY assume the MPI implementation to be CUDA aware
    // (i.e. when device_t is Kokkos::CUDA); this is a requirement tested by top-level cmake
    // TODO: if the MPI implementation is not cuda-aware (we need to first transfer data on host,
    // before calling p4est_transfer_fixed); this maybe useful for debug on a platform where
    // cuda-aware MPI is not available (even though, it is always possible to recompile locally
    // OpenMPI from source with cuda-awareness)
    MeshPartitioner_helper<dim, device_t>::transfer_fixed(
      m_lb_gfq_new.data(),
      m_lb_gfq_old.data(),
      m_par_env.mpi_comm(),
      KALYPSSO_COMM_MESH_PARTITIONER_TAG,
      userdata_new.data(), // after re-partition
      userdata_old.data(), // before re-partition
      datasize_per_octant_bytes);

  } // repartition_userdata - face-center block data

  // =======================================================
  // =======================================================
  //! user data load balancing (repartitioning) with MPI data communication.
  //!
  //! MPI communication pattern is provided by
  //! p4est, and used inside API p4est_transfer_fixed
  //! \note load balancing is also called re-partitioning
  //!
  //! \param[in] userdata_old a DataArrayGhostedBlock_t (device Kokkos::View) before re-partition
  //! \param[in,out] userdata_new a DataArrayGhostedBlock_t (device Kokkos::View) after
  //! re-partition, userdata_new will be properly resized
  //!
  //! Implementation note: use old and new gfq, and p4est_transfer_fixed to re-partition user data
  //!
  //! Important note: we actually transfer the entire block (ghost included)
  //! if this is a performance problem, we will refactor and transfer only the inner part (by
  //! removing the ghost cells.)
  template <typename T>
  void
  repartition_userdata(DataArrayGhostedBlock_t<T>   userdata_old,
                       DataArrayGhostedBlock_t<T> & userdata_new)
  {

    repartition_userdata(userdata_old.view(), userdata_new.view_ref());

  } // repartition_userdata - ghosted block

  // =======================================================
  // =======================================================
  //! user data load balancing (repartitioning) with MPI data communication.
  //!
  //! MPI communication pattern is provided by
  //! p4est, and used inside API p4est_transfer_fixed
  //! \note load balancing is also called re-partitioning
  //!
  //! \param[in] userdata_old a DataArrayBlock_t (device Kokkos::View) before re-partition
  //! \param[in,out] userdata_new a DataArrayBlock_t (device Kokkos::View) after re-partition,
  //! userdata_new will be properly resized
  //!
  //! Implementation note: use old and new gfq, and p4est_transfer_fixed to re-partition user data
  template <typename T>
  void
  repartition_userdata(DataArrayLeafAoS_t<T> userdata_old, DataArrayLeafAoS_t<T> & userdata_new)
  {
    Kokkos::Profiling::ScopedRegion myprof("MeshPartitioner::repartition_userdata - leaf");

    // if there is only one MPI rank, nothing to do
    if (m_par_env.nRanks() < 2)
      return;

    // resize userdata_new (actually just reallocate memory without initializing)
    // so that it can at least old data for the new partition
    if (userdata_new.extent(0) < static_cast<size_t>(m_local_num_quadrants_new))
    {
      Kokkos::realloc(Kokkos::view_alloc(Kokkos::WithoutInitializing),
                      userdata_new,
                      static_cast<size_t>(m_local_num_quadrants_new),
                      userdata_old.extent(1));
    }

    // size in bytes of data per octant
    const auto datasize_per_octant_bytes = userdata_old.extent(1) * sizeof(T);

    // \note userdata_old is a Kokkos::View with Left memory layout, so that all data attached to
    // a octant is memory contiguous, so we don't need any auxiliary array, userdata_old can be
    // directly used in the p4est_transfer_fixed API \note here we strongly assume the MPI
    // implementation to be CUDA aware (i.e. when device_t is Kokkos::CUDA); this is a requirement
    // tested by top-level cmake
    // TODO: if the MPI implementation is not cuda-aware (we need to first transfer data on host,
    // before calling p4est_transfer_fixed); this maybe useful for debug on a platform where
    // cuda-aware MPI is not available (even though, it is always possible to recompile locally
    // OpenMPI from source with cuda-awareness)
    p4est_t::transfer_fixed(m_lb_gfq_new.data(),
                            m_lb_gfq_old.data(),
                            m_par_env.mpi_comm(),
                            KALYPSSO_COMM_MESH_PARTITIONER_TAG,
                            userdata_new.data(), // after re-partition
                            userdata_old.data(), // before re-partition
                            datasize_per_octant_bytes);

  } // repartition_userdata

  // =======================================================
  // =======================================================
  //! user data load balancing (repartitioning) with MPI data communication.
  //!
  //! MPI communication pattern is provided by
  //! p4est, and used inside API p4est_transfer_fixed
  //! \note load balancing is also called re-partitioning
  //!
  //! \param[in] userdata_old a MaterialPresenceView_t before re-partition
  //! \param[in,out] userdata_new a MaterialPresenceView_t after re-partition, userdata_new will be
  //!                properly resized
  //!
  //! Implementation note: use old and new gfq, and p4est_transfer_fixed to re-partition user data
  //! Because a MaterialPresenceView_t is simply a collection of integers for each octant, we can
  //! easily reuse the already created transfer functions.
  void
  repartition_userdata(MaterialPresenceView_t userdata_old, MaterialPresenceView_t & userdata_new)
  {
    Kokkos::Profiling::ScopedRegion myprof(
      "MeshPartitioner::repartition_userdata - material presence");

    // if there is only one MPI rank, nothing to do
    if (m_par_env.nRanks() < 2)
      return;

    // resize userdata_new so that it can at least old data for the new partition
    userdata_new.resize(m_local_num_quadrants_new);

    // size in bytes of data per octant
    const size_t datasize_per_octant_bytes =
      static_cast<size_t>(userdata_old.block_length_per_octant()) *
      sizeof(typename MaterialPresenceView_t::BitBlock_t);

    // \note userdata_old is a Kokkos::View with Left memory layout, so that all data attached to
    // a octant is memory contiguous, so we don't need any auxiliary array, userdata_old can be
    // directly used in the p4est_transfer_fixed API \note here we strongly assume the MPI
    // implementation to be CUDA aware (i.e. when device_t is Kokkos::CUDA); this is a requirement
    // tested by top-level cmake
    // TODO: if the MPI implementation is not cuda-aware (we need to first transfer data on host,
    // before calling p4est_transfer_fixed); this maybe useful for debug on a platform where
    // cuda-aware MPI is not available (even though, it is always possible to recompile locally
    // OpenMPI from source with cuda-awareness)
    MeshPartitioner_helper<dim, device_t>::transfer_fixed(
      m_lb_gfq_new.data(),
      m_lb_gfq_old.data(),
      m_par_env.mpi_comm(),
      KALYPSSO_COMM_MESH_PARTITIONER_TAG,
      userdata_new.data(), // after re-partition
      userdata_old.data(), // before re-partition
      datasize_per_octant_bytes);
  } // repartition_userdata

  // =======================================================
  // =======================================================
  //! cell-center user data load balancing (repartitioning) with MPI data communication.
  //!
  //! MPI communication pattern is provided by
  //! p4est, and used inside API p4est_transfer_fixed
  //! \note load balancing is also called re-partitioning
  //!
  //! \param[in] userdata_old a DataArrayBlockMultiVar_t (device Kokkos::View) before re-partition
  //! \param[in,out] userdata_new a DataArrayBlockMultiVar_t (device Kokkos::View) after
  //! re-partition, is is supposed to be properly sized
  //!
  //! Implementation note: use old and new gfq, and p4est_transfer_fixed to re-partition user data
  template <typename T>
  void
  repartition_userdata(DataArrayBlockMultiVar_t<T>   userdata_old,
                       DataArrayBlockMultiVar_t<T> & userdata_new)
  {
    Kokkos::Profiling::ScopedRegion myprof(
      "MeshPartitioner::repartition_userdata - cell-center block");

    // if there is only one MPI rank, nothing to do
    if (m_par_env.nRanks() < 2)
      return;

    // size in bytes of data per block
    const size_t datasize_per_block_bytes =
      static_cast<size_t>(userdata_old.num_cells()) * sizeof(T);

    // using HostSpace = typename HostDevice::execution_space;
    const auto old_offsets =
      DataArrayBlockMultiVar_t<T>::Offsets_t::create_host_mirror_view_and_copy(
        userdata_old.offsets());
    const auto new_offsets =
      DataArrayBlockMultiVar_t<T>::Offsets_t::create_host_mirror_view_and_copy(
        userdata_new.offsets());

    MeshPartitioner_helper<dim, device_t>::transfer_fixed_multi_var(
      m_lb_gfq_new.data(),
      m_lb_gfq_old.data(),
      m_par_env.mpi_comm(),
      KALYPSSO_COMM_MESH_PARTITIONER_TAG,
      userdata_new.storage().data(), // after re-partition
      userdata_old.storage().data(), // before re-partition
      datasize_per_block_bytes,
      old_offsets.data(),
      new_offsets.data());

  } // repartition_userdata - cell-center block data

  // =======================================================
  // =======================================================
  //! user data load balancing (repartitioning) with MPI data communication.
  //!
  //! MPI communication pattern is provided by
  //! p4est, and used inside API p4est_transfer_fixed
  //! \note load balancing is also called re-partitioning
  //!
  //! \param[in] userdata_old a DataArrayBlock_t (device Kokkos::View) before re-partition
  //! \param[in,out] userdata_new a DataArrayBlock_t (device Kokkos::View) after re-partition,
  //! userdata_new will be properly resized
  //!
  //! Implementation note: use old and new gfq, and p4est_transfer_fixed to re-partition user data
  template <typename T>
  void
  repartition_userdata(DataArrayLeafSoA_t<T> userdata_old, DataArrayLeafSoA_t<T> & userdata_new)
  {
    Kokkos::Profiling::ScopedRegion myprof("MeshPartitioner::repartition_userdata - leaf");

    // if there is only one MPI rank, nothing to do
    if (m_par_env.nRanks() < 2)
      return;

    if constexpr (0 == 1)
    {
      // currently DataArrayLeaf_t use's a layout where the octant id is the first index (SoA).
      // to be able to transfer data:
      // - we either need to transpose data (this is a question of AoS versus SoA)
      // - or perform transfer variable per variable
      // TODO

      // resize userdata_new (actually just reallocate memory without initializing)
      // so that it can at least old data for the new partition
      if (userdata_new.extent(0) < static_cast<size_t>(m_local_num_quadrants_new))
      {
        Kokkos::realloc(Kokkos::view_alloc(Kokkos::WithoutInitializing),
                        userdata_new,
                        static_cast<size_t>(m_local_num_quadrants_new),
                        userdata_old.extent(1));
      }

      // size in bytes of data per octant
      const auto datasize_per_octant_bytes = userdata_old.extent(1) * sizeof(T);

      // \note userdata_old is a Kokkos::View with Left memory layout, so that all data attached to
      // a octant is memory contiguous, so we don't need any auxiliary array, userdata_old can be
      // directly used in the p4est_transfer_fixed API \note here we strongly assume the MPI
      // implementation to be CUDA aware (i.e. when device_t is Kokkos::CUDA); this is a requirement
      // tested by top-level cmake
      // TODO: if the MPI implementation is not cuda-aware (we need to first transfer data on host,
      // before calling p4est_transfer_fixed); this maybe useful for debug on a platform where
      // cuda-aware MPI is not available (even though, it is always possible to recompile locally
      // OpenMPI from source with cuda-awareness)
      p4est_t::transfer_fixed(m_lb_gfq_new.data(),
                              m_lb_gfq_old.data(),
                              m_par_env.mpi_comm(),
                              KALYPSSO_COMM_MESH_PARTITIONER_TAG,
                              userdata_new.data(), // after re-partition
                              userdata_old.data(), // before re-partition
                              datasize_per_octant_bytes);
    }
    else
    {
      KALYPSSO_ERROR("Please implement me");
    }
  } // repartition_userdata

private:
  //! config map (input parameter)
  const ConfigMap & m_config_map;

  //! parallel environment
  const ParallelEnv & m_par_env;

  /*
   * Data array used for MPI load balancing / mesh partitioning
   */

  //! array of global index to the first quadrant of each MPI process, before re-partitioning
  std::vector<p4est_gloidx_t> m_lb_gfq_old;

  //! array of global index to the first quadrant of each MPI process, after re-partitioning
  std::vector<p4est_gloidx_t> m_lb_gfq_new;

  //! local (current MPI process) number of quadrants
  int32_t m_local_num_quadrants_new;

}; // class MeshPartitioner

template <size_t dim, typename device_t>
typename MeshPartitioner<dim, device_t>::WeightsCallback::Data *
  MeshPartitioner<dim, device_t>::WeightsCallback::data = nullptr;

} // namespace kalypsso

#endif // KALYPSSO_CORE_MESHPARTITIONER_H_
