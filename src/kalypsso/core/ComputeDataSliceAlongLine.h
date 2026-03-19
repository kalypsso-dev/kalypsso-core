// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeDataSliceAlongLine.h
 *
 * \brief Contains the functor that extracts a data slice along a line.
 *
 * This functor is able to slice a DataArrayBlock along a line across domain decomposition
 * and save data in a file using Numpy file format. This is currently limited to lines which
 * direction is axis aligned.
 *
 * \todo refactor for supporting non-axis aligned slice (even more complex shape defined by
 * equation of type f(x,y,z)=0)
 *
 * \note The functionality is similar to Paraview filter "slice".
 */

#ifndef KALYPSSO_CORE_COMPUTE_DATA_SLICE_ALONG_LINE_H_
#define KALYPSSO_CORE_COMPUTE_DATA_SLICE_ALONG_LINE_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kalypsso_macros.h>
#include <kalypsso/core/kalypsso_data_container.h>
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <vector>
#include <string>

namespace kalypsso
{

namespace core
{

template <size_t dim, typename device_t>
class ComputeDataSliceAlongLine
{
public:
  //! Array of Orchard keys
  using OrchardKeys = typename orchard_key_base_t<device_t>::view_t;

  //! Tag used in marking cells
  struct TagMarkCells
  {};

  //! Tag used to write in final array of cell-centered data
  struct TagWriteCellData
  {};

  //! Tag used to write in final array of face-centered data
  struct TagWriteFaceData
  {};

  /**
   * \brief Computes a slice along a line and saves it in cnpy format.
   *
   * \param data Data array to slice.
   * \param start_octant The first owned octant.
   * \param end_octant The last owned octant, excluded.
   * \param start_point The segment's start.
   * \param end_point The segments' end.
   * \param keys Orchard keys.
   * \param fm Hydro variables to index mapper.
   * \param var Variable to save.
   * \param file_prefix Files prefix.
   * \param parallel_env Parallel environment.
   * \param config_map Inputted config map.
   */
  static void
  apply(const DataArrayBlock<dim, real_t, device_t> & data,
        const int32_t                                 start_octant,
        const int32_t                                 end_octant,
        const Kokkos::Array<real_t, dim>              start_point,
        const Kokkos::Array<real_t, dim>              end_point,
        const OrchardKeys &                           keys,
        const int32_t                                 var,
        const std::string &                           file_prefix,
        const ParallelEnv &                           parallel_env,
        const ConfigMap &                             config_map);

  /**
   * \brief Computes a slice along a line and saves it in cnpy format.
   *
   * Same as above, but the start and end point are automatically determined to be the middle box
   * axis along a given direction.
   *
   * \param data Data array to slice.
   * \param start_octant The first owned octant.
   * \param end_octant The last owned octant, excluded.
   * \param direction The line direction.
   * \param keys Orchard keys.
   * \param fm Hydro variables to index mapper.
   * \param var Variable to save.
   * \param file_prefix Files prefix.
   * \param parallel_env Parallel environment.
   * \param config_map Inputted config map.
   */
  static void
  apply(const DataArrayBlock<dim, real_t, device_t> & data,
        const int32_t                                 start_octant,
        const int32_t                                 end_octant,
        const int32_t                                 direction,
        const OrchardKeys &                           keys,
        const int32_t                                 var,
        const std::string &                           file_prefix,
        const ParallelEnv &                           parallel_env,
        const ConfigMap &                             config_map);

  /**
   * \brief Computes a slice along a line and saves it in cnpy format.
   *
   * Same as above, but writes multiple scalar values.
   *
   * \param data Data array to slice.
   * \param start_octant The first owned octant.
   * \param end_octant The last owned octant, excluded.
   * \param direction The line direction.
   * \param keys Orchard keys.
   * \param fm Hydro variables to index mapper.
   * \param vars Vector of variables to save.
   * \param var_names Vector of variables names.
   * \param file_prefix Files prefix.
   * \param parallel_env Parallel environment.
   * \param config_map Inputted config map.
   */
  static void
  apply(const DataArrayBlock<dim, real_t, device_t> & data,
        const int32_t                                 start_octant,
        const int32_t                                 end_octant,
        const int32_t                                 direction,
        const OrchardKeys &                           keys,
        const std::vector<int32_t>                    vars,
        const std::vector<std::string>                var_names,
        const std::string &                           file_prefix,
        const ParallelEnv &                           parallel_env,
        const ConfigMap &                             config_map);

  /**
   * \brief Computes a slice along a line and saves it in cnpy format.
   *
   * Same as above, but writes multiple scalar values.
   *
   * \param data_cell Cell-centered data array to slice.
   * \param data_face Face-centered data array to slice.
   * \param start_octant The first owned octant.
   * \param end_octant The last owned octant, excluded.
   * \param direction The line direction.
   * \param keys Orchard keys.
   * \param fm Hydro variables to index mapper.
   * \param cell_vars Vector of cell-centered variables to save.
   * \param cell_var_names Vector of cell-centered variables names.
   * \param face_vars Vector of face-centered variables to save.
   * \param face_var_names Vector of face-centered variables names.
   * \param file_prefix Files prefix.
   * \param parallel_env Parallel environment.
   * \param config_map Inputted config map.
   */
  static void
  apply(const DataArrayBlock<dim, real_t, device_t> &     data_cell,
        const FaceDataArrayBlock<dim, real_t, device_t> & data_face,
        const int32_t                                     start_octant,
        const int32_t                                     end_octant,
        const int32_t                                     direction,
        const OrchardKeys &                               keys,
        const std::vector<int32_t>                        cell_vars,
        const std::vector<std::string>                    cell_var_names,
        const std::vector<int32_t>                        face_vars,
        const std::vector<std::string>                    face_var_names,
        const std::string &                               file_prefix,
        const ParallelEnv &                               parallel_env,
        const ConfigMap &                                 config_map);

  static Kokkos::Array<real_t, dim>
  get_mid_box_start_point(const ConfigMap & config_map, const int32_t direction);

  static Kokkos::Array<real_t, dim>
  get_mid_box_end_point(const ConfigMap & config_map, const int32_t direction);

  void
  set_var(int32_t var)
  {
    m_var = var;
  }

  /**
   * \brief Kokkos kernel that will mark cells the segment crosses.
   *
   * \param i_global The global index of the cell.
   */
  KOKKOS_FUNCTION void
  operator()(const TagMarkCells, const int32_t & i_global) const;

  /**
   * \brief Kokkos kernel that will write cell-centered data into the final array.
   *
   * \param i_global The global index of the cell.
   */
  KOKKOS_FUNCTION void
  operator()(const TagWriteCellData, const int32_t & i_global) const;

  /**
   * \brief Kokkos kernel that will write face-centered data into the final array.
   *
   * \param i_global The global index of the cell.
   */
  KOKKOS_FUNCTION void
  operator()(const TagWriteFaceData, const int32_t & i_global) const;

private:
  //! Kokkos execution space
  using ExecutionSpace = typename device_t::execution_space;

  /**
   * \brief Create and fills in internal data used for the computation.
   *
   * \param data Data array to slice.
   * \param keys Orchard keys.
   * \param var Variable to save.
   * \param start_index The first owned cell.
   * \param end_index The last owned cell, excluded.
   * \param start_point The segment's start.
   * \param end_point The segments' end.
   * \param config_map Inputted config map.
   */
  ComputeDataSliceAlongLine(const DataArrayBlock<dim, real_t, device_t> & data,
                            const OrchardKeys &                           keys,
                            const int32_t                                 var,
                            const int32_t                                 start_index,
                            const int32_t                                 end_index,
                            const Kokkos::Array<real_t, dim>              start_point,
                            const Kokkos::Array<real_t, dim>              end_point,
                            const ConfigMap &                             config_map);

  /**
   * \brief Create and fills in internal data used for the computation.
   *
   * \note face-data will be converted into cell centered values.
   *
   * \param data_cell Cell-centered data array to slice.
   * \param data_face Face-centered data array to slice.
   * \param keys Orchard keys.
   * \param var Variable to save.
   * \param start_index The first owned cell.
   * \param end_index The last owned cell, excluded.
   * \param start_point The segment's start.
   * \param end_point The segments' end.
   * \param config_map Inputted config map.
   */
  ComputeDataSliceAlongLine(const DataArrayBlock<dim, real_t, device_t> &     data_cell,
                            const FaceDataArrayBlock<dim, real_t, device_t> & data_face,
                            const OrchardKeys &                               keys,
                            const int32_t                                     var,
                            const int32_t                                     start_index,
                            const int32_t                                     end_index,
                            const Kokkos::Array<real_t, dim>                  start_point,
                            const Kokkos::Array<real_t, dim>                  end_point,
                            const ConfigMap &                                 config_map);

  //! Cell-centered data array to slice
  DataArrayBlock<dim, real_t, device_t> m_data_cell;

  //! Face-centered data array to slice (optional)
  FaceDataArrayBlock<dim, real_t, device_t> m_data_face;

  //! The Orchard keys
  OrchardKeys m_keys;

  //! Tree scaling factor
  real_t m_scaling_factor;

  //! Variable id
  int32_t m_var;

  //! Minimum corner of the mesh in real space
  Kokkos::Array<real_t, dim> m_corner;

  //! Start octant
  int32_t m_start_index;

  //! Offsets for output
  Kokkos::View<int32_t *, device_t> m_offsets;

  //! Positions
  Kokkos::View<real_t *, device_t> m_positions;

  //! AMR data
  Kokkos::View<uint8_t *, device_t> m_amr;

  //! sliced data
  Kokkos::View<real_t *, device_t> m_data_slice;

  //! slice axis
  ComponentIndex3D m_dir;

  //! Start point
  Kokkos::Array<real_t, dim> m_start;

  //! End point
  Kokkos::Array<real_t, dim> m_end;
};

extern template class ComputeDataSliceAlongLine<2, kalypsso::DefaultDevice>;
extern template class ComputeDataSliceAlongLine<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_COMPUTE_DATA_SLICE_ALONG_LINE_H_
