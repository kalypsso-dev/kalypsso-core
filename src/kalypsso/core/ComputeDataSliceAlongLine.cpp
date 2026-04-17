// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeDataSliceAlongLine.cpp
 *
 * \brief Contains the definition of ComputeDataSliceAlongLine.
 */

#include <kalypsso/core/ComputeDataSliceAlongLine.h>

#include <kalypsso/core/cnpy_io.h>
#include <kalypsso/core/orchard_key_utils.h>

namespace kalypsso
{

namespace core
{

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
ComputeDataSliceAlongLine<dim, device_t>::ComputeDataSliceAlongLine(
  const DataArrayBlock<dim, real_t, device_t> & data,
  const OrchardKeys &                           keys,
  const int32_t                                 var,
  const int32_t                                 start_index,
  const int32_t                                 end_index,
  const Kokkos::Array<real_t, dim>              start_point,
  const Kokkos::Array<real_t, dim>              end_point,
  const ConfigMap &                             config_map)
  : m_data_cell(data)
  , m_keys(keys)
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_var(var)
  , m_corner(get_xyz_min<dim>(config_map))
  , m_start_index(start_index)
  , m_offsets("Offsets", static_cast<size_t>(end_index - start_index))
  , m_positions("Position", 0)
  , m_amr("AMR", 0)
  , m_data_slice("Data", 0)
  , m_start(start_point)
  , m_end(end_point)
{
  bool on_dir[dim];
  on_dir[IX] = !ISFUZZYNULL(start_point[IX] - end_point[IX]);
  on_dir[IY] = !ISFUZZYNULL(start_point[IY] - end_point[IY]);
  if constexpr (dim == 3)
    on_dir[IZ] = !ISFUZZYNULL(start_point[IZ] - end_point[IZ]);

  if constexpr (dim == 2)
  {
    assertm(((on_dir[IX] + on_dir[IY]) == 1), "Points must be on a single axis");
  }
  else if constexpr (dim == 3)
  {
    assertm(((on_dir[IX] + on_dir[IY] + on_dir[IZ]) == 1), "Points must be on a single axis");
  }

  for (uint8_t dir = IX; dir < dim; dir++)
    m_dir = on_dir[dir] ? ComponentIndex3D(dir) : m_dir;
} // ComputeDataSliceAlongLine<dim, device_t>::ComputeDataSliceAlongLine

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
ComputeDataSliceAlongLine<dim, device_t>::ComputeDataSliceAlongLine(
  const DataArrayBlock<dim, real_t, device_t> &     data_cell,
  const FaceDataArrayBlock<dim, real_t, device_t> & data_face,
  const OrchardKeys &                               keys,
  const int32_t                                     var,
  const int32_t                                     start_index,
  const int32_t                                     end_index,
  const Kokkos::Array<real_t, dim>                  start_point,
  const Kokkos::Array<real_t, dim>                  end_point,
  const ConfigMap &                                 config_map)
  : m_data_cell(data_cell)
  , m_data_face(data_face)
  , m_keys(keys)
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_var(var)
  , m_corner(get_xyz_min<dim>(config_map))
  , m_start_index(start_index)
  , m_offsets("Offsets", static_cast<size_t>(end_index - start_index))
  , m_positions("Position", 0)
  , m_amr("AMR", 0)
  , m_data_slice("Data", 0)
  , m_start(start_point)
  , m_end(end_point)
{
  bool on_dir[dim];
  on_dir[IX] = !ISFUZZYNULL(start_point[IX] - end_point[IX]);
  on_dir[IY] = !ISFUZZYNULL(start_point[IY] - end_point[IY]);
  if constexpr (dim == 3)
    on_dir[IZ] = !ISFUZZYNULL(start_point[IZ] - end_point[IZ]);

  if constexpr (dim == 2)
  {
    assertm(((on_dir[IX] + on_dir[IY]) == 1), "Points must be on a single axis");
  }
  else if constexpr (dim == 3)
  {
    assertm(((on_dir[IX] + on_dir[IY] + on_dir[IZ]) == 1), "Points must be on a single axis");
  }

  for (uint8_t dir = IX; dir < dim; dir++)
    m_dir = on_dir[dir] ? ComponentIndex3D(dir) : m_dir;
} // ComputeDataSliceAlongLine<dim, device_t>::ComputeDataSliceAlongLine

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
ComputeDataSliceAlongLine<dim, device_t>::apply(const DataArrayBlock<dim, real_t, device_t> & data,
                                                const int32_t                    start_octant,
                                                const int32_t                    end_octant,
                                                const Kokkos::Array<real_t, dim> start_point,
                                                const Kokkos::Array<real_t, dim> end_point,
                                                const OrchardKeys &              keys,
                                                const int32_t                    var,
                                                const std::string &              file_prefix,
                                                const ParallelEnv &              parallel_env,
                                                const ConfigMap &                config_map)
{
  const auto                               nb_cells = data.num_cells();
  const auto                               start_index = start_octant * nb_cells;
  const auto                               end_index = end_octant * nb_cells;
  ComputeDataSliceAlongLine<dim, device_t> functor(
    data, keys, var, start_index, end_index, start_point, end_point, config_map);

  Kokkos::RangePolicy<ExecutionSpace, TagMarkCells> policy0(start_index, end_index);
  Kokkos::parallel_for("kalypsso::core::ComputeDataSliceAlongLine::MarkCells", policy0, functor);

  Kokkos::RangePolicy<ExecutionSpace> policy1(0, end_index - start_index);
  int32_t                             total_num_cells = 0;
  Kokkos::parallel_scan(
    "kalypsso::core::ComputeDataSliceAlongLine::Offsets",
    policy1,
    KOKKOS_LAMBDA(const int32_t i, int32_t & offset, bool is_final) {
      if (functor.m_offsets(i) == -1)
        return;

      if (is_final)
        functor.m_offsets(i) = offset;
      offset += 1;
    },
    total_num_cells);

  Kokkos::realloc(functor.m_positions, static_cast<size_t>(total_num_cells));
  Kokkos::realloc(functor.m_amr, static_cast<size_t>(total_num_cells));
  Kokkos::realloc(functor.m_data_slice, static_cast<size_t>(total_num_cells));

  Kokkos::RangePolicy<ExecutionSpace, TagWriteCellData> policy2(start_index, end_index);
  Kokkos::parallel_for("kalypsso::core::ComputeDataSliceAlongLine::WriteData", policy2, functor);

  auto positions_host =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, functor.m_positions);
  auto amr_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, functor.m_amr);
  auto data_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, functor.m_data_slice);

  if (parallel_env.nRanks() == 1)
  {
    save_cnpy(positions_host, file_prefix + "_positions");
    save_cnpy(amr_host, file_prefix + "_level");
    save_cnpy(data_host, file_prefix + "_data");
  }
  else
  {
    const auto rank = std::to_string(parallel_env.rank());
    save_cnpy(positions_host, file_prefix + "_" + rank + "_positions");
    save_cnpy(amr_host, file_prefix + "_" + rank + "_level");
    save_cnpy(data_host, file_prefix + "_" + rank + "_data");
  }
} // ComputeDataSliceAlongLine<dim, device_t>::apply

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
ComputeDataSliceAlongLine<dim, device_t>::apply(const DataArrayBlock<dim, real_t, device_t> & data,
                                                const int32_t                    start_octant,
                                                const int32_t                    end_octant,
                                                const Kokkos::Array<real_t, dim> start_point,
                                                const Kokkos::Array<real_t, dim> end_point,
                                                const OrchardKeys &              keys,
                                                const std::vector<int32_t>       vars,
                                                const std::vector<std::string>   var_names,
                                                const std::string &              file_prefix,
                                                const ParallelEnv &              parallel_env,
                                                const ConfigMap &                config_map)
{
  const auto                               nb_cells = data.num_cells();
  const auto                               start_index = start_octant * nb_cells;
  const auto                               end_index = end_octant * nb_cells;
  ComputeDataSliceAlongLine<dim, device_t> functor(
    data, keys, vars[0], start_index, end_index, start_point, end_point, config_map);

  Kokkos::RangePolicy<ExecutionSpace, TagMarkCells> policy0(start_index, end_index);
  Kokkos::parallel_for("kalypsso::core::ComputeDataSliceAlongLine::MarkCells", policy0, functor);

  Kokkos::RangePolicy<ExecutionSpace> policy1(0, end_index - start_index);
  int32_t                             total_num_cells = 0;
  Kokkos::parallel_scan(
    "kalypsso::core::ComputeDataSliceAlongLine::Offsets",
    policy1,
    KOKKOS_LAMBDA(const int32_t i, int32_t & offset, bool is_final) {
      if (functor.m_offsets(i) == -1)
        return;

      if (is_final)
        functor.m_offsets(i) = offset;
      offset += 1;
    },
    total_num_cells);

  Kokkos::realloc(functor.m_positions, static_cast<size_t>(total_num_cells));
  Kokkos::realloc(functor.m_amr, static_cast<size_t>(total_num_cells));
  Kokkos::realloc(functor.m_data_slice, static_cast<size_t>(total_num_cells));

  for (size_t iv = 0; iv < vars.size(); ++iv)
  {

    functor.m_var = vars[iv];

    Kokkos::RangePolicy<ExecutionSpace, TagWriteCellData> policy2(start_index, end_index);
    Kokkos::parallel_for("kalypsso::core::ComputeDataSliceAlongLine::WriteData", policy2, functor);

    auto positions_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, functor.m_positions);
    auto amr_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, functor.m_amr);
    auto data_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, functor.m_data_slice);

    if (parallel_env.nRanks() == 1)
    {
      if (iv == 0)
      {
        save_cnpy(positions_host, file_prefix + "_positions");
        save_cnpy(amr_host, file_prefix + "_level");
      }
      save_cnpy(data_host, file_prefix + "_" + var_names[iv]);
    }
    else
    {
      const auto rank = std::to_string(parallel_env.rank());
      if (iv == 0)
      {
        save_cnpy(positions_host, file_prefix + "_" + rank + "_positions");
        save_cnpy(amr_host, file_prefix + "_" + rank + "_level");
      }
      save_cnpy(data_host, file_prefix + "_" + rank + "_" + var_names[iv]);
    }
  }
} // ComputeDataSliceAlongLine<dim, device_t>::apply

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
ComputeDataSliceAlongLine<dim, device_t>::apply(const DataArrayBlock<dim, real_t, device_t> & data,
                                                const int32_t       start_octant,
                                                const int32_t       end_octant,
                                                const int32_t       direction,
                                                const OrchardKeys & keys,
                                                const int32_t       var,
                                                const std::string & file_prefix,
                                                const ParallelEnv & parallel_env,
                                                const ConfigMap &   config_map)
{
  const Kokkos::Array<real_t, dim> start_point = get_mid_box_start_point(config_map, direction);
  const Kokkos::Array<real_t, dim> end_point = get_mid_box_end_point(config_map, direction);

  ComputeDataSliceAlongLine<dim, device_t>::apply(data,
                                                  start_octant,
                                                  end_octant,
                                                  start_point,
                                                  end_point,
                                                  keys,
                                                  var,
                                                  file_prefix,
                                                  parallel_env,
                                                  config_map);

} // ComputeDataSliceAlongLine<dim, device_t>::apply

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
ComputeDataSliceAlongLine<dim, device_t>::apply(const DataArrayBlock<dim, real_t, device_t> & data,
                                                const int32_t                  start_octant,
                                                const int32_t                  end_octant,
                                                const int32_t                  direction,
                                                const OrchardKeys &            keys,
                                                const std::vector<int32_t>     vars,
                                                const std::vector<std::string> var_names,
                                                const std::string &            file_prefix,
                                                const ParallelEnv &            parallel_env,
                                                const ConfigMap &              config_map)
{
  const Kokkos::Array<real_t, dim> start_point = get_mid_box_start_point(config_map, direction);
  const Kokkos::Array<real_t, dim> end_point = get_mid_box_end_point(config_map, direction);

  ComputeDataSliceAlongLine<dim, device_t>::apply(data,
                                                  start_octant,
                                                  end_octant,
                                                  start_point,
                                                  end_point,
                                                  keys,
                                                  vars,
                                                  var_names,
                                                  file_prefix,
                                                  parallel_env,
                                                  config_map);

} // ComputeDataSliceAlongLine<dim, device_t>::apply

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
ComputeDataSliceAlongLine<dim, device_t>::apply(
  const DataArrayBlock<dim, real_t, device_t> &     data_cell,
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
  const ConfigMap &                                 config_map)
{
  const Kokkos::Array<real_t, dim> start_point = get_mid_box_start_point(config_map, direction);
  const Kokkos::Array<real_t, dim> end_point = get_mid_box_end_point(config_map, direction);

  const auto                               nb_cells = data_cell.num_cells();
  const auto                               start_index = start_octant * nb_cells;
  const auto                               end_index = end_octant * nb_cells;
  ComputeDataSliceAlongLine<dim, device_t> functor(data_cell,
                                                   data_face,
                                                   keys,
                                                   cell_vars[0],
                                                   start_index,
                                                   end_index,
                                                   start_point,
                                                   end_point,
                                                   config_map);

  Kokkos::RangePolicy<ExecutionSpace, TagMarkCells> policy0(start_index, end_index);
  Kokkos::parallel_for("kalypsso::core::ComputeDataSliceAlongLine::MarkCells", policy0, functor);

  Kokkos::RangePolicy<ExecutionSpace> policy1(0, end_index - start_index);
  int32_t                             total_num_cells = 0;
  Kokkos::parallel_scan(
    "kalypsso::core::ComputeDataSliceAlongLine::Offsets",
    policy1,
    KOKKOS_LAMBDA(const int32_t i, int32_t & offset, bool is_final) {
      if (functor.m_offsets(i) == -1)
        return;

      if (is_final)
        functor.m_offsets(i) = offset;
      offset += 1;
    },
    total_num_cells);

  Kokkos::realloc(functor.m_positions, static_cast<size_t>(total_num_cells));
  Kokkos::realloc(functor.m_amr, static_cast<size_t>(total_num_cells));
  Kokkos::realloc(functor.m_data_slice, static_cast<size_t>(total_num_cells));

  // cell variables
  for (size_t iv = 0; iv < cell_vars.size(); ++iv)
  {

    functor.m_var = cell_vars[iv];

    Kokkos::RangePolicy<ExecutionSpace, TagWriteCellData> policy2(start_index, end_index);
    Kokkos::parallel_for("kalypsso::core::ComputeDataSliceAlongLine::WriteData", policy2, functor);

    auto positions_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, functor.m_positions);
    auto amr_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, functor.m_amr);
    auto data_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, functor.m_data_slice);

    if (parallel_env.nRanks() == 1)
    {
      if (iv == 0)
      {
        save_cnpy(positions_host, file_prefix + "_positions");
        save_cnpy(amr_host, file_prefix + "_level");
      }
      save_cnpy(data_host, file_prefix + "_" + cell_var_names[iv]);
    }
    else
    {
      const auto rank = std::to_string(parallel_env.rank());
      if (iv == 0)
      {
        save_cnpy(positions_host, file_prefix + "_" + rank + "_positions");
        save_cnpy(amr_host, file_prefix + "_" + rank + "_level");
      }
      save_cnpy(data_host, file_prefix + "_" + rank + "_" + cell_var_names[iv]);
    }
  }

  // face variables
  for (size_t iv = 0; iv < face_vars.size(); ++iv)
  {

    functor.m_var = face_vars[iv];

    Kokkos::RangePolicy<ExecutionSpace, TagWriteFaceData> policy2(start_index, end_index);
    Kokkos::parallel_for("kalypsso::core::ComputeDataSliceAlongLine::WriteData", policy2, functor);

    auto positions_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, functor.m_positions);
    auto amr_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, functor.m_amr);
    auto data_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, functor.m_data_slice);

    if (parallel_env.nRanks() == 1)
    {
      save_cnpy(data_host, file_prefix + "_" + face_var_names[iv]);
    }
    else
    {
      const auto rank = std::to_string(parallel_env.rank());
      save_cnpy(data_host, file_prefix + "_" + rank + "_" + face_var_names[iv]);
    }
  }

} // ComputeDataSliceAlongLine<dim, device_t>::apply

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
Kokkos::Array<real_t, dim>
ComputeDataSliceAlongLine<dim, device_t>::get_mid_box_start_point(const ConfigMap & config_map,
                                                                  const int32_t     direction)
{
  const auto xmin = config_map.getReal("mesh", "xmin", KALYPSSO_NUM(0.0));
  const auto ymin = config_map.getReal("mesh", "ymin", KALYPSSO_NUM(0.0));
  const auto zmin = config_map.getReal("mesh", "zmin", KALYPSSO_NUM(0.0));
  const auto nbrick_x = config_map.getInteger("p4est_connectivity", "nbrick_x", 1);
  const auto nbrick_y = config_map.getInteger("p4est_connectivity", "nbrick_y", 1);
  const auto nbrick_z = config_map.getInteger("p4est_connectivity", "nbrick_z", 1);
  const auto scaling_factor = config_map.getReal("mesh", "scaling_factor", KALYPSSO_NUM(1.0));

  Kokkos::Array<real_t, dim> xyz;

  xyz[IX] = direction == IX
              ? xmin
              : xmin + static_cast<real_t>(nbrick_x) * scaling_factor * HALF_F + KALYPSSO_NUM(1e-5);

  xyz[IY] = direction == IY
              ? ymin
              : ymin + static_cast<real_t>(nbrick_y) * scaling_factor * HALF_F + KALYPSSO_NUM(1e-5);

  if constexpr (dim == 3)
  {
    xyz[IZ] = direction == IZ ? zmin
                              : zmin + static_cast<real_t>(nbrick_z) * scaling_factor * HALF_F +
                                  KALYPSSO_NUM(1e-5);
  }

  return xyz;

} // ComputeDataSliceAlongLine<dim, device_t>::get_mid_box_start_point

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
Kokkos::Array<real_t, dim>
ComputeDataSliceAlongLine<dim, device_t>::get_mid_box_end_point(const ConfigMap & config_map,
                                                                const int32_t     direction)
{
  const auto xmin = config_map.getReal("mesh", "xmin", KALYPSSO_NUM(0.0));
  const auto ymin = config_map.getReal("mesh", "ymin", KALYPSSO_NUM(0.0));
  const auto zmin = config_map.getReal("mesh", "zmin", KALYPSSO_NUM(0.0));
  const auto nbrick_x = config_map.getInteger("p4est_connectivity", "nbrick_x", 1);
  const auto nbrick_y = config_map.getInteger("p4est_connectivity", "nbrick_y", 1);
  const auto nbrick_z = config_map.getInteger("p4est_connectivity", "nbrick_z", 1);
  const auto scaling_factor = config_map.getReal("mesh", "scaling_factor", KALYPSSO_NUM(1.0));

  Kokkos::Array<real_t, dim> xyz;

  xyz[IX] = direction == IX
              ? xmin + static_cast<real_t>(nbrick_x) * scaling_factor
              : xmin + static_cast<real_t>(nbrick_x) * scaling_factor * HALF_F + KALYPSSO_NUM(1e-5);

  xyz[IY] = direction == IY
              ? ymin + static_cast<real_t>(nbrick_y) * scaling_factor
              : ymin + static_cast<real_t>(nbrick_y) * scaling_factor * HALF_F + KALYPSSO_NUM(1e-5);

  if constexpr (dim == 3)
  {
    xyz[IZ] = direction == IZ ? zmin + static_cast<real_t>(nbrick_z) * scaling_factor
                              : zmin + static_cast<real_t>(nbrick_z) * scaling_factor * HALF_F +
                                  KALYPSSO_NUM(1e-5);
  }

  return xyz;

} // ComputeDataSliceAlongLine<dim, device_t>::get_mid_box_end_point

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
ComputeDataSliceAlongLine<dim, device_t>::operator()(const TagMarkCells,
                                                     const int32_t & i_global) const
{
  const auto block_size = m_data_cell.block_size();

  const auto nb_cells = m_data_cell.num_cells();
  const auto i_oct = i_global / nb_cells;
  const auto i_cell = i_global - nb_cells * i_oct;

  const auto i_coord = cellindex_to_coord<dim>(i_cell, m_data_cell.block_size());

  const auto key = m_keys(i_oct);
  const auto level = orchard_key_t<dim>::level(m_keys(i_oct));
  const auto xyz_vertex =
    orchard_key_to_cell_coord<dim>(key, i_coord, m_data_cell.block_size()[IX]);
  const auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_corner);
  const auto dx = compute_cell_length<dim>(level, block_size[IX]) * m_scaling_factor;

  // signed dim
  constexpr auto sdim = static_cast<int>(dim);

  const auto dir0 = static_cast<size_t>((m_dir + 0) % sdim); // Segment axis
  const auto dir1 = static_cast<size_t>((m_dir + 1) % sdim);
  const auto dir2 = static_cast<size_t>((m_dir + 2) % sdim);

  // Check if segment crosses cell
  bool cross = false;
  if constexpr (dim == 3)
    cross = xyz[dir1] - dx / 2 < m_start[dir1] && xyz[dir1] + dx / 2 > m_end[dir1] &&
            xyz[dir2] - dx / 2 < m_start[dir2] && xyz[dir2] + dx / 2 > m_end[dir2] &&
            xyz[dir0] > m_start[dir0] && xyz[dir0] < m_end[dir0];
  else if constexpr (dim == 2)
    cross = xyz[dir1] - dx / 2 < m_start[dir1] && xyz[dir1] + dx / 2 > m_end[dir1] &&
            xyz[dir0] > m_start[dir0] && xyz[dir0] < m_end[dir0];

  m_offsets(i_global - m_start_index) = cross ? 1 : -1;
}

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
ComputeDataSliceAlongLine<dim, device_t>::operator()(const TagWriteCellData,
                                                     const int32_t & i_global) const
{
  const auto offset = m_offsets(i_global - m_start_index);
  if (offset == -1)
    return;

  const auto nb_cells = m_data_cell.num_cells();
  const auto i_oct = i_global / nb_cells;
  const auto i_cell = i_global - nb_cells * i_oct;

  const auto i_coord = cellindex_to_coord<dim>(i_cell, m_data_cell.block_size());

  const auto key = m_keys(i_oct);
  const auto level = orchard_key_t<dim>::level(m_keys(i_oct));
  const auto xyz_vertex =
    orchard_key_to_cell_coord<dim>(key, i_coord, m_data_cell.block_size()[IX]);
  const auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_corner);

  m_positions(offset) = xyz[m_dir];
  m_amr(offset) = level;
  m_data_slice(offset) = m_data_cell(i_cell, m_var, i_oct);
} // ComputeDataSliceAlongLine<dim, device_t>::operator

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
ComputeDataSliceAlongLine<dim, device_t>::operator()(const TagWriteFaceData,
                                                     const int32_t & i_global) const
{
  const auto offset = m_offsets(i_global - m_start_index);
  if (offset == -1)
    return;

  const auto nb_cells = m_data_cell.num_cells();
  const auto i_oct = i_global / nb_cells;
  const auto i_cell = i_global - nb_cells * i_oct;

  const auto i_coord = cellindex_to_coord<dim>(i_cell, m_data_cell.block_size());

  const auto key = m_keys(i_oct);
  const auto level = orchard_key_t<dim>::level(m_keys(i_oct));
  const auto xyz_vertex =
    orchard_key_to_cell_coord<dim>(key, i_coord, m_data_cell.block_size()[IX]);
  const auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_corner);

  m_positions(offset) = xyz[m_dir];
  m_amr(offset) = level;

  if constexpr (dim == 2)
  {
    if (m_var == IX)
    {
      m_data_slice(offset) = (m_data_face(i_coord[IX], i_coord[IY], m_var, i_oct) +
                              m_data_face(i_coord[IX] + 1, i_coord[IY], m_var, i_oct)) /
                             2;
    }
    else if (m_var == IY)
    {
      m_data_slice(offset) = (m_data_face(i_coord[IX], i_coord[IY], m_var, i_oct) +
                              m_data_face(i_coord[IX], i_coord[IY] + 1, m_var, i_oct)) /
                             2;
    }
    else if (m_var == IZ)
    {
      m_data_slice(offset) = m_data_face(i_coord[IX], i_coord[IY], m_var, i_oct);
    }
  }
  else if constexpr (dim == 3)
  {
    if (m_var == IX)
    {
      m_data_slice(offset) =
        (m_data_face(i_coord[IX], i_coord[IY], i_coord[IZ], m_var, i_oct) +
         m_data_face(i_coord[IX] + 1, i_coord[IY], i_coord[IZ], m_var, i_oct)) /
        2;
    }
    else if (m_var == IY)
    {
      m_data_slice(offset) =
        (m_data_face(i_coord[IX], i_coord[IY], i_coord[IZ], m_var, i_oct) +
         m_data_face(i_coord[IX], i_coord[IY] + 1, i_coord[IZ], m_var, i_oct)) /
        2;
    }
    else if (m_var == IY)
    {
      m_data_slice(offset) =
        (m_data_face(i_coord[IX], i_coord[IY], i_coord[IZ], m_var, i_oct) +
         m_data_face(i_coord[IX], i_coord[IY], i_coord[IZ] + 1, m_var, i_oct)) /
        2;
    }
  }

} // ComputeDataSliceAlongLine<dim, device_t>::operator

// ================================================================================================
// ================================================================================================
template class ComputeDataSliceAlongLine<2, kalypsso::DefaultDevice>;
template class ComputeDataSliceAlongLine<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso
