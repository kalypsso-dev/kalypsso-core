// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file DataWriter.cpp
 */

#include "DataWriter.h"

namespace kalypsso
{

// ================================================================================
// ================================================================================
template <size_t dim, typename device_t, typename model_t>
void
DataWriter<dim, device_t, model_t>::save(std::string              filename,
                                         DataArrayLeaf_t          userdataLeaf,
                                         DataArrayBlock_t         userdataBlock,
                                         const ConfigMap &        config_map,
                                         AMRmesh<dim> /*const*/ & amr_mesh,
                                         model_t const &          model)
{
#ifdef KALYPSSO_CORE_USE_HDF5
  const auto fm = model.get_fieldmap();
  const auto id2names = model.get_id2names_map();
  const auto names2id = model.get_names2id_map();

  auto userdataLeaf_host = Kokkos::create_mirror_view(userdataLeaf);
  Kokkos::deep_copy(userdataLeaf_host, userdataLeaf);

  auto userdataBlock_host = DataArrayBlock_t::create_host_mirror_view_and_copy(userdataBlock);

  std::string outputDir = config_map.getString("output", "outputDir", "./");

  HDF5_Xdmf_Writer_legacy_t writer(amr_mesh.forest(),
                                   amr_mesh.geometry(),
                                   config_map,
                                   userdataBlock_host.block_size(),
                                   get_shift<dim>(0));
  writer.update_mesh_info();
  writer.open(filename, outputDir);
  writer.write_header(0.0);

  // write user the test data (all enabled scalar fields)
  {
    std::string write_variables = config_map.getString("output", "write_variables", "");

    for (auto & it : id2names)
    {
      auto varId = static_cast<typename model_t::VarId>(it.first);
      auto name = id2names.at(varId);
      if (write_variables.find(name) != std::string::npos)
      {
        writer.write_quadrant_attribute(userdataBlock_host, fm[varId], name);
      }
    }
  }

  // close the file
  writer.write_footer();
  writer.close();

  // now write data attached to octree leaf only (e.g. leaf level)
  writer.set_leaf_mode();
  writer.set_write_mesh_info(true);

  std::string filename2 = filename + "_quad";

  writer.open(filename2, outputDir);
  writer.write_header(0.0);

  // write user the test data (all scalar fields, here only one)
  {
    std::string write_variables = config_map.getString("output", "write_variables", "");

    for (auto & it : id2names)
    {
      auto varId = static_cast<typename model_t::VarId>(it.first);
      auto name = id2names.at(varId);
      if (write_variables.find(name) != std::string::npos)
      {
        writer.write_quadrant_attribute(userdataLeaf_host, fm[varId], name);
      }
    }
  }

  // close the file
  writer.write_footer();
  writer.close();
#else
  std::cout << "HDF5 output not available\n";
#endif
} // save - HDF5_Xdmf_Writer_legacy

// ================================================================================
// ================================================================================
template <size_t dim, typename device_t, typename model_t>
void
DataWriter<dim, device_t, model_t>::save(std::string                             filename,
                                         DataArrayLeaf_t                         userdataLeaf,
                                         DataArrayBlock_t                        userdataBlock,
                                         const ConfigMap &                       config_map,
                                         const ParallelEnv &                     par_env,
                                         std::shared_ptr<MeshMap<dim, device_t>> mesh_map,
                                         bool                                    use_outside_quads,
                                         model_t const &                         model)
{
#ifdef KALYPSSO_CORE_USE_HDF5
  const auto fm = model.get_fieldmap();
  const auto id2names = model.get_id2names_map();
  const auto names2id = model.get_names2id_map();

  auto userdataLeaf_host = Kokkos::create_mirror_view(userdataLeaf);
  Kokkos::deep_copy(userdataLeaf_host, userdataLeaf);

  auto userdataBlock_host = DataArrayBlock_t::create_host_mirror_view_and_copy(userdataBlock);

  std::string outputDir = config_map.getString("output", "outputDir", "./");

  std::string filename_cell = filename + "_cell";
  std::string filename_leaf = filename + "_leaf";

  if (use_outside_quads)
  {
    filename_cell = filename_cell + "_outside";
    filename_leaf = filename_leaf + "_outside";
  }

  HDF5_Xdmf_Writer_t writer(par_env, config_map, mesh_map);

  const auto amr_mesh_info = mesh_map->get_amr_mesh_info();

  //
  // block mode
  //
  writer.set_block_mode();
  writer.use_outside_quads(use_outside_quads);
  writer.set_write_mesh_info(true);
  writer.open(filename_cell, outputDir);
  writer.write_header(0.0);

  const int32_t iOct_begin =
    use_outside_quads ? amr_mesh_info.local_num_quadrants() + amr_mesh_info.local_num_ghosts() : 0;
  const int32_t nbOcts = use_outside_quads ? amr_mesh_info.local_num_quadrants_outside()
                                           : userdataBlock_host.num_quadrants();


  // write user data - block mode - all enabled scalar fields
  {
    std::string write_variables = config_map.getString("output", "write_variables", "");

    for (auto & it : id2names)
    {
      auto varId = static_cast<typename model_t::VarId>(it.first);
      auto name = id2names.at(varId);
      if (write_variables.find(name) != std::string::npos)
      {
        writer.write_quadrant_attribute(userdataBlock_host, fm[varId], name, iOct_begin, nbOcts);
      }
    } // end for it
  }

  // close the file
  writer.write_footer();
  writer.close();

  //
  // now write data attached to octree leaf only (e.g. leaf level)
  //
  writer.set_leaf_mode();
  writer.use_outside_quads(use_outside_quads);
  writer.set_write_mesh_info(true);
  writer.open(filename_leaf, outputDir);
  writer.write_header(0.0);

  // write user the test data (all scalar fields, here only one)
  {
    std::string write_variables = config_map.getString("output", "write_variables", "");

    // extract slice as a contiguous memory buffer
    auto nbVar = userdataLeaf_host.extent(1);

    if (use_outside_quads)
    {
      using DataArrayLeaf_t = DataArrayLeaf<real_t, device_t>;
      using DataArrayLeafHost_t = typename DataArrayLeaf_t::host_mirror_type;
      DataArrayLeafHost_t userdataLeaf_outside_host(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "dummy_data_outside"),
        static_cast<size_t>(amr_mesh_info.local_num_quadrants_outside()),
        nbVar);

      const auto offset = amr_mesh_info.local_num_quadrants() + amr_mesh_info.local_num_ghosts();

      const auto nbOctOutside = amr_mesh_info.local_num_quadrants_outside();
      Kokkos::parallel_for(
        "extract outside data",
        Kokkos::RangePolicy<Kokkos::OpenMP>(0, nbOctOutside * static_cast<int64_t>(nbVar)),
        KOKKOS_LAMBDA(int64_t index) {
          // index = iOct + nbOct * varId
          const auto varId = static_cast<typename model_t::VarId>(index / nbOctOutside);
          const auto iOct = index - varId * nbOctOutside;
          userdataLeaf_outside_host(iOct, fm[varId]) = userdataLeaf_host(iOct + offset, fm[varId]);
        });

      // make sure all kokkos kernels are done before performing actual writing
      Kokkos::fence();

      for (auto & it : id2names)
      {
        auto varId = static_cast<typename model_t::VarId>(it.first);
        auto name = id2names.at(varId);
        if (write_variables.find(name) != std::string::npos)
        {
          writer.write_quadrant_attribute(userdataLeaf_outside_host, fm[varId], name);
        }
      }
    }
    else
    {
      for (auto & it : id2names)
      {
        auto varId = static_cast<typename model_t::VarId>(it.first);
        auto name = id2names.at(varId);
        if (write_variables.find(name) != std::string::npos)
        {
          writer.write_quadrant_attribute(userdataLeaf_host, fm[varId], name);
        }
      }
    } // end if use_outside_quads
  } // end write leaf data

  // close the file
  writer.write_footer();
  writer.close();
#else
  std::cout << "HDF5 output not available\n";
#endif

} // save - HDF5_Xdmf_Writer

// // ================================================================================
// // ================================================================================
// template <size_t dim, typename device_t, typename model_t>
// void
// DataWriter<dim, device_t, model_t>::save(std::string              filename,
//                                          DataArrayLeaf_t          userdataLeaf,
//                                          const ConfigMap &        config_map,
//                                          AMRmesh<dim> /*const*/ & amr_mesh,
//                                          model_t const &          model)
// {
// #ifdef KALYPSSO_CORE_USE_HDF5
//   const auto fm = model.get_fieldmap();
//   const auto id2names = model.get_id2names_map();
//   const auto names2id = model.get_names2id_map();

//   auto userdataLeaf_host = Kokkos::create_mirror_view(userdataLeaf);
//   Kokkos::deep_copy(userdataLeaf_host, userdataLeaf);

//   std::string outputDir = config_map.getString("output", "outputDir", "./");

//   HDF5_Xdmf_Writer_legacy_t writer(
//     amr_mesh.forest(), amr_mesh.geometry(), config_map, get_block_size<dim>(1),
//     get_shift<dim>(0));
//   writer.update_mesh_info();

//   // write data attached to octree leaf only (e.g. leaf level)
//   writer.set_leaf_mode();
//   writer.set_write_mesh_info(true);

//   std::string filename2 = filename + "_quad";

//   writer.open(filename2, outputDir);
//   writer.write_header(0.0);

//   // write user the test data (all scalar fields, here only one)
//   {
//     std::string write_variables = config_map.getString("output", "write_variables", "");

//     for (auto & it : id2names)
//     {
//       auto varId = static_cast<typename model_t::VarId>(it.first);
//       auto name = id2names.at(varId);
//       if (write_variables.find(name) != std::string::npos)
//       {
//         writer.write_quadrant_attribute(userdataLeaf_host, fm[varId], name);
//       }
//     }
//   }

//   // close the file
//   writer.write_footer();
//   writer.close();
// #else
//   std::cout << "HDF5 output not available\n";
// #endif

// } // save

// ================================================================================
// ================================================================================
template <size_t dim, typename device_t, typename model_t>
void
DataWriter<dim, device_t, model_t>::save(std::string              filename,
                                         DataArrayBlock_t         userdataBlock,
                                         const ConfigMap &        config_map,
                                         /*const*/ AMRmesh<dim> & amr_mesh,
                                         model_t const &          model)
{
#ifdef KALYPSSO_CORE_USE_HDF5
  const auto fm = model.get_fieldmap();
  const auto id2names = model.get_id2names_map();
  const auto names2id = model.get_names2id_map();

  auto userdataBlock_host = DataArrayBlock_t::create_host_mirror_view_and_copy(userdataBlock);

  std::string outputDir = config_map.getString("output", "outputDir", "./");

  HDF5_Xdmf_Writer_legacy_t writer(amr_mesh.forest(),
                                   amr_mesh.geometry(),
                                   config_map,
                                   userdataBlock.block_size(),
                                   get_shift<dim>(0));
  writer.update_mesh_info();
  writer.open(filename, outputDir);
  writer.write_header(0.0);

  // write user the test data (all enabled scalar fields)
  {
    std::string write_variables = config_map.getString("output", "write_variables", "");

    for (auto & it : id2names)
    {
      auto varId = static_cast<typename model_t::VarId>(it.first);
      auto name = id2names.at(varId);

      if (write_variables.find(name) != std::string::npos)
      {
        writer.write_quadrant_attribute(userdataBlock_host, fm[varId], name);
      }
    }
  }

  // close the file
  writer.write_footer();
  writer.close();
#else
  std::cout << "HDF5 output not available\n";
#endif

} // save - DataArrayBlock_t

// ================================================================================
// ================================================================================
template <size_t dim, typename device_t, typename model_t>
void
DataWriter<dim, device_t, model_t>::save_scalar(std::string              filename,
                                                DataArrayBlock_t         userdataBlock,
                                                int                      ivar,
                                                const ConfigMap &        config_map,
                                                /*const*/ AMRmesh<dim> & amr_mesh)
{
#ifdef KALYPSSO_CORE_USE_HDF5

  auto userdataBlock_host = DataArrayBlock_t::create_host_mirror_view_and_copy(userdataBlock);

  std::string outputDir = config_map.getString("output", "outputDir", "./");

  HDF5_Xdmf_Writer_legacy_t writer(amr_mesh.forest(),
                                   amr_mesh.geometry(),
                                   config_map,
                                   userdataBlock.block_size(),
                                   get_shift<dim>(0));

  writer.update_mesh_info();
  writer.open(filename, outputDir);
  writer.write_header(0.0);

  // write user the test data (all enabled scalar fields)
  writer.write_quadrant_attribute(userdataBlock_host, ivar, "var_" + std::to_string(ivar));

  // close the file
  writer.write_footer();
  writer.close();
#else
  std::cout << "HDF5 output not available\n";
#endif

} // save - DataArrayBlock_t

// ================================================================================
// ================================================================================
template <size_t dim, typename device_t, typename model_t>
void
DataWriter<dim, device_t, model_t>::save(std::string              filename,
                                         DataArrayGhostedBlock_t  userdata_ghosted_block,
                                         const ConfigMap &        config_map,
                                         /*const*/ AMRmesh<dim> & amr_mesh,
                                         bool                     save_full,
                                         model_t const &          model)
{
#ifdef KALYPSSO_CORE_USE_HDF5
  const auto fm = model.get_fieldmap();
  const auto id2names = model.get_id2names_map();
  const auto names2id = model.get_names2id_map();

  auto userdata_ghosted_block_host =
    DataArrayBlock_t::create_host_mirror_view_and_copy(userdata_ghosted_block.data());

  auto total_block_sizes = userdata_ghosted_block.ghosted_block_size();

  std::string outputDir = config_map.getString("output", "outputDir", "./");

  // default we save the overlapping part
  const auto start_index = userdata_ghosted_block.get_start_overlap();
  const auto block_size_overlap = userdata_ghosted_block.get_block_size_overlap();

  HDF5_Xdmf_Writer_legacy_t writer(
    amr_mesh.forest(), amr_mesh.geometry(), config_map, block_size_overlap, start_index);

  writer.set_write_mesh_info(true);

  if (save_full)
  {
    writer.set_block_mode(total_block_sizes, get_shift<dim>(0));
  }

  writer.update_mesh_info();
  writer.open(filename, outputDir);
  writer.write_header(0.0);

  // write user the test data (all enabled scalar fields)
  {
    std::string write_variables = config_map.getString("output", "write_variables", "");

    for (auto & it : id2names)
    {
      auto varId = static_cast<typename model_t::VarId>(it.first);
      auto name = id2names.at(varId);
      if (write_variables.find(name) != std::string::npos)
      {
        writer.write_quadrant_attribute(userdata_ghosted_block_host, fm[varId], name);
      }
    }
  }

  // close the file
  writer.write_footer();
  writer.close();
#else
  std::cout << "HDF5 output not available\n";
#endif

} // save - DataArrayGhostedBlock_t

// ================================================================================
// ================================================================================
template <size_t dim, typename device_t, typename model_t>
void
DataWriter<dim, device_t, model_t>::save(std::string              filename,
                                         FaceDataArrayBlock_t     facedata,
                                         const ConfigMap &        config_map,
                                         /*const*/ AMRmesh<dim> & amr_mesh,
                                         std::string              varname)
{
#ifdef KALYPSSO_CORE_USE_HDF5
  std::string outputDir = config_map.getString("output", "outputDir", "./");

  const auto celldata = FaceDataArrayBlock_t::to_DataArrayBlock(facedata);
  const auto celldata_host = DataArrayBlock_t::create_host_mirror_view_and_copy(celldata);

  HDF5_Xdmf_Writer_legacy_t writer(
    amr_mesh.forest(), amr_mesh.geometry(), config_map, celldata.block_size(), get_shift<dim>(0));

  writer.update_mesh_info();
  writer.open(filename, outputDir);
  writer.write_header(0.0);

  if constexpr (dim == 2)
  {
    writer.write_quadrant_attribute(celldata_host, 0, varname + "_x_left");
    writer.write_quadrant_attribute(celldata_host, 1, varname + "_x_right");
    writer.write_quadrant_attribute(celldata_host, 2, varname + "_y_left");
    writer.write_quadrant_attribute(celldata_host, 3, varname + "_y_right");
    writer.write_quadrant_attribute(celldata_host, 4, varname + "_z");
  }
  else if constexpr (dim == 3)
  {
    writer.write_quadrant_attribute(celldata_host, 0, varname + "_x_left");
    writer.write_quadrant_attribute(celldata_host, 1, varname + "_x_right");
    writer.write_quadrant_attribute(celldata_host, 2, varname + "_y_left");
    writer.write_quadrant_attribute(celldata_host, 3, varname + "_y_right");
    writer.write_quadrant_attribute(celldata_host, 4, varname + "_z_left");
    writer.write_quadrant_attribute(celldata_host, 5, varname + "_z_right");
  }

  //   const auto data_x = FaceDataArrayBlock_t::to_DataArrayBlockCentered(userdataBlock_face,
  //   IX); const auto data_x_host = DataArrayBlock_t::create_host_mirror_view_and_copy(data_x);

  //   writer.write_quadrant_attribute(data_x_host, 0, varname + "_x");

  //   const auto data_y = FaceDataArrayBlock_t::to_DataArrayBlockCentered(userdataBlock_face,
  //   IY); const auto data_y_host = DataArrayBlock_t::create_host_mirror_view_and_copy(data_y);
  //   writer.write_quadrant_attribute(data_y_host, 0, varname + "_y");

  //   const auto data_z = FaceDataArrayBlock_t::to_DataArrayBlockCentered(userdataBlock_face,
  //   IZ); const auto data_z_host = DataArrayBlock_t::create_host_mirror_view_and_copy(data_z);
  //   writer.write_quadrant_attribute(data_z_host, 0, varname + "_z");

  // close the file
  writer.write_footer();
  writer.close();
#else
  std::cout << "HDF5 output not available\n";
#endif

} // save - FaceDataArrayBlock_t

template class DataWriter<2, kalypsso::DefaultDevice, core::models::Hydro>;
template class DataWriter<3, kalypsso::DefaultDevice, core::models::Hydro>;

// only instantiate those class when the default device is not on host
#if defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP) || \
  defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_THREADS) ||  \
  defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL)
#else
template class DataWriter<2, kalypsso::HostDevice, core::models::Hydro>;
template class DataWriter<3, kalypsso::HostDevice, core::models::Hydro>;
#endif

} // namespace kalypsso
