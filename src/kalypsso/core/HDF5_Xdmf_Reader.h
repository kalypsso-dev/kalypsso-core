// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file HDF5_Xdmf_Reader.h
 * \brief same functionality as HDF5_IO but only relying on orchard key hashmap, no p4est here.
 *
 * The main reason for redesigning io is to be ease implementing a class that could dump regular
 * quadrants (inside domain), as well as outside quadrants.
 */
#ifndef KALYPSSO_CORE_HDF5_XDMF_READER_H_
#define KALYPSSO_CORE_HDF5_XDMF_READER_H_

#include <hdf5.h>
#include <hdf5_hl.h>

#include <kalypsso/core/enums.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayLeaf, DataArrayLeafHost

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>
#include <kalypsso/utils/p4est/p4est_wrapper.h>

#include <kalypsso/core/kalypsso_core_base.h> // includes kalypsso_core_config.h
#include <kalypsso/core/AMRmesh_utils.h>
#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/core/brick_connectivity_utils.h>
#include <kalypsso/core/brick_utils.h>
#include <kalypsso/core/AMRMeshInfo.h>
#include <kalypsso/core/MeshMap.h>
#include <kalypsso/core/AMRMeshMonitoring.h> // for compute_level_histogram
#include <kalypsso/core/scan_utils.h>
#include <kalypsso/core/MaterialPresence.h>

#include <kalypsso/core/HDF5_IO_common.h>

#include <highfive/highfive.hpp>

#include <vector>
#include <numeric>    // for std::reduce
#include <functional> // for std::multiplies

namespace kalypsso
{

// ================================================================================================
// ================================================================================================
/**
 * \brief HDF5 reader for both quadrant inside / outside domain.
 *
 * Optionally a Xdmf file is written so that hdf5 can be opened directly in paraview using VTU data
 * format (VTK Unstructured Mesh)
 *
 * \tparam dim = 2 or 3
 */
template <size_t dim, typename device_t>
class HDF5_Xdmf_Reader
{

public:
  // using DataArrayLeafHost_t = DataArrayLeafHost<real_t, device_t>;
  using DataArrayBlockLegacyHost_t = DataArrayBlockLegacyHost<real_t, device_t>;
  using DataArrayBlockHost_t = DataArrayBlock<dim, real_t, HostDevice>;
  using DataArrayBlockMultiVarHost_t = DataArrayBlockMultiVar<dim, real_t, HostDevice>;
  using MaterialPresenceHost_t = MaterialPresenceView<HostDevice>;
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  using p4est_t = typename p4est::Wrapper<dim>;

  using MeshMap_t = MeshMap<dim, device_t>;

  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;
  using orchard_key_view_host_t = typename orchard_key_base_t<device_t>::view_host_t;

  /**
   * \brief Constructor.
   *
   * The reader is has no opened files by default except a main xmf file
   * that contains a collection of all the files that are to be written
   * next as a `Temporal` Collection.
   *
   * \param[in] par_env is the parallel environment (to access MPI_Comm)
   * \param[in] config_map to access parameters settings
   * \param[in] mesh_map to get access to orchard keys (as kokkos view) and amr_mesh_info
   *
   */
  HDF5_Xdmf_Reader(ParallelEnv const &        par_env,
                   ConfigMap const &          config_map,
                   std::shared_ptr<MeshMap_t> mesh_map);

  //! destructor
  virtual ~HDF5_Xdmf_Reader();

  /**
   * To be called as often as the mesh changes (refine/coarsen/balance/repartition).
   *
   * Recompute local (current MPI proc) and global (all MPI proc) number
   * of quadrants (forest leaves) and mesh block nodes.
   */
  void
  update_mesh_info();

  /**
   * \brief Open the HDF5 file for reading.
   */
  void
  open(std::string basename, std::string outDir);

  /**
   * \brief Open the HDF5 file for reading.
   */
  void
  open(std::string full_path);

  /**
   * \brief Close the HDF5 file.
   */
  void
  close();

  /**
   * Read scalar attribute.
   *
   * \param[in] group_name hdf5 group name
   * \param[in] var_name scalar variable name
   * \param[out] data variable value
   */
  template <typename data_t>
  uint64_t
  read_scalar_attribute(const std::string & group_name,
                        const std::string & var_name,
                        data_t &            data);


  /**
   * \brief Read cell-centered attribute.
   *
   * \param [in] name The name of the attribute.
   * \param [in] data The data to be read.
   * \param [in] dimData In the case of a vector, this is the dimension of each
   *             element in the vector field.
   * \param [in] ftype The type of the attribute. See supported types in
   *             the io_attribute_type_t enum.
   *
   * \return number of bytes written
   */
  template <typename data_t>
  uint64_t
  read_cc_attribute(const std::string & name,
                    data_t *            data,
                    size_t              dimData,
                    io_attribute_type_t ftype);

  /**
   * \brief Read face-centered data.
   *
   * Be aware that face data can't be added to xmdf and read directly in Paraview.
   * Currently this is only useful for checkpoint/restart a FaceDataArrayBlock.
   * At some point, we will see if we could update our paraview plugin to read face data, convert
   * them into cell-centered data on the fly.
   *
   * \param [in] name The name of the attribute.
   * \param [out] data Pointer where data will be copied.
   * \param [in] dimData In the case of a vector, this is the dimension of each
   *             element in the vector field.
   * \param [in] number of face elements per octants (see
   *             FaceDataArrayBlock::num_elements_per_octant).
   *
   * \return number of bytes read
   */
  template <typename data_t>
  uint64_t
  read_fc_attribute(const std::string & name,
                    data_t *            data,
                    size_t              dimData,
                    size_t              num_elements_per_oct);

  /**
   * \brief Read a cell-centered scalar attributes.
   *
   * \param [out] datah a host Kokkos View with the user data (block data)
   * \param [in] varIdx scalar index identify scalar field to read
   * \param [in] varName variable name
   * \param [in] iOct_begin index to first octant to read
   * \param [in] nbOcts number of octants to read
   */
  uint64_t
  read_quadrant_attribute(DataArrayBlockHost_t datah,
                          int32_t              varIdx,
                          const std::string    varName,
                          int32_t              iOct_begin,
                          int32_t              nbOcts);

  /**
   * \brief Read face-centered data.
   *
   * \param [out] face_data a device view of some face data.
   * \param [in] varName variable name
   */
  uint64_t
  read_quadrant_attribute(FaceDataArrayBlock_t face_data, const std::string varName);

  /**
   * \brief Read all cell-centered scalar attributes.
   *
   * \param [out] datah a host multivar array
   * \param [in] matph a host material presence object
   * \param [in] num_mats the number of materials
   * \param [in] num_vars_per_mat the number of variables per material
   * \param [in] mat_varIdx scalar index identify scalar field to read related to materials
   * \param [in] varName variable name (prefix)
   */
  // uint64_t
  // read_quadrant_multi_mat_attribute(DataArrayBlockMultiVarHost_t datah,
  //                                   MaterialPresenceHost_t       matph,
  //                                   int32_t                      num_mats,
  //                                   int32_t                      num_vars_per_mat,
  //                                   int32_t                      mat_varIdx,
  //                                   std::string                  varName,
  //                                   int32_t                      iOct_begin,
  //                                   int32_t                      nbOcts);

  ParallelEnv const & m_par_env;
  ConfigMap const &   m_config_map;

  std::shared_ptr<MeshMap_t> m_mesh_map;

  int m_bx; //!< block size along x
  int m_by; //!< block size along y
  int m_bz; //!< block size along z

  uint32_t m_nbCellsPerLeaf; //!< only useful for block amr

  HighFive::File * m_hdf5_file;      //!< HDF5 file descriptor
  FILE *           m_xdmf_file;      //!< XDMF file descriptor
  FILE *           m_main_xdmf_file; //!< main XDMF file descriptor

  // store information about nodes for writing node data

  uint64_t m_global_num_quads;  //!< global accumulated (all MPI proc) number of quads/octs
  uint32_t m_local_num_quads;   //!< local (current MPI proc) number of quads/octs
  uint64_t m_global_first_quad; //!< global index of first quadrant in current MPI proc

  bool m_use_outside_quads = false;

  void
  use_outside_quads(bool value);

private:
  /*
   * HDF5 utilities.
   */

  /**
   * \brief Read a given dataset from the HDF5 file.
   *
   * \param [in] fd An open file descriptor to a HDF5 file.
   * \param [in] dataset_name The name of the dataset we are writing.
   * \param [out] data_ptr memory pointer to data read from file.
   * \param [in] dims The global dimensions of the dataset.
   * \param [in] count The local dimensions of the dataset.
   * \param [in] offset The offset of the local data with respect to the global
   * positioning.
   *
   * \return total number of bytes written.
   *
   * \sa H5TPublic.h
   */
  template <typename data_t>
  uint64_t
  io_hdf5_readv(HighFive::File &                 hdf5_file,
                std::string                      dataset_prefix,
                const std::string &              dataset_name,
                data_t *                         data_ptr,
                std::vector<std::size_t> const & count,
                std::vector<std::size_t> const & offset);

}; // class HDF5_Xdmf_Reader

// =============================================================================
// =============================================================================
// class HDF5_Xdmf_Reader definition
// =============================================================================
// =============================================================================

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
HDF5_Xdmf_Reader<dim, device_t>::HDF5_Xdmf_Reader(ParallelEnv const &        par_env,
                                                  ConfigMap const &          config_map,
                                                  std::shared_ptr<MeshMap_t> mesh_map)
  : m_par_env(par_env)
  , m_config_map(config_map)
  , m_mesh_map(mesh_map)
  , m_bx(config_map.getInteger("amr", "bx", 1))
  , m_by(config_map.getInteger("amr", "by", 1))
  , m_bz(config_map.getInteger("amr", "bz", 1))
  , m_nbCellsPerLeaf(dim == 2 ? static_cast<uint32_t>(m_bx * m_by)
                              : static_cast<uint32_t>(m_bx * m_by * m_bz))
  , m_hdf5_file()
  , m_global_num_quads(0)
  , m_local_num_quads(0)
  , m_global_first_quad(0)
{

  // TODO
  update_mesh_info();

} // HDF5_Xdmf_Reader<dim, device_t>::HDF5_Xdmf_Reader

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
HDF5_Xdmf_Reader<dim, device_t>::~HDF5_Xdmf_Reader()
{

  // close hdf5 file
  close();

} // HDF5_Xdmf_Reader<dim, device_t>::~HDF5_Xdmf_Reader

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Reader<dim, device_t>::update_mesh_info()
{

  m_local_num_quads = static_cast<uint32_t>(m_mesh_map->get_amr_mesh_info().local_num_quadrants());
  m_global_num_quads =
    static_cast<uint64_t>(m_mesh_map->get_amr_mesh_info().global_num_quadrants());
  m_global_first_quad =
    static_cast<uint64_t>(m_mesh_map->get_amr_mesh_info().global_first_quadrant());

} // HDF5_Xdmf_Reader<dim, device_t>::update_mesh_info

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Reader<dim, device_t>::open(std::string basename, std::string outDir)
{
  // build filename
  std::string filename = basename + ".h5";
  std::string full_path = outDir + "/" + filename;

  this->open(full_path);

} // HDF5_Xdmf_Reader<dim, device_t>::open

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Reader<dim, device_t>::open(std::string full_path)
{
  /*
   * create file access property list
   */
  HighFive::FileAccessProps fapl;
#ifdef KALYPSSO_CORE_USE_MPI
  fapl.add(HighFive::MPIOFileAccess{ m_par_env.mpi_comm(), MPI_INFO_NULL });
  // all metadata are written using collective IO
  fapl.add(HighFive::MPIOCollectiveMetadata{});
#endif // KALYPSSO_CORE_USE_MPI

  /*
   * Open parallel HDF5 resources.
   */
  m_hdf5_file = new HighFive::File(full_path, HighFive::File::ReadOnly, fapl);

} // HDF5_Xdmf_Reader<dim, device_t>::open

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Reader<dim, device_t>::close()
{
  // flush HDF5
  if (m_hdf5_file != nullptr)
  {
    m_hdf5_file->flush();
    delete m_hdf5_file;
    m_hdf5_file = nullptr;
  }

} // HDF5_Xdmf_Reader<dim, device_t>::close

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
template <typename data_t>
uint64_t
HDF5_Xdmf_Reader<dim, device_t>::read_scalar_attribute(const std::string & group_name,
                                                       const std::string & var_name,
                                                       data_t &            data)
{
  if (m_hdf5_file->exist(group_name))
  {
    // const auto group = m_hdf5_file->getGroup(group_name);
    // const auto attribute = group.getAttribute(var_name);
    const auto attribute = m_hdf5_file->getGroup(group_name).getAttribute(var_name);
    data = attribute.read<data_t>();
  }

  return sizeof(data_t);

} // HDF5_Xdmf_Reader<dim, device_t>::read_scalar_attribute

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
template <typename data_t>
uint64_t
HDF5_Xdmf_Reader<dim, device_t>::read_cc_attribute(const std::string & data_name,
                                                   data_t *            data,
                                                   size_t              dimData,
                                                   io_attribute_type_t ftype)
{
  uint64_t num_bytes = 0;

  std::vector<std::size_t> dims{ 0 };
  std::vector<std::size_t> count{ 0 };
  std::vector<std::size_t> start{ 0 };

  if (dimData > 0)
  {
    dims.resize(2);
    count.resize(2);
    start.resize(2);
  }

  if (ftype == IO_CELL_SCALAR)
  {

    dims[0] = static_cast<size_t>(m_mesh_map->get_amr_mesh_info().global_num_quadrants()) *
              m_nbCellsPerLeaf;
    if (dimData > 0)
      dims[1] = dimData;

    count[0] =
      static_cast<size_t>(m_mesh_map->get_amr_mesh_info().local_num_quadrants()) * m_nbCellsPerLeaf;
    if (dimData > 0)
      count[1] = dims[1];

    // get global index of the first octant of current mpi processor
    start[0] = static_cast<size_t>(m_mesh_map->get_amr_mesh_info().global_first_quadrant()) *
               m_nbCellsPerLeaf;
    if (dimData > 0)
      start[1] = 0;
  }

  num_bytes += io_hdf5_readv(*m_hdf5_file, "/celldata/", data_name, data, count, start);

  return num_bytes;

} // HDF5_Xdmf_Reader<dim, device_t>::read_cc_attribute

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
template <typename data_t>
uint64_t
HDF5_Xdmf_Reader<dim, device_t>::read_fc_attribute(const std::string & data_name,
                                                   data_t *            data,
                                                   size_t              dimData,
                                                   size_t              num_elements_per_oct)
{
  uint64_t num_bytes = 0;

  std::vector<std::size_t> dims{ 0 };
  std::vector<std::size_t> count{ 0 };
  std::vector<std::size_t> start{ 0 };

  if (dimData > 0)
  {
    dims.resize(2);
    count.resize(2);
    start.resize(2);
  }

  dims[0] = static_cast<size_t>(m_mesh_map->get_amr_mesh_info().global_num_quadrants()) *
            num_elements_per_oct;

  count[0] = static_cast<size_t>(m_mesh_map->get_amr_mesh_info().local_num_quadrants()) *
             num_elements_per_oct;

  // get global index of the first octant of current mpi processor
  start[0] = static_cast<size_t>(m_mesh_map->get_amr_mesh_info().global_first_quadrant()) *
             num_elements_per_oct;

  num_bytes += io_hdf5_readv(*m_hdf5_file, "/facedata/", data_name, data, count, start);

  return num_bytes;

} // HDF5_Xdmf_Reader<dim, device_t>::read_fc_attribute

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Reader<dim, device_t>::read_quadrant_attribute(DataArrayBlockHost_t datah,
                                                         int32_t              varIdx,
                                                         const std::string    varName,
                                                         int32_t              iOct_begin,
                                                         int32_t              nbOcts)
{
  uint64_t      num_bytes = 0;
  const int32_t nbCellsPerOct = dim == 2 ? m_bx * m_by : m_bx * m_by * m_bz;

  assertm(datah.num_cells() == nbCellsPerOct,
          "HDF5_Xdmf_Reader::read_quadrant_attribute wrong data size");

  // we need to gather data corresponding to a given scalar variable
  using DataArrayScalar = Kokkos::View<real_t *, Kokkos::HostSpace>;

  // remember that
  // - data.extent(0) is the number of cells per octant
  // - data.extent(1) is the number of scalar fields
  // - data.extent(2) is the total number of oct in current MPI process
  DataArrayScalar dataVar =
    DataArrayScalar("scalar_array_for_hdf5_io", static_cast<size_t>(nbCellsPerOct * nbOcts));

  // actual data writing
  num_bytes += read_cc_attribute(varName, dataVar.data(), 0, IO_CELL_SCALAR);

  // copy dataVar into datah
  Kokkos::parallel_for(
    "HDF5_Xdmf_Reader::read_quadrant_attribute",
    Kokkos::RangePolicy<Kokkos::OpenMP>(iOct_begin, iOct_begin + nbOcts),
    KOKKOS_LAMBDA(int32_t iOct) {
      for (int32_t iCell = 0; iCell < nbCellsPerOct; ++iCell)
        datah(iCell, varIdx, iOct) = dataVar(iCell + nbCellsPerOct * (iOct - iOct_begin));
    });

  // make sure all kokkos kernels are done before actual writing
  Kokkos::fence();

  return num_bytes;

} // HDF5_Xdmf_Reader<dim, device_t>::read_quadrant_attribute

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Reader<dim, device_t>::read_quadrant_attribute(FaceDataArrayBlock_t face_data,
                                                         const std::string    varName)
{

  uint64_t num_bytes = 0;
  size_t   num_elements_per_octant = static_cast<size_t>(face_data.num_elements_per_octant());

  auto face_data_storage = face_data.logical_view();
  auto face_data_storage_host = Kokkos::create_mirror_view(Kokkos::HostSpace{}, face_data_storage);

  // actual data writing
  num_bytes +=
    read_fc_attribute(varName, face_data_storage_host.data(), 0, num_elements_per_octant);

  Kokkos::deep_copy(face_data_storage, face_data_storage_host);

  return num_bytes;

} // HDF5_Xdmf_Reader<dim, device_t>::read_quadrant_attribute - FaceDataArrayBlock

// =======================================================
// =======================================================
// template <size_t dim, typename device_t>
// uint64_t
// HDF5_Xdmf_Reader<dim, device_t>::read_quadrant_multi_mat_attribute(
//   DataArrayBlockMultiVarHost_t datah,
//   MaterialPresenceHost_t       matph,
//   int32_t                      num_mats,
//   int32_t                      num_vars_per_mat,
//   int32_t                      mat_varIdx,
//   std::string                  varName,
//   int32_t                      iOct_begin,
//   int32_t                      nbOcts)
// {
//   varName = "_" + varName;

//   uint64_t num_bytes = 0;
//   int32_t  nbCellsPerOct = dim == 2 ? m_bx * m_by : m_bx * m_by * m_bz;

//   int32_t                  size_x = m_bx + 2 * m_gx;
//   int32_t                  size_y = m_by + 2 * m_gy;
//   [[maybe_unused]] int32_t size_z = m_bz + 2 * m_gz;

//   // we don't want to capture this in kokkos lambda's
//   auto                  bx = m_bx;
//   auto                  by = m_by;
//   [[maybe_unused]] auto bz = m_bz;

//   auto                  gx = m_gx;
//   auto                  gy = m_gy;
//   [[maybe_unused]] auto gz = m_gz;

//   if (m_extract_block_ghost)
//   {
//     [[maybe_unused]] int32_t nbCellsPerOct_ghosted =
//       dim == 2 ? size_x * size_y : size_x * size_y * size_z;

//     assertm(datah.num_cells() == nbCellsPerOct_ghosted,
//             "HDF5_Xdmf_Reader::read_quadrant_multi_mat_attribute wrong data size");
//   }
//   else
//   {
//     assertm(datah.num_cells() == nbCellsPerOct,
//             "HDF5_Xdmf_Reader::read_quadrant_multi_mat_attribute wrong data size");
//   }

//   // we need to gather data corresponding to a given scalar variable
//   using DataArrayScalar = Kokkos::View<real_t *, Kokkos::HostSpace>;

//   // array used for each material
//   DataArrayScalar dataVar("scalar_array_for_hdf5_io", static_cast<size_t>(nbCellsPerOct *
//   nbOcts));

//   for (int32_t imat = 0; imat < num_mats; imat++)
//   {
//     if (m_extract_block_ghost == false)
//     {
//       Kokkos::parallel_for(
//         "HDF5_Xdmf_Reader::read_quadrant_multi_mat_attribute",
//         Kokkos::RangePolicy<Kokkos::OpenMP>(iOct_begin, iOct_begin + nbOcts),
//         KOKKOS_LAMBDA(int32_t iOct) {
//           if (matph.get(iOct, imat))
//           {
//             const auto mat_id = matph.material_index(iOct, imat);
//             const auto var_id = mat_id * num_vars_per_mat + mat_varIdx;
//             for (int32_t iCell = 0; iCell < nbCellsPerOct; ++iCell)
//               dataVar(iCell + nbCellsPerOct * (iOct - iOct_begin)) = datah(iCell, var_id, iOct);
//           }
//           else // We set the value to nan to indicates that the value "does not exist"
//             for (int32_t iCell = 0; iCell < nbCellsPerOct; ++iCell)
//               dataVar(iCell + nbCellsPerOct * (iOct - iOct_begin)) = nan("");
//         });
//     }
//     else
//     {
//       Kokkos::parallel_for(
//         "HDF5_Xdmf_Reader::read_quadrant_multi_mat_attribute",
//         Kokkos::RangePolicy<Kokkos::OpenMP>(iOct_begin, iOct_begin + nbOcts),
//         KOKKOS_LAMBDA(int32_t iOct) {

//       // the only reason of the following dummy code to be here, is that cuda nvcc compile
//       doesn't
//       // support capturing variables inside the inner lambda inside a constexpr if
//       // TODO: remove ASAP nvcc is fixed
// #ifdef __NVCC__
//           [[maybe_unused]] int dummy = 0;
//           if (gx == 0 or gy == 0 or gz == 0 or bx == 0 or by == 0 or bz == 0 or size_x == 0 or
//               size_y == 0 or size_z == 0 or datah.num_cells() == 0 or dataVar.extent(0) == 0 or
//               mat_varIdx == -1 or iOct_begin == 0)
//             dummy++;
// #endif

//           if (matph.get(iOct, imat))
//           {
//             const auto mat_id = matph.material_index(iOct, imat);
//             const auto var_id = mat_id * num_vars_per_mat + mat_varIdx;

//             for (int32_t iCell = 0; iCell < nbCellsPerOct; ++iCell)
//             {
//               if constexpr (dim == 2)
//               {
//                 // iCell = ix + iy*m_bx
//                 const auto iy = iCell / bx;
//                 const auto ix = iCell - iy * bx;

//                 const auto iCell2 = (ix + gx) + size_x * (iy + gy);
//                 dataVar(iCell + nbCellsPerOct * (iOct - iOct_begin)) = datah(iCell2, var_id,
//                 iOct);
//               }
//               else if constexpr (dim == 3)
//               {
//                 // iCell = ix + iy*m_bx + iz*m_bx*m_by
//                 const auto iz = iCell / bx / by;
//                 const auto tmp = iCell - iz * bx * by;
//                 const auto iy = tmp / bx;
//                 const auto ix = tmp - iy * bx;

//                 auto iCell2 = (ix + gx) + size_x * (iy + gy) + size_x * size_y * (iz + gz);
//                 dataVar(iCell + nbCellsPerOct * (iOct - iOct_begin)) = datah(iCell2, var_id,
//                 iOct);
//               }
//             }
//           }
//           else // We set the value to nan to indicates that the value "does not exist"
//             for (int32_t iCell = 0; iCell < nbCellsPerOct; ++iCell)
//               dataVar(iCell + nbCellsPerOct * (iOct - iOct_begin)) = nan("");
//         });
//     }

//     // make sure all kokkos kernels are done before actual writing
//     Kokkos::fence();

//     // actual data writing
//     num_bytes += read_cc_attribute(std::to_string(imat) + varName, dataVar.data(), 0,
//     IO_CELL_SCALAR);
//   }

//   return num_bytes;

// } // HDF5_Xdmf_Reader<dim, device_t>::read_quadrant_multi_mat_attribute

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
template <typename data_t>
uint64_t
HDF5_Xdmf_Reader<dim, device_t>::io_hdf5_readv(HighFive::File &                 hdf5_file,
                                               std::string                      dataset_prefix,
                                               const std::string &              dataset_name,
                                               data_t *                         data_ptr,
                                               std::vector<std::size_t> const & count,
                                               std::vector<std::size_t> const & offset)
{
  if (hdf5_file.exist(dataset_prefix + dataset_name))
  {

    // create dataset
    HighFive::DataSet dataset = hdf5_file.getDataSet(dataset_prefix + dataset_name);

    auto xfer_props = HighFive::DataTransferProps{};
#ifdef KALYPSSO_CORE_USE_MPI
    xfer_props.add(HighFive::UseCollectiveIO{});
#endif // KALYPSSO_CORE_USE_MPI

    dataset.select(offset, count).read_raw(data_ptr, xfer_props);
#ifdef KALYPSSO_CORE_USE_MPI
    check_collective_io(xfer_props);
#endif // KALYPSSO_CORE_USE_MPI

    // return the total number of bytes written in local MPI process
    return sizeof(data_t) * std::reduce(count.begin(), count.end(), 1u, std::multiplies<>());
  }
  else
  {
    const std::string err_msg =
      "DataSet " + dataset_prefix + dataset_name + " doesn't exist; you can't perform a restart";
    Kokkos::abort(err_msg.c_str());
    return 0;
  }
} // HDF5_Xdmf_Reader<dim, device_t>::io_hdf5_readv

} // namespace kalypsso

#endif // KALYPSSO_CORE_HDF5_XDMF_READER_H_
