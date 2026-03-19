// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file HDF5_Xdmf_Writer_legacy.h
 */
#ifndef KALYPSSO_CORE_HDF5_XDMF_WRITER_LEGACY_H_
#define KALYPSSO_CORE_HDF5_XDMF_WRITER_LEGACY_H_

#include <hdf5.h>
#include <hdf5_hl.h>

#include <kalypsso/core/enums.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost

#include <kalypsso/utils/config/ConfigMap.h>

#include <kalypsso/utils/p4est/p4est_wrapper.h>
#include <kalypsso/core/kalypsso_core_base.h>
#include <kalypsso/core/AMRmesh_utils.h>
#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/brick_connectivity_utils.h>

#include <kalypsso/core/HDF5_IO_common.h>

#include <highfive/highfive.hpp>

#include <iostream> // for std::cerr

namespace kalypsso
{

// =====================================================================
// =====================================================================
/**
 * \brief HDF5 XDMF writer (legacy version).
 *
 * prefer using HDF5_Xdmf_Writer which is additionally able to write outside cell data.
 *
 * \tparam dim = 2 or 3
 */
template <size_t dim, typename device_t>
class HDF5_Xdmf_Writer_legacy
{

public:
  using DataArrayLeafHost_t = DataArrayLeafHost<real_t, device_t>;
  using DataArrayBlockHost_t = DataArrayBlock<dim, real_t, HostDevice>;

  using p4est_t = typename p4est::Wrapper<dim>;

  enum orchard_key_dump_type_t : uint32_t
  {
    REDUCED_LOCAL = 0,
    REDUCED_GLOBAL = 1,
    FULL = 2
  };

  /**
   * \brief Constructor.
   *
   * The writer is has no opened files by default except a main xmf file
   * that contains a collection of all the files that are to be written
   * next as a `Temporal` Collection.
   *
   * \param [in] forest p4est main object
   * \param [in] geometry p4est geometry (can be nullptr)
   * \param [in] ConfigMap
   * \param [in] write_xdmf_main is a boolean flag to enable/disable write the main xdmf file
   *
   */
  HDF5_Xdmf_Writer_legacy(forest_t<dim> *       forest,
                          geometry_t<dim> *     geom,
                          const ConfigMap &     config_map,
                          block_size_t<dim>     block_size,
                          coord_t<dim, int32_t> start_index,
                          std::string           xdmf_main = "_main");

  //! destructor
  virtual ~HDF5_Xdmf_Writer_legacy();

  /**
   * To be called as often as the mesh changes (refine/coarsen/balance/repartition).
   *
   * Recompute local (current MPI proc) and global (all MPI proc) number
   * of quadrants (forest leaves) and mesh block nodes.
   */
  void
  update_mesh_info();

  /**
   * \brief Open the HDF5 and XMF files for writing.
   *
   * Also includes this file inside the main xmf file, if needed.
   */
  void
  open(std::string basename, std::string outDir);

  /**
   * \brief Close the HDF5 and XMF files.
   *
   * Also closes the main xmf file, if it was opened.
   */
  void
  close();

  /**
   * \brief Write the header for the XMF and HDF5 files.
   *
   * The header includes the node information, connectivity information and
   * the treeid, level or mpi rank for each quadrant, if required.
   *
   * In the case of the XMF file, this will defined the topology and geometry
   * of the mesh and point to the relevant fields in the HDF5 file.
   */
  uint64_t
  write_header(double time);

  /**
   * \brief Write the XMF footer.
   *
   * Closes all the XML tags.
   */
  int
  write_footer();


  /**
   * \brief Write a node-centered or cell-centered attribute.
   *
   * \param [in] name The name of the attribute.
   * \param [in] data The data to be written.
   * \param [in] dimData In the case of a vector, this is the dimension of each
   * element in the vector field.
   * \param [in] ftype The type of the attribute. See supported types in
   *  the io_attribute_type_t enum.
   * \param [in] dtype The type of the data we are writing. This is given as a
   * native HDF5 type. See the types defined in the H5Tpublic.h header.
   * \param [in] wtype The type of the data written to the file. The
   * conversion between the data type and the written data is handled
   * by HDF5.
   */
  template <typename data_t>
  uint64_t
  write_attribute(const std::string & name,
                  data_t *            data,
                  size_t              dimData,
                  io_attribute_type_t ftype);

  /**
   * \brief Write all cell-centered scalar attributes.
   *
   * \param [in] datah a host Kokkos View with the user data
   * \param [in] varIdx scalar index identify scalar field to write
   * \param [in] varName variable name
   */
  uint64_t
  write_quadrant_attribute(DataArrayLeafHost_t datah, int32_t varIdx, const std::string varName);

  /**
   * \brief Write all cell-centered scalar attributes when block amr is enabled.
   *
   * \param [in] datah a host Kokkos View with the user data (block data)
   * \param [in] varIdx scalar index identify scalar field to write
   * \param [in] varName variable name
   */
  uint64_t
  write_quadrant_attribute(DataArrayBlockHost_t datah, int32_t varIdx, const std::string varName);

  forest_t<dim> *   m_forest;   //!< p4est main object
  geometry_t<dim> * m_geom;     //!< p4est geometry
  std::string       m_basename; //!< the base name of the two files

  const ConfigMap & m_config_map;

  bool m_write_mesh_info;                  //!< write mesh info (oct level, mpi proc, ...)
  bool m_write_tree;                       //!< default write_mesh_info (false)
  bool m_write_level;                      //!< default write_mesh_info (false)
  bool m_write_rank;                       //!< default write_mesh_info (false)
  bool m_write_reduced_orchard_key_global; //!< default write_mesh_info (false)
  bool m_write_reduced_orchard_key_local;  //!< default write_mesh_info (false)
  bool m_write_full_orchard_key;           //!< default write_mesh_info (false)
  bool m_write_at_domain_border;           //!< default write_mesh_info (false)
  bool m_write_at_tree_border;             //!< default write_mesh_info (false)

  bool m_write_block_data; //!< if true, expect a DataArrayBlock object

  //! sizes of the block of cells per octant
  block_size_t<dim> m_block_size;

  //! start index (used when user wants to extract a block from the original data block of cells)
  coord_t<dim, int32_t> m_istart;

  uint8_t  m_nbNodesPerCell; //!< 4 (2d) or 8 (3d)
  uint32_t m_nbCellsPerLeaf; //!< only useful for block amr

  HighFive::File * m_hdf5_file;      //!< HDF5 file descriptor
  FILE *           m_xdmf_file;      //!< XDMF file descriptor
  FILE *           m_main_xdmf_file; //!< main XDMF file descriptor

  bool m_write_iOct; //!< default false

  int m_mpiRank; //!< mpi rank (as stored in p4est)

  // store information about nodes for writing node data

  uint64_t m_global_num_nodes; //!< global accumulated (all MPI proc) number of nodes
  uint32_t m_local_num_nodes;  //!< local (current MPI proc) number of nodes
  uint64_t m_global_num_quads; //!< global accumulated (all MPI proc) number of quads/octs
  uint32_t m_local_num_quads;  //!< local (current MPI proc) number of quads/octs

private:
  /*
   * XMDF utilities.
   */

  /**
   * \brief Write the header of the main XMF file.
   */
  void
  io_xdmf_write_main_header();

  /**
   * \brief Write the XMF header information: topology and geometry.
   */
  void
  io_xdmf_write_header(double time);

  /**
   * \brief Write the include for the current file.
   */
  void
  io_xdmf_write_main_include(const std::string & name);

  /**
   * \brief Write information about an attribute.
   *
   * \param [in] fd   file descriptor for xmf file
   * \param [in] basename The basename.
   * \param [in] name The name of the attribute.
   * \param [in] type The type.
   * \param [in] dims The dimensions of the attribute. If it is a scalar, dims[1]
   * will be ignored.
   */
  void
  io_xdmf_write_attribute(const std::string &         name,
                          const std::string &         number_type,
                          io_attribute_type_t         type,
                          std::vector<size_t> const & dims);

  /**
   * \brief Close the remaining tags for the main file.
   */
  void
  io_xdmf_write_main_footer();

  /**
   * \brief Close all remaining tags.
   *
   * \param[in] fd xmff file descriptor
   */
  void
  io_xdmf_write_footer();

  /*
   * HDF5 utilities.
   */
  /**
   * \brief Write a given dataset into the HDF5 file.
   *
   * \param [in] fd An open file descriptor to a HDF5 file.
   * \param [in] name The name of the dataset we are writing.
   * \param [in] data The data to write.
   * \param [in] dtype_id The native HDF5 type of the given data.
   * \param [in] wtype_id The native HDF5 type of the written data.
   * \param [in] rank The rank of the dataset. 1 if it is a vector, 2 for a matrix.
   * \param [in] dims The global dimensions of the dataset.
   * \param [in] count The local dimensions of the dataset.
   * \param [in] start The offset of the local data with respect to the global
   * positioning.
   *
   * \return total number of bytes written.
   *
   * \sa H5TPublic.h
   */
  template <typename data_t>
  uint64_t
  io_hdf5_writev(HighFive::File &                 hdf5_file,
                 const std::string &              dataset_name,
                 data_t *                         data_ptr,
                 std::vector<std::size_t> const & dims,
                 std::vector<std::size_t> const & count,
                 std::vector<std::size_t> const & offset);

  /**
   * \brief Compute and write the coordinates of all the mesh nodes.
   *
   * Mesh coordinates is only necessary if one wants to load data into paraview as a VTU
   * (VTK Unstructured mesh)
   *
   * \return number of bytes written
   */
  uint64_t
  io_hdf5_write_coordinates();

  /**
   * \brief Compute and write the connectivity information for each quadrant.
   * \return number of bytes written
   */
  uint64_t
  io_hdf5_write_connectivity();

  /**
   * \brief Compute and write the tree id for each quadrant.
   * \return number of bytes written
   */
  uint64_t
  io_hdf5_write_tree();

  /**
   * \brief Compute and write the level for each quadrant.
   * \return number of bytes written
   */
  uint64_t
  io_hdf5_write_level();

  /**
   * \brief Write local quadrant id (local to current MPI process).
   * Purely for debug purpose.
   * \return number of bytes written
   */
  uint64_t
  io_hdf5_write_iOct();

  /**
   * \brief Compute and write the MPI rank for each quadrant.
   * \return number of bytes written
   */
  uint64_t
  io_hdf5_write_rank();

  /**
   * \brief Compute and write the reduced orchard key for each quadrant.
   *
   * reduced orchard key is obtain by considering only the octant part (discard level and tree),
   * and then right shift the bits by level_max (to keep only most significant bits)
   *
   *
   * \param[in] global if global is true, compute reduced key with tree, else without tree
   * \return number of bytes written
   */
  uint64_t
  io_hdf5_write_reduced_orchard_key(orchard_key_dump_type_t type);

  /**
   * \brief Compute and write for each quadrant a boolean value indicating if quadrant touches
   * external border.
   * \return number of bytes written
   */
  uint64_t
  io_hdf5_write_at_domain_border();

  /**
   * \brief Compute and write for each quadrant a boolean value indicating if quadrant touches
   * tree border.
   * \return number of bytes written
   */
  uint64_t
  io_hdf5_write_at_tree_border();

  //! get cell linear index from (jx,jy,jz)
  int32_t
  subCellIndex(int jx, int jy, int jz);

  //! get node linear index from (jx,jy,jz)
  int32_t
  subNodeIndex(int jx, int jy, int jz);

public:
  //! change block size for writing DataArrayBlock.
  void
  set_block_mode(block_size_t<dim> block_size, coord_t<dim> start_index);

  //! switch to leaf mode for writing DataArrayLeaf.
  //! one cell per (octree) leaf.
  void
  set_leaf_mode();

  //! Configure if we want to write mesh info.
  //!
  //! \param[in] value, if true we write all mesh info if not specified in input parameter
  //! \note if write_mesh_info is set in input parameter it will be used in constructor
  //! outside of constructor this method can be used to change behavior of writer.
  void
  set_write_mesh_info(bool value)
  {
    m_write_mesh_info = value;
    m_write_tree = m_config_map.getBool("output", "write_mesh_tree", m_write_mesh_info);
    m_write_level = m_config_map.getBool("output", "write_mesh_level", m_write_mesh_info);
    m_write_rank = m_config_map.getBool("output", "write_mesh_rank", m_write_mesh_info);
    m_write_reduced_orchard_key_global =
      m_config_map.getBool("output", "write_mesh_reduced_orchard_key_global", m_write_mesh_info);
    m_write_reduced_orchard_key_local =
      m_config_map.getBool("output", "write_mesh_reduced_orchard_key_local", m_write_mesh_info);
    m_write_full_orchard_key =
      m_config_map.getBool("output", "write_mesh_full_orchard_key", m_write_mesh_info);
    m_write_at_domain_border =
      m_config_map.getBool("output", "write_mesh_at_domain_border", m_write_mesh_info);
    m_write_at_tree_border =
      m_config_map.getBool("output", "write_mesh_at_tree_border", m_write_mesh_info);
  } // set_write_mesh_info

}; // class HDF5_Xdmf_Writer_legacy

// =============================================================================
// =============================================================================
// class HDF5_Xdmf_Writer_legacy definition
// =============================================================================
// =============================================================================

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
HDF5_Xdmf_Writer_legacy<dim, device_t>::HDF5_Xdmf_Writer_legacy(forest_t<dim> *       forest,
                                                                geometry_t<dim> *     geom,
                                                                const ConfigMap &     config_map,
                                                                block_size_t<dim>     block_size,
                                                                coord_t<dim, int32_t> start_index,
                                                                std::string xdmf_main_suffix)
  : m_forest(forest)
  , m_geom(geom)
  , m_basename("")
  , m_config_map(config_map)
  , m_write_mesh_info(false)
  , m_write_tree(false)
  , m_write_level(false)
  , m_write_rank(false)
  , m_write_reduced_orchard_key_global(false)
  , m_write_reduced_orchard_key_local(false)
  , m_write_full_orchard_key(false)
  , m_write_at_domain_border(false)
  , m_write_at_tree_border(false)
  , m_write_block_data(Kokkos::dim_prod(block_size) > 1)
  , m_block_size(block_size)
  , m_istart(start_index)
  , m_nbNodesPerCell(dim == TWO_D ? IO_NODES_PER_CELL_2D : IO_NODES_PER_CELL_3D)
  , m_nbCellsPerLeaf(static_cast<uint32_t>(Kokkos::dim_prod(block_size)))
{

  // set write_mesh_info from config, but it can be changed afterwards
  set_write_mesh_info(m_config_map.getBool("output", "write_mesh_info", false));

  m_write_iOct = m_config_map.getBool("output", "write_iOct", false);

  update_mesh_info();

  m_hdf5_file = 0;
  m_xdmf_file = nullptr;

  m_mpiRank = m_forest->mpirank;

  // is actually hdf5 enabled ?
  bool hdf5_enabled = m_config_map.getBool("output", "hdf5_enabled", false);

  if (m_mpiRank == 0 and hdf5_enabled)
  {

    std::string outputPrefix = m_config_map.getString("output", "outputPrefix", "output");
    std::string outputDir = m_config_map.getString("output", "outputDir", "output");

    std::string filename;
    filename = outputDir + "/" + outputPrefix + xdmf_main_suffix + ".xmf";

    // KALYPSSO_GLOBAL_INFOF("Writing main XMDF file \"%s\".\n", filename.c_str());
    m_main_xdmf_file = fopen(filename.c_str(), "w");
    io_xdmf_write_main_header();
  }
  else
  {
    m_main_xdmf_file = nullptr;
  }

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::HDF5_Xdmf_Writer_legacy

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
HDF5_Xdmf_Writer_legacy<dim, device_t>::~HDF5_Xdmf_Writer_legacy()
{

  /*
   * Only rank 0 needs to close the main xdmf file.
   */
  if (m_mpiRank == 0 and m_main_xdmf_file != nullptr)
  {
    io_xdmf_write_main_footer();

    fflush(m_main_xdmf_file);
    fclose(m_main_xdmf_file);
    m_main_xdmf_file = nullptr;
  }

  // close other file
  close();

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::~HDF5_Xdmf_Writer_legacy

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer_legacy<dim, device_t>::update_mesh_info()
{

  m_local_num_quads = static_cast<uint32_t>(m_forest->local_num_quadrants);
  m_global_num_quads = static_cast<uint64_t>(m_forest->global_num_quadrants);

  m_local_num_nodes = m_nbNodesPerCell * m_local_num_quads;
  m_global_num_nodes = m_nbNodesPerCell * m_global_num_quads;

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::update_mesh_info

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer_legacy<dim, device_t>::open(std::string basename, std::string outDir)
{
  // build filename
  std::string filename = basename + ".h5";
  std::string full_path = outDir + "/" + filename;
  m_basename = basename;

  /*
   * create file access property list
   */
  HighFive::FileAccessProps fapl;
#ifdef KALYPSSO_CORE_USE_MPI
  fapl.add(HighFive::MPIOFileAccess{ m_forest->mpicomm, MPI_INFO_NULL });
  // all metadata are written using collective IO
  fapl.add(HighFive::MPIOCollectiveMetadata{});
#endif // KALYPSSO_CORE_USE_MPI

  /*
   * Open parallel HDF5 resources.
   */
  m_hdf5_file = new HighFive::File(full_path, HighFive::File::Truncate, fapl);

  // open xdmf files (one for each hdf5, a main xdmf file)
  if (m_mpiRank == 0)
  {

    filename = basename + ".xmf";
    full_path = outDir + "/" + filename;
    m_xdmf_file = fopen(full_path.c_str(), "w");

    if (m_main_xdmf_file)
    {
      io_xdmf_write_main_include(filename);
    }
  }

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::open

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer_legacy<dim, device_t>::close()
{
  // close HDF5
  if (m_hdf5_file != nullptr)
  {
    m_hdf5_file->flush();
    delete m_hdf5_file;
    m_hdf5_file = nullptr;
  }

  // close XDMF file descriptor
  if (m_xdmf_file)
  {
    fflush(m_xdmf_file);
    fclose(m_xdmf_file);
    m_xdmf_file = nullptr;
  }

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::close

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer_legacy<dim, device_t>::write_header(double time)
{

  // write the xmdf file first
  if (m_mpiRank == 0)
  {
    io_xdmf_write_header(time);
  }

  uint64_t num_bytes = 0;

  // and write stuff into the hdf file
  num_bytes += io_hdf5_write_coordinates();
  num_bytes += io_hdf5_write_connectivity();
  num_bytes += io_hdf5_write_level();
  num_bytes += io_hdf5_write_tree();
  num_bytes += io_hdf5_write_rank();
  num_bytes += io_hdf5_write_iOct();
  if (m_write_reduced_orchard_key_global)
    num_bytes += io_hdf5_write_reduced_orchard_key(REDUCED_GLOBAL);
  if (m_write_reduced_orchard_key_local)
    num_bytes += io_hdf5_write_reduced_orchard_key(REDUCED_LOCAL);
  if (m_write_full_orchard_key)
    num_bytes += io_hdf5_write_reduced_orchard_key(FULL);
  num_bytes += io_hdf5_write_at_domain_border();
  num_bytes += io_hdf5_write_at_tree_border();

  return num_bytes;

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::write_header

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
int
HDF5_Xdmf_Writer_legacy<dim, device_t>::write_footer()
{
  if (m_mpiRank == 0)
  {
    io_xdmf_write_footer();
  }

  return 0;

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::write_footer

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
template <typename data_t>
uint64_t
HDF5_Xdmf_Writer_legacy<dim, device_t>::write_attribute(const std::string & name,
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

  if (ftype == IO_CELL_SCALAR || ftype == IO_CELL_VECTOR)
  {

    dims[0] = static_cast<size_t>(m_forest->global_num_quadrants) * m_nbCellsPerLeaf;
    if (dimData > 0)
      dims[1] = dimData;

    count[0] = static_cast<size_t>(m_forest->local_num_quadrants) * m_nbCellsPerLeaf;
    if (dimData > 0)
      count[1] = dims[1];

    // get global index of the first octant of current mpi processor
    start[0] = static_cast<size_t>(m_forest->global_first_quadrant[m_mpiRank]) * m_nbCellsPerLeaf;
    if (dimData > 0)
      start[1] = 0;
  }
  else
  {

    // is this relevant ?

    // dims[0] = m_global_num_nodes;
    // if (dimData > 0)
    //   dims[1] = dim;

    // count[0] = m_local_num_nodes;
    // if (dimData > 0)
    //   count[1] = dims[1];

    // start[0] = m_start_nodes;
    // if (dimData > 0)
    //   start[1] = 0;
  }

  if (m_mpiRank == 0)
  {
    const char * dtype_str = hdf5_native_type_to_string<data_t>();
    io_xdmf_write_attribute(name, dtype_str, ftype, dims);
  }

  num_bytes += io_hdf5_writev(*m_hdf5_file, name, data, dims, count, start);

  return num_bytes;

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::write_attribute

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer_legacy<dim, device_t>::write_quadrant_attribute(DataArrayLeafHost_t datah,
                                                                 int32_t             varIdx,
                                                                 const std::string   varName)
{


  uint64_t num_bytes = 0;

  // if DataArray has a left layout, we only need to define
  // a slice to actual scalar data
  // if DataArrayLeafHost_t has right layout, we need to actually extract
  // the slide so that it is memory contiguous
  if (std::is_same<typename DataArrayLeafHost_t::array_layout, Kokkos::LayoutLeft>::value)
  {

    auto dataVar = Kokkos::subview(datah, Kokkos::ALL(), varIdx);

    // actual data writing
    num_bytes += write_attribute(varName, dataVar.data(), 0, IO_CELL_SCALAR);
  }
  else
  {

    using DataArrayScalar = Kokkos::View<real_t *, Kokkos::HostSpace>;

    DataArrayScalar dataVar = DataArrayScalar("scalar_array_for_hdf5_io", datah.extent(0));

    size_t nbOcts = datah.extent(0);

    Kokkos::parallel_for(
      "HDF5_Xdmf_Writer_legacy::write_quadrant_attribute",
      Kokkos::RangePolicy<Kokkos::OpenMP>(0, nbOcts),
      KOKKOS_LAMBDA(uint32_t iOct) { dataVar(iOct) = datah(iOct, varIdx); });

    // actual data writing
    num_bytes += write_attribute(varName, dataVar.data(), 0, IO_CELL_SCALAR);
  }

  return num_bytes;

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::write_quadrant_attribute

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer_legacy<dim, device_t>::write_quadrant_attribute(DataArrayBlockHost_t datah,
                                                                 int32_t              varIdx,
                                                                 const std::string    varName)
{

  uint64_t num_bytes = 0;
  int32_t  nbCellsPerOct = Kokkos::dim_prod(m_block_size);

  int32_t nbOcts = datah.num_quadrants();

  // we need to gather data corresponding to a given scalar variable
  using DataArrayScalar = Kokkos::View<real_t *, Kokkos::HostSpace>;

  // remember that
  // - data.extent(0) is the number of cells per octant
  // - data.extent(1) is the number of scalar fields
  // - data.extent(2) is the total number of oct in current MPI process
  DataArrayScalar dataVar =
    DataArrayScalar("scalar_array_for_hdf5_io", static_cast<size_t>(nbCellsPerOct * nbOcts));

  auto block_size = m_block_size;
  auto istart = m_istart;

  Kokkos::parallel_for(
    "HDF5_Xdmf_Writer_legacy::write_quadrant_attribute",
    Kokkos::RangePolicy<Kokkos::OpenMP>(0, nbOcts * nbCellsPerOct),
    KOKKOS_LAMBDA(int32_t global_index) {
      const auto iOct_local = global_index / nbCellsPerOct;
      const auto cell_index = global_index - iOct_local * nbCellsPerOct;
      const auto coords = cellindex_to_coord<dim>(cell_index, block_size);

    // the only reason of the following dummy code to be here, is that cuda nvcc compile doesn't
    // support capturing variables inside the inner lambda inside a constexpr if
    //
    // strangely, nvc++ is ok and don't need that
    // TODO: remove theses lines when nvcc will have support for this.
#ifdef __NVCC__
      [[maybe_unused]] int dummy = 0;
      if (dataVar.extent(0) == 0 or datah.num_cells() == 0 or varIdx == 0 or istart[IX] == 0)
        dummy++;
#endif
      if constexpr (dim == 2)
      {
        dataVar(cell_index + nbCellsPerOct * iOct_local) =
          datah(coords[IX] + istart[IX], coords[IY] + istart[IY], varIdx, iOct_local);
      }
      else if constexpr (dim == 3)
      {
        dataVar(cell_index + nbCellsPerOct * iOct_local) = datah(coords[IX] + istart[IX],
                                                                 coords[IY] + istart[IY],
                                                                 coords[IZ] + istart[IZ],
                                                                 varIdx,
                                                                 iOct_local);
      }
    });

  // actual data writing
  num_bytes += write_attribute(varName, dataVar.data(), 0, IO_CELL_SCALAR);

  return num_bytes;

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::write_quadrant_attribute

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
template <typename data_t>
uint64_t
HDF5_Xdmf_Writer_legacy<dim, device_t>::io_hdf5_writev(HighFive::File &    hdf5_file,
                                                       const std::string & dataset_name,
                                                       data_t *            data_ptr,
                                                       std::vector<std::size_t> const & dims,
                                                       std::vector<std::size_t> const & count,
                                                       std::vector<std::size_t> const & offset)
{
  // create dataset
  HighFive::DataSet dataset =
    hdf5_file.createDataSet<data_t>(dataset_name, HighFive::DataSpace(dims));

  auto xfer_props = HighFive::DataTransferProps{};
#ifdef KALYPSSO_CORE_USE_MPI
  xfer_props.add(HighFive::UseCollectiveIO{});
#endif // KALYPSSO_CORE_USE_MPI

  dataset.select(offset, count).write_raw(data_ptr, xfer_props);
#ifdef KALYPSSO_CORE_USE_MPI
  check_collective_io(xfer_props);
#endif // KALYPSSO_CORE_USE_MPI

  // Let's ensure that everything has been written do disk.
  hdf5_file.flush();

  // return the total number of bytes written in local MPI process
  return sizeof(data_t) * std::reduce(count.begin(), count.end(), 1u, std::multiplies<>());

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::io_hdf5_writev

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer_legacy<dim, device_t>::io_hdf5_write_coordinates()
{

  const auto NB_CHILDREN = p4est_t::NB_CHILDREN;
  const auto quadrant_array_index = &p4est_t::quadrant_array_index;
  const auto tree_array_index = &p4est_t::tree_array_index;

  uint64_t num_bytes = 0;

  if (m_write_block_data)
  {
    uint32_t nbNodesPerLeaf = static_cast<uint32_t>(Kokkos::dim_prod(m_block_size + 1));

    uint64_t totalNumOfCoords = 3 * m_local_num_quads * nbNodesPerLeaf;

    std::vector<float> data(totalNumOfCoords);

    /*
     * construct the list of node coordinates
     */

    // bool use_block_amr = true;
    std::array<int, 3> bSize{ m_block_size[IX], m_block_size[IY], 1 };
    if constexpr (dim == 3)
    {
      bSize[IZ] = m_block_size[IZ];
    }

    // array of local trees
    sc_array_t * trees = m_forest->trees;

    uint32_t iOct = 0;

    // loop over all local tree
    for (auto jt = m_forest->first_local_tree; jt <= m_forest->last_local_tree; ++jt)
    {
      // get current tree
      tree_t<dim> * tree = tree_array_index(trees, jt);

      // get quadrant array of current tree
      sc_array_t * quadrants = &(tree->quadrants);

      // loop over all local quadrant
      for (size_t jq = 0; jq < quadrants->elem_count; ++jq)
      {
        quadrant_t<dim> * q = quadrant_array_index(quadrants, jq);

        // the offset of the first node of the current quadrant
        size_t offset = 3 * nbNodesPerLeaf * iOct;

        // compute the coordinates of the NB_CHILDREN nodes and add them to
        // the data array

        io_fill_coordinates<dim>(m_forest, m_geom, jt, q, &(data[offset]), bSize);

        // next quad/octant
        ++iOct;

      } // end for quadrant/octant in current tree

    } // end for tree

    // get prepared for hdf5 writing

    std::vector<std::size_t> dims{ 0, 0 };
    std::vector<std::size_t> count{ 0, 0 };
    std::vector<std::size_t> start{ 0, 0 };

    // get the dimensions and offset of the node coordinates array
    dims[0] = m_global_num_quads * nbNodesPerLeaf;
    dims[1] = 3;

    count[0] = m_local_num_quads * nbNodesPerLeaf;
    count[1] = 3;

    // get global index of the first octant of current mpi processor
    start[0] = static_cast<size_t>(m_forest->global_first_quadrant[m_mpiRank]) * nbNodesPerLeaf;
    start[1] = 0;

    // write the node coordinates
    num_bytes += io_hdf5_writev(*m_hdf5_file, "coordinates", data.data(), dims, count, start);
  }
  else // cell-based AMR, one cell per quadrant
  {

    // not using block AMR
    std::array<int, 3> bSize{ -1, -1, -1 };

    // array with all local nodes coordinates
    std::vector<float> data(3 * m_local_num_nodes);

    /*
     * construct the list of node coordinates
     */

    // array of local trees
    sc_array_t * trees = m_forest->trees;

    uint32_t iOct = 0;

    // loop over all local tree
    for (auto jt = m_forest->first_local_tree; jt <= m_forest->last_local_tree; ++jt)
    {
      // get current tree
      tree_t<dim> * tree = tree_array_index(trees, jt);

      // get quadrant array of current tree
      sc_array_t * quadrants = &(tree->quadrants);

      // loop over all local quadrant
      for (size_t jq = 0; jq < quadrants->elem_count; ++jq)
      {
        quadrant_t<dim> * q = quadrant_array_index(quadrants, jq);

        // the offset of the first node of the current quadrant
        size_t offset = 3 * NB_CHILDREN * iOct;

        // compute the coordinates of the NB_CHILDREN nodes and add them to
        // the data array
        io_fill_coordinates<dim>(m_forest, m_geom, jt, q, &(data[offset]), bSize);

        // next quad/octant
        ++iOct;

      } // end for quadrant/octant in current tree

    } // end for tree

    // get prepared for hdf5 writing

    std::vector<std::size_t> dims{ 0, 0 };
    std::vector<std::size_t> count{ 0, 0 };
    std::vector<std::size_t> start{ 0, 0 };

    // get the dimensions and offset of the node coordinates array
    dims[0] = m_global_num_nodes;
    dims[1] = 3;

    count[0] = m_local_num_nodes;
    count[1] = 3;

    // get global index of the first octant of current mpi processor
    start[0] = m_nbNodesPerCell * static_cast<size_t>(m_forest->global_first_quadrant[m_mpiRank]);
    start[1] = 0;

    // write the node coordinates
    num_bytes += io_hdf5_writev(*m_hdf5_file, "coordinates", data.data(), dims, count, start);
  }

  return num_bytes;

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::io_hdf5_write_coordinates

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer_legacy<dim, device_t>::io_hdf5_write_connectivity()
{

  uint64_t num_bytes = 0;

  if (m_write_block_data)
  {

    uint32_t nbNodesPerLeaf = static_cast<uint32_t>(Kokkos::dim_prod(m_block_size + 1));

    std::vector<int64_t> data(m_local_num_quads * m_nbCellsPerLeaf * m_nbNodesPerCell);

    // first node of current MPI process
    uint64_t globalNodeOffset =
      nbNodesPerLeaf * static_cast<uint64_t>(m_forest->global_first_quadrant[m_mpiRank]);

    uint64_t nbConnectivityPerLeaf = m_nbCellsPerLeaf * m_nbNodesPerCell;

    // get connectivity data
    for (uint32_t iLeaf = 0; iLeaf < m_local_num_quads; ++iLeaf)
    {
      uint64_t localNodeOffset = nbNodesPerLeaf * iLeaf;

      // sweep subcells
      int nz = 1;
      if constexpr (dim == 3)
      {
        nz = m_block_size[IZ];
      }

      for (int jz = 0; jz < nz; ++jz)
      {
        for (int jy = 0; jy < m_block_size[IY]; ++jy)
        {
          for (int jx = 0; jx < m_block_size[IX]; ++jx)
          {

            int64_t nodeOffset = static_cast<int64_t>(globalNodeOffset + localNodeOffset);

            uint64_t idx = nbConnectivityPerLeaf * iLeaf +
                           static_cast<uint64_t>(subCellIndex(jx, jy, jz)) * m_nbNodesPerCell;

            data[idx + 0] = nodeOffset + subNodeIndex(jx, jy, jz);
            data[idx + 1] = nodeOffset + subNodeIndex(jx + 1, jy, jz);
            data[idx + 2] = nodeOffset + subNodeIndex(jx + 1, jy + 1, jz);
            data[idx + 3] = nodeOffset + subNodeIndex(jx, jy + 1, jz);

            if (dim == THREE_D)
            {
              data[idx + 4] = nodeOffset + subNodeIndex(jx, jy, jz + 1);
              data[idx + 5] = nodeOffset + subNodeIndex(jx + 1, jy, jz + 1);
              data[idx + 6] = nodeOffset + subNodeIndex(jx + 1, jy + 1, jz + 1);
              data[idx + 7] = nodeOffset + subNodeIndex(jx, jy + 1, jz + 1);
            }

          } // end for jx
        } // end for jy
      } // end for jz

    } // end for iLeaf

    // now write connectivity with hdf5

    std::vector<std::size_t> dims{ 0, 0 };
    std::vector<std::size_t> count{ 0, 0 };
    std::vector<std::size_t> start{ 0, 0 };

    // get the dimensions and offsets for each connectivity array
    dims[0] = m_global_num_quads * m_nbCellsPerLeaf;
    dims[1] = m_nbNodesPerCell;

    count[0] = m_local_num_quads * m_nbCellsPerLeaf;
    count[1] = m_nbNodesPerCell;

    start[0] = static_cast<size_t>(m_forest->global_first_quadrant[m_mpiRank]) * m_nbCellsPerLeaf;
    start[1] = 0;

    // write the node coordinates
    num_bytes += io_hdf5_writev(*m_hdf5_file, "connectivity", data.data(), dims, count, start);
  }
  else
  { // regular AMR mesh, i.e. one cell per quad/oct

    uint32_t node[8] = { 0, 1, 3, 2, 0, 0, 0, 0 };

    if (dim == 3)
    {
      node[4] = 4;
      node[5] = 5;
      node[6] = 7;
      node[7] = 6;
    }

    std::vector<int64_t> data(m_local_num_quads * m_nbNodesPerCell);

    int64_t in = 0;

    // get connectivity data
    for (uint32_t i = 0; i < m_local_num_quads; ++i)
    {

      for (uint32_t j = 0; j < m_nbNodesPerCell; ++j)
      {
        uint64_t idx = m_nbNodesPerCell * i + j;

        data[idx] =
          static_cast<int64_t>(m_nbNodesPerCell) * m_forest->global_first_quadrant[m_mpiRank] + in +
          static_cast<int64_t>(node[j]);
      } // end for j

      in += m_nbNodesPerCell;

    } // end for i

    // now write connectivity with hdf5

    std::vector<std::size_t> dims{ 0, 0 };
    std::vector<std::size_t> count{ 0, 0 };
    std::vector<std::size_t> start{ 0, 0 };

    // get the dimensions and offsets for each connectivity array
    dims[0] = m_global_num_quads;
    dims[1] = m_nbNodesPerCell;

    count[0] = m_local_num_quads;
    count[1] = m_nbNodesPerCell;

    start[0] = static_cast<size_t>(m_forest->global_first_quadrant[m_mpiRank]);
    start[1] = 0;

    // write the node coordinates
    num_bytes += io_hdf5_writev(*m_hdf5_file, "connectivity", data.data(), dims, count, start);
  }

  return num_bytes;

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::io_hdf5_write_connectivity

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer_legacy<dim, device_t>::io_hdf5_write_level()
{

  // const auto NB_CHILDREN = p4est_t::NB_CHILDREN;
  const auto quadrant_array_index = &p4est_t::quadrant_array_index;
  const auto tree_array_index = &p4est_t::tree_array_index;

  if (!this->m_write_level)
  {
    return 0;
  }

  uint32_t nbData = m_local_num_quads * m_nbCellsPerLeaf;

  std::vector<int> data(nbData);

  // gather level for each local quadrant
  // p4est::locidx_t iOct = 0;
  sc_array_t * trees = m_forest->trees;

  uint32_t i = 0;

  // loop over all local tree
  for (auto jt = m_forest->first_local_tree; jt <= m_forest->last_local_tree; ++jt)
  {
    // get current tree
    tree_t<dim> * tree = tree_array_index(trees, jt);

    // get quadrant array of current tree
    sc_array_t * quadrants = &(tree->quadrants);

    // loop over all local quadrant
    for (size_t jq = 0; jq < quadrants->elem_count; ++jq)
    {
      quadrant_t<dim> * q = quadrant_array_index(quadrants, jq);

      for (uint32_t j = 0; j < m_nbCellsPerLeaf; ++j)
      {
        data[i] = q->level;
        ++i;
      }

      // data[iOct] = q->level;

      // next quad/octant
      //++iOct;

    } // end for quadrant/octant in current tree

  } // end for tree

  return write_attribute("level", data.data(), 0, IO_CELL_SCALAR);

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::io_hdf5_write_level

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer_legacy<dim, device_t>::io_hdf5_write_tree()
{

  // const auto NB_CHILDREN = p4est_t::NB_CHILDREN;
  // const auto quadrant_array_index = &p4est_t::quadrant_array_index;
  const auto tree_array_index = &p4est_t::tree_array_index;

  if (!this->m_write_tree)
  {
    return 0;
  }

  uint32_t nbData = m_local_num_quads * m_nbCellsPerLeaf;

  std::vector<int> data(nbData);

  // gather level for each local quadrant
  // p4est::locidx_t iOct = 0;
  sc_array_t * trees = m_forest->trees;

  uint32_t i = 0;

  // loop over all local tree
  for (auto jt = m_forest->first_local_tree; jt <= m_forest->last_local_tree; ++jt)
  {
    // get current tree
    tree_t<dim> * tree = tree_array_index(trees, jt);

    // get quadrant array of current tree
    sc_array_t * quadrants = &(tree->quadrants);

    // loop over all local quadrant
    for (size_t jq = 0; jq < quadrants->elem_count; ++jq)
    {
      // quadrant_t<dim> * q = quadrant_array_index(quadrants, jq);

      for (uint32_t j = 0; j < m_nbCellsPerLeaf; ++j)
      {
        data[i] = jt;
        ++i;
      }

      // next quad/octant
      //++iOct;

    } // end for quadrant/octant in current tree

  } // end for tree

  return write_attribute("treeid", data.data(), 0, IO_CELL_SCALAR);

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::io_hdf5_write_tree

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer_legacy<dim, device_t>::io_hdf5_write_iOct()
{

  if (!this->m_write_iOct)
  {
    return 0;
  }

  uint32_t nbData = m_local_num_quads * m_nbCellsPerLeaf;

  std::vector<uint32_t> data(nbData);

  // gather level for each local quadrant
  uint32_t i = 0;
  for (uint32_t iLeaf = 0; iLeaf < m_local_num_quads; ++iLeaf)
  {
    for (uint32_t j = 0; j < m_nbCellsPerLeaf; ++j)
    {
      data[i] = iLeaf;
      ++i;
    }
  }

  return write_attribute("iOct", data.data(), 0, IO_CELL_SCALAR);

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::io_hdf5_write_iOct

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer_legacy<dim, device_t>::io_hdf5_write_rank()
{

  if (!this->m_write_rank)
  {
    return 0;
  }

  uint32_t nbData = m_local_num_quads * m_nbCellsPerLeaf;

  std::vector<int> data(nbData);

  // gather rank for each local quadrant
  for (uint32_t i = 0; i < nbData; ++i)
  {
    data[i] = m_mpiRank;
  }

  return write_attribute("rank", data.data(), 0, IO_CELL_SCALAR);

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::io_hdf5_write_rank

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer_legacy<dim, device_t>::io_hdf5_write_reduced_orchard_key(
  orchard_key_dump_type_t type)
{

  // const auto NB_CHILDREN = p4est_t::NB_CHILDREN;
  const auto quadrant_array_index = &p4est_t::quadrant_array_index;
  const auto tree_array_index = &p4est_t::tree_array_index;

  uint32_t nbData = m_local_num_quads * m_nbCellsPerLeaf;

  std::vector<uint64_t> data(nbData);
  size_t                icell = 0;

  // check if connectivity is either "unit" or "brick"
  auto conn_name = m_config_map.getString("amr", "connectivity", "invalid_connectivity");

  if (conn_name == "unit" or conn_name == "brick")
  {
    // gather orchard for each local quadrant / cell

    // brick sizes
    brick_size_t<dim> brick_sizes;
    brick_sizes[0] = 1;
    brick_sizes[1] = 1;
    if constexpr (dim == 3)
      brick_sizes[2] = 1;

    if (conn_name == "brick")
    {
      brick_sizes = get_brick_sizes<dim>(m_config_map);
    }

    // tree linear index to xyz converter
    BrickConnectivityData<dim> convert(brick_sizes);

    // array of local trees
    sc_array_t * trees = m_forest->trees;

    auto max_level = m_config_map.getInteger("amr", "level_max", 0);

    // loop over all local tree
    for (auto jt = m_forest->first_local_tree; jt <= m_forest->last_local_tree; ++jt)
    {
      // get current tree
      tree_t<dim> * tree = tree_array_index(trees, jt);

      // get tree coordinate
      auto tree_xyz = convert.toXYZ(jt);

      // get quadrant array of current tree
      sc_array_t * quadrants = &(tree->quadrants);

      // loop over all local quadrant
      for (size_t jq = 0; jq < quadrants->elem_count; ++jq)
      {
        quadrant_t<dim> * q = quadrant_array_index(quadrants, jq);

        Kokkos::Array<uint32_t, dim> octCoord;
        octCoord[0] =
          static_cast<uint32_t>(q->x >> (p4est_t::QMAXLEVEL - orchard_key_t<dim>::NUM_LEVELS + 2));
        octCoord[1] =
          static_cast<uint32_t>(q->y >> (p4est_t::QMAXLEVEL - orchard_key_t<dim>::NUM_LEVELS + 2));
        if constexpr (dim == 3)
        {
          octCoord[2] = static_cast<uint32_t>(
            q->z >> (p4est_t::QMAXLEVEL - orchard_key_t<dim>::NUM_LEVELS + 2));
        }

        if (type == FULL)
        {
          uint64_t orchard_key =
            orchard_key_t<dim>::encode_orchard(tree_xyz, octCoord, static_cast<uint8_t>(q->level));
          for (size_t i = 0; i < m_nbCellsPerLeaf; ++i)
          {
            data[icell] = orchard_key;
            icell++;
          }
        }
        else
        {
          // reduced key
          octCoord[0] >>= (orchard_key_t<dim>::NUM_LEVELS - max_level);
          octCoord[1] >>= (orchard_key_t<dim>::NUM_LEVELS - max_level);
          if constexpr (dim == 3)
            octCoord[2] >>= (orchard_key_t<dim>::NUM_LEVELS - max_level);

          uint64_t orchard_key = orchard_key_t<dim>::encode_orchard(tree_xyz, octCoord, 0);
          uint64_t morton_oct = orchard_key_t<dim>::morton_octant(orchard_key);
          uint64_t morton_key = orchard_key_t<dim>::morton_tree(orchard_key);
          uint64_t reduced_key =
            type == REDUCED_GLOBAL
              ? morton_oct + (morton_key << (dim * static_cast<uint32_t>(max_level)))
              : morton_oct;

          for (size_t i = 0; i < m_nbCellsPerLeaf; ++i)
          {
            data[icell] = reduced_key; // morton_oct; // orchard_key;
            icell++;
          }
        }

      } // end for quadrant/octant in current tree

    } // end for tree
  }
  else
  {
    // to be clarified, we could defined an orchard key, but the meaning is unclear
    for (uint32_t i = 0; i < nbData; ++i)
    {
      data[i] = 0;
    }
  }

  std::string attribute_str = type == FULL             ? "full_orchard_key"
                              : type == REDUCED_GLOBAL ? "reduced_orchard_key_global"
                                                       : "reduced_orchard_key_local";

  return write_attribute(attribute_str, data.data(), 0, IO_CELL_SCALAR);


} // HDF5_Xdmf_Writer_legacy<dim, device_t>::io_hdf5_write_reduced_orchard_key

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer_legacy<dim, device_t>::io_hdf5_write_at_domain_border()
{

  if (!this->m_write_at_domain_border)
  {
    return 0;
  }

  // const auto NB_CHILDREN = p4est_t::NB_CHILDREN;
  const auto quadrant_array_index = &p4est_t::quadrant_array_index;
  const auto tree_array_index = &p4est_t::tree_array_index;

  uint32_t nbData = m_local_num_quads * m_nbCellsPerLeaf;

  std::vector<uint32_t> data(nbData);
  size_t                icell = 0;

  // check if connectivity is either "unit" or "brick"
  auto conn_name = m_config_map.getString("amr", "connectivity", "invalid_connectivity");

  if (conn_name == "unit" or conn_name == "brick")
  {
    // gather orchard for each local quadrant / cell

    // brick sizes
    brick_size_t<dim> brick_sizes;
    brick_sizes[0] = 1;
    brick_sizes[1] = 1;
    if constexpr (dim == 3)
      brick_sizes[2] = 1;

    if (conn_name == "brick")
    {
      brick_sizes = get_brick_sizes<dim>(m_config_map);
    }

    // tree linear index to xyz converter
    BrickConnectivityData<dim> convert(brick_sizes);

    // array of local trees
    sc_array_t * trees = m_forest->trees;

    // loop over all local tree
    for (auto jt = m_forest->first_local_tree; jt <= m_forest->last_local_tree; ++jt)
    {
      // get current tree
      tree_t<dim> * tree = tree_array_index(trees, jt);

      // get tree coordinate
      auto tree_xyz = convert.toXYZ(jt);

      // get quadrant array of current tree
      sc_array_t * quadrants = &(tree->quadrants);

      // loop over all local quadrant
      for (size_t jq = 0; jq < quadrants->elem_count; ++jq)
      {
        quadrant_t<dim> * q = quadrant_array_index(quadrants, jq);

        Kokkos::Array<uint32_t, dim> octCoord;
        octCoord[0] =
          static_cast<uint32_t>(q->x >> (p4est_t::QMAXLEVEL - orchard_key_t<dim>::NUM_LEVELS + 2));
        octCoord[1] =
          static_cast<uint32_t>(q->y >> (p4est_t::QMAXLEVEL - orchard_key_t<dim>::NUM_LEVELS + 2));
        if constexpr (dim == 3)
        {
          octCoord[2] = static_cast<uint32_t>(
            q->z >> (p4est_t::QMAXLEVEL - orchard_key_t<dim>::NUM_LEVELS + 2));
        }

        uint64_t orchard_key =
          orchard_key_t<dim>::encode_orchard(tree_xyz, octCoord, static_cast<uint8_t>(q->level));

        for (size_t i = 0; i < m_nbCellsPerLeaf; ++i)
        {
          uint32_t atDomainBorder = static_cast<uint32_t>(
            orchard_key_t<dim>::is_at_any_domain_border(orchard_key, brick_sizes));
          data[icell] = atDomainBorder;
          icell++;
        }

      } // end for quadrant/octant in current tree

    } // end for tree
  }
  else
  {
    // not defined, so use "zero" as default value
    // writing this attribute should be disabled in config_map
    for (uint32_t i = 0; i < nbData; ++i)
    {
      data[i] = 0;
    }
  }

  return write_attribute("at_domain_border", data.data(), 0, IO_CELL_SCALAR);

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::io_hdf5_write_at_domain_border

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer_legacy<dim, device_t>::io_hdf5_write_at_tree_border()
{

  if (!this->m_write_at_tree_border)
  {
    return 0;
  }

  // const auto NB_CHILDREN = p4est_t::NB_CHILDREN;
  const auto quadrant_array_index = &p4est_t::quadrant_array_index;
  const auto tree_array_index = &p4est_t::tree_array_index;

  uint32_t nbData = m_local_num_quads * m_nbCellsPerLeaf;

  std::vector<uint32_t> data(nbData);
  size_t                icell = 0;

  // check if connectivity is either "unit" or "brick"
  auto conn_name = m_config_map.getString("amr", "connectivity", "invalid_connectivity");

  if (conn_name == "unit" or conn_name == "brick")
  {
    // gather orchard for each local quadrant / cell

    // brick sizes
    brick_size_t<dim> brick_sizes;
    brick_sizes[0] = 1;
    brick_sizes[1] = 1;
    if constexpr (dim == 3)
      brick_sizes[2] = 1;

    if (conn_name == "brick")
    {
      brick_sizes = get_brick_sizes<dim>(m_config_map);
    }

    // tree linear index to xyz converter
    BrickConnectivityData<dim> convert(brick_sizes);

    // array of local trees
    sc_array_t * trees = m_forest->trees;

    // loop over all local tree
    for (auto jt = m_forest->first_local_tree; jt <= m_forest->last_local_tree; ++jt)
    {
      // get current tree
      tree_t<dim> * tree = tree_array_index(trees, jt);

      // get tree coordinate
      auto tree_xyz = convert.toXYZ(jt);

      // get quadrant array of current tree
      sc_array_t * quadrants = &(tree->quadrants);

      // loop over all local quadrant
      for (size_t jq = 0; jq < quadrants->elem_count; ++jq)
      {
        quadrant_t<dim> * q = quadrant_array_index(quadrants, jq);

        Kokkos::Array<uint32_t, dim> octCoord;
        octCoord[0] =
          static_cast<uint32_t>(q->x >> (p4est_t::QMAXLEVEL - orchard_key_t<dim>::NUM_LEVELS + 2));
        octCoord[1] =
          static_cast<uint32_t>(q->y >> (p4est_t::QMAXLEVEL - orchard_key_t<dim>::NUM_LEVELS + 2));
        if constexpr (dim == 3)
        {
          octCoord[2] = static_cast<uint32_t>(
            q->z >> (p4est_t::QMAXLEVEL - orchard_key_t<dim>::NUM_LEVELS + 2));
        }

        uint64_t orchard_key =
          orchard_key_t<dim>::encode_orchard(tree_xyz, octCoord, static_cast<uint8_t>(q->level));

        uint32_t at_any_tree_border =
          static_cast<uint32_t>(orchard_key_t<dim>::is_at_any_tree_border(orchard_key));

        uint32_t at_any_tree_corner =
          static_cast<uint32_t>(orchard_key_t<dim>::is_at_any_tree_corner(orchard_key));

        uint32_t at_any_tree_edge = 0;
        if constexpr (dim == 3)
          at_any_tree_edge =
            static_cast<uint32_t>(orchard_key_t<dim>::is_at_any_tree_edge(orchard_key));

        for (size_t i = 0; i < m_nbCellsPerLeaf; ++i)
        {
          data[icell] = at_any_tree_border + at_any_tree_corner + at_any_tree_edge;
          icell++;
        }

      } // end for quadrant/octant in current tree

    } // end for tree
  }
  else
  {
    // not defined, so use "zero" as default value
    // writing this attribute should be disabled in config_map
    for (uint32_t i = 0; i < nbData; ++i)
    {
      data[i] = 0;
    }
  }

  return write_attribute("at_tree_border", data.data(), 0, IO_CELL_SCALAR);

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::io_hdf5_write_at_tree_border

// =======================================================
// =======================================================
// Private members
// =======================================================
// =======================================================

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer_legacy<dim, device_t>::io_xdmf_write_main_header()
{

  FILE * fd = m_main_xdmf_file;

  fprintf(fd, "<?xml version=\"1.0\" ?>\n");
  fprintf(fd, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
  fprintf(fd,
          "<Xdmf xmlns:xi=\"http://www.w3.org/2001/XInclude\""
          " Version=\"2.0\">\n");
  fprintf(fd, "  <Domain Name=\"MainTimeSeries\">\n");
  fprintf(fd,
          "    <Grid Name=\"MainTimeSeries\" GridType=\"Collection\""
          " CollectionType=\"Temporal\">\n");

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::io_xdmf_write_main_header

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer_legacy<dim, device_t>::io_xdmf_write_header(double time)
{

  FILE * fd = this->m_xdmf_file;
  size_t global_num_cells = this->m_global_num_quads;
  size_t global_num_nodes = global_num_cells * m_nbNodesPerCell;

  if (m_write_block_data)
  {

    global_num_cells *= m_nbCellsPerLeaf;

    uint32_t nbNodesPerLeaf = static_cast<uint32_t>(Kokkos::dim_prod(m_block_size + 1));

    global_num_nodes = this->m_global_num_quads * nbNodesPerLeaf;
  }

  const std::string IO_TOPOLOGY_TYPE = dim == TWO_D ? IO_TOPOLOGY_TYPE_2D : IO_TOPOLOGY_TYPE_3D;

  fprintf(fd, "<?xml version=\"1.0\" ?>\n");
  fprintf(fd, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
  fprintf(fd, "<Xdmf Version=\"2.0\">\n");
  fprintf(fd, "  <Domain>\n");
  fprintf(fd, "    <Grid Name=\"%s\" GridType=\"Uniform\">\n", this->m_basename.c_str());
  fprintf(fd, "      <Time TimeType=\"Single\" Value=\"%g\" />\n", time);

  // Connectivity
  fprintf(fd,
          "      <Topology TopologyType=\"%s\" NumberOfElements=\"%lu\""
          ">\n",
          IO_TOPOLOGY_TYPE.c_str(),
          global_num_cells);
  fprintf(fd,
          "        <DataItem Dimensions=\"%lu %d\" DataType=\"Int\""
          " Format=\"HDF\">\n",
          global_num_cells,
          m_nbNodesPerCell);
  fprintf(fd, "         %s.h5:/connectivity\n", this->m_basename.c_str());
  fprintf(fd, "        </DataItem>\n");
  fprintf(fd, "      </Topology>\n");

  // Points
  fprintf(fd, "      <Geometry GeometryType=\"XYZ\">\n");
  fprintf(fd,
          "        <DataItem Dimensions=\"%lu 3\" %s Format=\"HDF\">\n",
          global_num_nodes,
          IO_XDMF_NUMBER_TYPE);
  fprintf(fd, "         %s.h5:/coordinates\n", this->m_basename.c_str());
  fprintf(fd, "        </DataItem>\n");
  fprintf(fd, "      </Geometry>\n");

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::io_xdmf_write_header

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer_legacy<dim, device_t>::io_xdmf_write_main_include(const std::string & name)
{

  FILE * fd = this->m_main_xdmf_file;

  fprintf(fd,
          "      <xi:include href=\"%s\""
          " xpointer=\"xpointer(//Xdmf/Domain/Grid)\" />\n",
          name.c_str());

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::io_xdmf_write_main_include

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer_legacy<dim, device_t>::io_xdmf_write_attribute(const std::string & name,
                                                                const std::string & number_type,
                                                                io_attribute_type_t type,
                                                                std::vector<size_t> const & dims)
{

  FILE *              fd = this->m_xdmf_file;
  const std::string & basename = this->m_basename;

  fprintf(fd, "      <Attribute Name=\"%s\"", name.c_str());
  switch (type)
  {
    case IO_CELL_SCALAR:
      fprintf(fd, " AttributeType=\"Scalar\" Center=\"Cell\">\n");
      break;
    case IO_CELL_VECTOR:
      fprintf(fd, " AttributeType=\"Vector\" Center=\"Cell\">\n");
      break;
    case IO_NODE_SCALAR:
      fprintf(fd, " AttributeType=\"Scalar\" Center=\"Node\">\n");
      break;
    case IO_NODE_VECTOR:
      fprintf(fd, " AttributeType=\"Vector\" Center=\"Node\">\n");
      break;
    default:
      std::cerr << "Unsupported field type.\n";
      return;
  }

  fprintf(fd, "        <DataItem %s Format=\"HDF\"", number_type.c_str());
  if (type == IO_CELL_SCALAR || type == IO_NODE_SCALAR)
  {
    fprintf(fd, " Dimensions=\"%zu\">\n", dims[0]);
  }
  else
  {
    fprintf(fd, " Dimensions=\"%zu %zu\">\n", dims[0], dims[1]);
  }

  fprintf(fd, "         %s.h5:/%s\n", basename.c_str(), name.c_str());
  fprintf(fd, "        </DataItem>\n");
  fprintf(fd, "      </Attribute>\n");

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::io_xdmf_write_attribute

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer_legacy<dim, device_t>::io_xdmf_write_main_footer()
{

  FILE * fd = this->m_main_xdmf_file;

  fprintf(fd, "    </Grid>\n");
  fprintf(fd, "  </Domain>\n");
  fprintf(fd, "</Xdmf>\n");

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::io_xdmf_write_main_footer

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer_legacy<dim, device_t>::io_xdmf_write_footer()
{

  FILE * fd = this->m_xdmf_file;

  fprintf(fd, "    </Grid>\n");
  fprintf(fd, "  </Domain>\n");
  fprintf(fd, "</Xdmf>\n");

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::io_xdmf_write_footer

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
int32_t
HDF5_Xdmf_Writer_legacy<dim, device_t>::subCellIndex(int jx, int jy, int jz)
{

  return jx + m_block_size[IX] * (jy + m_block_size[IY] * jz);

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::subCellIndex

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
int32_t
HDF5_Xdmf_Writer_legacy<dim, device_t>::subNodeIndex(int jx, int jy, int jz)
{

  return jx + (m_block_size[IX] + 1) * (jy + (m_block_size[IY] + 1) * jz);

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::subNodeIndex

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer_legacy<dim, device_t>::set_block_mode(block_size_t<dim> block_size,
                                                       coord_t<dim>      start_index)
{

  m_block_size = block_size;
  m_nbCellsPerLeaf = static_cast<uint32_t>(Kokkos::dim_prod(block_size));
  m_istart = start_index;

  m_write_block_data = (Kokkos::dim_prod(block_size) > 1);

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::set_block_mode

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer_legacy<dim, device_t>::set_leaf_mode()
{

  m_block_size[IX] = 1;
  m_block_size[IY] = 1;
  if constexpr (dim == 3)
  {
    m_block_size[IZ] = 1;
  }
  m_nbCellsPerLeaf = static_cast<uint32_t>(Kokkos::dim_prod(m_block_size));

  m_istart[IX] = 0;
  m_istart[IY] = 0;
  if constexpr (dim == 3)
  {
    m_istart[IZ] = 0;
  }

  m_write_block_data = false;

} // HDF5_Xdmf_Writer_legacy<dim, device_t>::set_leaf_mode

// =====================================================================
// =====================================================================

} // namespace kalypsso

#endif // KALYPSSO_CORE_HDF5_XDMF_WRITER_LEGACY_H_
