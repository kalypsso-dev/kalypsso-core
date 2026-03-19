// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file HDF5_Xdmf_Writer.h
 * \brief same functionality as HDF5_IO but only relying on orchard key hashmap, no p4est here.
 *
 * The main reason for redesigning io is to be ease implementing a class that could dump regular
 * quadrants (inside domain), as well as outside quadrants.
 */
#ifndef KALYPSSO_CORE_HDF5_XDMF_WRITER_H_
#define KALYPSSO_CORE_HDF5_XDMF_WRITER_H_

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
 * \brief HDF5 writer for both quadrant inside / outside domain.
 *
 * Optionally a Xdmf file is written so that hdf5 can be opened directly in paraview using VTU data
 * format (VTK Unstructured Mesh)
 *
 *
 * \tparam dim = 2 or 3
 */
template <size_t dim, typename device_t>
class HDF5_Xdmf_Writer
{

public:
  using DataArrayLeafHost_t = DataArrayLeafHost<real_t, device_t>;
  using DataArrayBlockLegacyHost_t = DataArrayBlockLegacyHost<real_t, device_t>;
  using DataArrayBlockHost_t = DataArrayBlock<dim, real_t, HostDevice>;
  using DataArrayBlockMultiVarHost_t = DataArrayBlockMultiVar<dim, real_t, HostDevice>;
  using MaterialPresenceHost_t = MaterialPresenceView<HostDevice>;
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  using p4est_t = typename p4est::Wrapper<dim>;

  using MeshMap_t = MeshMap<dim, device_t>;

  enum orchard_key_dump_type_t : uint32_t
  {
    REDUCED_LOCAL = 0,
    REDUCED_GLOBAL = 1,
    FULL = 2
  };

  enum writing_mode_t : uint32_t
  {
    BLOCK_MODE = 0,
    LEAF_MODE = 1
  };

  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;
  using orchard_key_view_host_t = typename orchard_key_base_t<device_t>::view_host_t;

  /**
   * \brief Constructor.
   *
   * The writer is has no opened files by default except a main xmf file
   * that contains a collection of all the files that are to be written
   * next as a `Temporal` Collection.
   *
   * \param[in] config_map to access parameters settings
   * \param[in] par_env is the parallel environment (to access MPI_Comm)
   * \param[in] keys is a Kokkos::View of orchard keys
   * \param[in] amr_mesh_info access number of owned, MPI ghosts and outside quadrants
   * \param[in] xdmf_main_suffix is a string used to create filename for main xdmf file
   *
   */
  HDF5_Xdmf_Writer(ParallelEnv const &        par_env,
                   ConfigMap const &          config_map,
                   std::shared_ptr<MeshMap_t> mesh_map,
                   std::string                xdmf_main_suffix = "_main");

  //! destructor
  virtual ~HDF5_Xdmf_Writer();

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

  //! write amr metadata (orchard key array, ...)
  uint64_t
  write_amr_metadata(orchard_key_view_host_t keys);

  /**
   * \brief Write the header for the XMF and HDF5 files.
   *
   * The header includes the node information, connectivity information and
   * the treeid, level or mpi rank for each quadrant, if required.
   *
   * In the case of the XMF file, this will defined the topology and geometry
   * of the mesh and point to the relevant fields in the HDF5 file.
   *
   * \return number of bytes written
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
   * Write scalar attribute.
   *
   * \param[in] group_name hdf5 group name
   * \param[in] var_name scalar variable name
   * \param[in] data variable value
   */
  template <typename data_t>
  uint64_t
  write_scalar_attribute(const std::string & group_name, const std::string & var_name, data_t data);

  /**
   * \brief Write cell-centered data.
   *
   * \param [in] name The name of the attribute.
   * \param [in] data The data to be written.
   * \param [in] dimData In the case of a vector, this is the dimension of each
   *             element in the vector field.
   * \param [in] ftype The type of the attribute. See supported types in
   *             the io_attribute_type_t enum.
   *
   * \return number of bytes written
   */
  template <typename data_t>
  uint64_t
  write_cc_attribute(const std::string & name,
                     data_t *            data,
                     size_t              dimData,
                     io_attribute_type_t ftype);

  /**
   * \brief Write face-centered data.
   *
   * Be aware that face data can't be added to xmdf and read directly in Paraview.
   * Currently this is only useful for checkpoint/restart a FaceDataArrayBlock.
   * At some point, we will see if we could update our paraview plugin to read face data, convert
   * them into cell-centered data on the fly.
   *
   * \param [in] name The name of the attribute.
   * \param [in] data The data to be written.
   * \param [in] dimData In the case of a vector, this is the dimension of each
   *             element in the vector field.
   * \param [in] number of face elements per octants (see
   *             FaceDataArrayBlock::num_elements_per_octant).
   *
   * \return number of bytes written
   */
  template <typename data_t>
  uint64_t
  write_fc_attribute(const std::string & name,
                     data_t *            data,
                     size_t              dimData,
                     size_t              num_elements_per_oct);

  /**
   * \brief Write a cell-centered scalar dataset.
   *
   * \param [in] datah a host Kokkos View with the user data
   * \param [in] varIdx scalar index identify scalar field to write
   * \param [in] varName variable name
   *
   * \return number of bytes written
   */
  uint64_t
  write_quadrant_attribute(DataArrayLeafHost_t datah, int32_t varIdx, const std::string varName);

  /**
   * \brief Write a cell-centered scalar attributes when block amr is enabled.
   *
   * \param [in] datah a host Kokkos View with the user data (block data)
   * \param [in] varIdx scalar index identify scalar field to write
   * \param [in] varName variable name
   * \param [in] iOct_begin index to first octant to write
   * \param [in] nbOcts number of octants to write
   */
  uint64_t
  write_quadrant_attribute(DataArrayBlockHost_t datah,
                           int32_t              varIdx,
                           const std::string    varName,
                           int32_t              iOct_begin,
                           int64_t              nbOcts);

  /**
   * \brief Write face-centered attributes when block amr is enabled.
   *
   * \param [in] face_data a device view of some face data.
   * \param [in] varName variable name
   */
  uint64_t
  write_quadrant_attribute(FaceDataArrayBlock_t face_data, const std::string varName);

  /**
   * \brief Write all cell-centered scalar attributes when block amr is enabled.
   *
   * \param [in] datah a host multivar array
   * \param [in] matph a host material presence object
   * \param [in] num_mats the number of materials
   * \param [in] num_vars_per_mat the number of variables per material
   * \param [in] mat_varIdx scalar index identify scalar field to write related to materials
   * \param [in] varName variable name (prefix)
   */
  uint64_t
  write_quadrant_multi_mat_attribute(DataArrayBlockMultiVarHost_t datah,
                                     MaterialPresenceHost_t       matph,
                                     int32_t                      num_vars_per_mat,
                                     int32_t                      global_var_id,
                                     std::string                  varName,
                                     int32_t                      iOct_begin,
                                     int32_t                      nbOcts);

  /**
   * write a scalar array (one scale per quadrant/octant).
   */
  template <typename host_view_t>
  uint64_t
  write_quadrant_leaf_scalar(host_view_t data_h, const std::string dataName);


  ParallelEnv const & m_par_env;
  ConfigMap const &   m_config_map;

  std::shared_ptr<MeshMap_t> m_mesh_map;
  std::string                m_basename;     //!< the base name of the two files
  writing_mode_t             m_writing_mode; //!< writing mode (leaf or block)

  bool m_write_xdmf;                       //!< write xdmf
  bool m_write_mesh_info;                  //!< write mesh info (oct level, mpi proc, ...)
  bool m_write_tree;                       //!< default write_mesh_info (false)
  bool m_write_level;                      //!< default write_mesh_info (false)
  bool m_write_rank;                       //!< default write_mesh_info (false)
  bool m_write_reduced_orchard_key_global; //!< default write_mesh_info (false)
  bool m_write_reduced_orchard_key_local;  //!< default write_mesh_info (false)
  bool m_write_full_orchard_key;           //!< default write_mesh_info (false)
  bool m_write_at_domain_border;           //!< default write_mesh_info (false)
  bool m_write_at_tree_border;             //!< default write_mesh_info (false)
  bool m_write_is_outside;                 //!< default write_mesh_info (false)

  bool m_write_block_data; //!< if true, expect a DataArrayBlock object

  int m_bx; //!< block size along x
  int m_by; //!< block size along y
  int m_bz; //!< block size along z

  bool m_extract_block_ghost; //!< remove block ghost before writing data
  int  m_gx;                  //!< ghost width along x
  int  m_gy;                  //!< ghost width along y
  int  m_gz;                  //!< ghost width along z

  uint8_t  m_nbNodesPerCell; //!< 4 (2d) or 8 (3d)
  uint32_t m_nbCellsPerLeaf; //!< only useful for block amr

  HighFive::File * m_hdf5_file;      //!< HDF5 file descriptor
  FILE *           m_xdmf_file;      //!< XDMF file descriptor
  FILE *           m_main_xdmf_file; //!< main XDMF file descriptor

  bool m_write_iOct; //!< default false

  // store information about nodes for writing node data

  uint64_t m_global_num_nodes;  //!< global accumulated (all MPI proc) number of nodes
  uint32_t m_local_num_nodes;   //!< local (current MPI proc) number of nodes
  uint64_t m_global_num_quads;  //!< global accumulated (all MPI proc) number of quads/octs
  uint32_t m_local_num_quads;   //!< local (current MPI proc) number of quads/octs
  uint64_t m_global_first_quad; //!< global index of first quadrant in current MPI proc

  bool m_use_outside_quads = false;

  void
  use_outside_quads(bool value);

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
   * \param [in] dataset_name The name of the dataset we are writing.
   * \param [in] data_ptr pointer to data to write.
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
  io_hdf5_writev(HighFive::File &                 hdf5_file,
                 const std::string                dataset_prefix,
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

  /**
   * \brief Write for each quadrant a integer value indicating if quadrant is outside domain.
   * \return number of bytes written
   */
  uint64_t
  io_hdf5_write_is_outside();

  //! get cell linear index from (jx,jy,jz)
  int32_t
  subCellIndex(int jx, int jy, int jz);

  //! get node linear index from (jx,jy,jz)
  int32_t
  subNodeIndex(int jx, int jy, int jz);

public:
  //! switch to block mode for writing DataArrayBlock.
  //! block sizes are set from config_map
  void
  set_block_mode();

  void
  set_block_mode(block_size_t<dim> block_size, block_size_t<dim> ghost_width);

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
    m_write_is_outside = m_config_map.getBool("output", "write_mesh_is_outside", m_write_mesh_info);
  } // set_write_mesh_info

}; // class HDF5_Xdmf_Writer

// =============================================================================
// =============================================================================
// class HDF5_Xdmf_Writer definition
// =============================================================================
// =============================================================================

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
HDF5_Xdmf_Writer<dim, device_t>::HDF5_Xdmf_Writer(ParallelEnv const &        par_env,
                                                  ConfigMap const &          config_map,
                                                  std::shared_ptr<MeshMap_t> mesh_map,
                                                  std::string                xdmf_main_suffix)
  : m_par_env(par_env)
  , m_config_map(config_map)
  , m_mesh_map(mesh_map)
  , m_basename("")
  , m_writing_mode(BLOCK_MODE)
  , m_write_xdmf(config_map.getBool("output", "write_xdmf", true))
  , m_write_mesh_info(config_map.getBool("output", "write_mesh_info", false))
  , m_write_tree(config_map.getBool("output", "write_mesh_tree", m_write_mesh_info))
  , m_write_level(config_map.getBool("output", "write_mesh_level", m_write_mesh_info))
  , m_write_rank(config_map.getBool("output", "write_mesh_rank", m_write_mesh_info))
  , m_write_reduced_orchard_key_global(
      config_map.getBool("output", "write_mesh_reduced_orchard_key_global", m_write_mesh_info))
  , m_write_reduced_orchard_key_local(
      config_map.getBool("output", "write_mesh_reduced_orchard_key_local", m_write_mesh_info))
  , m_write_full_orchard_key(
      config_map.getBool("output", "write_mesh_full_orchard_key", m_write_mesh_info))
  , m_write_at_domain_border(
      config_map.getBool("output", "write_mesh_at_domain_border", m_write_mesh_info))
  , m_write_at_tree_border(
      config_map.getBool("output", "write_mesh_at_tree_border", m_write_mesh_info))
  , m_write_is_outside(config_map.getBool("output", "write_mesh_is_outside", m_write_mesh_info))
  , m_write_block_data(true)
  , m_bx(1)
  , m_by(1)
  , m_bz(1)
  , m_extract_block_ghost(false)
  , m_gx(0)
  , m_gy(0)
  , m_gz(0)
  , m_nbNodesPerCell(dim == TWO_D ? IO_NODES_PER_CELL_2D : IO_NODES_PER_CELL_3D)
  , m_nbCellsPerLeaf(dim == 2 ? static_cast<uint32_t>(m_bx * m_by)
                              : static_cast<uint32_t>(m_bx * m_by * m_bz))
  , m_hdf5_file()
  , m_xdmf_file(NULL)
  , m_main_xdmf_file(NULL)
  , m_write_iOct(config_map.getBool("output", "write_iOct", false))
  , m_global_num_nodes(0)
  , m_local_num_nodes(0)
  , m_global_num_quads(0)
  , m_local_num_quads(0)
  , m_global_first_quad(0)
  , m_use_outside_quads(false)
{
  // only meaningful when one wants to write block data (i.e. cell-wise)
  // if true, write data cell-wise (bx*by*bz data per leaf quadrant)
  // if false, write data quadrant-wise (i.e. only one data per quadrant)
  if (m_writing_mode == BLOCK_MODE)
    set_block_mode();
  else
    set_leaf_mode();

  update_mesh_info();

  // is actually hdf5 enabled ?
  const bool hdf5_enabled = m_config_map.getBool("output", "hdf5_enabled", false);

  // when nOutput==0, it means no output at all
  const bool nOutput = config_map.getInteger("run", "noutput", 100);

  if (m_par_env.rank() == 0 and hdf5_enabled and nOutput != 0 and m_write_xdmf)
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

} // HDF5_Xdmf_Writer<dim, device_t>::HDF5_Xdmf_Writer

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
HDF5_Xdmf_Writer<dim, device_t>::~HDF5_Xdmf_Writer()
{

  /*
   * Only rank 0 needs to close the main xdmf file.
   */
  if (m_par_env.rank() == 0 and m_write_xdmf and m_main_xdmf_file != nullptr)
  {
    io_xdmf_write_main_footer();

    fflush(m_main_xdmf_file);
    fclose(m_main_xdmf_file);
    m_main_xdmf_file = nullptr;
  }

  // close the other files (hdf5 + xdmf)
  close();

} // HDF5_Xdmf_Writer<dim, device_t>::~HDF5_Xdmf_Writer

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer<dim, device_t>::update_mesh_info()
{

  m_local_num_quads = static_cast<uint32_t>(m_mesh_map->get_amr_mesh_info().local_num_quadrants());
  m_global_num_quads =
    static_cast<uint64_t>(m_mesh_map->get_amr_mesh_info().global_num_quadrants());
  m_global_first_quad =
    static_cast<uint64_t>(m_mesh_map->get_amr_mesh_info().global_first_quadrant());

  m_local_num_nodes = m_nbNodesPerCell * m_local_num_quads;
  m_global_num_nodes = m_nbNodesPerCell * m_global_num_quads;

} // HDF5_Xdmf_Writer<dim, device_t>::update_mesh_info

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer<dim, device_t>::use_outside_quads(bool value)
{
  m_use_outside_quads = value;

  if (value)
  {
    // use outside quads
    m_local_num_quads =
      static_cast<uint32_t>(m_mesh_map->get_amr_mesh_info().local_num_quadrants_outside());
    m_global_num_quads =
      static_cast<uint64_t>(m_mesh_map->get_amr_mesh_info().global_num_quadrants_outside());
    m_global_first_quad =
      static_cast<uint64_t>(m_mesh_map->get_amr_mesh_info().global_first_quadrant_outside());

    m_local_num_nodes = m_nbNodesPerCell * m_local_num_quads;
    m_global_num_nodes = m_nbNodesPerCell * m_global_num_quads;
  }
  else
  {
    // use inside quads
    m_local_num_quads =
      static_cast<uint32_t>(m_mesh_map->get_amr_mesh_info().local_num_quadrants());
    m_global_num_quads =
      static_cast<uint64_t>(m_mesh_map->get_amr_mesh_info().global_num_quadrants());
    m_global_first_quad =
      static_cast<uint64_t>(m_mesh_map->get_amr_mesh_info().global_first_quadrant());

    m_local_num_nodes = m_nbNodesPerCell * m_local_num_quads;
    m_global_num_nodes = m_nbNodesPerCell * m_global_num_quads;
  }

} // HDF5_Xdmf_Writer<dim, device_t>::use_outside_quads

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer<dim, device_t>::open(std::string basename, std::string outDir)
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
  fapl.add(HighFive::MPIOFileAccess{ m_par_env.mpi_comm(), MPI_INFO_NULL });
  // all metadata are written using collective IO
  fapl.add(HighFive::MPIOCollectiveMetadata{});
#endif // KALYPSSO_CORE_USE_MPI

  /*
   * Open parallel HDF5 resources.
   */
  m_hdf5_file = new HighFive::File(full_path, HighFive::File::Truncate, fapl);

  // add creation date (collectively), creation is initialized in root MPI process (rank 0)
  {
    auto date = get_current_date();

#ifdef KALYPSSO_CORE_USE_MPI
    auto date_size = date.size();
    m_par_env.comm().MPI_Bcast(&date_size, 1, 0);
    if (m_par_env.rank() != 0)
      date.resize(date_size);

    m_par_env.comm().MPI_Bcast(const_cast<char *>(date.data()), static_cast<int>(date_size), 0);
#endif // KALYPSSO_CORE_USE_MPI

    m_hdf5_file->createAttribute<std::string>("creation_date", date);
  }

  // add a group "amr" for storing amr related metadata
  // everything that is needed to reconstruct geometrical mesh information
  {
    auto group = m_hdf5_file->createGroup("amr");
    group.createAttribute("dim", dim);

    std::array<int, dim> block_sizes;
    block_sizes[IX] = m_bx;
    block_sizes[IY] = m_by;
    if constexpr (dim == 3)
      block_sizes[IZ] = m_bz;
    group.createAttribute("block_sizes", block_sizes);

    const auto brick_sizes =
      to_std_array<typename brick_size_t<dim>::value_type, dim>(get_brick_sizes<dim>(m_config_map));
    group.createAttribute("brick_sizes", brick_sizes);

    const auto xyz_min = to_std_array<real_t, dim>(get_xyz_min<dim>(m_config_map));
    group.createAttribute("xyz_min", xyz_min);

    const auto scaling_factor = get_scaling_factor(m_config_map);
    group.createAttribute("scaling_factor", scaling_factor);

    group.createAttribute("level_min", m_config_map.getInteger("amr", "level_min", 0));
    group.createAttribute("level_max", m_config_map.getInteger("amr", "level_max", 0));

    // write level histogram
    {
      const auto level_min = m_config_map.getInteger("amr", "level_min", 0);
      const auto level_max = m_config_map.getInteger("amr", "level_max", 0);

      auto monitor = AMRMeshMonitoring<dim, device_t>(m_config_map);

      // all MPI procesus computes its local histogram
      const auto level_histo = monitor.compute_level_histogram(m_par_env, *m_mesh_map);
      const auto level_histo_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, level_histo);

      // convert histogram into a std::vector for ease using createAttribute right after
      std::size_t      nb_levels = static_cast<size_t>(level_max - level_min + 1);
      std::vector<int> level_histo_vector(nb_levels);
      for (size_t l = 0; l < nb_levels; ++l)
      {
        level_histo_vector[l] = level_histo_host[l];
      }

      group.createAttribute("level_histogram", level_histo_vector);
    }
  }

  // open xdmf files (one for each hdf5, a main xdmf file)
  if (m_par_env.rank() == 0 and m_write_xdmf)
  {

    filename = basename + ".xmf";
    full_path = outDir + "/" + filename;
    m_xdmf_file = fopen(full_path.c_str(), "w");

    if (m_main_xdmf_file)
    {
      io_xdmf_write_main_include(filename);
    }
  }

} // HDF5_Xdmf_Writer<dim, device_t>::open

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer<dim, device_t>::close()
{
  // flush HDF5
  if (m_hdf5_file != nullptr)
  {
    m_hdf5_file->flush();
    delete m_hdf5_file;
    m_hdf5_file = nullptr;
  }

  // close XDMF file descriptor
  if (m_write_xdmf and m_xdmf_file)
  {
    fflush(m_xdmf_file);
    fclose(m_xdmf_file);
    m_xdmf_file = nullptr;
  }

} // HDF5_Xdmf_Writer<dim, device_t>::close

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer<dim, device_t>::write_amr_metadata(orchard_key_view_host_t keys)
{
  std::vector<std::size_t> dims{ static_cast<size_t>(
    m_mesh_map->get_amr_mesh_info().global_num_quadrants()) };
  std::vector<std::size_t> count{ static_cast<size_t>(
    m_mesh_map->get_amr_mesh_info().local_num_quadrants()) };
  std::vector<std::size_t> start{ static_cast<size_t>(
    m_mesh_map->get_amr_mesh_info().global_first_quadrant()) };

  using data_t = typename orchard_key_view_host_t::value_type;

  // write amr keys
  {
    // create dataset
    HighFive::DataSet dataset =
      m_hdf5_file->createDataSet<data_t>("/amr/keys", HighFive::DataSpace(dims));

    auto xfer_props = HighFive::DataTransferProps{};
#ifdef KALYPSSO_CORE_USE_MPI
    xfer_props.add(HighFive::UseCollectiveIO{});
#endif // KALYPSSO_CORE_USE_MPI

    dataset.select(start, count).write_raw(keys.data(), xfer_props);
#ifdef KALYPSSO_CORE_USE_MPI
    check_collective_io(xfer_props);
#endif // KALYPSSO_CORE_USE_MPI
  }

  // write level index
  {
    const auto level_min = static_cast<uint8_t>(m_config_map.getInteger("amr", "level_min", 4));
    const auto level_max = static_cast<uint8_t>(m_config_map.getInteger("amr", "level_max", 4));
    auto       level_indexes = compute_index_by_level<Kokkos::OpenMP, dim>(
      keys, level_min, level_max, m_mesh_map->get_amr_mesh_info().local_num_quadrants(), m_par_env);

    // create dataset
    HighFive::DataSet dataset =
      m_hdf5_file->createDataSet<data_t>("/amr/level_indexes", HighFive::DataSpace(dims));

    auto xfer_props = HighFive::DataTransferProps{};
#ifdef KALYPSSO_CORE_USE_MPI
    xfer_props.add(HighFive::UseCollectiveIO{});
#endif // KALYPSSO_CORE_USE_MPI

    dataset.select(start, count).write_raw(level_indexes.data(), xfer_props);
#ifdef KALYPSSO_CORE_USE_MPI
    check_collective_io(xfer_props);
#endif // KALYPSSO_CORE_USE_MPI
  }

  // Let's ensure that everything has been written do disk.
  m_hdf5_file->flush();

  // return the total number of bytes written in local MPI process
  return sizeof(data_t) * count[0];

} // HDF5_Xdmf_Writer<dim, device_t>::write_amr_metadata

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer<dim, device_t>::write_header(double time)
{

  // write the xmdf file first
  if (m_par_env.rank() == 0 and m_write_xdmf)
  {
    io_xdmf_write_header(time);
  }

  uint64_t num_bytes = 0;

  // and write stuff into the hdf5 file

  // when write_xdmf is false, no xdmf file is produce, no coordinates/connectivity written in hdf5
  // file
  if (m_write_xdmf)
  {
    num_bytes += io_hdf5_write_coordinates();
    num_bytes += io_hdf5_write_connectivity();
  }
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
  num_bytes += io_hdf5_write_is_outside();

  return num_bytes;

} // HDF5_Xdmf_Writer<dim, device_t>::write_header

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
int
HDF5_Xdmf_Writer<dim, device_t>::write_footer()
{
  if (m_par_env.rank() == 0 and m_write_xdmf)
  {
    io_xdmf_write_footer();
  }

  return 0;

} // HDF5_Xdmf_Writer<dim, device_t>::write_footer

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
template <typename data_t>
uint64_t
HDF5_Xdmf_Writer<dim, device_t>::write_scalar_attribute(const std::string & group_name,
                                                        const std::string & var_name,
                                                        data_t              data)
{
  auto group = m_hdf5_file->exist(group_name) ? m_hdf5_file->getGroup(group_name)
                                              : m_hdf5_file->createGroup(group_name);

  group.createAttribute(var_name, data);

  return sizeof(data_t);
} // HDF5_Xdmf_Writer<dim, device_t>::write_scalar_attribute

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
template <typename data_t>
uint64_t
HDF5_Xdmf_Writer<dim, device_t>::write_cc_attribute(const std::string & name,
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

    if (m_use_outside_quads)
    {
      dims[0] =
        static_cast<size_t>(m_mesh_map->get_amr_mesh_info().global_num_quadrants_outside()) *
        m_nbCellsPerLeaf;
      if (dimData > 0)
        dims[1] = dimData;

      count[0] =
        static_cast<size_t>(m_mesh_map->get_amr_mesh_info().local_num_quadrants_outside()) *
        m_nbCellsPerLeaf;
      if (dimData > 0)
        count[1] = dims[1];

      // get global index of the first octant of current mpi processor
      start[0] =
        static_cast<size_t>(m_mesh_map->get_amr_mesh_info().global_first_quadrant_outside()) *
        m_nbCellsPerLeaf;
      if (dimData > 0)
        start[1] = 0;
    }
    else
    {
      dims[0] = static_cast<size_t>(m_mesh_map->get_amr_mesh_info().global_num_quadrants()) *
                m_nbCellsPerLeaf;
      if (dimData > 0)
        dims[1] = dimData;

      count[0] = static_cast<size_t>(m_mesh_map->get_amr_mesh_info().local_num_quadrants()) *
                 m_nbCellsPerLeaf;
      if (dimData > 0)
        count[1] = dims[1];

      // get global index of the first octant of current mpi processor
      start[0] = static_cast<size_t>(m_mesh_map->get_amr_mesh_info().global_first_quadrant()) *
                 m_nbCellsPerLeaf;
      if (dimData > 0)
        start[1] = 0;
    }
  }

  if (m_par_env.rank() == 0 and m_write_xdmf)
  {
    const char * dtype_str = hdf5_native_type_to_string<data_t>();
    io_xdmf_write_attribute(name, dtype_str, ftype, dims);
  }

  num_bytes += io_hdf5_writev(*m_hdf5_file, "/celldata/", name, data, dims, count, start);

  return num_bytes;

} // HDF5_Xdmf_Writer<dim, device_t>::write_cc_attribute

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
template <typename data_t>
uint64_t
HDF5_Xdmf_Writer<dim, device_t>::write_fc_attribute(const std::string & name,
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

  num_bytes += io_hdf5_writev(*m_hdf5_file, "/facedata/", name, data, dims, count, start);

  return num_bytes;

} // HDF5_Xdmf_Writer<dim, device_t>::write_fc_attribute

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer<dim, device_t>::write_quadrant_attribute(DataArrayLeafHost_t datah,
                                                          int32_t             varIdx,
                                                          const std::string   varName)
{

  assertm(m_write_block_data == false, "[HDF5_Xdmf_Writer] m_write_block_data must be false here.");

  uint64_t num_bytes = 0;

  // if DataArray has a left layout, we only need to define
  // a slice to actual scalar data
  // if DataArrayLeafHost_t has right layout, we need to actually extract
  // the slide so that it is memory contiguous
  if (std::is_same<typename DataArrayLeafHost_t::array_layout, Kokkos::LayoutLeft>::value)
  {

    uint32_t begin = m_use_outside_quads ? m_mesh_map->get_amr_mesh_info().local_num_quadrants() +
                                             m_mesh_map->get_amr_mesh_info().local_num_ghosts()
                                         : 0;
    uint32_t nbOcts = m_use_outside_quads
                        ? m_mesh_map->get_amr_mesh_info().local_num_quadrants_outside()
                        : m_mesh_map->get_amr_mesh_info().local_num_quadrants();

    auto dataVar = Kokkos::subview(datah, std::make_pair(begin, begin + nbOcts), varIdx);

    // actual data writing
    num_bytes += write_cc_attribute(varName, dataVar.data(), 0, IO_CELL_SCALAR);
  }
  else
  {
    using DataArrayScalar = Kokkos::View<real_t *, Kokkos::HostSpace>;

    int32_t begin = m_use_outside_quads ? m_mesh_map->get_amr_mesh_info().local_num_quadrants() +
                                            m_mesh_map->get_amr_mesh_info().local_num_ghosts()
                                        : 0;
    int32_t nbOcts = m_use_outside_quads
                       ? m_mesh_map->get_amr_mesh_info().local_num_quadrants_outside()
                       : m_mesh_map->get_amr_mesh_info().local_num_quadrants();

    DataArrayScalar dataVar("scalar_array_for_hdf5_io", static_cast<size_t>(nbOcts));

    Kokkos::parallel_for(
      "HDF5_Xdmf_Writer::write_quadrant_attribute",
      Kokkos::RangePolicy<Kokkos::OpenMP>(begin, begin + nbOcts),
      KOKKOS_LAMBDA(int32_t iOct) { dataVar(iOct - begin) = datah(iOct, varIdx); });

    // make sure all kokkos kernels are done before actual writing
    Kokkos::fence();

    // actual data writing
    num_bytes += write_cc_attribute(varName, dataVar.data(), 0, IO_CELL_SCALAR);
  }

  return num_bytes;

} // HDF5_Xdmf_Writer<dim, device_t>::write_quadrant_attribute

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer<dim, device_t>::write_quadrant_attribute(DataArrayBlockHost_t datah,
                                                          int32_t              varIdx,
                                                          const std::string    varName,
                                                          int32_t              iOct_begin,
                                                          int64_t              nbOcts)
{
  uint64_t num_bytes = 0;
  int32_t  nbCellsPerOct = dim == 2 ? m_bx * m_by : m_bx * m_by * m_bz;

  int32_t                  size_x = m_bx + 2 * m_gx;
  int32_t                  size_y = m_by + 2 * m_gy;
  [[maybe_unused]] int32_t size_z = m_bz + 2 * m_gz;

  // we don't want to capture this in kokkos lambda's
  auto                  bx = m_bx;
  auto                  by = m_by;
  [[maybe_unused]] auto bz = m_bz;

  auto                  gx = m_gx;
  auto                  gy = m_gy;
  [[maybe_unused]] auto gz = m_gz;

  if (m_extract_block_ghost)
  {
    [[maybe_unused]] int32_t nbCellsPerOct_ghosted =
      dim == 2 ? size_x * size_y : size_x * size_y * size_z;

    assertm(datah.num_cells() == nbCellsPerOct_ghosted,
            "HDF5_Xdmf_Writer::write_quadrant_attribute wrong data size");
  }
  else
  {
    assertm(datah.num_cells() == nbCellsPerOct,
            "HDF5_Xdmf_Writer::write_quadrant_attribute wrong data size");
  }

  // we need to gather data corresponding to a given scalar variable
  using DataArrayScalar = Kokkos::View<real_t *, Kokkos::HostSpace>;

  // remember that
  // - data.extent(0) is the number of cells per octant
  // - data.extent(1) is the number of scalar fields
  // - data.extent(2) is the total number of oct in current MPI process
  DataArrayScalar dataVar =
    DataArrayScalar("scalar_array_for_hdf5_io", static_cast<size_t>(nbCellsPerOct * nbOcts));

  if (m_extract_block_ghost == false)
  {
    Kokkos::parallel_for(
      "HDF5_Xdmf_Writer::write_quadrant_attribute",
      Kokkos::RangePolicy<Kokkos::OpenMP>(iOct_begin, iOct_begin + nbOcts),
      KOKKOS_LAMBDA(int32_t iOct) {
        for (int32_t iCell = 0; iCell < nbCellsPerOct; ++iCell)
          dataVar(iCell + nbCellsPerOct * (iOct - iOct_begin)) = datah(iCell, varIdx, iOct);
      });
  }
  else
  {
    Kokkos::parallel_for(
      "HDF5_Xdmf_Writer::write_quadrant_attribute",
      Kokkos::RangePolicy<Kokkos::OpenMP>(iOct_begin, iOct_begin + nbOcts),
      KOKKOS_LAMBDA(int32_t iOct) {

    // the only reason of the following dummy code to be here, is that cuda nvcc compile doesn't
    // support capturing variables inside the inner lambda inside a constexpr if
    // TODO: remove ASAP nvcc is fixed
#ifdef __NVCC__
        [[maybe_unused]] int dummy = 0;
        if (gx == 0 or gy == 0 or gz == 0 or bx == 0 or by == 0 or bz == 0 or size_x == 0 or
            size_y == 0 or size_z == 0 or datah.num_cells() == 0 or dataVar.extent(0) == 0 or
            varIdx == -1 or iOct_begin == 0)
          dummy++;
#endif

        for (int32_t iCell = 0; iCell < nbCellsPerOct; ++iCell)
        {
          if constexpr (dim == 2)
          {
            // iCell = ix + iy*m_bx
            const auto iy = iCell / bx;
            const auto ix = iCell - iy * bx;

            const auto iCell2 = (ix + gx) + size_x * (iy + gy);
            dataVar(iCell + nbCellsPerOct * (iOct - iOct_begin)) = datah(iCell2, varIdx, iOct);
          }
          else if constexpr (dim == 3)
          {
            // iCell = ix + iy*m_bx + iz*m_bx*m_by
            const auto iz = iCell / bx / by;
            const auto tmp = iCell - iz * bx * by;
            const auto iy = tmp / bx;
            const auto ix = tmp - iy * bx;

            auto iCell2 = (ix + gx) + size_x * (iy + gy) + size_x * size_y * (iz + gz);
            dataVar(iCell + nbCellsPerOct * (iOct - iOct_begin)) = datah(iCell2, varIdx, iOct);
          }
        }
      });
  }

  // make sure all kokkos kernels are done before actual writing
  Kokkos::fence();

  // actual data writing
  num_bytes += write_cc_attribute(varName, dataVar.data(), 0, IO_CELL_SCALAR);

  return num_bytes;

} // HDF5_Xdmf_Writer<dim, device_t>::write_quadrant_attribute

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer<dim, device_t>::write_quadrant_attribute(FaceDataArrayBlock_t face_data,
                                                          const std::string    varName)
{
  uint64_t num_bytes = 0;
  size_t   num_elements_per_octant = static_cast<size_t>(face_data.num_elements_per_octant());

  auto face_data_storage = face_data.logical_view();
  auto face_data_storage_host =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, face_data_storage);

  // make sure all kokkos copy are done before actual writing
  Kokkos::fence();

  // actual data writing
  num_bytes +=
    write_fc_attribute(varName, face_data_storage_host.data(), 0, num_elements_per_octant);

  return num_bytes;

} // HDF5_Xdmf_Writer<dim, device_t>::write_quadrant_attribute - FaceDataArrayBlock_t

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer<dim, device_t>::write_quadrant_multi_mat_attribute(
  DataArrayBlockMultiVarHost_t datah,
  MaterialPresenceHost_t       matph,
  int32_t                      num_vars_per_mat,
  int32_t                      global_var_id,
  std::string                  varName,
  int32_t                      iOct_begin,
  int32_t                      nbOcts)
{
  uint64_t num_bytes = 0;
  int32_t  nbCellsPerOct = dim == 2 ? m_bx * m_by : m_bx * m_by * m_bz;

  int32_t                  size_x = m_bx + 2 * m_gx;
  int32_t                  size_y = m_by + 2 * m_gy;
  [[maybe_unused]] int32_t size_z = m_bz + 2 * m_gz;

  // we don't want to capture this in kokkos lambda's
  auto                  bx = m_bx;
  auto                  by = m_by;
  [[maybe_unused]] auto bz = m_bz;

  auto                  gx = m_gx;
  auto                  gy = m_gy;
  [[maybe_unused]] auto gz = m_gz;

  const auto ivar = global_var_id % num_vars_per_mat;
  const auto imat = (global_var_id - ivar) / num_vars_per_mat;

  if (m_extract_block_ghost)
  {
    [[maybe_unused]] int32_t nbCellsPerOct_ghosted =
      dim == 2 ? size_x * size_y : size_x * size_y * size_z;

    assertm(datah.num_cells() == nbCellsPerOct_ghosted,
            "HDF5_Xdmf_Writer::write_quadrant_multi_mat_attribute wrong data size");
  }
  else
  {
    assertm(datah.num_cells() == nbCellsPerOct,
            "HDF5_Xdmf_Writer::write_quadrant_multi_mat_attribute wrong data size");
  }

  // we need to gather data corresponding to a given scalar variable
  using DataArrayScalar = Kokkos::View<real_t *, Kokkos::HostSpace>;

  // array used for each material
  DataArrayScalar dataVar("scalar_array_for_hdf5_io", static_cast<size_t>(nbCellsPerOct * nbOcts));

  if (m_extract_block_ghost == false)
  {
    Kokkos::parallel_for(
      "HDF5_Xdmf_Writer::write_quadrant_multi_mat_attribute",
      Kokkos::RangePolicy<Kokkos::OpenMP>(iOct_begin, iOct_begin + nbOcts),
      KOKKOS_LAMBDA(int32_t iOct) {
        if (matph.get(iOct, imat))
        {
          const auto mat_id = matph.material_index(iOct, imat);
          const auto var_id = mat_id * num_vars_per_mat + ivar;
          for (int32_t iCell = 0; iCell < nbCellsPerOct; ++iCell)
            dataVar(iCell + nbCellsPerOct * (iOct - iOct_begin)) = datah(iCell, var_id, iOct);
        }
        else // We set the value to nan to indicates that the value "does not exist"
          for (int32_t iCell = 0; iCell < nbCellsPerOct; ++iCell)
            dataVar(iCell + nbCellsPerOct * (iOct - iOct_begin)) = nan("");
      });
  }
  else
  {
    Kokkos::parallel_for(
      "HDF5_Xdmf_Writer::write_quadrant_multi_mat_attribute",
      Kokkos::RangePolicy<Kokkos::OpenMP>(iOct_begin, iOct_begin + nbOcts),
      KOKKOS_LAMBDA(int32_t iOct) {

    // the only reason of the following dummy code to be here, is that cuda nvcc compile doesn't
    // support capturing variables inside the inner lambda inside a constexpr if
    // TODO: remove ASAP nvcc is fixed
#ifdef __NVCC__
        [[maybe_unused]] int dummy = 0;
        if (gx == 0 or gy == 0 or gz == 0 or bx == 0 or by == 0 or bz == 0 or size_x == 0 or
            size_y == 0 or size_z == 0 or datah.num_cells() == 0 or dataVar.extent(0) == 0 or
            ivar == -1 or iOct_begin == 0)
          dummy++;
#endif

        if (matph.get(iOct, imat))
        {
          const auto mat_id = matph.material_index(iOct, imat);
          const auto var_id = mat_id * num_vars_per_mat + ivar;

          for (int32_t iCell = 0; iCell < nbCellsPerOct; ++iCell)
          {
            if constexpr (dim == 2)
            {
              // iCell = ix + iy*m_bx
              const auto iy = iCell / bx;
              const auto ix = iCell - iy * bx;

              const auto iCell2 = (ix + gx) + size_x * (iy + gy);
              dataVar(iCell + nbCellsPerOct * (iOct - iOct_begin)) = datah(iCell2, var_id, iOct);
            }
            else if constexpr (dim == 3)
            {
              // iCell = ix + iy*m_bx + iz*m_bx*m_by
              const auto iz = iCell / bx / by;
              const auto tmp = iCell - iz * bx * by;
              const auto iy = tmp / bx;
              const auto ix = tmp - iy * bx;

              auto iCell2 = (ix + gx) + size_x * (iy + gy) + size_x * size_y * (iz + gz);
              dataVar(iCell + nbCellsPerOct * (iOct - iOct_begin)) = datah(iCell2, var_id, iOct);
            }
          }
        }
        else // We set the value to nan to indicates that the value "does not exist"
          for (int32_t iCell = 0; iCell < nbCellsPerOct; ++iCell)
            dataVar(iCell + nbCellsPerOct * (iOct - iOct_begin)) = nan("");
      });
  }

  // make sure all kokkos kernels are done before actual writing
  Kokkos::fence();

  // actual data writing
  num_bytes += write_cc_attribute(varName, dataVar.data(), 0, IO_CELL_SCALAR);

  return num_bytes;
}

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
template <typename host_view_t>
uint64_t
HDF5_Xdmf_Writer<dim, device_t>::write_quadrant_leaf_scalar(host_view_t       data_h,
                                                            const std::string dataName)
{
  static_assert(Kokkos::is_view<host_view_t>::value);
  static_assert(host_view_t::rank == 1);
  static_assert(KokkosExt::is_accessible_from_host<host_view_t>::value);

  assertm(m_writing_mode == LEAF_MODE,
          "[HDF5_Xdmf_Writer::write_quadrant_leaf_scalar] writing mode must be LEAF_MODE here.");

  [[maybe_unused]] const auto nbOcts = m_mesh_map->get_amr_mesh_info().local_num_quadrants();

  assertm(data_h.extent(0) >= static_cast<size_t>(nbOcts),
          "[HDF5_Xdmf_Writer::write_quadrant_leaf_scalar] input data is too short.");

  return write_cc_attribute(dataName, data_h.data(), 0, IO_CELL_SCALAR);

} // HDF5_Xdmf_Writer<dim, device_t>::write_quadrant_leaf_scalar

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
template <typename data_t>
uint64_t
HDF5_Xdmf_Writer<dim, device_t>::io_hdf5_writev(HighFive::File &                 hdf5_file,
                                                std::string                      dataset_prefix,
                                                const std::string &              dataset_name,
                                                data_t *                         data_ptr,
                                                std::vector<std::size_t> const & dims,
                                                std::vector<std::size_t> const & count,
                                                std::vector<std::size_t> const & offset)
{
  // create dataset
  HighFive::DataSet dataset =
    hdf5_file.createDataSet<data_t>(dataset_prefix + dataset_name, HighFive::DataSpace(dims));

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

} // HDF5_Xdmf_Writer<dim, device_t>::io_hdf5_writev

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer<dim, device_t>::io_hdf5_write_coordinates()
{

  // get up to date orchard keys
  const auto orchard_keys = m_mesh_map->orchard_keys_host();

  const auto brick_sizes = get_brick_sizes<dim>(m_config_map);

  // get periodicity property from config map
  const auto is_brick_periodic = get_brick_periodicity<dim>(m_config_map);

  // get geometrical scaling factor
  const auto scaling_factor = get_scaling_factor(m_config_map);

  // get domain lower left corner
  const auto xyz_min = get_xyz_min<dim>(m_config_map);

  // do we have at least one non-periodic border ?
  bool has_non_periodic_border = false;
  for (size_t idim = 0; idim < dim; ++idim)
  {
    if (!is_brick_periodic[idim])
      has_non_periodic_border = true;
  }

  uint64_t num_bytes = 0;

  if (m_write_block_data)
  {
    uint32_t nbNodesPerLeaf = dim == TWO_D
                                ? static_cast<uint32_t>((m_bx + 1) * (m_by + 1))
                                : static_cast<uint32_t>((m_bx + 1) * (m_by + 1) * (m_bz + 1));
    uint64_t totalNumOfCoords =
      3 * m_local_num_quads * static_cast<uint32_t>((m_bx + 1) * (m_by + 1) * (m_bz + 1));

    std::vector<float> data(totalNumOfCoords);

    /*
     * construct the list of node coordinates (using either inside or outside quadrants)
     */

    uint32_t inode = 0;
    for (int32_t iOct = 0; iOct < static_cast<int32_t>(m_local_num_quads); ++iOct)
    {
      auto iOct2 = m_use_outside_quads
                     ? iOct + m_mesh_map->get_amr_mesh_info().local_num_quadrants() +
                         m_mesh_map->get_amr_mesh_info().local_num_ghosts()
                     : iOct;

      const auto key = static_cast<uint64_t>(orchard_keys(iOct2));

      // compute cell length in vertex space
      const auto dx_cell = compute_cell_length<dim>(key, m_bx);

      // get lower left corner coordinates (in p4est connectivity coordinates units)
      auto lower_left_coord = orchard_key_to_vertex_coord<dim>(key, false);

      // if using outside quads, we need to shift coordinates, to actually get outside coordinates
      // TODO: refactor code below, it is too tightly linked to
      // MeshMap::compute_orchard_keys_view_host
      if (m_use_outside_quads and has_non_periodic_border)
      {
        lower_left_coord = outside_key_to_vertex_coord<dim>(key, false, brick_sizes);
      } // end if m_use_outside_quads

      // cell corner coordinates (in p4est connectivity coordinates units)
      Kokkos::Array<real_t, dim> vertex_coords;

      if constexpr (dim == 2)
      {
        for (int32_t iy = 0; iy < m_by + 1; ++iy)
        {
          for (int32_t ix = 0; ix < m_bx + 1; ++ix)
          {
            vertex_coords[IX] = lower_left_coord[IX] + ix * dx_cell;
            vertex_coords[IY] = lower_left_coord[IY] + iy * dx_cell;

            auto real_coords =
              vertex_coord_to_real_space<2>(vertex_coords, scaling_factor, xyz_min);

            data[3 * inode + 0] = static_cast<float>(real_coords[IX]);
            data[3 * inode + 1] = static_cast<float>(real_coords[IY]);
            data[3 * inode + 2] = static_cast<float>(0.0);
            inode++;
          }
        }
      }
      else if constexpr (dim == 3)
      {
        for (int32_t iz = 0; iz < m_bz + 1; ++iz)
        {
          for (int32_t iy = 0; iy < m_by + 1; ++iy)
          {
            for (int32_t ix = 0; ix < m_bx + 1; ++ix)
            {
              vertex_coords[IX] = lower_left_coord[IX] + ix * dx_cell;
              vertex_coords[IY] = lower_left_coord[IY] + iy * dx_cell;
              vertex_coords[IZ] = lower_left_coord[IZ] + iz * dx_cell;

              auto real_coords =
                vertex_coord_to_real_space<3>(vertex_coords, scaling_factor, xyz_min);

              data[3 * inode + 0] = static_cast<float>(real_coords[IX]);
              data[3 * inode + 1] = static_cast<float>(real_coords[IY]);
              data[3 * inode + 2] = static_cast<float>(real_coords[IZ]);
              inode++;
            }
          }
        }
      }
    } // end for iOct

    // get prepared for writing hdf5 file

    std::vector<std::size_t> dims{ 0, 0 };
    std::vector<std::size_t> count{ 0, 0 };
    std::vector<std::size_t> start{ 0, 0 };

    // get the dimensions and offset of the node coordinates array
    dims[0] = m_global_num_quads * nbNodesPerLeaf;
    dims[1] = 3;

    count[0] = m_local_num_quads * nbNodesPerLeaf;
    count[1] = 3;

    // get global index of the first octant of current mpi processor
    start[0] = m_global_first_quad * nbNodesPerLeaf;
    start[1] = 0;

    // write the node coordinates
    num_bytes += io_hdf5_writev(
      *m_hdf5_file, "/unstructured_mesh/", "coordinates", data.data(), dims, count, start);
  }
  else // cell-based AMR, one cell per quadrant
  {

    // array with all local nodes coordinates
    std::vector<float> data(3 * m_local_num_nodes);

    /*
     * construct the list of node coordinates
     */

    uint32_t inode = 0;
    for (int32_t iOct = 0; iOct < static_cast<int32_t>(m_local_num_quads); ++iOct)
    {

      auto iOct2 = m_use_outside_quads
                     ? iOct + m_mesh_map->get_amr_mesh_info().local_num_quadrants() +
                         m_mesh_map->get_amr_mesh_info().local_num_ghosts()
                     : iOct;

      const auto key = static_cast<uint64_t>(orchard_keys(iOct2));

      // compute cell length in vertex space
      const auto dx_cell = compute_cell_length<dim>(key, 1);

      // get lower left corner coordinates
      auto lower_left_coord = orchard_key_to_vertex_coord<dim>(key, false);

      // if using outside quads, we need to shift coordinates, to actually get outside coordinates
      // TODO: refactor code below, it is too tightly linked to
      // MeshMap::compute_orchard_keys_view_host
      if (m_use_outside_quads and has_non_periodic_border)
      {
        lower_left_coord = outside_key_to_vertex_coord<dim>(key, false, brick_sizes);
      } // end if m_use_outside_quads

      // cell corner coordinates (in p4est connectivity coordinates units)
      Kokkos::Array<real_t, dim> vertex_coords;

      if constexpr (dim == 2)
      {
        for (uint8_t iy = 0; iy < 2; ++iy)
        {
          for (uint8_t ix = 0; ix < 2; ++ix)
          {
            vertex_coords[IX] = lower_left_coord[IX] + ix * dx_cell;
            vertex_coords[IY] = lower_left_coord[IY] + iy * dx_cell;

            auto real_coords =
              vertex_coord_to_real_space<2>(vertex_coords, scaling_factor, xyz_min);

            data[3 * inode + 0] = static_cast<float>(real_coords[IX]);
            data[3 * inode + 1] = static_cast<float>(real_coords[IY]);
            data[3 * inode + 2] = static_cast<float>(0.0);
            inode++;
          }
        }
      }
      else if constexpr (dim == 3)
      {
        for (uint8_t iz = 0; iz < 2; ++iz)
        {
          for (uint8_t iy = 0; iy < 2; ++iy)
          {
            for (uint8_t ix = 0; ix < 2; ++ix)
            {
              vertex_coords[IX] = lower_left_coord[IX] + ix * dx_cell;
              vertex_coords[IY] = lower_left_coord[IY] + iy * dx_cell;
              vertex_coords[IZ] = lower_left_coord[IZ] + iz * dx_cell;

              auto real_coords =
                vertex_coord_to_real_space<3>(vertex_coords, scaling_factor, xyz_min);

              data[3 * inode + 0] = static_cast<float>(real_coords[IX]);
              data[3 * inode + 1] = static_cast<float>(real_coords[IY]);
              data[3 * inode + 2] = static_cast<float>(real_coords[IZ]);
              inode++;
            }
          }
        }
      }
    } // end for iOct

    // get prepared for writing hdf5 file

    std::vector<std::size_t> dims{ 0, 0 };
    std::vector<std::size_t> count{ 0, 0 };
    std::vector<std::size_t> start{ 0, 0 };

    // get the dimensions and offset of the node coordinates array
    dims[0] = m_global_num_nodes;
    dims[1] = 3;

    count[0] = m_local_num_nodes;
    count[1] = 3;

    // get global index of the first octant of current mpi processor
    start[0] = m_nbNodesPerCell * m_global_first_quad;
    start[1] = 0;

    // write the node coordinates
    num_bytes += io_hdf5_writev(
      *m_hdf5_file, "/unstructured_mesh/", "coordinates", data.data(), dims, count, start);

  } // end one cell per quadrant

  return num_bytes;

} // HDF5_Xdmf_Writer<dim, device_t>::io_hdf5_write_coordinates

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer<dim, device_t>::io_hdf5_write_connectivity()
{

  uint64_t num_bytes = 0;

  if (m_write_block_data)
  {

    int nbNodesPerLeaf =
      dim == TWO_D ? (m_bx + 1) * (m_by + 1) : (m_bx + 1) * (m_by + 1) * (m_bz + 1);

    // TODO: clarification: should/can data type used here be int64_t ?
    // - from xdmf specification it is not clear
    // - paraview seems ok to read int64_t
    std::vector<int64_t> data(m_local_num_quads * m_nbCellsPerLeaf * m_nbNodesPerCell);

    // first node of current MPI process
    int64_t globalNodeOffset = nbNodesPerLeaf * static_cast<int64_t>(m_global_first_quad);

    const auto nbConnectivityPerLeaf = m_nbCellsPerLeaf * m_nbNodesPerCell;

    // get connectivity data
    for (size_t iLeaf = 0; iLeaf < m_local_num_quads; ++iLeaf)
    {
      int64_t localNodeOffset = nbNodesPerLeaf * static_cast<int64_t>(iLeaf);

      // sweep subcells
      int nz = dim == 2 ? 1 : m_bz;
      for (int jz = 0; jz < nz; ++jz)
      {
        for (int jy = 0; jy < m_by; ++jy)
        {
          for (int jx = 0; jx < m_bx; ++jx)
          {

            int64_t nodeOffset = globalNodeOffset + localNodeOffset;

            size_t idx = nbConnectivityPerLeaf * iLeaf +
                         static_cast<size_t>(subCellIndex(jx, jy, jz)) * m_nbNodesPerCell;

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

    start[0] = m_global_first_quad * m_nbCellsPerLeaf;
    start[1] = 0;

    // write cell-to-cell connectivity
    num_bytes += io_hdf5_writev(
      *m_hdf5_file, "/unstructured_mesh/", "connectivity", data.data(), dims, count, start);
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

    uint32_t in = 0;

    // get connectivity data
    for (uint32_t i = 0; i < m_local_num_quads; ++i)
    {

      for (uint32_t j = 0; j < m_nbNodesPerCell; ++j)
      {
        size_t idx = m_nbNodesPerCell * i + j;

        data[idx] = static_cast<int64_t>(m_nbNodesPerCell * m_global_first_quad + in + node[j]);
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

    start[0] = m_global_first_quad;
    start[1] = 0;

    // write cell-to-cell connectivity
    num_bytes += io_hdf5_writev(
      *m_hdf5_file, "/unstructured_mesh/", "connectivity", data.data(), dims, count, start);
  }

  return num_bytes;

} // HDF5_Xdmf_Writer<dim, device_t>::io_hdf5_write_connectivity

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer<dim, device_t>::io_hdf5_write_level()
{

  if (!this->m_write_level)
  {
    return 0;
  }

  const auto orchard_keys = m_mesh_map->orchard_keys_host();

  const uint32_t nbData = m_local_num_quads * m_nbCellsPerLeaf;

  std::vector<int> data(nbData);

  size_t i = 0;
  for (int32_t iOct = 0; iOct < static_cast<int32_t>(m_local_num_quads); ++iOct)
  {
    auto iOct2 = m_use_outside_quads
                   ? iOct + m_mesh_map->get_amr_mesh_info().local_num_quadrants() +
                       m_mesh_map->get_amr_mesh_info().local_num_ghosts()
                   : iOct;

    auto level = orchard_key_t<dim>::level(orchard_keys(iOct2));

    for (uint32_t j = 0; j < m_nbCellsPerLeaf; ++j)
    {
      data[i] = level;
      ++i;
    }
  } // end for iOct

  return write_cc_attribute("level", data.data(), 0, IO_CELL_SCALAR);

} // HDF5_Xdmf_Writer<dim, device_t>::io_hdf5_write_level

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer<dim, device_t>::io_hdf5_write_tree()
{

  if (!this->m_write_tree)
  {
    return 0;
  }

  const auto orchard_keys = m_mesh_map->orchard_keys_host();

  const auto brick_sizes = get_brick_sizes<dim>(m_config_map);

  // tree linear index to xyz converter
  BrickConnectivityData<dim> convert(brick_sizes);

  uint32_t nbData = m_local_num_quads * m_nbCellsPerLeaf;

  std::vector<int> data(nbData);

  size_t i = 0;

  auto get_treeid = [&](uint64_t key) {
    auto tree_coords = orchard_key_t<dim>::get_tree_coords(key);

    if constexpr (dim == 2)
      return convert.treeId(tree_coords[0], tree_coords[1]);
    else if constexpr (dim == 3)
      return convert.treeId(tree_coords[0], tree_coords[1], tree_coords[2]);
  };


  for (int32_t iOct = 0; iOct < static_cast<int32_t>(m_local_num_quads); ++iOct)
  {

    const auto iOct2 = m_use_outside_quads
                         ? iOct + m_mesh_map->get_amr_mesh_info().local_num_quadrants() +
                             m_mesh_map->get_amr_mesh_info().local_num_ghosts()
                         : iOct;

    const auto treeid = get_treeid(orchard_keys(iOct2));

    for (uint32_t j = 0; j < m_nbCellsPerLeaf; ++j)
    {
      data[i] = treeid;
      ++i;
    }
  } // end for iOct

  return write_cc_attribute("treeid", data.data(), 0, IO_CELL_SCALAR);

} // HDF5_Xdmf_Writer<dim, device_t>::io_hdf5_write_tree

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer<dim, device_t>::io_hdf5_write_iOct()
{

  if (!this->m_write_iOct)
  {
    return 0;
  }

  uint32_t nbData = m_local_num_quads * m_nbCellsPerLeaf;

  std::vector<uint32_t> data(nbData);

  // gather level for each local quadrant
  size_t i = 0;
  for (int32_t iOct = 0; iOct < static_cast<int32_t>(m_local_num_quads); ++iOct)
  {

    auto iOct2 = m_use_outside_quads
                   ? iOct + m_mesh_map->get_amr_mesh_info().local_num_quadrants() +
                       m_mesh_map->get_amr_mesh_info().local_num_ghosts()
                   : iOct;

    for (uint32_t j = 0; j < m_nbCellsPerLeaf; ++j)
    {
      data[i] = static_cast<uint32_t>(iOct2);
      ++i;
    }
  }

  return write_cc_attribute("iOct", data.data(), 0, IO_CELL_SCALAR);

} // HDF5_Xdmf_Writer<dim, device_t>::io_hdf5_write_iOct

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer<dim, device_t>::io_hdf5_write_rank()
{

  if (!this->m_write_rank)
  {
    return 0;
  }

  uint32_t nbData = m_local_num_quads * m_nbCellsPerLeaf;

  std::vector<int> data(nbData);

  // gather rank for each local quadrant
  for (size_t i = 0; i < nbData; ++i)
  {
    data[i] = m_par_env.rank();
  }

  return write_cc_attribute("rank", data.data(), 0, IO_CELL_SCALAR);

} // HDF5_Xdmf_Writer<dim, device_t>::io_hdf5_write_rank

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer<dim, device_t>::io_hdf5_write_reduced_orchard_key(orchard_key_dump_type_t type)
{
  const auto orchard_keys = m_mesh_map->orchard_keys_host();

  const uint32_t nbData = m_local_num_quads * m_nbCellsPerLeaf;

  std::vector<uint64_t> data(nbData);

  // check if connectivity is either "unit" or "brick"
  auto conn_name = m_config_map.getString("amr", "connectivity", "invalid_connectivity");

  if (conn_name == "brick")
  {

    const auto brick_sizes = get_brick_sizes<dim>(m_config_map);

    // tree linear index to xyz converter
    BrickConnectivityData<dim> convert(brick_sizes);

    auto max_level = m_config_map.getInteger("amr", "level_max", 0);

    size_t icell = 0;
    for (int32_t iOct = 0; iOct < static_cast<int32_t>(m_local_num_quads); ++iOct)
    {
      auto iOct2 = m_use_outside_quads
                     ? iOct + m_mesh_map->get_amr_mesh_info().local_num_quadrants() +
                         m_mesh_map->get_amr_mesh_info().local_num_ghosts()
                     : iOct;

      const auto key = orchard_keys(iOct2);

      if (type == FULL)
      {
        for (size_t i = 0; i < m_nbCellsPerLeaf; ++i)
        {
          data[icell] = key;
          icell++;
        }
      }
      else
      {
        // reduced key
        auto octCoords = orchard_key_t<dim>::get_octant_coords(key);
        auto treeCoords = orchard_key_t<dim>::get_tree_coords(key);

        // shift bit to reduced to the effective number of levels allowed in input parameters
        octCoords[IX] >>= (orchard_key_t<dim>::MAX_LEVEL - max_level);
        octCoords[IY] >>= (orchard_key_t<dim>::MAX_LEVEL - max_level);
        if constexpr (dim == 3)
          octCoords[IZ] >>= (orchard_key_t<dim>::MAX_LEVEL - max_level);

        // re-encode key using only tree and octant information
        // set level to zero (reduced key, no level involved)
        uint64_t orchard_key = orchard_key_t<dim>::encode_orchard(treeCoords, octCoords, 0);
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

    } // end for iOct
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

  return write_cc_attribute(attribute_str, data.data(), 0, IO_CELL_SCALAR);


} // HDF5_Xdmf_Writer<dim, device_t>::io_hdf5_write_reduced_orchard_key

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer<dim, device_t>::io_hdf5_write_at_domain_border()
{

  if (!this->m_write_at_domain_border)
  {
    return 0;
  }

  const auto orchard_keys = m_mesh_map->orchard_keys_host();

  const uint32_t nbData = m_local_num_quads * m_nbCellsPerLeaf;

  std::vector<uint32_t> data(nbData);
  size_t                icell = 0;

  // check if connectivity is either "unit" or "brick"
  auto conn_name = m_config_map.getString("amr", "connectivity", "invalid_connectivity");

  if (conn_name == "brick")
  {
    // gather orchard key for each local quadrant / cell

    // brick sizes
    const auto brick_sizes = get_brick_sizes<dim>(m_config_map);

    // tree linear index to xyz converter
    BrickConnectivityData<dim> convert(brick_sizes);

    for (int32_t iOct = 0; iOct < static_cast<int32_t>(m_local_num_quads); ++iOct)
    {
      auto iOct2 = m_use_outside_quads
                     ? iOct + m_mesh_map->get_amr_mesh_info().local_num_quadrants() +
                         m_mesh_map->get_amr_mesh_info().local_num_ghosts()
                     : iOct;

      auto orchard_key = orchard_keys(iOct2);

      for (size_t i = 0; i < m_nbCellsPerLeaf; ++i)
      {
        auto atDomainBorder = static_cast<uint32_t>(
          orchard_key_t<dim>::is_at_any_domain_border(orchard_key, brick_sizes));
        data[icell] = atDomainBorder;
        icell++;
      }
    } // end for iOct
  }
  else
  {
    // not defined, so use "zero" as default value
    // writing this attribute should be disabled in config_map
    for (size_t i = 0; i < nbData; ++i)
    {
      data[i] = 0;
    }
  }

  return write_cc_attribute("at_domain_border", data.data(), 0, IO_CELL_SCALAR);

} // HDF5_Xdmf_Writer<dim, device_t>::io_hdf5_write_at_domain_border

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer<dim, device_t>::io_hdf5_write_at_tree_border()
{

  if (!this->m_write_at_tree_border)
  {
    return 0;
  }

  const auto orchard_keys = m_mesh_map->orchard_keys_host();

  const uint32_t nbData = m_local_num_quads * m_nbCellsPerLeaf;

  std::vector<uint32_t> data(nbData);
  size_t                icell = 0;

  // check if connectivity is either "unit" or "brick"
  auto conn_name = m_config_map.getString("amr", "connectivity", "invalid_connectivity");

  if (conn_name == "brick")
  {
    // gather orchard for each local quadrant / cell

    // brick sizes
    const auto brick_sizes = get_brick_sizes<dim>(m_config_map);

    // tree linear index to xyz converter
    BrickConnectivityData<dim> convert(brick_sizes);

    // loop over all octants
    for (int32_t iOct = 0; iOct < static_cast<int32_t>(m_local_num_quads); ++iOct)
    {
      auto iOct2 = m_use_outside_quads
                     ? iOct + m_mesh_map->get_amr_mesh_info().local_num_quadrants() +
                         m_mesh_map->get_amr_mesh_info().local_num_ghosts()
                     : iOct;

      auto orchard_key = orchard_keys(iOct2);

      auto at_any_tree_border =
        static_cast<uint32_t>(orchard_key_t<dim>::is_at_any_tree_border(orchard_key));

      auto at_any_tree_corner =
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

    } // end for iOct
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

  return write_cc_attribute("at_tree_border", data.data(), 0, IO_CELL_SCALAR);

} // HDF5_Xdmf_Writer<dim, device_t>::io_hdf5_write_at_tree_border

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
HDF5_Xdmf_Writer<dim, device_t>::io_hdf5_write_is_outside()
{

  if (!this->m_write_is_outside)
  {
    return 0;
  }

  const auto orchard_keys = m_mesh_map->orchard_keys_host();

  const uint32_t nbData = m_local_num_quads * m_nbCellsPerLeaf;

  std::vector<uint32_t> data(nbData);
  size_t                icell = 0;

  // check if connectivity is either "unit" or "brick"
  auto conn_name = m_config_map.getString("amr", "connectivity", "invalid_connectivity");

  if (conn_name == "brick")
  {
    // gather orchard for each local quadrant / cell

    // brick sizes
    const auto brick_sizes = get_brick_sizes<dim>(m_config_map);

    // tree linear index to xyz converter
    BrickConnectivityData<dim> convert(brick_sizes);

    // loop over all octants
    for (int32_t iOct = 0; iOct < static_cast<int32_t>(m_local_num_quads); ++iOct)
    {
      auto iOct2 = m_use_outside_quads
                     ? iOct + m_mesh_map->get_amr_mesh_info().local_num_quadrants() +
                         m_mesh_map->get_amr_mesh_info().local_num_ghosts()
                     : iOct;

      auto orchard_key = orchard_keys(iOct2);

      for (size_t i = 0; i < m_nbCellsPerLeaf; ++i)
      {
        data[icell] = static_cast<uint32_t>(orchard_key_t<dim>::is_outside(orchard_key));
        icell++;
      }

    } // end for iOct
  }
  else
  {
    // not defined, so use "zero" as default value
    // writing this attribute should be disabled in config_map
    for (size_t i = 0; i < nbData; ++i)
    {
      data[i] = 0;
    }
  }

  return write_cc_attribute("is_outside", data.data(), 0, IO_CELL_SCALAR);

} // HDF5_Xdmf_Writer<dim, device_t>::io_hdf5_write_is_outside

// =======================================================
// =======================================================
// Private members
// =======================================================
// =======================================================

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer<dim, device_t>::io_xdmf_write_main_header()
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

} // HDF5_Xdmf_Writer<dim, device_t>::io_xdmf_write_main_header

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer<dim, device_t>::io_xdmf_write_header(double time)
{

  FILE * fd = this->m_xdmf_file;
  size_t global_num_cells = this->m_global_num_quads;
  size_t global_num_nodes = global_num_cells * m_nbNodesPerCell;

  if (m_write_block_data)
  {

    global_num_cells *= m_nbCellsPerLeaf;

    int nbNodesPerLeaf =
      dim == TWO_D ? (m_bx + 1) * (m_by + 1) : (m_bx + 1) * (m_by + 1) * (m_bz + 1);

    global_num_nodes = this->m_global_num_quads * static_cast<size_t>(nbNodesPerLeaf);
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
  fprintf(fd, "         %s.h5:/unstructured_mesh/connectivity\n", this->m_basename.c_str());
  fprintf(fd, "        </DataItem>\n");
  fprintf(fd, "      </Topology>\n"); // Points
  fprintf(fd, "      <Geometry GeometryType=\"XYZ\">\n");
  fprintf(fd,
          "        <DataItem Dimensions=\"%lu 3\" %s Format=\"HDF\">\n",
          global_num_nodes,
          IO_XDMF_NUMBER_TYPE);
  fprintf(fd, "         %s.h5:/unstructured_mesh/coordinates\n", this->m_basename.c_str());
  fprintf(fd, "        </DataItem>\n");
  fprintf(fd, "      </Geometry>\n");

} // HDF5_Xdmf_Writer<dim, device_t>::io_xdmf_write_header

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer<dim, device_t>::io_xdmf_write_main_include(const std::string & name)
{

  FILE * fd = this->m_main_xdmf_file;

  fprintf(fd,
          "      <xi:include href=\"%s\""
          " xpointer=\"xpointer(//Xdmf/Domain/Grid)\" />\n",
          name.c_str());

} // HDF5_Xdmf_Writer<dim, device_t>::io_xdmf_write_main_include

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer<dim, device_t>::io_xdmf_write_attribute(const std::string &         name,
                                                         const std::string &         number_type,
                                                         io_attribute_type_t         type,
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
    // case IO_CELL_VECTOR:
    //   fprintf(fd, " AttributeType=\"Vector\" Center=\"Cell\">\n");
    //   break;
    case IO_NODE_SCALAR:
      fprintf(fd, " AttributeType=\"Scalar\" Center=\"Node\">\n");
      break;
    // case IO_NODE_VECTOR:
    //   fprintf(fd, " AttributeType=\"Vector\" Center=\"Node\">\n");
    //   break;
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
    assertm(dims.size() == 2, "[HDF5_Xdmf_Writer] std::vector dims must have size equal to 2.");
    fprintf(fd, " Dimensions=\"%zu %zu\">\n", dims[0], dims[1]);
  }

  fprintf(fd, "         %s.h5:/celldata/%s\n", basename.c_str(), name.c_str());
  fprintf(fd, "        </DataItem>\n");
  fprintf(fd, "      </Attribute>\n");

} // HDF5_Xdmf_Writer<dim, device_t>::io_xdmf_write_attribute

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer<dim, device_t>::io_xdmf_write_main_footer()
{

  FILE * fd = this->m_main_xdmf_file;

  fprintf(fd, "    </Grid>\n");
  fprintf(fd, "  </Domain>\n");
  fprintf(fd, "</Xdmf>\n");

} // HDF5_Xdmf_Writer<dim, device_t>::io_xdmf_write_main_footer

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer<dim, device_t>::io_xdmf_write_footer()
{

  FILE * fd = this->m_xdmf_file;

  fprintf(fd, "    </Grid>\n");
  fprintf(fd, "  </Domain>\n");
  fprintf(fd, "</Xdmf>\n");

} // HDF5_Xdmf_Writer<dim, device_t>::io_xdmf_write_footer

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
int32_t
HDF5_Xdmf_Writer<dim, device_t>::subCellIndex(int jx, int jy, int jz)
{

  return jx + m_bx * (jy + m_by * jz);

} // HDF5_Xdmf_Writer<dim, device_t>::subCellIndex

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
int32_t
HDF5_Xdmf_Writer<dim, device_t>::subNodeIndex(int jx, int jy, int jz)
{

  return jx + (m_bx + 1) * (jy + (m_by + 1) * jz);

} // HDF5_Xdmf_Writer<dim, device_t>::subNodeIndex

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer<dim, device_t>::set_block_mode()
{

  m_writing_mode = BLOCK_MODE;
  m_bx = m_config_map.getInteger("amr", "bx", 1);
  m_by = m_config_map.getInteger("amr", "by", 1);
  m_bz = dim == 2 ? 1 : m_config_map.getInteger("amr", "bz", 1);

  m_write_block_data = true;
  m_extract_block_ghost = false;
  m_nbCellsPerLeaf =
    dim == 2 ? static_cast<uint32_t>(m_bx * m_by) : static_cast<uint32_t>(m_bx * m_by * m_bz);

} // HDF5_Xdmf_Writer<dim, device_t>::set_block_mode()

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer<dim, device_t>::set_block_mode(block_size_t<dim> block_size,
                                                block_size_t<dim> ghost_width)
{

  m_writing_mode = BLOCK_MODE;
  m_bx = block_size[0];
  m_by = block_size[1];
  m_bz = dim == 3 ? block_size[2] : 1;

  m_write_block_data = true;
  m_extract_block_ghost = true;
  m_nbCellsPerLeaf =
    dim == 2 ? static_cast<uint32_t>(m_bx * m_by) : static_cast<uint32_t>(m_bx * m_by * m_bz);

  m_gx = ghost_width[0];
  m_gy = ghost_width[1];
  m_gz = dim == 3 ? ghost_width[2] : 0;

} // HDF5_Xdmf_Writer<dim, device_t>::set_block_mode()

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
HDF5_Xdmf_Writer<dim, device_t>::set_leaf_mode()
{

  m_writing_mode = LEAF_MODE;
  m_bx = 1;
  m_by = 1;
  m_bz = 1;
  m_write_block_data = false;
  m_nbCellsPerLeaf =
    dim == 2 ? static_cast<uint32_t>(m_bx * m_by) : static_cast<uint32_t>(m_bx * m_by * m_bz);

} // HDF5_Xdmf_Writer<dim, device_t>::set_block_mode()

// =====================================================================
// =====================================================================

} // namespace kalypsso

#endif // KALYPSSO_CORE_HDF5_XDMF_WRITER_H_
