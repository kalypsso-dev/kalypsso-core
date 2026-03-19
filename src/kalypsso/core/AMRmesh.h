// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file AMRmesh.h
 * \brief Container for p4est mesh data (p4est, ghost, connectivity).
 *
 */
#ifndef KALYPSSO_CORE_AMRMESH_H_
#define KALYPSSO_CORE_AMRMESH_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm, kalypsso_core_config.h
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/p4est/p4est_wrapper.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>
#include <kalypsso/core/AMRmesh_utils.h>
#include <kalypsso/core/kokkos_shared.h>

#include <iostream>

#ifdef KALYPSSO_CORE_USE_MPI
#  include <mpi.h>
#endif // KALYPSSO_CORE_USE_MPI

namespace kalypsso
{

// forward declare
template <size_t dim>
class AMRmesh;

/**
 * Callback to perform uniform refinement (input to p4est_refine).
 *
 * The one and only use of this callback is to feed p4est::Wrapper<dim>::new_forest with a callback
 * function called one for every quadrant in the forest creation.
 */
template <size_t dim>
static int
uniform_refine_fn(typename kalypsso::p4est::Wrapper<dim>::forest_t *   forest,
                  [[maybe_unused]] typename kalypsso::p4est::topidx_t  which_tree,
                  typename kalypsso::p4est::Wrapper<dim>::quadrant_t * quadrant)
{
  using p4est_userdata_t = typename AMRmesh<dim>::p4est_userdata_t;

  p4est_userdata_t * p4est_userdata = static_cast<p4est_userdata_t *>(forest->user_pointer);
  const auto         level_min = p4est_userdata->level_min;

  if (quadrant->level < level_min)
    return 1;

  return 0;
} // uniform_refine_fn

/**
 * Default callback used when creating the forest.
 *
 * All such callback should initialize the delay memory parameter (see coarsen_delay in p4est
 * sources).
 *
 * The one and only use of this callback is to feed p4est::Wrapper<dim>::new_forest with a callback
 * function called one for every quadrant in the forest creation.
 */
template <size_t dim>
static void
initial_quad_fn([[maybe_unused]] typename kalypsso::p4est::Wrapper<dim>::forest_t * forest,
                [[maybe_unused]] typename kalypsso::p4est::topidx_t                 which_tree,
                typename kalypsso::p4est::Wrapper<dim>::quadrant_t *                quadrant)
{

  // initialize delay memory in the quadrants' user field.
  // user_int is a p4est internal metadata field (avail for each quadrant)
  quadrant->p.user_int = 0;

} // initial_quad_fn


/**
 * \class AMRmesh
 *
 * This class is doing :
 * - RAII, resources management (alloc / dealloc, resize) for p4est objects: connectivity,
 * geometry, forest and ghost.
 *
 */
template <size_t dim>
class AMRmesh
{
public:
  //! type alias to access p4est C API (2D or 3D)
  using p4est_t = typename p4est::Wrapper<dim>;

  using forest_t = typename p4est_t::forest_t;
  using tree_t = typename p4est_t::tree_t;
  using connectivity_t = typename p4est_t::connectivity_t;
  using geometry_t = typename p4est_t::geometry_t;
  using quadrant_t = typename p4est_t::quadrant_t;
  using ghost_t = typename p4est_t::ghost_t;

  //! p4est user-defined meta data attached to each quadrant (allocated inside p4est)
  //! currently not used, TODO: evaluate if we really need it, and for what purpose
  // struct user_data_t
  // {
  //   double x;
  // };

  //! Additional data structure type passed to p4est_new last argument.
  //! there is only one instance per forest; it can be used to passed metadata can be used
  //! inside p4est callback's
  struct p4est_userdata_t
  {
    geometry_t * geom;
    uint8_t      level_min;
    uint8_t      level_max;
  };

  //! constructor : create/allocate a p4est object with connectivity and geometry
  AMRmesh([[maybe_unused]] ParallelEnv const & par_env, const ConfigMap & config_map);

  //! destructor : destroy connectivity, geometry and p4est mesh
  ~AMRmesh();

  //! remove copy constructor
  AMRmesh(const AMRmesh & other) = delete;

  //! remove copy assign constructor
  AMRmesh &
  operator=(const AMRmesh & other) = delete;

  //! reset p4est resource
  void
  reset_p4est_resources();

  //! create p4est resources from file (created by p4est_save).
  //! this is essentially a wrapper around p4est_load
  void
  create_p4est_resources_from_file(ParallelEnv const & par_env,
                                   ConfigMap const &   config_map,
                                   std::string const & p4est_filename);

  forest_t *
  forest()
  {
    return m_forest;
  }

  const forest_t *
  forest() const
  {
    return m_forest;
  }

  connectivity_t *
  connectivity()
  {
    return m_connectivity;
  }

  const connectivity_t *
  connectivity() const
  {
    return m_connectivity;
  }

  std::string
  connectivity_name()
  {
    return m_connectivity_name;
  }

  const std::string &
  connectivity_name() const
  {
    return m_connectivity_name;
  }

  geometry_t *
  geometry()
  {
    return m_geometry;
  }

  const geometry_t *
  geometry() const
  {
    return m_geometry;
  }

  std::string
  geometry_name()
  {
    return m_geometry_name;
  }

  const std::string &
  geometry_name() const
  {
    return m_geometry_name;
  }

  ghost_t *
  ghost()
  {
    return m_ghost;
  }

  const ghost_t *
  ghost() const
  {
    return m_ghost;
  }

  void
  reset_ghost()
  {
    if (m_ghost != nullptr)
    {
      p4est_t::ghost_destroy(m_ghost);
      m_ghost = nullptr;
    }

    m_ghost = p4est_t::ghost_new(m_forest, p4est_t::CONNECT_FULL);
  }

  //! get number of owned quadrants / octants in current MPI process
  int32_t
  local_num_quadrants() const
  {
    return m_forest->local_num_quadrants;
  }

  //! get number of quadrants / octants in current MPI process
  int32_t
  global_num_quadrants() const
  {
    return m_forest->global_num_quadrants;
  }

  //! get the total number of ghost quadrants / octants in current MPI process
  int32_t
  local_num_ghosts() const
  {
    return static_cast<int32_t>(m_ghost->ghosts.elem_count);
  }

  //! get total number of owned + ghost quadrants in current MPI process
  //! WARNING/FYI: not counting outside quadrant here (this is something we added, not part of
  //! p4est)
  int32_t
  local_num_quadrants_total() const
  {
    return m_forest->local_num_quadrants + static_cast<int32_t>(m_ghost->ghosts.elem_count);
  }

  //! get the number of ghost quadrant in local MPI process to be filled with data received from
  //! MPI process of rank 'iproc' \param[in] iproc is the MPI process rank from which we receive
  //! data.
  //!
  //! Remainder proc_offsets is an array of size mpi size + 1, so no overflow here
  int32_t
  local_num_ghosts(int iproc) const
  {
    assertm(iproc >= 0 and iproc < m_forest->mpisize,
            "[AMRmesh::local_num_ghosts] wrong MPI rank value.");
    return m_ghost->proc_offsets[iproc + 1] - m_ghost->proc_offsets[iproc];
  }

  //! get number of owned mirror quadrants / octants in current MPI process
  //! involved in MPI ghost communication.
  //! Important \note
  //! if a mirror quadrant is involved in several communication (e.g. at at corner where several MPI
  //! sub-domains are neighbors) this quadrant is counted as many times.
  //! As a remainder: ghost->mirrors.elem_count gives the number of unique mirror quadrants in
  //! current MPI process not to be confused with ghost->mirror_proc_offsets[m_par_env.size()] which
  //! is what we are looking for.
  int32_t
  local_num_mirrors() const
  {
    return m_ghost->mirror_proc_offsets[m_forest->mpisize];
  }

  //! number of local (owned by current MPI process) mirror quadrants involved in ghost
  //! communication with MPI process of rank 'iproc'
  //!
  //! \param[in] iproc is the rank of the remote process where mirror quadrant are to be sent
  int32_t
  local_num_mirrors(int iproc) const
  {
    assertm(iproc >= 0 and iproc < m_forest->mpisize,
            "[AMRmesh::local_num_mirrors] wrong MPI rank value.");
    return m_ghost->mirror_proc_offsets[iproc + 1] - m_ghost->mirror_proc_offsets[iproc];
  }

  //! return the num of mirror quadrants in current process.
  //! each quadrant is counted only once, not matter if quadrant is involved in several MPI
  //! communications.
  int32_t
  local_num_mirrors_unique() const
  {
    return m_ghost->mirrors.elem_count;
  }

private:
  //! the MAIN p4est object for accessing all mesh information
  forest_t * m_forest;

  //! inter-tree connectivity.
  //! see https://github.com/cburstedde/p4est/blob/master/src/p4est_connectivity.h#L152
  connectivity_t * m_connectivity;

  //! inter-tree connectivity name (only "brick" is supported by kalypsso currently)
  std::string m_connectivity_name;

  //! additional geometrical mapping (x,y,z) => (X,Y,Z) from the connectivity space to another space
  //! used to modify/deform a mesh
  //! see https://github.com/cburstedde/p4est/blob/master/src/p4est_geometry.h#L59
  //! currently not used nor supported in kalypsso; we only use the brick connectivity without
  //! additional mapping.
  //! TODO: see any real application might need this.
  geometry_t * m_geometry;

  //! geometry name
  std::string m_geometry_name;

  //! the main p4est object for handling /transfer MPI ghost quadrant metadata.
  //! ghost user data transfer is done elsewhere
  ghost_t * m_ghost = nullptr;

  //! TODO: clarify if we need it in kalypsso.
  p4est_userdata_t m_p4est_userdata;

  //! helper to initialize a p4est connectivity
  connectivity_t *
  create_connectivity(const ConfigMap & config_map)
  {
    m_connectivity_name = config_map.getString("amr", "connectivity", "invalid_connectivity");

    // if connectivity name is not provided in input parameter file,
    // then force it, or maybe we should abort - TBD
    if (m_connectivity_name.compare("invalid_connectivity") == 0)
    {
      m_connectivity_name = dim == 2 ? "shell2d" : "shell";
      std::cout
        << "KALYPSSO warning : p4est connectivity missing in input parameter file !!! Please "
           "check !\n";
    }

    return p4est_t::my_connectivity_new_byname(m_connectivity_name.c_str(), config_map);
  }

  //! helper to initialize a p4est geometry (can be nullptr)
  geometry_t *
  create_geometry(connectivity_t * conn, const ConfigMap & config_map)
  {
    m_geometry_name = config_map.getString("amr", "geometry", "no_geometry");

    return p4est_t::my_geometry_new_byname(m_geometry_name.c_str(), conn, config_map);
  }

}; // class AMRmesh

// =====================================================================
// =====================================================================
// constructor
template <size_t dim>
AMRmesh<dim>::AMRmesh([[maybe_unused]] ParallelEnv const & par_env, const ConfigMap & config_map)
{
  m_connectivity = create_connectivity(config_map);

  m_geometry = create_geometry(m_connectivity, config_map);

  // initial refinement to reach at least level_min
  m_p4est_userdata.level_min = static_cast<uint8_t>(config_map.getInteger("amr", "level_min", 4));
  m_p4est_userdata.level_max = static_cast<uint8_t>(config_map.getInteger("amr", "level_max", 4));
  m_p4est_userdata.geom = m_geometry;

  m_forest = p4est_t::new_forest(
#ifdef KALYPSSO_CORE_USE_MPI
    par_env.mpi_comm(), // MPI communicator
#else
    sc_MPI_COMM_WORLD,
#endif
    m_connectivity,                          // tree connectivity
    0,                                       // minimal initial quad per MPI proc
    0,                                       // min level
    0,                                       // fill uniform is false
    0,                                       // sizeof(user_data_t),  // currently not used
    initial_quad_fn<dim>,                    // no user data init callback function
    static_cast<void *>(&m_p4est_userdata)); // p4est user data

  p4est_t::refine(m_forest, 1, uniform_refine_fn<dim>, initial_quad_fn<dim>);

  reset_ghost();

} // AMRmesh

// =====================================================================
// =====================================================================
template <size_t dim>
AMRmesh<dim>::~AMRmesh()
{
  reset_p4est_resources();
} // ~AMRmesh

// =====================================================================
// =====================================================================
template <size_t dim>
void
AMRmesh<dim>::reset_p4est_resources()
{
  if (m_ghost != nullptr)
  {
    p4est_t::ghost_destroy(m_ghost);
    m_ghost = nullptr;
  }

  // destroy p4est
  if (m_forest != nullptr)
  {
    p4est_t::destroy(m_forest);
    m_forest = nullptr;
  }

  // destroy geometry
  if (m_geometry != nullptr)
  {
    p4est_t::geometry_destroy(m_geometry);
    m_geometry = nullptr;
  }

  // destroy connectivity
  if (m_connectivity != nullptr)
  {
    p4est_t::connectivity_destroy(m_connectivity);
    m_connectivity = nullptr;
  }

} // AMRmesh::reset_p4est_resources

// =====================================================================
// =====================================================================
template <size_t dim>
void
AMRmesh<dim>::create_p4est_resources_from_file([[maybe_unused]] ParallelEnv const & par_env,
                                               ConfigMap const &                    config_map,
                                               std::string const &                  p4est_filename)
{
  reset_p4est_resources();

  m_forest = p4est_t::load_ext(p4est_filename.c_str(),
#ifdef KALYPSSO_CORE_USE_MPI
                               par_env.mpi_comm(),
#else
                               sc_MPI_COMM_WORLD,
#endif
                               0 /* data size */,
                               0 /* load_data */,
                               1 /* autopartition */,
                               1 /* broadcast header */,
                               nullptr /* user pointer */,
                               &m_connectivity);

  m_geometry = create_geometry(m_connectivity, config_map);

  // initial refinement to reach at least level_min
  m_p4est_userdata.level_min = static_cast<uint8_t>(config_map.getInteger("amr", "level_min", 4));
  m_p4est_userdata.level_max = static_cast<uint8_t>(config_map.getInteger("amr", "level_max", 4));
  m_p4est_userdata.geom = m_geometry;

  m_forest->user_pointer = static_cast<void *>(&m_p4est_userdata);

  reset_ghost();

} // AMRmesh<dim>::create_p4est_resources_from_file

} // namespace kalypsso

#endif // KALYPSSO_CORE_AMRMESH_H_
