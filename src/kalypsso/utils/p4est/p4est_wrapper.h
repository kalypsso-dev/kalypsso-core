// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * @file p4est_wrapper.h
 * @date 17 octobre 2020
 * @author pkestene
 * @note
 *
 * Add a wrapper unifying 2D / 3D interface to p4est.
 *
 * This file is part of the Kalypsso software project.
 *
 */
#ifndef KALYPSSO_UTILS_P4EST_P4ESTWRAPPER_H_
#define KALYPSSO_UTILS_P4EST_P4ESTWRAPPER_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/kalypsso_macros.h>

#include <type_traits>

#include <p4est.h> // for 2D
#include <p4est_extended.h>
#include <p4est_ghost.h>
#include <p4est_bits.h>
#include <p4est_vtk.h>
#include <p4est_connectivity.h>
#include <p4est_geometry.h>
#include <p4est_mesh.h>
#include <p4est_nodes.h>
#include <p4est_lnodes.h>
#include <p4est_communication.h>
#include <p4est_extended.h>
#include <p4est_wrap.h>

#include <p8est.h> // for 3D
#include <p8est_extended.h>
#include <p8est_ghost.h>
#include <p8est_bits.h>
#include <p8est_vtk.h>
#include <p8est_connectivity.h>
#include <p8est_geometry.h>
#include <p8est_mesh.h>
#include <p8est_nodes.h>
#include <p8est_lnodes.h>
#include <p8est_communication.h>
#include <p8est_extended.h>
#include <p8est_wrap.h>

#include <cstdint> // for standard integral types
#include <cstring> // for memset

#if defined(KALYPSSO_CORE_USE_MPI)
#  include <mpi.h>
#else
// typedef int MPI_Comm;
#endif


namespace kalypsso
{

namespace p4est
{

using qcoord_t = p4est_qcoord_t;
using topidx_t = p4est_topidx_t;
using locidx_t = p4est_locidx_t;
using gloidx_t = p4est_gloidx_t;

/** flags used in refine / coarsen callback functions */
static constexpr int P4EST_WRAP_NONE = 0;
static constexpr int P4EST_WRAP_REFINE = 0x01;
static constexpr int P4EST_WRAP_COARSEN = 0x02;

// quadrant routines
static inline qcoord_t
get_x(p4est_quadrant_t * quad)
{
  return quad->x;
}
static inline qcoord_t
get_y(p4est_quadrant_t * quad)
{
  return quad->y;
}
static inline qcoord_t
get_z(p4est_quadrant_t * quad)
{
  (void)(quad);
  return 0;
}

static inline qcoord_t
get_x(p8est_quadrant_t * quad)
{
  return quad->x;
}
static inline qcoord_t
get_y(p8est_quadrant_t * quad)
{
  return quad->y;
}
static inline qcoord_t
get_z(p8est_quadrant_t * quad)
{
  return quad->z;
}

// same for independent nodes
static inline qcoord_t
get_x(p4est_indep_t * quad)
{
  return quad->x;
}
static inline qcoord_t
get_y(p4est_indep_t * quad)
{
  return quad->y;
}
static inline qcoord_t
get_z(p4est_indep_t * quad)
{
  (void)(quad);
  return 0;
}

static inline qcoord_t
get_x(p8est_indep_t * quad)
{
  return quad->x;
}
static inline qcoord_t
get_y(p8est_indep_t * quad)
{
  return quad->y;
}
static inline qcoord_t
get_z(p8est_indep_t * quad)
{
  return quad->z;
}

/**
 * A structure aimed at being specialized for 2D / 3D, and routing
 * to the right p4est/p8est data/routines.
 *
 * Loosely adapted from Deal.II, file p4est_wrappers.
 * http://www.dealii.org/
 *
 * Note for later: c++14 allows the following helper templated alias:
 *
 *   template< bool B, class T, class F >
 *   using conditional_t = typename conditional<B,T,F>::type;
 *
 * For now, just stick to plain c++11.
 *
 */
template <size_t dim>
struct Wrapper
{

  /** Maximum level for representing nodes, 30 in 2D and 3D since mid-2020 */
  // static constexpr int MAXLEVEL = dim == 2 ? P4EST_MAXLEVEL : P8EST_MAXLEVEL;
  static constexpr int MAXLEVEL = P4EST_MAXLEVEL;

  /** Maximum level for representing quadrants, 29 in 2D and 3D since mid-2020
   */
  // static constexpr int QMAXLEVEL = dim == 2 ? P4EST_QMAXLEVEL : P8EST_QMAXLEVEL;
  static constexpr int QMAXLEVEL = P4EST_QMAXLEVEL;

  /** The number of children of a quadrant also the number of corners */
  static constexpr int NB_CHILDREN = dim == 2 ? P4EST_CHILDREN : P8EST_CHILDREN;

  /** The number of children/corners touching one face */
  static constexpr int HALF = dim == 2 ? P4EST_HALF : P8EST_HALF;

  /** The number of faces per cell is 2*dim */
  static constexpr uint32_t NB_FACES = dim == 2 ? P4EST_FACES : P8EST_FACES;

  static constexpr uint32_t NB_NODES_PER_FACE = HALF;

  /** Store the corner numbers 0..3 or 0..7 for each tree face. */
  static const int face_corners[NB_FACES][NB_NODES_PER_FACE];

  /** Store the face numbers in the face neighbor's system. */
  static const int face_dual[2 * dim];

  // type alias
  using connectivity_t =
    typename std::conditional<dim == 2, p4est_connectivity_t, p8est_connectivity_t>::type;
  using geometry_t = typename std::conditional<dim == 2, p4est_geometry_t, p8est_geometry_t>::type;
  using forest_t = typename std::conditional<dim == 2, p4est_t, p8est_t>::type;
  using tree_t = typename std::conditional<dim == 2, p4est_tree_t, p8est_tree_t>::type;
  using quadrant_t = typename std::conditional<dim == 2, p4est_quadrant_t, p8est_quadrant_t>::type;
  using ghost_t = typename std::conditional<dim == 2, p4est_ghost_t, p8est_ghost_t>::type;
  using connect_type_t =
    typename std::conditional<dim == 2, p4est_connect_type_t, p8est_connect_type_t>::type;
  using balance_type_t = connect_type_t;
  using vtk_context_t =
    typename std::conditional<dim == 2, p4est_vtk_context_t, p8est_vtk_context_t>::type;
  using forest_wrap_t = typename std::conditional<dim == 2, p4est_wrap_t, p8est_wrap_t>::type;

  using transfer_context_t =
    typename std::conditional<dim == 2, p4est_transfer_context_t, p8est_transfer_context_t>::type;

  static const connect_type_t CONNECT_SELF;
  static const connect_type_t CONNECT_FACE;
  static const connect_type_t CONNECT_EDGE;
  static const connect_type_t CONNECT_CORNER;
  static const connect_type_t CONNECT_FULL;

  // mesh types
  using mesh_t = typename std::conditional<dim == 2, p4est_mesh_t, p8est_mesh_t>::type;
  using mesh_face_neighbor_t = typename std::
    conditional<dim == 2, p4est_mesh_face_neighbor_t, p8est_mesh_face_neighbor_t>::type;

  // nodes / lnodes types
  using nodes_t = typename std::conditional<dim == 2, p4est_nodes_t, p8est_nodes_t>::type;
  using indep_t = typename std::conditional<dim == 2, p4est_indep_t, p8est_indep_t>::type;
  using lnodes_t = typename std::conditional<dim == 2, p4est_lnodes_t, p8est_lnodes_t>::type;
  using lnodes_code_t =
    typename std::conditional<dim == 2, p4est_lnodes_code_t, p8est_lnodes_code_t>::type;
  using lnodes_rank_t =
    typename std::conditional<dim == 2, p4est_lnodes_rank_t, p8est_lnodes_rank_t>::type;
  using lnodes_buffer_t =
    typename std::conditional<dim == 2, p4est_lnodes_buffer_t, p8est_lnodes_buffer_t>::type;

  // iteration types
  using volume_info_t =
    typename std::conditional<dim == 2, p4est_iter_volume_info_t, p8est_iter_volume_info_t>::type;
  using face_info_t =
    typename std::conditional<dim == 2, p4est_iter_face_info_t, p8est_iter_face_info_t>::type;
  using face_corner_t =
    typename std::conditional<dim == 2, p4est_iter_corner_info_t, p8est_iter_corner_info_t>::type;

  using iter_face_side_t =
    typename std::conditional<dim == 2, p4est_iter_face_side, p8est_iter_face_side_t>::type;

  // iteration callback types
  using volume_cb_t =
    typename std::conditional<dim == 2, p4est_iter_volume_t, p8est_iter_volume_t>::type;
  using face_cb_t = typename std::conditional<dim == 2, p4est_iter_face_t, p8est_iter_face_t>::type;
  using corner_cb_t =
    typename std::conditional<dim == 2, p4est_iter_corner_t, p8est_iter_corner_t>::type;

  // callback type alias
  using init_cb_t = typename std::conditional<dim == 2, p4est_init_t, p8est_init_t>::type;
  using refine_cb_t = typename std::conditional<dim == 2, p4est_refine_t, p8est_refine_t>::type;
  using coarsen_cb_t = typename std::conditional<dim == 2, p4est_coarsen_t, p8est_coarsen_t>::type;
  using weight_cb_t = typename std::conditional<dim == 2, p4est_weight_t, p8est_weight_t>::type;
  using replace_cb_t = typename std::conditional<dim == 2, p4est_replace_t, p8est_replace_t>::type;

  static void
  qcoord_to_vertex(connectivity_t * connectivity, topidx_t treeid, qcoord_t xyz[3], double vxyz[3]);

  static int (&quadrant_is_equal)(const quadrant_t * q1, const quadrant_t * q2);

  static int (&quadrant_is_ancestor)(const quadrant_t * q, const quadrant_t * r);

  static void (&quadrant_child)(const quadrant_t * q, quadrant_t * r, int child_id);

  static int (&quadrant_child_id)(const quadrant_t * q);

  // connectivity / geometry routines
  static connectivity_t * (&connectivity_new_byname)(const char * name);

  // kalypsso's own connectivity by name constructor
  static connectivity_t * (&my_connectivity_new_byname)(const char *      name,
                                                        const ConfigMap & config_map);

  // kalypsso's own geometry by name constructor
  static geometry_t * (&my_geometry_new_byname)(const char *      name,
                                                connectivity_t *  conn,
                                                const ConfigMap & config_map);

  static void (&connectivity_destroy)(connectivity_t * connectivity);

  static int (&connectivity_save)(const char * filename, connectivity_t * connectivity);

  static int (&connectivity_is_valid)(connectivity_t * connectivity);

  static connectivity_t * (&connectivity_load)(const char * filename, size_t * length);

  static connectivity_t * (&connectivity_read_inp)(const char * filename);

  static void (&geometry_connectivity_X)(geometry_t * geom,
                                         topidx_t     which_tree,
                                         const double abc[3],
                                         double       xyz[3]);

  static void (&geometry_destroy)(geometry_t * geometry);

  // forest routines
  static forest_t * (&new_forest)(sc_MPI_Comm      mpicomm,
                                  connectivity_t * connectivity,
                                  locidx_t         min_quadrants,
                                  int              min_level,
                                  int              fill_uniform,
                                  size_t           data_size,
                                  init_cb_t        init_fn,
                                  void *           user_pointer);

  static void (&destroy)(forest_t * forest);

  static forest_t * (&load)(const char *      filename,
                            sc_MPI_Comm       mpicomm,
                            size_t            data_size,
                            int               load_data,
                            void *            user_pointer,
                            connectivity_t ** connectivity);

  static forest_t * (&load_ext)(const char *      filename,
                                sc_MPI_Comm       mpicomm,
                                size_t            data_size,
                                int               load_data,
                                int               autopartition,
                                int               broadcasthead,
                                void *            user_pointer,
                                connectivity_t ** connectivity);

  static void (&refine)(forest_t *  forest,
                        int         refine_recursive,
                        refine_cb_t refine_fn,
                        init_cb_t   init_fn);

  static void (&refine_ext)(forest_t *   forest,
                            int          refine_recursive,
                            int          maxlevel,
                            refine_cb_t  refine_fn,
                            init_cb_t    init_fn,
                            replace_cb_t replace_fn);

  static void (&coarsen)(forest_t *   forest,
                         int          coarsen_recursive,
                         coarsen_cb_t coarsen_fn,
                         init_cb_t    init_fn);

  static void (&coarsen_ext)(forest_t *   p4est,
                             int          coarsen_recursive,
                             int          callback_orphans,
                             coarsen_cb_t coarsen_fn,
                             init_cb_t    init_fn,
                             replace_cb_t replace_fn);

  static void (&balance)(forest_t * forest, connect_type_t btype, init_cb_t init_fn);

  static void (&balance_ext)(forest_t *     forest,
                             connect_type_t btype,
                             init_cb_t      init_fn,
                             replace_cb_t   replace_fn);

  static void (&partition)(forest_t * forest, int allow_for_coarsening, weight_cb_t weight_fn);

  static gloidx_t (&partition_ext)(forest_t *  forest,
                                   int         partition_for_coarsening,
                                   weight_cb_t weight_fn);

  static void
  iterate_volume(forest_t * forest, ghost_t * ghost, void * user_data, volume_cb_t volume_cb);

  static void (&save)(const char * filename, forest_t * forest, int save_data);

  static void (&save_ext)(const char * filename,
                          forest_t *   forest,
                          int          save_data,
                          int          save_partition);

  static unsigned int (&checksum)(forest_t * forest);

  /*
   * VTK related routines.
   */

  static void (&vtk_write_file)(forest_t * forest, geometry_t * geom, const char * baseName);

  static vtk_context_t * (&vtk_context_new)(forest_t * forest, const char * filename);

  static void (&vtk_context_set_geom)(vtk_context_t * cont, geometry_t * geom);

  static void (&vtk_context_set_scale)(vtk_context_t * cont, double scale);

  static void (&vtk_context_set_continuous)(vtk_context_t * cont, int continuous);

  static void (&vtk_context_destroy)(vtk_context_t * context);

  static vtk_context_t * (&vtk_write_header)(vtk_context_t * cont);

  static vtk_context_t * (&vtk_write_cell_dataf)(vtk_context_t * cont,
                                                 int             write_tree,
                                                 int             write_level,
                                                 int             write_rank,
                                                 int             wrap_rank,
                                                 int             num_cell_scalars,
                                                 int             num_cell_vectors,
                                                 ...);

  // static
  // vtk_context_t* (&vtk_write_cell_data) (vtk_context_t * cont,
  //                                        int write_tree,
  //                                        int write_level,
  //                                        int write_rank,
  //                                        int wrap_rank,
  //                                        int num_cell_scalars,
  //                                        int num_cell_vectors,
  //                                        const char *filenames[],
  //                                        sc_array_t * values[]);

  static vtk_context_t * (&vtk_write_point_dataf)(vtk_context_t * cont,
                                                  int             num_point_scalars,
                                                  int             num_point_vectors,
                                                  ...);

  static int (&vtk_write_footer)(vtk_context_t * cont);

  /*
   * ghost related routines.
   */
  static int (&ghost_is_valid)(forest_t * forest, ghost_t * ghost);

  static size_t (&ghost_memory_used)(ghost_t * ghost);

  static ghost_t * (&ghost_new)(forest_t * forest, connect_type_t btype);

  static void (&ghost_destroy)(ghost_t * ghost);

  static void (&ghost_exchange_data)(forest_t * forest, ghost_t * ghost, void * ghost_data);

  static int (&is_balanced)(forest_t * forest, connect_type_t btype);

  static unsigned (&ghost_checksum)(forest_t * forest, ghost_t * ghost);

  /*
   * mesh related routines.
   */
  static mesh_t * (&mesh_new)(forest_t * p4est, ghost_t * ghost, connect_type_t btype);

  static mesh_t * (&mesh_new_ext)(forest_t *     p4est,
                                  ghost_t *      ghost,
                                  int            compute_tree_index,
                                  int            compute_level_lists,
                                  connect_type_t btype);

  static void (&mesh_destroy)(mesh_t * mesh);

  static quadrant_t * (&mesh_quadrant_cumulative)(forest_t * forest,
                                                  mesh_t *   mesh,
                                                  locidx_t   cumulative_id,
                                                  topidx_t * which_tree,
                                                  locidx_t * quadrant_id);

  static void (&mesh_face_neighbor_init2)(mesh_face_neighbor_t * mfn,
                                          forest_t *             forest,
                                          ghost_t *              ghost,
                                          mesh_t *               mesh,
                                          topidx_t               which_tree,
                                          locidx_t               quadrant_id);

  static quadrant_t * (&mesh_face_neighbor_next)(mesh_face_neighbor_t * mfn,
                                                 topidx_t *             ntree,
                                                 locidx_t *             nquad,
                                                 int *                  nface,
                                                 int *                  nrank);

  /*
   * nodes / lnodes related routines
   */
  static nodes_t * (&nodes_new)(forest_t * forest, ghost_t * ghost);

  /**
   * \brief Create a new list of nodes without any ghost data.
   *
   * When there is no ghost data, p4est_nodes_new doesn't compute all the
   * global information about owned nodes and offsets, so we do it here
   * instead.
   *
   * \note this is useful only in HDF5 IO routines.
   */
  static nodes_t *
  nodes_new2(forest_t * forest);

  static void (&nodes_destroy)(nodes_t * nodes);

  static int (&nodes_is_valid)(forest_t * forest, nodes_t * nodes);

  static lnodes_t * (&lnodes_new)(forest_t * forest, ghost_t * ghost_layer, int degree);

  static void (&lnodes_destroy)(lnodes_t * lnodes);

  static lnodes_buffer_t * (&lnodes_share_all)(sc_array_t * node_data, lnodes_t * lnodes);

  static void (&lnodes_buffer_destroy)(lnodes_buffer_t * buffer);

  /*
   * Communication / user data transfer related routines.
   * Wrapper from p4est_communication.h and p8est_communication.h
   */
  static void (&comm_count_quadrants)(forest_t * forest);

  static void (&comm_count_pertree)(forest_t * forest, gloidx_t * pertree);

  static void (&transfer_fixed)(const gloidx_t * dest_gfq,
                                const gloidx_t * src_gfq,
                                sc_MPI_Comm      mpicomm,
                                int              tag,
                                void *           dest_data,
                                const void *     src_data,
                                size_t           data_size);

  static int (&bsearch_partition)(gloidx_t target, const gloidx_t * gfq, int nmemb);

  static transfer_context_t * (&transfer_fixed_begin)(const gloidx_t * dest_gfq,
                                                      const gloidx_t * src_gfq,
                                                      sc_MPI_Comm      mpicomm,
                                                      int              tag,
                                                      void *           dest_data,
                                                      const void *     src_data,
                                                      size_t           data_size);

  static void (&transfer_fixed_end)(transfer_context_t * tc);

  static void (&transfer_custom)(const gloidx_t * dest_gfq,
                                 const gloidx_t * src_gfq,
                                 sc_MPI_Comm      mpicomm,
                                 int              tag,
                                 void *           dest_data,
                                 const int *      dest_sizes,
                                 const void *     src_data,
                                 const int *      src_sizes);

  static transfer_context_t * (&transfer_custom_begin)(const gloidx_t * dest_gfq,
                                                       const gloidx_t * src_gfq,
                                                       sc_MPI_Comm      mpicomm,
                                                       int              tag,
                                                       void *           dest_data,
                                                       const int *      dest_sizes,
                                                       const void *     src_data,
                                                       const int *      src_sizes);

  static void (&transfer_custom_end)(transfer_context_t * tc);

  static void (&transfer_items)(const gloidx_t * dest_gfq,
                                const gloidx_t * src_gfq,
                                sc_MPI_Comm      mpicomm,
                                int              tag,
                                void *           dest_data,
                                const int *      dest_counts,
                                const void *     src_data,
                                const int *      src_counts,
                                size_t           item_size);

  static transfer_context_t * (&transfer_items_begin)(const gloidx_t * dest_gfq,
                                                      const gloidx_t * src_gfq,
                                                      sc_MPI_Comm      mpicomm,
                                                      int              tag,
                                                      void *           dest_data,
                                                      const int *      dest_counts,
                                                      const void *     src_data,
                                                      const int *      src_counts,
                                                      size_t           item_size);

  static void (&transfer_items_end)(transfer_context_t * tc);

  static void (&transfer_end)(transfer_context_t * tc);

  /*
   * array related routines. TODO: check about inlining the following two.
   */
  static tree_t * (&tree_array_index)(sc_array_t * array, topidx_t it);

  static quadrant_t * (&quadrant_array_index)(sc_array_t * array, size_t it);

  static void (&reset_data)(forest_t * forest,
                            size_t     data_size,
                            init_cb_t  init_fn,
                            void *     user_pointer);

  static size_t (&forest_memory_used)(forest_t * forest);

  static size_t (&connectivity_memory_used)(connectivity_t * connectivity);

  static void
  quadrant_reset(quadrant_t * quad)
  {
    if (dim == 2)
    {
      ((void)std::memset((quad), -1, sizeof(quadrant_t)));
    }
    else
    {
      ((void)std::memset((quad), -1, sizeof(quadrant_t)));
    }
  }

  /** The length of a side of the root quadrant */
  static constexpr uint32_t ROOT_LEN = (1 << MAXLEVEL);

  /** The length of a quadrant of level l */
  static inline uint32_t
  QUADRANT_LEN(int level)
  {
    return 1 << (MAXLEVEL - (level));
  }

  /** The offset of the highest (farthest from the origin) quadrant at level l*/
  static inline int32_t
  LAST_OFFSET(int level)
  {
    return ROOT_LEN - QUADRANT_LEN(level);
  }

  /**
   * \brief Compute the length of a quadrant from its level.
   *
   * NOTE: Copied from p4est quadrant_ext.c
   */
  static double
  quadrant_length_level(int level)
  {
    return static_cast<double>(QUADRANT_LEN(level)) / static_cast<double>(ROOT_LEN);
  }

  /** Transform a quadrant coordinate into the space spanned by tree vertices and
   *  taking geometry into account.
   * \param [in] connectivity     Connectivity must provide the vertices.
   * \param [in] geom             Geometry.
   * \param [in] treeid           Identify the tree that contains x, y.
   * \param [in] x, y, z          Quadrant coordinates relative to treeid.
   * \param [out] vxyz            Transformed coordinates in vertex space.
   *
   * Note that z is not used in 2D, however vxyz may be fully populated (see e.g.
   * icosahedron geometry).
   */
  static void
  qcoord_to_vertex_with_geom(geometry_t * geom,
                             topidx_t     treeid,
                             qcoord_t     x,
                             qcoord_t     y,
                             qcoord_t     z,
                             double       vxyz[3])
  {

    const double intsize = 1.0 / ROOT_LEN;
    double       xyz_logic[3] = { intsize * x, intsize * y, intsize * z };

    if (dim == 2)
      xyz_logic[2] = 0;

    // from logical coordinates to physical coordinates
    geom->X(geom, treeid, xyz_logic, vxyz);

  } // qcoord_to_vertex_with_geom

}; // struct Wrapper<size_t dim>

// some template implementation
template <size_t dim>
typename Wrapper<dim>::nodes_t *
Wrapper<dim>::nodes_new2(forest_t * forest)
{
  nodes_t *   nodes = nodes_new(forest, nullptr);
  int         rank = forest->mpirank;
  sc_MPI_Comm mpicomm = forest->mpicomm;

  KALYPSSO_DISABLE_NVCC_WARNINGS_PUSH()
  nodes->global_owned_indeps = P4EST_ALLOC(locidx_t, forest->mpisize);
  KALYPSSO_DISABLE_NVCC_WARNINGS_POP()

  nodes->num_owned_indeps = nodes->global_owned_indeps[rank] =
    nodes->indep_nodes.elem_count + nodes->face_hangings.elem_count;

  if (dim == 3)
    nodes->num_owned_indeps += nodes->edge_hangings.elem_count +

                               MPI_Allgather(&(nodes->num_owned_indeps),
                                             1,
                                             P4EST_MPI_LOCIDX,
                                             nodes->global_owned_indeps,
                                             1,
                                             P4EST_MPI_LOCIDX,
                                             mpicomm);

  nodes->offset_owned_indeps = 0;
  for (int i = 0; i < rank; ++i)
  {
    nodes->offset_owned_indeps += nodes->global_owned_indeps[i];
  }

  return nodes;

} // Wrapper<dim>::nodes_new2


} // namespace p4est

template <size_t dim>
using forest_t = typename p4est::Wrapper<dim>::forest_t;

template <size_t dim>
using tree_t = typename p4est::Wrapper<dim>::tree_t;

template <size_t dim>
using quadrant_t = typename p4est::Wrapper<dim>::quadrant_t;

template <size_t dim>
using ghost_t = typename p4est::Wrapper<dim>::ghost_t;

template <size_t dim>
using geometry_t = typename p4est::Wrapper<dim>::geometry_t;

} // namespace kalypsso

#endif // KALYPSSO_UTILS_P4EST_P4ESTWRAPPER_H_
