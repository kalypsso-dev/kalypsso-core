// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * @file p4est_wrapper.cpp
 * @date 17 octobre 2020
 * @author pkestene
 * @note
 *
 * Add a wrapper unifying 2D / 3D interface to p4est.
 *
 * This file is part of the Kalypsso software project.
 *
 */
#include <kalypsso/utils/p4est/p4est_wrapper.h>
#include <kalypsso/utils/p4est/connectivity.h>
#include <kalypsso/utils/p4est/geometry.h>

#include <cstring>

namespace kalypsso
{

namespace p4est
{

/*****************************************************************
 * Static reference function member initialization for Wrapper<2>.
 *****************************************************************/
template <>
const int Wrapper<2>::face_corners[][NB_NODES_PER_FACE] = { { 0, 2 },
                                                            { 1, 3 },
                                                            { 0, 1 },
                                                            { 2, 3 } };

template <>
const int Wrapper<2>::face_dual[] = { 1, 0, 3, 2 };

template <>
void
Wrapper<2>::qcoord_to_vertex(connectivity_t * connectivity,
                             topidx_t         treeid,
                             qcoord_t         xyz[3],
                             double           vxyz[3])
{
  p4est_qcoord_to_vertex(connectivity, treeid, xyz[0], xyz[1], vxyz);
}

template <>
int (&Wrapper<2>::quadrant_is_equal)(const quadrant_t * q1,
                                     const quadrant_t * q2) = p4est_quadrant_is_equal;

template <>
int (&Wrapper<2>::quadrant_is_ancestor)(const quadrant_t * q,
                                        const quadrant_t * r) = p4est_quadrant_is_ancestor;

template <>
void (&Wrapper<2>::quadrant_child)(const quadrant_t * q,
                                   quadrant_t *       r,
                                   int                child_id) = p4est_quadrant_child;

template <>
int (&Wrapper<2>::quadrant_child_id)(const quadrant_t * q) = p4est_quadrant_child_id;

// connectivity / geometry routines
template <>
Wrapper<2>::connectivity_t * (&Wrapper<2>::connectivity_new_byname)(const char * name) =
  p4est_connectivity_new_byname;

template <>
Wrapper<2>::connectivity_t * (&Wrapper<2>::my_connectivity_new_byname)(
  const char *      name,
  const ConfigMap & config_map) = connectivity_2d_new_byname;

template <>
Wrapper<2>::geometry_t * (&Wrapper<2>::my_geometry_new_byname)(const char *      name,
                                                               connectivity_t *  conn,
                                                               const ConfigMap & config_map) =
  geometry_2d_new_byname;

template <>
void (&Wrapper<2>::connectivity_destroy)(connectivity_t * connectivity) =
  p4est_connectivity_destroy;

template <>
int (&Wrapper<2>::connectivity_save)(const char *     filename,
                                     connectivity_t * connectivity) = p4est_connectivity_save;

template <>
int (&Wrapper<2>::connectivity_is_valid)(connectivity_t * connectivity) =
  p4est_connectivity_is_valid;

template <>
Wrapper<2>::connectivity_t * (
  &Wrapper<2>::connectivity_load)(const char * filename, size_t * length) = p4est_connectivity_load;

template <>
Wrapper<2>::connectivity_t * (&Wrapper<2>::connectivity_read_inp)(const char * filename) =
  p4est_connectivity_read_inp;

template <>
void (&Wrapper<2>::geometry_connectivity_X)(geometry_t * geometry,
                                            topidx_t     which_tree,
                                            const double abc[3],
                                            double       xyz[3]) = p4est_geometry_connectivity_X;
template <>
void (&Wrapper<2>::geometry_destroy)(geometry_t * geometry) = p4est_geometry_destroy;

// forest routines
template <>
Wrapper<2>::forest_t * (&Wrapper<2>::new_forest)(sc_MPI_Comm      mpicomm,
                                                 connectivity_t * connectivity,
                                                 locidx_t         min_quadrants,
                                                 int              min_level,
                                                 int              fill_uniform,
                                                 size_t           data_size,
                                                 init_cb_t        init_fn,
                                                 void *           user_pointer) = p4est_new_ext;

template <>
void (&Wrapper<2>::destroy)(forest_t * forest) = p4est_destroy;

template <>
Wrapper<2>::forest_t * (&Wrapper<2>::load)(const char *      filename,
                                           sc_MPI_Comm       mpicomm,
                                           size_t            data_size,
                                           int               load_data,
                                           void *            user_pointer,
                                           connectivity_t ** connectivity) = p4est_load;

template <>
Wrapper<2>::forest_t * (&Wrapper<2>::load_ext)(const char *      filename,
                                               sc_MPI_Comm       mpicomm,
                                               size_t            data_size,
                                               int               load_data,
                                               int               autopartition,
                                               int               broadcasthead,
                                               void *            user_pointer,
                                               connectivity_t ** connectivity) = p4est_load_ext;

template <>
void (&Wrapper<2>::refine)(forest_t *  forest,
                           int         refine_recursive,
                           refine_cb_t refine_fn,
                           init_cb_t   init_fn) = p4est_refine;

template <>
void (&Wrapper<2>::refine_ext)(forest_t *   forest,
                               int          refine_recursive,
                               int          maxlevel,
                               refine_cb_t  refine_fn,
                               init_cb_t    init_fn,
                               replace_cb_t replace_fn) = p4est_refine_ext;

template <>
void (&Wrapper<2>::coarsen)(forest_t *   forest,
                            int          coarsen_recursive,
                            coarsen_cb_t coarsen_fn,
                            init_cb_t    init_fn) = p4est_coarsen;

template <>
void (&Wrapper<2>::coarsen_ext)(forest_t *   p4est,
                                int          coarsen_recursive,
                                int          callback_orphans,
                                coarsen_cb_t coarsen_fn,
                                init_cb_t    init_fn,
                                replace_cb_t replace_fn) = p4est_coarsen_ext;

template <>
void (&Wrapper<2>::balance)(forest_t *     forest,
                            connect_type_t btype,
                            init_cb_t      init_fn) = p4est_balance;

template <>
void (&Wrapper<2>::balance_ext)(forest_t *     forest,
                                connect_type_t btype,
                                init_cb_t      init_fn,
                                replace_cb_t   replace_fn) = p4est_balance_ext;

template <>
const Wrapper<2>::connect_type_t Wrapper<2>::CONNECT_SELF = P4EST_CONNECT_SELF;
template <>
const Wrapper<2>::connect_type_t Wrapper<2>::CONNECT_FACE = P4EST_CONNECT_FACE;
template <>
const Wrapper<2>::connect_type_t Wrapper<2>::CONNECT_EDGE = P4EST_CONNECT_FACE; // WOULDN'T BE USED
template <>
const Wrapper<2>::connect_type_t Wrapper<2>::CONNECT_CORNER = P4EST_CONNECT_CORNER;
template <>
const Wrapper<2>::connect_type_t Wrapper<2>::CONNECT_FULL = P4EST_CONNECT_FULL;

template <>
void (&Wrapper<2>::partition)(forest_t *  forest,
                              int         allow_for_coarsening,
                              weight_cb_t weight_fn) = p4est_partition;

template <>
gloidx_t (&Wrapper<2>::partition_ext)(forest_t *  forest,
                                      int         partition_for_coarsening,
                                      weight_cb_t weight_fn) = p4est_partition_ext;

template <>
void
Wrapper<2>::iterate_volume(forest_t *  forest,
                           ghost_t *   ghost,
                           void *      user_data,
                           volume_cb_t volume_cb)
{
  p4est_iterate(forest,
                ghost, // ghost layer
                user_data,
                volume_cb,
                nullptr, // iter_face,
                nullptr  // iter_corner
  );
}

template <>
void (&Wrapper<2>::save)(const char * filename, forest_t * forest, int save_data) = p4est_save;

template <>
void (&Wrapper<2>::save_ext)(const char * filename,
                             forest_t *   forest,
                             int          save_data,
                             int          save_partition) = p4est_save_ext;

template <>
unsigned int (&Wrapper<2>::checksum)(forest_t * forest) = p4est_checksum;

/*
 * VTK related routines
 */
template <>
void (&Wrapper<2>::vtk_write_file)(forest_t *   forest,
                                   geometry_t * geometry,
                                   const char * baseName) = p4est_vtk_write_file;

template <>
Wrapper<2>::vtk_context_t * (
  &Wrapper<2>::vtk_context_new)(forest_t * forest, const char * filename) = p4est_vtk_context_new;

template <>
void (&Wrapper<2>::vtk_context_set_geom)(vtk_context_t * cont,
                                         geometry_t *    geom) = p4est_vtk_context_set_geom;

template <>
void (&Wrapper<2>::vtk_context_set_scale)(vtk_context_t * cont,
                                          double          scale) = p4est_vtk_context_set_scale;

template <>
void (&Wrapper<2>::vtk_context_set_continuous)(vtk_context_t * cont,
                                               int continuous) = p4est_vtk_context_set_continuous;

template <>
void (&Wrapper<2>::vtk_context_destroy)(vtk_context_t * context) = p4est_vtk_context_destroy;

template <>
Wrapper<2>::vtk_context_t * (&Wrapper<2>::vtk_write_header)(vtk_context_t * cont) =
  p4est_vtk_write_header;

template <>
Wrapper<2>::vtk_context_t * (&Wrapper<2>::vtk_write_cell_dataf)(vtk_context_t * cont,
                                                                int             write_tree,
                                                                int             write_level,
                                                                int             write_rank,
                                                                int             wrap_rank,
                                                                int             num_cell_scalars,
                                                                int             num_cell_vectors,
                                                                ...) = p4est_vtk_write_cell_dataf;

// template<>
// Wrapper<2>::vtk_context_t* (&Wrapper<2>::vtk_write_cell_data) (vtk_context_t * cont,
//                                                                int write_tree,
//                                                                int write_level,
//                                                                int write_rank,
//                                                                int wrap_rank,
//                                                                int num_cell_scalars,
//                                                                int num_cell_vectors,
//                                                                const char *filenames[],
//                                                                sc_array_t * values[]) =
//   p4est_vtk_write_cell_data;

template <>
Wrapper<2>::vtk_context_t * (&Wrapper<2>::vtk_write_point_dataf)(vtk_context_t * cont,
                                                                 int             num_point_scalars,
                                                                 int             num_point_vectors,
                                                                 ...) = p4est_vtk_write_point_dataf;

template <>
int (&Wrapper<2>::vtk_write_footer)(vtk_context_t * cont) = p4est_vtk_write_footer;

/*
 * ghost related routine.
 */
template <>
int (&Wrapper<2>::ghost_is_valid)(forest_t * forest, ghost_t * ghost) = p4est_ghost_is_valid;

template <>
size_t (&Wrapper<2>::ghost_memory_used)(ghost_t * ghost) = p4est_ghost_memory_used;
template <>
Wrapper<2>::ghost_t * (&Wrapper<2>::ghost_new)(forest_t *     forest,
                                               connect_type_t btype) = p4est_ghost_new;

template <>
void (&Wrapper<2>::ghost_destroy)(ghost_t * ghost) = p4est_ghost_destroy;

template <>
void (&Wrapper<2>::ghost_exchange_data)(forest_t * forest,
                                        ghost_t *  ghost,
                                        void *     ghost_data) = p4est_ghost_exchange_data;

template <>
int (&Wrapper<2>::is_balanced)(forest_t * forest, connect_type_t btype) = p4est_is_balanced;

template <>
unsigned (&Wrapper<2>::ghost_checksum)(forest_t * forest, ghost_t * ghost) = p4est_ghost_checksum;

/*
 * mesh related routines.
 */
template <>
Wrapper<2>::mesh_t * (&Wrapper<2>::mesh_new)(forest_t *     p4est,
                                             ghost_t *      ghost,
                                             connect_type_t btype) = p4est_mesh_new;

template <>
Wrapper<2>::mesh_t * (&Wrapper<2>::mesh_new_ext)(forest_t *     p4est,
                                                 ghost_t *      ghost,
                                                 int            compute_tree_index,
                                                 int            compute_level_lists,
                                                 connect_type_t btype) = p4est_mesh_new_ext;

template <>
void (&Wrapper<2>::mesh_destroy)(mesh_t * mesh) = p4est_mesh_destroy;

template <>
Wrapper<2>::quadrant_t * (&Wrapper<2>::mesh_quadrant_cumulative)(forest_t * forest,
                                                                 mesh_t *   mesh,
                                                                 locidx_t   cumulative_id,
                                                                 topidx_t * which_tree,
                                                                 locidx_t * quadrant_id) =
  p4est_mesh_quadrant_cumulative;

template <>
void (&Wrapper<2>::mesh_face_neighbor_init2)(mesh_face_neighbor_t * mfn,
                                             forest_t *             forest,
                                             ghost_t *              ghost,
                                             mesh_t *               mesh,
                                             topidx_t               which_tree,
                                             locidx_t quadrant_id) = p4est_mesh_face_neighbor_init2;

template <>
Wrapper<2>::quadrant_t * (&Wrapper<2>::mesh_face_neighbor_next)(mesh_face_neighbor_t * mfn,
                                                                topidx_t *             ntree,
                                                                locidx_t *             nquad,
                                                                int *                  nface,
                                                                int *                  nrank) =
  p4est_mesh_face_neighbor_next;


/*
 * nodes / lnodes related routines.
 */
template <>
Wrapper<2>::nodes_t * (&Wrapper<2>::nodes_new)(forest_t * forest,
                                               ghost_t *  ghost) = p4est_nodes_new;

template <>
void (&Wrapper<2>::nodes_destroy)(nodes_t * nodes) = p4est_nodes_destroy;

template <>
int (&Wrapper<2>::nodes_is_valid)(forest_t * forest, nodes_t * nodes) = p4est_nodes_is_valid;

template <>
Wrapper<2>::lnodes_t * (&Wrapper<2>::lnodes_new)(forest_t * forest,
                                                 ghost_t *  ghost,
                                                 int        degree) = p4est_lnodes_new;

template <>
void (&Wrapper<2>::lnodes_destroy)(lnodes_t * lnodes) = p4est_lnodes_destroy;

template <>
Wrapper<2>::lnodes_buffer_t * (&Wrapper<2>::lnodes_share_all)(sc_array_t * node_data,
                                                              lnodes_t *   lnodes) =
  p4est_lnodes_share_all;

template <>
void (&Wrapper<2>::lnodes_buffer_destroy)(lnodes_buffer_t * buffer) = p4est_lnodes_buffer_destroy;

/*
 * Communication / user data transfer related routines.
 */
template <>
void (&Wrapper<2>::comm_count_quadrants)(forest_t * forest) = p4est_comm_count_quadrants;

template <>
void (&Wrapper<2>::comm_count_pertree)(forest_t * forest,
                                       gloidx_t * pertree) = p4est_comm_count_pertree;

template <>
void (&Wrapper<2>::transfer_fixed)(const gloidx_t * dest_gfq,
                                   const gloidx_t * src_gfq,
                                   sc_MPI_Comm      mpicomm,
                                   int              tag,
                                   void *           dest_data,
                                   const void *     src_data,
                                   size_t           data_size) = p4est_transfer_fixed;

template <>
int (&Wrapper<2>::bsearch_partition)(gloidx_t         target,
                                     const gloidx_t * gfq,
                                     int              nmemb) = p4est_bsearch_partition;

template <>
Wrapper<2>::transfer_context_t * (&Wrapper<2>::transfer_fixed_begin)(const gloidx_t * dest_gfq,
                                                                     const gloidx_t * src_gfq,
                                                                     sc_MPI_Comm      mpicomm,
                                                                     int              tag,
                                                                     void *           dest_data,
                                                                     const void *     src_data,
                                                                     size_t           data_size) =
  p4est_transfer_fixed_begin;

template <>
void (&Wrapper<2>::transfer_fixed_end)(transfer_context_t * tc) = p4est_transfer_fixed_end;

template <>
void (&Wrapper<2>::transfer_custom)(const gloidx_t * dest_gfq,
                                    const gloidx_t * src_gfq,
                                    sc_MPI_Comm      mpicomm,
                                    int              tag,
                                    void *           dest_data,
                                    const int *      dest_sizes,
                                    const void *     src_data,
                                    const int *      src_sizes) = p4est_transfer_custom;

template <>
Wrapper<2>::transfer_context_t * (&Wrapper<2>::transfer_custom_begin)(const gloidx_t * dest_gfq,
                                                                      const gloidx_t * src_gfq,
                                                                      sc_MPI_Comm      mpicomm,
                                                                      int              tag,
                                                                      void *           dest_data,
                                                                      const int *      dest_sizes,
                                                                      const void *     src_data,
                                                                      const int *      src_sizes) =
  p4est_transfer_custom_begin;

template <>
void (&Wrapper<2>::transfer_custom_end)(transfer_context_t * tc) = p4est_transfer_custom_end;

template <>
void (&Wrapper<2>::transfer_items)(const gloidx_t * dest_gfq,
                                   const gloidx_t * src_gfq,
                                   sc_MPI_Comm      mpicomm,
                                   int              tag,
                                   void *           dest_data,
                                   const int *      dest_counts,
                                   const void *     src_data,
                                   const int *      src_counts,
                                   size_t           item_size) = p4est_transfer_items;

template <>
Wrapper<2>::transfer_context_t * (&Wrapper<2>::transfer_items_begin)(const gloidx_t * dest_gfq,
                                                                     const gloidx_t * src_gfq,
                                                                     sc_MPI_Comm      mpicomm,
                                                                     int              tag,
                                                                     void *           dest_data,
                                                                     const int *      dest_counts,
                                                                     const void *     src_data,
                                                                     const int *      src_counts,
                                                                     size_t           item_size) =
  p4est_transfer_items_begin;

template <>
void (&Wrapper<2>::transfer_items_end)(transfer_context_t * tc) = p4est_transfer_items_end;

template <>
void (&Wrapper<2>::transfer_end)(transfer_context_t * tc) = p4est_transfer_end;


/*
 * array related routines.
 */
template <>
Wrapper<2>::tree_t * (&Wrapper<2>::tree_array_index)(sc_array_t * array,
                                                     topidx_t     it) = p4est_tree_array_index;

template <>
Wrapper<2>::quadrant_t * (&Wrapper<2>::quadrant_array_index)(sc_array_t * array, size_t it) =
  p4est_quadrant_array_index;

template <>
void (&Wrapper<2>::reset_data)(forest_t * forest,
                               size_t     data_size,
                               init_cb_t  init_fn,
                               void *     user_pointer) = p4est_reset_data;

template <>
size_t (&Wrapper<2>::forest_memory_used)(forest_t * forest) = p4est_memory_used;

template <>
size_t (&Wrapper<2>::connectivity_memory_used)(connectivity_t * conn) =
  p4est_connectivity_memory_used;


/*********************************************************************
 * Static reference function member initialization for Wrapper<3>.
 *********************************************************************/
template <>
const int Wrapper<3>::face_corners[][NB_NODES_PER_FACE] = { { 0, 2, 4, 6 }, { 1, 3, 5, 7 },
                                                            { 0, 1, 4, 5 }, { 2, 3, 6, 7 },
                                                            { 0, 1, 2, 3 }, { 4, 5, 6, 7 } };

template <>
const int Wrapper<3>::face_dual[] = { 1, 0, 3, 2, 5, 4 };

template <>
void
Wrapper<3>::qcoord_to_vertex(connectivity_t * connectivity,
                             topidx_t         treeid,
                             qcoord_t         xyz[3],
                             double           vxyz[3])
{
  p8est_qcoord_to_vertex(connectivity, treeid, xyz[0], xyz[1], xyz[2], vxyz);
}

template <>
int (&Wrapper<3>::quadrant_is_equal)(const quadrant_t * q1,
                                     const quadrant_t * q2) = p8est_quadrant_is_equal;

template <>
int (&Wrapper<3>::quadrant_is_ancestor)(const quadrant_t * q,
                                        const quadrant_t * r) = p8est_quadrant_is_ancestor;

template <>
void (&Wrapper<3>::quadrant_child)(const quadrant_t * q,
                                   quadrant_t *       r,
                                   int                child_id) = p8est_quadrant_child;

template <>
int (&Wrapper<3>::quadrant_child_id)(const quadrant_t * q) = p8est_quadrant_child_id;

// connectivity / geometry routines
template <>
Wrapper<3>::connectivity_t * (&Wrapper<3>::connectivity_new_byname)(const char * name) =
  p8est_connectivity_new_byname;

template <>
Wrapper<3>::connectivity_t * (&Wrapper<3>::my_connectivity_new_byname)(
  const char *      name,
  const ConfigMap & config_map) = connectivity_3d_new_byname;

template <>
Wrapper<3>::geometry_t * (&Wrapper<3>::my_geometry_new_byname)(const char *      name,
                                                               connectivity_t *  conn,
                                                               const ConfigMap & config_map) =
  geometry_3d_new_byname;

template <>
void (&Wrapper<3>::connectivity_destroy)(connectivity_t * connectivity) =
  p8est_connectivity_destroy;

template <>
int (&Wrapper<3>::connectivity_save)(const char *     filename,
                                     connectivity_t * connectivity) = p8est_connectivity_save;

template <>
int (&Wrapper<3>::connectivity_is_valid)(connectivity_t * connectivity) =
  p8est_connectivity_is_valid;

template <>
Wrapper<3>::connectivity_t * (
  &Wrapper<3>::connectivity_load)(const char * filename, size_t * length) = p8est_connectivity_load;

template <>
Wrapper<3>::connectivity_t * (&Wrapper<3>::connectivity_read_inp)(const char * filename) =
  p8est_connectivity_read_inp;

template <>
void (&Wrapper<3>::geometry_connectivity_X)(geometry_t * geometry,
                                            topidx_t     which_tree,
                                            const double abc[3],
                                            double       xyz[3]) = p8est_geometry_connectivity_X;

template <>
void (&Wrapper<3>::geometry_destroy)(geometry_t * geometry) = p8est_geometry_destroy;

// forest routines
template <>
Wrapper<3>::forest_t * (&Wrapper<3>::new_forest)(sc_MPI_Comm      mpicomm,
                                                 connectivity_t * connectivity,
                                                 locidx_t         min_quadrants,
                                                 int              min_level,
                                                 int              fill_uniform,
                                                 size_t           data_size,
                                                 init_cb_t        init_fn,
                                                 void *           user_pointer) = p8est_new_ext;

template <>
void (&Wrapper<3>::destroy)(forest_t * forest) = p8est_destroy;

template <>
Wrapper<3>::forest_t * (&Wrapper<3>::load)(const char *      filename,
                                           sc_MPI_Comm       mpicomm,
                                           size_t            data_size,
                                           int               load_data,
                                           void *            user_pointer,
                                           connectivity_t ** connectivity) = p8est_load;

template <>
Wrapper<3>::forest_t * (&Wrapper<3>::load_ext)(const char *      filename,
                                               sc_MPI_Comm       mpicomm,
                                               size_t            data_size,
                                               int               load_data,
                                               int               autopartition,
                                               int               broadcasthead,
                                               void *            user_pointer,
                                               connectivity_t ** connectivity) = p8est_load_ext;

template <>
void (&Wrapper<3>::refine)(forest_t *  forest,
                           int         refine_recursive,
                           refine_cb_t refine_fn,
                           init_cb_t   init_fn) = p8est_refine;

template <>
void (&Wrapper<3>::refine_ext)(forest_t *   forest,
                               int          refine_recursive,
                               int          maxlevel,
                               refine_cb_t  refine_fn,
                               init_cb_t    init_fn,
                               replace_cb_t replace_fn) = p8est_refine_ext;


template <>
void (&Wrapper<3>::coarsen)(forest_t *   forest,
                            int          coarsen_recursive,
                            coarsen_cb_t coarsen_fn,
                            init_cb_t    init_fn) = p8est_coarsen;

template <>
void (&Wrapper<3>::coarsen_ext)(forest_t *   p4est,
                                int          coarsen_recursive,
                                int          callback_orphans,
                                coarsen_cb_t coarsen_fn,
                                init_cb_t    init_fn,
                                replace_cb_t replace_fn) = p8est_coarsen_ext;

template <>
void (&Wrapper<3>::balance)(forest_t *     forest,
                            connect_type_t btype,
                            init_cb_t      init_fn) = p8est_balance;

template <>
void (&Wrapper<3>::balance_ext)(forest_t *     forest,
                                connect_type_t btype,
                                init_cb_t      init_fn,
                                replace_cb_t   replace_fn) = p8est_balance_ext;

template <>
const Wrapper<3>::connect_type_t Wrapper<3>::CONNECT_SELF = P8EST_CONNECT_SELF;
template <>
const Wrapper<3>::connect_type_t Wrapper<3>::CONNECT_FACE = P8EST_CONNECT_FACE;
template <>
const Wrapper<3>::connect_type_t Wrapper<3>::CONNECT_EDGE = P8EST_CONNECT_EDGE;
template <>
const Wrapper<3>::connect_type_t Wrapper<3>::CONNECT_CORNER = P8EST_CONNECT_CORNER;
template <>
const Wrapper<3>::connect_type_t Wrapper<3>::CONNECT_FULL = P8EST_CONNECT_FULL;

template <>
void (&Wrapper<3>::partition)(forest_t *  forest,
                              int         allow_for_coarsening,
                              weight_cb_t weight_fn) = p8est_partition;

template <>
gloidx_t (&Wrapper<3>::partition_ext)(forest_t *  forest,
                                      int         partition_for_coarsening,
                                      weight_cb_t weight_fn) = p8est_partition_ext;

template <>
void
Wrapper<3>::iterate_volume(forest_t *  forest,
                           ghost_t *   ghost,
                           void *      user_data,
                           volume_cb_t volume_cb)
{
  p8est_iterate(forest,
                ghost, // ghost layer
                user_data,
                volume_cb,
                nullptr, // iter_face
                nullptr, // iter_edge
                nullptr  // iter_corner
  );
}

template <>
void (&Wrapper<3>::save)(const char * filename, forest_t * forest, int save_data) = p8est_save;

template <>
void (&Wrapper<3>::save_ext)(const char * filename,
                             forest_t *   forest,
                             int          save_data,
                             int          save_partition) = p8est_save_ext;


template <>
unsigned int (&Wrapper<3>::checksum)(forest_t * forest) = p8est_checksum;

/*
 * VTK related routines
 */
template <>
void (&Wrapper<3>::vtk_write_file)(forest_t *   forest,
                                   geometry_t * geometry,
                                   const char * baseName) = p8est_vtk_write_file;

template <>
Wrapper<3>::vtk_context_t * (
  &Wrapper<3>::vtk_context_new)(forest_t * forest, const char * filename) = p8est_vtk_context_new;

template <>
void (&Wrapper<3>::vtk_context_set_geom)(vtk_context_t * cont,
                                         geometry_t *    geom) = p8est_vtk_context_set_geom;

template <>
void (&Wrapper<3>::vtk_context_set_scale)(vtk_context_t * cont,
                                          double          scale) = p8est_vtk_context_set_scale;

template <>
void (&Wrapper<3>::vtk_context_set_continuous)(vtk_context_t * cont,
                                               int continuous) = p8est_vtk_context_set_continuous;

template <>
void (&Wrapper<3>::vtk_context_destroy)(vtk_context_t * context) = p8est_vtk_context_destroy;

template <>
Wrapper<3>::vtk_context_t * (&Wrapper<3>::vtk_write_header)(vtk_context_t * cont) =
  p8est_vtk_write_header;

template <>
Wrapper<3>::vtk_context_t * (&Wrapper<3>::vtk_write_cell_dataf)(vtk_context_t * cont,
                                                                int             write_tree,
                                                                int             write_level,
                                                                int             write_rank,
                                                                int             wrap_rank,
                                                                int             num_cell_scalars,
                                                                int             num_cell_vectors,
                                                                ...) = p8est_vtk_write_cell_dataf;

// template<>
// Wrapper<3>::vtk_context_t* (&Wrapper<3>::vtk_write_cell_data) (vtk_context_t * cont,
//                                                                int write_tree,
//                                                                int write_level,
//                                                                int write_rank,
//                                                                int wrap_rank,
//                                                                int num_cell_scalars,
//                                                                int num_cell_vectors,
//                                                                const char *filenames[],
//                                                                sc_array_t * values[]) =
//                                                                p8est_vtk_write_cell_data;

template <>
Wrapper<3>::vtk_context_t * (&Wrapper<3>::vtk_write_point_dataf)(vtk_context_t * cont,
                                                                 int             num_point_scalars,
                                                                 int             num_point_vectors,
                                                                 ...) = p8est_vtk_write_point_dataf;

template <>
int (&Wrapper<3>::vtk_write_footer)(vtk_context_t * cont) = p8est_vtk_write_footer;

/*
 * ghost related routines.
 */
template <>
int (&Wrapper<3>::ghost_is_valid)(forest_t * forest, ghost_t * ghost) = p8est_ghost_is_valid;

template <>
size_t (&Wrapper<3>::ghost_memory_used)(ghost_t * ghost) = p8est_ghost_memory_used;

template <>
Wrapper<3>::ghost_t * (&Wrapper<3>::ghost_new)(forest_t *     forest,
                                               connect_type_t btype) = p8est_ghost_new;

template <>
void (&Wrapper<3>::ghost_destroy)(ghost_t * ghost) = p8est_ghost_destroy;

template <>
void (&Wrapper<3>::ghost_exchange_data)(forest_t * forest,
                                        ghost_t *  ghost,
                                        void *     ghost_data) = p8est_ghost_exchange_data;

template <>
int (&Wrapper<3>::is_balanced)(forest_t * forest, connect_type_t btype) = p8est_is_balanced;

template <>
unsigned (&Wrapper<3>::ghost_checksum)(forest_t * forest, ghost_t * ghost) = p8est_ghost_checksum;


/*
 * mesh related routines.
 */
template <>
Wrapper<3>::mesh_t * (&Wrapper<3>::mesh_new)(forest_t *     p4est,
                                             ghost_t *      ghost,
                                             connect_type_t btype) = p8est_mesh_new;

template <>
Wrapper<3>::mesh_t * (&Wrapper<3>::mesh_new_ext)(forest_t *     p4est,
                                                 ghost_t *      ghost,
                                                 int            compute_tree_index,
                                                 int            compute_level_lists,
                                                 connect_type_t btype) = p8est_mesh_new_ext;

template <>
void (&Wrapper<3>::mesh_destroy)(mesh_t * mesh) = p8est_mesh_destroy;

template <>
Wrapper<3>::quadrant_t * (&Wrapper<3>::mesh_quadrant_cumulative)(forest_t * forest,
                                                                 mesh_t *   mesh,
                                                                 locidx_t   cumulative_id,
                                                                 topidx_t * which_tree,
                                                                 locidx_t * quadrant_id) =
  p8est_mesh_quadrant_cumulative;

template <>
void (&Wrapper<3>::mesh_face_neighbor_init2)(mesh_face_neighbor_t * mfn,
                                             forest_t *             forest,
                                             ghost_t *              ghost,
                                             mesh_t *               mesh,
                                             topidx_t               which_tree,
                                             locidx_t quadrant_id) = p8est_mesh_face_neighbor_init2;

template <>
Wrapper<3>::quadrant_t * (&Wrapper<3>::mesh_face_neighbor_next)(mesh_face_neighbor_t * mfn,
                                                                topidx_t *             ntree,
                                                                locidx_t *             nquad,
                                                                int *                  nface,
                                                                int *                  nrank) =
  p8est_mesh_face_neighbor_next;

/*
 * nodes / lnodes related routines.
 */
template <>
Wrapper<3>::nodes_t * (&Wrapper<3>::nodes_new)(forest_t * forest,
                                               ghost_t *  ghost) = p8est_nodes_new;

template <>
void (&Wrapper<3>::nodes_destroy)(nodes_t * nodes) = p8est_nodes_destroy;

template <>
int (&Wrapper<3>::nodes_is_valid)(forest_t * forest, nodes_t * nodes) = p8est_nodes_is_valid;

template <>
Wrapper<3>::lnodes_t * (&Wrapper<3>::lnodes_new)(forest_t * forest,
                                                 ghost_t *  ghost,
                                                 int        degree) = p8est_lnodes_new;

template <>
void (&Wrapper<3>::lnodes_destroy)(lnodes_t * lnodes) = p8est_lnodes_destroy;

template <>
Wrapper<3>::lnodes_buffer_t * (&Wrapper<3>::lnodes_share_all)(sc_array_t * node_data,
                                                              lnodes_t *   lnodes) =
  p8est_lnodes_share_all;

template <>
void (&Wrapper<3>::lnodes_buffer_destroy)(lnodes_buffer_t * buffer) = p8est_lnodes_buffer_destroy;

/*
 * Communication / user data transfer related routines.
 */
template <>
void (&Wrapper<3>::comm_count_quadrants)(forest_t * forest) = p8est_comm_count_quadrants;

template <>
void (&Wrapper<3>::comm_count_pertree)(forest_t * forest,
                                       gloidx_t * pertree) = p8est_comm_count_pertree;

template <>
void (&Wrapper<3>::transfer_fixed)(const gloidx_t * dest_gfq,
                                   const gloidx_t * src_gfq,
                                   sc_MPI_Comm      mpicomm,
                                   int              tag,
                                   void *           dest_data,
                                   const void *     src_data,
                                   size_t           data_size) = p8est_transfer_fixed;

template <>
int (&Wrapper<3>::bsearch_partition)(gloidx_t         target,
                                     const gloidx_t * gfq,
                                     int              nmemb) = p8est_bsearch_partition;

template <>
Wrapper<3>::transfer_context_t * (&Wrapper<3>::transfer_fixed_begin)(const gloidx_t * dest_gfq,
                                                                     const gloidx_t * src_gfq,
                                                                     sc_MPI_Comm      mpicomm,
                                                                     int              tag,
                                                                     void *           dest_data,
                                                                     const void *     src_data,
                                                                     size_t           data_size) =
  p8est_transfer_fixed_begin;

template <>
void (&Wrapper<3>::transfer_fixed_end)(transfer_context_t * tc) = p8est_transfer_fixed_end;

template <>
void (&Wrapper<3>::transfer_custom)(const gloidx_t * dest_gfq,
                                    const gloidx_t * src_gfq,
                                    sc_MPI_Comm      mpicomm,
                                    int              tag,
                                    void *           dest_data,
                                    const int *      dest_sizes,
                                    const void *     src_data,
                                    const int *      src_sizes) = p8est_transfer_custom;

template <>
Wrapper<3>::transfer_context_t * (&Wrapper<3>::transfer_custom_begin)(const gloidx_t * dest_gfq,
                                                                      const gloidx_t * src_gfq,
                                                                      sc_MPI_Comm      mpicomm,
                                                                      int              tag,
                                                                      void *           dest_data,
                                                                      const int *      dest_sizes,
                                                                      const void *     src_data,
                                                                      const int *      src_sizes) =
  p8est_transfer_custom_begin;

template <>
void (&Wrapper<3>::transfer_custom_end)(transfer_context_t * tc) = p8est_transfer_custom_end;

template <>
void (&Wrapper<3>::transfer_items)(const gloidx_t * dest_gfq,
                                   const gloidx_t * src_gfq,
                                   sc_MPI_Comm      mpicomm,
                                   int              tag,
                                   void *           dest_data,
                                   const int *      dest_counts,
                                   const void *     src_data,
                                   const int *      src_counts,
                                   size_t           item_size) = p8est_transfer_items;

template <>
Wrapper<3>::transfer_context_t * (&Wrapper<3>::transfer_items_begin)(const gloidx_t * dest_gfq,
                                                                     const gloidx_t * src_gfq,
                                                                     sc_MPI_Comm      mpicomm,
                                                                     int              tag,
                                                                     void *           dest_data,
                                                                     const int *      dest_counts,
                                                                     const void *     src_data,
                                                                     const int *      src_counts,
                                                                     size_t           item_size) =
  p8est_transfer_items_begin;

template <>
void (&Wrapper<3>::transfer_items_end)(transfer_context_t * tc) = p8est_transfer_items_end;

template <>
void (&Wrapper<3>::transfer_end)(transfer_context_t * tc) = p8est_transfer_end;


/*
 * array related routines.
 */
template <>
Wrapper<3>::tree_t * (&Wrapper<3>::tree_array_index)(sc_array_t * array,
                                                     topidx_t     it) = p8est_tree_array_index;

template <>
Wrapper<3>::quadrant_t * (&Wrapper<3>::quadrant_array_index)(sc_array_t * array, size_t it) =
  p8est_quadrant_array_index;

template <>
void (&Wrapper<3>::reset_data)(forest_t * forest,
                               size_t     data_size,
                               init_cb_t  init_fn,
                               void *     user_pointer) = p8est_reset_data;

template <>
size_t (&Wrapper<3>::forest_memory_used)(forest_t * forest) = p8est_memory_used;

template <>
size_t (&Wrapper<3>::connectivity_memory_used)(connectivity_t * conn) =
  p8est_connectivity_memory_used;

} // namespace p4est

} // namespace kalypsso
