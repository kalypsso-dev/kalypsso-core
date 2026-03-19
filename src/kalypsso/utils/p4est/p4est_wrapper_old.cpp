// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "p4est_wrapper_old.h"

#include <cstring>

namespace kalypsso
{

namespace p4est
{

/*
 * Static reference function member initialization for Wrapper<2>.
 */
int (&Wrapper<2>::quadrant_is_equal)(const quadrant_t * q1,
                                     const quadrant_t * q2) = p4est_quadrant_is_equal;

int (&Wrapper<2>::quadrant_is_ancestor)(const quadrant_t * q,
                                        const quadrant_t * r) = p4est_quadrant_is_ancestor;

void (&Wrapper<2>::quadrant_child)(const quadrant_t * q,
                                   quadrant_t *       r,
                                   int                child_id) = p4est_quadrant_child;

int (&Wrapper<2>::quadrant_child_id)(const quadrant_t * q) = p4est_quadrant_child_id;

void (&Wrapper<2>::connectivity_destroy)(connectivity_t * connectivity) =
  p4est_connectivity_destroy;

int (&Wrapper<2>::connectivity_save)(const char *                 filename,
                                     Wrapper<2>::connectivity_t * connectivity) =
  p4est_connectivity_save;

int (&Wrapper<2>::connectivity_is_valid)(Wrapper<2>::connectivity_t * connectivity) =
  p4est_connectivity_is_valid;

Wrapper<2>::connectivity_t * (
  &Wrapper<2>::connectivity_load)(const char * filename, size_t * length) = p4est_connectivity_load;

void (&Wrapper<2>::geometry_destroy)(geometry_t * geometry) = p4est_geometry_destroy;

Wrapper<2>::forest_t * (&Wrapper<2>::new_forest)(MPI_Comm         mpicomm,
                                                 connectivity_t * connectivity,
                                                 locidx_t         min_quadrants,
                                                 int              min_level,
                                                 int              fill_uniform,
                                                 size_t           data_size,
                                                 init_cb_t        init_fn,
                                                 void *           user_pointer) = p4est_new_ext;

void (&Wrapper<2>::destroy)(Wrapper<2>::forest_t * forest) = p4est_destroy;

void (&Wrapper<2>::refine)(Wrapper<2>::forest_t * forest,
                           int                    refine_recursive,
                           refine_cb_t            refine_fn,
                           init_cb_t              init_fn) = p4est_refine;

void (&Wrapper<2>::coarsen)(Wrapper<2>::forest_t * forest,
                            int                    coarsen_recursive,
                            coarsen_cb_t           coarsen_fn,
                            init_cb_t              init_fn) = p4est_coarsen;

void (&Wrapper<2>::balance)(Wrapper<2>::forest_t *     forest,
                            Wrapper<2>::balance_type_t btype,
                            init_cb_t                  init_fn) = p4est_balance;


Wrapper<2>::gloidx_t (&Wrapper<2>::partition)(Wrapper<2>::forest_t * forest,
                                              int                    partition_for_coarsening,
                                              weight_cb_t weight_fn) = p4est_partition_ext;

void
Wrapper<2>::iterate_volume(forest_t * forest, void * user_data, volume_cb_t volume_cb)
{

  p4est_iterate(forest,
                nullptr, // ghost layer
                user_data,
                volume_cb,
                nullptr, // iter_face,
                nullptr  // iter_corner
  );

} // Wrapper<2>::iterate_volume

void (&Wrapper<2>::save)(const char *           filename,
                         Wrapper<2>::forest_t * forest,
                         int                    save_data) = p4est_save;

Wrapper<2>::forest_t * (&Wrapper<2>::load_ext)(const char *                  filename,
                                               MPI_Comm                      mpicomm,
                                               std::size_t                   data_size,
                                               int                           load_data,
                                               int                           autopartition,
                                               int                           broadcasthead,
                                               void *                        user_pointer,
                                               Wrapper<2>::connectivity_t ** connectivity) =
  p4est_load_ext;

unsigned int (&Wrapper<2>::checksum)(Wrapper<2>::forest_t * forest) = p4est_checksum;

void (&Wrapper<2>::vtk_write_file)(Wrapper<2>::forest_t *   forest,
                                   Wrapper<2>::geometry_t * geometry,
                                   const char *             baseName) = p4est_vtk_write_file;

Wrapper<2>::vtk_context_t * (
  &Wrapper<2>::vtk_context_new)(forest_t * forest, const char * filename) = p4est_vtk_context_new;

void (&Wrapper<2>::vtk_context_set_geom)(vtk_context_t * cont,
                                         geometry_t *    geom) = p4est_vtk_context_set_geom;

void (&Wrapper<2>::vtk_context_set_scale)(vtk_context_t * cont,
                                          double          scale) = p4est_vtk_context_set_scale;

void (&Wrapper<2>::vtk_context_set_continuous)(vtk_context_t * cont,
                                               int continuous) = p4est_vtk_context_set_continuous;

void (&Wrapper<2>::vtk_context_destroy)(vtk_context_t * context) = p4est_vtk_context_destroy;

Wrapper<2>::vtk_context_t * (&Wrapper<2>::vtk_write_header)(vtk_context_t * cont) =
  p4est_vtk_write_header;

Wrapper<2>::vtk_context_t * (&Wrapper<2>::vtk_write_cell_dataf)(vtk_context_t * cont,
                                                                int             write_tree,
                                                                int             write_level,
                                                                int             write_rank,
                                                                int             wrap_rank,
                                                                int             num_cell_scalars,
                                                                int             num_cell_vectors,
                                                                ...) = p4est_vtk_write_cell_dataf;

Wrapper<2>::vtk_context_t * (&Wrapper<2>::vtk_write_point_dataf)(vtk_context_t * cont,
                                                                 int             num_point_scalars,
                                                                 int             num_point_vectors,
                                                                 ...) = p4est_vtk_write_point_dataf;

int (&Wrapper<2>::vtk_write_footer)(vtk_context_t * cont) = p4est_vtk_write_footer;

Wrapper<2>::ghost_t * (&Wrapper<2>::ghost_new)(Wrapper<2>::forest_t *     forest,
                                               Wrapper<2>::balance_type_t btype) = p4est_ghost_new;

void (&Wrapper<2>::ghost_destroy)(Wrapper<2>::ghost_t * ghost) = p4est_ghost_destroy;

void (&Wrapper<2>::reset_data)(Wrapper<2>::forest_t * forest,
                               size_t                 data_size,
                               init_cb_t              init_fn,
                               void *                 user_pointer) = p4est_reset_data;

size_t (&Wrapper<2>::forest_memory_used)(Wrapper<2>::forest_t * forest) = p4est_memory_used;

size_t (&Wrapper<2>::connectivity_memory_used)(Wrapper<2>::connectivity_t * conn) =
  p4est_connectivity_memory_used;


/*
 * Static reference function member initialization for Wrapper<3>.
 */
int (&Wrapper<3>::quadrant_is_equal)(const quadrant_t * q1,
                                     const quadrant_t * q2) = p8est_quadrant_is_equal;

int (&Wrapper<3>::quadrant_is_ancestor)(const quadrant_t * q,
                                        const quadrant_t * r) = p8est_quadrant_is_ancestor;

void (&Wrapper<3>::quadrant_child)(const quadrant_t * q,
                                   quadrant_t *       r,
                                   int                child_id) = p8est_quadrant_child;

int (&Wrapper<3>::quadrant_child_id)(const quadrant_t * q) = p8est_quadrant_child_id;

void (&Wrapper<3>::connectivity_destroy)(connectivity_t * connectivity) =
  p8est_connectivity_destroy;

int (&Wrapper<3>::connectivity_save)(const char *                 filename,
                                     Wrapper<3>::connectivity_t * connectivity) =
  p8est_connectivity_save;

int (&Wrapper<3>::connectivity_is_valid)(Wrapper<3>::connectivity_t * connectivity) =
  p8est_connectivity_is_valid;

Wrapper<3>::connectivity_t * (
  &Wrapper<3>::connectivity_load)(const char * filename, size_t * length) = p8est_connectivity_load;

void (&Wrapper<3>::geometry_destroy)(geometry_t * geometry) = p8est_geometry_destroy;

Wrapper<3>::forest_t * (&Wrapper<3>::new_forest)(MPI_Comm         mpicomm,
                                                 connectivity_t * connectivity,
                                                 locidx_t         min_quadrants,
                                                 int              min_level,
                                                 int              fill_uniform,
                                                 size_t           data_size,
                                                 init_cb_t        init_fn,
                                                 void *           user_pointer) = p8est_new_ext;

void (&Wrapper<3>::destroy)(Wrapper<3>::forest_t * forest) = p8est_destroy;

void (&Wrapper<3>::refine)(Wrapper<3>::forest_t * forest,
                           int                    refine_recursive,
                           refine_cb_t            refine_fn,
                           init_cb_t              init_fn) = p8est_refine;

void (&Wrapper<3>::coarsen)(Wrapper<3>::forest_t * forest,
                            int                    coarsen_recursive,
                            coarsen_cb_t           coarsen_fn,
                            init_cb_t              init_fn) = p8est_coarsen;

void (&Wrapper<3>::balance)(Wrapper<3>::forest_t *     forest,
                            Wrapper<3>::balance_type_t btype,
                            init_cb_t                  init_fn) = p8est_balance;


Wrapper<3>::gloidx_t (&Wrapper<3>::partition)(Wrapper<3>::forest_t * forest,
                                              int                    partition_for_coarsening,
                                              weight_cb_t weight_fn) = p8est_partition_ext;

void
Wrapper<3>::iterate_volume(forest_t * forest, void * user_data, volume_cb_t volume_cb)
{

  p8est_iterate(forest,
                nullptr, // ghost layer
                user_data,
                volume_cb,
                nullptr, // iter_face
                nullptr, // iter_edge
                nullptr  // iter_corner
  );

} // Wrapper<3>::iterate_volume

void (&Wrapper<3>::save)(const char *           filename,
                         Wrapper<3>::forest_t * forest,
                         int                    save_data) = p8est_save;

Wrapper<3>::forest_t * (&Wrapper<3>::load_ext)(const char *                  filename,
                                               MPI_Comm                      mpicomm,
                                               std::size_t                   data_size,
                                               int                           load_data,
                                               int                           autopartition,
                                               int                           broadcasthead,
                                               void *                        user_pointer,
                                               Wrapper<3>::connectivity_t ** connectivity) =
  p8est_load_ext;

unsigned int (&Wrapper<3>::checksum)(Wrapper<3>::forest_t * forest) = p8est_checksum;

void (&Wrapper<3>::vtk_write_file)(Wrapper<3>::forest_t *   forest,
                                   Wrapper<3>::geometry_t * geometry,
                                   const char *             baseName) = p8est_vtk_write_file;

Wrapper<3>::vtk_context_t * (
  &Wrapper<3>::vtk_context_new)(forest_t * forest, const char * filename) = p8est_vtk_context_new;

void (&Wrapper<3>::vtk_context_set_geom)(vtk_context_t * cont,
                                         geometry_t *    geom) = p8est_vtk_context_set_geom;

void (&Wrapper<3>::vtk_context_set_scale)(vtk_context_t * cont,
                                          double          scale) = p8est_vtk_context_set_scale;

void (&Wrapper<3>::vtk_context_set_continuous)(vtk_context_t * cont,
                                               int continuous) = p8est_vtk_context_set_continuous;

void (&Wrapper<3>::vtk_context_destroy)(vtk_context_t * context) = p8est_vtk_context_destroy;

Wrapper<3>::vtk_context_t * (&Wrapper<3>::vtk_write_header)(vtk_context_t * cont) =
  p8est_vtk_write_header;

Wrapper<3>::vtk_context_t * (&Wrapper<3>::vtk_write_cell_dataf)(vtk_context_t * cont,
                                                                int             write_tree,
                                                                int             write_level,
                                                                int             write_rank,
                                                                int             wrap_rank,
                                                                int             num_cell_scalars,
                                                                int             num_cell_vectors,
                                                                ...) = p8est_vtk_write_cell_dataf;

Wrapper<3>::vtk_context_t * (&Wrapper<3>::vtk_write_point_dataf)(vtk_context_t * cont,
                                                                 int             num_point_scalars,
                                                                 int             num_point_vectors,
                                                                 ...) = p8est_vtk_write_point_dataf;

int (&Wrapper<3>::vtk_write_footer)(vtk_context_t * cont) = p8est_vtk_write_footer;

Wrapper<3>::ghost_t * (&Wrapper<3>::ghost_new)(Wrapper<3>::forest_t *     forest,
                                               Wrapper<3>::balance_type_t btype) = p8est_ghost_new;

void (&Wrapper<3>::ghost_destroy)(Wrapper<3>::ghost_t * ghost) = p8est_ghost_destroy;

void (&Wrapper<3>::reset_data)(Wrapper<3>::forest_t * forest,
                               size_t                 data_size,
                               init_cb_t              init_fn,
                               void *                 user_pointer) = p8est_reset_data;

size_t (&Wrapper<3>::forest_memory_used)(Wrapper<3>::forest_t * forest) = p8est_memory_used;

size_t (&Wrapper<3>::connectivity_memory_used)(Wrapper<3>::connectivity_t * conn) =
  p8est_connectivity_memory_used;

/* ============================================== */
/* ============================================== */
void
Wrapper<2>::quadrant_reset(quadrant_t * quad)
{

  ((void)std::memset((quad), -1, sizeof(quadrant_t)));

} // Wrapper<2>::quadrant_reset(quadrant_t *quad)

void
Wrapper<3>::quadrant_reset(quadrant_t * quad)
{

  ((void)std::memset((quad), -1, sizeof(quadrant_t)));

} // Wrapper<3>::quadrant_reset(quadrant_t *quad)

} // namespace p4est

} // namespace kalypsso
