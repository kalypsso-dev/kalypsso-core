// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef P4EST_WRAPPERS_OLD_H_
#define P4EST_WRAPPERS_OLD_H_

#include <p4est.h> // for 2D
#include <p4est_extended.h>
#include <p4est_ghost.h>
#include <p4est_bits.h>
#include <p4est_vtk.h>

#include <p8est.h> // for 3D
#include <p8est_extended.h>
#include <p8est_ghost.h>
#include <p8est_bits.h>
#include <p8est_vtk.h>

// mapping between logical and physical space
#include <p4est_geometry.h>
#include <p8est_geometry.h>

#include <cstdint>

#if defined(USE_MPI)
#  include <mpi.h>
#else
typedef int MPI_Comm;
#endif


namespace kalypsso
{

namespace p4est
{

/**
 * A structure aimed at being specialized for 2D / 3D, and routing
 * to the right p4est/p8est data/routines.
 *
 * Loosely adapted from Deal.II, file p4est_wrappers.
 * http://www.dealii.org/
 */
template <int dim>
struct Wrapper;

/* 2D specialization */
template <>
struct Wrapper<2>
{

  static constexpr int MAXLEVEL = 30;
  static constexpr int QMAXLEVEL = 29;

  // type alias
  using connectivity_t = p4est_connectivity_t;
  using geometry_t = p4est_geometry_t;
  using forest_t = p4est_t;
  using tree_t = p4est_tree_t;
  using quadrant_t = p4est_quadrant_t;
  using qcoord_t = p4est_qcoord_t;
  using topidx_t = p4est_topidx_t;
  using locidx_t = p4est_locidx_t;
  using gloidx_t = p4est_gloidx_t;
  using ghost_t = p4est_ghost_t;
  using balance_type_t = p4est_connect_type_t;
  using vtk_context_t = p4est_vtk_context_t;

  // iteration types
  using volume_info_t = p4est_iter_volume_info_t;
  using face_info_t = p4est_iter_face_info_t;
  using face_corner_t = p4est_iter_corner_info_t;

  // iteration callback types
  using volume_cb_t = p4est_iter_volume_t;
  using face_cb_t = p4est_iter_face_t;
  using corner_cb_t = p4est_iter_corner_t;

  // callback type alias
  using init_cb_t = p4est_init_t;
  using refine_cb_t = p4est_refine_t;
  using coarsen_cb_t = p4est_coarsen_t;
  using weight_cb_t = p4est_weight_t;

  // quadrant routines
  static inline qcoord_t
  get_x(quadrant_t * quad)
  {
    return quad->x;
  };
  static inline qcoord_t
  get_y(quadrant_t * quad)
  {
    return quad->y;
  };
  static inline qcoord_t
  get_z(quadrant_t * quad)
  {
    return 0.0;
  };

  static inline void
  qcoord_to_vertex(connectivity_t * connectivity, topidx_t treeid, qcoord_t xyz[3], double vxyz[3])
  {
    p4est_qcoord_to_vertex(connectivity, treeid, xyz[0], xyz[1], vxyz);
  };

  static int (&quadrant_is_equal)(const quadrant_t * q1, const quadrant_t * q2);

  static int (&quadrant_is_ancestor)(const quadrant_t * q, const quadrant_t * r);

  static void (&quadrant_child)(const quadrant_t * q, quadrant_t * r, int child_id);

  static int (&quadrant_child_id)(const quadrant_t * q);

  // connectivity / geometry routines
  static void (&connectivity_destroy)(connectivity_t * connectivity);

  static int (&connectivity_save)(const char * filename, connectivity_t * connectivity);
  static int (&connectivity_is_valid)(connectivity_t * connectivity);

  static connectivity_t * (&connectivity_load)(const char * filename, size_t * length);

  static void (&geometry_destroy)(geometry_t * geometry);

  // forest routines
  static forest_t * (&new_forest)(MPI_Comm         mpicomm,
                                  connectivity_t * connectivity,
                                  locidx_t         min_quadrants,
                                  int              min_level,
                                  int              fill_uniform,
                                  size_t           data_size,
                                  init_cb_t        init_fn,
                                  void *           user_pointer);

  static void (&destroy)(forest_t * forest);

  static void (&refine)(forest_t *  forest,
                        int         refine_recursive,
                        refine_cb_t refine_fn,
                        init_cb_t   init_fn);

  static void (&coarsen)(forest_t *   forest,
                         int          coarsen_recursive,
                         coarsen_cb_t coarsen_fn,
                         init_cb_t    init_fn);

  static void (&balance)(forest_t * forest, balance_type_t btype, init_cb_t init_fn);

  static gloidx_t (&partition)(forest_t *  forest,
                               int         partition_for_coarsening,
                               weight_cb_t weight_fn);

  static void
  iterate_volume(forest_t * forest, void * user_data, volume_cb_t volume_cb);


  static void (&save)(const char * filename, forest_t * forest, int save_data);

  static forest_t * (&load_ext)(const char *      filename,
                                MPI_Comm          mpicomm,
                                size_t            data_size,
                                int               load_data,
                                int               autopartition,
                                int               broadcasthead,
                                void *            user_pointer,
                                connectivity_t ** connectivity);


  static unsigned int (&checksum)(forest_t * forest);

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

  static vtk_context_t * (&vtk_write_point_dataf)(vtk_context_t * cont,
                                                  int             num_point_scalars,
                                                  int             num_point_vectors,
                                                  ...);

  static int (&vtk_write_footer)(vtk_context_t * cont);

  static ghost_t * (&ghost_new)(forest_t * forest, balance_type_t btype);

  static void (&ghost_destroy)(ghost_t * ghost);

  static void (&reset_data)(forest_t * forest,
                            size_t     data_size,
                            init_cb_t  init_fn,
                            void *     user_pointer);

  static size_t (&forest_memory_used)(forest_t * forest);

  static size_t (&connectivity_memory_used)(connectivity_t * connectivity);

  static void
  quadrant_reset(quadrant_t * quad);

  static constexpr uint32_t ROOT_LEN = ((qcoord_t)1 << MAXLEVEL);

  static inline uint32_t
  QUADRANT_LEN(int level)
  {
    return (qcoord_t)1 << (MAXLEVEL - (level));
  }

}; // struct Wrapper<2>

/* 3D specialization */
template <>
struct Wrapper<3>
{

  static constexpr int MAXLEVEL = 19;
  static constexpr int QMAXLEVEL = 18;

  // type alias
  using connectivity_t = p8est_connectivity_t;
  using geometry_t = p8est_geometry_t;
  using forest_t = p8est_t;
  using tree_t = p8est_tree_t;
  using quadrant_t = p8est_quadrant_t;
  using qcoord_t = p4est_qcoord_t; /* */
  using topidx_t = p4est_topidx_t; /* */
  using locidx_t = p4est_locidx_t; /* */
  using gloidx_t = p4est_gloidx_t; /* */
  using ghost_t = p8est_ghost_t;
  using balance_type_t = p8est_connect_type_t;
  using vtk_context_t = p8est_vtk_context_t;

  // iteration types
  using volume_info_t = p8est_iter_volume_info_t;
  using face_info_t = p8est_iter_face_info_t;
  using face_corner_t = p8est_iter_corner_info_t;

  // iteration callback types
  using volume_cb_t = p8est_iter_volume_t;
  using face_cb_t = p8est_iter_face_t;
  using corner_cb_t = p8est_iter_corner_t;

  // callback type alias
  using init_cb_t = p8est_init_t;
  using refine_cb_t = p8est_refine_t;
  using coarsen_cb_t = p8est_coarsen_t;
  using weight_cb_t = p8est_weight_t;

  // quadrant routines
  static inline qcoord_t
  get_x(quadrant_t * quad)
  {
    return quad->x;
  };
  static inline qcoord_t
  get_y(quadrant_t * quad)
  {
    return quad->y;
  };
  static inline qcoord_t
  get_z(quadrant_t * quad)
  {
    return quad->z;
  };

  static inline void
  qcoord_to_vertex(connectivity_t * connectivity, topidx_t treeid, qcoord_t xyz[3], double vxyz[3])
  {
    p8est_qcoord_to_vertex(connectivity, treeid, xyz[0], xyz[1], xyz[2], vxyz);
  }

  static int (&quadrant_is_equal)(const quadrant_t * q1, const quadrant_t * q2);

  static int (&quadrant_is_ancestor)(const quadrant_t * q, const quadrant_t * r);

  static void (&quadrant_child)(const quadrant_t * q, quadrant_t * r, int child_id);

  static int (&quadrant_child_id)(const quadrant_t * q);

  // connectivity / geometry routines
  static void (&connectivity_destroy)(connectivity_t * connectivity);

  static int (&connectivity_save)(const char * filename, connectivity_t * connectivity);
  static int (&connectivity_is_valid)(connectivity_t * connectivity);

  static connectivity_t * (&connectivity_load)(const char * filename, size_t * length);

  static void (&geometry_destroy)(geometry_t * geometry);

  // forest routines
  static forest_t * (&new_forest)(MPI_Comm         mpicomm,
                                  connectivity_t * connectivity,
                                  locidx_t         min_quadrants,
                                  int              min_level,
                                  int              fill_uniform,
                                  size_t           data_size,
                                  init_cb_t        init_fn,
                                  void *           user_pointer);

  static void (&destroy)(forest_t * forest);

  static void (&refine)(forest_t *  forest,
                        int         refine_recursive,
                        refine_cb_t refine_fn,
                        init_cb_t   init_fn);

  static void (&coarsen)(forest_t *   forest,
                         int          coarsen_recursive,
                         coarsen_cb_t coarsen_fn,
                         init_cb_t    init_fn);

  static void (&balance)(forest_t * forest, balance_type_t btype, init_cb_t init_fn);

  static gloidx_t (&partition)(forest_t *  forest,
                               int         partition_for_coarsening,
                               weight_cb_t weight_fn);

  static void
  iterate_volume(forest_t * forest, void * user_data, volume_cb_t volume_cb);

  static void (&save)(const char * filename, forest_t * forest, int save_data);

  static forest_t * (&load_ext)(const char *      filename,
                                MPI_Comm          mpicomm,
                                size_t            data_size,
                                int               load_data,
                                int               autopartition,
                                int               broadcasthead,
                                void *            user_pointer,
                                connectivity_t ** connectivity);

  static unsigned int (&checksum)(forest_t * forest);

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

  static vtk_context_t * (&vtk_write_point_dataf)(vtk_context_t * cont,
                                                  int             num_point_scalars,
                                                  int             num_point_vectors,
                                                  ...);

  static int (&vtk_write_footer)(vtk_context_t * cont);

  static ghost_t * (&ghost_new)(forest_t * forest, balance_type_t btype);

  static void (&ghost_destroy)(ghost_t * ghost);

  static void (&reset_data)(forest_t * forest,
                            size_t     data_size,
                            init_cb_t  init_fn,
                            void *     user_pointer);

  static size_t (&forest_memory_used)(forest_t * forest);

  static size_t (&connectivity_memory_used)(connectivity_t * connectivity);

  static void
  quadrant_reset(quadrant_t * quad);

  static constexpr uint32_t ROOT_LEN = ((qcoord_t)1 << MAXLEVEL);

  static inline uint32_t
  QUADRANT_LEN(int level)
  {
    return (qcoord_t)1 << (MAXLEVEL - (level));
  }

}; // struct Wrapper<3>

} // namespace p4est

} // namespace kalypsso

#endif // P4EST_WRAPPERS_OLD_H_
