// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file test_io.cpp
 */
/*
 * Usage: test_simple_vtkio <configuration> <level>
 *
 *        possible 2D configurations:
 *        o unit       Refinement on the unit square.
 *        o three      Refinement on a forest with three trees.
 *        o brick      Refinement on a regular grid of trees (orchard)
 *        o evil       Check second round of refinement with np=5 level=7
 *        o evil3      Check second round of refinement on three trees
 *        o pillow     Refinement on a 2-tree pillow-shaped domain.
 *        o moebius    Refinement on a 5-tree Moebius band.
 *        o star       Refinement on a 6-tree star shaped domain.
 *        o cubed      Refinement on a 6-tree cubed sphere surface.
 *        o disk       Refinement on a 5-tree spherical disk.
 *        o periodic   Refinement on the unit square with all-periodic b.c.
 *        o rotwrap    Refinement on the unit square with weird periodic b.c.
 *        o icosahedron   Refinement on the sphere
 *        o shell2d       Refinement on a 2d shell with geometry.
 *        o disk2d        Refinement on a 2d disk with geometry.
 *
 *        possible 3D configurations:
 *        o unit3      The unit cube.
 *        o brick3      Refinement on a regular grid of trees (orchard)
 *        o periodic3  The unit cube with all-periodic boundary conditions.
 *        o rotwrap3   The unit cube with various self-periodic b.c.
 *        o twocubes3  Two connected cubes.
 *        o twowrap3   Two cubes with periodically identified far ends.
 *        o rotcubes3  A collection of six connected rotated cubes.
 *        o shell3     A 24-tree discretization of a hollow sphere.
 *        o sphere3    A 13-tree discretization of a solid sphere.
 *        o torus      A configurable discretization of a solid torus.
 */

#if defined(KALYPSSO_IO_VTK)
#  include <kalypsso/core/SimpleVTKIO.h>
#elif defined(KALYPSSO_IO_HDF5_BLOCK) || defined(KALYPSSO_IO_HDF5_GHOSTED_BLOCK)
#  include <kalypsso/core/HDF5_Xdmf_Writer_legacy.h>
#endif

#include <kalypsso/core/io_utils.h>
#include <kalypsso/core/models/Hydro.h>
#include <kalypsso/core/kalypsso_data_container.h>
#include <kalypsso/core/enums.h>

#include <kalypsso/utils/p4est/p4est_wrapper.h>
#include <kalypsso/utils/p4est/connectivity.h>
#include <kalypsso/utils/p4est/geometry.h>

#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <cstdlib>

enum simple_config_t
{
  // 2D
  P4EST_CONFIG_NULL,
  P4EST_CONFIG_UNIT,
  P4EST_CONFIG_THREE,
  P4EST_CONFIG_BRICK,
  P4EST_CONFIG_EVIL,
  P4EST_CONFIG_EVIL3,
  P4EST_CONFIG_PILLOW,
  P4EST_CONFIG_MOEBIUS,
  P4EST_CONFIG_STAR,
  P4EST_CONFIG_CUBED,
  P4EST_CONFIG_DISK,
  P4EST_CONFIG_PERIODIC,
  P4EST_CONFIG_ROTWRAP,
  P4EST_CONFIG_ICOSAHEDRON,
  P4EST_CONFIG_DISK2D,
  P4EST_CONFIG_SHELL2D,
  // 3D
  P8EST_CONFIG_NULL,
  P8EST_CONFIG_UNIT,
  P8EST_CONFIG_PERIODIC,
  P8EST_CONFIG_BRICK,
  P8EST_CONFIG_ROTWRAP,
  P8EST_CONFIG_TWOCUBES,
  P8EST_CONFIG_TWOWRAP,
  P8EST_CONFIG_ROTCUBES,
  P8EST_CONFIG_SHELL,
  P8EST_CONFIG_SPHERE,
  P8EST_CONFIG_TORUS
};

struct user_data_t
{
  double x; // physical space
};

static int refine_level = 0;

// =======================================================================================
// =======================================================================================
template <int dim>
void
quadrant_center_vertex(typename kalypsso::p4est::Wrapper<dim>::connectivity_t * connectivity,
                       typename kalypsso::p4est::Wrapper<dim>::geometry_t *     geom,
                       typename kalypsso::p4est::topidx_t                       tree,
                       typename kalypsso::p4est::Wrapper<dim>::quadrant_t *     quad,
                       double                                                   xyz[3])
{

  using namespace kalypsso::p4est;

  uint32_t root_len = Wrapper<dim>::ROOT_LEN;
  uint32_t quad_len = Wrapper<dim>::QUADRANT_LEN(quad->level);

  qcoord_t half_length = quad_len / 2;

  double       h2 = 0.5 * quad_len / root_len;
  const double intsize = 1.0 / root_len;

  if (geom != nullptr)
  {

    double xyz_logic[3] = { 0., 0., 0. };

    /*
     * get coordinates at cell center
     */
    xyz_logic[0] = intsize * get_x(quad) + h2;
    xyz_logic[1] = intsize * get_y(quad) + h2;
    xyz_logic[2] = dim == 3 ? intsize * get_z(quad) + h2 : 0.0;

    // from logical coordinates to physical coordinates
    geom->X(geom, tree, xyz_logic, xyz);
  }
  else
  { // connectivity space (no deformation geometry)

    qcoord_t xyz_logic[3] = { get_x(quad) + half_length,
                              get_y(quad) + half_length,
                              get_z(quad) + half_length };

    Wrapper<dim>::qcoord_to_vertex(connectivity, tree, xyz_logic, xyz);

  } // end cartesian geometry

} // quadrant_center_vertex

// =======================================================================================
// =======================================================================================
template <int dim>
static void
init_fn(typename kalypsso::p4est::Wrapper<dim>::forest_t *   p4est,
        typename kalypsso::p4est::topidx_t                   which_tree,
        typename kalypsso::p4est::Wrapper<dim>::quadrant_t * quadrant)
{

  using namespace kalypsso::p4est;

  using connectivity_t = typename Wrapper<dim>::connectivity_t;
  using geometry_t = typename Wrapper<dim>::geometry_t;

  user_data_t * data = (user_data_t *)quadrant->p.user_data;

  // compute physical space coordinates, and put "x" into user_data

  connectivity_t * conn = (connectivity_t *)p4est->connectivity;
  geometry_t *     geom = (geometry_t *)p4est->user_pointer;

  double XYZ[3];

  quadrant_center_vertex<dim>(conn, geom, which_tree, quadrant, XYZ);

  data->x = XYZ[0];
}

// =======================================================================================
// =======================================================================================
template <int dim>
static int
refine_normal_fn(typename kalypsso::p4est::Wrapper<dim>::forest_t *   p4est,
                 typename kalypsso::p4est::topidx_t                   which_tree,
                 typename kalypsso::p4est::Wrapper<dim>::quadrant_t * quadrant)
{
  using namespace kalypsso::p4est;

  if ((int)quadrant->level >= (refine_level - (int)(which_tree % 3)))
  {
    return 0;
  }
  if (quadrant->level == 1 && Wrapper<dim>::quadrant_child_id(quadrant) == 3)
  {
    return 1;
  }
  if (quadrant->x == P4EST_LAST_OFFSET(2) && quadrant->y == P4EST_LAST_OFFSET(2))
  {
    return 1;
  }

  if (dim == 2)
  {
    if (get_x(quadrant) >= (int)Wrapper<dim>::QUADRANT_LEN(2))
    {
      return 0;
    }
  }
  else
  {
    if (get_z(quadrant) >= (int)Wrapper<dim>::QUADRANT_LEN(2))
    {
      return 0;
    }
  }

  return 1;
}

// ================================================================================================
// ================================================================================================
template <int dim>
static int
refine_sparse_fn(typename kalypsso::p4est::Wrapper<dim>::forest_t *   forest,
                 typename kalypsso::p4est::topidx_t                   which_tree,
                 typename kalypsso::p4est::Wrapper<dim>::quadrant_t * quadrant)
{
  using namespace kalypsso::p4est;
  // uint32_t quad_len = Wrapper<dim>::QUADRANT_LEN (quadrant->level);

  if (which_tree != 0)
  {
    return 0;
  }
  if ((int)quadrant->level >= refine_level)
  {
    return 0;
  }
  if (quadrant->level == 0)
  {
    return 1;
  }
  if (dim == 2)
  {
    if (get_x(quadrant) < (int)Wrapper<dim>::QUADRANT_LEN(2) && get_y(quadrant) > 0)
    {
      return 1;
    }
  }
  else
  {
    if (get_x(quadrant) < (int)Wrapper<dim>::QUADRANT_LEN(2) && get_y(quadrant) > 0 and
        get_z(quadrant) < (int)Wrapper<dim>::QUADRANT_LEN(2))
    {
      return 1;
    }
  }

  return 0;
}

template <int dim>
static int
refine_evil_fn(typename kalypsso::p4est::Wrapper<dim>::forest_t *   forest,
               typename kalypsso::p4est::topidx_t                   which_tree,
               typename kalypsso::p4est::Wrapper<dim>::quadrant_t * quadrant)
{
  if ((int)quadrant->level >= refine_level)
  {
    return 0;
  }
  if (forest->mpirank <= 1)
  {
    return 1;
  }

  return 0;
}

template <int dim>
static int
refine_evil3_fn(typename kalypsso::p4est::Wrapper<dim>::forest_t *   p4est,
                typename kalypsso::p4est::topidx_t                   which_tree,
                typename kalypsso::p4est::Wrapper<dim>::quadrant_t * quadrant)
{

  using namespace kalypsso::p4est;

  qcoord_t                          u2;
  typename Wrapper<dim>::quadrant_t ref;

  P4EST_QUADRANT_INIT(&ref);

  u2 = Wrapper<dim>::QUADRANT_LEN(2);

  if (which_tree == 0)
  {
    ref.x = 3 * u2;
    ref.y = 2 * u2;
  }
  else if (which_tree == 1)
  {
    ref.x = 2 * u2;
    ref.y = 3 * u2;
  }
  ref.level = 2;

  if ((int)quadrant->level >= refine_level)
  {
    return 0;
  }
  if ((which_tree == 0 || which_tree == 1) && (Wrapper<dim>::quadrant_is_equal(&ref, quadrant) ||
                                               Wrapper<dim>::quadrant_is_ancestor(&ref, quadrant)))
  {
    return 1;
  }

  return 0;
}

template <int dim>
static int
coarsen_evil_fn(typename kalypsso::p4est::Wrapper<dim>::forest_t *   forest,
                typename kalypsso::p4est::topidx_t                   which_tree,
                typename kalypsso::p4est::Wrapper<dim>::quadrant_t * quadrant[])
{
  if (forest->mpirank >= 2)
  {
    return 1;
  }

  return 0;
}

template <int dim>
static int
refine_icosahedron_fn(typename kalypsso::p4est::Wrapper<dim>::forest_t *   forest,
                      typename kalypsso::p4est::topidx_t                   which_tree,
                      typename kalypsso::p4est::Wrapper<dim>::quadrant_t * quadrant)
{
  using namespace kalypsso::p4est;
  using geometry_t = typename Wrapper<dim>::geometry_t;

  geometry_t * geom = (geometry_t *)forest->user_pointer;

  /* logical coordinates */
  double xyz[3] = { 0, 0, 0 };

  /* physical coordinates */
  double XYZ[3] = { 0, 0, 0 };

  double       h2 = 0.5 * Wrapper<dim>::QUADRANT_LEN(quadrant->level) / Wrapper<dim>::ROOT_LEN;
  const double intsize = 1.0 / Wrapper<dim>::ROOT_LEN;

  /*
   * get coordinates at cell center
   */
  xyz[0] = intsize * quadrant->x + h2;
  xyz[1] = intsize * quadrant->y + h2;
  xyz[2] = 0; // dim == 3 ? intsize * quadrant->z + h2 : 0;

  /* from logical coordinates to physical coordinates (cartesian) */
  geom->X(geom, which_tree, xyz, XYZ);

  if (quadrant->level > 6)
    return 0;
  if (XYZ[2] > 0 and quadrant->level >= 3)
    return 0;

  return 1;
}


template <int dim>
static int
refine_radius_fn(typename kalypsso::p4est::Wrapper<dim>::forest_t *   forest,
                 typename kalypsso::p4est::topidx_t                   which_tree,
                 typename kalypsso::p4est::Wrapper<dim>::quadrant_t * quadrant)
{

  using namespace kalypsso::p4est;
  using geometry_t = typename Wrapper<dim>::geometry_t;

  uint32_t root_len = Wrapper<dim>::ROOT_LEN;
  uint32_t quad_len = Wrapper<dim>::QUADRANT_LEN(quadrant->level);

  // stop criterion
  if ((int)quadrant->level >= refine_level)
  {
    return 0;
  }

  geometry_t * geom = (geometry_t *)forest->user_pointer;

  // logical coordinates
  double xyz[3] = { 0, 0, 0 };

  // physical coordinates (after geometry mapping )
  double XYZ[3] = { 0, 0, 0 };

  // half-size of the cell in logical coordinate
  double h2 = 0.5 * quad_len / root_len;

  const double intsize = 1.0 / root_len;

  /*
   * get coordinates at cell center
   */
  xyz[0] = intsize * get_x(quadrant) + h2;
  xyz[1] = intsize * get_y(quadrant) + h2;
  if (dim == 3)
    xyz[2] = intsize * get_z(quadrant) + h2;

  // apply mapping from logical coordinates to physical coordinates
  geom->X(geom, which_tree, xyz, XYZ);

  double radius = 0;
  if (dim == 3)
    radius = sqrt(XYZ[0] * XYZ[0] + XYZ[1] * XYZ[1] + XYZ[2] * XYZ[2]);
  else
    radius = sqrt(XYZ[0] * XYZ[0] + XYZ[1] * XYZ[1]);

  if (dim == 2)
  {
    if (radius < 0.7 and quadrant->level > 4)
      return 0;
    if (radius < 0.8 and quadrant->level > 5)
      return 0;
    if (radius < 0.9 and quadrant->level > 6)
      return 0;
  }
  else
  {
    if (radius < 0.7 and quadrant->level > 3)
      return 0;
    if (radius < 0.85 and quadrant->level > 4)
      return 0;
    if (radius < 0.95 and quadrant->level > 5)
      return 0;
    if (radius < 0.99 and quadrant->level > 6)
      return 0;
  }

  return 1;
}

// =======================================================================================
// =======================================================================================
/**
 * callback routine for p4est_iterate to generate fake data.
 */
template <int dim>
void
set_userdata_cb(typename kalypsso::p4est::Wrapper<dim>::volume_info_t * info, void * user_data)
{

  using namespace kalypsso::p4est;

  using forest_t = typename Wrapper<dim>::forest_t;
  using tree_t = typename Wrapper<dim>::tree_t;
  using quadrant_t = typename Wrapper<dim>::quadrant_t;

  /* we passed the array of values to fill as the user_data in the call
     to p4est_iterate */
  sc_array_t * scalar_data = (sc_array_t *)user_data;
  double *     this_data;
  forest_t *   forest = info->p4est;
  quadrant_t * q = info->quad;
  topidx_t     which_tree = info->treeid;
  locidx_t     local_id =
    info->quadid; /* this is the index of q *within its tree's numbering*.  We want to convert it
                     its index for all the quadrants on this process, which we do below */
  tree_t *      tree;
  user_data_t * udata = (user_data_t *)q->p.user_data;
  locidx_t      arrayoffset;

  tree = (tree_t *)(forest->trees->array + sizeof(tree_t) * (size_t)which_tree);

  /* compute id relative inside current MPI process */
  local_id += tree->quadrants_offset;
  arrayoffset = local_id;

  this_data = (double *)sc_array_index(scalar_data, arrayoffset);
  this_data[0] = udata->x;

} // set_userdata_cb

// =======================================================================================
// =======================================================================================
/**
 * callback routine for p4est_iterate to generate fake data.
 */
template <int dim>
void
set_orchard_key_cb(typename kalypsso::p4est::Wrapper<dim>::volume_info_t * info, void * user_data)
{

  using namespace kalypsso::p4est;

  using forest_t = typename Wrapper<dim>::forest_t;
  using tree_t = typename Wrapper<dim>::tree_t;
  using quadrant_t = typename Wrapper<dim>::quadrant_t;

  /* we passed the array of values to fill as the user_data in the call
     to p4est_iterate */
  sc_array_t * scalar_data = (sc_array_t *)user_data;
  double *     this_data;
  forest_t *   forest = info->p4est;
  quadrant_t * q = info->quad;
  topidx_t     which_tree = info->treeid;
  locidx_t     local_id =
    info->quadid; /* this is the index of q *within its tree's numbering*.  We want to convert it
                     its index for all the quadrants on this process, which we do below */
  tree_t *      tree;
  user_data_t * udata = (user_data_t *)q->p.user_data;
  locidx_t      arrayoffset;

  tree = (tree_t *)(forest->trees->array + sizeof(tree_t) * (size_t)which_tree);

  /* compute id relative inside current MPI process */
  local_id += tree->quadrants_offset;
  arrayoffset = local_id;

  this_data = (double *)sc_array_index(scalar_data, arrayoffset);
  this_data[0] = udata->x; // TODO - WIP

} // set_orchard_key_cb

// =======================================================================================
// =======================================================================================
/*
 * Output routine.
 */
template <int dim, typename device_t>
void
my_p4est_write_file(kalypsso::ParallelEnv const &                        par_env,
                    typename kalypsso::p4est::Wrapper<dim>::forest_t *   forest,
                    std::string                                          connectivity_name,
                    int                                                  brick_dim[3],
                    typename kalypsso::p4est::Wrapper<dim>::geometry_t * geom,
                    std::string                                          filename,
                    [[maybe_unused]] double                              scale)
{

  using namespace kalypsso::p4est;

  // makes enum Hydro::VarId available
  using Hydro = kalypsso::core::models::Hydro;

  // number of quadrant in local processor
  auto numOcts = forest->local_num_quadrants;

  // create fake data as a  scalar array
  sc_array_t * scalar_data;
  scalar_data = sc_array_new_size(sizeof(double), forest->local_num_quadrants);

  // Use the iterator to visit every cell and fill scalar_data vector
  Wrapper<dim>::iterate_volume(forest, nullptr, (void *)scalar_data, set_userdata_cb<dim>);

  // config map
#if defined(KALYPSSO_IO_VTK)
  kalypsso::ConfigMap config_map = kalypsso::broadcast_parameters("./test_io_vtk.ini");
#elif defined(KALYPSSO_IO_HDF5_BLOCK) || defined(KALYPSSO_IO_HDF5_GHOSTED_BLOCK)
  kalypsso::ConfigMap config_map = kalypsso::broadcast_parameters("./test_io_hdf5_block.ini");
#endif

  // ensure amr connectivity is set
  config_map.setString("amr", "connectivity", connectivity_name);

  // ensure brick sizes are set in config_map
  config_map.setInteger("p4est_connectivity", "nbrick_x", brick_dim[0]);
  config_map.setInteger("p4est_connectivity", "nbrick_y", brick_dim[1]);
  config_map.setInteger("p4est_connectivity", "nbrick_z", brick_dim[2]);

  // create fake DataArray as Kokkos::View
  //
  // HDF5 can write both block data and leaf data
  // VTK can (currently) only write leaf data

  [[maybe_unused]] const int nbvar = 5;

#if defined(KALYPSSO_IO_HDF5_BLOCK)
  using DataArrayBlock_t = kalypsso::DataArrayBlock<dim, kalypsso::real_t, device_t>;

  int32_t  bx = config_map.getInteger("amr", "bx", 1);
  int32_t  by = config_map.getInteger("amr", "by", 1);
  int32_t  bz = config_map.getInteger("amr", "bz", 1);
  uint32_t nbCells = dim == 2 ? bx * by : bx * by * bz;

  const auto block_sizes = [=]() {
    if constexpr (dim == 2)
      return kalypsso::block_size_t<2>{ bx, by };
    if constexpr (dim == 3)
      return kalypsso::block_size_t<3>{ bx, by, bz };
  }();

  auto fakeData_block = DataArrayBlock_t("fakeData_block", block_sizes, nbvar, numOcts);
  auto fakeData_block_h = DataArrayBlock_t::create_host_mirror_view(fakeData_block);
#elif defined(KALYPSSO_IO_HDF5_GHOSTED_BLOCK)
  using DataArrayGhostedBlock_t = kalypsso::DataArrayGhostedBlock<dim, kalypsso::real_t, device_t>;
  using DataArrayGhostedBlockHost_t =
    kalypsso::DataArrayGhostedBlock<dim, kalypsso::real_t, kalypsso::HostDevice>;

  // get block sizes
  int32_t bx = config_map.getInteger("amr", "bx", 1);
  int32_t by = config_map.getInteger("amr", "by", 1);
  int32_t bz = config_map.getInteger("amr", "bz", 1);

  const auto block_sizes = [=]() {
    if constexpr (dim == 2)
      return kalypsso::block_size_t<2>{ bx, by };
    if constexpr (dim == 3)
      return kalypsso::block_size_t<3>{ bx, by, bz };
  }();

  int nbCells = dim == 2 ? bx * by : bx * by * bz;

  // get ghost sizes
  const int32_t gx = config_map.getInteger("amr", "gx", 2);
  const int32_t gy = config_map.getInteger("amr", "gy", 2);
  const int32_t gz = config_map.getInteger("amr", "gz", 2);

  const auto ghosted_sizes = [=]() {
    if constexpr (dim == 2)
      return kalypsso::block_size_t<2>{ bx + 2 * gx, by + 2 * gy };
    if constexpr (dim == 3)
      return kalypsso::block_size_t<3>{ bx + 2 * gx, by + 2 * gy, bz + 2 * gz };
  }();

  const auto shift = [=]() {
    if constexpr (dim == 2)
      return kalypsso::block_size_t<2>{ -gx, -gy };
    if constexpr (dim == 3)
      return kalypsso::block_size_t<3>{ -gx, -gy, -gz };
  }();

  auto fakeData_ghosted_block = DataArrayGhostedBlock_t(block_sizes,
                                                        ghosted_sizes,
                                                        shift,
                                                        "test_data_ghosted_block",
                                                        nbvar,
                                                        forest->local_num_quadrants);

  auto fakeData_ghosted_block_h = DataArrayGhostedBlockHost_t(block_sizes,
                                                              ghosted_sizes,
                                                              shift,
                                                              "test_data_ghosted_block_host",
                                                              nbvar,
                                                              forest->local_num_quadrants);

  auto total_block_sizes = fakeData_ghosted_block.ghosted_block_size();
  auto tx = total_block_sizes[kalypsso::IX];
  auto ty = total_block_sizes[kalypsso::IY];
  auto tz = [=]() {
    if constexpr (dim == 2)
      return 1;
    if constexpr (dim == 3)
      return total_block_sizes[kalypsso::IZ];
  }();

#endif

  using DataArrayLeaf_t = kalypsso::DataArrayLeaf<kalypsso::real_t, device_t>;

  auto fakeData_leaf = DataArrayLeaf_t("fakeData_leaf", numOcts, 5);
  auto fakeData_leaf_h = Kokkos::create_mirror(fakeData_leaf);

  // model
  Hydro::Settings settings{ dim };
  Hydro           model(settings);

  // build list of variables to write by parsing config field "output/write_variables"
  // list is actually a map of names to enum
  kalypsso::names2index_t names2index; // this is initially empty
  kalypsso::build_var_to_write_map(names2index, model.get_names2id_map(), config_map);

  auto       fm = model.get_fieldmap();
  const auto id2names = model.get_id2names_map();

  // fill fakeData on host
#if defined(KALYPSSO_IO_HDF5_BLOCK)
  for (locidx_t iOct = 0; iOct < numOcts; ++iOct)
  {
    // global index
    gloidx_t iOct_g = iOct + forest->global_first_quadrant[forest->mpirank];

    double * ptr = (double *)sc_array_index(scalar_data, iOct);
    for (uint32_t icell = 0; icell < nbCells; ++icell)
    {
      fakeData_block_h(icell, fm[Hydro::ID], iOct) = ptr[0];
      fakeData_block_h(icell, fm[Hydro::IE], iOct) = 42.0;
      fakeData_block_h(icell, fm[Hydro::IU], iOct) = 1.0 * iOct_g + icell * 1.0 / nbCells;
      fakeData_block_h(icell, fm[Hydro::IV], iOct) = 2.0;
    }
  }
#elif defined(KALYPSSO_IO_HDF5_GHOSTED_BLOCK)
  for (locidx_t iOct = 0; iOct < numOcts; ++iOct)
  {
    // global index
    gloidx_t iOct_g = iOct + forest->global_first_quadrant[forest->mpirank];

    double * ptr = (double *)sc_array_index(scalar_data, iOct);

    const auto shift = fakeData_ghosted_block.shift();
    const auto ghosted_size = fakeData_ghosted_block.ghosted_block_size();

    if constexpr (dim == 2)
    {
      for (int32_t j = shift[kalypsso::IY]; j < shift[kalypsso::IY] + ghosted_size[kalypsso::IY];
           ++j)
        for (int32_t i = shift[kalypsso::IX]; i < shift[kalypsso::IX] + ghosted_size[kalypsso::IX];
             ++i)
        {
          if (i >= 0 and i < bx and j >= 0 and j < by)
          {
            auto icell = (i + gx) + bx * (j + gy);
            fakeData_ghosted_block_h(i, j, fm[Hydro::ID], iOct) = ptr[0];
            fakeData_ghosted_block_h(i, j, fm[Hydro::IE], iOct) = 42.0;
            fakeData_ghosted_block_h(i, j, fm[Hydro::IU], iOct) =
              1.0 * iOct_g + icell * 1.0 / nbCells;
            fakeData_ghosted_block_h(i, j, fm[Hydro::IV], iOct) = 2.0;
          }
          else
          {
            fakeData_ghosted_block_h(i, j, fm[Hydro::ID], iOct) = -0.1;
            fakeData_ghosted_block_h(i, j, fm[Hydro::IE], iOct) = 1.0;
            fakeData_ghosted_block_h(i, j, fm[Hydro::IU], iOct) = 2.0;
            fakeData_ghosted_block_h(i, j, fm[Hydro::IV], iOct) = 3.0;
          }
        }
    }
    if constexpr (dim == 3)
    {
      for (int32_t k = shift[kalypsso::IZ]; k < shift[kalypsso::IZ] + ghosted_size[kalypsso::IZ];
           ++k)
        for (int32_t j = shift[kalypsso::IY]; j < shift[kalypsso::IY] + ghosted_size[kalypsso::IY];
             ++j)
          for (int32_t i = shift[kalypsso::IX];
               i < shift[kalypsso::IX] + ghosted_size[kalypsso::IX];
               ++i)
          {
            if (i >= 0 and i < bx and j >= 0 and j < by and k >= 0 and k < bz)
            {
              auto icell = (i + gx) + bx * (j + gy) + bx * by * (k + gz);
              fakeData_ghosted_block_h(i, j, k, fm[Hydro::ID], iOct) = ptr[0];
              fakeData_ghosted_block_h(i, j, k, fm[Hydro::IE], iOct) = 42.0;
              fakeData_ghosted_block_h(i, j, k, fm[Hydro::IU], iOct) =
                1.0 * iOct_g + icell * 1.0 / nbCells;
              fakeData_ghosted_block_h(i, j, k, fm[Hydro::IV], iOct) = 2.0;
              fakeData_ghosted_block_h(i, j, k, fm[Hydro::IW], iOct) = 3.0;
            }
            else
            {
              fakeData_ghosted_block_h(i, j, k, fm[Hydro::ID], iOct) = 0.0;
              fakeData_ghosted_block_h(i, j, k, fm[Hydro::IE], iOct) = 1.0;
              fakeData_ghosted_block_h(i, j, k, fm[Hydro::IU], iOct) = 2.0;
              fakeData_ghosted_block_h(i, j, k, fm[Hydro::IV], iOct) = 3.0;
              fakeData_ghosted_block_h(i, j, k, fm[Hydro::IW], iOct) = 4.0;
            }
          }
    }
  }
  Kokkos::deep_copy(fakeData_ghosted_block.flat_view(), fakeData_ghosted_block_h.flat_view());
#endif

  for (locidx_t i = 0; i < numOcts; ++i)
  {
    double * ptr = (double *)sc_array_index(scalar_data, i);
    fakeData_leaf_h(i, fm[Hydro::ID]) = ptr[0];
    fakeData_leaf_h(i, fm[Hydro::IE]) = 42.0;
    fakeData_leaf_h(i, fm[Hydro::IU]) = 1.0 * i;
    fakeData_leaf_h(i, fm[Hydro::IV]) = 2.0;
  }

  // and copy to device
  // usually, it is the other way around
  Kokkos::deep_copy(fakeData_leaf, fakeData_leaf_h);

#if defined(KALYPSSO_IO_HDF5_BLOCK)
  Kokkos::deep_copy(fakeData_block.logical_view(), fakeData_block_h.logical_view());
#endif

#if defined(KALYPSSO_IO_VTK)
  // finally write vtk file
  kalypsso::writeVTK<dim, device_t, kalypsso::core::models::Hydro>(
    forest, geom, fakeData_leaf, model, filename);
#elif defined(KALYPSSO_IO_HDF5_BLOCK) || defined(KALYPSSO_IO_HDF5_GHOSTED_BLOCK)

  const auto start_index = [=]() {
    if constexpr (dim == 2)
      return kalypsso::coord_t<2>{ 0, 0 };
    if constexpr (dim == 3)
      return kalypsso::coord_t<3>{ 0, 0, 0 };
  }();

  // don't write mesh info at cell level (only at block level)
  kalypsso::HDF5_Xdmf_Writer_legacy<dim, device_t> writer(
    forest, geom, config_map, block_sizes, start_index);

  std::string outputDir = config_map.getString("output", "outputDir", "./");

#  if defined(KALYPSSO_IO_HDF5_GHOSTED_BLOCK)
  // start index is opposite of shift
  const auto start_index2 = [=]() {
    if constexpr (dim == 2)
      return kalypsso::coord_t<2>{ gx, gy };
    if constexpr (dim == 3)
      return kalypsso::coord_t<3>{ gx, gy, gz };
  }();

  // 1. save only the inner part (should produce exactly the same hdf5 as KALYPSSO_IO_HDF5_BLOCK)
  // this can be check using h5diff
  writer.set_block_mode(block_sizes, start_index2);

  // 2. save all (inner part and ghost cells)
  // writer.set_block_mode(ghosted_sizes, start_index);
#  endif

  writer.set_write_mesh_info(true);
  writer.update_mesh_info();
  writer.open(filename, outputDir);
  writer.write_header(0.0);

  // write user the fake data (all scalar fields, here only one)
#  if defined(KALYPSSO_IO_HDF5_BLOCK)
  writer.write_quadrant_attribute(fakeData_block_h, fm[Hydro::ID], id2names.at(Hydro::ID));
  writer.write_quadrant_attribute(fakeData_block_h, fm[Hydro::IE], id2names.at(Hydro::IE));
  writer.write_quadrant_attribute(fakeData_block_h, fm[Hydro::IU], id2names.at(Hydro::IU));
  writer.write_quadrant_attribute(fakeData_block_h, fm[Hydro::IV], id2names.at(Hydro::IV));
#  elif defined(KALYPSSO_IO_HDF5_GHOSTED_BLOCK)
  Kokkos::deep_copy(fakeData_ghosted_block_h.flat_view(), fakeData_ghosted_block.flat_view());
  writer.write_quadrant_attribute(
    fakeData_ghosted_block_h.data(), fm[Hydro::ID], id2names.at(Hydro::ID));
  writer.write_quadrant_attribute(
    fakeData_ghosted_block_h.data(), fm[Hydro::IE], id2names.at(Hydro::IE));
  writer.write_quadrant_attribute(
    fakeData_ghosted_block_h.data(), fm[Hydro::IU], id2names.at(Hydro::IU));
  writer.write_quadrant_attribute(
    fakeData_ghosted_block_h.data(), fm[Hydro::IV], id2names.at(Hydro::IV));
#  endif

  // check if we want to write velocity or rhoV vector fields
  // std::string write_variables = config_map.getString("output", "write_variables", "");

  // close the file
  writer.write_footer();
  writer.close();

  //
  // now write leaf data
  //
  std::string filename2 = filename + "_quad";

  writer.set_leaf_mode();
  writer.set_write_mesh_info(true);

  // writer2.update_mesh_info();
  writer.open(filename2, outputDir);
  writer.write_header(0.0);

  writer.write_quadrant_attribute(fakeData_leaf_h, fm[Hydro::ID], id2names.at(Hydro::ID));

  // close the file
  writer.write_footer();
  writer.close();

#endif

  sc_array_destroy(scalar_data);

} // my_p4est_write_file

template <int dim>
using connectivity_data_t =
  std::tuple<typename kalypsso::p4est::Wrapper<dim>::connectivity_t *, std::string>;

/* ================================================================= */
/* ================================================================= */
/* ================================================================= */
template <int dim>
connectivity_data_t<dim>
create_connectivity(simple_config_t config, int brick_dim[3]);

template <>
connectivity_data_t<2>
create_connectivity<2>(simple_config_t config, int brick_dim[3])
{

  typename kalypsso::p4est::Wrapper<2>::connectivity_t * connectivity = nullptr;
  std::string                                            connectivity_name = "unknown";

  if (config == P4EST_CONFIG_UNIT)
  {
    connectivity = p4est_connectivity_new_unitsquare();
    connectivity_name = "unit";
  }
  else if (config == P4EST_CONFIG_THREE || config == P4EST_CONFIG_EVIL3)
  {
    connectivity = p4est_connectivity_new_corner();
    connectivity_name = "corner";
  }
  else if (config == P4EST_CONFIG_BRICK)
  {
    connectivity = p4est_connectivity_new_brick(brick_dim[0], brick_dim[1], 1, 1);
    connectivity_name = "brick";
  }
  else if (config == P4EST_CONFIG_PILLOW)
  {
    connectivity = p4est_connectivity_new_pillow();
    connectivity_name = "pillow";
  }
  else if (config == P4EST_CONFIG_MOEBIUS)
  {
    connectivity = p4est_connectivity_new_moebius();
    connectivity_name = "moebius";
  }
  else if (config == P4EST_CONFIG_STAR)
  {
    connectivity = p4est_connectivity_new_star();
    connectivity_name = "star";
  }
  else if (config == P4EST_CONFIG_CUBED)
  {
    connectivity = p4est_connectivity_new_cubed();
    connectivity_name = "cubed";
  }
  else if (config == P4EST_CONFIG_DISK)
  {
#if defined(KALYPSSO_CORE_USE_OLD_P4EST_API)
    connectivity = p4est_connectivity_new_disk();
#else
    // non periodic disk
    connectivity = p4est_connectivity_new_disk(0, 0);
#endif // USE_OLD_P4EST_API
    connectivity_name = "disk";
  }
  else if (config == P4EST_CONFIG_PERIODIC)
  {
    connectivity = p4est_connectivity_new_periodic();
    connectivity_name = "periodic";
  }
  else if (config == P4EST_CONFIG_ROTWRAP)
  {
    connectivity = p4est_connectivity_new_rotwrap();
    connectivity_name = "rotwrap";
  }
  else if (config == P4EST_CONFIG_ICOSAHEDRON)
  {
    connectivity = p4est_connectivity_new_icosahedron();
    connectivity_name = "icosahedron";
  }
  else if (config == P4EST_CONFIG_DISK2D)
  {
    connectivity = p4est_connectivity_new_disk2d();
    connectivity_name = "disk2d";
  }
  else if (config == P4EST_CONFIG_SHELL2D)
  {
    connectivity = p4est_connectivity_new_shell2d();
    connectivity_name = "shell2d";
  }
  else
  {
    connectivity = p4est_connectivity_new_unitsquare();
    connectivity_name = "unit";
  }

  return { connectivity, connectivity_name };

} // create_connectivity<2>

template <>
connectivity_data_t<3>
create_connectivity<3>(simple_config_t config, int brick_dim[3])
{

  kalypsso::p4est::Wrapper<3>::connectivity_t * connectivity = nullptr;
  std::string                                   connectivity_name = "unknown";

  if (config == P8EST_CONFIG_UNIT)
  {
    connectivity = p8est_connectivity_new_unitcube();
    connectivity_name = "unit";
  }
  else if (config == P8EST_CONFIG_PERIODIC)
  {
    connectivity = p8est_connectivity_new_periodic();
    connectivity_name = "periodic";
  }
  else if (config == P8EST_CONFIG_BRICK)
  {
    connectivity = p8est_connectivity_new_brick(brick_dim[0], brick_dim[1], brick_dim[2], 1, 1, 1);
    connectivity_name = "brick";
  }
  else if (config == P8EST_CONFIG_ROTWRAP)
  {
    connectivity = p8est_connectivity_new_rotwrap();
    connectivity_name = "rotwrap";
  }
  else if (config == P8EST_CONFIG_TWOCUBES)
  {
    connectivity = p8est_connectivity_new_twocubes();
    connectivity_name = "twocubes";
  }
  else if (config == P8EST_CONFIG_TWOWRAP)
  {
    connectivity = p8est_connectivity_new_twowrap();
    connectivity_name = "twowrap";
  }
  else if (config == P8EST_CONFIG_ROTCUBES)
  {
    connectivity = p8est_connectivity_new_rotcubes();
    connectivity_name = "rotcubes";
  }
  else if (config == P8EST_CONFIG_SHELL)
  {
    connectivity = p8est_connectivity_new_shell();
    connectivity_name = "shell";
  }
  else if (config == P8EST_CONFIG_SPHERE)
  {
    connectivity = p8est_connectivity_new_sphere();
    connectivity_name = "sphere";
  }
  else if (config == P8EST_CONFIG_TORUS)
  {
    connectivity = p8est_connectivity_new_torus(8);
    connectivity_name = "torus";
  }

  return { connectivity, connectivity_name };

} // create_connectivity<3>

/* ================================================================= */
/* ================================================================= */
/* ================================================================= */
template <int dim>
typename kalypsso::p4est::Wrapper<dim>::geometry_t *
create_geometry(simple_config_t                                          config,
                typename kalypsso::p4est::Wrapper<dim>::connectivity_t * connectivity,
                double                                                   R0,
                double                                                   R1);

template <>
kalypsso::p4est::Wrapper<2>::geometry_t *
create_geometry<2>(simple_config_t                               config,
                   kalypsso::p4est::Wrapper<2>::connectivity_t * connectivity,
                   double                                        R0,
                   double                                        R1)
{

  kalypsso::p4est::Wrapper<2>::geometry_t * geom = nullptr;

  if (config == P4EST_CONFIG_ICOSAHEDRON)
  {
    geom = p4est_geometry_new_icosahedron(connectivity, R0);
  }
  else if (config == P4EST_CONFIG_DISK2D)
  {
    geom = p4est_geometry_new_disk2d(connectivity, R0, R1);
  }
  else if (config == P4EST_CONFIG_SHELL2D)
  {
    geom = p4est_geometry_new_shell2d(connectivity, R0, R1);
  }

  return geom;

} // create_geometry

template <>
kalypsso::p4est::Wrapper<3>::geometry_t *
create_geometry<3>(simple_config_t                               config,
                   kalypsso::p4est::Wrapper<3>::connectivity_t * connectivity,
                   double                                        R0,
                   double                                        R1)
{

  kalypsso::p4est::Wrapper<3>::geometry_t * geom = nullptr;

  if (config == P8EST_CONFIG_SHELL)
  {
    geom = p8est_geometry_new_shell(connectivity, 1., .44);
  }
  else if (config == P8EST_CONFIG_SPHERE)
  {
    geom = p8est_geometry_new_sphere(connectivity, 1., 0.7, 0.5);
  }
  else if (config == P8EST_CONFIG_TORUS)
  {
    geom = p8est_geometry_new_torus(connectivity, 0.44, 1.0, 3.0);
  }

  return geom;
} // create_geometry

/* ============================================================ */
/* ============================================================ */
/* ============================================================ */
template <int dim, typename device_t>
void
run_test(simple_config_t config, const kalypsso::ParallelEnv & par_env, int argc, char * argv[])
{

  if (dim == 2 and par_env.rank() == 0)
    printf("Running a 2D test\n");
  if (dim == 3 and par_env.rank() == 0)
    printf("Running a 3D test\n");

  using namespace kalypsso::p4est;

  typename Wrapper<dim>::forest_t *   forest;
  typename Wrapper<dim>::geometry_t * geom = nullptr;
  typename Wrapper<dim>::refine_cb_t  refine_fn;
  typename Wrapper<dim>::coarsen_cb_t coarsen_fn;

  // only meaningful when using brick connectivity
  int brick_dim[3];
  brick_dim[0] = argc > 3 ? atoi(argv[3]) : 2;
  brick_dim[1] = argc > 4 ? atoi(argv[4]) : 3;
  brick_dim[2] = argc > 5 ? atoi(argv[5]) : 4;

  // create connectivity and forest structures
  auto [connectivity, connectivity_name] = create_connectivity<dim>(config, brick_dim);

  // create geometry
  double R0 = 0.44;
  double R1 = 1.0;
  if (argc >= 4)
    R0 = atof(argv[3]);
  if (argc >= 5)
    R1 = atof(argv[4]);
  geom = create_geometry<dim>(config, connectivity, R0, R1);

  // assign refine_fn, coarsen_fn
  refine_fn = refine_normal_fn<dim>;
  coarsen_fn = nullptr;

  if (config == P4EST_CONFIG_EVIL)
  {
    refine_fn = refine_evil_fn<dim>;
    coarsen_fn = coarsen_evil_fn<dim>;
  }
  else if (config == P4EST_CONFIG_EVIL3)
  {
    refine_fn = refine_evil3_fn<dim>;
    coarsen_fn = nullptr;
  }
  else if (config == P4EST_CONFIG_DISK2D)
  {
    refine_fn = refine_radius_fn<dim>;
    coarsen_fn = nullptr;
  }
  else if (config == P4EST_CONFIG_ICOSAHEDRON)
  {
    refine_fn = refine_icosahedron_fn<dim>;
    coarsen_fn = nullptr;
  }
  else if (config == P4EST_CONFIG_SHELL2D)
  {
    refine_fn = refine_radius_fn<dim>;
    coarsen_fn = nullptr;
  }
  else if (config == P8EST_CONFIG_TWOCUBES)
  {
    refine_fn = refine_sparse_fn<dim>;
  }
  else if (config == P8EST_CONFIG_TWOWRAP)
  {
    refine_fn = refine_sparse_fn<dim>;
  }
  else
  {
    refine_fn = refine_normal_fn<dim>;
    coarsen_fn = nullptr;
  }

  /*
   * set geometry as p4est user pointer so that we can
   * retrieve it in the callbacks.
   */
  forest = Wrapper<dim>::new_forest(
#ifdef KALYPSSO_CORE_USE_MPI
    par_env.mpi_comm(),
#else
    sc_MPI_COMM_WORLD,
#endif // KALYPSSO_CORE_USE_MPI
    connectivity,
    dim == 2 ? 4 : 4,
    0,
    0,
    sizeof(user_data_t),
    init_fn<dim>,
    geom);

  double scale = 1.0;

#if defined(KALYPSSO_IO_HDF5_BLOCK)
  my_p4est_write_file<dim, device_t>(
    par_env, forest, connectivity_name, brick_dim, geom, "test_new_block", scale);
#elif defined(KALYPSSO_IO_HDF5_GHOSTED_BLOCK)
  my_p4est_write_file<dim, device_t>(
    par_env, forest, connectivity_name, brick_dim, geom, "test_new_block_ghosted", scale);
#else
  my_p4est_write_file<dim, device_t>(
    par_env, forest, connectivity_name, brick_dim, geom, "test_new", scale);
#endif

  // refinement and coarsening
  Wrapper<dim>::refine(forest, 1, refine_fn, init_fn<dim>);
  if (coarsen_fn != nullptr)
  {
    Wrapper<dim>::coarsen(forest, 1, coarsen_fn, init_fn<dim>);
  }
#if defined(KALYPSSO_IO_HDF5_BLOCK)
  my_p4est_write_file<dim, device_t>(
    par_env, forest, connectivity_name, brick_dim, geom, "test_refined_block", scale);
#elif defined(KALYPSSO_IO_HDF5_GHOSTED_BLOCK)
  my_p4est_write_file<dim, device_t>(
    par_env, forest, connectivity_name, brick_dim, geom, "test_refined_block_ghosted", scale);
#else
  my_p4est_write_file<dim, device_t>(
    par_env, forest, connectivity_name, brick_dim, geom, "test_refined", scale);
#endif

  // balance
  Wrapper<dim>::balance(forest, Wrapper<dim>::CONNECT_FULL, init_fn<dim>);
#if defined(KALYPSSO_IO_HDF5_BLOCK)
  my_p4est_write_file<dim, device_t>(
    par_env, forest, connectivity_name, brick_dim, geom, "test_balanced_block", scale);
#elif defined(KALYPSSO_IO_HDF5_GHOSTED_BLOCK)
  my_p4est_write_file<dim, device_t>(
    par_env, forest, connectivity_name, brick_dim, geom, "test_balanced_block_ghosted", scale);
#else
  my_p4est_write_file<dim, device_t>(
    par_env, forest, connectivity_name, brick_dim, geom, "test_balanced", scale);
#endif

  [[maybe_unused]] auto crc = Wrapper<dim>::checksum(forest);

  // partition
  Wrapper<dim>::partition(forest, 0, nullptr);
#if defined(KALYPSSO_IO_HDF5_BLOCK)
  my_p4est_write_file<dim, device_t>(
    par_env, forest, connectivity_name, brick_dim, geom, "test_partition_block", scale);
#elif defined(KALYPSSO_IO_HDF5_GHOSTED_BLOCK)
  my_p4est_write_file<dim, device_t>(
    par_env, forest, connectivity_name, brick_dim, geom, "test_partition_block_ghosted", scale);
#else
  my_p4est_write_file<dim, device_t>(
    par_env, forest, connectivity_name, brick_dim, geom, "test_partition", scale);
#endif

#ifdef P4EST_ENABLE_DEBUG
  // rebalance should not change checksum
  Wrapper<dim>::balance(forest, Wrapper<dim>::CONNECT_FULL, init_fn<dim>);
  P4EST_ASSERT(Wrapper<dim>::checksum(forest) == crc);
#endif

  // destroy the p4est and its connectivity structure
  Wrapper<dim>::destroy(forest);
  if (geom != nullptr)
  {
    Wrapper<dim>::geometry_destroy(geom);
  }
  Wrapper<dim>::connectivity_destroy(connectivity);

} // run_test

// ======================================================
// ======================================================
// ======================================================
int
main(int argc, char ** argv)
{

  simple_config_t config;
  int             wrongusage;
  const char *    usage;

  {
    kalypsso::ParallelEnv par_env(argc, argv);

    // Process command line arguments
    usage =
      "Arguments: <configuration> <level>\n"
      "   2D configuration can be any of\n"
      "      unit|three|brick|evil|evil3|pillow|moebius|\n"
      "      star|cubed|disk|periodic|rotwrap|icosahedron|disk2d|shell2d\n"
      "   3D configuration can be any of\n"
      "      unit3|periodic3|brick3|rotwrap3|twocubes3|twowrap3|rotcubes3|shell3|sphere3|torus\n"
      "   Level controls the maximum depth of refinement\n"
      "\n"
      "Example run:\n"
      "  mpirun -np 3 ./test_io_hdf5_block unit 6\n"
      "  mpirun -np 6 ./test_io_hdf5_block brick 5 5 3\n"
      "  mpirun -np 3 ./test_io_hdf5_ghosted_block unit 6\n"
      "  mpirun -np 6 ./test_io_hdf5_ghosted_block brick 5 5 3\n";
    wrongusage = 0;
    config = P4EST_CONFIG_NULL;
    if (!wrongusage && argc < 3)
    {
      wrongusage = 1;
    }
    if (!wrongusage)
    {
      if (!strcmp(argv[1], "unit"))
      {
        config = P4EST_CONFIG_UNIT;
      }
      else if (!strcmp(argv[1], "three"))
      {
        config = P4EST_CONFIG_THREE;
      }
      else if (!strcmp(argv[1], "brick"))
      {
        config = P4EST_CONFIG_BRICK;
      }
      else if (!strcmp(argv[1], "evil"))
      {
        config = P4EST_CONFIG_EVIL;
      }
      else if (!strcmp(argv[1], "evil3"))
      {
        config = P4EST_CONFIG_EVIL3;
      }
      else if (!strcmp(argv[1], "pillow"))
      {
        config = P4EST_CONFIG_PILLOW;
      }
      else if (!strcmp(argv[1], "moebius"))
      {
        config = P4EST_CONFIG_MOEBIUS;
      }
      else if (!strcmp(argv[1], "star"))
      {
        config = P4EST_CONFIG_STAR;
      }
      else if (!strcmp(argv[1], "cubed"))
      {
        config = P4EST_CONFIG_CUBED;
      }
      else if (!strcmp(argv[1], "disk"))
      {
        config = P4EST_CONFIG_DISK;
      }
      else if (!strcmp(argv[1], "periodic"))
      {
        config = P4EST_CONFIG_PERIODIC;
      }
      else if (!strcmp(argv[1], "rotwrap"))
      {
        config = P4EST_CONFIG_ROTWRAP;
      }
      else if (!strcmp(argv[1], "icosahedron"))
      {
        config = P4EST_CONFIG_ICOSAHEDRON;
      }
      else if (!strcmp(argv[1], "disk2d"))
      {
        config = P4EST_CONFIG_DISK2D;
      }
      else if (!strcmp(argv[1], "shell2d"))
      {
        config = P4EST_CONFIG_SHELL2D;
      }
      // 3D config
      else if (!strcmp(argv[1], "unit3"))
      {
        config = P8EST_CONFIG_UNIT;
      }
      else if (!strcmp(argv[1], "periodic3"))
      {
        config = P8EST_CONFIG_PERIODIC;
      }
      else if (!strcmp(argv[1], "brick3"))
      {
        config = P8EST_CONFIG_BRICK;
      }
      else if (!strcmp(argv[1], "rotwrap3"))
      {
        config = P8EST_CONFIG_ROTWRAP;
      }
      else if (!strcmp(argv[1], "twocubes3"))
      {
        config = P8EST_CONFIG_TWOCUBES;
      }
      else if (!strcmp(argv[1], "twowrap3"))
      {
        config = P8EST_CONFIG_TWOWRAP;
      }
      else if (!strcmp(argv[1], "rotcubes3"))
      {
        config = P8EST_CONFIG_ROTCUBES;
      }
      else if (!strcmp(argv[1], "shell3"))
      {
        config = P8EST_CONFIG_SHELL;
      }
      else if (!strcmp(argv[1], "sphere3"))
      {
        config = P8EST_CONFIG_SPHERE;
      }
      else if (!strcmp(argv[1], "torus"))
      {
        config = P8EST_CONFIG_TORUS;
      }
      else
      {
        wrongusage = 1;
      }
    }
    if (wrongusage)
    {
      P4EST_GLOBAL_LERROR(usage);
      sc_abort_collective("Usage error");
    }

    // assign variables based on configuration
    refine_level = atoi(argv[2]);

    // run some tests
    if (config <= P4EST_CONFIG_SHELL2D)
      run_test<2, kalypsso::DefaultDevice>(config, par_env, argc, argv);
    else
      run_test<3, kalypsso::DefaultDevice>(config, par_env, argc, argv);
  }

  return EXIT_SUCCESS;
}
