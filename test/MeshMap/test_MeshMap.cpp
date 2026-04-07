// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file test_MeshMap.cpp
 *
 * purpose: just illustrate usage of class MeshMap, especially method create_orchard_keys_view or
 * fill_map
 */

#include <kalypsso/core/kalypsso_core_config.h> // for KALYPSSO_CORE_USE_HDF5, ...
#include <kalypsso/core/MeshMap.h>

#include <kalypsso/core/AMRmesh.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/cmdline_utils.h>

#ifdef KALYPSSO_CORE_USE_HDF5
#  include <kalypsso/core/HDF5_Xdmf_Writer.h>
#endif

#include <kalypsso/core/scan_utils.h>

#include <kalypsso/utils/p4est/p4est_wrapper.h>
#include <kalypsso/utils/p4est/connectivity.h>
#include <kalypsso/utils/p4est/geometry.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <memory>
#include <cstdlib>

namespace kalypsso
{

static int refine_level = 0;

// =============================================================================
// =============================================================================
template <size_t dim>
static int
refine_normal_fn(typename p4est::Wrapper<dim>::forest_t *   forest,
                 typename p4est::topidx_t                   which_tree,
                 typename p4est::Wrapper<dim>::quadrant_t * quadrant)
{
  using Wrapper = typename p4est::Wrapper<dim>;

  using p4est_userdata_t = typename AMRmesh<dim>::p4est_userdata_t;

  p4est_userdata_t * p4est_userdata = static_cast<p4est_userdata_t *>(forest->user_pointer);
  [[maybe_unused]] const auto level_min = p4est_userdata->level_min;
  const auto                  level_max = p4est_userdata->level_max;

  // when reaching max level, do not refine anymore
  if (quadrant->level == level_max)
    return 0;

  if (which_tree == 0 and static_cast<int>(quadrant->level) < 4)
  {
    double x = (static_cast<double>(p4est::get_x(quadrant))) / Wrapper::ROOT_LEN;
    double y = (static_cast<double>(p4est::get_y(quadrant))) / Wrapper::ROOT_LEN;
    if (x > 0.25 and x < 0.35 and y > 0.85 and y < 0.95)
      return 1;
  }

  if (which_tree == 1 and static_cast<int>(quadrant->level) < 4)
  {
    double x = (static_cast<double>(p4est::get_x(quadrant))) / Wrapper::ROOT_LEN;
    double y = (static_cast<double>(p4est::get_y(quadrant))) / Wrapper::ROOT_LEN;
    if (x > 0.45 and x < 0.55 and y > 0.45 and y < 0.55)
      return 1;
  }

  if (static_cast<int>(quadrant->level) >= (kalypsso::refine_level - (which_tree % 3)))
  {
    return 0;
  }
  if (quadrant->level == 1 && Wrapper::quadrant_child_id(quadrant) == 3)
  {
    return 1;
  }
  if (quadrant->x == P4EST_LAST_OFFSET(2) && quadrant->y == P4EST_LAST_OFFSET(2))
  {
    return 1;
  }

  if constexpr (dim == 2)
  {
    if (p4est::get_x(quadrant) >= static_cast<int>(Wrapper::QUADRANT_LEN(2)))
    {
      return 0;
    }
  }
  else
  {
    if (p4est::get_z(quadrant) >= static_cast<int>(Wrapper::QUADRANT_LEN(2)))
    {
      return 0;
    }
  }

  return 1;

} // refine_normal_fn

// =============================================================================
// =============================================================================
template <size_t dim>
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

  qcoord_t half_length = static_cast<qcoord_t>(quad_len) / 2;

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

// =============================================================================
// =============================================================================
template <size_t dim, typename device_t>
auto
setup_initial_data_leaf(typename MeshMap<dim, device_t>::orchard_key_view_t orchard_keys_device,
                        AMRMeshInfo                                         amr_mesh_info,
                        ConfigMap const & config_map) -> DataArrayLeaf<real_t, device_t>
{
  const auto brick_sizes = get_brick_sizes<dim>(config_map);

  using DataArrayLeaf_t = DataArrayLeaf<real_t, device_t>;

  //
  // create some test user data (on leaf, not cells for simplicity), and upload to device
  //
  DataArrayLeaf_t userdataLeaf(Kokkos::view_alloc(Kokkos::WithoutInitializing, "dummy_data"),
                               static_cast<size_t>(amr_mesh_info.local_num_quadrants_total()),
                               1);

  {
    using exec_space = typename device_t::execution_space;

    // get geometrical scaling factor
    const auto scaling_factor = get_scaling_factor(config_map);

    // get domain lower left corner
    const auto xyz_min = get_xyz_min<dim>(config_map);

    Kokkos::parallel_for(
      "Fill dummy data",
      Kokkos::RangePolicy<exec_space>(0, amr_mesh_info.local_num_quadrants_total()),
      KOKKOS_LAMBDA(const int32_t iOct) {
        constexpr bool use_center = false;

        const auto key = orchard_keys_device(iOct);

        auto xyz_vertex = orchard_key_to_vertex_coord<dim>(key, use_center);
        if (iOct >= amr_mesh_info.local_num_quadrants() + amr_mesh_info.local_num_ghosts() and
            iOct < amr_mesh_info.local_num_quadrants() + amr_mesh_info.local_num_ghosts() +
                     amr_mesh_info.local_num_quadrants_outside())
        {
          xyz_vertex = outside_key_to_vertex_coord<dim>(key, false, brick_sizes);
        }
        const auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, scaling_factor, xyz_min);

      // the only reason of the following dummy code to be here, is that cuda nvcc compile
      // doesn't support capturing variables inside the inner lambda inside a constexpr if
      //
      // strangely, nvc++ is ok and don't need that
      // TODO: remove theses lines when nvcc will have support for this.
#ifdef __NVCC__
        [[maybe_unused]] int dummy = 0;
        if (userdataLeaf.extent(0) == 0)
          dummy++;
#endif
        if constexpr (dim == 2)
          userdataLeaf(iOct, 0) = xyz[IX] + xyz[IY];
        else if constexpr (dim == 3)
          userdataLeaf(iOct, 0) = xyz[IX] + xyz[IY] + xyz[IZ];
      });
  }

  return userdataLeaf;

} // setup_initial_data_leaf

// =============================================================================
// =============================================================================
template <size_t dim, typename device_t>
auto
setup_initial_data_block(typename MeshMap<dim, device_t>::orchard_key_view_t orchard_keys_device,
                         AMRMeshInfo                                         amr_mesh_info,
                         ConfigMap const & config_map) -> DataArrayBlock<dim, real_t, device_t>
{

  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using exec_space = typename device_t::execution_space;

  const auto brick_sizes = get_brick_sizes<dim>(config_map);

  //
  // create some data
  //

  // get block sizes
  const auto bx = config_map.getInteger("amr", "bx", 1);
  const auto by = config_map.getInteger("amr", "by", 1);
  const auto bz = config_map.getInteger("amr", "bz", 1);

  const auto block_sizes = [=]() {
    if constexpr (dim == 2)
      return kalypsso::block_size_t<2>{ bx, by };
    if constexpr (dim == 3)
      return kalypsso::block_size_t<3>{ bx, by, bz };
  }();

  //
  // create some test user data (on cells), and upload to device
  //
  auto userdataBlock =
    DataArrayBlock_t("test_data_block", block_sizes, 1, amr_mesh_info.local_num_quadrants_total());

  {
    // number of quadrants in current MPI process
    const int32_t nbOcts = amr_mesh_info.local_num_quadrants_total();

    // Kokkos team policy type alias
    using team_policy_t = Kokkos::TeamPolicy<exec_space, Kokkos::IndexType<int32_t>>;

    team_policy_t policy(nbOcts,
                         Kokkos::AUTO() /* number of threads per team is chosen by kokkos */);
    using thread_t = typename team_policy_t::member_type;

    // get geometrical scaling factor
    const auto scaling_factor = get_scaling_factor(config_map);

    // get domain lower left corner
    const auto xyz_min = get_xyz_min<dim>(config_map);

    Kokkos::parallel_for(
      "Fill_test_data_block", policy, KOKKOS_LAMBDA(const thread_t & member) {
        // block sizes (bx,by,bz) are captured by the lambda

        // get lower left corner real space coordinates
        constexpr bool use_center = false;

        // number of cells per octant
        const int32_t nbCells = dim == 2 ? bx * by : bx * by * bz;

        // the first octant to process is indexed by the team id
        int32_t iOct = member.league_rank();

        // get orchard key of current block/octant
        const auto key = orchard_keys_device(iOct);

        // get octant level
        const auto level = orchard_key_t<dim>::level(key);

        // compute cell length in real space
        // const double dx_cell = compute_cell_length<dim>(level, bx);

        // get real space coordinates of lower left corner of the block
        auto xyz_corner_vertex = orchard_key_to_vertex_coord<dim>(key, use_center);

        if (iOct >= amr_mesh_info.local_num_quadrants() + amr_mesh_info.local_num_ghosts() and
            iOct < amr_mesh_info.local_num_quadrants() + amr_mesh_info.local_num_ghosts() +
                     amr_mesh_info.local_num_quadrants_outside())
        {
          xyz_corner_vertex = outside_key_to_vertex_coord<dim>(key, false, brick_sizes);
        }

      // the only reason of the following dummy code to be here, is that cuda nvcc compile
      // doesn't support capturing variables inside the inner lambda inside a constexpr if
      //
      // strangely, nvc++ is ok and don't need that
      // TODO: remove theses lines when nvcc will have support for this.
#ifdef __NVCC__
        [[maybe_unused]] int dummy = 0;
        if (userdataBlock.num_cells() == 0)
          dummy++;
#endif

        // initialize cell id
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, nbCells), [=](const int32_t icell) {
          // compute ix,iy,iz of local cell inside
          // block from index
          auto iCoord = icell_to_icoord<dim>(icell, bx);
          auto xyz_cell_vertex =
            compute_cell_coordinates<dim>(level, xyz_corner_vertex, iCoord, block_sizes);

          const auto xyz_cell =
            vertex_coord_to_real_space<dim>(xyz_cell_vertex, scaling_factor, xyz_min);

          if constexpr (dim == 2)
            userdataBlock(icell, 0, iOct) = xyz_cell[IX] + xyz_cell[IY];
          if constexpr (dim == 3)
            userdataBlock(icell, 0, iOct) = xyz_cell[IX] + xyz_cell[IY] + xyz_cell[IZ];
        }); // end TeamVectorRange
      });
  }

  return userdataBlock;

} // setup_initial_data_block

// =============================================================================
// =============================================================================
template <size_t dim>
static void
userdata_init_fn(typename kalypsso::p4est::Wrapper<dim>::forest_t *   p4est,
                 typename kalypsso::p4est::topidx_t                   which_tree,
                 typename kalypsso::p4est::Wrapper<dim>::quadrant_t * quadrant)
{

  using namespace kalypsso::p4est;

  using connectivity_t = typename Wrapper<dim>::connectivity_t;
  using geometry_t = typename Wrapper<dim>::geometry_t;

  // using udata_t = typename AMRmesh<dim>::user_data_t;
  // udata_t * udata = (udata_t *)quadrant->p.user_data;

  // compute physical space coordinates, and put "x" into user_data

  connectivity_t * conn = static_cast<connectivity_t *>(p4est->connectivity);

  using p4est_userdata_t = typename AMRmesh<dim>::p4est_userdata_t;
  p4est_userdata_t * p4est_userdata = static_cast<p4est_userdata_t *>(p4est->user_pointer);

  geometry_t * geom = p4est_userdata->geom;

  double XYZ[3];

  quadrant_center_vertex<dim>(conn, geom, which_tree, quadrant, XYZ);

  // initialize user data for current quandrant
  // udata->x = XYZ[0];

} // userdata_init_fn

// =============================================================================
// =============================================================================
template <size_t dim, typename device_t>
void
run_test(const ParallelEnv & par_env, int argc, char * argv[])
{
  using namespace p4est;
#ifdef KALYPSSO_CORE_USE_HDF5
  // using DataArrayLeafHost_t = DataArrayLeafHost<real_t, device_t>;
  // using DataArrayBlockHost_t = DataArrayBlockHost<real_t, device_t>;
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
#endif

  // check if user passed a custom ini filename
  std::string config_filename = kalypsso::cmdline_get_string(argv, argv + argc, "--ini");

  // provide a default config filename (that exists)
  if (config_filename.size() == 0)
    config_filename = dim == 2 ? "./test_MeshMap_brick_2d.ini" : "./test_MeshMap_brick_3d.ini";

  ConfigMap config_map = broadcast_parameters(config_filename);

  if (par_env.rank() == 0)
    printf("================================================\n");

  // create a p4est object (using brick connectivity)

  typename Wrapper<dim>::refine_cb_t  refine_fn = refine_normal_fn<dim>;
  typename Wrapper<dim>::coarsen_cb_t coarsen_fn = nullptr;
  // using balance_type_t = typename Wrapper<dim>::balance_type_t;

  AMRmesh<dim> amr_mesh(par_env, config_map);

  auto forest = amr_mesh.forest();
  // auto geom = amr_mesh.geometry();

  auto conn_name = amr_mesh.connectivity_name();
  auto geom_name = amr_mesh.geometry_name();

  // refinement and coarsening
  Wrapper<dim>::refine(forest, 1, refine_fn, userdata_init_fn<dim>);
  if (coarsen_fn != nullptr)
  {
    Wrapper<dim>::coarsen(forest, 1, coarsen_fn, userdata_init_fn<dim>);
  }

  if (par_env.rank() == 0)
  {
    printf("Running a %ld test with connectivity %s and geometry %s\n",
           dim,
           conn_name.c_str(),
           geom_name.c_str());
  }

  printf("Forest has %ld global octants, %d local octants on MPI proc %d.\n",
         forest->global_num_quadrants,
         forest->local_num_quadrants,
         par_env.rank());

  // balance
  Wrapper<dim>::balance(forest, Wrapper<dim>::CONNECT_FULL, userdata_init_fn<dim>);

  // partition
  Wrapper<dim>::partition(forest, 0, nullptr);

  // initialize ghost
  amr_mesh.reset_ghost();

  auto ghost = amr_mesh.ghost();

  auto mesh_map = std::make_shared<MeshMap<dim, device_t>>(config_map, par_env);

  mesh_map->compute_outside_quad_info(forest, ghost);
  mesh_map->update_hashmap_serial(forest, ghost);

  mesh_map->update_amr_mesh_info(forest, ghost);
  auto amr_mesh_info = mesh_map->get_amr_mesh_info();
  if (par_env.rank() == 0)
  {
    amr_mesh_info.print();
  }

  // retrieve amr keys
  mesh_map->update_orchard_keys(forest, ghost);
  auto orchard_keys_host = mesh_map->orchard_keys_host();
  auto orchard_keys_device = mesh_map->orchard_keys_clone();

  auto conformal_full_status = mesh_map->update_conformal_full_status();
  auto conformal_full_status_h =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, conformal_full_status);


  if (par_env.rank() == 0)
  {
    for (int qid = 0; qid < forest->local_num_quadrants; ++qid)
    {
      const auto status = conformal_full_status_h(qid);
      printf("[rank = %d] orchard_key[%d] = %zu conformal_full_status=%ld (face %d %d %d %d edge "
             "%d %d %d %d)\n",
             par_env.rank(),
             qid,
             orchard_keys_host(qid),
             static_cast<uint64_t>(status),
             conformal_full_status_t<dim>::face_xmin(status),
             conformal_full_status_t<dim>::face_xmax(status),
             conformal_full_status_t<dim>::face_ymin(status),
             conformal_full_status_t<dim>::face_ymax(status),
             conformal_full_status_t<dim>::edge_xmin_ymin(status),
             conformal_full_status_t<dim>::edge_xmax_ymin(status),
             conformal_full_status_t<dim>::edge_xmin_ymax(status),
             conformal_full_status_t<dim>::edge_xmax_ymax(status));
    }
  }

  // testing scan_utils, example parameters
  // level_min = 1
  // level_max = 4
  // mpirun -np 1 ./test_MeshMap --ini test_MeshMap_brick_2d.ini --refine_level 3
  {
    uint8_t level_min = static_cast<uint8_t>(config_map.getInteger("amr", "level_min", 4));
    uint8_t level_max = static_cast<uint8_t>(config_map.getInteger("amr", "level_max", 4));
    auto    level_indexes =
      compute_index_by_level<device_t, dim>(orchard_keys_device,
                                            level_min,
                                            level_max,
                                            mesh_map->get_amr_mesh_info().local_num_quadrants(),
                                            par_env);
    auto level_indexes_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, level_indexes);

    for (size_t ikey = 0; ikey < orchard_keys_device.size(); ++ikey)
    {
      printf("i=%lu key=%lu level=%d index=%lu\n",
             ikey,
             orchard_keys_host(ikey),
             orchard_key_t<dim>::level(orchard_keys_host(ikey)),
             level_indexes_host(ikey));
    }
  }

#ifdef KALYPSSO_CORE_USE_HDF5
  // testing IO
  {
    bool use_outside_quads = false;
    if (kalypsso::cmdline_arg_exists(argv, argv + argc, "--outside"))
      use_outside_quads = true;

    // first create dummy data
    auto userdataLeaf_d =
      setup_initial_data_leaf<dim, device_t>(orchard_keys_device, amr_mesh_info, config_map);
    auto userdataLeaf_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, userdataLeaf_d);

    auto userdataBlock_d =
      setup_initial_data_block<dim, device_t>(orchard_keys_device, amr_mesh_info, config_map);
    auto userdataBlock_h = DataArrayBlock_t::create_host_mirror_view_and_copy(userdataBlock_d);

    // write hdf5/xmdf
    HDF5_Xdmf_Writer<dim, device_t> writer(par_env, config_map, mesh_map);

    std::string outputDir = config_map.getString("output", "outputDir", "./");

    std::string filename_cell = "test_MeshMap_cell";
    std::string filename_leaf = "test_MeshMap_leaf";

    if (use_outside_quads)
    {
      filename_cell = filename_cell + "_outside";
      filename_leaf = filename_leaf + "_outside";
    }

    // block mode
    writer.set_block_mode();
    writer.use_outside_quads(use_outside_quads);
    writer.set_write_mesh_info(true);
    writer.open(filename_cell, outputDir);
    writer.write_header(0.0);
    writer.write_amr_metadata(orchard_keys_host);
    {
      // write dummy test data
      if (use_outside_quads)
      {
        writer.write_quadrant_attribute(userdataBlock_h,
                                        0,
                                        "dummy",
                                        amr_mesh_info.local_num_quadrants() +
                                          amr_mesh_info.local_num_ghosts(),
                                        amr_mesh_info.local_num_quadrants_outside());
      }
      else
      {
        writer.write_quadrant_attribute(
          userdataBlock_h, 0, "dummy", 0, amr_mesh_info.local_num_quadrants());
      }
    }
    writer.write_footer();
    writer.close();

    // leaf mode
    writer.set_leaf_mode();
    writer.use_outside_quads(use_outside_quads);
    writer.set_write_mesh_info(true);
    writer.open(filename_leaf, outputDir);
    writer.write_header(0.0);
    writer.write_amr_metadata(orchard_keys_host);
    {
      // write dummy test data
      if (use_outside_quads)
      {
        // auto userdataLeaf_outside_h = Kokkos::subview(
        //   userdataLeaf_h,
        //   std::make_pair(amr_mesh_info.local_num_quadrants + amr_mesh_info.local_num_ghosts,
        //                  amr_mesh_info.local_num_quadrants + amr_mesh_info.local_num_ghosts +
        //                    amr_mesh_info.local_num_quadrants_outside),
        //   Kokkos::ALL);

        // extract slice as a contiguous memory buffer
        using DataArrayLeaf_t = DataArrayLeaf<real_t, device_t>;
        using DataArrayLeafHost_t = typename DataArrayLeaf_t::host_mirror_type;
        DataArrayLeafHost_t userdataLeaf_outside_h(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "dummy_data_outside"),
          static_cast<size_t>(amr_mesh_info.local_num_quadrants_outside()),
          1);

        const auto offset = amr_mesh_info.local_num_quadrants() + amr_mesh_info.local_num_ghosts();
        Kokkos::parallel_for(
          "extract outside data",
          Kokkos::RangePolicy<Kokkos::OpenMP>(0, amr_mesh_info.local_num_quadrants_outside()),
          KOKKOS_LAMBDA(int32_t iOct) {
            userdataLeaf_outside_h(iOct, 0) = userdataLeaf_h(iOct + offset, 0);
          });

        writer.write_quadrant_attribute(userdataLeaf_h, 0, "dummy");
      }
      else
      {
        writer.write_quadrant_attribute(userdataLeaf_h, 0, "dummy");
      }
    }
    writer.write_footer();
    writer.close();
  }
#else
  std::cout << "HDF5 output unavailable.\n";
#endif

} // run_test

} // namespace kalypsso

// =============================================================================
// =============================================================================
// =============================================================================
int
main(int argc, char * argv[])
{

  {
    kalypsso::ParallelEnv par_env(argc, argv);

    if (kalypsso::cmdline_arg_exists(argv, argv + argc, "--help"))
    {
      if (par_env.rank() == 0)
      {
        // clang-format off
        std::cout << "Example cmdline: \"mpirun -np 4 ./test_MeshMap --ini test_MeshMap_brick_2d.ini --refine_level 6\"\n";
        // clang-format on
      }
      return 0;
    }

    // assign static global variables based on configuration
    if (kalypsso::cmdline_arg_exists(argv, argv + argc, "--refine_level"))
      kalypsso::refine_level = kalypsso::cmdline_get_integer(argv, argv + argc, "--refine_level");

    if (par_env.rank() == 0)
      printf("Using refine_level=%d\n", kalypsso::refine_level);

    // parse command line : 2d or 3d ?
    bool run_3d = kalypsso::cmdline_arg_exists(argv, argv + argc, "--3d");

    run_3d ? kalypsso::run_test<3, kalypsso::DefaultDevice>(par_env, argc, argv)
           : kalypsso::run_test<2, kalypsso::DefaultDevice>(par_env, argc, argv);
  }

  return 0;
} // main
