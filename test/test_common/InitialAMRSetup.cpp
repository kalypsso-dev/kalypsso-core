// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitialAMRSetup.cpp
 *
 * this is not a test, but we define a base class for multiple tests
 */
#include "InitialAMRSetup.h"

namespace kalypsso
{

int InitialAMRSetupBase::refine_level = 0;

// =============================================================================
// =============================================================================
template <size_t dim, typename device_t, typename Function>
InitialAMRSetup<dim, device_t, Function>::InitialAMRSetup(const ParallelEnv & par_env,
                                                          const ConfigMap &   config_map,
                                                          const Function      f)
  : m_par_env(par_env)
  , m_config_map(config_map)
  , m_amr_mesh(new AMRmesh<dim>(par_env, config_map))
  , m_mesh_map(new MeshMap<dim, device_t>(config_map, par_env))
  , m_f(f)
{

  if (InitialAMRSetupBase::refine_level == 0)
  {
    if (m_par_env.rank() == 0)
    {
      std::cout << "WARNING: InitialAMRSetupBase::refine was not set in main. Using default "
                   "value instead (i.e. 6)\n";
    }
    InitialAMRSetupBase::refine_level = 6;
  }

  auto conn_name = m_config_map.getString("amr", "connectivity", "invalid_connectivity");
  if (conn_name == "brick")
  {
    m_brick_sizes = get_brick_sizes<dim>(m_config_map);

    m_is_brick_periodic[IX] = static_cast<bool>(
      m_config_map.getInteger("p4est_connectivity", "periodic_x", CONNECTIVITY_PERIODIC_FALSE));
    m_is_brick_periodic[IY] = static_cast<bool>(
      m_config_map.getInteger("p4est_connectivity", "periodic_y", CONNECTIVITY_PERIODIC_FALSE));
    if constexpr (dim == 3)
      m_is_brick_periodic[IZ] = static_cast<bool>(
        m_config_map.getInteger("p4est_connectivity", "periodic_z", CONNECTIVITY_PERIODIC_FALSE));

    // get block sizes
    const auto bx = m_config_map.getInteger("amr", "bx", 1);
    const auto by = m_config_map.getInteger("amr", "by", 1);
    const auto bz = m_config_map.getInteger("amr", "bz", 1);

    m_block_sizes[IX] = bx;
    m_block_sizes[IY] = by;
    if constexpr (dim == 3)
      m_block_sizes[IZ] = bz;

    // get ghost block sizes
    const auto gx = m_config_map.getInteger("amr", "gx", 2);
    const auto gy = m_config_map.getInteger("amr", "gy", 2);
    const auto gz = m_config_map.getInteger("amr", "gz", 2);

    m_ghost_sizes[IX] = gx;
    m_ghost_sizes[IY] = gy;
    if constexpr (dim == 3)
      m_ghost_sizes[IZ] = gz;
    ;
  }
} // InitialAMRSetup

// =============================================================================
// =============================================================================
template <size_t dim, typename device_t, typename Function>
void
InitialAMRSetup<dim, device_t, Function>::setup_initial_mesh(bool norefine)
{

  auto forest = m_amr_mesh->forest();
  auto geom = m_amr_mesh->geometry();

  auto conn_name = m_config_map.getString("amr", "connectivity", "invalid_connectivity");
  assertm(conn_name == "brick", "This test requires p4est connectivity to be \"brick\" !");

  auto refine_type = m_config_map.getString("amr", "refine_type", "normal");

  [[maybe_unused]] auto geom_name = geom == nullptr ? "no_geometry" : geom->name;

  KALYPSSO_INFO(
    "Running a {}D test with connectivity {} and geometry {}", dim, conn_name.c_str(), geom_name);

  //
  // apply some simple initial refinement
  //
  {
    Kokkos::Profiling::ScopedRegion prof("initial_AMR_p4est");
    if (norefine)
      p4est::Wrapper<dim>::refine(forest, 1, norefine_fn<dim>, nullptr);
    else
    {
      if (refine_type == "normal")
        p4est::Wrapper<dim>::refine(forest, 1, refine_normal_fn<dim>, nullptr);
      else if (refine_type == "simple")
        p4est::Wrapper<dim>::refine(forest, 1, refine_simple_fn<dim>, nullptr);
    }
    p4est::Wrapper<dim>::coarsen(forest, 1, coarsen_normal_fn<dim>, nullptr);
    p4est::Wrapper<dim>::balance(forest, p4est::Wrapper<dim>::CONNECT_FULL, nullptr);
  }

  // init mesh ghost
  m_amr_mesh->reset_ghost();
  auto ghost = m_amr_mesh->ghost();

  KALYPSSO_INFO_ALL("before our own AMR cycle, local_num_quadrants = {}, local_num_ghosts = {}",
                    m_amr_mesh->local_num_quadrants(),
                    m_amr_mesh->local_num_ghosts());

  // update number of outside quad
  m_mesh_map->compute_outside_quad_info(forest, ghost);

  // update MeshMap orchard keys array
  m_mesh_map->update_orchard_keys(forest, ghost);

  // fill device hash map
  m_mesh_map->update_hashmap(forest, ghost);

  // compute number of quadrants per types (owned, ghost, outside, outside_ghost)
  m_mesh_map->update_amr_mesh_info(forest, ghost);

  KALYPSSO_INFO_ALL("Forest has {} global octants, {} local octants.",
                    m_mesh_map->get_amr_mesh_info().global_num_quadrants(),
                    m_mesh_map->get_amr_mesh_info().local_num_quadrants());

} // setup_initial_mesh

// =============================================================================
// =============================================================================
template <size_t dim, typename device_t, typename Function>
auto
InitialAMRSetup<dim, device_t, Function>::setup_initial_data_leaf(
  typename MeshMap<dim, device_t>::orchard_key_view_t orchard_keys_device) -> DataArrayLeaf_t
{

  //
  // create some data
  //

  // now generate some user data (block based AMR)
  HydroParams params = HydroParams(m_config_map);

  const int nbvar = nbvar_hydro<dim>();

  // get domain lower left corner
  const auto xyz_min = get_xyz_min<dim>(m_config_map);

  // get geometrical scaling factor
  const auto scaling_factor = get_scaling_factor(m_config_map);

  //
  // create some test user data (on leaf, not cells for simplicity), and upload to device
  //
  DataArrayLeaf_t userdataLeaf = DataArrayLeaf_t(
    "test_data",
    static_cast<size_t>(m_mesh_map->get_amr_mesh_info().local_num_quadrants_total()),
    static_cast<size_t>(nbvar));

  {
    Kokkos::parallel_for(
      "Fill_test_data",
      Kokkos::RangePolicy<exec_space>(0, m_mesh_map->get_amr_mesh_info().local_num_quadrants()),
      KOKKOS_LAMBDA(const int i) {
        constexpr bool use_center = false;
        const auto     key = orchard_keys_device(i);
        const auto     xyz_vertex = orchard_key_to_vertex_coord<dim>(key, use_center);
        const auto     xyz = vertex_coord_to_real_space<dim>(xyz_vertex, scaling_factor, xyz_min);
        userdataLeaf(i, 0) = xyz[IX] + xyz[IY];
        userdataLeaf(i, 1) = xyz[IX];
        userdataLeaf(i, 2) = xyz[IY];
        userdataLeaf(i, 3) = KALYPSSO_NUM(1.0) * static_cast<real_t>(i);
      });
  }

  return userdataLeaf;

} // setup_initial_data_leaf

// =============================================================================
// =============================================================================
template <size_t dim, typename device_t, typename Function>
auto
InitialAMRSetup<dim, device_t, Function>::setup_initial_data_block_new(
  typename MeshMap<dim, device_t>::orchard_key_view_t orchard_keys_device) -> DataArrayBlock_t
{
  //
  // create some data
  //

  // now generate some user data (block based AMR)
  HydroParams params = HydroParams(m_config_map);

  const int nbvar = nbvar_hydro<dim>();

  // get block sizes
  const auto bx = m_config_map.getInteger("amr", "bx", 1);
  const auto by = m_config_map.getInteger("amr", "by", 1);
  const auto bz = m_config_map.getInteger("amr", "bz", 1);

  const auto bSizes = [=]() {
    if constexpr (dim == 2)
      return block_size_t<2>{ bx, by };
    if constexpr (dim == 3)
      return block_size_t<3>{ bx, by, bz };
  }();


  // get domain lower left corner
  const auto xyz_min = get_xyz_min<dim>(m_config_map);

  // get geometrical scaling factor
  const auto scaling_factor = get_scaling_factor(m_config_map);

  //
  // create some test user data (on cells), and upload to device
  //
  auto userdataBlock =
    DataArrayBlock_t("test_data_block",
                     bSizes, // dim == 2 ? bx * by : bx * by * bz,
                     nbvar,
                     m_mesh_map->get_amr_mesh_info().local_num_quadrants_total());

  {
    // number of quadrants in current MPI process
    const auto nbOcts = m_mesh_map->get_amr_mesh_info().local_num_quadrants();

    // Kokkos team policy type alias
    using team_policy_t = Kokkos::TeamPolicy<exec_space, Kokkos::IndexType<int32_t>>;

    team_policy_t policy(nbOcts,
                         Kokkos::AUTO() /* number of threads per team is chosen by kokkos */);
    using thread_t = typename team_policy_t::member_type;

    auto block_sizes = m_block_sizes;
    auto f = m_f;

    Kokkos::parallel_for(
      "Fill_test_data_block", policy, KOKKOS_LAMBDA(const thread_t & member) {
        // block sizes (bx,by,bz) are captured by the lambda

        // get lower left corner real space coordinates
        constexpr bool use_center = false;

        // number of cells per octant
        const auto nbCells = dim == 2 ? bx * by : bx * by * bz;

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

      // the only reason of the following dummy code to be here, is that cuda nvcc compile
      // doesn't support capturing variables inside the inner lambda inside a constexpr if
      //
      // strangely, nvc++ is ok and don't need that
      // TODO: remove theses lines when nvcc will have support for this.
#ifdef __NVCC__
        [[maybe_unused]] int dummy = 0;
        if (f(0, 0, 0) != f(0, 0, 0) or userdataBlock(0, 0, 0) != userdataBlock(0, 0, 0))
          dummy++;
#endif

        // initialize cell id
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, nbCells), [=](const int32_t icell) {
          // compute ix,iy,iz of local cell inside
          // block from index
          auto       iCoord = icell_to_icoord<dim>(icell, bx);
          const auto xyz_cell_vertex =
            compute_cell_coordinates<dim>(level, xyz_corner_vertex, iCoord, block_sizes);
          const auto xyz_cell =
            vertex_coord_to_real_space<dim>(xyz_cell_vertex, scaling_factor, xyz_min);
          for (int ivar = 0; ivar < nbvar; ++ivar)
          {
            if constexpr (dim == 2)
              userdataBlock(icell, ivar, iOct) = f(xyz_cell[IX], xyz_cell[IY], ivar);
            if constexpr (dim == 3)
              userdataBlock(icell, ivar, iOct) = f(xyz_cell[IX], xyz_cell[IY], xyz_cell[IZ], ivar);
          }
        }); // end TeamVectorRange
      });
  }

  return userdataBlock;
} // setup_initial_data_block_new

// =============================================================================
// =============================================================================
template <size_t dim, typename device_t, typename Function>
auto
InitialAMRSetup<dim, device_t, Function>::setup_initial_data_block_flux(
  typename MeshMap<dim, device_t>::orchard_key_view_t orchard_keys_device,
  int                                                 direction) -> DataArrayBlock_t
{
  //
  // create some data
  //

  // now generate some user data (block based AMR)
  HydroParams params = HydroParams(m_config_map);

  const int nbvar = nbvar_hydro<dim>();

  // get block sizes
  const auto bx = m_config_map.getInteger("amr", "bx", 1);
  const auto by = m_config_map.getInteger("amr", "by", 1);
  const auto bz = m_config_map.getInteger("amr", "bz", 1);

  const auto bSizes = [=]() {
    if constexpr (dim == 2)
      return block_size_t<2>{ bx, by };
    if constexpr (dim == 3)
      return block_size_t<3>{ bx, by, bz };
  }();

  // flux size
  auto fSizes = bSizes;
  fSizes[direction]++;

  // get domain lower left corner
  const auto xyz_min = get_xyz_min<dim>(m_config_map);

  // get geometrical scaling factor
  const auto scaling_factor = get_scaling_factor(m_config_map);

  //
  // create some test user data (flux), and upload to device
  //
  auto userdata_flux = DataArrayBlock_t(
    "test_data_block", fSizes, nbvar, m_mesh_map->get_amr_mesh_info().local_num_quadrants_total());

  {
    // number of quadrants in current MPI process
    const auto nbOcts = m_mesh_map->get_amr_mesh_info().local_num_quadrants();

    const auto nbIters = nbOcts * userdata_flux.num_cells();

    // auto block_sizes = m_block_sizes;
    auto f = m_f;

    Kokkos::parallel_for(
      "Fill_test_data_block",
      Kokkos::RangePolicy<exec_space>(0, nbIters),
      KOKKOS_LAMBDA(const int32_t & global_index) {
        auto const iOct = global_index / userdata_flux.num_cells();
        auto const flux_index = global_index - iOct * userdata_flux.num_cells();

      // the only reason of the following dummy code to be here, is that cuda nvcc compile
      // doesn't support capturing variables inside the inner lambda inside a constexpr if
      //
      // strangely, nvc++ is ok and don't need that
      // TODO: remove theses lines when nvcc will have support for this.
#ifdef __NVCC__
        [[maybe_unused]] int dummy = 0;
        if (f(0, 0, 0) == f(0, 0, 0))
          dummy++;
#endif

        // get orchard key of current block/octant
        const auto key = orchard_keys_device(iOct);

        // compute ix,iy,iz of local cell inside
        // block from index
        const auto ijk = cell_index_unravel<dim>(flux_index, fSizes);
        const auto face_multiindex = to_face_multiindex<dim>(ijk, direction);

        const auto xyz_face = orchard_key_to_facecenter_real_space<dim>(
          key, face_multiindex, bSizes, scaling_factor, xyz_min);

        for (int ivar = 0; ivar < nbvar; ++ivar)
        {
          if constexpr (dim == 2)
          {
            userdata_flux(flux_index, ivar, iOct) = f(xyz_face[IX], xyz_face[IY], ivar);
          }
          else if constexpr (dim == 3)
          {
            userdata_flux(flux_index, ivar, iOct) =
              f(xyz_face[IX], xyz_face[IY], xyz_face[IZ], ivar);
          }
        }
      });
  }

  return userdata_flux;

} // setup_initial_data_block_flux

// =============================================================================
// =============================================================================
template <size_t dim, typename device_t, typename Function>
auto
InitialAMRSetup<dim, device_t, Function>::setup_initial_data_block_face(
  typename MeshMap<dim, device_t>::orchard_key_view_t orchard_keys_device,
  bool use_face_averaged_values) -> FaceDataArrayBlock_t
{
  //
  // create some data
  //

  // now generate some user data (block based AMR)
  HydroParams params = HydroParams(m_config_map);

  // get domain lower left corner
  const auto xyz_min = get_xyz_min<dim>(m_config_map);

  // get geometrical scaling factor
  const auto scaling_factor = get_scaling_factor(m_config_map);

  const auto num_octants = m_mesh_map->get_amr_mesh_info().local_num_quadrants_total();

  //
  // create some test user data (on cells), and upload to device
  //
  FaceDataArrayBlock_t userdataBlock =
    FaceDataArrayBlock_t("test_data_block", m_block_sizes, num_octants);

  const auto nbFacesPerLeaf = userdataBlock.num_elements_per_octant();

  const auto total_num_faces = num_octants * nbFacesPerLeaf;

  {
    // number of quadrants in current MPI process
    const auto block_sizes = m_block_sizes;
    auto       f = m_f;

    Kokkos::parallel_for(
      "Fill_test_data_block",
      Kokkos::RangePolicy<exec_space>(0, total_num_faces),
      KOKKOS_LAMBDA(const int32_t & global_index) {
        // block sizes (bx,by,bz) are captured by the lambda

        const auto iOct = global_index / nbFacesPerLeaf;
        const auto face_flat_index = global_index - iOct * nbFacesPerLeaf;

        const auto face_indexes = face_flat_index_unravel<dim>(
          face_flat_index, block_sizes, userdataBlock.offsets(), userdataBlock.shift());

        // get orchard key of current block/octant
        const auto key = orchard_keys_device(iOct);

        auto const & bSize = block_sizes[IX];

        // compute physical x,y,z for that face center
        const auto xyz = orchard_key_to_facecenter_real_space<dim>(
          key, face_indexes, bSize, scaling_factor, xyz_min);

        const auto level = orchard_key_t<dim>::level(key);

        // compute cell size
        const auto dx = compute_cell_length<dim>(level, block_sizes[IX]) * scaling_factor;

        [[maybe_unused]] const auto use_face_averaged_values_d = use_face_averaged_values;

      // the only reason of the following dummy code to be here, is that cuda nvcc compile
      // doesn't support capturing variables inside the inner lambda inside a constexpr if
      //
      // strangely, nvc++ is ok and don't need that
      // TODO: remove theses lines when nvcc will have support for this.
#ifdef __NVCC__
        [[maybe_unused]] int dummy = 0;
        if (f(0, 0, 0) != f(0, 0, 0))
          dummy++;
#endif

        if constexpr (dim == 2)
        {

          auto const & i = face_indexes[IX];
          auto const & j = face_indexes[IY];
          auto const & ivar = face_indexes[dim];

          if constexpr (f.has_face_averaged_values)
          {
            if (use_face_averaged_values_d)
              userdataBlock(i, j, ivar, iOct) = f.faverage(xyz[IX], xyz[IY], dx, dx, ivar);
            else
              userdataBlock(i, j, ivar, iOct) = f(xyz[IX], xyz[IY], ivar);
          }
          else
          {
            userdataBlock(i, j, ivar, iOct) = f(xyz[IX], xyz[IY], ivar);
          }
        }
        else if constexpr (dim == 3)
        {
          auto const & i = face_indexes[IX];
          auto const & j = face_indexes[IY];
          auto const & k = face_indexes[IZ];
          auto const & ivar = face_indexes[dim];

          userdataBlock(i, j, k, ivar, iOct) = f(xyz[IX], xyz[IY], xyz[IZ], ivar);
        }
      });
  }

  return userdataBlock;
} // setup_initial_data_block_face

// =============================================================================
// =============================================================================
template <size_t dim, typename device_t, typename Function>
auto
InitialAMRSetup<dim, device_t, Function>::setup_initial_data_ghosted_block(
  typename MeshMap<dim, device_t>::orchard_key_view_t orchard_keys_device,
  bool                                                fill_ghosts) -> DataArrayGhostedBlock_t
{
  //
  // create some data
  //

  // now generate some user data (block based AMR)
  HydroParams params = HydroParams(m_config_map);

  const int nbvar = nbvar_hydro<dim>();

  //
  // create some test user data (on cells, taking into account ghost cell, set to zero), and
  // upload to device
  //
  auto ghosted_block_sizes = m_block_sizes;
  ghosted_block_sizes[IX] += 2 * m_ghost_sizes[IX];
  ghosted_block_sizes[IY] += 2 * m_ghost_sizes[IY];
  if constexpr (dim == 3)
  {
    ghosted_block_sizes[IZ] += 2 * m_ghost_sizes[IZ];
  }

  shift_t<dim> shift;
  shift[IX] = -m_ghost_sizes[IX];
  shift[IY] = -m_ghost_sizes[IY];
  if constexpr (dim == 3)
  {
    shift[IZ] = -m_ghost_sizes[IZ];
  }

  auto userdata_ghosted_block =
    DataArrayGhostedBlock_t(m_block_sizes,
                            ghosted_block_sizes,
                            shift,
                            "test_data_ghosted_block",
                            nbvar,
                            m_mesh_map->get_amr_mesh_info().local_num_quadrants_total());

  auto block_sizes = m_block_sizes;
  auto brick_sizes = m_brick_sizes;

  auto ghosted_block_size = userdata_ghosted_block.ghosted_block_size();

  // get domain lower left corner
  const auto xyz_min = get_xyz_min<dim>(m_config_map);

  // get geometrical scaling factor
  const auto scaling_factor = get_scaling_factor(m_config_map);

  {
    // number of quadrants in current MPI process
    const auto nbOcts = m_amr_mesh->local_num_quadrants_total();
    const auto num_vars = nbvar;

    const auto tx = ghosted_block_size[IX];
    const auto ty = ghosted_block_size[IY];
    const auto tz = [=]() {
      if constexpr (dim == 2)
        return 1;
      if constexpr (dim == 3)
        return ghosted_block_size[IZ];
    }();

    const int32_t bx = m_block_sizes[IX];
    const int32_t by = m_block_sizes[IY];
    const int32_t bz = [=]() {
      if constexpr (dim == 3)
        return m_block_sizes[IZ];
      else
        return 1;
    }();

    const int32_t gx = m_ghost_sizes[IX];
    const int32_t gy = m_ghost_sizes[IY];
    const int32_t gz = [=]() {
      if constexpr (dim == 3)
        return m_ghost_sizes[IZ];
      else
        return 1;
    }();

    // Kokkos team policy type alias
    using team_policy_t = Kokkos::TeamPolicy<exec_space, Kokkos::IndexType<int32_t>>;

    team_policy_t policy(nbOcts * num_vars,
                         Kokkos::AUTO() /* number of threads per team is chosen by kokkos */);
    using thread_t = typename team_policy_t::member_type;

    auto f = m_f;

    const auto first_outside_quad_local_id =
      m_mesh_map->get_amr_mesh_info().first_outside_quad_local_id();

    const auto is_brick_periodic = m_is_brick_periodic;

    Kokkos::parallel_for(
      "Fill_test_data_block", policy, KOKKOS_LAMBDA(const thread_t & member) {
        // block sizes (bx,by,bz) are captured by the lambda

        // get lower left corner real space coordinates
        constexpr bool use_center = false;

        // number of cells per octant (ghost included)
        // const uint32_t nbCells = dim == 2 ? tx * ty : tx * ty * tz;

        // get (octant id, ivar) from  the team id
        // using left indexing, we get
        // tmp = ivar + iOct * num_vars
        const auto tmp = member.league_rank();
        const auto iOct = tmp / num_vars;
        const auto ivar = tmp - iOct * num_vars;

        // get orchard key of current block/octant
        const auto key = orchard_keys_device(iOct);

        // get octant level
        const auto level = orchard_key_t<dim>::level(key);

        // compute cell length in real space
        // const double dx_cell = compute_cell_length<dim>(level, bx);

        // get real space coordinates of lower left corner of the block
        const auto xyz_corner = (iOct >= first_outside_quad_local_id)
                                  ? outside_key_to_vertex_coord<dim>(key, use_center, brick_sizes)
                                  : orchard_key_to_vertex_coord<dim>(key, use_center);

      // the only reason of the following dummy code to be here, is that cuda nvcc compile
      // doesn't support capturing variables inside the inner lambda inside a constexpr if
      //
      // strangely, nvc++ is ok and don't need that
      // TODO: remove theses lines when nvcc will have support for this.
#ifdef __NVCC__
        [[maybe_unused]] int dummy = 0;
        if (tx == 0 or ty == 0 or tz == 0 or gx == 0 or gy == 0 or gz == 0 or bx == 0 or by == 0 or
            bz == 0 or userdata_ghosted_block.num_cells() == 0 or block_sizes[IX] == 0 or
            fill_ghosts != fill_ghosts or f(0, 0, 0) != f(0, 0, 0) or brick_sizes[IX] == 0 or
            is_brick_periodic[IX] == 0 or scaling_factor != scaling_factor or
            xyz_min[0] != xyz_min[0])
          dummy++;
#endif

        // clang-format off
          if constexpr (dim == 2)
          {
            Kokkos::parallel_for(
              Kokkos::TeamVectorMDRange<
              Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
              thread_t>(member, tx, ty),
              [=](const int32_t i, const int32_t j) {
                Kokkos::Array<int32_t, dim> iCoord{ i - gx, j - gy };
                auto xyz_cell_vertex = compute_cell_coordinates<dim>(
                  level, xyz_corner, iCoord, block_sizes);

                // take into account mesh periodicity
                if (is_brick_periodic[IX])
                {
                    if (xyz_cell_vertex[IX] < 0)
                      xyz_cell_vertex[IX] += static_cast<real_t>(brick_sizes[IX]);
                    if (xyz_cell_vertex[IX] > brick_sizes[IX])
                      xyz_cell_vertex[IX] -= static_cast<real_t>(brick_sizes[IX]);
                }
                if (is_brick_periodic[IY])
                {
                  if (xyz_cell_vertex[IY] < 0)
                    xyz_cell_vertex[IY] += static_cast<real_t>(brick_sizes[IY]);
                  if (xyz_cell_vertex[IY] > brick_sizes[IY])
                    xyz_cell_vertex[IY] += static_cast<real_t>(-brick_sizes[IY]);
                }
                const auto xyz_cell = vertex_coord_to_real_space<dim>(xyz_cell_vertex, scaling_factor, xyz_min);

                if (i >= gx and i < gx + bx and j >= gy and j < gy + by)
                {
                  userdata_ghosted_block(i-gx, j-gy, ivar, iOct) =
                    f(xyz_cell[IX], xyz_cell[IY], ivar);
                }
                else
                {
                  userdata_ghosted_block(i-gx, j-gy, ivar, iOct) =
                    fill_ghosts ? f(xyz_cell[IX], xyz_cell[IY], ivar) : 0;
                }

              }); // end TeamVectorRange
          }
        // clang-format on

        else if constexpr (dim == 3)
        {
          Kokkos::parallel_for(
            Kokkos::TeamVectorMDRange<Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                                      thread_t>(member, tx, ty, tz),
            [=](const int32_t i, const int32_t j, const int32_t k) {
              Kokkos::Array<int32_t, dim> iCoord{ i - gx, j - gy, k - gz };
              auto                        xyz_cell_vertex =
                compute_cell_coordinates<dim>(level, xyz_corner, iCoord, block_sizes);

              // take into account mesh periodicity
              if (is_brick_periodic[IX])
              {
                if (xyz_cell_vertex[IX] < 0)
                  xyz_cell_vertex[IX] += static_cast<real_t>(brick_sizes[IX]);
                if (xyz_cell_vertex[IX] > brick_sizes[IX])
                  xyz_cell_vertex[IX] -= static_cast<real_t>(brick_sizes[IX]);
              }
              if (is_brick_periodic[IY])
              {
                if (xyz_cell_vertex[IY] < 0)
                  xyz_cell_vertex[IY] += static_cast<real_t>(brick_sizes[IY]);
                if (xyz_cell_vertex[IY] > brick_sizes[IY])
                  xyz_cell_vertex[IY] += static_cast<real_t>(-brick_sizes[IY]);
              }
              if (is_brick_periodic[IZ])
              {
                if (xyz_cell_vertex[IZ] < 0)
                  xyz_cell_vertex[IZ] += static_cast<real_t>(brick_sizes[IZ]);
                if (xyz_cell_vertex[IZ] > brick_sizes[IZ])
                  xyz_cell_vertex[IZ] += static_cast<real_t>(-brick_sizes[IZ]);
              }

              const auto xyz_cell =
                vertex_coord_to_real_space<dim>(xyz_cell_vertex, scaling_factor, xyz_min);

              if (i >= gx and i < gx + bx and j >= gy and j < gy + by and k >= gz and k < gz + bz)
              {
                userdata_ghosted_block(i - gx, j - gy, k - gz, ivar, iOct) =
                  f(xyz_cell[IX], xyz_cell[IY], xyz_cell[IZ], ivar);
              }
              else
              {
                userdata_ghosted_block(i - gx, j - gy, k - gz, ivar, iOct) =
                  fill_ghosts ? f(xyz_cell[IX], xyz_cell[IY], xyz_cell[IZ], ivar) : 0;
              }
            }); // end TeamVectorRange
        }
      });
  }

  return userdata_ghosted_block;

} // setup_initial_data_ghosted_block

// =============================================================================
// =============================================================================
template <size_t dim, typename device_t, typename Function>
auto
InitialAMRSetup<dim, device_t, Function>::compute_diff_ghosted_block(
  typename MeshMap<dim, device_t>::orchard_key_view_t orchard_keys_device,
  DataArrayGhostedBlock_t                             data1,
  DataArrayGhostedBlock_t                             data2) -> DataArrayGhostedBlock_t
{
  // KOKKOS_ASSERT(data1.num_vars == data2.num_vars);

  using DirectAccess = typename DataArrayGhostedBlock_t::DirectAccess;

  const auto block_sizes = m_block_sizes;
  const auto brick_sizes = m_brick_sizes;

  const auto num_vars = data1.num_vars();
  const auto num_octs = data1.num_quadrants();

  const auto total_block_size = data1.ghosted_block_size();

  const auto tx = total_block_size[IX];
  const auto ty = total_block_size[IY];
  const auto tz = [=]() {
    if constexpr (dim == 2)
      return 1;
    if constexpr (dim == 3)
      return total_block_size[IZ];
  }();

  // shift
  const int32_t                  sx = data1.shift()[IX];
  const int32_t                  sy = data1.shift()[IY];
  [[maybe_unused]] const int32_t sz = [=]() {
    if constexpr (dim == 3)
      return data1.shift()[IZ];
    else
      return 0;
  }();

  // get domain lower left corner
  const auto xyz_min = get_xyz_min<dim>(m_config_map);

  // get geometrical scaling factor
  const auto scaling_factor = get_scaling_factor(m_config_map);

  const auto first_outside_quad_local_id =
    m_mesh_map->get_amr_mesh_info().first_outside_quad_local_id();

  const auto is_brick_periodic = m_is_brick_periodic;

  auto f = m_f;

  DataArrayGhostedBlock_t data_diff = DataArrayGhostedBlock_t(data1.block_size(),
                                                              data1.ghosted_block_size(),
                                                              data1.shift(),
                                                              "userdata_ghosted_block_diff",
                                                              num_vars,
                                                              num_octs);

  constexpr real_t tol = KALYPSSO_NUM(1e-7);

  // Kokkos team policy type alias
  using team_policy_t = Kokkos::TeamPolicy<exec_space, Kokkos::IndexType<int32_t>>;

  team_policy_t policy(num_octs * num_vars,
                       Kokkos::AUTO() /* number of threads per team is chosen by kokkos */);
  using thread_t = typename team_policy_t::member_type;

  Kokkos::parallel_for(
    "compare data", policy, KOKKOS_LAMBDA(const thread_t & member) {
      // get lower left corner real space coordinates
      constexpr bool use_center = false;

      const auto tmp = member.league_rank();
      const auto iOct = tmp / num_vars;
      const auto ivar = tmp - iOct * num_vars;

      // get orchard key of current block/octant
      const auto key = orchard_keys_device(iOct);

      // get octant level
      const auto level = orchard_key_t<dim>::level(key);

      // get real space coordinates of lower left corner of the block
      const auto xyz_corner = (iOct >= first_outside_quad_local_id)
                                ? outside_key_to_vertex_coord<dim>(key, use_center, brick_sizes)
                                : orchard_key_to_vertex_coord<dim>(key, use_center);

      // get the cell length
      real_t dx = compute_cell_length<dim>(level, block_sizes[IX]) * scaling_factor;
      real_t dy = compute_cell_length<dim>(level, block_sizes[IY]) * scaling_factor;

    // the only reason of the following dummy code to be here, is that cuda nvcc compile doesn't
    // support capturing variables inside the inner lambda inside a constexpr if
    //
    // strangely, nvc++ is ok and don't need that
    // TODO: remove theses lines when nvcc will have support for this.
#ifdef __NVCC__
      [[maybe_unused]] int dummy = 0;
      if (tx == 0 or ty == 0 or tz == 0 or sx == 0 or sy == 0 or sz == 0 or
          data1.num_cells() == 0 or data2.num_cells() == 0 or data_diff.num_cells() == 0 or
          block_sizes[IX] == 0 or brick_sizes[IX] == 0 or tol < 0 or f(0, 0, 0) != f(0, 0, 0) or
          is_brick_periodic[IX] == 0 or scaling_factor != scaling_factor or
          xyz_min[0] != xyz_min[0])
        dummy++;
#endif
      if constexpr (dim == 2)
      {
        Kokkos::parallel_for(
          Kokkos::TeamVectorMDRange<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                                    thread_t>(member, tx, ty),
          [=](const int32_t i, const int32_t j) {
            data_diff(i, j, ivar, iOct, DirectAccess{}) =
              data1(i, j, ivar, iOct, DirectAccess{}) - data2(i, j, ivar, iOct, DirectAccess{});

            // try to evaluate if we can account the difference by the
            // coarse-graining that appear in the ghost layer surrounding a
            // small with coarser block neighbor if the difference is still
            // large, it means we have a bug

            if (fabs(data_diff(i, j, ivar, iOct, DirectAccess{})) > tol)
            {
              Kokkos::Array<int32_t, dim> iCoord{ i + sx, j + sy };
              auto                        xyz_cell_vertex =
                compute_cell_coordinates<dim>(level, xyz_corner, iCoord, block_sizes);

              if (is_brick_periodic[IX])
              {
                if (xyz_cell_vertex[IX] < 0)
                  xyz_cell_vertex[IX] += static_cast<real_t>(brick_sizes[IX]);
                if (xyz_cell_vertex[IX] > brick_sizes[IX])
                  xyz_cell_vertex[IX] -= static_cast<real_t>(brick_sizes[IX]);
              }
              if (is_brick_periodic[IY])
              {
                if (xyz_cell_vertex[IY] < 0)
                  xyz_cell_vertex[IY] += static_cast<real_t>(brick_sizes[IY]);
                if (xyz_cell_vertex[IY] > brick_sizes[IY])
                  xyz_cell_vertex[IY] += static_cast<real_t>(-brick_sizes[IY]);
              }

              auto & xx = xyz_cell_vertex[IX];
              if ((-sx) % 2 == 0)
              {
                xx = (i % 2 == 0) ? xx + dx / 2 : xx - dx / 2;
              }
              else
              {
                xx = (i % 2 == 0) ? xx - dx / 2 : xx + dx / 2;
              }

              auto & yy = xyz_cell_vertex[IY];
              if ((-sy) % 2 == 0)
              {
                yy = (j % 2 == 0) ? yy + dy / 2 : yy - dy / 2;
              }
              else
              {
                yy = (j % 2 == 0) ? yy - dy / 2 : yy + dy / 2;
              }

              const auto xyz_cell =
                vertex_coord_to_real_space<dim>(xyz_cell_vertex, scaling_factor, xyz_min);

              data_diff(i, j, ivar, iOct, DirectAccess{}) =
                data1(i, j, ivar, iOct, DirectAccess{}) - f(xyz_cell[IX], xyz_cell[IY], ivar);
            }
          });
      }
      else if constexpr (dim == 3)
      {
        real_t dz = compute_cell_length<dim>(level, block_sizes[IZ]) * scaling_factor;

        Kokkos::parallel_for(
          Kokkos::TeamVectorMDRange<Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                                    thread_t>(member, tx, ty, tz),
          [=](const int32_t i, const int32_t j, const int32_t k) {
            data_diff(i, j, k, ivar, iOct, DirectAccess{}) =
              data1(i, j, k, ivar, iOct, DirectAccess{}) -
              data2(i, j, k, ivar, iOct, DirectAccess{});
            // try to evaluate if we can account the difference by the
            // coarse-graining that appear in the ghost layer surrounding a
            // small with coarser block neighbor if the difference is still
            // large, it means we have a bug

            if (fabs(data_diff(i, j, k, ivar, iOct, DirectAccess{})) > tol)
            {
              Kokkos::Array<int32_t, dim> iCoord{ i + sx, j + sy, k + sz };
              auto                        xyz_cell_vertex =
                compute_cell_coordinates<dim>(level, xyz_corner, iCoord, block_sizes);

              if (is_brick_periodic[IX])
              {
                if (xyz_cell_vertex[IX] < 0)
                  xyz_cell_vertex[IX] += static_cast<real_t>(brick_sizes[IX]);
                if (xyz_cell_vertex[IX] > brick_sizes[IX])
                  xyz_cell_vertex[IX] -= static_cast<real_t>(brick_sizes[IX]);
              }

              if (is_brick_periodic[IY])
              {
                if (xyz_cell_vertex[IY] < 0)
                  xyz_cell_vertex[IY] += static_cast<real_t>(brick_sizes[IY]);
                if (xyz_cell_vertex[IY] > brick_sizes[IY])
                  xyz_cell_vertex[IY] += static_cast<real_t>(-brick_sizes[IY]);
              }

              if (is_brick_periodic[IZ])
              {
                if (xyz_cell_vertex[IZ] < 0)
                  xyz_cell_vertex[IZ] += static_cast<real_t>(brick_sizes[IZ]);
                if (xyz_cell_vertex[IZ] > brick_sizes[IZ])
                  xyz_cell_vertex[IZ] += static_cast<real_t>(-brick_sizes[IZ]);
              }

              auto & xx = xyz_cell_vertex[IX];
              if ((-sx) % 2 == 0)
              {
                xx = (i % 2 == 0) ? xx + dx / 2 : xx - dx / 2;
              }
              else
              {
                xx = (i % 2 == 0) ? xx - dx / 2 : xx + dx / 2;
              }

              auto & yy = xyz_cell_vertex[IY];
              if ((-sy) % 2 == 0)
              {
                yy = (j % 2 == 0) ? yy + dy / 2 : yy - dy / 2;
              }
              else
              {
                yy = (j % 2 == 0) ? yy - dy / 2 : yy + dy / 2;
              }

              auto & zz = xyz_cell_vertex[IZ];
              if ((-sz) % 2 == 0)
              {
                zz = (k % 2 == 0) ? zz + dz / 2 : zz - dz / 2;
              }
              else
              {
                zz = (k % 2 == 0) ? zz - dz / 2 : zz + dz / 2;
              }

              const auto xyz_cell =
                vertex_coord_to_real_space<dim>(xyz_cell_vertex, scaling_factor, xyz_min);

              data_diff(i, j, k, ivar, iOct, DirectAccess{}) =
                data1(i, j, k, ivar, iOct, DirectAccess{}) -
                f(xyz_cell[IX], xyz_cell[IY], xyz_cell[IZ], ivar);
            }
          });
      }
    });

  return data_diff;

} // compute_diff_ghosted_block

template class InitialAMRSetup<2, kalypsso::DefaultDevice, InitGaussian>;
template class InitialAMRSetup<3, kalypsso::DefaultDevice, InitGaussian>;

template class InitialAMRSetup<2, kalypsso::DefaultDevice, InitHat>;
template class InitialAMRSetup<3, kalypsso::DefaultDevice, InitHat>;

template class InitialAMRSetup<2, kalypsso::DefaultDevice, InitFuncParabola>;
template class InitialAMRSetup<3, kalypsso::DefaultDevice, InitFuncParabola>;

template class InitialAMRSetup<2, kalypsso::DefaultDevice, InitFuncSineWave>;
template class InitialAMRSetup<3, kalypsso::DefaultDevice, InitFuncSineWave>;

template class InitialAMRSetup<2, kalypsso::DefaultDevice, InitFunc1>;
template class InitialAMRSetup<3, kalypsso::DefaultDevice, InitFunc1>;

template class InitialAMRSetup<2, kalypsso::DefaultDevice, InitFunc2>;
template class InitialAMRSetup<3, kalypsso::DefaultDevice, InitFunc2>;

template class InitialAMRSetup<2, kalypsso::DefaultDevice, InitFunc3>;
template class InitialAMRSetup<3, kalypsso::DefaultDevice, InitFunc3>;

// only instantiate those class when the default device is not on host
#if defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP) || \
  defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_THREADS) ||  \
  defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL)
#else
template class InitialAMRSetup<2, kalypsso::HostDevice, InitFuncSineWave>;
template class InitialAMRSetup<3, kalypsso::HostDevice, InitFuncSineWave>;
#endif

} // namespace kalypsso
