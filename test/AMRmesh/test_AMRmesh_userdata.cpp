// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file test_AMRmesh_userdata.cpp
 */

#include <kalypsso/core/kalypsso_core_config.h> // for KALYPSSO_CORE_USE_HDF5, ...
#include <kalypsso/core/AMRmesh.h>
#include <kalypsso/core/MeshMap.h> // for orchard_key_view_t
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/cmdline_utils.h>
#include <kalypsso/utils/log/kalypsso_log.h>
#include <kalypsso/core/models/Hydro.h>
#include <kalypsso/core/models/HydroState.h>
#include <kalypsso/core/io_utils.h>
#ifdef KALYPSSO_CORE_USE_HDF5
#  include <kalypsso/core/HDF5_Xdmf_Writer_legacy.h>
#endif
#include <kalypsso/core/utils_block.h>
#include <kalypsso/core/orchard_key_utils.h>

#include <kalypsso/utils/p4est/p4est_wrapper.h>
#include <kalypsso/utils/p4est/connectivity.h>
#include <kalypsso/utils/p4est/geometry.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <test_common/DataWriter.h>

#include <cstdlib>

namespace kalypsso
{

// ============================================================
// ============================================================
// ============================================================
template <int dim, typename device_t>
void
run_test(const ParallelEnv & par_env, const ConfigMap & config_map)
{
  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  using namespace p4est;

  using Hydro_t = core::models::Hydro;

  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  if (par_env.rank() == 0)
    printf("================================================\n");

  auto amr_mesh = std::make_shared<AMRmesh<dim>>(par_env, config_map);

  amr_mesh->reset_ghost();

  auto forest = amr_mesh->forest();
  // auto conn   = amr_mesh->connectivity();
  auto geom = amr_mesh->geometry();

  // geometry not supported here (for now)
  assert(geom == nullptr);

  auto conn_name = config_map.getString("amr", "connectivity", "invalid_connectivity");
  auto geom_name = geom == nullptr ? "no_geometry" : geom->name;

  if (par_env.rank() == 0)
  {
    printf("Running a %dD test with connectivity %s and geometry %s\n",
           dim,
           conn_name.c_str(),
           geom_name);
  }

  printf("Forest has %ld global octants, %d local octants on MPI proc %d.\n",
         forest->global_num_quadrants,
         forest->local_num_quadrants,
         par_env.rank());

  // now generate some user data (block based AMR)
  HydroParams params = HydroParams(config_map);
  // BlastParams blastParams = BlastParams(config_map);

  uint32_t bx = (uint32_t)config_map.getInteger("amr", "bx", 0);
  uint32_t by = (uint32_t)config_map.getInteger("amr", "by", 0);
  uint32_t bz = (uint32_t)config_map.getInteger("amr", "bz", 0);

  block_size_t<dim> block_sizes;
  block_sizes[IX] = bx;
  block_sizes[IY] = by;
  if constexpr (dim == 3)
    block_sizes[IZ] = bz;

  Kokkos::Array<uint8_t, 3> brick_sizes;
  brick_sizes[0] = static_cast<uint8_t>(config_map.getInteger("p4est_connectivity", "nbrick_x", 2));
  brick_sizes[1] = static_cast<uint8_t>(config_map.getInteger("p4est_connectivity", "nbrick_y", 3));
  if constexpr (dim == 3)
    brick_sizes[2] =
      static_cast<uint8_t>(config_map.getInteger("p4est_connectivity", "nbrick_z", 4));

  const int nbvar = nbvar_hydro<dim>();

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  Hydro_t model = Hydro_t(params.dimType);
  // FieldMap<core::models::Hydro> fm;
  // fm.setup(params, config_map);
  // auto fm = fieldMgr.get_id2index();
  const auto fm = model.get_fieldmap();

  // variable map
  const auto id2names = model.get_id2names_map();

  auto userdataBlock =
    DataArrayBlock_t("fake_data_block", block_sizes, nbvar, amr_mesh->local_num_quadrants());

  // create orchard keys as a view
  MeshMap<dim, device_t> mesh_map(config_map, par_env);

  mesh_map.update_orchard_keys(amr_mesh->forest(), amr_mesh->ghost());
  auto orchard_keys_host = mesh_map.orchard_keys_host();
  auto orchard_keys_device = mesh_map.orchard_keys();

  const auto nbCellsPerLeaf = userdataBlock.num_cells();
  const auto nbCellsTotal = amr_mesh->local_num_quadrants() * nbCellsPerLeaf;

  Kokkos::parallel_for(
    "init_data",
    Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
    KOKKOS_LAMBDA(uint32_t global_index) {
      constexpr auto ID = core::models::Hydro::ID;
      // constexpr auto IP = core::models::Hydro::IP;
      constexpr auto IE = core::models::Hydro::IE;
      constexpr auto IU = core::models::Hydro::IU;
      constexpr auto IV = core::models::Hydro::IV;
      constexpr auto IW = core::models::Hydro::IW;

      // convert global index into
      // - octant id
      // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
      const auto iOct = global_index / nbCellsPerLeaf;
      const auto cell_index = global_index - iOct * nbCellsPerLeaf;

      // compute ix,iy,iz of local cell inside
      // block from index
      auto iCoord = cellindex_to_coord<dim>(cell_index, block_sizes);

      // get block orchard key
      const auto key = orchard_keys_device(iOct);

      // get block level
      // const auto level = orchard_key_t<dim>::level(key);

      // compute physical x,y,z for that cell (cell center)
      const auto xyz = orchard_key_to_cell_coord<dim>(key, iCoord, block_sizes[IX]);

      auto e = xyz[IX] * xyz[IX] + xyz[IY] * xyz[IY];
      if constexpr (dim == 3)
        e += xyz[IZ] * xyz[IZ];

      userdataBlock(cell_index, fm[ID], iOct) = xyz[0] + xyz[1];
      userdataBlock(cell_index, fm[IE], iOct) = e;
      userdataBlock(cell_index, fm[IU], iOct) = 0;
      userdataBlock(cell_index, fm[IV], iOct) = 1;
      if constexpr (dim == 3)
        userdataBlock(cell_index, fm[IW], iOct) = 42;
    });


  {
    std::string outputDir = config_map.getString("output", "outputDir", "./");
    std::string filename = dim == 2 ? "test_userdata_2d" : "test_userdata_3d";

    DataWriter<dim, device_t, Hydro_t>::save(filename, userdataBlock, config_map, *amr_mesh, model);
  }

} // run_test

} // namespace kalypsso

// ======================================================
// ======================================================
bool
hasEnding(std::string const & fullString, std::string const & ending)
{
  if (fullString.length() >= ending.length())
  {
    return (0 ==
            fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
  }
  else
  {
    return false;
  }
}

// ======================================================
// ======================================================
//! parse command line
bool
arg_exists(char ** begin, char ** end, const std::string & arg)
{
  return std::find(begin, end, arg) != end;
}

// ======================================================
// ======================================================
std::string
get_ini_filename(char ** begin, char ** end, const std::string & arg)
{
  char ** itr = std::find(begin, end, arg);
  if (itr != end && ++itr != end)
  {
    return std::string(*itr);
  }
  return std::string("");
}

// ======================================================
// ======================================================
// ======================================================
int
main(int argc, char ** argv)
{

  {
    // initialize mpi, kokkos and p4est
    kalypsso::ParallelEnv par_env(argc, argv);

#ifdef KALYPSSO_CORE_USE_SPDLOG
    kalypsso::kalypsso_spdlog_config(argc, argv, par_env.rank(), par_env.size());
#endif

    if (kalypsso::cmdline_arg_exists(argv, argv + argc, "--help"))
    {
      if (par_env.rank() == 0)
      {
        // clang-format off
        std::cout << "Example cmdline: \"mpirun -np 4 ./test_AMRmesh_userdata --ini test_AMRmesh_userdata_2d.ini --refine_level 6\"\n";
        // clang-format on
      }
      return 0;
    }

    // parse command line : 2d or 3d ?
    bool threed = arg_exists(argv, argv + argc, "--3d");

    // check if user passed a custom ini filename
    std::string config_filename = get_ini_filename(argv, argv + argc, "--ini");

    // provide a default config filename (that exists)
    if (config_filename.size() == 0)
    {
      config_filename =
        threed ? "./test_AMRmesh_userdata_3d.ini" : "./test_AMRmesh_userdata_2d.ini";
    }

    kalypsso::ConfigMap config_map = kalypsso::broadcast_parameters(config_filename);

    // check if input file is valid, i.e. parameter run/dimension is 2 or 3
    const auto dim = kalypsso::get_dim(config_map);
    assertm(dim == 2 or dim == 3, "[test_AMRmesh_userdata] Wrong dimension");

    if (dim == 2)
    {
      kalypsso::run_test<2, kalypsso::DefaultDevice>(par_env, config_map);
    }
    else if (dim == 3)
    {
      kalypsso::run_test<3, kalypsso::DefaultDevice>(par_env, config_map);
    }
    else
    {
      if (par_env.rank() == 0)
      {
        std::cout << "Input file is not valid ! check parameter run/dimension.\n";
      }
    }
  }

  return 0;
}
