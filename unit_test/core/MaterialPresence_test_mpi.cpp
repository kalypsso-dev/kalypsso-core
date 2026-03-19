// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/core/MaterialPresence.h>

#include <kalypsso/core/AMRmesh.h>
#include <kalypsso/core/MeshMap.h>
#include <kalypsso/core/AMRContext.h>
#include <kalypsso/core/MeshGhostsExchanger.h>

#include "gtest/gtest.h"
#include "../main_kalypsso_unittest.h"

namespace kalypsso
{

// ================================================================================================
// ================================================================================================
TEST(kalypsso_shared_test, MaterialPresence_exchange)
{
  static char         null_char = '\0';
  static char *       empty_string = &null_char;
  const ParallelEnv & par_env = **g_par_env_ptr;
  ConfigMap           config_map(empty_string, 1);

  config_map.setInteger("amr", "level_min", 0);
  config_map.setInteger("amr", "level_max", 0);
  config_map.setString("amr", "connectivity", "brick");

  config_map.setInteger("p4est_connectivity", "nbrick_x", 2);
  config_map.setInteger("p4est_connectivity", "nbrick_y", 2);

  AMRmesh<2>                               amr_mesh(par_env.mpi_comm(), config_map);
  MeshMap<2, HostDevice>                   amr_mesh_map(config_map, par_env);
  MaterialPresenceExchanger<2, HostDevice> mat_pres_exchanger(
    config_map, par_env, amr_mesh, amr_mesh_map);

  auto rebuild_forest = [&]() {
    amr_mesh.reset_ghost();
    amr_mesh_map.compute_outside_quad_info(amr_mesh.forest(), amr_mesh.ghost());
    amr_mesh_map.update_hashmap(amr_mesh.forest(), amr_mesh.ghost(), true);
    amr_mesh_map.update_mirror_orchard_keys(amr_mesh.forest(), amr_mesh.ghost());
    amr_mesh_map.update_amr_mesh_info(amr_mesh.forest(), amr_mesh.ghost());
    amr_mesh_map.update_orchard_keys(amr_mesh.forest(), amr_mesh.ghost());
    amr_mesh_map.update_conformal_status();
  };

  rebuild_forest();

  using MatPresView_t = MaterialPresenceView<HostDevice>;
  MatPresView_t mat_pres("mat pres src", 3, amr_mesh.local_num_quadrants_total());
  mat_pres.resets();

  {
    mat_pres.set(0, par_env.rank() % 3);
    mat_pres.set(1, 0);
  }

  mat_pres_exchanger.exchange(mat_pres);

  {
    EXPECT_EQ(mat_pres.get(2, 0), par_env.rank() == 1);
    EXPECT_EQ(mat_pres.get(2, 1), par_env.rank() == 0);
    EXPECT_FALSE(mat_pres.get(2, 2));

    EXPECT_TRUE(mat_pres.get(3, 0));
    EXPECT_FALSE(mat_pres.get(3, 1));
    EXPECT_FALSE(mat_pres.get(3, 2));
  }
}

} // namespace kalypsso
