// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file multimaterial_utils.h
 */
#ifndef KALYPSSO_CORE_MULTIMATERIAL_UTILS_H_
#define KALYPSSO_CORE_MULTIMATERIAL_UTILS_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/utils/config/ConfigMap.h>

namespace kalypsso
{

/**
 * Read material id in region.
 *
 * \param[in] i_region region id
 * \param[in] config_map application's configuration parameters
 */
inline int
get_material_id_in_region(const int32_t i_region, const ConfigMap & config_map)
{
  const auto section = "region" + std::to_string(i_region);
  const int  i_mat = config_map.getInteger(section, "material_id", -1);
  return i_mat;
}

/**
 * Get region to material mapping (number of regions is known at compile time).
 */
template <size_t NB_REGIONS>
Kokkos::Array<int, NB_REGIONS>
get_region_to_material_id_mapping(ConfigMap const & config_map)
{
  Kokkos::Array<int, NB_REGIONS> region_to_material_id;
  for (int i_region = 0; i_region < static_cast<int>(NB_REGIONS); ++i_region)
  {
    region_to_material_id[i_region] = get_material_id_in_region(i_region, config_map);
  }

  return region_to_material_id;

} // get_region_to_material_id_mapping

/**
 * Get region to material mapping (number of regions is only known at runtime time).
 */
template <typename device_t>
Kokkos::View<int *, device_t>
get_region_to_material_id_mapping_view(ConfigMap const & config_map, int nb_regions)
{
  Kokkos::View<int *, device_t> region_to_material_id("region_to_material_id",
                                                      static_cast<size_t>(nb_regions));
  auto region_to_material_id_host = Kokkos::create_mirror_view(region_to_material_id);

  for (int i_region = 0; i_region < nb_regions; ++i_region)
  {
    region_to_material_id_host(i_region) = get_material_id_in_region(i_region, config_map);
  }
  Kokkos::deep_copy(region_to_material_id, region_to_material_id_host);

  return region_to_material_id;

} // get_region_to_material_id_mapping_view

} // namespace kalypsso

#endif // KALYPSSO_CORE_MULTIMATERIAL_UTILS_H_
