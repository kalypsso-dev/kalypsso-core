// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MieGruneisenEosArray.cpp
 */
#include <kalypsso/core/eos/MieGruneisenEosArray.h>
#include <kalypsso/utils/log/kalypsso_log.h>

#include <type_traits> // for std::is_same_v

namespace kalypsso
{

namespace core
{

namespace eos
{

// =====================================================================
// =====================================================================
size_t
get_number_of_eos(ConfigMap const & config_map, MG_EOS_TYPE mg_eos_type)
{
  size_t     num_mat = 0;
  const auto nmat = static_cast<size_t>(config_map.getInteger("run", "nmat", 2));

  for (size_t i_mat = 0; i_mat < nmat; ++i_mat)
  {
    if (get_mg_eos_type(i_mat, config_map) == mg_eos_type)
    {
      num_mat++;
    }
  }

  return num_mat;
} // get_number_of_eos

// =====================================================================
// =====================================================================
template <typename device_t>
MieGruneisenEosArray<device_t>::MieGruneisenEosArray(ConfigMap const & config_map)
  : m_num_material(static_cast<size_t>(config_map.getInteger("run", "nmat", 2)))
  , m_mg_eos_ig("mg_eos_ig", get_number_of_eos(config_map, MG_EOS_TYPE::MG_IDEAL_GAS))
  , m_mg_eos_sg("mg_eos_sg", get_number_of_eos(config_map, MG_EOS_TYPE::MG_STIFFENED_GAS))
  , m_mg_eos_vdw("mg_eos_vdw", get_number_of_eos(config_map, MG_EOS_TYPE::MG_VANDERWAALS_GAS))
  , m_mg_eos_sw("mg_eos_sw", get_number_of_eos(config_map, MG_EOS_TYPE::MG_SHOCKWAVE))
  , m_mg_eos_sw2("mg_eos_sw2", get_number_of_eos(config_map, MG_EOS_TYPE::MG_SHOCKWAVE2))
  , m_mg_eos_cc("mg_eos_cc", get_number_of_eos(config_map, MG_EOS_TYPE::MG_COCHRAN_CHAN))
  , m_mg_eos_jwl("mg_eos_jwl", get_number_of_eos(config_map, MG_EOS_TYPE::MG_JWL))
  , m_material_eos_type("material_eos_type", m_num_material)
  , m_material_eos_id("material_eos_id", m_num_material)
{
  // EOS histogram
  auto eos_histogram = Kokkos::View<int *, HostDevice>("EOS histogram", MG_EOS_TYPE::MG_NUM);
  Kokkos::deep_copy(eos_histogram, 0);

  auto material_eos_type_host = material_eos_type_t::create_host_mirror_view(m_material_eos_type);
  auto material_eos_id_host = material_eos_id_t::create_host_mirror_view(m_material_eos_id);

  for (size_t i = 0; i < m_num_material; ++i)
    material_eos_id_host(i) = 0;

  // init material eos type array (on host)
  for (size_t i_mat = 0; i_mat < m_num_material; ++i_mat)
  {
    const auto eos_type = get_mg_eos_type(i_mat, config_map);
    assertm(eos_type != +MG_EOS_TYPE::MG_INVALID,
            "Wrong Mie-Gruneisen eos type. Please check input file.");
    const auto eos_type_int = eos_type._to_integral();

    material_eos_type_host(i_mat) = eos_type_int;
    material_eos_id_host(i_mat) = eos_histogram(eos_type_int);
    eos_histogram(eos_type_int) = eos_histogram(eos_type_int) + 1;
  }

  // create host mirror views
  auto mg_eos_ig_host = Kokkos::create_mirror_view(m_mg_eos_ig);
  auto mg_eos_sg_host = Kokkos::create_mirror_view(m_mg_eos_sg);
  auto mg_eos_vdw_host = Kokkos::create_mirror_view(m_mg_eos_vdw);
  auto mg_eos_sw_host = Kokkos::create_mirror_view(m_mg_eos_sw);
  auto mg_eos_sw2_host = Kokkos::create_mirror_view(m_mg_eos_sw2);
  auto mg_eos_cc_host = Kokkos::create_mirror_view(m_mg_eos_cc);
  auto mg_eos_jwl_host = Kokkos::create_mirror_view(m_mg_eos_jwl);

  // initialize all eos
  for (size_t i_mat = 0; i_mat < m_num_material; ++i_mat)
  {
    const auto eos_type = MG_EOS_TYPE::_from_integral_unchecked(material_eos_type_host(i_mat));
    const auto material_eos_id = material_eos_id_host(i_mat);

    if (eos_type == +MG_EOS_TYPE::MG_IDEAL_GAS)
    {
      mg_eos_ig_host(material_eos_id) = MieGruneisenEosIdealGas(i_mat, config_map);
    }
    else if (eos_type == +MG_EOS_TYPE::MG_STIFFENED_GAS)
    {
      mg_eos_sg_host(material_eos_id) = MieGruneisenEosStiffenedGas(i_mat, config_map);
    }
    else if (eos_type == +MG_EOS_TYPE::MG_VANDERWAALS_GAS)
    {
      mg_eos_vdw_host(material_eos_id) = MieGruneisenEosVanDerWaalsGas(i_mat, config_map);
    }
    else if (eos_type == +MG_EOS_TYPE::MG_SHOCKWAVE)
    {
      mg_eos_sw_host(material_eos_id) = MieGruneisenEosSW(i_mat, config_map);
    }
    else if (eos_type == +MG_EOS_TYPE::MG_SHOCKWAVE2)
    {
      mg_eos_sw2_host(material_eos_id) = MieGruneisenEosSW2(i_mat, config_map);
    }
    else if (eos_type == +MG_EOS_TYPE::MG_COCHRAN_CHAN)
    {
      mg_eos_cc_host(material_eos_id) = MieGruneisenEosCochranChan(i_mat, config_map);
    }
    else if (eos_type == +MG_EOS_TYPE::MG_JWL)
    {
      mg_eos_jwl_host(material_eos_id) = MieGruneisenEosJWL(i_mat, config_map);
    }
  }

  // copy to device
  Kokkos::deep_copy(m_material_eos_id.logical_view(), material_eos_id_host.logical_view());
  Kokkos::deep_copy(m_material_eos_type.logical_view(), material_eos_type_host.logical_view());

  Kokkos::deep_copy(m_mg_eos_ig, mg_eos_ig_host);
  Kokkos::deep_copy(m_mg_eos_sg, mg_eos_sg_host);
  Kokkos::deep_copy(m_mg_eos_vdw, mg_eos_vdw_host);
  Kokkos::deep_copy(m_mg_eos_sw, mg_eos_sw_host);
  Kokkos::deep_copy(m_mg_eos_sw2, mg_eos_sw2_host);
  Kokkos::deep_copy(m_mg_eos_cc, mg_eos_cc_host);
  Kokkos::deep_copy(m_mg_eos_jwl, mg_eos_jwl_host);

  // monitoring
  if constexpr (std::is_same_v<device_t, HostDevice>)
  {
    KALYPSSO_INFO("Initializing Mie-Gruneisen EOS array on host");
  }
  else
  {
    KALYPSSO_INFO("Initializing Mie-Gruneisen EOS array on device");
  }

  for (size_t i_mat = 0; i_mat < m_num_material; ++i_mat)
  {
    const auto eos_type = MG_EOS_TYPE::_from_integral_unchecked(material_eos_type_host(i_mat));
    KALYPSSO_INFO(
      "material #{} eos={} id={}", i_mat, eos_type._to_string(), material_eos_id_host(i_mat));
    const auto material_eos_id = material_eos_id_host(i_mat);

    if (eos_type == +MG_EOS_TYPE::MG_IDEAL_GAS)
    {
      mg_eos_ig_host(material_eos_id).print();
    }
    else if (eos_type == +MG_EOS_TYPE::MG_STIFFENED_GAS)
    {
      mg_eos_sg_host(material_eos_id).print();
    }
    else if (eos_type == +MG_EOS_TYPE::MG_VANDERWAALS_GAS)
    {
      mg_eos_vdw_host(material_eos_id).print();
    }
    else if (eos_type == +MG_EOS_TYPE::MG_SHOCKWAVE)
    {
      mg_eos_sw_host(material_eos_id).print();
    }
    else if (eos_type == +MG_EOS_TYPE::MG_SHOCKWAVE2)
    {
      mg_eos_sw2_host(material_eos_id).print();
    }
    else if (eos_type == +MG_EOS_TYPE::MG_COCHRAN_CHAN)
    {
      mg_eos_cc_host(material_eos_id).print();
    }
    else if (eos_type == +MG_EOS_TYPE::MG_JWL)
    {
      mg_eos_jwl_host(material_eos_id).print();
    }
  }

} // MieGruneisenEosArray<device_t>::MieGruneisenEosArray

// =====================================================================
// =====================================================================
template <typename device_t>
real_t
MieGruneisenEosArray<device_t>::material_gruneisen_param(int i_mat, real_t rho) const
{
  // get EOS type
  const auto eos_type = MG_EOS_TYPE::_from_integral_unchecked(m_material_eos_type(i_mat));

  // get material eos id
  const auto material_eos_id = m_material_eos_id(i_mat);

  if (eos_type == +MG_EOS_TYPE::MG_IDEAL_GAS)
  {
    return m_mg_eos_ig(material_eos_id).gamma(rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_STIFFENED_GAS)
  {
    return m_mg_eos_sg(material_eos_id).gamma(rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_VANDERWAALS_GAS)
  {
    return m_mg_eos_vdw(material_eos_id).gamma(rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_SHOCKWAVE)
  {
    return m_mg_eos_sw(material_eos_id).gamma(rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_SHOCKWAVE2)
  {
    return m_mg_eos_sw2(material_eos_id).gamma(rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_JWL)
  {
    return m_mg_eos_jwl(material_eos_id).gamma(rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_COCHRAN_CHAN)
  {
    return m_mg_eos_cc(material_eos_id).gamma(rho);
  }

  return ZERO_F;

} // MieGruneisenEosArray<device_t>::material_gruneisen_param

// =====================================================================
// =====================================================================
template <typename device_t>
real_t
MieGruneisenEosArray<device_t>::material_specific_eint_ref(int i_mat, real_t rho) const
{
  // get EOS type
  const auto eos_type = MG_EOS_TYPE::_from_integral_unchecked(m_material_eos_type(i_mat));

  // get material eos id
  const auto material_eos_id = m_material_eos_id(i_mat);

  if (eos_type == +MG_EOS_TYPE::MG_IDEAL_GAS)
  {
    return m_mg_eos_ig(material_eos_id).eint_ref(rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_STIFFENED_GAS)
  {
    return m_mg_eos_sg(material_eos_id).eint_ref(rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_VANDERWAALS_GAS)
  {
    return m_mg_eos_vdw(material_eos_id).eint_ref(rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_SHOCKWAVE)
  {
    return m_mg_eos_sw(material_eos_id).eint_ref(rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_SHOCKWAVE2)
  {
    return m_mg_eos_sw2(material_eos_id).eint_ref(rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_JWL)
  {
    return m_mg_eos_jwl(material_eos_id).eint_ref(rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_COCHRAN_CHAN)
  {
    return m_mg_eos_cc(material_eos_id).eint_ref(rho);
  }

  return ZERO_F;

} // MieGruneisenEosArray<device_t>::material_specific_eint_ref

// =====================================================================
// =====================================================================
template <typename device_t>
real_t
MieGruneisenEosArray<device_t>::material_pressure_ref(int i_mat, real_t rho) const
{
  // get EOS type
  const auto eos_type = MG_EOS_TYPE::_from_integral_unchecked(m_material_eos_type(i_mat));

  // get material eos id
  const auto material_eos_id = m_material_eos_id(i_mat);

  if (eos_type == +MG_EOS_TYPE::MG_IDEAL_GAS)
  {
    return m_mg_eos_ig(material_eos_id).pressure_ref(rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_STIFFENED_GAS)
  {
    return m_mg_eos_sg(material_eos_id).pressure_ref(rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_VANDERWAALS_GAS)
  {
    return m_mg_eos_vdw(material_eos_id).pressure_ref(rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_SHOCKWAVE)
  {
    return m_mg_eos_sw(material_eos_id).pressure_ref(rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_SHOCKWAVE2)
  {
    return m_mg_eos_sw2(material_eos_id).pressure_ref(rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_JWL)
  {
    return m_mg_eos_jwl(material_eos_id).pressure_ref(rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_COCHRAN_CHAN)
  {
    return m_mg_eos_cc(material_eos_id).pressure_ref(rho);
  }

  return ZERO_F;

} // MieGruneisenEosArray<device_t>::material_pressure_ref

// =====================================================================
// =====================================================================
template <typename device_t>
real_t
MieGruneisenEosArray<device_t>::material_sound_speed_square(int    i_mat,
                                                            real_t pressure,
                                                            real_t rho) const
{
  // get EOS type
  const auto eos_type = MG_EOS_TYPE::_from_integral_unchecked(m_material_eos_type(i_mat));

  // get material eos id
  const auto material_eos_id = m_material_eos_id(i_mat);

  if (eos_type == +MG_EOS_TYPE::MG_IDEAL_GAS)
  {
    return m_mg_eos_ig(material_eos_id).sound_speed_square(pressure, rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_STIFFENED_GAS)
  {
    return m_mg_eos_sg(material_eos_id).sound_speed_square(pressure, rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_VANDERWAALS_GAS)
  {
    return m_mg_eos_vdw(material_eos_id).sound_speed_square(pressure, rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_SHOCKWAVE)
  {
    return m_mg_eos_sw(material_eos_id).sound_speed_square(pressure, rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_SHOCKWAVE2)
  {
    return m_mg_eos_sw2(material_eos_id).sound_speed_square(pressure, rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_JWL)
  {
    return m_mg_eos_jwl(material_eos_id).sound_speed_square(pressure, rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_COCHRAN_CHAN)
  {
    return m_mg_eos_cc(material_eos_id).sound_speed_square(pressure, rho);
  }

  return ZERO_F;

} // MieGruneisenEosArray<device_t>::material_sound_speed_square

// =====================================================================
// =====================================================================
template <typename device_t>
real_t
MieGruneisenEosArray<device_t>::material_bulk_modulus(int i_mat, real_t pressure, real_t rho) const
{

  // const auto c2 = material_sound_speed_square(i_mat, pressure, rho);
  // return rho * c2;

  // get EOS type
  const auto eos_type = MG_EOS_TYPE::_from_integral_unchecked(m_material_eos_type(i_mat));

  // get material eos id
  const auto material_eos_id = m_material_eos_id(i_mat);

  if (eos_type == +MG_EOS_TYPE::MG_IDEAL_GAS)
  {
    return m_mg_eos_ig(material_eos_id).bulk_modulus(pressure, rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_STIFFENED_GAS)
  {
    return m_mg_eos_sg(material_eos_id).bulk_modulus(pressure, rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_VANDERWAALS_GAS)
  {
    return m_mg_eos_vdw(material_eos_id).bulk_modulus(pressure, rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_SHOCKWAVE)
  {
    return m_mg_eos_sw(material_eos_id).bulk_modulus(pressure, rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_SHOCKWAVE2)
  {
    return m_mg_eos_sw2(material_eos_id).bulk_modulus(pressure, rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_JWL)
  {
    return m_mg_eos_jwl(material_eos_id).bulk_modulus(pressure, rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_COCHRAN_CHAN)
  {
    return m_mg_eos_cc(material_eos_id).bulk_modulus(pressure, rho);
  }

  return ZERO_F;

} // MieGruneisenEosArray<device_t>::material_bulk_modulus

// explicit template instantiation
#ifdef KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE
template class MieGruneisenEosArray<HostDevice>;
#endif
template class MieGruneisenEosArray<DefaultDevice>;

} // namespace eos
} // namespace core
} // namespace kalypsso
