// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MieGruneisenMixture.cpp
 */
#include <kalypsso/core/eos/MieGruneisenMixture.h>
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
MieGruneisenMixture<device_t>::MieGruneisenMixture(ConfigMap const & config_map)
  : m_num_material(static_cast<size_t>(config_map.getInteger("run", "nmat", 2)))
  , m_mg_eos_ig("mg_eos_ig", get_number_of_eos(config_map, MG_EOS_TYPE::MG_IDEAL_GAS))
  , m_mg_eos_sg("mg_eos_sg", get_number_of_eos(config_map, MG_EOS_TYPE::MG_STIFFENED_GAS))
  , m_mg_eos_vdw("mg_eos_vdw", get_number_of_eos(config_map, MG_EOS_TYPE::MG_VANDERWAALS_GAS))
  , m_mg_eos_sw("mg_eos_sw", get_number_of_eos(config_map, MG_EOS_TYPE::MG_SHOCKWAVE))
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
  Kokkos::deep_copy(m_mg_eos_cc, mg_eos_cc_host);
  Kokkos::deep_copy(m_mg_eos_jwl, mg_eos_jwl_host);

  // monitoring
  if constexpr (std::is_same_v<device_t, HostDevice>)
  {
    KALYPSSO_INFO("Initializing Mie-Gruneisen mixture on host");
  }
  else
  {
    KALYPSSO_INFO("Initializing Mie-Gruneisen mixture on device");
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
    else if (eos_type == +MG_EOS_TYPE::MG_COCHRAN_CHAN)
    {
      mg_eos_cc_host(material_eos_id).print();
    }
    else if (eos_type == +MG_EOS_TYPE::MG_JWL)
    {
      mg_eos_jwl_host(material_eos_id).print();
    }
  }


} // MieGruneisenMixture<device_t>::MieGruneisenMixture

// =====================================================================
// =====================================================================
template <typename device_t>
real_t
MieGruneisenMixture<device_t>::material_gruneisen_param(int i_mat, real_t rho) const
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
  else if (eos_type == +MG_EOS_TYPE::MG_JWL)
  {
    return m_mg_eos_jwl(material_eos_id).gamma(rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_COCHRAN_CHAN)
  {
    return m_mg_eos_cc(material_eos_id).gamma(rho);
  }

  return ZERO_F;

} // MieGruneisenMixture<device_t>::material_gruneisen_param

// =====================================================================
// =====================================================================
template <typename device_t>
real_t
MieGruneisenMixture<device_t>::material_specific_eint_ref(int i_mat, real_t rho) const
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
  else if (eos_type == +MG_EOS_TYPE::MG_JWL)
  {
    return m_mg_eos_jwl(material_eos_id).eint_ref(rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_COCHRAN_CHAN)
  {
    return m_mg_eos_cc(material_eos_id).eint_ref(rho);
  }

  return ZERO_F;

} // MieGruneisenMixture<device_t>::material_specific_eint_ref

// =====================================================================
// =====================================================================
template <typename device_t>
real_t
MieGruneisenMixture<device_t>::material_pressure_ref(int i_mat, real_t rho) const
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
  else if (eos_type == +MG_EOS_TYPE::MG_JWL)
  {
    return m_mg_eos_jwl(material_eos_id).pressure_ref(rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_COCHRAN_CHAN)
  {
    return m_mg_eos_cc(material_eos_id).pressure_ref(rho);
  }

  return ZERO_F;

} // MieGruneisenMixture<device_t>::material_pressure_ref

// =====================================================================
// =====================================================================
template <typename device_t>
real_t
MieGruneisenMixture<device_t>::material_sound_speed_square(int    i_mat,
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
  else if (eos_type == +MG_EOS_TYPE::MG_JWL)
  {
    return m_mg_eos_jwl(material_eos_id).sound_speed_square(pressure, rho);
  }
  else if (eos_type == +MG_EOS_TYPE::MG_COCHRAN_CHAN)
  {
    return m_mg_eos_cc(material_eos_id).sound_speed_square(pressure, rho);
  }

  return ZERO_F;

} // MieGruneisenMixture<device_t>::material_sound_speed_square

// =====================================================================
// =====================================================================
template <typename device_t>
real_t
MieGruneisenMixture<device_t>::material_bulk_modulus(int i_mat, real_t pressure, real_t rho) const
{

  const auto c2 = material_sound_speed_square(i_mat, pressure, rho);
  return rho * c2;

} // MieGruneisenMixture<device_t>::material_bulk_modulus

// =====================================================================
// =====================================================================
template <typename device_t>
real_t
MieGruneisenMixture<device_t>::mixture_gruneisen_param(real_t phi0,
                                                       real_t phi1,
                                                       real_t phi_rho0,
                                                       real_t phi_rho1) const
{
  real_t tmp = ZERO_F;

  // material 0
  // if (phi0 > LOW_PHI)
  {
    tmp += phi0 / material_gruneisen_param(0, phi_rho0 / phi0);
  }

  // material 1
  // if (phi1 > LOW_PHI)
  {
    tmp += phi1 / material_gruneisen_param(1, phi_rho1 / phi1);
  }

  return 1 / tmp;

} // MieGruneisenMixture<device_t>::mixture_gruneisen_param

// =====================================================================
// =====================================================================
template <typename device_t>
real_t
MieGruneisenMixture<device_t>::mixture_pressure(real_t rho,
                                                real_t eint,
                                                real_t phi0,
                                                real_t phi1,
                                                real_t phi_rho0,
                                                real_t phi_rho1) const
{
  // use eq (34) from Wallis et all, A unified diffuse interface method for the interaction
  // of rigid bodies with elastoplastic solids and multi-phase mixtures.
  // J. Appl. Phys. 131, 104901 (2022)
  // https://doi.org/10.1063/5.0079970

  real_t tmp = ZERO_F;

  // material 0
  const auto rho0 = phi0 > LOW_PHI ? phi_rho0 / phi0 : ONE_F;
  tmp += phi_rho0 * material_specific_eint_ref(0, rho0) -
         phi0 * material_pressure_ref(0, rho0) / material_gruneisen_param(0, rho0);


  // material 1
  const auto rho1 = phi1 > LOW_PHI ? phi_rho1 / phi1 : ONE_F;
  tmp += phi_rho1 * material_specific_eint_ref(1, rho1) -
         phi1 * material_pressure_ref(1, rho1) / material_gruneisen_param(1, rho1);

  const auto pressure_mix =
    mixture_gruneisen_param(phi0, phi1, phi_rho0, phi_rho1) * (rho * eint - tmp);

  return pressure_mix;

} // MieGruneisenMixture<device_t>::mixture_pressure

// =====================================================================
// =====================================================================
template <typename device_t>
real_t
MieGruneisenMixture<device_t>::mixture_volumic_eint([[maybe_unused]] real_t rho,
                                                    real_t                  pressure,
                                                    real_t                  phi0,
                                                    real_t                  phi1,
                                                    real_t                  phi_rho0,
                                                    real_t                  phi_rho1) const
{
  // use eq (33) from Wallis et all, A unified diffuse interface method for the interaction
  // of rigid bodies with elastoplastic solids and multi-phase mixtures.
  // J. Appl. Phys. 131, 104901 (2022)
  // https://doi.org/10.1063/5.0079970

  real_t eint = ZERO_F;

  // material 0
  const auto rho0 = phi0 > LOW_PHI ? phi_rho0 / phi0 : ONE_F;
  eint += phi_rho0 * material_specific_eint_ref(0, rho0);
  eint += phi0 * (pressure - material_pressure_ref(0, rho0)) / material_gruneisen_param(0, rho0);

  // material 1
  const auto rho1 = phi1 > LOW_PHI ? phi_rho1 / phi1 : ONE_F;
  eint += phi_rho1 * material_specific_eint_ref(1, rho1);
  eint += phi1 * (pressure - material_pressure_ref(1, rho1)) / material_gruneisen_param(1, rho1);

  return eint;

} // MieGruneisenMixture<device_t>::mixture_volumic_eint

// =====================================================================
// =====================================================================
template <typename device_t>
real_t
MieGruneisenMixture<device_t>::mixture_specific_eint(real_t rho,
                                                     real_t pressure,
                                                     real_t phi0,
                                                     real_t phi1,
                                                     real_t phi_rho0,
                                                     real_t phi_rho1) const
{
  return mixture_volumic_eint(rho, pressure, phi0, phi1, phi_rho0, phi_rho1) / rho;
} // MieGruneisenMixture<device_t>::mixture_specific_eint

// =====================================================================
// =====================================================================
template <typename device_t>
real_t
MieGruneisenMixture<device_t>::mixture_sound_speed_square(real_t rho,
                                                          real_t pressure,
                                                          real_t phi0,
                                                          real_t phi1,
                                                          real_t phi_rho0,
                                                          real_t phi_rho1) const
{
  // use eq (31) from Wallis et all, A unified diffuse interface method for the interaction
  // of rigid bodies with elastoplastic solids and multi-phase mixtures.
  // J. Appl. Phys. 131, 104901 (2022)
  // https://doi.org/10.1063/5.0079970
  //
  // The formula from this article is slightly erroneous, but we use the correct one

  real_t tmp = ZERO_F;

  // material 0
  const auto rho0 = phi0 > LOW_PHI ? phi_rho0 / phi0 : ONE_F;
  const auto Y0 = phi_rho0 / rho; // mass fraction
  tmp += Y0 * material_sound_speed_square(0, pressure, rho0) / material_gruneisen_param(0, rho0);

  // material 1
  const auto rho1 = phi1 > LOW_PHI ? phi_rho1 / phi1 : ONE_F;
  const auto Y1 = phi_rho1 / rho; // mass fraction
  tmp += Y1 * material_sound_speed_square(1, pressure, rho1) / material_gruneisen_param(1, rho1);

  return mixture_gruneisen_param(phi0, phi1, phi_rho0, phi_rho1) * tmp;

} // MieGruneisenMixture<device_t>::mixture_sound_speed_square

// =====================================================================
// =====================================================================
template <typename device_t>
real_t
MieGruneisenMixture<device_t>::mixture_sound_speed(real_t rho,
                                                   real_t pressure,
                                                   real_t phi0,
                                                   real_t phi1,
                                                   real_t phi_rho0,
                                                   real_t phi_rho1) const
{

  return sqrt(mixture_sound_speed_square(rho, pressure, phi0, phi1, phi_rho0, phi_rho1));

} // MieGruneisenMixture<device_t>::mixture_sound_speed

// =====================================================================
// =====================================================================
template <typename device_t>
real_t
MieGruneisenMixture<device_t>::mixture_bulk_modulus(real_t rho,
                                                    real_t pressure,
                                                    real_t phi0,
                                                    real_t phi1,
                                                    real_t phi_rho0,
                                                    real_t phi_rho1) const
{

  return rho * mixture_sound_speed_square(rho, pressure, phi0, phi1, phi_rho0, phi_rho1);

} // MieGruneisenMixture<device_t>::mixture_bulk_modulus

// explicit template instantiation
#ifdef KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE
template class MieGruneisenMixture<HostDevice>;
#endif
template class MieGruneisenMixture<DefaultDevice>;

} // namespace eos
} // namespace core
} // namespace kalypsso
