// SPDX-FileCopyrightText: 2026 kalypsso-core authors
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
template <typename device_t>
MieGruneisenMixture<device_t>::MieGruneisenMixture(ConfigMap const & config_map)
  : m_eos_array(config_map)
  , m_small_p(config_map.getReal("mixture", "smallp", KALYPSSO_NUM(1e-9)))
{} // MieGruneisenMixture<device_t>::MieGruneisenMixture

// =====================================================================
// =====================================================================
template <typename device_t>
KOKKOS_FUNCTION real_t
MieGruneisenMixture<device_t>::mixture_gruneisen_param(real_t alpha0,
                                                       real_t alpha1,
                                                       real_t alpha_rho0,
                                                       real_t alpha_rho1) const
{
  auto tmp = ZERO_F;

  // material 0
  if (alpha0 > LOW_ALPHA)
  {
    const auto Gamma0 = m_eos_array.material_gruneisen_param(0, alpha_rho0 / alpha0);
    tmp += alpha0 / Gamma0;
  }

  // material 1
  if (alpha1 > LOW_ALPHA)
  {
    const auto Gamma1 = m_eos_array.material_gruneisen_param(1, alpha_rho1 / alpha1);
    tmp += alpha1 / Gamma1;
  }

  return ONE_F / tmp;

} // MieGruneisenMixture<device_t>::mixture_gruneisen_param

// =====================================================================
// =====================================================================
template <typename device_t>
KOKKOS_FUNCTION real_t
MieGruneisenMixture<device_t>::mixture_pressure(real_t rho,
                                                real_t eint,
                                                real_t alpha0,
                                                real_t alpha1,
                                                real_t alpha_rho0,
                                                real_t alpha_rho1) const
{
  // use eq (34) from Wallis et all, A unified diffuse interface method for the interaction
  // of rigid bodies with elastoplastic solids and multi-phase mixtures.
  // J. Appl. Phys. 131, 104901 (2022)
  // https://doi.org/10.1063/5.0079970

  auto tmp = ZERO_F;

  // material 0
  const auto rho0 = alpha0 > LOW_ALPHA ? alpha_rho0 / alpha0 : ONE_F;
  const auto eint0 = m_eos_array.material_specific_eint_ref(0, rho0);
  const auto pref0 = m_eos_array.material_pressure_ref(0, rho0);
  const auto Gamma0 = m_eos_array.material_gruneisen_param(0, rho0);
  tmp += alpha0 > LOW_ALPHA ? alpha_rho0 * eint0 - alpha0 * pref0 / Gamma0 : ZERO_F;

  // material 1
  const auto rho1 = alpha1 > LOW_ALPHA ? alpha_rho1 / alpha1 : ONE_F;
  const auto eint1 = m_eos_array.material_specific_eint_ref(1, rho1);
  const auto pref1 = m_eos_array.material_pressure_ref(1, rho1);
  const auto Gamma1 = m_eos_array.material_gruneisen_param(1, rho1);
  tmp += alpha1 > LOW_ALPHA ? alpha_rho1 * eint1 - alpha1 * pref1 / Gamma1 : ZERO_F;

  const auto pressure_mix =
    mixture_gruneisen_param(alpha0, alpha1, alpha_rho0, alpha_rho1) * (rho * eint - tmp);

  // return pressure_mix;
  return pressure_mix < m_small_p ? m_small_p : pressure_mix;

} // MieGruneisenMixture<device_t>::mixture_pressure

// =====================================================================
// =====================================================================
template <typename device_t>
KOKKOS_FUNCTION real_t
MieGruneisenMixture<device_t>::mixture_volumic_eint([[maybe_unused]] real_t rho,
                                                    real_t                  pressure,
                                                    real_t                  alpha0,
                                                    real_t                  alpha1,
                                                    real_t                  alpha_rho0,
                                                    real_t                  alpha_rho1) const
{
  // use eq (33) from Wallis et all, A unified diffuse interface method for the interaction
  // of rigid bodies with elastoplastic solids and multi-phase mixtures.
  // J. Appl. Phys. 131, 104901 (2022)
  // https://doi.org/10.1063/5.0079970

  real_t eint = ZERO_F;

  // material 0
  if (alpha0 > LOW_ALPHA)
  {
    const auto rho0 = alpha_rho0 / alpha0;
    const auto eint0 = m_eos_array.material_specific_eint_ref(0, rho0);
    eint += alpha_rho0 * eint0;
    eint += alpha0 * (pressure - m_eos_array.material_pressure_ref(0, rho0)) /
            m_eos_array.material_gruneisen_param(0, rho0);
  }

  // material 1
  if (alpha1 > LOW_ALPHA)
  {
    const auto rho1 = alpha_rho1 / alpha1;
    const auto eint1 = m_eos_array.material_specific_eint_ref(1, rho1);
    eint += alpha_rho1 * eint1;
    eint += alpha1 * (pressure - m_eos_array.material_pressure_ref(1, rho1)) /
            m_eos_array.material_gruneisen_param(1, rho1);
  }
  return eint;

} // MieGruneisenMixture<device_t>::mixture_volumic_eint

// =====================================================================
// =====================================================================
template <typename device_t>
KOKKOS_FUNCTION real_t
MieGruneisenMixture<device_t>::mixture_specific_eint(real_t rho,
                                                     real_t pressure,
                                                     real_t alpha0,
                                                     real_t alpha1,
                                                     real_t alpha_rho0,
                                                     real_t alpha_rho1) const
{
  return mixture_volumic_eint(rho, pressure, alpha0, alpha1, alpha_rho0, alpha_rho1) / rho;
} // MieGruneisenMixture<device_t>::mixture_specific_eint

// =====================================================================
// =====================================================================
template <typename device_t>
KOKKOS_FUNCTION real_t
MieGruneisenMixture<device_t>::mixture_sound_speed_square(real_t rho,
                                                          real_t pressure,
                                                          real_t alpha0,
                                                          real_t alpha1,
                                                          real_t alpha_rho0,
                                                          real_t alpha_rho1) const
{
  // use eq (31) from Wallis et all, A unified diffuse interface method for the interaction
  // of rigid bodies with elastoplastic solids and multi-phase mixtures.
  // J. Appl. Phys. 131, 104901 (2022)
  // https://doi.org/10.1063/5.0079970
  //
  // Important note:
  // - The formula from this article is slightly erroneous (additional volume fraction at
  // numerator shouldn't exist)
  // - the formula is slightly rewritten to treat bulk modulus (rho*c^2) as whole; the formula is
  // much more numerically stable as it avoid using rho0 and rho1 which may be ill-defined in
  // regions where volume fraction is very small

  auto tmp = ZERO_F;

  // material 0
  if (alpha0 > LOW_ALPHA)
  {
    const auto rho0 = alpha_rho0 / alpha0;
    tmp += alpha0 * m_eos_array.material_bulk_modulus(0, pressure, rho0) /
           m_eos_array.material_gruneisen_param(0, rho0);
  }

  // material 1
  if (alpha1 > LOW_ALPHA)
  {
    const auto rho1 = alpha_rho1 / alpha1;
    tmp += alpha1 * m_eos_array.material_bulk_modulus(1, pressure, rho1) /
           m_eos_array.material_gruneisen_param(1, rho1);
  }

  return mixture_gruneisen_param(alpha0, alpha1, alpha_rho0, alpha_rho1) / rho * tmp;

} // MieGruneisenMixture<device_t>::mixture_sound_speed_square

// =====================================================================
// =====================================================================
template <typename device_t>
KOKKOS_FUNCTION real_t
MieGruneisenMixture<device_t>::mixture_sound_speed(real_t rho,
                                                   real_t pressure,
                                                   real_t alpha0,
                                                   real_t alpha1,
                                                   real_t alpha_rho0,
                                                   real_t alpha_rho1) const
{

  return sqrt(mixture_sound_speed_square(rho, pressure, alpha0, alpha1, alpha_rho0, alpha_rho1));

} // MieGruneisenMixture<device_t>::mixture_sound_speed

// =====================================================================
// =====================================================================
template <typename device_t>
KOKKOS_FUNCTION real_t
MieGruneisenMixture<device_t>::mixture_bulk_modulus(real_t rho,
                                                    real_t pressure,
                                                    real_t alpha0,
                                                    real_t alpha1,
                                                    real_t alpha_rho0,
                                                    real_t alpha_rho1) const
{

  return rho * mixture_sound_speed_square(rho, pressure, alpha0, alpha1, alpha_rho0, alpha_rho1);

} // MieGruneisenMixture<device_t>::mixture_bulk_modulus

// explicit template instantiation
#ifdef KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE
template class MieGruneisenMixture<HostDevice>;
#endif
template class MieGruneisenMixture<DefaultDevice>;

} // namespace eos
} // namespace core
} // namespace kalypsso
