// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file IdealGasMixture.h
 *
 * This is only useful when using at least two material to define pressure and
 * internal  energy in mixed cells.
 */
#ifndef KALYPSSO_CORE_EOS_IDEAL_GAS_MIXTURE_H_
#define KALYPSSO_CORE_EOS_IDEAL_GAS_MIXTURE_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/eos/eos_utils.h>

#include <kalypsso/core/eos/IdealGasEos.h>

namespace kalypsso
{

namespace core
{

namespace eos
{

//! type alias for an array of equation of state
template <typename device_t>
using ig_eos_array_t = Kokkos::View<IdealGasEos *, device_t>;

// ==========================================================================
// ==========================================================================
// ==========================================================================
/**
 * Ideal gas mixture helper class.
 *
 * In a multifluid simulation, when each fluid obeys a ideal gas EOS, by weighting each eos by
 * the the volume fraction \f$z_i\f$, summing all equations and using isobaric closure, one obtains
 * the mixture pressure:
 *
 * \f$ p = \frac{\rho e}{\epsilon} \f$,
 *
 * where \f$ \epsilon = \sum_i \frac{z_i}{\gamma_i-1}\f$
 *
 * The mixture quantities are:
 *
 * - volumic fraction conservation : \f$ \sum_i \phi_i = 1 \f$
 * - density : \f$ \rho = \sum_i \phi_i \rho_i \f$
 * - internal energy : \f$ e_{int} = \sum_i $\phi_i eint_i \f$
 * - pressure : \f$ \forall i, p_i = p \f$
 *
 */
template <typename device_t>
class IdealGasMixture
{

public:
  IdealGasMixture() = default;

  IdealGasMixture(const ConfigMap & config_map)
    : m_num_material(static_cast<size_t>(config_map.getInteger("run", "nmat", 2)))
    , m_ig_eos("eos ideal gas eos array", m_num_material)
    , m_small_p(config_map.getReal("ideal_gas", "mixture_smallp", KALYPSSO_NUM(1e-9)))
  {
    const auto eos_type = get_eos_type(config_map);

    // only initialize eos array if ideal gas is requested
    if (eos_type == +EOS_TYPE::IDEAL_GAS)
    {
      // create ideal eos array
      auto ig_eos_host = Kokkos::create_mirror_view(m_ig_eos);
      for (size_t i_mat = 0; i_mat < m_num_material; i_mat++)
        ig_eos_host(i_mat) = IdealGasEos(i_mat, config_map);
      Kokkos::deep_copy(m_ig_eos, ig_eos_host);
    }
  }

  //! number of materials
  size_t m_num_material;

  //! array of equation of states, one per material
  ig_eos_array_t<device_t> m_ig_eos;

  //! small pressure guard
  const real_t m_small_p;

  /**
   * Compute mixture pressure.
   *
   * To be used only when there are two materials
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  mixture_pressure(real_t rho, real_t eint, real_t phi0) const
  {
    const real_t epsilon = compute_epsilon(phi0);
    const auto   pressure = rho * eint / epsilon;
    return pressure < m_small_p ? m_small_p : pressure;
  }

  /**
   * Compute mixture specific internal energy.
   *
   * To be used only when there are two materials
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  mixture_specific_eint(real_t pressure, real_t rho, real_t phi0) const
  {
    const real_t epsilon = compute_epsilon(phi0);
    return pressure * epsilon / rho;
  }

  /**
   * Compute mixture speed of sound.
   *
   * To be used only when there are two materials
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  mixture_sound_speed(real_t pressure, real_t rho, real_t phi0) const
  {
    const real_t epsilon = compute_epsilon(phi0);
    return sqrt(1 / rho * (pressure * (1 + 1 / epsilon)));
  }

private:
  /**
   * Compute epsilon (mixture gas) from volumic fraction (material 0, phi = phi_0).
   *
   * To be used only when there are two materials.
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  compute_epsilon(real_t phi0) const
  {
    return phi0 * m_ig_eos(0).one_over_gammam1() + (1 - phi0) * m_ig_eos(1).one_over_gammam1();
  }

}; // struct IdealGasMixture

} // namespace eos

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_EOS_IDEAL_GAS_MIXTURE_H_
