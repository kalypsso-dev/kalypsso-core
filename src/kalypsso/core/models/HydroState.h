// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file HydroState.h
 */
#ifndef KALYPSSO_CORE_MODELS_HYDRO_STATE_H_
#define KALYPSSO_CORE_MODELS_HYDRO_STATE_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

constexpr int HYDRO_2D_NBVAR = 4;
constexpr int HYDRO_3D_NBVAR = 5;

template <size_t dim>
constexpr int
nbvar_hydro()
{
  if constexpr (dim == 2)
    return HYDRO_2D_NBVAR;
  else
    return HYDRO_3D_NBVAR;
}

template <size_t nbvar>
using StateNd = Kokkos::Array<real_t, nbvar>;

template <size_t dim>
using HydroState = StateNd<nbvar_hydro<dim>()>;

template <size_t dim>
using GradTensor = Kokkos::Array<real_t, dim * dim>;

using HydroState2d = HydroState<2>;
using HydroState3d = HydroState<3>;
using GravityState = Kokkos::Array<real_t, 3>;

constexpr int HYDRO_NBVAR_CELL = 5;
} // namespace kalypsso

#endif // KALYPSSO_CORE_MODELS_HYDRO_STATE_H_
