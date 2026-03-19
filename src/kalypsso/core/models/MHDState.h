// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MHDState.h
 */
#ifndef KALYPSSO_CORE_MODELS_MHD_STATE_H_
#define KALYPSSO_CORE_MODELS_MHD_STATE_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

template <size_t nbvar>
using StateNd = Kokkos::Array<real_t, nbvar>;

template <size_t dim>
using GradTensor = Kokkos::Array<real_t, dim * dim>;

template <size_t dim>
using DivergenceB = Kokkos::Array<real_t, dim>;

using GravityState = Kokkos::Array<real_t, 3>;

constexpr int HYDRO_NBVAR_CELL = 5;
constexpr int MHD_NBVAR_CELL = 8;
constexpr int MHD_NBVAR_FACE = 11;
constexpr int MHD_NBVAR_SPLIT = 11;

template <size_t dim>
constexpr int
nbvar_hydro_only()
{
  if constexpr (dim == 2)
    return 4;
  else if constexpr (dim == 3)
    return 5;
}

template <size_t dim>
using HydroState = Kokkos::Array<real_t, nbvar_hydro_only<dim>()>;

//! state variable array with cell-centered magnetic field
using MHDStateCell = Kokkos::Array<real_t, MHD_NBVAR_CELL>;

//! state variable array with cell-centered and split magnetic field
using MHDSplitStateCell = Kokkos::Array<real_t, MHD_NBVAR_SPLIT>;

//! state variables array with cell-centered hydrodynamics var and
//! face-centered magnetic field (left and right)
using MHDStateFace = Kokkos::Array<real_t, MHD_NBVAR_FACE>;

//! Face-centered magnetic field as a 6-component vector.
//! left and right faces in all three direction (X,Y and Z).
//!
//! designed to be use with enum FaceMagneticFieldId
using FaceMagState = Kokkos::Array<real_t, 6>;

//! magnetic field component array
using BField = Kokkos::Array<real_t, 3>;

} // namespace kalypsso

#endif // KALYPSSO_CORE_MODELS_MHD_STATE_H_
