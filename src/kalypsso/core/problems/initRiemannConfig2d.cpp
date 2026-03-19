// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/core/problems/initRiemannConfig2d.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

// =====================================================================
// =====================================================================
KOKKOS_FUNCTION
RiemannConfig<2>::HydroStates_t
getRiemannConfig2d(int numConfig)
{
  constexpr auto ID = core::models::Hydro::ID;
  constexpr auto IP = core::models::Hydro::IP;
  constexpr auto IU = core::models::Hydro::IU;
  constexpr auto IV = core::models::Hydro::IV;
  // constexpr auto IW = core::models::Hydro::IW;

  RiemannConfig<2>::HydroStates_t  Us;
  RiemannConfig<2>::HydroState_t & U0 = Us[0];
  RiemannConfig<2>::HydroState_t & U1 = Us[1];
  RiemannConfig<2>::HydroState_t & U2 = Us[2];
  RiemannConfig<2>::HydroState_t & U3 = Us[3];

  switch (numConfig)
  {

    case 0:
      // Config 1
      U0[ID] = KALYPSSO_NUM(1.0);
      U0[IU] = KALYPSSO_NUM(0.0);
      U0[IV] = KALYPSSO_NUM(0.0);
      U0[IP] = KALYPSSO_NUM(1.0);

      U1[ID] = KALYPSSO_NUM(0.5197);
      U1[IU] = KALYPSSO_NUM(-0.7259);
      U1[IV] = KALYPSSO_NUM(0.0);
      U1[IP] = KALYPSSO_NUM(0.4);

      U2[ID] = KALYPSSO_NUM(0.1072);
      U2[IU] = KALYPSSO_NUM(-0.7259);
      U2[IV] = KALYPSSO_NUM(-1.4045);
      U2[IP] = KALYPSSO_NUM(0.0439);

      U3[ID] = KALYPSSO_NUM(0.2579);
      U3[IU] = KALYPSSO_NUM(0.0);
      U3[IV] = KALYPSSO_NUM(-1.4045);
      U3[IP] = KALYPSSO_NUM(0.15);

      break;
    case 1:
      // Config 2
      U0[ID] = KALYPSSO_NUM(1.0);
      U0[IU] = KALYPSSO_NUM(0.0);
      U0[IV] = KALYPSSO_NUM(0.0);
      U0[IP] = KALYPSSO_NUM(1.0);

      U1[ID] = KALYPSSO_NUM(0.5197);
      U1[IU] = KALYPSSO_NUM(-0.7259);
      U1[IV] = KALYPSSO_NUM(0.0);
      U1[IP] = KALYPSSO_NUM(0.4);

      U2[ID] = KALYPSSO_NUM(1.0);
      U2[IU] = KALYPSSO_NUM(-0.7259);
      U2[IV] = KALYPSSO_NUM(-0.7259);
      U2[IP] = KALYPSSO_NUM(1.0);

      U3[ID] = KALYPSSO_NUM(0.5197);
      U3[IU] = KALYPSSO_NUM(0.0);
      U3[IV] = KALYPSSO_NUM(-0.7259);
      U3[IP] = KALYPSSO_NUM(0.4);

      break;
    case 2:
      // Config 3
      U0[ID] = KALYPSSO_NUM(1.5);
      U0[IU] = KALYPSSO_NUM(0.0);
      U0[IV] = KALYPSSO_NUM(0.0);
      U0[IP] = KALYPSSO_NUM(1.5);

      U1[ID] = KALYPSSO_NUM(0.5323);
      U1[IU] = KALYPSSO_NUM(1.206);
      U1[IV] = KALYPSSO_NUM(0.0);
      U1[IP] = KALYPSSO_NUM(0.3);

      U2[ID] = KALYPSSO_NUM(0.138);
      U2[IU] = KALYPSSO_NUM(1.206);
      U2[IV] = KALYPSSO_NUM(1.206);
      U2[IP] = KALYPSSO_NUM(0.029);

      U3[ID] = KALYPSSO_NUM(0.5323);
      U3[IU] = KALYPSSO_NUM(0.0);
      U3[IV] = KALYPSSO_NUM(1.206);
      U3[IP] = KALYPSSO_NUM(0.3);

      break;
    case 3:
      // Config 4
      U0[ID] = KALYPSSO_NUM(1.1);
      U0[IU] = KALYPSSO_NUM(0.0);
      U0[IV] = KALYPSSO_NUM(0.0);
      U0[IP] = KALYPSSO_NUM(1.1);

      U1[ID] = KALYPSSO_NUM(0.5065);
      U1[IU] = KALYPSSO_NUM(0.8939);
      U1[IV] = KALYPSSO_NUM(0.0);
      U1[IP] = KALYPSSO_NUM(0.35);

      U2[ID] = KALYPSSO_NUM(1.1);
      U2[IU] = KALYPSSO_NUM(0.8939);
      U2[IV] = KALYPSSO_NUM(0.8939);
      U2[IP] = KALYPSSO_NUM(1.1);

      U3[ID] = KALYPSSO_NUM(0.5065);
      U3[IU] = KALYPSSO_NUM(0.0);
      U3[IV] = KALYPSSO_NUM(0.8939);
      U3[IP] = KALYPSSO_NUM(0.35);

      break;
    case 4:
      // Config 5
      U0[ID] = KALYPSSO_NUM(1.0);
      U0[IU] = KALYPSSO_NUM(-0.75);
      U0[IV] = KALYPSSO_NUM(-0.5);
      U0[IP] = KALYPSSO_NUM(1.0);

      U1[ID] = KALYPSSO_NUM(2.0);
      U1[IU] = KALYPSSO_NUM(-0.75);
      U1[IV] = KALYPSSO_NUM(0.5);
      U1[IP] = KALYPSSO_NUM(1.0);

      U2[ID] = KALYPSSO_NUM(1.0);
      U2[IU] = KALYPSSO_NUM(0.75);
      U2[IV] = KALYPSSO_NUM(0.5);
      U2[IP] = KALYPSSO_NUM(1.0);

      U3[ID] = KALYPSSO_NUM(3.0);
      U3[IU] = KALYPSSO_NUM(0.75);
      U3[IV] = KALYPSSO_NUM(-0.5);
      U3[IP] = KALYPSSO_NUM(1.0);

      break;
    case 5:
      // Config 6
      U0[ID] = KALYPSSO_NUM(1.0);
      U0[IU] = KALYPSSO_NUM(0.75);
      U0[IV] = KALYPSSO_NUM(-0.5);
      U0[IP] = KALYPSSO_NUM(1.0);

      U1[ID] = KALYPSSO_NUM(2.0);
      U1[IU] = KALYPSSO_NUM(0.75);
      U1[IV] = KALYPSSO_NUM(0.5);
      U1[IP] = KALYPSSO_NUM(0.5);

      U2[ID] = KALYPSSO_NUM(1.0);
      U2[IU] = KALYPSSO_NUM(-0.75);
      U2[IV] = KALYPSSO_NUM(0.5);
      U2[IP] = KALYPSSO_NUM(1.0);

      U3[ID] = KALYPSSO_NUM(3.0);
      U3[IU] = KALYPSSO_NUM(-0.75);
      U3[IV] = KALYPSSO_NUM(-0.5);
      U3[IP] = KALYPSSO_NUM(1.0);

      break;
    case 6:
      // Config 7
      U0[ID] = KALYPSSO_NUM(1.0);
      U0[IU] = KALYPSSO_NUM(0.1);
      U0[IV] = KALYPSSO_NUM(0.1);
      U0[IP] = KALYPSSO_NUM(1.0);

      U1[ID] = KALYPSSO_NUM(0.5197);
      U1[IU] = KALYPSSO_NUM(-0.6259);
      U1[IV] = KALYPSSO_NUM(0.1);
      U1[IP] = KALYPSSO_NUM(0.4);

      U2[ID] = KALYPSSO_NUM(0.8);
      U2[IU] = KALYPSSO_NUM(0.1);
      U2[IV] = KALYPSSO_NUM(0.1);
      U2[IP] = KALYPSSO_NUM(0.4);

      U3[ID] = KALYPSSO_NUM(0.5197);
      U3[IU] = KALYPSSO_NUM(0.1);
      U3[IV] = KALYPSSO_NUM(-0.6259);
      U3[IP] = KALYPSSO_NUM(0.4);

      break;
    case 7:
      // Config 8
      U0[ID] = KALYPSSO_NUM(0.5197);
      U0[IU] = KALYPSSO_NUM(0.1);
      U0[IV] = KALYPSSO_NUM(0.1);
      U0[IP] = KALYPSSO_NUM(0.4);

      U1[ID] = KALYPSSO_NUM(1.0);
      U1[IU] = KALYPSSO_NUM(-0.6259);
      U1[IV] = KALYPSSO_NUM(0.1);
      U1[IP] = KALYPSSO_NUM(1.0);

      U2[ID] = KALYPSSO_NUM(0.8);
      U2[IU] = KALYPSSO_NUM(0.1);
      U2[IV] = KALYPSSO_NUM(0.1);
      U2[IP] = KALYPSSO_NUM(1.0);

      U3[ID] = KALYPSSO_NUM(1.0);
      U3[IU] = KALYPSSO_NUM(0.1);
      U3[IV] = KALYPSSO_NUM(-0.6259);
      U3[IP] = KALYPSSO_NUM(1.0);

      break;
    case 8:
      // Config 9
      U0[ID] = KALYPSSO_NUM(1.0);
      U0[IU] = KALYPSSO_NUM(0.0);
      U0[IV] = KALYPSSO_NUM(0.3);
      U0[IP] = KALYPSSO_NUM(1.0);

      U1[ID] = KALYPSSO_NUM(2.0);
      U1[IU] = KALYPSSO_NUM(0.0);
      U1[IV] = KALYPSSO_NUM(-0.3);
      U1[IP] = KALYPSSO_NUM(1.0);

      U2[ID] = KALYPSSO_NUM(1.039);
      U2[IU] = KALYPSSO_NUM(0.0);
      U2[IV] = KALYPSSO_NUM(-0.8133);
      U2[IP] = KALYPSSO_NUM(0.4);

      U3[ID] = KALYPSSO_NUM(0.5197);
      U3[IU] = KALYPSSO_NUM(0.0);
      U3[IV] = KALYPSSO_NUM(-0.4259);
      U3[IP] = KALYPSSO_NUM(0.4);

      break;
    case 9:
      // Config 10
      U0[ID] = KALYPSSO_NUM(1.0);
      U0[IU] = KALYPSSO_NUM(0.0);
      U0[IV] = KALYPSSO_NUM(0.4297);
      U0[IP] = KALYPSSO_NUM(1.0);

      U1[ID] = KALYPSSO_NUM(0.5);
      U1[IU] = KALYPSSO_NUM(0.0);
      U1[IV] = KALYPSSO_NUM(0.6076);
      U1[IP] = KALYPSSO_NUM(1.0);

      U2[ID] = KALYPSSO_NUM(0.2281);
      U2[IU] = KALYPSSO_NUM(0.0);
      U2[IV] = KALYPSSO_NUM(-0.6076);
      U2[IP] = KALYPSSO_NUM(0.3333);

      U3[ID] = KALYPSSO_NUM(0.4562);
      U3[IU] = KALYPSSO_NUM(0.0);
      U3[IV] = KALYPSSO_NUM(-0.4259);
      U3[IP] = KALYPSSO_NUM(0.3333);

      break;
    case 10:
      // Config 11
      U0[ID] = KALYPSSO_NUM(1.0);
      U0[IU] = KALYPSSO_NUM(0.1);
      U0[IV] = KALYPSSO_NUM(0.0);
      U0[IP] = KALYPSSO_NUM(1.0);

      U1[ID] = KALYPSSO_NUM(0.5313);
      U1[IU] = KALYPSSO_NUM(0.8276);
      U1[IV] = KALYPSSO_NUM(0.0);
      U1[IP] = KALYPSSO_NUM(0.4);

      U2[ID] = KALYPSSO_NUM(0.8);
      U2[IU] = KALYPSSO_NUM(0.1);
      U2[IV] = KALYPSSO_NUM(0.0);
      U2[IP] = KALYPSSO_NUM(0.4);

      U3[ID] = KALYPSSO_NUM(0.5313);
      U3[IU] = KALYPSSO_NUM(0.1);
      U3[IV] = KALYPSSO_NUM(0.7276);
      U3[IP] = KALYPSSO_NUM(0.4);

      break;
    case 11:
      // Config 12
      U0[ID] = KALYPSSO_NUM(0.5313);
      U0[IU] = KALYPSSO_NUM(0.0);
      U0[IV] = KALYPSSO_NUM(0.0);
      U0[IP] = KALYPSSO_NUM(0.4);

      U1[ID] = KALYPSSO_NUM(1.0);
      U1[IU] = KALYPSSO_NUM(0.7276);
      U1[IV] = KALYPSSO_NUM(0.0);
      U1[IP] = KALYPSSO_NUM(1.0);

      U2[ID] = KALYPSSO_NUM(0.8);
      U2[IU] = KALYPSSO_NUM(0.0);
      U2[IV] = KALYPSSO_NUM(0.0);
      U2[IP] = KALYPSSO_NUM(1.0);

      U3[ID] = KALYPSSO_NUM(1.0);
      U3[IU] = KALYPSSO_NUM(0.0);
      U3[IV] = KALYPSSO_NUM(0.7276);
      U3[IP] = KALYPSSO_NUM(1.0);

      break;
    case 12:
      // Config 13
      U0[ID] = KALYPSSO_NUM(1.0);
      U0[IU] = KALYPSSO_NUM(0.0);
      U0[IV] = KALYPSSO_NUM(-0.3);
      U0[IP] = KALYPSSO_NUM(1.0);

      U1[ID] = KALYPSSO_NUM(2.0);
      U1[IU] = KALYPSSO_NUM(0.0);
      U1[IV] = KALYPSSO_NUM(0.3);
      U1[IP] = KALYPSSO_NUM(1.0);

      U2[ID] = KALYPSSO_NUM(1.0625);
      U2[IU] = KALYPSSO_NUM(0.0);
      U2[IV] = KALYPSSO_NUM(0.8145);
      U2[IP] = KALYPSSO_NUM(0.4);

      U3[ID] = KALYPSSO_NUM(0.5313);
      U3[IU] = KALYPSSO_NUM(0.0);
      U3[IV] = KALYPSSO_NUM(0.4276);
      U3[IP] = KALYPSSO_NUM(0.4);

      break;
    case 13:
      // Config 14
      U0[ID] = KALYPSSO_NUM(2.0);
      U0[IU] = KALYPSSO_NUM(0.0);
      U0[IV] = KALYPSSO_NUM(-0.5606);
      U0[IP] = KALYPSSO_NUM(8.0);

      U1[ID] = KALYPSSO_NUM(1.0);
      U1[IU] = KALYPSSO_NUM(0.0);
      U1[IV] = KALYPSSO_NUM(-1.2172);
      U1[IP] = KALYPSSO_NUM(8.0);

      U2[ID] = KALYPSSO_NUM(0.4736);
      U2[IU] = KALYPSSO_NUM(0.0);
      U2[IV] = KALYPSSO_NUM(1.2172);
      U2[IP] = KALYPSSO_NUM(2.6667);

      U3[ID] = KALYPSSO_NUM(0.9474);
      U3[IU] = KALYPSSO_NUM(0.0);
      U3[IV] = KALYPSSO_NUM(1.1606);
      U3[IP] = KALYPSSO_NUM(2.6667);

      break;
    case 14:
      // Config 15
      U0[ID] = KALYPSSO_NUM(1.0);
      U0[IU] = KALYPSSO_NUM(0.1);
      U0[IV] = KALYPSSO_NUM(-0.3);
      U0[IP] = KALYPSSO_NUM(1.0);

      U1[ID] = KALYPSSO_NUM(0.5197);
      U1[IU] = KALYPSSO_NUM(-0.6259);
      U1[IV] = KALYPSSO_NUM(-0.3);
      U1[IP] = KALYPSSO_NUM(0.4);

      U2[ID] = KALYPSSO_NUM(0.8);
      U2[IU] = KALYPSSO_NUM(0.1);
      U2[IV] = KALYPSSO_NUM(-0.3);
      U2[IP] = KALYPSSO_NUM(0.4);

      U3[ID] = KALYPSSO_NUM(0.5313);
      U3[IU] = KALYPSSO_NUM(0.1);
      U3[IV] = KALYPSSO_NUM(0.4276);
      U3[IP] = KALYPSSO_NUM(0.4);

      break;
    case 15:
      // Config 16
      U0[ID] = KALYPSSO_NUM(0.5313);
      U0[IU] = KALYPSSO_NUM(0.1);
      U0[IV] = KALYPSSO_NUM(0.1);
      U0[IP] = KALYPSSO_NUM(0.4);

      U1[ID] = KALYPSSO_NUM(1.0222);
      U1[IU] = KALYPSSO_NUM(-0.6179);
      U1[IV] = KALYPSSO_NUM(0.1);
      U1[IP] = KALYPSSO_NUM(1.0);

      U2[ID] = KALYPSSO_NUM(0.8);
      U2[IU] = KALYPSSO_NUM(0.1);
      U2[IV] = KALYPSSO_NUM(0.1);
      U2[IP] = KALYPSSO_NUM(1.0);

      U3[ID] = KALYPSSO_NUM(1.0);
      U3[IU] = KALYPSSO_NUM(0.1);
      U3[IV] = KALYPSSO_NUM(0.8276);
      U3[IP] = KALYPSSO_NUM(1.0);

      break;
    case 16:
      // Config 17
      U0[ID] = KALYPSSO_NUM(1.0);
      U0[IU] = KALYPSSO_NUM(0.0);
      U0[IV] = KALYPSSO_NUM(-0.4);
      U0[IP] = KALYPSSO_NUM(1.0);

      U1[ID] = KALYPSSO_NUM(2.0);
      U1[IU] = KALYPSSO_NUM(0.0);
      U1[IV] = KALYPSSO_NUM(-0.3);
      U1[IP] = KALYPSSO_NUM(1.0);

      U2[ID] = KALYPSSO_NUM(1.0625);
      U2[IU] = KALYPSSO_NUM(0.0);
      U2[IV] = KALYPSSO_NUM(0.2145);
      U2[IP] = KALYPSSO_NUM(0.4);

      U3[ID] = KALYPSSO_NUM(0.5197);
      U3[IU] = KALYPSSO_NUM(0.0);
      U3[IV] = KALYPSSO_NUM(-1.1259);
      U3[IP] = KALYPSSO_NUM(0.4);

      break;
    case 17:
      // Config 18
      U0[ID] = KALYPSSO_NUM(1.0);
      U0[IU] = KALYPSSO_NUM(0.0);
      U0[IV] = KALYPSSO_NUM(1.0);
      U0[IP] = KALYPSSO_NUM(1.0);

      U1[ID] = KALYPSSO_NUM(2.0);
      U1[IU] = KALYPSSO_NUM(0.0);
      U1[IV] = KALYPSSO_NUM(-0.3);
      U1[IP] = KALYPSSO_NUM(1.0);

      U2[ID] = KALYPSSO_NUM(1.0625);
      U2[IU] = KALYPSSO_NUM(0.0);
      U2[IV] = KALYPSSO_NUM(0.2145);
      U2[IP] = KALYPSSO_NUM(0.4);

      U3[ID] = KALYPSSO_NUM(0.5197);
      U3[IU] = KALYPSSO_NUM(0.0);
      U3[IV] = KALYPSSO_NUM(0.2741);
      U3[IP] = KALYPSSO_NUM(0.4);

      break;
    case 18:
      // Config 19
      U0[ID] = KALYPSSO_NUM(1.0);
      U0[IU] = KALYPSSO_NUM(0.0);
      U0[IV] = KALYPSSO_NUM(0.3);
      U0[IP] = KALYPSSO_NUM(1.0);

      U1[ID] = KALYPSSO_NUM(2.0);
      U1[IU] = KALYPSSO_NUM(0.0);
      U1[IV] = KALYPSSO_NUM(-0.3);
      U1[IP] = KALYPSSO_NUM(1.0);

      U2[ID] = KALYPSSO_NUM(1.0625);
      U2[IU] = KALYPSSO_NUM(0.0);
      U2[IV] = KALYPSSO_NUM(0.2145);
      U2[IP] = KALYPSSO_NUM(0.4);

      U3[ID] = KALYPSSO_NUM(0.5197);
      U3[IU] = KALYPSSO_NUM(0.0);
      U3[IV] = KALYPSSO_NUM(-0.4259);
      U3[IP] = KALYPSSO_NUM(0.4);

      break;
    default:
      // Config 1
      U0[ID] = KALYPSSO_NUM(1.0);
      U0[IU] = KALYPSSO_NUM(0.0);
      U0[IV] = KALYPSSO_NUM(0.0);
      U0[IP] = KALYPSSO_NUM(1.0);

      U1[ID] = KALYPSSO_NUM(0.5197);
      U1[IU] = KALYPSSO_NUM(-0.7259);
      U1[IV] = KALYPSSO_NUM(0.0);
      U1[IP] = KALYPSSO_NUM(0.4);

      U2[ID] = KALYPSSO_NUM(0.1072);
      U2[IU] = KALYPSSO_NUM(-0.7259);
      U2[IV] = KALYPSSO_NUM(-1.4045);
      U2[IP] = KALYPSSO_NUM(0.0439);

      U3[ID] = KALYPSSO_NUM(0.2579);
      U3[IU] = KALYPSSO_NUM(0.0);
      U3[IV] = KALYPSSO_NUM(-1.4045);
      U3[IP] = KALYPSSO_NUM(0.15);
  }

  return Us;

} // getRiemannConfig2d

// =====================================================================
// =====================================================================
KOKKOS_FUNCTION
RiemannConfig<3>::HydroStates_t
getRiemannConfig3d(int numConfig)
{
  constexpr auto ID = core::models::Hydro::ID;
  constexpr auto IP = core::models::Hydro::IP;
  constexpr auto IU = core::models::Hydro::IU;
  constexpr auto IV = core::models::Hydro::IV;
  constexpr auto IW = core::models::Hydro::IW;

  RiemannConfig<3>::HydroStates_t  Us;
  RiemannConfig<3>::HydroState_t & U0 = Us[0];
  RiemannConfig<3>::HydroState_t & U1 = Us[1];
  RiemannConfig<3>::HydroState_t & U2 = Us[2];
  RiemannConfig<3>::HydroState_t & U3 = Us[3];
  RiemannConfig<3>::HydroState_t & U4 = Us[4];
  RiemannConfig<3>::HydroState_t & U5 = Us[5];
  RiemannConfig<3>::HydroState_t & U6 = Us[6];
  RiemannConfig<3>::HydroState_t & U7 = Us[7];

  // only 1 config supported for now
  // if you feel an interesting config is missing, please help implement

  switch (numConfig)
  {

    case 2:
      // Config 3
      U0[ID] = KALYPSSO_NUM(0.5323);
      U0[IU] = KALYPSSO_NUM(0.0);
      U0[IV] = KALYPSSO_NUM(0.0);
      U0[IW] = KALYPSSO_NUM(1.206);
      U0[IP] = KALYPSSO_NUM(0.3);

      U1[ID] = KALYPSSO_NUM(0.138);
      U1[IU] = KALYPSSO_NUM(1.206);
      U1[IV] = KALYPSSO_NUM(0.0);
      U1[IW] = KALYPSSO_NUM(1.206);
      U1[IP] = KALYPSSO_NUM(0.029);

      U2[ID] = KALYPSSO_NUM(0.138);
      U2[IU] = KALYPSSO_NUM(1.206);
      U2[IV] = KALYPSSO_NUM(1.206);
      U2[IW] = KALYPSSO_NUM(1.206);
      U2[IP] = KALYPSSO_NUM(0.029);

      U3[ID] = KALYPSSO_NUM(0.138);
      U3[IU] = KALYPSSO_NUM(0.0);
      U3[IV] = KALYPSSO_NUM(1.206);
      U3[IW] = KALYPSSO_NUM(1.206);
      U3[IP] = KALYPSSO_NUM(0.029);

      U4[ID] = KALYPSSO_NUM(1.5);
      U4[IU] = KALYPSSO_NUM(0.0);
      U4[IV] = KALYPSSO_NUM(0.0);
      U4[IW] = KALYPSSO_NUM(0.0);
      U4[IP] = KALYPSSO_NUM(1.5);

      U5[ID] = KALYPSSO_NUM(0.5323);
      U5[IU] = KALYPSSO_NUM(1.206);
      U5[IV] = KALYPSSO_NUM(0.0);
      U5[IW] = KALYPSSO_NUM(0.0);
      U5[IP] = KALYPSSO_NUM(0.3);

      U6[ID] = KALYPSSO_NUM(0.138);
      U6[IU] = KALYPSSO_NUM(1.206);
      U6[IV] = KALYPSSO_NUM(1.206);
      U6[IW] = KALYPSSO_NUM(0.0);
      U6[IP] = KALYPSSO_NUM(0.029);

      U7[ID] = KALYPSSO_NUM(0.5323);
      U7[IU] = KALYPSSO_NUM(0.0);
      U7[IV] = KALYPSSO_NUM(1.206);
      U7[IW] = KALYPSSO_NUM(0.0);
      U7[IP] = KALYPSSO_NUM(0.3);
      break;

    case 3:
      // Config 4
      U0[ID] = KALYPSSO_NUM(1.1);
      U0[IU] = KALYPSSO_NUM(0.0);
      U0[IV] = KALYPSSO_NUM(0.0);
      U0[IW] = KALYPSSO_NUM(0.0);
      U0[IP] = KALYPSSO_NUM(1.1);

      U1[ID] = KALYPSSO_NUM(0.5065);
      U1[IU] = KALYPSSO_NUM(0.8939);
      U1[IV] = KALYPSSO_NUM(0.0);
      U1[IW] = KALYPSSO_NUM(0.0);
      U1[IP] = KALYPSSO_NUM(0.35);

      U2[ID] = KALYPSSO_NUM(1.1);
      U2[IU] = KALYPSSO_NUM(0.8939);
      U2[IV] = KALYPSSO_NUM(0.8939);
      U2[IW] = KALYPSSO_NUM(0.0);
      U2[IP] = KALYPSSO_NUM(1.1);

      U3[ID] = KALYPSSO_NUM(0.5065);
      U3[IU] = KALYPSSO_NUM(0.0);
      U3[IV] = KALYPSSO_NUM(0.8939);
      U3[IW] = KALYPSSO_NUM(0.0);
      U3[IP] = KALYPSSO_NUM(0.35);

      U4[ID] = KALYPSSO_NUM(0.5065);
      U4[IU] = KALYPSSO_NUM(0.0);
      U4[IV] = KALYPSSO_NUM(0.0);
      U4[IW] = KALYPSSO_NUM(0.8939);
      U4[IP] = KALYPSSO_NUM(0.35);

      U5[ID] = KALYPSSO_NUM(1.1);
      U5[IU] = KALYPSSO_NUM(0.8939);
      U5[IV] = KALYPSSO_NUM(0.0);
      U5[IW] = KALYPSSO_NUM(0.8939);
      U5[IP] = KALYPSSO_NUM(1.1);

      U6[ID] = KALYPSSO_NUM(1.1);
      U6[IU] = KALYPSSO_NUM(0.8939);
      U6[IV] = KALYPSSO_NUM(0.8939);
      U6[IW] = KALYPSSO_NUM(0.8939);
      U6[IP] = KALYPSSO_NUM(1.1);

      U7[ID] = KALYPSSO_NUM(0.5065);
      U7[IU] = KALYPSSO_NUM(0.0);
      U7[IV] = KALYPSSO_NUM(0.8939);
      U7[IW] = KALYPSSO_NUM(0.8939);
      U7[IP] = KALYPSSO_NUM(0.35);
      break;

    default:
      // Config 4
      U0[ID] = KALYPSSO_NUM(1.1);
      U0[IU] = KALYPSSO_NUM(0.0);
      U0[IV] = KALYPSSO_NUM(0.0);
      U0[IW] = KALYPSSO_NUM(0.0);
      U0[IP] = KALYPSSO_NUM(1.1);

      U1[ID] = KALYPSSO_NUM(0.5065);
      U1[IU] = KALYPSSO_NUM(0.8939);
      U1[IV] = KALYPSSO_NUM(0.0);
      U1[IW] = KALYPSSO_NUM(0.0);
      U1[IP] = KALYPSSO_NUM(0.35);

      U2[ID] = KALYPSSO_NUM(1.1);
      U2[IU] = KALYPSSO_NUM(0.8939);
      U2[IV] = KALYPSSO_NUM(0.8939);
      U2[IW] = KALYPSSO_NUM(0.0);
      U2[IP] = KALYPSSO_NUM(1.1);

      U3[ID] = KALYPSSO_NUM(0.5065);
      U3[IU] = KALYPSSO_NUM(0.0);
      U3[IV] = KALYPSSO_NUM(0.8939);
      U3[IW] = KALYPSSO_NUM(0.0);
      U3[IP] = KALYPSSO_NUM(0.35);

      U4[ID] = KALYPSSO_NUM(0.5065);
      U4[IU] = KALYPSSO_NUM(0.0);
      U4[IV] = KALYPSSO_NUM(0.0);
      U4[IW] = KALYPSSO_NUM(0.8939);
      U4[IP] = KALYPSSO_NUM(0.35);

      U5[ID] = KALYPSSO_NUM(1.1);
      U5[IU] = KALYPSSO_NUM(0.8939);
      U5[IV] = KALYPSSO_NUM(0.0);
      U5[IW] = KALYPSSO_NUM(0.8939);
      U5[IP] = KALYPSSO_NUM(1.1);

      U6[ID] = KALYPSSO_NUM(1.1);
      U6[IU] = KALYPSSO_NUM(0.8939);
      U6[IV] = KALYPSSO_NUM(0.8939);
      U6[IW] = KALYPSSO_NUM(0.8939);
      U6[IP] = KALYPSSO_NUM(1.1);

      U7[ID] = KALYPSSO_NUM(0.5065);
      U7[IU] = KALYPSSO_NUM(0.0);
      U7[IV] = KALYPSSO_NUM(0.8939);
      U7[IW] = KALYPSSO_NUM(0.8939);
      U7[IP] = KALYPSSO_NUM(0.35);
  }

  return Us;

} // getRiemannConfig3d

} // namespace kalypsso
