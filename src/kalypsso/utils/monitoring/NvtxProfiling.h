#ifndef KALYPSSO_UTILS_MONITORING_NVTXPROFILING_H_
#define KALYPSSO_UTILS_MONITORING_NVTXPROFILING_H_

#include <cstdint>

namespace kalypsso
{

// Copyright (c) 2018, Eyal Rozenberg
// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
struct Color_t
{
  using underlying_type = ::std::uint32_t;
  using channel_value = ::std::uint8_t;
  channel_value alpha, red, green, blue;

  static constexpr Color_t from_hex(underlying_type raw_argb) noexcept {
    return {
      static_cast<channel_value> ((raw_argb >> 24) & 0xFF),
      static_cast<channel_value> ((raw_argb >> 16) & 0xFF),
      static_cast<channel_value> ((raw_argb >>  8) & 0xFF),
      static_cast<channel_value> ((raw_argb >>  0) & 0xFF),
    };
  }

  operator underlying_type() const noexcept { return as_hex(); }
  underlying_type as_hex() const noexcept
  {
    return
      static_cast<underlying_type>(alpha)  << 24 |
      static_cast<underlying_type>(red)    << 16 |
      static_cast<underlying_type>(green)  <<  8 |
      static_cast<underlying_type>(blue)   <<  0;
  }

  // Use color definition: Google Material Design colors
  // See https://material.google.com/style/color.html
  // see also https://github.com/yse/easy_profiler

  static constexpr Color_t Red50()   noexcept { return from_hex(0xffffebee); }
  static constexpr Color_t Red100()  noexcept { return from_hex(0xffffcdd2); }
  static constexpr Color_t Red200()  noexcept { return from_hex(0xffef9a9a); }
  static constexpr Color_t Red300()  noexcept { return from_hex(0xffe57373); }
  static constexpr Color_t Red400()  noexcept { return from_hex(0xffef5350); }
  static constexpr Color_t Red500()  noexcept { return from_hex(0xfff44336); }
  static constexpr Color_t Red600()  noexcept { return from_hex(0xffe53935); }
  static constexpr Color_t Red700()  noexcept { return from_hex(0xffd32f2f); }
  static constexpr Color_t Red800()  noexcept { return from_hex(0xffc62828); }
  static constexpr Color_t Red900()  noexcept { return from_hex(0xffb71c1c); }
  static constexpr Color_t RedA100() noexcept { return from_hex(0xffff8a80); }
  static constexpr Color_t RedA200() noexcept { return from_hex(0xffff5252); }
  static constexpr Color_t RedA400() noexcept { return from_hex(0xffff1744); }
  static constexpr Color_t RedA700() noexcept { return from_hex(0xffd50000); }

  static constexpr Color_t Pink50()   noexcept { return from_hex(0xfffce4ec); }
  static constexpr Color_t Pink100()  noexcept { return from_hex(0xfff8bbd0); }
  static constexpr Color_t Pink200()  noexcept { return from_hex(0xfff48fb1); }
  static constexpr Color_t Pink300()  noexcept { return from_hex(0xfff06292); }
  static constexpr Color_t Pink400()  noexcept { return from_hex(0xffec407a); }
  static constexpr Color_t Pink500()  noexcept { return from_hex(0xffe91e63); }
  static constexpr Color_t Pink600()  noexcept { return from_hex(0xffd81b60); }
  static constexpr Color_t Pink700()  noexcept { return from_hex(0xffc2185b); }
  static constexpr Color_t Pink800()  noexcept { return from_hex(0xffad1457); }
  static constexpr Color_t Pink900()  noexcept { return from_hex(0xff880e4f); }
  static constexpr Color_t PinkA100() noexcept { return from_hex(0xffff80ab); }
  static constexpr Color_t PinkA200() noexcept { return from_hex(0xffff4081); }
  static constexpr Color_t PinkA400() noexcept { return from_hex(0xfff50057); }
  static constexpr Color_t PinkA700() noexcept { return from_hex(0xffc51162); }

  static constexpr Color_t Purple50 ()  noexcept { return from_hex(0xfff3e5f5); }
  static constexpr Color_t Purple100()  noexcept { return from_hex(0xffe1bee7); }
  static constexpr Color_t Purple200()  noexcept { return from_hex(0xffce93d8); }
  static constexpr Color_t Purple300()  noexcept { return from_hex(0xffba68c8); }
  static constexpr Color_t Purple400()  noexcept { return from_hex(0xffab47bc); }
  static constexpr Color_t Purple500()  noexcept { return from_hex(0xff9c27b0); }
  static constexpr Color_t Purple600()  noexcept { return from_hex(0xff8e24aa); }
  static constexpr Color_t Purple700()  noexcept { return from_hex(0xff7b1fa2); }
  static constexpr Color_t Purple800()  noexcept { return from_hex(0xff6a1b9a); }
  static constexpr Color_t Purple900()  noexcept { return from_hex(0xff4a148c); }
  static constexpr Color_t PurpleA100() noexcept { return from_hex(0xffea80fc); }
  static constexpr Color_t PurpleA200() noexcept { return from_hex(0xffe040fb); }
  static constexpr Color_t PurpleA400() noexcept { return from_hex(0xffd500f9); }
  static constexpr Color_t PurpleA700() noexcept { return from_hex(0xffaa00ff); }

  static constexpr Color_t DeepPurple50()   noexcept { return from_hex(0xffede7f6); }
  static constexpr Color_t DeepPurple100()  noexcept { return from_hex(0xffd1c4e9); }
  static constexpr Color_t DeepPurple200()  noexcept { return from_hex(0xffb39ddb); }
  static constexpr Color_t DeepPurple300()  noexcept { return from_hex(0xff9575cd); }
  static constexpr Color_t DeepPurple400()  noexcept { return from_hex(0xff7e57c2); }
  static constexpr Color_t DeepPurple500()  noexcept { return from_hex(0xff673ab7); }
  static constexpr Color_t DeepPurple600()  noexcept { return from_hex(0xff5e35b1); }
  static constexpr Color_t DeepPurple700()  noexcept { return from_hex(0xff512da8); }
  static constexpr Color_t DeepPurple800()  noexcept { return from_hex(0xff4527a0); }
  static constexpr Color_t DeepPurple900()  noexcept { return from_hex(0xff311b92); }
  static constexpr Color_t DeepPurpleA100() noexcept { return from_hex(0xffb388ff); }
  static constexpr Color_t DeepPurpleA200() noexcept { return from_hex(0xff7c4dff); }
  static constexpr Color_t DeepPurpleA400() noexcept { return from_hex(0xff651fff); }
  static constexpr Color_t DeepPurpleA700() noexcept { return from_hex(0xff6200ea); }

  static constexpr Color_t Indigo50()   noexcept { return from_hex(0xffe8eaf6); }
  static constexpr Color_t Indigo100()  noexcept { return from_hex(0xffc5cae9); }
  static constexpr Color_t Indigo200()  noexcept { return from_hex(0xff9fa8da); }
  static constexpr Color_t Indigo300()  noexcept { return from_hex(0xff7986cb); }
  static constexpr Color_t Indigo400()  noexcept { return from_hex(0xff5c6bc0); }
  static constexpr Color_t Indigo500()  noexcept { return from_hex(0xff3f51b5); }
  static constexpr Color_t Indigo600()  noexcept { return from_hex(0xff3949ab); }
  static constexpr Color_t Indigo700()  noexcept { return from_hex(0xff303f9f); }
  static constexpr Color_t Indigo800()  noexcept { return from_hex(0xff283593); }
  static constexpr Color_t Indigo900()  noexcept { return from_hex(0xff1a237e); }
  static constexpr Color_t IndigoA100() noexcept { return from_hex(0xff8c9eff); }
  static constexpr Color_t IndigoA200() noexcept { return from_hex(0xff536dfe); }
  static constexpr Color_t IndigoA400() noexcept { return from_hex(0xff3d5afe); }
  static constexpr Color_t IndigoA700() noexcept { return from_hex(0xff304ffe); }

  static constexpr Color_t Blue50()   noexcept { return from_hex(0xffe3f2fd); }
  static constexpr Color_t Blue100()  noexcept { return from_hex(0xffbbdefb); }
  static constexpr Color_t Blue200()  noexcept { return from_hex(0xff90caf9); }
  static constexpr Color_t Blue300()  noexcept { return from_hex(0xff64b5f6); }
  static constexpr Color_t Blue400()  noexcept { return from_hex(0xff42a5f5); }
  static constexpr Color_t Blue500()  noexcept { return from_hex(0xff2196f3); }
  static constexpr Color_t Blue600()  noexcept { return from_hex(0xff1e88e5); }
  static constexpr Color_t Blue700()  noexcept { return from_hex(0xff1976d2); }
  static constexpr Color_t Blue800()  noexcept { return from_hex(0xff1565c0); }
  static constexpr Color_t Blue900()  noexcept { return from_hex(0xff0d47a1); }
  static constexpr Color_t BlueA100() noexcept { return from_hex(0xff82b1ff); }
  static constexpr Color_t BlueA200() noexcept { return from_hex(0xff448aff); }
  static constexpr Color_t BlueA400() noexcept { return from_hex(0xff2979ff); }
  static constexpr Color_t BlueA700() noexcept { return from_hex(0xff2962ff); }

  static constexpr Color_t LightBlue50()   noexcept { return from_hex(0xffe1f5fe); }
  static constexpr Color_t LightBlue100()  noexcept { return from_hex(0xffb3e5fc); }
  static constexpr Color_t LightBlue200()  noexcept { return from_hex(0xff81d4fa); }
  static constexpr Color_t LightBlue300()  noexcept { return from_hex(0xff4fc3f7); }
  static constexpr Color_t LightBlue400()  noexcept { return from_hex(0xff29b6f6); }
  static constexpr Color_t LightBlue500()  noexcept { return from_hex(0xff03a9f4); }
  static constexpr Color_t LightBlue600()  noexcept { return from_hex(0xff039be5); }
  static constexpr Color_t LightBlue700()  noexcept { return from_hex(0xff0288d1); }
  static constexpr Color_t LightBlue800()  noexcept { return from_hex(0xff0277bd); }
  static constexpr Color_t LightBlue900()  noexcept { return from_hex(0xff01579b); }
  static constexpr Color_t LightBlueA100() noexcept { return from_hex(0xff80d8ff); }
  static constexpr Color_t LightBlueA200() noexcept { return from_hex(0xff40c4ff); }
  static constexpr Color_t LightBlueA400() noexcept { return from_hex(0xff00b0ff); }
  static constexpr Color_t LightBlueA700() noexcept { return from_hex(0xff0091ea); }

  static constexpr Color_t Cyan50()   noexcept { return from_hex(0xffe0f7fa); }
  static constexpr Color_t Cyan100()  noexcept { return from_hex(0xffb2ebf2); }
  static constexpr Color_t Cyan200()  noexcept { return from_hex(0xff80deea); }
  static constexpr Color_t Cyan300()  noexcept { return from_hex(0xff4dd0e1); }
  static constexpr Color_t Cyan400()  noexcept { return from_hex(0xff26c6da); }
  static constexpr Color_t Cyan500()  noexcept { return from_hex(0xff00bcd4); }
  static constexpr Color_t Cyan600()  noexcept { return from_hex(0xff00acc1); }
  static constexpr Color_t Cyan700()  noexcept { return from_hex(0xff0097a7); }
  static constexpr Color_t Cyan800()  noexcept { return from_hex(0xff00838f); }
  static constexpr Color_t Cyan900()  noexcept { return from_hex(0xff006064); }
  static constexpr Color_t CyanA100() noexcept { return from_hex(0xff84ffff); }
  static constexpr Color_t CyanA200() noexcept { return from_hex(0xff18ffff); }
  static constexpr Color_t CyanA400() noexcept { return from_hex(0xff00e5ff); }
  static constexpr Color_t CyanA700() noexcept { return from_hex(0xff00b8d4); }

  static constexpr Color_t Teal50()   noexcept { return from_hex(0xffe0f2f1); }
  static constexpr Color_t Teal100()  noexcept { return from_hex(0xffb2dfdb); }
  static constexpr Color_t Teal200()  noexcept { return from_hex(0xff80cbc4); }
  static constexpr Color_t Teal300()  noexcept { return from_hex(0xff4db6ac); }
  static constexpr Color_t Teal400()  noexcept { return from_hex(0xff26a69a); }
  static constexpr Color_t Teal500()  noexcept { return from_hex(0xff009688); }
  static constexpr Color_t Teal600()  noexcept { return from_hex(0xff00897b); }
  static constexpr Color_t Teal700()  noexcept { return from_hex(0xff00796b); }
  static constexpr Color_t Teal800()  noexcept { return from_hex(0xff00695c); }
  static constexpr Color_t Teal900()  noexcept { return from_hex(0xff004d40); }
  static constexpr Color_t TealA100() noexcept { return from_hex(0xffa7ffeb); }
  static constexpr Color_t TealA200() noexcept { return from_hex(0xff64ffda); }
  static constexpr Color_t TealA400() noexcept { return from_hex(0xff1de9b6); }
  static constexpr Color_t TealA700() noexcept { return from_hex(0xff00bfa5); }

  static constexpr Color_t Green50()   noexcept { return from_hex(0xffe8f5e9); }
  static constexpr Color_t Green100()  noexcept { return from_hex(0xffc8e6c9); }
  static constexpr Color_t Green200()  noexcept { return from_hex(0xffa5d6a7); }
  static constexpr Color_t Green300()  noexcept { return from_hex(0xff81c784); }
  static constexpr Color_t Green400()  noexcept { return from_hex(0xff66bb6a); }
  static constexpr Color_t Green500()  noexcept { return from_hex(0xff4caf50); }
  static constexpr Color_t Green600()  noexcept { return from_hex(0xff43a047); }
  static constexpr Color_t Green700()  noexcept { return from_hex(0xff388e3c); }
  static constexpr Color_t Green800()  noexcept { return from_hex(0xff2e7d32); }
  static constexpr Color_t Green900()  noexcept { return from_hex(0xff1b5e20); }
  static constexpr Color_t GreenA100() noexcept { return from_hex(0xffb9f6ca); }
  static constexpr Color_t GreenA200() noexcept { return from_hex(0xff69f0ae); }
  static constexpr Color_t GreenA400() noexcept { return from_hex(0xff00e676); }
  static constexpr Color_t GreenA700() noexcept { return from_hex(0xff00c853); }

  static constexpr Color_t LightGreen50()   noexcept { return from_hex(0xfff1f8e9); }
  static constexpr Color_t LightGreen100()  noexcept { return from_hex(0xffdcedc8); }
  static constexpr Color_t LightGreen200()  noexcept { return from_hex(0xffc5e1a5); }
  static constexpr Color_t LightGreen300()  noexcept { return from_hex(0xffaed581); }
  static constexpr Color_t LightGreen400()  noexcept { return from_hex(0xff9ccc65); }
  static constexpr Color_t LightGreen500()  noexcept { return from_hex(0xff8bc34a); }
  static constexpr Color_t LightGreen600()  noexcept { return from_hex(0xff7cb342); }
  static constexpr Color_t LightGreen700()  noexcept { return from_hex(0xff689f38); }
  static constexpr Color_t LightGreen800()  noexcept { return from_hex(0xff558b2f); }
  static constexpr Color_t LightGreen900()  noexcept { return from_hex(0xff33691e); }
  static constexpr Color_t LightGreenA100() noexcept { return from_hex(0xffccff90); }
  static constexpr Color_t LightGreenA200() noexcept { return from_hex(0xffb2ff59); }
  static constexpr Color_t LightGreenA400() noexcept { return from_hex(0xff76ff03); }
  static constexpr Color_t LightGreenA700() noexcept { return from_hex(0xff64dd17); }

  static constexpr Color_t Lime50()   noexcept { return from_hex(0xfff9ebe7); }
  static constexpr Color_t Lime100()  noexcept { return from_hex(0xfff0f4c3); }
  static constexpr Color_t Lime200()  noexcept { return from_hex(0xffe6ee9c); }
  static constexpr Color_t Lime300()  noexcept { return from_hex(0xffdce775); }
  static constexpr Color_t Lime400()  noexcept { return from_hex(0xffd4e157); }
  static constexpr Color_t Lime500()  noexcept { return from_hex(0xffcddc39); }
  static constexpr Color_t Lime600()  noexcept { return from_hex(0xffc0ca33); }
  static constexpr Color_t Lime700()  noexcept { return from_hex(0xffafb42b); }
  static constexpr Color_t Lime800()  noexcept { return from_hex(0xff9e9d24); }
  static constexpr Color_t Lime900()  noexcept { return from_hex(0xff827717); }
  static constexpr Color_t LimeA100() noexcept { return from_hex(0xfff4ff81); }
  static constexpr Color_t LimeA200() noexcept { return from_hex(0xffeeff41); }
  static constexpr Color_t LimeA400() noexcept { return from_hex(0xffc6ff00); }
  static constexpr Color_t LimeA700() noexcept { return from_hex(0xffaeea00); }

  static constexpr Color_t Yellow50()   noexcept { return from_hex(0xfffffde7); }
  static constexpr Color_t Yellow100()  noexcept { return from_hex(0xfffff9c4); }
  static constexpr Color_t Yellow200()  noexcept { return from_hex(0xfffff59d); }
  static constexpr Color_t Yellow300()  noexcept { return from_hex(0xfffff176); }
  static constexpr Color_t Yellow400()  noexcept { return from_hex(0xffffee58); }
  static constexpr Color_t Yellow500()  noexcept { return from_hex(0xffffeb3b); }
  static constexpr Color_t Yellow600()  noexcept { return from_hex(0xfffdd835); }
  static constexpr Color_t Yellow700()  noexcept { return from_hex(0xfffbc02d); }
  static constexpr Color_t Yellow800()  noexcept { return from_hex(0xfff9a825); }
  static constexpr Color_t Yellow900()  noexcept { return from_hex(0xfff57f17); }
  static constexpr Color_t YellowA100() noexcept { return from_hex(0xffffff8d); }
  static constexpr Color_t YellowA200() noexcept { return from_hex(0xffffff00); }
  static constexpr Color_t YellowA400() noexcept { return from_hex(0xffffea00); }
  static constexpr Color_t YellowA700() noexcept { return from_hex(0xffffd600); }

  static constexpr Color_t Amber50()   noexcept { return from_hex(0xfffff8e1); }
  static constexpr Color_t Amber100()  noexcept { return from_hex(0xffffecb3); }
  static constexpr Color_t Amber200()  noexcept { return from_hex(0xffffe082); }
  static constexpr Color_t Amber300()  noexcept { return from_hex(0xffffd54f); }
  static constexpr Color_t Amber400()  noexcept { return from_hex(0xffffca28); }
  static constexpr Color_t Amber500()  noexcept { return from_hex(0xffffc107); }
  static constexpr Color_t Amber600()  noexcept { return from_hex(0xffffb300); }
  static constexpr Color_t Amber700()  noexcept { return from_hex(0xffffa000); }
  static constexpr Color_t Amber800()  noexcept { return from_hex(0xffff8f00); }
  static constexpr Color_t Amber900()  noexcept { return from_hex(0xffff6f00); }
  static constexpr Color_t AmberA100() noexcept { return from_hex(0xffffe57f); }
  static constexpr Color_t AmberA200() noexcept { return from_hex(0xffffd740); }
  static constexpr Color_t AmberA400() noexcept { return from_hex(0xffffc400); }
  static constexpr Color_t AmberA700() noexcept { return from_hex(0xffffab00); }

  static constexpr Color_t Orange50()   noexcept { return from_hex(0xfffff3e0); }
  static constexpr Color_t Orange100()  noexcept { return from_hex(0xffffe0b2); }
  static constexpr Color_t Orange200()  noexcept { return from_hex(0xffffcc80); }
  static constexpr Color_t Orange300()  noexcept { return from_hex(0xffffb74d); }
  static constexpr Color_t Orange400()  noexcept { return from_hex(0xffffa726); }
  static constexpr Color_t Orange500()  noexcept { return from_hex(0xffff9800); }
  static constexpr Color_t Orange600()  noexcept { return from_hex(0xfffb8c00); }
  static constexpr Color_t Orange700()  noexcept { return from_hex(0xfff57c00); }
  static constexpr Color_t Orange800()  noexcept { return from_hex(0xffef6c00); }
  static constexpr Color_t Orange900()  noexcept { return from_hex(0xffe65100); }
  static constexpr Color_t OrangeA100() noexcept { return from_hex(0xffffd180); }
  static constexpr Color_t OrangeA200() noexcept { return from_hex(0xffffab40); }
  static constexpr Color_t OrangeA400() noexcept { return from_hex(0xffff9100); }
  static constexpr Color_t OrangeA700() noexcept { return from_hex(0xffff6d00); }

  static constexpr Color_t DeepOrange50()   noexcept { return from_hex(0xfffbe9e7); }
  static constexpr Color_t DeepOrange100()  noexcept { return from_hex(0xffffccbc); }
  static constexpr Color_t DeepOrange200()  noexcept { return from_hex(0xffffab91); }
  static constexpr Color_t DeepOrange300()  noexcept { return from_hex(0xffff8a65); }
  static constexpr Color_t DeepOrange400()  noexcept { return from_hex(0xffff7043); }
  static constexpr Color_t DeepOrange500()  noexcept { return from_hex(0xffff5722); }
  static constexpr Color_t DeepOrange600()  noexcept { return from_hex(0xfff4511e); }
  static constexpr Color_t DeepOrange700()  noexcept { return from_hex(0xffe64a19); }
  static constexpr Color_t DeepOrange800()  noexcept { return from_hex(0xffd84315); }
  static constexpr Color_t DeepOrange900()  noexcept { return from_hex(0xffbf360c); }
  static constexpr Color_t DeepOrangeA100() noexcept { return from_hex(0xffff9e80); }
  static constexpr Color_t DeepOrangeA200() noexcept { return from_hex(0xffff6e40); }
  static constexpr Color_t DeepOrangeA400() noexcept { return from_hex(0xffff3d00); }
  static constexpr Color_t DeepOrangeA700() noexcept { return from_hex(0xffdd2c00); }

  static constexpr Color_t Brown50()  noexcept { return from_hex(0xffefebe9); }
  static constexpr Color_t Brown100() noexcept { return from_hex(0xffd7ccc8); }
  static constexpr Color_t Brown200() noexcept { return from_hex(0xffbcaaa4); }
  static constexpr Color_t Brown300() noexcept { return from_hex(0xffa1887f); }
  static constexpr Color_t Brown400() noexcept { return from_hex(0xff8d6e63); }
  static constexpr Color_t Brown500() noexcept { return from_hex(0xff795548); }
  static constexpr Color_t Brown600() noexcept { return from_hex(0xff6d4c41); }
  static constexpr Color_t Brown700() noexcept { return from_hex(0xff5d4037); }
  static constexpr Color_t Brown800() noexcept { return from_hex(0xff4e342e); }
  static constexpr Color_t Brown900() noexcept { return from_hex(0xff3e2723); }

  static constexpr Color_t Grey50()  noexcept { return from_hex(0xfffafafa); }
  static constexpr Color_t Grey100() noexcept { return from_hex(0xfff5f5f5); }
  static constexpr Color_t Grey200() noexcept { return from_hex(0xffeeeeee); }
  static constexpr Color_t Grey300() noexcept { return from_hex(0xffe0e0e0); }
  static constexpr Color_t Grey400() noexcept { return from_hex(0xffbdbdbd); }
  static constexpr Color_t Grey500() noexcept { return from_hex(0xff9e9e9e); }
  static constexpr Color_t Grey600() noexcept { return from_hex(0xff757575); }
  static constexpr Color_t Grey700() noexcept { return from_hex(0xff616161); }
  static constexpr Color_t Grey800() noexcept { return from_hex(0xff424242); }
  static constexpr Color_t Grey900() noexcept { return from_hex(0xff212121); }

  static constexpr Color_t BlueGrey50()  noexcept { return from_hex(0xffeceff1); }
  static constexpr Color_t BlueGrey100() noexcept { return from_hex(0xffcfd8dc); }
  static constexpr Color_t BlueGrey200() noexcept { return from_hex(0xffb0bec5); }
  static constexpr Color_t BlueGrey300() noexcept { return from_hex(0xff90a4ae); }
  static constexpr Color_t BlueGrey400() noexcept { return from_hex(0xff78909c); }
  static constexpr Color_t BlueGrey500() noexcept { return from_hex(0xff607d8b); }
  static constexpr Color_t BlueGrey600() noexcept { return from_hex(0xff546e7a); }
  static constexpr Color_t BlueGrey700() noexcept { return from_hex(0xff455a64); }
  static constexpr Color_t BlueGrey800() noexcept { return from_hex(0xff37474f); }
  static constexpr Color_t BlueGrey900() noexcept { return from_hex(0xff263238); }

  static constexpr Color_t Black()  noexcept { return from_hex(0xff000000); }
  static constexpr Color_t White()  noexcept { return from_hex(0xffffffff); }
  static constexpr Color_t Null()   noexcept { return from_hex(0x00000000); }

  static constexpr Color_t Red()         noexcept { return Red500(); }
  static constexpr Color_t DarkRed()     noexcept { return Red900(); }
  static constexpr Color_t Coral()       noexcept { return Red200(); }
  static constexpr Color_t RichRed()     noexcept { return from_hex(0xffff0000); }
  static constexpr Color_t Pink()        noexcept { return Pink500(); }
  static constexpr Color_t Rose()        noexcept { return PinkA100(); }
  static constexpr Color_t Purple()      noexcept { return Purple500(); }
  static constexpr Color_t Magenta()     noexcept { return PurpleA200(); }
  static constexpr Color_t DarkMagenta() noexcept { return PurpleA700(); }
  static constexpr Color_t DeepPurple()  noexcept { return DeepPurple500(); }
  static constexpr Color_t Indigo()      noexcept { return Indigo500(); }
  static constexpr Color_t Blue()        noexcept { return Blue500(); }
  static constexpr Color_t DarkBlue()    noexcept { return Blue900(); }
  static constexpr Color_t RichBlue()    noexcept { return from_hex(0xff0000ff); }
  static constexpr Color_t LightBlue()   noexcept { return LightBlue500(); }
  static constexpr Color_t SkyBlue()     noexcept { return LightBlueA100(); }
  static constexpr Color_t Navy()        noexcept { return LightBlue800(); }
  static constexpr Color_t Cyan()        noexcept { return Cyan500(); }
  static constexpr Color_t DarkCyan()    noexcept { return Cyan900(); }
  static constexpr Color_t Teal()        noexcept { return Teal500(); }
  static constexpr Color_t DarkTeal()    noexcept { return Teal900(); }
  static constexpr Color_t Green()       noexcept { return Green500(); }
  static constexpr Color_t DarkGreen()   noexcept { return Green900(); }
  static constexpr Color_t RichGreen()   noexcept { return from_hex(0xff00ff00); }
  static constexpr Color_t LightGreen()  noexcept { return LightGreen500(); }
  static constexpr Color_t Mint()        noexcept { return LightGreen900(); }
  static constexpr Color_t Lime()        noexcept { return Lime500(); }
  static constexpr Color_t Olive()       noexcept { return Lime900(); }
  static constexpr Color_t Yellow()      noexcept { return Yellow500(); }
  static constexpr Color_t RichYellow()  noexcept { return YellowA200(); }
  static constexpr Color_t Amber()       noexcept { return Amber500(); }
  static constexpr Color_t Gold()        noexcept { return Amber300(); }
  static constexpr Color_t PaleGold()    noexcept { return AmberA100(); }
  static constexpr Color_t Orange()      noexcept { return Orange500(); }
  static constexpr Color_t Skin()        noexcept { return Orange100(); }
  static constexpr Color_t DeepOrange()  noexcept { return DeepOrange500(); }
  static constexpr Color_t Brick()       noexcept { return DeepOrange900(); }
  static constexpr Color_t Brown()       noexcept { return Brown500(); }
  static constexpr Color_t DarkBrown()   noexcept { return Brown900(); }
  static constexpr Color_t CreamWhite()  noexcept { return Orange50(); }
  static constexpr Color_t Wheat()       noexcept { return Amber100(); }
  static constexpr Color_t Grey()        noexcept { return Grey500(); }
  static constexpr Color_t Dark()        noexcept { return Grey900(); }
  static constexpr Color_t Silver()      noexcept { return Grey300(); }
  static constexpr Color_t BlueGrey()    noexcept { return BlueGrey500(); }

  // old colors
  static constexpr Color_t FullRed()     noexcept { return from_hex(0x00FF0000); }
  static constexpr Color_t FullGreen()   noexcept { return from_hex(0x0000FF00); }
  static constexpr Color_t FullBlue()    noexcept { return from_hex(0x000000FF); }
  static constexpr Color_t FullYellow()  noexcept { return from_hex(0x00FFFF00); }
  static constexpr Color_t FullOrange()  noexcept { return from_hex(0x00FF8000); }
  static constexpr Color_t LightRed()    noexcept { return from_hex(0x00FFDDDD); }
  static constexpr Color_t LightYellow() noexcept { return from_hex(0x00FFFFDD); }
  static constexpr Color_t LightOrange() noexcept { return from_hex(0x00FFAA00); }
  static constexpr Color_t DarkYellow()  noexcept { return from_hex(0x00888800); }
  static constexpr Color_t DarkOrange()  noexcept { return from_hex(0x00FF8C00); }
  static constexpr Color_t LightPurple() noexcept { return from_hex(0x009370DB); }
  static constexpr Color_t Lavender()    noexcept { return from_hex(0x00E6E6FA); }
  static constexpr Color_t Violet()      noexcept { return from_hex(0x00EE82EE); }
  static constexpr Color_t Salmon()      noexcept { return from_hex(0x00FA8072); }
  static constexpr Color_t HotPink()     noexcept { return from_hex(0x00FF69B4); }
  static constexpr Color_t DeepPink()    noexcept { return from_hex(0x00FF1493); }
  static constexpr Color_t NavajoWhite() noexcept { return from_hex(0x00FFDEAD); }
  static constexpr Color_t Turquoise()   noexcept { return from_hex(0x0040E0D0); }
  static constexpr Color_t Khaki()       noexcept { return from_hex(0x00F0E68C); }
  static constexpr Color_t DarkKhaki()   noexcept { return from_hex(0x00BDB76B); }
  static constexpr Color_t DeepSkyBlue() noexcept { return from_hex(0x0000BFFF); }
  static constexpr Color_t DodgerBlue()  noexcept { return from_hex(0x001E90FF); }
  static constexpr Color_t RoyalBlue()   noexcept { return from_hex(0x004169E1); }
  static constexpr Color_t SteelBlue()   noexcept { return from_hex(0x00B0C4DE); }
  static constexpr Color_t BdxRed()      noexcept { return from_hex(0x00800000); }
  static constexpr Color_t Chocolate()   noexcept { return from_hex(0x00D2691E); }
  static constexpr Color_t SaddleBrown() noexcept { return from_hex(0x008B4513); }
  static constexpr Color_t SandyBrown()  noexcept { return from_hex(0x00F4A460); }
  static constexpr Color_t RosyBrown()   noexcept { return from_hex(0x00BC8F8F); }
  static constexpr Color_t Burlywood()   noexcept { return from_hex(0x00DEB887); }

  static constexpr Color_t DarkGray()    noexcept { return from_hex(0x00A9A9A9); }
  static constexpr Color_t Gray()        noexcept { return from_hex(0x00808080); }
  static constexpr Color_t SlateGray()   noexcept { return from_hex(0x00708090); }

  static constexpr Color_t Default() noexcept { return Wheat(); }

}; // class Color_t
// clang-format on

} // namespace kalypsso

#endif // KALYPSSO_UTILS_MONITORING_NVTXPROFILING_H_
