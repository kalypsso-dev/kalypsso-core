// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MHD.h
 */
#ifndef KALYPSSO_CORE_MODELS_MHD_H
#define KALYPSSO_CORE_MODELS_MHD_H

#include <kalypsso/core/FieldMap.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/models/MHDSettings.h>

#include <cstdint>
#include <string>
#include <unordered_map>

namespace kalypsso
{

namespace core
{

namespace models
{

// =============================================================
// =============================================================
class MHD
{

public:
  using Id_t = int32_t;

  //! MHD field ids
  enum VarId : Id_t
  {
    ID = 0,          /*!< ID Density field index */
    IP = 1,          /*!< IP Pressure/Energy field index */
    IE = 1,          /*!< IE Energy/Pressure field index */
    IU = 2,          /*!< X velocity / momentum index */
    IV = 3,          /*!< Y velocity / momentum index */
    IW = 4,          /*!< Z velocity / momentum index */
    IA = 5,          /*!< X magnetic field index */
    IB = 6,          /*!< Y magnetic field index */
    IC = 7,          /*!< Z magnetic field index */
    IBX = 5,         /*!< X magnetic field index */
    IBY = 6,         /*!< Y magnetic field index */
    IBZ = 7,         /*!< Z magnetic field index */
    IAL = 5,         /*!< X magnetic field index */
    IBL = 6,         /*!< Y magnetic field index */
    ICL = 7,         /*!< Z magnetic field index */
    IAR = 8,         /*!< X magnetic field index */
    IBR = 9,         /*!< Y magnetic field index */
    ICR = 10,        /*!< Z magnetic field index */
    IGX = 11,        /*!< X gravitational field index */
    IGY = 12,        /*!< Y gravitational field index */
    IGZ = 13,        /*!< Z gravitational field index */
    IA0 = 14,        /*!< X zero order magnetic field index */
    IB0 = 15,        /*!< Y zero order magnetic field index */
    IC0 = 16,        /*!< Z zero order magnetic field index */
    VARID_COUNT = 17 /*!< invalid index, just counting number of fields */
  };

  enum GradTensorId : Id_t
  {
    IUX = 0, /*!< du/dx */
    IUY = 1, /*!< du/dy */
    IUZ = 2, /*!< du/dz */
    IVX = 3, /*!< dv/dx */
    IVY = 4, /*!< dv/dy */
    IVZ = 5, /*!< dv/dz */
    IWX = 6, /*!< dw/dx */
    IWY = 7, /*!< dw/dy */
    IWZ = 8, /*!< dw/dz */
  };

  enum DivergenceId : Id_t
  {
    IXA = 0, /*!dA/dx*/
    IYB = 1, /*!dB/dy*/
    IZC = 2, /*!dC/dz*/
  };

  //! Face-centered magnetic field component enumeration.
  //! To be used with FaceMagState.
  enum FaceMagneticFieldId : Id_t
  {
    AL = IAL - IAL, /*!< left face along  X magnetic field index */
    BL = IBL - IAL, /*!< left face along  Y magnetic field index */
    CL = ICL - IAL, /*!< left face along  Z magnetic field index */
    AR = IAR - IAL, /*!< right face along X magnetic field index */
    BR = IBR - IAL, /*!< right face along Y magnetic field index */
    CR = ICR - IAL, /*!< right face along Z magnetic field index */
  };

  //! a dictionary of all variables names and corresponding id (enum)
  //! this map is initialized in MHD.cpp
  static const id2names_t m_id2names_all;

public:
  //! constructor
  MHD(ConfigMap const & config_map) { setup(config_map); }

  //! initialize model enabled variable maps
  void
  setup(ConfigMap const & config_map)
  {

    m_fieldmap.enable(ID);
    m_fieldmap.enable(IE);
    m_fieldmap.enable(IU);
    m_fieldmap.enable(IV);
    m_fieldmap.enable(IW);
    m_fieldmap.enable(IA);
    m_fieldmap.enable(IB);
    m_fieldmap.enable(IC);

    const auto mhd_settings = MHDSettings(config_map);

    if (mhd_settings.mag_field_split_enabled)
    {
      m_fieldmap.enable(IA0);
      m_fieldmap.enable(IB0);
      m_fieldmap.enable(IC0);
    }

    // insert some fields in names2id and id2names maps
    for (int32_t varIdInt = 0; varIdInt != VARID_COUNT; ++varIdInt)
    {
      const VarId varId = static_cast<VarId>(varIdInt);
      if (m_fieldmap.enabled(varId))
      {
        const auto iter = m_id2names_all.find(varId);

        if (iter != m_id2names_all.end())
        {
          m_names2id[iter->second] = varId;
          m_id2names[varId] = iter->second;
        }
      }
    } // for varIdInt

  } // setup

  FieldMap<MHD>
  get_fieldmap()
  {
    return m_fieldmap;
  }

  const FieldMap<MHD> &
  get_fieldmap() const
  {
    return m_fieldmap;
  }

  //! get id to names map for enabled variables
  const names2id_t &
  get_names2id_map() const
  {
    return m_names2id;
  }

  //! get names to id map for enabled variables
  const id2names_t &
  get_id2names_map() const
  {
    return m_id2names;
  }

private:
  //! Kokkos::Array mapping enums to index of active variables
  FieldMap<MHD> m_fieldmap;

  //! map of ids to names
  names2id_t m_names2id;

  //! map of ids to names
  id2names_t m_id2names;

}; // class MHD

} // namespace models

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_MODELS_MHD_H
