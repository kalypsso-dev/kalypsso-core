// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file Hydro.h
 */
#ifndef KALYPSSO_CORE_MODELS_HYDRO_H
#define KALYPSSO_CORE_MODELS_HYDRO_H

#include <kalypsso/core/FieldMap.h>

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
class Hydro
{

public:
  struct Settings
  {
    size_t dim = 2;

    Settings() = default;

    Settings(size_t dim_)
      : dim(dim_)
    {}

  }; // struct Settings

  using Id_t = int32_t;

  //! hydro field ids
  enum VarId : Id_t
  {
    ID = 0,         /*!< ID Density field index */
    IP = 1,         /*!< IP Pressure/Energy field index */
    IE = 1,         /*!< IE Energy/Pressure field index */
    IU = 2,         /*!< X velocity / momentum index */
    IV = 3,         /*!< Y velocity / momentum index */
    IW = 4,         /*!< Z velocity / momentum index */
    IGX = 5,        /*!< X gravitational field index */
    IGY = 6,        /*!< Y gravitational field index */
    IGZ = 7,        /*!< Z gravitational field index */
    VARID_COUNT = 8 /*!< invalid index, just counting number of fields */
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

  //! a dictionary of all variables names and corresponding id (enum)
  //! this map is initialized in Hydro.cpp
  static const id2names_t m_id2names_all;

public:
  Hydro()
    : m_settings()
  {
    setup();
  }

  //! constructor
  Hydro(Settings settings)
    : m_settings(settings)
  {
    setup();
  }

  //! constructor
  Hydro(size_t dim)
    : m_settings(dim)
  {
    setup();
  }

  //! initialize model enabled variable maps
  void
  setup()
  {

    m_fieldmap.enable(ID);
    m_fieldmap.enable(IE);
    m_fieldmap.enable(IU);
    m_fieldmap.enable(IV);
    if (m_settings.dim == 3)
    {
      m_fieldmap.enable(IW);
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

  FieldMap<Hydro>
  get_fieldmap()
  {
    return m_fieldmap;
  }

  const FieldMap<Hydro> &
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
  //! model settings
  Settings m_settings;

  //! Kokkos::Array mapping enums to index of active variables
  FieldMap<Hydro> m_fieldmap;

  //! map of ids to names
  names2id_t m_names2id;

  //! map of ids to names
  id2names_t m_id2names;

}; // class Hydro

} // namespace models

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_MODELS_HYDRO_H
