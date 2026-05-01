// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FieldMap.h
 * \brief Define class FieldMap
 *
 * \date July, 11th 2022
 */
#ifndef KALYPSSO_CORE_FIELDMAP_H_
#define KALYPSSO_CORE_FIELDMAP_H_

#include <unordered_map>
#include <set>

#include <kalypsso/core/enums.h>
#include <kalypsso/core/HydroParams.h>

namespace kalypsso
{

//! a convenience alias to map variable names to id
using names2id_t = std::unordered_map<std::string, int32_t>;

//! same as above but mapping memory index (instead of enum)
using names2index_t = std::unordered_map<std::string, int32_t>;

//! a convenience alias to map id to variable names
using id2names_t = std::unordered_map<int32_t, std::string>;

/**
 * A simple class to map id (enum) to index for a given model.
 *
 * the actual model is responsible for activating / deactivating some scalar fields.
 * This map is used to access user data in DataArray, by "Id" instead of index.
 *
 * The main purpose of this class is to provide operator[].
 * eg:
 * Kokkos::View<real_t **> data;
 * Hydro::Settings setting(...);
 * Hydro hydro_model(settings);
 * auto fm =
 *
 * To clarify again:
 * - id is an enum value
 * - index is an integer between 0 and scalarFieldNb-1,
 *         used to access a DataArray
 *
 * \tparam Model must define an enum called VarId
 *
 * This class can be used inside a Kokkos parallel region.
 */
template <class Model>
class FieldMap
{

public:
  using VarId = typename Model::VarId;
  using Id_t = typename Model::Id_t;
  static constexpr Id_t VARCOUNT = VarId::VARID_COUNT;

private:
  Kokkos::Array<int32_t, VARCOUNT> id2index{};
  Kokkos::Array<bool, VARCOUNT>    field_enabled{};
  int32_t                          m_nbfields = 0;

public:
  void
  enable(VarId id)
  {
    id2index[static_cast<size_t>(id)] = m_nbfields;

    // we add field only if field is not already enabled
    if (!field_enabled[id])
    {
      field_enabled[id] = true;
      m_nbfields++;
    }
  }

  std::set<VarId>
  enabled_fields() const
  {
    std::set<VarId> res;
    for (size_t i = 0; i < VARCOUNT; ++i)
      if (field_enabled[i])
        res.insert(static_cast<VarId>(i));
    return res;
  }

  KOKKOS_INLINE_FUNCTION
  int32_t
  nbfields() const
  {
    return m_nbfields;
  }

  KOKKOS_INLINE_FUNCTION
  int32_t
  operator[](VarId id) const
  {
    assert(field_enabled[static_cast<size_t>(id)]); // This variable is not active
    return id2index[static_cast<size_t>(id)];
  }

  KOKKOS_INLINE_FUNCTION
  bool
  enabled(VarId id) const
  {
    return field_enabled[static_cast<size_t>(id)];
  }

}; // class FieldMap

} // namespace kalypsso

#endif // KALYPSSO_CORE_FIELDMAP_H_
