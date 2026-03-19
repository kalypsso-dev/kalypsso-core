/**
 * \file ParameterValue.h
 * \brief a simple class to represent a runtime configurable parameter.
 *
 * This source code is currently not used but was considered as useful for
 * potential refactoring.
 *
 */
#ifndef KALYPSSO_UTILS_CONFIG_PARAMETERVALUE_H_
#define KALYPSSO_UTILS_CONFIG_PARAMETERVALUE_H_

#include <kalypsso/core/kalypsso_core_config.h> // for KALYPSSO_CORE_USE_DOUBLE, ...

#include <cstdint>
#include <string>
#include <variant>

namespace kalypsso
{

/**
 * A data structure holding data associated to runtime modifiable parameters.
 *
 * All parameters are stored in a std::map<std::string, ParameterValue>.
 */
struct ParameterValue
{
  using param_t = std::variant<bool, int32_t, int64_t, float, double, std::string>;

  //! Holded value (string, integer, float, double, etc ...)
  param_t value;

  //! A minimal doc string for this parameter
  std::string doc_string;

  //! Status boolean variable set to true if parameter was initialized from the default value.
  //! By default is is true, and reset to false if parameter is present in input file.
  bool defaulted;

  //! Set to true the first time API touch this parameter.
  //! Should be useful
  bool used;

  ParameterValue()
    : value(false)
    , doc_string("No documentation available")
    , defaulted(true)
    , used(false)
  {}

  template <typename T>
  ParameterValue(T _value, std::string _doc_string)
    : value(_value)
    , doc_string(_doc_string)
    , defaulted(true)
    , used(false)
  {}

  ~ParameterValue() = default;

}; // ParameterValue

} // namespace kalypsso

#endif // KALYPSSO_UTILS_CONFIG_PARAMETERVALUE_H_
