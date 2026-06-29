// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ConfigMap.h
 * \brief Define an object will allow to easily retrieve parameter from a dictionary.
 *
 */
#ifndef KALYPSSO_UTILS_CONFIGMAP_CONFIGMAP_H_
#define KALYPSSO_UTILS_CONFIGMAP_CONFIGMAP_H_

#include <kalypsso/core/kalypsso_core_config.h> // for KALYPSSO_CORE_USE_DOUBLE, ...

#include <map>
#include <string>
#include <cstdint>
#include <vector>
#include <regex>
#include <algorithm>
#include <iostream>

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#ifdef KALYPSSO_CORE_USE_MPI
#  include <mpi.h>
#endif

namespace kalypsso
{

/**
 * check if a given item is present in a vector of items.
 */
template <typename T>
bool
is_present(std::vector<T> const & vec, T const & item)
{
  return std::find(vec.begin(), vec.end(), item) != vec.end();
}

/**
 * \class ConfigMap ConfigMap.h
 * \brief This is a specialized version of ConfigMap which reads and parses a INI
 * file into a key-value map (implemented using std::map). This class
 * is useful to gather parameters.
 */
class ConfigMap
{
public:
  ConfigMap(std::string filename);
  ConfigMap(char *& buffer, size_t buffer_size);
  ~ConfigMap() = default;

  /**
   * Return the result of ini_parse(), i.e., 0 on success, line number of
   * first error on parse error, or -1 on file open error.
   */
  int
  parse_error();

  // =================================================================================

  //! Get a string value from INI file, returning default_value if not found.
  std::string
  getString(std::string section, std::string name, std::string default_value) const;

  //! Set a string value to section/name.
  void
  setString(std::string section, std::string name, std::string value);

  //! Get a vector<string> value from INI file, returning default_value if not found.
  std::vector<std::string>
  getStringVector(std::string              section,
                  std::string              name,
                  std::vector<std::string> default_value) const;

  //! Set a vector of strings value to section/name.
  void
  setStringVector(std::string section, std::string name, std::vector<std::string> value);

  // =================================================================================

  //! Get an integer (int) value from INI file, returning default_value if not found.
  int
  getInteger(std::string section, std::string name, int default_value) const;

  //! Set an integer value to an section/name, reverse operation of set getInteger.
  void
  setInteger(std::string section, std::string name, int value);

  //! Get a vector of integer (int) value from INI file, returning default_value if not found.
  std::vector<int>
  getIntegerVector(std::string section, std::string name, std::vector<int> default_value) const;

  //! Set a vector of integer to an section/name, reverse operation of set getIntegerVector.
  void
  setIntegerVector(std::string section, std::string name, std::vector<int> value);

  //! Get a vector of integer (int) value from INI file, returning default_value if not found.
  //! Interesting when we know at compile time how many values are required
  template <size_t N>
  auto
  getIntegerVector(std::string           section,
                   std::string           name,
                   Kokkos::Array<int, N> default_value) const -> Kokkos::Array<int, N>

  {
    auto vec_int = getIntegerVector(section, name, std::vector<int>{});

    if (vec_int.size() >= N)
    {
      Kokkos::Array<int, N> res;

      for (size_t i = 0; i < N; ++i)
      {
        res[i] = vec_int[i];
      }

      return res;
    }

    return default_value;
  }

  // =================================================================================

  //! Get an integer (64 bits) value from INI file, returning default_value if not found.
  int64_t
  getInteger64(std::string section, std::string name, int64_t default_value) const;

  //! Set an integer value to an section/name, reverse operation of set getInteger.
  void
  setInteger64(std::string section, std::string name, int64_t value);

  //! Get a vector of 64 bit integers (int) value from INI file, returning default_value if not
  //! found.
  std::vector<int64_t>
  getInteger64Vector(std::string          section,
                     std::string          name,
                     std::vector<int64_t> default_value) const;

  //! Set a vector of 64 bits integers to an section/name, reverse operation of set
  //! getInteger64Vector.
  void
  setInteger64Vector(std::string section, std::string name, std::vector<int64_t> value);

  //! Get a vector of 64 bits integer (int) value from INI file, returning default_value if not
  //! found. Interesting when we know at compile time how many values are required
  template <size_t N>
  auto
  getInteger64Vector(std::string               section,
                     std::string               name,
                     Kokkos::Array<int64_t, N> default_value) const -> Kokkos::Array<int64_t, N>

  {
    auto vec_int = getInteger64Vector(section, name, std::vector<int64_t>{});

    if (vec_int.size() >= N)
    {
      Kokkos::Array<int64_t, N> res;

      for (size_t i = 0; i < N; ++i)
      {
        res[i] = vec_int[i];
      }

      return res;
    }

    return default_value;
  }

  // =================================================================================

  //! Get a floating point value from the map.
  float
  getFloat(std::string section, std::string name, float default_value) const;

  //! Set a floating point value to a section/name.
  void
  setFloat(std::string section, std::string name, float value);

  //! Get a vector of float values from INI file, returning default_value if not found.
  std::vector<float>
  getFloatVector(std::string section, std::string name, std::vector<float> default_value) const;

  //! Set a vector of float to an section/name, reverse operation of set getFloatVector.
  void
  setFloatVector(std::string section, std::string name, std::vector<float> value);

  //! Get a vector of float values from INI file, returning default_value if not found.
  //! Interesting when we know at compile time how many values are required
  template <size_t N>
  auto
  getFloatVector(std::string             section,
                 std::string             name,
                 Kokkos::Array<float, N> default_value) const -> Kokkos::Array<float, N>

  {
    auto vec_float = getFloatVector(section, name, std::vector<float>{});

    if (vec_float.size() >= N)
    {
      Kokkos::Array<float, N> res;

      for (size_t i = 0; i < N; ++i)
      {
        res[i] = vec_float[i];
      }

      return res;
    }

    return default_value;
  }

  // =================================================================================

  //! Get a floating point value from the map.
  double
  getDouble(std::string section, std::string name, double default_value) const;

  //! Set a floating point value to a section/name.
  void
  setDouble(std::string section, std::string name, double value);

  //! Get a vector of double values from INI file, returning default_value if not found.
  std::vector<double>
  getDoubleVector(std::string section, std::string name, std::vector<double> default_value) const;

  //! Set a vector of double to an section/name, reverse operation of set getDoubleVector.
  void
  setDoubleVector(std::string section, std::string name, std::vector<double> value);

  //! Get a vector of double values from INI file, returning default_value if not found.
  //! Interesting when we know at compile time how many values are required
  template <size_t N>
  auto
  getDoubleVector(std::string              section,
                  std::string              name,
                  Kokkos::Array<double, N> default_value) const -> Kokkos::Array<double, N>
  {
    auto vec_double = getDoubleVector(section, name, std::vector<double>{});

    if (vec_double.size() >= N)
    {
      Kokkos::Array<double, N> res;

      for (size_t i = 0; i < N; ++i)
      {
        res[i] = vec_double[i];
      }

      return res;
    }

    return default_value;
  }

#ifdef KALYPSSO_CORE_USE_DOUBLE
  //! Get a floating point value from the map.
  double
  getReal(std::string section, std::string name, double default_value) const;

  //! Get a vector of floating point values from the map.
  std::vector<double>
  getRealVector(std::string section, std::string name, std::vector<double> default_value) const;

  //! Get a vector of real values from INI file, returning default_value if not found.
  //! Interesting when we know at compile time how many values are required
  template <size_t N>
  auto
  getRealVector(std::string              section,
                std::string              name,
                Kokkos::Array<double, N> default_value) const -> Kokkos::Array<double, N>
  {
    return getDoubleVector(section, name, default_value);
  }
#else
  //! Get a floating point value from the map.
  float
  getReal(std::string section, std::string name, float default_value) const;

  //! Get a vector of floating point values from the map.
  std::vector<float>
  getRealVector(std::string section, std::string name, std::vector<float> default_value) const;

  //! Get a vector of real values from INI file, returning default_value if not found.
  //! Interesting when we know at compile time how many values are required
  template <size_t N>
  auto
  getRealVector(std::string             section,
                std::string             name,
                Kokkos::Array<float, N> default_value) const -> Kokkos::Array<float, N>
  {
    return getFloatVector(section, name, default_value);
  }
#endif // KALYPSSO_CORE_USE_DOUBLE

  // =================================================================================

  //! Get a boolean value from the map.
  bool
  getBool(std::string section, std::string name, bool default_value) const;

  //! Set a boolean value to a section/name.
  void
  setBool(std::string section, std::string name, bool value);

  //! Get a vector of boolean values from the map.
  std::vector<bool>
  getBoolVector(std::string section, std::string name, std::vector<bool> default_value) const;

  //! Set a vector of boolean values to a section/name.
  void
  setBoolVector(std::string section, std::string name, std::vector<bool> value);

  //! Get a vector of bool values from INI file, returning default_value if not found.
  //! Interesting when we know at compile time how many values are required
  template <size_t N>
  auto
  getBoolVector(std::string            section,
                std::string            name,
                Kokkos::Array<bool, N> default_value) const -> Kokkos::Array<bool, N>
  {
    auto vec_bool = getBoolVector(section, name, std::vector<bool>{});

    if (vec_bool.size() >= N)
    {
      Kokkos::Array<bool, N> res;

      for (size_t i = 0; i < N; ++i)
      {
        res[i] = vec_bool[i];
      }

      return res;
    }

    return default_value;
  }

  // =================================================================================

  //! Print the content of a ConfigMap object
  friend std::ostream &
  operator<<(std::ostream & os, const ConfigMap & cfg);

private:
  int m_parse_error;

  std::map<std::string, std::string> m_values;

  static std::string
  makeKey(std::string section, std::string name);

  static int
  valueHandler(void * user, const char * section, const char * name, const char * value);

  /**
   * \brief Tokenize a given string according to the regex
   * and remove the empty tokens.
   *
   * \param str
   * \param re
   * \return std::vector<std::string>
   */
  std::vector<std::string>
  tokenize(const std::string str, const std::regex re) const
  {
    std::sregex_token_iterator it{ str.begin(), str.end(), re, -1 };
    std::vector<std::string>   tokenized{ it, {} };

    // Additional check to remove empty strings
    tokenized.erase(std::remove_if(tokenized.begin(),
                                   tokenized.end(),
                                   [](std::string const & s) { return s.size() == 0; }),
                    tokenized.end());

    return tokenized;
  }

  template <typename T>
  std::vector<T>
  tokenize_as(const std::string str)
  {}

  /**
   * Print default value for a scalar parameter.
   */
  template <typename T>
  void
  print_default_value_scalar(std::string const & section,
                             std::string const & name,
                             T                   default_value) const
  {
    int myRank = 0;
    int nTasks = 1;
#ifdef KALYPSSO_CORE_USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nTasks);
#endif

    if (myRank == 0)
    {
      std::cout << "Using default value for parameter " << section << ":" << name << " = "
                << default_value << "\n";
    }
  } // print_default_value_scalar

  /**
   * Print default value for a scalar parameter.
   */
  template <typename T>
  void
  print_default_value_vector(std::string const &    section,
                             std::string const &    name,
                             std::vector<T> const & default_value) const
  {
    int myRank = 0;
    int nTasks = 1;
#ifdef KALYPSSO_CORE_USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nTasks);
#endif

    if (myRank == 0)
    {
      for (size_t i = 0; i < default_value.size(); ++i)
      {
        std::cout << "Using default value for parameter " << section << ":" << name << "[" << i
                  << "]" << " = " << default_value[i] << "\n";
      }
    }
  } // print_default_value

}; // class ConfigMap

/**
 * Builds a ConfigMap object from the input parameter file.
 *
 * ConfigMap is return by value here.
 */
ConfigMap
broadcast_parameters(std::string filename);

} // namespace kalypsso

#endif // KALYPSSO_UTILS_CONFIGMAP_CONFIGMAP_H_
