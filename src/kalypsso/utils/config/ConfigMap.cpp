// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ConfigMap.cpp
 * \brief Implement ConfigMap, essentially a INIReader with additional get methods.
 *
 */
#include "ConfigMap.h"
#include <cstdlib> // for strtof
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm> // for std::transform
#include <numeric>   // for std::accumulate
#include <cerrno>
#include "ini.h"

#include <cctype> // for std::tolower

namespace kalypsso
{

// =======================================================
// =======================================================
ConfigMap::ConfigMap(std::string filename)
{
  m_parse_error = ini_parse(filename.c_str(), valueHandler, this);

  if (this->m_parse_error == -1)
  {
    printf("Error : can't read input parameter file \"%s\"\nPlease check file actually exists !\n",
           filename.c_str());
    std::abort();
  }

} // ConfigMap::ConfigMap

// =======================================================
// =======================================================
ConfigMap::ConfigMap(char *& buffer, size_t buffer_size)
{
  m_parse_error = ini_parse_string_length(buffer, buffer_size, valueHandler, this);
} // ConfigMap::ConfigMap

// =======================================================
// =======================================================
int
ConfigMap::parse_error()
{
  return m_parse_error;
} // ConfigMap::parse_error

// =======================================================
// =======================================================
std::string
ConfigMap::getString(std::string section, std::string name, std::string default_value) const
{

  const std::string key = makeKey(section, name);
  // for const correctness use method at instead of operator[]
  return m_values.count(key) ? m_values.at(key) : default_value;

} // ConfigMap::getString

// =======================================================
// =======================================================
void
ConfigMap::setString(std::string section, std::string name, std::string value)
{

  std::string key = makeKey(section, name);
  m_values[key] = value;

} // ConfigMap::setString

// =======================================================
// =======================================================
std::vector<std::string>
ConfigMap::getStringVector(std::string              section,
                           std::string              name,
                           std::vector<std::string> default_value) const
{
  const std::string key = makeKey(section, name);

  if (m_values.count(key))
  {
    const auto tmp = m_values.at(key);

    // tokenize tmp using a regular expression
    // delimiter is whitespace or comma
    const std::regex re(R"([\s|,]+)");

    return tokenize(tmp, re);
  }

#ifdef KALYPSSO_CORE_CONFIGMAP_VERBOSE
  print_default_value_vector(section, name, default_value);
#endif // KALYPSSO_CORE_CONFIGMAP_VERBOSE
  return default_value;

} // ConfigMap::getStringVector

// =======================================================
// =======================================================
void
ConfigMap::setStringVector(std::string section, std::string name, std::vector<std::string> value)
{

  std::string key = makeKey(section, name);
  auto        dash_fold = [](std::string a, std::string b) { return std::move(a) + ',' + b; };

  m_values[key] = std::accumulate(std::next(value.begin()), value.end(), value[0], dash_fold);

} // ConfigMap::setStringVector

// =======================================================
// =======================================================
int
ConfigMap::getInteger(std::string section, std::string name, int default_value) const
{
  std::string  valstr = getString(section, name, "");
  const char * c_str = valstr.c_str();
  char *       end_ptr;

  // This parses "1234" (decimal) and also "0x4D2" (hex)
  int n = static_cast<int>(strtol(c_str, &end_ptr, 0));

  if (end_ptr == c_str)
  {
#ifdef KALYPSSO_CORE_CONFIGMAP_VERBOSE
    print_default_value_scalar(section, name, default_value);
#endif // KALYPSSO_CORE_CONFIGMAP_VERBOSE
    return default_value;
  }
  else if (*end_ptr != '\0')
  {
    std::cerr << "Unable to convert string \"" << valstr << "\" to integer" << "\n";
    Kokkos::abort("Invalid config map.");
  }

  return n;

} // ConfigMap::getInteger

// =======================================================
// =======================================================
void
ConfigMap::setInteger(std::string section, std::string name, int value)
{
  std::stringstream ss;
  ss << value;

  setString(section, name, ss.str());

} // ConfigMap::setInteger

// =======================================================
// =======================================================
std::vector<int>
ConfigMap::getIntegerVector(std::string      section,
                            std::string      name,
                            std::vector<int> default_value) const
{
  auto str_vec = getStringVector(section, name, std::vector<std::string>{});

  if (str_vec.size() == 0)
  {
#ifdef KALYPSSO_CORE_CONFIGMAP_VERBOSE
    print_default_value_vector(section, name, default_value);
#endif // KALYPSSO_CORE_CONFIGMAP_VERBOSE
    return default_value;
  }

  std::vector<int> result;
  result.reserve(str_vec.size());
  std::transform(
    str_vec.begin(), str_vec.end(), std::back_inserter(result), [&default_value](const auto & str) {
      const char * c_str = str.c_str();
      char *       end_ptr;

      // This parses "1234" (decimal) and also "0x4D2" (hex)
      int n = static_cast<int>(strtol(c_str, &end_ptr, 0));

      if (end_ptr == c_str)
      {
        return default_value[0];
      }
      else if (*end_ptr != '\0')
      {
        std::cerr << "Unable to convert string \"" << str << "\" to integer" << "\n";
        Kokkos::abort("Invalid config map.");
      }
      return n;
    });

  return result;

} // ConfigMap::getIntegerVector

// =======================================================
// =======================================================
void
ConfigMap::setIntegerVector(std::string section, std::string name, std::vector<int> value)
{
  std::vector<std::string> strVector;
  strVector.reserve(value.size());
  std::transform(value.begin(), value.end(), std::back_inserter(strVector), [](const auto & val) {
    std::stringstream ss;
    ss << val;
    return ss.str();
  });

  setStringVector(section, name, strVector);

} // ConfigMap::setIntegerVector

// =======================================================
// =======================================================
int64_t
ConfigMap::getInteger64(std::string section, std::string name, int64_t default_value) const
{
  std::string  valstr = getString(section, name, "");
  const char * c_str = valstr.c_str();
  char *       end_ptr;

  // This parses "1234" (decimal) and also "0x4D2" (hex)
  int64_t n = static_cast<int64_t>(strtoll(c_str, &end_ptr, 0));

  if (end_ptr == c_str)
  {
#ifdef KALYPSSO_CORE_CONFIGMAP_VERBOSE
    print_default_value_scalar(section, name, default_value);
#endif // KALYPSSO_CORE_CONFIGMAP_VERBOSE
    return default_value;
  }
  else if (*end_ptr != '\0')
  {
    std::cerr << "Unable to convert string \"" << valstr << "\" to integer (64 bits)" << "\n";
    Kokkos::abort("Invalid config map.");
  }

  return n;

} // ConfigMap::getInteger64

// =======================================================
// =======================================================
void
ConfigMap::setInteger64(std::string section, std::string name, int64_t value)
{
  std::stringstream ss;
  ss << value;

  setString(section, name, ss.str());

} // ConfigMap::setInteger64

// =======================================================
// =======================================================
std::vector<int64_t>
ConfigMap::getInteger64Vector(std::string          section,
                              std::string          name,
                              std::vector<int64_t> default_value) const
{
  auto str_vec = getStringVector(section, name, std::vector<std::string>{});

  if (str_vec.size() == 0)
  {
#ifdef KALYPSSO_CORE_CONFIGMAP_VERBOSE
    print_default_value_vector(section, name, default_value);
#endif // KALYPSSO_CORE_CONFIGMAP_VERBOSE
    return default_value;
  }

  std::vector<int64_t> result;
  result.reserve(str_vec.size());
  std::transform(
    str_vec.begin(), str_vec.end(), std::back_inserter(result), [&default_value](const auto & str) {
      const char * c_str = str.c_str();
      char *       end_ptr;

      // This parses "1234" (decimal) and also "0x4D2" (hex)
      int64_t n = static_cast<int64_t>(strtoll(c_str, &end_ptr, 0));

      if (end_ptr == c_str)
      {
        return default_value[0];
      }
      else if (*end_ptr != '\0')
      {
        std::cerr << "Unable to convert string \"" << str << "\" to 64 bits integer" << "\n";
        Kokkos::abort("Invalid config map.");
      }
      return n;
    });

  return result;

} // ConfigMap::getInteger64Vector

// =======================================================
// =======================================================
void
ConfigMap::setInteger64Vector(std::string section, std::string name, std::vector<int64_t> value)
{
  std::vector<std::string> strVector;
  strVector.reserve(value.size());
  std::transform(value.begin(), value.end(), std::back_inserter(strVector), [](const auto & val) {
    std::stringstream ss;
    ss << val;
    return ss.str();
  });

  setStringVector(section, name, strVector);

} // ConfigMap::setInteger64Vector

// =======================================================
// =======================================================
float
ConfigMap::getFloat(std::string section, std::string name, float default_value) const
{
  std::string  valstr = getString(section, name, "");
  const char * c_str = valstr.c_str();
  char *       end_ptr;

  const auto valFloat = strtof(c_str, &end_ptr);

  if (end_ptr == c_str)
  {
#ifdef KALYPSSO_CORE_CONFIGMAP_VERBOSE
    print_default_value_scalar(section, name, default_value);
#endif // KALYPSSO_CORE_CONFIGMAP_VERBOSE
    return default_value;
  }
  else if (*end_ptr != '\0')
  {
    std::cerr << "Unable to convert string \"" << valstr << "\" to float" << "\n";
    Kokkos::abort("Invalid config map.");
  }

  return valFloat;

} // ConfigMap::getFloat

// =======================================================
// =======================================================
void
ConfigMap::setFloat(std::string section, std::string name, float value)
{

  std::stringstream ss;
  ss << value;

  setString(section, name, ss.str());

} // ConfigMap::setFloat

// =======================================================
// =======================================================
std::vector<float>
ConfigMap::getFloatVector(std::string        section,
                          std::string        name,
                          std::vector<float> default_value) const
{
  auto str_vec = getStringVector(section, name, std::vector<std::string>{});

  if (str_vec.size() == 0)
  {
#ifdef KALYPSSO_CORE_CONFIGMAP_VERBOSE
    print_default_value_vector(section, name, default_value);
#endif // KALYPSSO_CORE_CONFIGMAP_VERBOSE
    return default_value;
  }

  std::vector<float> result;
  result.reserve(str_vec.size());
  std::transform(
    str_vec.begin(), str_vec.end(), std::back_inserter(result), [&default_value](const auto & str) {
      const char * c_str = str.c_str();
      char *       end_ptr;

      const auto valf = strtof(c_str, &end_ptr);

      if (end_ptr == c_str)
      {
        return default_value[0];
      }
      else if (*end_ptr != '\0')
      {
        std::cerr << "Unable to convert string \"" << str << "\" to float" << "\n";
        Kokkos::abort("Invalid config map.");
      }
      return valf;
    });

  return result;

} // ConfigMap::getFloatVector

// =======================================================
// =======================================================
void
ConfigMap::setFloatVector(std::string section, std::string name, std::vector<float> value)
{
  std::vector<std::string> strVector;
  strVector.reserve(value.size());
  std::transform(value.begin(), value.end(), std::back_inserter(strVector), [](const float & val) {
    std::stringstream ss;
    ss << val;
    return ss.str();
  });

  setStringVector(section, name, strVector);

} // ConfigMap::setFloatVector

// =======================================================
// =======================================================
double
ConfigMap::getDouble(std::string section, std::string name, double default_value) const
{
  std::string  valstr = getString(section, name, "");
  const char * c_str = valstr.c_str();
  char *       end_ptr;

  const auto valDouble = strtod(c_str, &end_ptr);

  if (end_ptr == c_str)
  {
#ifdef KALYPSSO_CORE_CONFIGMAP_VERBOSE
    print_default_value_scalar(section, name, default_value);
#endif // KALYPSSO_CORE_CONFIGMAP_VERBOSE
    return default_value;
  }
  else if (*end_ptr != '\0')
  {
    std::cerr << "Unable to convert string \"" << valstr << "\" to double" << "\n";
    Kokkos::abort("Invalid config map.");
  }

  return valDouble;

} // ConfigMap::getDouble

// =======================================================
// =======================================================
void
ConfigMap::setDouble(std::string section, std::string name, double value)
{

  std::stringstream ss;
  ss << value;

  setString(section, name, ss.str());

} // ConfigMap::setDouble

// =======================================================
// =======================================================
std::vector<double>
ConfigMap::getDoubleVector(std::string         section,
                           std::string         name,
                           std::vector<double> default_value) const
{
  auto str_vec = getStringVector(section, name, std::vector<std::string>{});

  if (str_vec.size() == 0)
  {
#ifdef KALYPSSO_CORE_CONFIGMAP_VERBOSE
    print_default_value_vector(section, name, default_value);
#endif // KALYPSSO_CORE_CONFIGMAP_VERBOSE
    return default_value;
  }

  std::vector<double> result;
  result.reserve(str_vec.size());
  std::transform(
    str_vec.begin(), str_vec.end(), std::back_inserter(result), [&default_value](const auto & str) {
      const char * c_str = str.c_str();
      char *       end_ptr;

      double valf = strtod(c_str, &end_ptr);

      if (end_ptr == c_str)
      {
        return default_value[0];
      }
      else if (*end_ptr != '\0')
      {
        std::cerr << "Unable to convert string \"" << str << "\" to double" << "\n";
        Kokkos::abort("Invalid config map.");
      }
      return valf;
    });

  return result;

} // ConfigMap::getDoubleVector

// =======================================================
// =======================================================
void
ConfigMap::setDoubleVector(std::string section, std::string name, std::vector<double> value)
{
  std::vector<std::string> strVector;
  strVector.reserve(value.size());
  std::transform(value.begin(), value.end(), std::back_inserter(strVector), [](const double & val) {
    std::stringstream ss;
    ss << val;
    return ss.str();
  });

  setStringVector(section, name, strVector);

} // ConfigMap::setDoubleVector

#ifdef KALYPSSO_CORE_USE_DOUBLE

// =======================================================
// =======================================================
double
ConfigMap::getReal(std::string section, std::string name, double default_value) const
{
  return getDouble(section, name, default_value);
} // ConfigMap::getReal

// =======================================================
// =======================================================
std::vector<double>
ConfigMap::getRealVector(std::string         section,
                         std::string         name,
                         std::vector<double> default_value) const
{
  return getDoubleVector(section, name, default_value);
} // ConfigMap::getRealVector

#else

// =======================================================
// =======================================================
float
ConfigMap::getReal(std::string section, std::string name, float default_value) const
{
  return getFloat(section, name, default_value);
} // ConfigMap::getReal

// =======================================================
// =======================================================
std::vector<float>
ConfigMap::getRealVector(std::string        section,
                         std::string        name,
                         std::vector<float> default_value) const
{
  return getFloatVector(section, name, default_value);
} // ConfigMap::getRealVector

#endif // KALYPSSO_CORE_USE_DOUBLE

// =======================================================
// =======================================================
bool
ConfigMap::getBool(std::string section, std::string name, bool default_value) const
{
  bool        val = default_value;
  std::string valstr = getString(section, name, "");

  if (!valstr.compare("1") or !valstr.compare("yes") or !valstr.compare("true") or
      !valstr.compare("on"))
    val = true;
  if (!valstr.compare("0") or !valstr.compare("no") or !valstr.compare("false") or
      !valstr.compare("off"))
    val = false;

  // if valstr is empty, return the default value
  if (!valstr.size())
  {
#ifdef KALYPSSO_CORE_CONFIGMAP_VERBOSE
    print_default_value_scalar(section, name, default_value);
#endif // KALYPSSO_CORE_CONFIGMAP_VERBOSE
    val = default_value;
  }

  return val;

} // ConfigMap::getBool

// =======================================================
// =======================================================
void
ConfigMap::setBool(std::string section, std::string name, bool value)
{

  if (value)
    setString(section, name, "true");
  else
    setString(section, name, "false");

} // ConfigMap::setBool

// =======================================================
// =======================================================
std::vector<bool>
ConfigMap::getBoolVector(std::string       section,
                         std::string       name,
                         std::vector<bool> default_value) const
{
  auto str_vec = getStringVector(section, name, std::vector<std::string>{});

  if (str_vec.size() == 0)
  {
#ifdef KALYPSSO_CORE_CONFIGMAP_VERBOSE
    print_default_value_vector(section, name, default_value);
#endif // KALYPSSO_CORE_CONFIGMAP_VERBOSE
    return default_value;
  }

  std::vector<bool> result;
  result.reserve(str_vec.size());
  std::transform(str_vec.begin(),
                 str_vec.end(),
                 std::back_inserter(result),
                 [&default_value](const auto & valstr) {
                   bool val = default_value.size() > 0 ? default_value[0] : false;
                   if (!valstr.compare("1") or !valstr.compare("yes") or !valstr.compare("true") or
                       !valstr.compare("on"))
                     val = true;
                   if (!valstr.compare("0") or !valstr.compare("no") or !valstr.compare("false") or
                       !valstr.compare("off"))
                     val = false;
                   return val;
                 });

  return result;

} // ConfigMap::getBoolVector

// =======================================================
// =======================================================
void
ConfigMap::setBoolVector(std::string section, std::string name, std::vector<bool> value)
{
  std::vector<std::string> strVector;
  strVector.reserve(value.size());
  std::transform(value.begin(), value.end(), std::back_inserter(strVector), [](const bool & val) {
    std::stringstream ss;
    ss << (val ? "true" : "false");
    return ss.str();
  });

  setStringVector(section, name, strVector);

} // ConfigMap::setBoolVector

// =======================================================
// =======================================================
std::ostream &
operator<<(std::ostream & os, const ConfigMap & cfg)
{
  std::map<std::string, std::string>::const_iterator it;
  for (it = cfg.m_values.begin(); it != cfg.m_values.end(); it++)
  {
    os << (*it).first << " = " << (*it).second << "\n";
  }
  return os;
}

// =======================================================
// =======================================================
std::string
ConfigMap::makeKey(std::string section, std::string name)
{
  std::string key = section + "." + name;

  // Convert to lower case to make lookups case-insensitive
  std::transform(
    key.begin(), key.end(), key.begin(), [](unsigned char c) { return std::tolower(c); });

  return key;

} // ConfigMap::makeKey

// =======================================================
// =======================================================
int
ConfigMap::valueHandler(void * user, const char * section, const char * name, const char * value)
{
  if (!name) // Happens when INI_CALL_HANDLER_ON_NEW_SECTION enabled
    return 1;
  ConfigMap * reader = static_cast<ConfigMap *>(user);
  const auto  key = makeKey(section, name);
  if (reader->m_values[key].size() > 0)
    reader->m_values[key] += "\n";
  reader->m_values[key] += value ? value : "";

  return 1;

} // ConfigMap::valueHandler

// =======================================================
// =======================================================
ConfigMap
broadcast_parameters(std::string filename)
{
#ifdef KALYPSSO_CORE_USE_MPI

  int myRank;
  int nTasks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nTasks);

  char * buffer = nullptr;
  int    buffer_size = 0;

  // MPI rank 0 reads parameter file
  if (myRank == 0)
  {

    // open file and go to the end to get file size in bytes
    std::ifstream filein(filename.c_str(), std::ifstream::ate);
    if (filein.good() and filein.is_open())
    {
      // file size must be smaller than 2^32 bytes (should be ok for a config file)
      int file_size = static_cast<int>(filein.tellg());

      filein.seekg(0); // rewind

      buffer_size = file_size;
      buffer = new char[buffer_size];

      filein.read(buffer, buffer_size);
    }
    else
    {
      printf(
        "Error : can't read input parameter file \"%s\"\nPlease check file actually exists !\n",
        filename.c_str());
      std::abort();
    }
  }

  // broadcast buffer size (collective)
  MPI_Bcast(&buffer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (buffer_size > 0)
  {

    // all other MPI task need to allocate buffer
    if (myRank > 0)
    {
      // printf("I'm rank %d allocating buffer of size %d\n",myRank,buffer_size);
      buffer = new char[buffer_size];
    }

    // broastcast buffer itself (collective)
    MPI_Bcast(&buffer[0], buffer_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // now all MPI rank should have buffer filled, try to build a ConfigMap
    ConfigMap config_map(buffer, static_cast<size_t>(buffer_size));

    if (buffer)
      delete[] buffer;

    return config_map;
  }

  // return default config_map (surely not what we wanted)
  char *    null_buffer = nullptr;
  ConfigMap defaultConfig(null_buffer, 0);

  return defaultConfig;

#else
  ConfigMap config_map(filename);
  return config_map;
#endif

} // broadcast_parameters

} // namespace kalypsso
