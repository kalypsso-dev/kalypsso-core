// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <fstream>
#include <kalypsso/utils/config/ConfigMap.h>

namespace kalypsso
{

int
config_map_test()
{
  // make test.ini file
  std::fstream iniFile;
  iniFile.open("./test.ini", std::ios_base::out);
  iniFile << "; Test config file for ini_test.c" << std::endl;

  iniFile << "[Protocol]             ; Protocol configuration" << std::endl;
  iniFile << "Version=6              ; IPv6" << std::endl;

  iniFile << "[User]" << std::endl;
  iniFile << "Name = Bob Smith       ; Spaces around '=' are stripped" << std::endl;
  iniFile << "Email = bob@smith.com  ; And comments (like this) ignored" << std::endl;
  iniFile.close();

  // create a INIReader instance
  ConfigMap reader("./test.ini");

  if (reader.parse_error() < 0)
  {
    std::cout << "Can't load 'test.ini'\n";
    return 1;
  }
  std::cout << "Config loaded from 'test.ini': version="
            << reader.getInteger("protocol", "version", -1)
            << ", name=" << reader.getString("user", "name", "UNKNOWN")
            << ", email=" << reader.getString("user", "email", "UNKNOWN") << "\n";

  ConfigMap reader2 = reader;
  std::cout << std::endl;
  std::cout << "Config loaded from reader2: version="
            << reader.getInteger("protocol", "version", -1)
            << ", name=" << reader.getString("user", "name", "UNKNOWN")
            << ", email=" << reader.getString("user", "email", "UNKNOWN") << "\n";

  return 0;
}

} // namespace kalypsso

// ========================================================================
// ========================================================================
int
main([[maybe_unused]] int argc, [[maybe_unused]] char * argv[])
{

  const auto res = kalypsso::config_map_test();

  return res;
}
