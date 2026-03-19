// SPDX-FileCopyrightText: 2025 kalypsso-core authors
// © Commissariat a l'Energie Atomique et aux Energies Alternatives (CEA)
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/utils/config/ConfigMap.h>

#include <sstream>

#include <gtest/gtest.h>

namespace kalypsso
{

TEST(kalypsso_shared_ConfigMap_test, config_map_test)
{
  // make test.ini file
  std::stringstream ini_data;
  ini_data << "# Test config for unit testing " << std::endl;

  ini_data << "[run] " << std::endl;
  ini_data << "solver_name=godunov ; solver name" << std::endl;
  ini_data << "materials=air,water r22, copper ; solver material name" << std::endl;
  ini_data << "some_ints= 34 , 2,-2 9,0 ; some integers" << std::endl;
  ini_data << "some_floats= -1.0f 6.5 3.2e5 1.34e-4 ; some floats" << std::endl;
  ini_data << "some_bools= yes no, 1, 0 true, false ; some boolean values" << std::endl;
  ini_data << "other_ints= 34 , 2, -2 ; some integers" << std::endl;
  ini_data << "drinks=water,milk,chocolate " << std::endl;
  ini_data << " ,coffee,tea" << std::endl;


  char * buffer = strdup(ini_data.str().c_str());
  size_t buf_size = ini_data.str().size();

  // create a config_map
  ConfigMap config_map(buffer, buf_size);

  // test string
  EXPECT_EQ(config_map.getString("run", "solver_name", "unknown"), "godunov");
  EXPECT_EQ(config_map.getString("run", "solver_name2", "unknown"), "unknown");

  {
    // test read vector of string
    auto materials = config_map.getStringVector("run", "materials", std::vector<std::string>{ "" });

    EXPECT_EQ(materials[0], "air");
    EXPECT_EQ(materials[1], "water");
    EXPECT_EQ(materials[2], "r22");
    EXPECT_EQ(materials[3], "copper");

    auto materials2 =
      config_map.getStringVector("run", "materials2", std::vector<std::string>{ "" });
    EXPECT_EQ(materials2.size(), 1);
    EXPECT_EQ(materials2[0], "");

    auto materials3 =
      config_map.getStringVector("run", "materials3", std::vector<std::string>{ "gold", "silver" });
    EXPECT_EQ(materials3.size(), 2);
    EXPECT_EQ(materials3[0], "gold");
  }

  {
    // test write vector of string
    config_map.setStringVector(
      "run", "materials4", std::vector<std::string>{ "wood ", " iron", " water " });
    auto materials4 =
      config_map.getStringVector("run", "materials4", std::vector<std::string>{ "" });
    EXPECT_EQ(materials4.size(), 3);
    EXPECT_EQ(materials4[0], "wood");
    EXPECT_EQ(materials4[1], "iron");
    EXPECT_EQ(materials4[2], "water");
  }

  auto some_ints = config_map.getIntegerVector("run", "some_ints", std::vector<int>{ 8, 8 });
  EXPECT_EQ(some_ints[0], 34);
  EXPECT_EQ(some_ints[1], 2);
  EXPECT_EQ(some_ints[2], -2);
  EXPECT_EQ(some_ints[3], 9);
  EXPECT_EQ(some_ints[4], 0);

  // vector of 64 bit integers
  {
    config_map.setInteger64Vector("run", "int64", std::vector<int64_t>{ 34000000000, 68000000000 });
    auto int64 = config_map.getInteger64Vector("run", "int64", std::vector<int64_t>{ 0 });
    EXPECT_EQ(int64.size(), 2);
    EXPECT_EQ(int64[0], 34000000000);
    EXPECT_EQ(int64[1], 68000000000);

    auto int32 = config_map.getIntegerVector("run", "int64", std::vector<int32_t>{ 0 });
    EXPECT_EQ(int32.size(), 2);
    EXPECT_NE(int32[0], 34000000000);
    EXPECT_NE(int32[1], 68000000000);
  }

  // vector of floats
  {
    config_map.setFloatVector("run", "floats", std::vector<float>{ 1.0f, 1.34e-4f });
    auto floats = config_map.getFloatVector("run", "floats", std::vector<float>{ 0 });
    EXPECT_EQ(floats.size(), 2);
    EXPECT_EQ(floats[0], 1.0f);
    EXPECT_EQ(floats[1], 1.34e-4f);

    auto floats2 = config_map.getFloatVector("run", "some_floats", std::vector<float>{});
    EXPECT_EQ(floats2.size(), 4);
    EXPECT_EQ(floats2[0], -1.0f);
    EXPECT_EQ(floats2[1], 6.5);
    EXPECT_EQ(floats2[2], 3.2e5);
    EXPECT_EQ(floats2[3], 1.34e-4f);
  }

  // vector of bools
  {
    auto bools = config_map.getBoolVector("run", "some_bools", std::vector<bool>{ false });
    EXPECT_EQ(bools.size(), 6);
    EXPECT_EQ(bools[0], true);
    EXPECT_EQ(bools[1], false);
    EXPECT_EQ(bools[2], true);
    EXPECT_EQ(bools[3], false);
    EXPECT_EQ(bools[4], true);
    EXPECT_EQ(bools[5], false);
  }

  // vector of ints as a Kokkos::Array
  {
    auto vec_ints =
      config_map.getIntegerVector<3>("run", "other_ints", Kokkos::Array<int, 3>{ 0, 1, 2 });

    EXPECT_EQ(vec_ints[0], 34);
    EXPECT_EQ(vec_ints[1], 2);
    EXPECT_EQ(vec_ints[2], -2);
  }

  {
    auto vec_ints =
      config_map.getIntegerVector<2>("run", "other_ints", Kokkos::Array<int, 2>{ 0, 1 });

    EXPECT_EQ(vec_ints[0], 34);
    EXPECT_EQ(vec_ints[1], 2);
  }

  // a multiline list
  {
    auto drinks = config_map.getStringVector("run", "drinks", std::vector<std::string>{});
    EXPECT_EQ(drinks.size(), 5);
    EXPECT_EQ(drinks[0], "water");
    EXPECT_EQ(drinks[4], "tea");
  }

  free(buffer);
}

} // namespace kalypsso
