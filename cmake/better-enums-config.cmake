message("[kalypsso-core] Create imported target for better-enums")

add_library(better-enums INTERFACE IMPORTED GLOBAL)

# cmake-format: off
set_target_properties(
  better-enums PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/external/better-enums)
# cmake-format: on

# create library alias
add_library(better-enums::better-enums ALIAS better-enums)
