# SPDX-License-Identifier: Apache-2.0
#
# adapted from https://github.com/apache/kvrocks/tree/unstable

include_guard()

include(FetchContent)

macro(parse_var arg key value)
  string(REGEX REPLACE "^(.+)=(.+)$" "\\1;\\2" REGEX_RESULT ${arg})
  list(GET REGEX_RESULT 0 ${key})
  list(GET REGEX_RESULT 1 ${value})
endmacro()

function(FetchContent_MakeAvailableWithArgs dep)
  if(NOT ${dep}_POPULATED)
    message("Fetching ${dep}...")
    FetchContent_Populate(${dep})

    foreach(arg IN LISTS ARGN)
      parse_var(${arg} key value)
      set(${key}_OLD ${${key}})
      set(${key}
          ${value}
          CACHE INTERNAL "")
      message("In ${dep}: ${key} set to ${value}")
    endforeach()

    add_subdirectory(${${dep}_SOURCE_DIR} ${${dep}_BINARY_DIR} EXCLUDE_FROM_ALL)

    foreach(arg IN LISTS ARGN)
      parse_var(${arg} key value)
      set(${key}
          ${${key}_OLD}
          CACHE INTERNAL "")
    endforeach()
  endif()
endfunction()
