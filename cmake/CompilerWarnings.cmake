# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 Jason Turner
# SPDX-FileCopyrightText: 2025 kalypsso-core authors
# original version: https://github.com/cpp-best-practices/cmake_template

#
# Default values for warning flags are lightweight; Wall is a minimum.
#
# This macro will set the following variables (if not already set on the command-line or via
# environment variables):
#
# * CLANG_C_WARNINGS_DEFAULT
# * CLANG_CXX_WARNINGS_DEFAULT
# * GCC_C_WARNINGS_DEFAULT
# * GCC_CXX_WARNINGS_DEFAULT
# * CUDA_WARNINGS_DEFAULT
#
# example:
#
# * using cmake variable: GCC_CXX_WARNINGS_DEFAULT="-Wall;-Wshadow"
# * using environment variable: export GCC_CXX_WARNINGS_DEFAULT="-Wall;-Wshadow"
#
macro(kalypsso_set_default_warning_flags)

  # use basic default warnings flags
  if(DEFINED ENV{CLANG_C_WARNINGS_DEFAULT})
    set(CLANG_C_WARNINGS_DEFAULT $ENV{CLANG_C_WARNINGS_DEFAULT})
  elseif(NOT DEFINED CLANG_C_WARNINGS_DEFAULT)
    set(CLANG_C_WARNINGS_DEFAULT -Wall)
  endif()

  if(DEFINED ENV{CLANG_CXX_WARNINGS_DEFAULT})
    set(CLANG_CXX_WARNINGS_DEFAULT $ENV{CLANG_CXX_WARNINGS_DEFAULT})
  elseif(NOT DEFINED CLANG_CXX_WARNINGS_DEFAULT)
    set(CLANG_CXX_WARNINGS_DEFAULT -Wall)
  endif()

  if(DEFINED ENV{GCC_C_WARNINGS_DEFAULT})
    set(GCC_C_WARNINGS_DEFAULT $ENV{GCC_C_WARNINGS_DEFAULT})
  elseif(NOT DEFINED GCC_C_WARNINGS_DEFAULT)
    set(GCC_C_WARNINGS_DEFAULT -Wall)
  endif()

  if(DEFINED ENV{GCC_CXX_WARNINGS_DEFAULT})
    set(GCC_CXX_WARNINGS_DEFAULT $ENV{GCC_CXX_WARNINGS_DEFAULT})
  elseif(NOT DEFINED GCC_CXX_WARNINGS_DEFAULT)
    set(GCC_CXX_WARNINGS_DEFAULT -Wall)
  endif()

  if(DEFINED ENV{CUDA_WARNINGS_DEFAULT})
    set(CUDA_WARNINGS_DEFAULT $ENV{CUDA_WARNINGS_DEFAULT})
  elseif(NOT DEFINED CUDA_WARNINGS_DEFAULT)
    set(CUDA_WARNINGS_DEFAULT -Wall)
  endif()

  if(DEFINED ENV{KOKKOS_DEVICE_SPECIFIC_WARNINGS_DEFAULT})
    set(KOKKOS_DEVICE_SPECIFIC_WARNINGS_DEFAULT $ENV{KOKKOS_DEVICE_SPECIFIC_WARNINGS_DEFAULT})
  elseif(NOT DEFINED KOKKOS_DEVICE_SPECIFIC_WARNINGS_DEFAULT)
    set(KOKKOS_DEVICE_SPECIFIC_WARNINGS_DEFAULT)
  endif()

endmacro(kalypsso_set_default_warning_flags)

kalypsso_set_default_warning_flags()

#
# Setup developer warnings (full set)
#
# This function will do two things:
#
# * warning flags setup: either use default warnings flags, or use the full set of developer warning
#   flags
# * define a library target name "kalypsso_warning" that every final target to link with in order to
#   transitively receive the compilation flags; that way each time the flag set changes, it will
#   trigger re-compilation of every depending targets.
#
function(
  kalypsso_set_project_warnings
  project_name
  WARNINGS_AS_ERRORS
  CLANG_C_WARNINGS
  CLANG_CXX_WARNINGS
  GCC_C_WARNINGS
  GCC_CXX_WARNINGS
  CUDA_WARNINGS
  KOKKOS_DEVICE_SPECIFIC_WARNINGS)

  if("${CLANG_C_WARNINGS}" STREQUAL "")
    set(CLANG_C_WARNINGS
        -Wall
        -Wextra # reasonable and standard
        -Wshadow # warn the user if a variable declaration shadows one from a parent context
        # catch hard to track down memory errors
        -Wcast-align # warn for potential performance problem casts
        -Wunused # warn on anything being unused
        -Wpedantic # warn if non-standard C++ is used
        -Wconversion # warn on type conversions that may lose data
        -Wsign-conversion # warn on type conversions that may lose data
        -Wnull-dereference # warn if a null dereference is detected
        -Wdouble-promotion # warn if float is implicit promoted to double
        -Wformat=2 # warn on security issues around functions that format output (ie printf)
        -Wimplicit-fallthrough # warn on statements that fallthrough without an explicit annotation
    )
    if(KALYPSSO_CORE_DISABLE_DEPRECATED_WARNING)
      set(CLANG_C_WARNINGS ${CLANG_C_WARNINGS} -Wno-deprecated-declarations # do not warn about
                                                                            # deprecated features
      )
    endif()
    if(DEFINED KALYPSSO_CORE_KOKKOS_BACKEND)
      if(KALYPSSO_CORE_KOKKOS_BACKEND MATCHES "Cuda")
        # don't do anything, -Wold-style-cast generates too many false positive warnings
      endif()
    endif()
  else()
    set(CLANG_C_WARNINGS ${${CLANG_C_WARNINGS}})
  endif()
  message(STATUS "Clang C warnings flags are: ${CLANG_C_WARNINGS}")

  if("${CLANG_CXX_WARNINGS}" STREQUAL "")
    set(CLANG_CXX_WARNINGS
        ${CLANG_C_WARNING}
        -Wnon-virtual-dtor # warn the user if a class with virtual functions has a non-virtual
                           # destructor. This helps
        # catch hard to track down memory errors
        -Woverloaded-virtual # warn if you overload (not override) a virtual function
    )
  else()
    set(CLANG_CXX_WARNINGS ${${CLANG_CXX_WARNINGS}})
  endif()
  message(STATUS "Clang CXX warnings flags are: ${CLANG_CXX_WARNINGS}")

  if("${GCC_C_WARNINGS}" STREQUAL "")
    set(GCC_C_WARNINGS
        ${CLANG_C_WARNINGS}
        -Wmisleading-indentation # warn if indentation implies blocks where blocks do not exist
        -Wduplicated-cond # warn if if / else chain has duplicated conditions
        -Wduplicated-branches # warn if if / else branches have duplicated code
        -Wlogical-op # warn about logical operations being used where bitwise were probably wanted
    )
  else()
    set(GCC_C_WARNINGS ${${GCC_C_WARNINGS}})
  endif()
  message(STATUS "Gcc C warnings flags are: ${GCC_C_WARNINGS}")

  if("${GCC_CXX_WARNINGS}" STREQUAL "")
    set(GCC_CXX_WARNINGS ${GCC_C_WARNINGS} -Wuseless-cast # warn if you perform a cast to the same
                                                          # type
    )
  else()
    set(GCC_CXX_WARNINGS ${${GCC_CXX_WARNINGS}})
  endif()
  if(DEFINED KALYPSSO_CORE_KOKKOS_BACKEND)
    if(KALYPSSO_CORE_KOKKOS_BACKEND MATCHES "Cuda")
      # don't do anything about -Wold-style-cast as it generates too many false positive warnings
      # suppress warning 20208 : 'long double' is treated as 'double' in device code
      set(KOKKOS_DEVICE_SPECIFIC_WARNINGS ${KOKKOS_DEVICE_SPECIFIC_WARNINGS} -Xcudafe
                                          --diag_suppress=20208)

      if(KALYPSSO_CORE_CUDA_LINEINFO)
        set(KOKKOS_DEVICE_SPECIFIC_WARNINGS ${KOKKOS_DEVICE_SPECIFIC_WARNINGS} -lineinfo)
      endif(KALYPSSO_CORE_CUDA_LINEINFO)

      if(KALYPSSO_CORE_CUDA_RES_USAGE)
        set(KOKKOS_DEVICE_SPECIFIC_WARNINGS ${KOKKOS_DEVICE_SPECIFIC_WARNINGS} -res-usage)
      endif(KALYPSSO_CORE_CUDA_RES_USAGE)

      if(KALYPSSO_CORE_CUDA_PTXAS_VERBOSE)
        set(KOKKOS_DEVICE_SPECIFIC_WARNINGS ${KOKKOS_DEVICE_SPECIFIC_WARNINGS} -Xptxas=-v)
      endif(KALYPSSO_CORE_CUDA_PTXAS_VERBOSE)

    else()
      set(GCC_CXX_WARNINGS ${GCC_C_WARNINGS} -Wold-style-cast # warn for c-style casts
      )
    endif()
  endif()
  message(STATUS "Gcc CXX warnings flags are: ${GCC_CXX_WARNINGS}")

  if("${CUDA_WARNINGS}" STREQUAL "")
    set(CUDA_WARNINGS
        -Wall
        -Wextra
        -Wunused
        -Wconversion
        -Wshadow
        -Xcudafe
        --diag_suppress=20208
        # TODO add more Cuda warnings
    )
  else()
    set(CUDA_WARNINGS ${${CUDA_WARNINGS}})
  endif()
  message(STATUS "Cuda warnings flags are: ${CUDA_WARNINGS}")

  if(WARNINGS_AS_ERRORS)
    message(TRACE "Warnings are treated as errors")
    list(APPEND CLANG_C_WARNINGS -Werror)
    list(APPEND CLANG_CXX_WARNINGS -Werror)
    list(APPEND GCC_C_WARNINGS -Werror)
    list(APPEND GCC_CXX_WARNINGS -Werror)
    list(APPEND MSVC_WARNINGS /WX)
  endif()

  if(MSVC)
    set(PROJECT_WARNINGS_C ${MSVC_WARNINGS})
    set(PROJECT_WARNINGS_CXX ${MSVC_WARNINGS})
    set(PROJECT_WARNINGS_CXX_HOST ${MSVC_WARNINGS})
  elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    set(PROJECT_WARNINGS_C ${CLANG_C_WARNINGS})
    set(PROJECT_WARNINGS_CXX ${CLANG_CXX_WARNINGS})
    set(PROJECT_WARNINGS_CXX_HOST ${CLANG_CXX_WARNINGS})
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(PROJECT_WARNINGS_C ${GCC_C_WARNINGS})
    set(PROJECT_WARNINGS_CXX ${GCC_CXX_WARNINGS} ${KOKKOS_DEVICE_SPECIFIC_WARNINGS})
    set(PROJECT_WARNINGS_CXX_HOST ${GCC_CXX_WARNINGS})
  else()
    message(AUTHOR_WARNING "No compiler warnings set for CXX compiler: '${CMAKE_CXX_COMPILER_ID}'")
    # TODO support Intel compiler
  endif()

  set(PROJECT_WARNINGS_CUDA "${CUDA_WARNINGS}")

  target_compile_options(
    ${project_name}_host_only
    INTERFACE # C++ warnings
              $<$<COMPILE_LANGUAGE:CXX>:${PROJECT_WARNINGS_CXX_HOST}>
              # C warnings
              $<$<COMPILE_LANGUAGE:C>:${PROJECT_WARNINGS_C}>
              # Cuda warnings
              $<$<COMPILE_LANGUAGE:CUDA>:${PROJECT_WARNINGS_CUDA}>)
  target_compile_options(
    ${project_name}
    INTERFACE # C++ warnings
              $<$<COMPILE_LANGUAGE:CXX>:${PROJECT_WARNINGS_CXX}>
              # C warnings
              $<$<COMPILE_LANGUAGE:C>:${PROJECT_WARNINGS_C}>
              # Cuda warnings
              $<$<COMPILE_LANGUAGE:CUDA>:${PROJECT_WARNINGS_CUDA}>)
endfunction()
