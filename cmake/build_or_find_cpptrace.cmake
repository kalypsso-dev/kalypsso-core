# check if user requested a build of kalypsso-core with cpptrace enabled, so make it available
#
# Notes on important cpptrace options
#
# * set CPPTRACE_USE_EXTERNAL_LIBDWARF to ON if you want to use libdwarf from your system (by
#   default cpptrace will download libdwarf sources)
# * set CPPTRACE_USE_EXTERNAL_ZSTD to ON if you want to use libzstd from your system (by
#   default cpptrace will download zstd sources)
# * set CPPTRACE_UNWIND_WITH_UNWIND to ON is recommended for stack trace print
#
# About libunwind: on ubuntu 24.04 there are two packages providing libunwind:
#
# - libunwind-dev which packages https://github.com/libunwind/libunwind
# - libunwind-18-dev provided by llvm
#
# You should prefer libunwind-dev over libunwind-18-dev because the first one contains a pkg-config file
# and while the second does'nt.
#
include(cmake/cmake_utils.cmake)

if(KALYPSSO_CORE_USE_CPPTRACE)

  message("[kalypsso-core / cpptrace] enabled")

  include(FetchContent)

  if(DEFINED CPPTRACE_UNWIND_WITH_UNWIND)
    if(NOT CPPTRACE_UNWIND_WITH_UNWIND)
      message(STATUS "CPPTRACE_UNWIND_WITH_UNWIND is OFF; it is recommended to set it ON for ")
    endif()
  endif()

  FetchContent_Declare(cpptrace_external SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/cpptrace SYSTEM)

  # libunwind is necessary to print stack

  fetchcontent_makeavailablewithargs(cpptrace_external "CPPTRACE_UNWIND_WITH_LIBUNWIND=ON"
                                     "CPPTRACE_UNWIND_WITH_UNWIND=ON")

endif(KALYPSSO_CORE_USE_CPPTRACE)
