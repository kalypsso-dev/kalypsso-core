# gnu compatibility : https://cmake.org/cmake/help/latest/module/GNUInstallDirs.html
include(GNUInstallDirs)

# some important variables
set(INSTALL_BINDIR
    ${CMAKE_INSTALL_BINDIR}
    CACHE STRING "Installation directory for binaries, relative to ${CMAKE_INSTALL_PREFIX}.")

set(INSTALL_LIBDIR
    ${CMAKE_INSTALL_LIBDIR}
    CACHE STRING "Installation directory for libraries, relative to ${CMAKE_INSTALL_PREFIX}.")

set(INSTALL_INCLUDEDIR
    ${CMAKE_INSTALL_INCLUDEDIR}
    CACHE STRING "Installation directory for include files, relative to ${CMAKE_INSTALL_PREFIX}.")

set(INSTALL_PKGCONFIG_DIR
    ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    CACHE PATH
          "Installation directory for pkgconfig (.pc) files, relative to ${CMAKE_INSTALL_PREFIX}.")

set(INSTALL_CMAKE_DIR
    ${CMAKE_INSTALL_LIBDIR}/cmake
    CACHE STRING "Installation directory for cmake files, relative to ${CMAKE_INSTALL_PREFIX}.")
