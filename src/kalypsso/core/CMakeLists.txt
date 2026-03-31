# ########################################################################################
# ##### KALYPSSO_CORE_CONFIG ######
# ########################################################################################
add_library(kalypsso_core_config INTERFACE)

# setup include path for including kalypsso_core_config.h
target_include_directories(
  kalypsso_core_config
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../..>
            $<INSTALL_INTERFACE:include/kalypsso/core>)

add_library(kalypsso::core_config ALIAS kalypsso_core_config)

# ##################################################################################################
# ##### KALYPSSO_CORE ######
# ##################################################################################################
add_library(
  kalypsso_core
  AlternatingDirectionScheduler.cpp
  AlternatingDirectionScheduler.h
  AMRContext.h
  AMRCycleDriver.h
  amr_hashmap.h
  amr_key.h
  AMRmesh.h
  AMRMeshInfo.h
  AMRMeshMonitoring.cpp
  AMRMeshMonitoring.h
  AMRmesh_utils.h
  BinomialCoef.h
  BitFieldHelper.h
  BitFieldInteger.h
  brick_base.h
  brick_connectivity_utils.h
  brick_utils.h
  CheckEdgeSiblingsConnectivity.cpp
  CheckEdgeSiblingsConnectivity.h
  CheckFaceBorderCompatibility.cpp
  CheckFaceBorderCompatibility.h
  cmdline_utils.h
  ComputeConformalFullStatus.h
  ComputeConformalStatus.h
  ComputeCurvature.cpp
  ComputeCurvature.h
  ComputeDivergence.cpp
  ComputeDivergence.h
  ComputeError.h
  ComputeFilteredCurvature.cpp
  ComputeFilteredCurvature.h
  ComputeGhostedScalarField.cpp
  ComputeGhostedScalarField.h
  ComputeInterfaceNormalVector.cpp
  ComputeInterfaceNormalVector.h
  ComputeRefineFlags.h
  ComputeRefineFlags.cpp
  ComputeRefineFlagsMM.h
  ComputeRefineFlagsMM.cpp
  ComputeRefineFlags_utils.h
  ComputeSchlieren.h
  ComputeSchlieren.cpp
  ComputeSmoothInterfaceFunction.cpp
  ComputeSmoothInterfaceFunction.h
  ComputeVolumeIntegralValue.h
  ComputeVolumeIntegralValue.cpp
  config_utils.h
  ConformalFaceStatus.h
  ConformalFullStatus.h
  ConformalNeighborStatus.h
  ConservativityCheck.cpp
  ConservativityCheck.h
  DataArray.h
  DataArrayBlock.h
  DataArrayBlockMultiVar.h
  DataArrayBlock_utils.h
  DataArrayGhostedBlock.h
  DataArrayGhostedBlock_old.h
  DataArrayUtils.cpp
  DataArrayUtils.h
  EdgeDataArrayBlock.h
  EdgeDataArrayBlock_utils.h
  enums.h
  eos/eos_utils.h
  eos/IdealGasEos.h
  eos/IdealGasMixture.h
  eos/MieGruneisenEosCochranChan.h
  eos/MieGruneisenEosIdealGas.h
  eos/MieGruneisenEosJWL.h
  eos/MieGruneisenEosStiffenedGas.h
  eos/MieGruneisenEosSW.h
  eos/MieGruneisenEosVanDerWaalsGas.h
  eos/StiffenedGasEos.h
  eos/StiffenedGasMixture.h
  eos/VanDerWaalsGasEos.h
  ExtractNonGhostedArray.h
  FaceDataArrayBlock.h
  FaceDataArrayBlock_utils.h
  FieldMap.h
  FillBlockGhosts_common.h
  FillBlockGhostCells.cpp
  FillBlockGhostCells.h
  FillBlockGhostCellsInPlace.cpp
  FillBlockGhostCellsInPlace.h
  FillBlockGhostCells_v2.cpp
  FillBlockGhostCells_v2.h
  FillBlockGhostFaces.cpp
  FillBlockGhostFaces.h
  FillBlockGhostFluxes.cpp
  FillBlockGhostFluxes.h
  FillBlockGhosts_common.h
  FillOutside_utils.h
  FiniteDifferenceData.h
  FirstOrderDerivativeFiniteDifference.cpp
  FirstOrderDerivativeFiniteDifference.h
  geometry_utils.h
  GravityField.h
  GravityField.cpp
  HDF5_IO_common.h
  HDF5_Xdmf_Writer.h
  HDF5_Xdmf_Writer_legacy.h
  HydroParams.cpp
  HydroParams.h
  init_func.h
  InterfaceNormalVectorAlgorithmParams.h
  io_utils.cpp
  io_utils.h
  kalypsso_comm_config.h
  kalypsso_core_base.h
  kalypsso_core_build_info.h
  kalypsso_core_build_info.cpp
  kalypsso_core_git_info.h
  kalypsso_core_git_info.cpp
  kalypsso_data_container.h
  kalypsso_macros.h
  Kokkos_Array_extensions.h
  Kokkos_extensions.h
  kokkos_shared.h
  LinearCombination.h
  Locations.h
  MaterialPresence.h
  MeshMap.h
  MeshMap.cpp
  MeshPartitioner.h
  MeshPartitioner_helper.h
  mesh_utils.h
  misc_utils.cpp
  misc_utils.h
  models/Hydro.cpp
  models/Hydro.h
  models/HydroSettings.cpp
  models/HydroSettings.h
  models/HydroState.h
  models/MHD.cpp
  models/MHD.h
  models/MHDSettings.cpp
  models/MHDSettings.h
  models/MHDState.h
  models/riemann_solver_types.h
  models/RiemannSolvers.h
  models/RiemannSolvers_MHD.h
  models/mhd_utils.h
  models/utils_hydro.h
  morton_utils.h
  multimaterial_utils.h
  MultiMatFillBlockGhostCells.h
  myKokkos_Sort.h
  orchard_key_base.h
  orchard_key.h
  orchard_key_impl_2d.h
  orchard_key_impl_3d.h
  orchard_key_utils.h
  OutputParams.cpp
  OutputParams.h
  OutsideQuadsInfo.h
  p4est_utils.h
  physical_constants.h
  Point.h
  problems/init_cond_utils.cpp
  problems/init_cond_utils.h
  problems/AlfvenParams.h
  problems/BlastParams.h
  problems/BreakingWaveParams.h
  problems/DiamagCavityParams.h
  problems/DoubleMachReflectionParams.h
  problems/DropletAdvectionParams.h
  problems/FieldLoopAdvectionParams.h
  problems/FourQuadrantParams.h
  problems/GreshoVortexParams.h
  problems/ImplodeParams.h
  problems/initRiemannConfig2d.cpp
  problems/initRiemannConfig2d.h
  problems/IsentropicVortexParams.h
  problems/KHParams.h
  problems/MHDShockTubeParams.h
  problems/OrszagTangParams.h
  problems/RayleighTaylorParams.h
  problems/ShuOsherParams.h
  problems/SodParams.h
  problems/StaticDropletParams.h
  problems/TriplePointParams.h
  problems/TwoFluidShockTubeParams.h
  problems/UnderwaterExplosionParams.h
  problems/WoodwardParams.h
  prolongation.h
  real_type.h
  region_utils.h
  scan_utils.h
  SimpleVTKIO.h
  SmoothInterfaceFunctionData.cpp
  SmoothInterfaceFunctionData.h
  SolverBase.cpp
  SolverBase.h
  StencilHelper.h
  StencilHelper.cpp
  THINCParams.h
  TimeIntegratorConfig.h
  UserDataRemapper.cpp
  UserDataRemapper.h
  UserDataRemapperImplBCC.cpp
  UserDataRemapperImplBCC.h
  UserDataRemapperImplBFC.cpp
  UserDataRemapperImplBFC.h
  UserDataRemapperImplMD.cpp
  UserDataRemapperImplMD.h
  UserDataRemapperImplMP.cpp
  UserDataRemapperImplMP.h
  utils_block.h
  vof/interface_tracking_utils.h
  vof/interface_tracking_utils.cpp
  vof/youngs.h)

if(KALYPSSO_CORE_USE_MPI)
  target_sources(
    kalypsso_core
    PRIVATE MaterialPresenceExchanger.cpp MaterialPresenceExchanger.h MeshGhostsExchanger.cpp
            MeshGhostsExchanger.h MeshGhostsExchangerMultiVar.cpp MeshGhostsExchangerMultiVar.h)
endif(KALYPSSO_CORE_USE_MPI)

if(KALYPSSO_CORE_USE_CPPTRACE)
  target_sources(kalypsso_core PRIVATE cpptrace_utils.h cpptrace_utils.cpp)
endif(KALYPSSO_CORE_USE_CPPTRACE)

if(KALYPSSO_CORE_USE_HDF5)
  target_sources(kalypsso_core PRIVATE HDF5_IO_common.cpp)
endif(KALYPSSO_CORE_USE_HDF5)

if(KALYPSSO_CORE_USE_CNPY)
  target_sources(kalypsso_core PRIVATE ComputeDataSliceAlongLine.cpp ComputeDataSliceAlongLine.h)

  target_sources(kalypsso_core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cnpy_io.h>
                                      $<INSTALL_INTERFACE:include/kalypsso/core/cnpy_io.h>)
endif(KALYPSSO_CORE_USE_CNPY)

target_link_libraries(kalypsso_core PRIVATE kalypsso_warnings)

target_link_libraries(
  kalypsso_core PUBLIC kalypsso_core_config better-enums::better-enums kalypsso::monitoring
                       kalypsso::log Kokkos::kokkos HighFive::HighFive)

if(KALYPSSO_CORE_USE_MPI)
  target_link_libraries(kalypsso_core PUBLIC kalypsso::p4est_wrap kalypsso::mpiUtils)
endif(KALYPSSO_CORE_USE_MPI)

if(KALYPSSO_CORE_USE_HDF5)
  target_link_libraries(kalypsso_core PUBLIC ${HDF5_LIBRARIES})
endif(KALYPSSO_CORE_USE_HDF5)

if(KALYPSSO_CORE_USE_CNPY)
  target_link_libraries(kalypsso_core PUBLIC cnpy::cnpy)
endif(KALYPSSO_CORE_USE_CNPY)

if(KALYPSSO_CORE_USE_CPPTRACE)
  target_link_libraries(kalypsso_core PUBLIC cpptrace::cpptrace)

  add_executable(kalypsso_stack_printer kalypsso_stack_printer.cpp)
  target_link_libraries(kalypsso_stack_printer PUBLIC cpptrace::cpptrace)
endif(KALYPSSO_CORE_USE_CPPTRACE)

# declare library alias
add_library(kalypsso::core ALIAS kalypsso_core)

#
# Get prepared for install
#
include(${PROJECT_SOURCE_DIR}/cmake/setup_install.cmake)

#
# install libraries and headers
#
install(
  TARGETS kalypsso_core kalypsso_core_config
  EXPORT kalypsso-targets
  LIBRARY DESTINATION ${INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${INSTALL_LIBDIR}
  RUNTIME DESTINATION ${INSTALL_BINDIR})

#
# install headers (exclude cpp files)
#
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DESTINATION ${INSTALL_INCLUDEDIR}/kalypsso
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.cpp" EXCLUDE
  PATTERN "*.in" EXCLUDE)
