add_executable(test_p4est_wrapper "")
target_sources(test_p4est_wrapper PUBLIC test_p4est_wrapper.cpp)
target_link_libraries(test_p4est_wrapper PUBLIC kalypsso::p4est_wrap kalypsso::config
                                                kalypsso::core_config)

file(GLOB files_to_remove ${CMAKE_CURRENT_BINARY_DIR}/*.vtu ${CMAKE_CURRENT_BINARY_DIR}/*.pvtu
     ${CMAKE_CURRENT_BINARY_DIR}/*.visit)
set_property(
  DIRECTORY
  APPEND
  PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${files_to_remove})

add_test(NAME p4est_wrapper_brick COMMAND test_p4est_wrapper brick ${KALYPSSO_TEST_AMR_MAX_LEVEL} 3
                                          3)
set_property(TEST p4est_wrapper_brick PROPERTY LABELS p4est_wrapper 2D)

add_test(NAME p4est_wrapper_brick3 COMMAND test_p4est_wrapper brick3 ${KALYPSSO_TEST_AMR_MAX_LEVEL}
                                           4 3 2)
set_property(TEST p4est_wrapper_brick3 PROPERTY LABELS p4est_wrapper 3D)
