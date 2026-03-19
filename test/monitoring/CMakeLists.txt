add_executable(test_profiling test_profiling.cpp)

target_link_libraries(test_profiling PUBLIC kalypsso::core kalypsso::config kalypsso::monitoring)

configure_file(test_config.ini test_config.ini COPYONLY)
