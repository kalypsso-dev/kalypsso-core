# this macro is only meant to be used with add_custom_command, to check if a given file name exist
# at build time (not configure time)
#
# see
# https://stackoverflow.com/questions/18785168/cmake-check-if-file-exists-at-build-time-rather-than-during-cmake-configuration

string(ASCII 27 Esc)
set(COLOUR_RESET "${Esc}[m")
set(RED "${Esc}[31m")
set(GREEN "${Esc}[32m")

if(EXISTS ${FileToCheck})
  message("${GREEN}${FileToCheck} exists.${COLOUR_RESET}")
else()
  message("${RED}${FileToCheck} doesn't exist.${COLOUR_RESET}")
endif()
