cmake_minimum_required(VERSION 3.25)

# ---- json library ----

# dont run tests
set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(json)

# ----------------------