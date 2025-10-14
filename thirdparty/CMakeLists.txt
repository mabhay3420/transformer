cmake_minimum_required(VERSION 3.25)

# ---- mlx library ----

set(MLX_BUILD_TESTS OFF CACHE BOOL "Disable MLX tests" FORCE)
set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "Disable MLX examples" FORCE)
set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "Disable MLX benchmarks" FORCE)
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Disable MLX python bindings" FORCE)
set(MLX_BUILD_CUDA OFF CACHE BOOL "Disable MLX CUDA backend" FORCE)
set(MLX_BUILD_CPU ON CACHE BOOL "Enable MLX CPU backend" FORCE)
set(MLX_BUILD_METAL ON CACHE BOOL "Enable MLX Metal backend" FORCE)

add_subdirectory(mlx)

if(TARGET mlx)
  target_compile_options(mlx PRIVATE "-U__ARM_FEATURE_BF16")
endif()

# ----------------------
