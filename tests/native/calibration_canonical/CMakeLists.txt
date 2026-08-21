cmake_minimum_required(VERSION 3.16)
project(simllm_calibration_canonical_verify LANGUAGES CXX)

add_executable(calibration_canonical_verify canonical_verify.cpp)
target_compile_features(calibration_canonical_verify PRIVATE cxx_std_17)
set_target_properties(
    calibration_canonical_verify
    PROPERTIES
    CXX_EXTENSIONS OFF
)

if(MSVC)
    target_compile_options(calibration_canonical_verify PRIVATE /W4 /WX /permissive-)
else()
    target_compile_options(
        calibration_canonical_verify
        PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
    )
endif()
