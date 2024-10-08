cmake_minimum_required(VERSION 3.13.0)
project(privacy_experiments)


# Build run executable
add_executable(laplace1d $<TARGET_OBJECTS:bayesmix>
    unidimensional_laplace/main.cc
    utils.h
    utils.cc
)

add_executable(wavelet $<TARGET_OBJECTS:bayesmix>
    wavelets/main.cc
    utils.h
    utils.cc
)

add_executable(compare $<TARGET_OBJECTS:bayesmix>
    compare/main.cc
    utils.h
    utils.cc
)

add_executable(gaussian $<TARGET_OBJECTS:bayesmix>
    gaussian/main.cc
    utils.h
    utils.cc
)

add_executable(global $<TARGET_OBJECTS:bayesmix>
    global/main.cc
    utils.h
    utils.cc
)


add_executable(blood $<TARGET_OBJECTS:bayesmix>
    blood_donors/main.cc
    utils.h
    utils.cc
)

add_executable(run_private $<TARGET_OBJECTS:bayesmix>
    run_from_file.cc
    utils.h
    utils.cc
)

target_include_directories(laplace1d PUBLIC ${INCLUDE_PATHS})
target_link_libraries(laplace1d PUBLIC ${LINK_LIBRARIES})
target_compile_options(laplace1d PUBLIC ${COMPILE_OPTIONS})

target_include_directories(wavelet PUBLIC ${INCLUDE_PATHS})
target_link_libraries(wavelet PUBLIC ${LINK_LIBRARIES})
target_compile_options(wavelet PUBLIC ${COMPILE_OPTIONS})

target_include_directories(compare PUBLIC ${INCLUDE_PATHS})
target_link_libraries(compare PUBLIC ${LINK_LIBRARIES})
target_compile_options(compare PUBLIC ${COMPILE_OPTIONS})

target_include_directories(gaussian PUBLIC ${INCLUDE_PATHS})
target_link_libraries(gaussian PUBLIC ${LINK_LIBRARIES})
target_compile_options(gaussian PUBLIC ${COMPILE_OPTIONS})

target_include_directories(global PUBLIC ${INCLUDE_PATHS})
target_link_libraries(global PUBLIC ${LINK_LIBRARIES})
target_compile_options(global PUBLIC ${COMPILE_OPTIONS})

target_include_directories(blood PUBLIC ${INCLUDE_PATHS})
target_link_libraries(blood PUBLIC ${LINK_LIBRARIES})
target_compile_options(blood PUBLIC ${COMPILE_OPTIONS})


target_include_directories(run_private PUBLIC ${INCLUDE_PATHS})
target_link_libraries(run_private PUBLIC ${LINK_LIBRARIES})
target_compile_options(run_private PUBLIC ${COMPILE_OPTIONS})