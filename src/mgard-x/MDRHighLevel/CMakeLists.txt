MgardXGenerateSourceAllCombinations("MDRHighLevel")
# MgardXGenerateSourceAllCombinations("MDRefactor")
# MgardXGenerateSourceAllCombinations("MDRecontruct")
# MgardXGenerateSourceAllCombinations("MDRequest")
set(MGARD_X_SERIAL_SRC ${MGARD_X_SERIAL_SRC} PARENT_SCOPE)
set(MGARD_X_OPENMP_SRC ${MGARD_X_OPENMP_SRC} PARENT_SCOPE)
set(MGARD_X_CUDA_SRC ${MGARD_X_CUDA_SRC} PARENT_SCOPE)
set(MGARD_X_HIP_SRC ${MGARD_X_HIP_SRC} PARENT_SCOPE)
set(MGARD_X_SYCL_SRC ${MGARD_X_SYCL_SRC} PARENT_SCOPE)

list(APPEND MGARD_X_SRC  
      ${CMAKE_CURRENT_SOURCE_DIR}/DynamicAPI.cpp)

set(MGARD_X_SRC ${MGARD_X_SRC} PARENT_SCOPE)
