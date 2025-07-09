list(APPEND MGARD_X_HEADER
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearQuantization.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/HybridHierarchyLinearQuantization.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LocalQuantization.hpp
    )
set(MGARD_X_HEADER ${MGARD_X_HEADER} PARENT_SCOPE)