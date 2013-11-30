SET(CMAKE_C_FLAGS "-std=gnu99 -pedantic -fPIC -pedantic -fopenmp")
SET(CMAKE_CXX_FLAGS "-std=c++0x -Wall -fPIC -pedantic -fopenmp")

add_definitions(-DBOOST_UBLAS_NDEBUG)

add_library(dbscan dbscan.cpp)

add_executable(test_dbscan test_dbscan.cpp)
target_link_libraries(test_dbscan dbscan)