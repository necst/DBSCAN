SET(CMAKE_C_FLAGS "-std=gnu99 -pedantic -fPIC -pedantic -fopenmp -mfpmath=sse")
SET(CMAKE_CXX_FLAGS "-std=c++0x -Wall -fPIC -pedantic -fopenmp -mfpmath=sse")

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON) 
set(Boost_USE_STATIC_RUNTIME OFF)

FIND_PACKAGE(PythonInterp)
FIND_PACKAGE(PythonLibs)
find_package(Boost REQUIRED COMPONENTS python)

add_definitions(-DBOOST_UBLAS_NDEBUG)

add_library(dbscan dbscan.cpp)

add_executable(test_dbscan test_dbscan.cpp)
target_link_libraries(test_dbscan dbscan)

if (PYTHONLIBS_FOUND AND Boost_FOUND)	
	include_directories( 
		${PYTHON_INCLUDE_PATH} 
	)
	         
	ADD_LIBRARY(pydbscan SHARED PyDBSCAN.cpp)
	target_link_libraries(pydbscan dbscan boost_python)

	SET_TARGET_PROPERTIES(pydbscan
		PROPERTIES
		    MAJOR_VERSION 1
		    MINOR_VERSION 0
		        LIBRARY_OUTPUT_NAME pydbscan
		        PREFIX ""
	)

endif(PYTHONLIBS_FOUND AND Boost_FOUND)
