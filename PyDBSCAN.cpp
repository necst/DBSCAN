#include <boost/python.hpp>
#include "dbscan.h"

using namespace boost::python;
using namespace clustering;

struct ublas_matrix_to_python
{
	static PyObject* convert( DBSCAN::ClusterData const& C )
	{
		PyObject * result = PyList_New( C.size1() );

		for (size_t i = 0; i < C.size1(); ++i)
		{
			PyObject * l = PyList_New( C.size2() );

			for (size_t j = 0; j < C.size2(); ++j)
			{	
				PyList_SetItem( l, j, PyFloat_FromDouble( C(i, j) ) );
			}

			PyList_SetItem( result, i, l );
		}

		return result;
	}
};

class PyDBSCAN : public DBSCAN
{
public:
	PyDBSCAN() : DBSCAN() {}

	void pyfit( boost::python::list& pylist )
	{
		auto num_samples = boost::python::len( pylist );

		if ( num_samples == 0 )
		{
			return;
		}

		boost::python::extract< boost::python::list > first_elem( pylist[ 0 ] );

		auto num_features = boost::python::len( first_elem );

		DBSCAN::ClusterData C( num_samples, num_features );

		for ( int i = 0; i < num_samples; ++i )
        {
        	boost::python::list sublist = boost::python::extract< boost::python::list >( pylist[ i ] );

        	for ( int j = 0; j < num_features; ++j )
        	{
        		C(i, j) = boost::python::extract<double>( sublist[ j ] );	
        	}
        }

        fit( C );
	}

	void pywfit( boost::python::list& pylist, boost::python::list& pyweights )
	{
		auto num_samples = boost::python::len( pylist );

		if ( num_samples == 0 )
		{
			return;
		}

		boost::python::extract< boost::python::list > first_elem( pylist[ 0 ] );

		auto num_features = boost::python::len( first_elem );

		DBSCAN::ClusterData C( num_samples, num_features );

		for ( int i = 0; i < num_samples; ++i )
        {
        	boost::python::list sublist = boost::python::extract< boost::python::list >( pylist[ i ] );

        	for ( int j = 0; j < num_features; ++j )
        	{
        		C(i, j) = boost::python::extract<double>( sublist[ j ] );	
        	}
        }

        auto num_weights = boost::python::len( pyweights );

        DBSCAN::FeaturesWeights W( num_weights );

        for ( int i = 0; i < num_weights; ++i )
        {
        	W(i) = boost::python::extract<double>( pyweights[ i ] );
        }

     	wfit( C, W );   
	}

	boost::python::list pyget_labels()
	{
		boost::python::list list;

		for (const auto & l : get_labels() )
		{
			list.append(l);
		}

		return list;
	}
};

BOOST_PYTHON_MODULE(pydbscan)
{
    def( "gen_cluster_data", & PyDBSCAN::gen_cluster_data    );
    //def( "send_stat", & python_send_stat );

    class_< PyDBSCAN >( "DBSCAN" )
    	.def( "init"             	, & PyDBSCAN::init 			)
        .def( "fit"             	, & PyDBSCAN::pyfit  		)
        .def( "wfit"             	, & PyDBSCAN::pywfit  		)
        .def( "get_labels"        	, & PyDBSCAN::pyget_labels  )
    ;

    to_python_converter<DBSCAN::ClusterData, ublas_matrix_to_python, false>();
}
