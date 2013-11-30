#include "dbscan.h"

using namespace clustering;

int main()
{
	DBSCAN::ClusterData cl_d = DBSCAN::gen_cluster_data( 225, 3000 );

	DBSCAN dbs(3.0, 5);

	dbs.fit( cl_d );

	std::cout << dbs << std::endl;

	return 0;
}