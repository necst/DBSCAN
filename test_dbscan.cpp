#include "dbscan.h"

using namespace clustering;

int main()
{
	DBSCAN::ClusterData cl_d = DBSCAN::gen_cluster_data( 225, 10000 );

	DBSCAN dbs(0.1, 5, 1);

	dbs.fit( cl_d );

	//std::cout << dbs << std::endl;

	return 0;
}
