#include "dbscan.h"

using namespace clustering;

int main()
{
	DBSCAN::ClusterData cl_d = DBSCAN::gen_cluster_data(1, 1000);

	DBSCAN dbs(0.01, 2, 1);

	dbs.fit( cl_d );

	std::cout << dbs << std::endl;

	return 0;
}
