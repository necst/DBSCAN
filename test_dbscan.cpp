#include <fstream>
#include <map>
#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/timer/timer.hpp>

#include "dbscan.h"

using namespace clustering;

int main(int argc, char **argv)
{
	//ClusterData::Data cl_d = DBSCAN::gen_cluster_data(1, 1000);

    boost::timer::cpu_timer t;

    std::ifstream ifs;
    ifs.open (argv[1], std::ifstream::in);

	ClusterData cl_d = DBSCAN::read_cluster_data(ifs, 1, atoi(argv[3]));

    DBSCAN::Distance threshold = boost::lexical_cast<DBSCAN::Distance>(argv[2]);
    uint32_t threads = 3;
    uint32_t min_elements = 2;

    //std::cerr << "threshold " << threshold << std::endl;
    //std::cerr << "min_elements " << min_elements << std::endl;
    //std::cerr << "threads " << threads << std::endl;

	DBSCAN dbs(threshold, min_elements, threads);

    t.start();
	dbs.fit( cl_d.m );
    t.stop();
    std::cout << t.format(16, "cl_time %w") << std::endl;

    std::map<ClusterData::Klass, uint32_t> size;

    dbs.prepare_cluster_index();

    for (const auto & c : dbs.get_cluster_index() )
    {
        std::cout << c.first << ": ";;

        for (const auto & e : c.second ) {
            std::cout << e << " ";
        }

        std::cout << std::endl;
    }

    //average intra and inter cluster distance
    DBSCAN::ClusterIndex idx = dbs.get_cluster_index();
    for (DBSCAN::ClusterIndex::iterator idxit = idx.begin(); idxit != idx.end(); ++idxit)
    {
        ClusterData::Klass klass = (*idxit).first;
        std::set<uint32_t> elements = (*idxit).second;

        double avg_intra = 0.0;
        double couples = 0.0;

        std::cout << "avg_intra " << klass << " " << elements.size() << " ";
        for (std::set<uint32_t>::iterator it = elements.begin(); it != elements.end(); ++it)
        {
            size_t i = *it; //i-th element

            for (std::set<uint32_t>::iterator jt = elements.begin(); jt != elements.end(); ++jt)
            {
                size_t j = *jt; //j-th element

                if (i != j) //if they form a couple
                {
                    ++couples;

                    ublas::matrix_row<ClusterData::Data> U (cl_d.m, i);
                    ublas::matrix_row<ClusterData::Data> V (cl_d.m, j);

                    //distance computation between two elements (i.e., rows)
                    for (const auto e : DBSCAN::distance(U, V))
                    {
                        avg_intra += fabs(e);
                    }
                }
            }
        }
        avg_intra /= couples;

        std::cout << avg_intra << std::endl;
    }

    std::cout << "clusters " << idx.size() << std::endl;

    //cluster 1
    for (DBSCAN::ClusterIndex::iterator idxit = idx.begin(); idxit != idx.end(); ++idxit)
    {
        std::cout << "cluster_size " << (*idxit).first << " " << (*idxit).second.size() << std::endl;

        //skip noise
        //if ((*idxit).first != -1)
        {
            //cluster 2
            for (DBSCAN::ClusterIndex::iterator idxjt = idx.begin(); idxjt != idx.end(); ++idxjt)
            {
                //if they make a couple
                if ((*idxit).first != (*idxjt).first)
                {
                    double avg_inter = 0.0;
                    double couples = 0.0;

                    std::set<uint32_t> e1 = (*idxit).second;
                    std::set<uint32_t> e2 = (*idxjt).second;

                    //elements in the first
                    for (std::set<uint32_t>::iterator it = e1.begin(); it != e1.end(); ++it)
                    {
                        size_t i = *it;

                        //elements in the second
                        for (std::set<uint32_t>::iterator jt = e2.begin(); jt != e2.end(); ++jt)
                        {
                            size_t j = *jt;

                            ++couples;

                            ublas::matrix_row<ClusterData::Data> U (cl_d.m, i);
                            ublas::matrix_row<ClusterData::Data> V (cl_d.m, j);

                            //distance computation between two elements (i.e., rows)
                            for (const auto e : DBSCAN::distance(U, V))
                            {
                                avg_inter += fabs(e);
                            }
                        }
                    }

                    avg_inter /= couples;

                    std::cout << "avg_inter " << (*idxit).first << "-" << (*idxjt).first << " " << avg_inter << std::endl;
                }
            }
        }
    }

    DBSCAN::Labels l = dbs.get_labels();
    for (size_t i = 0; i < l.size(); ++i)
    {
        std::cout << "cluster " << cl_d.m_ids[i] << " " << l[i] << std::endl;
        if (size.count(l[i]) > 0)
            ++size[l[i]];
        else
            size[l[i]] = 0;
    }

    for (std::map<ClusterData::Klass, uint32_t>::iterator it = size.begin(); it != size.end(); ++it)
        std::cout << it->second << std::endl;

	return 0;
}
