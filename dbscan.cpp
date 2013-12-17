#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/algorithm/minmax.hpp>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>
#include <map>
#include <algorithm>    // std::unique, std::distance
#include <omp.h>

#include "dbscan.h"

namespace clustering
{
    void ClusterData::init(size_t elements_num, size_t features_num)
    {
        m_elements_num = elements_num;
        m_features_num = features_num;
    }

    ClusterData::ClusterData(size_t elements_num, size_t features_num)
        : m_elements_num( elements_num )
          , m_features_num( features_num )
          , m( elements_num, features_num )
    {
        reset();
    }

    void ClusterData::reset()
    {
        m_ids.clear();
        m_cl.clear();
        m.clear();
    }

    size_t ClusterData::getElementsNum()
    {
        return m_elements_num;
    }

    size_t ClusterData::getFeaturesNum()
    {
        return m_features_num;
    }

    ClusterData::~ClusterData()
    {

    }

    //------------------------------------------------------------------

    void DBSCAN::init(double eps, size_t min_elems, int num_threads)
    {
        m_eps = eps;
        m_min_elems = min_elems;
        m_num_threads = num_threads;
    }

    DBSCAN::DBSCAN(double eps, size_t min_elems, int num_threads)
        : m_eps( eps )
          , m_min_elems( min_elems )
          , m_num_threads( num_threads )
          , m_dmin(0.0)
          , m_dmax(0.0)
    {
        reset();
    }

    DBSCAN::~DBSCAN()
    {

    }

    ClusterData::Data DBSCAN::gen_cluster_data( size_t features_num, size_t elements_num )
    {
        ClusterData::Data cl_d( elements_num, features_num );
        ulong64 random;

        //elements -> by row
        for (size_t i = 0; i < elements_num; ++i)
        {
            //features -> by column
            for (size_t j = 0; j < features_num; ++j)
            {
                random =
                    (((ulong64) rand() <<  0) & 0x000000000000FFFFull) |
                    (((ulong64) rand() << 16) & 0x00000000FFFF0000ull) |
                    (((ulong64) rand() << 32) & 0x0000FFFF00000000ull) |
                    (((ulong64) rand() << 48) & 0xFFFF000000000000ull);
                cl_d(i, j) = random;
            }
        }

        return cl_d;
    }

    ClusterData DBSCAN::read_cluster_data(std::istream & is, size_t features_num, size_t elements_num )
    {
        /**
         * Read data from stream.
         *
         * N = features_num
         * M = elements_num
         *
         * Expected format is:
         *
         * ClassElement1 stringIdElement1 ulong64feature1 ulong64feature2 ... ulong64featureN
         * ClassElement2 stringidelement2 ulong64feature1 ulong64feature2 ... ulong64featureN
         * ClassElement3 stringidelement3 ulong64feature1 ulong64feature2 ... ulong64featureN
         *       .
         *       .
         *       .
         * ClassElementM stringidelement3 ulong64feature1 ulong64feature2 ... ulong64featureN
         */
        ClusterData cl_d( elements_num, features_num );

        std::vector<std::string> vec;
        std::string line;
        typedef boost::tokenizer<boost::escaped_list_separator<char> > Tokenizer;
        Tokenizer::iterator it;

        for (size_t i = 0; std::getline(is, line) && i < elements_num; ++i)
        {
            size_t j = 0;
            Tokenizer tok(line);

            it = tok.begin();
            cl_d.m_cl.push_back(boost::lexical_cast<ClusterData::Klass>(*it));
            ++it;

            cl_d.m_ids.push_back(boost::lexical_cast<ClusterData::Id>(*it));
            ++it;

            for (; j < features_num && it != tok.end(); ++j)
            {
                cl_d.m(i, j) = boost::lexical_cast<ulong64>(*it);
            }
        }

        return cl_d;
    }

    DBSCAN::FeaturesWeights DBSCAN::std_weights( size_t s )
    {
        // num cols
        DBSCAN::FeaturesWeights ws( s );

        for (size_t i = 0; i < s; ++i)
        {
            ws(i) = 1.0;
        }

        return ws;
    }

    void DBSCAN::reset()
    {
        m_labels.clear();
        m_cluster_index.clear();
    }

    void DBSCAN::prepare_labels( size_t s )
    {
        m_labels.resize(s);

        for( auto & l : m_labels)
        {
            l = -1; //noise by default
        }
    }

    ublas::vector<DBSCAN::Distance> DBSCAN::distance(
            ublas::matrix_row<ClusterData::Data> & rowA,
            ublas::matrix_row<ClusterData::Data> & rowB)
    {
        ublas::vector<DBSCAN::Distance> d(rowA.size());

        for (size_t i = 0; i < rowA.size(); ++i)
        {
            //Hamming
            d(i) = (DBSCAN::Distance)ph_hamming_distance((ulong64)rowA(i), (ulong64)rowB(i));

            //Euclidean
            //d(i) = (DBSCAN::Distance)(rowA(i)-rowB(i));
        }

        return d;
    }

    const DBSCAN::DistanceMatrix DBSCAN::calc_dist_matrix(
            const ClusterData::Data & C, const DBSCAN::FeaturesWeights & W, bool normalize)
    {
        ClusterData::Data cl_d = C;

        // by-column normalization
        if (normalize)
        {
            omp_set_dynamic(0);
            omp_set_num_threads( m_num_threads );

#pragma omp parallel for
            for (size_t i = 0; i < cl_d.size2(); ++i)
            {
                ublas::matrix_column<ClusterData::Data> col(cl_d, i);

                const auto r = minmax_element( col.begin(), col.end() );

                double data_min = *r.first;
                double data_range = *r.second - *r.first;

                if (data_range == 0.0) {
                    data_range = 1.0;
                }

                const double scale = 1/data_range;
                const double min = -1.0 * data_min * scale;

                col *= scale;
                col.plus_assign(
                        ublas::scalar_vector< typename ublas::matrix_column<ClusterData::Data>::value_type >(col.size(), min) );
            }
        }

        // rows x rows
        DBSCAN::DistanceMatrix d_m( cl_d.size1(), cl_d.size1() );
        ublas::vector<double> d_max( cl_d.size1() );
        ublas::vector<double> d_min( cl_d.size1() );

        omp_set_dynamic(0);
        omp_set_num_threads( m_num_threads );
#pragma omp parallel for
        for (size_t i = 0; i < cl_d.size1(); ++i)
        {
            for (size_t j = i; j < cl_d.size1(); ++j)
            {
                d_m(i, j) = 0.0;

                if (i != j)
                {
                    ublas::matrix_row<ClusterData::Data> U (cl_d, i);
                    ublas::matrix_row<ClusterData::Data> V (cl_d, j);

                    //distance computation between two elements (i.e., rows)
                    int k = 0;
                    for (const auto e : distance(U, V))
                    {
                        d_m(i, j) += fabs(e)*W[k++];
                    }

                    d_m(j, i) = d_m(i, j);
                }
            }

            const auto cur_row = ublas::matrix_row<DBSCAN::DistanceMatrix>(d_m, i);
            const auto mm = minmax_element( cur_row.begin(), cur_row.end() );

            d_max(i) = *mm.second;
            d_min(i) = *mm.first;
        }

        if (normalize)
        {
            m_dmin = *(min_element( d_min.begin(), d_min.end() ));
            m_dmax = *(max_element( d_max.begin(), d_max.end() ));

            m_eps = (m_dmax - m_dmin) * m_eps + m_dmin;
        }

        return d_m;
    }

    DBSCAN::Neighbors DBSCAN::find_neighbors(const DBSCAN::DistanceMatrix & D, uint32_t pid)
    {
        Neighbors ne;
        DBSCAN::Distance d;

        for (uint32_t j = 0; j < D.size1(); ++j)
        {
            d = D(pid, j);

            if 	( d <= m_eps )
            {
                ne.push_back(j);
            }
        }

        return ne;
    }

    void DBSCAN::dbscan( const DBSCAN::DistanceMatrix & dm )
    {
        std::vector<uint8_t> visited( dm.size1() );

        uint32_t cluster_id = 0;

        //for each element
        for (uint32_t pid = 0; pid < dm.size1(); ++pid)
        {
            //only if not visited yet
            if ( !visited[pid] )
            {
                visited[pid] = 1;

                //look for neighbors
                Neighbors ne = find_neighbors(dm, pid );

                //if enough neighbors
                if (ne.size() >= m_min_elems)
                {
                    //assign that element to the current cluster
                    m_labels[pid] = cluster_id;

                    //for each element in the neighborhood
                    for (uint32_t i = 0; i < ne.size(); ++i)
                    {
                        uint32_t nPid = ne[i];

                        //if not visited yet
                        if ( !visited[nPid] )
                        {
                            //mark it as visited
                            visited[nPid] = 1;

                            //find its neighbors
                            Neighbors ne1 = find_neighbors(dm, nPid);

                            //if enough neighbors
                            if ( ne1.size() >= m_min_elems )
                            {
                                //expand neighbors
                                for (const auto & n1 : ne1)
                                {
                                    ne.push_back(n1);
                                }
                            }
                        }

                        //if the last element in the neighborhood was noise
                        // mark it as non-noise anymore and include it to the
                        // current cluster
                        if ( m_labels[nPid] == -1 )
                        {
                            m_labels[nPid] = cluster_id;
                        }
                    }

                    ++cluster_id; //create new cluster
                }
            }
        }
    }

    void DBSCAN::fit( const ClusterData::Data & C, bool normalize )
    {
        const DBSCAN::FeaturesWeights W = DBSCAN::std_weights( C.size2() );
        wfit( C, W, normalize );
    }
    void DBSCAN::fit_precomputed( const DBSCAN::DistanceMatrix & D )
    {
        prepare_labels( D.size1() );
        dbscan( D );
    }

    void DBSCAN::wfit( const ClusterData::Data & C, const DBSCAN::FeaturesWeights & W, bool normalize )
    {
        prepare_labels( C.size1() );
        const DBSCAN::DistanceMatrix D = calc_dist_matrix( C, W, normalize );
        dbscan( D );
    }

    const DBSCAN::Labels & DBSCAN::get_labels() const
    {
        return m_labels;
    }

    void DBSCAN::prepare_cluster_index()
    {
        if (m_cluster_index.size() == 0) {
            //key->value (element_id -> label)
            for (size_t element = 0; element < m_labels.size(); ++element)
            {
                m_cluster_index[m_labels[element]].insert(element);
            }
        }
    }

    const DBSCAN::ClusterIndex & DBSCAN::get_cluster_index() const
    {
        return m_cluster_index;
    }

    std::ostream& operator<<(std::ostream& o, DBSCAN & d)
    {
        o << "[";
        for ( const auto & l : d.get_labels() )
        {
            o << " " << l;
        }
        o << "] " << std::endl;

        return o;
    }

    std::ostream& operator<<(std::ostream& o, ClusterData & cl_d)
    {
        for (size_t i = 0; i < cl_d.getElementsNum(); ++i)
        {
            o << cl_d.m_cl[i] << " " << cl_d.m_ids[i];

            for (size_t j = 0; j < cl_d.getFeaturesNum(); ++j)
                o << " " << cl_d.m(i, j);

            o << std::endl;
        }

        return o;
    }
}
