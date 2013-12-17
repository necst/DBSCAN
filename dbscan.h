#include <vector>
#include <string>
#include <istream>
#include <set>
#include <map>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include <pHash.h>

using namespace boost::numeric;

namespace clustering
{
    class ClusterData {
        public:
            typedef ublas::matrix<ulong64> Data;
            typedef int32_t Klass;
            typedef string Id;
            typedef std::vector<Id> Ids;
            typedef std::vector<ClusterData::Klass> Classes;

            ClusterData(size_t elements_num, size_t features_num);

            ClusterData();
            ~ClusterData();

            void init(size_t elements_num, size_t features_num);

            Classes & getClasses();
            Ids & getIds();
            size_t getElementsNum();
            size_t getFeaturesNum();

            void reset();

            Classes m_cl;
            Ids m_ids;
            Data m;

        private:
            size_t m_features_num;
            size_t m_elements_num;
    };

	class DBSCAN
	{
	public:
        typedef double Distance;
		typedef ublas::vector<double> FeaturesWeights;
		typedef ublas::matrix<Distance> DistanceMatrix;
		typedef std::vector<uint32_t> Neighbors;
		typedef ClusterData::Classes Labels;
        typedef std::map<ClusterData::Klass, std::set<uint32_t> > ClusterIndex;

		static ClusterData::Data gen_cluster_data( size_t features_num, size_t elements_num );
        static ClusterData read_cluster_data( std::istream & is, size_t features_num, size_t elements_num );
		static FeaturesWeights std_weights( size_t s );
        static ublas::vector<Distance> distance(
            ublas::matrix_row<ClusterData::Data> & rowA,
            ublas::matrix_row<ClusterData::Data> & rowB);

		DBSCAN(double eps, size_t min_elems, int num_threads=1);

		DBSCAN();
		~DBSCAN();

		void init(double eps, size_t min_elems, int num_threads=1);
		void fit( const ClusterData::Data & C, bool normalize=false );
		void fit_precomputed( const DistanceMatrix & D );
		void wfit( const ClusterData::Data & C, const FeaturesWeights & W, bool normalize=false );
        void prepare_cluster_index();
		void reset();

		const Labels & get_labels() const;
		const ClusterIndex & get_cluster_index() const;

	private:

		void prepare_labels( size_t s );
		const DistanceMatrix calc_dist_matrix( const ClusterData::Data & C, const FeaturesWeights & W, bool normalize=false);
		Neighbors find_neighbors(const DistanceMatrix & D, uint32_t pid);
		void dbscan( const DistanceMatrix & dm );

		double m_eps;
		size_t m_min_elems;
		int m_num_threads;
		double m_dmin;
		double m_dmax;

		Labels m_labels;
        ClusterIndex m_cluster_index;
	};

	std::ostream& operator<<(std::ostream& o, DBSCAN & d);
    std::ostream& operator<<(std::ostream& o, ClusterData & cl_d);
}
