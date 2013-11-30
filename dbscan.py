import numpy as np
from pydbscan import DBSCAN, gen_cluster_data

db = DBSCAN()
db.init( 5.5, 5, 4 )

cld = gen_cluster_data(225, 10000)

db.fit( cld )

print db.get_labels()