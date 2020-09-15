import sys
sys.path.insert(0, '/home/pragna/Documents/Documents/collision/collision_model/')
sys.path.insert(0, '/home/pragna/Documents/Documents/collision/collision_model/main/')
# sys.path.insert(0, '/home/pragna/Documents/Documents/collision/collision_model/main/')
from get_centroids_Octree import get_centroids_Octree
from input_jt_pos import get_input_joint_pos
from segmentation.region_grow import region_growing
from main_nn import main_nn
#if __name__ == "__main__":
def collision_gradient(q_input):
    ### get_centroids_Octree is a function to generate the octree and produce the object centroids.
    # It takes the current scene as pcd and produce centroid in .dat files
    # A folder is involved: the pcd files are in segmented_pcd/recorded_pcd_fg inside home
    # A .dat file is created for each pcd inside segmented_pcd/recorded_pcd_fg, since we do not need to train here, so a single consolidated centroid.dat file is not required.
    # This individual .dat files contains all the centroids of the individual pcds
    centroid_file = get_centroids_Octree(pcd='/home/pragna/segment_pcd/recorded_pcd/510090000.pcd')
    print("Centroid file", centroid_file)
    # get_input_config is a function to get the current joint angles.
    # It is input from the Ozan's program. It is a numpy 1D- numpy array
    # jt_vec = get_input_joint_pos(q_input)
    jac_input = main_nn(centroid_file=centroid_file, input_config=q_input)
    return jac_input
