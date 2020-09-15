from sympy import *
import pandas as pd
import pickle
from scipy.spatial.distance import cdist
import math
import numpy as np
import sys
sys.path.insert(0, "/home/pragna/kinpy")
import kinpy
import open3d as o3d
import os
import glob

def RotX(angle):
    cti =cos(angle)
    sti =sin(angle)
    res = Matrix([[1, 0, 0, 0], [0 , cti,  -sti,   0],[ 0, sti, cti, 0],[0, 0, 0, 1]])
    return res

def RotY(angle):
    cti=cos(angle)
    sti=sin(angle)
    res = Matrix([[  cti, 0, sti, 0],[ 0, 1, 0,0],[ -sti, 0, cti, 0],[0, 0, 0, 1]])
    return res

def RotZ(angle):
    cti = cos(angle);
    sti = sin(angle);
    res = Matrix([[cti, - sti, 0, 0],[ sti, cti, 0, 0],[ 0, 0, 1, 0],[ 0, 0, 0, 1]])
    return res

def TranX(t):
    res = Matrix([[ 1, 0, 0, t], [ 0, 1, 0, 0],[ 0, 0, 1, 0],[0,0,0,1]])
    return res

def TranY(t):
    res = Matrix([[ 1, 0, 0, 0], [ 0, 1, 0, t],[ 0, 0, 1, 0],[0,0,0,1]])
    return res

def TranZ(t):
    res = Matrix([[ 1, 0, 0, 0], [ 0, 1, 0, 0],[ 0, 0, 1, t],[0,0,0,1]])
    return res


def Skew(v):
    res = Matrix([[0, - v[2], v[1]], [v[2], 0, - v[0]], [- v[1], v[0], 0]])
    return res

def Jacobian(endEffector, t0, t1, t2, t3, t4, t5, t6, z0, z1, z2, z3, z4, z5, z6):

    res = zeros(6, 7)

    res[0: 3, 0] = (Transpose(Skew(endEffector - t0))) * z0
    res[0: 3, 1] = (Transpose(Skew(endEffector - t1))) * z1
    res[0: 3, 2] = (Transpose(Skew(endEffector - t2))) * z2
    res[0: 3, 3] = (Transpose(Skew(endEffector - t3))) * z3
    res[0: 3, 4] = (Transpose(Skew(endEffector - t4))) * z4
    res[0: 3, 5] = (Transpose(Skew(endEffector - t5))) * z5
    res[0: 3, 6] = (Transpose(Skew(endEffector - t6))) * z6
#
    res[3: 6, 0] = z0;
    res[3: 6, 1] = z1;
    res[3: 6, 2] = z2;
    res[3: 6, 3] = z3;
    res[3: 6, 4] = z4;
    res[3: 6, 5] = z5;
    res[3: 6, 6] = z6;
    return res


def Gen3DirKin(q1, q2, q3, q4, q5, q6, q7):

    d1 = -128.4
    theta1 = 0
    a1 = 0
    alpha1 = pi / 2
    b1 = -5.4
    beta1 = 0

    d2 = -6.4;
    theta2 = 0
    a2 = 0
    alpha2 = -pi / 2
    b2 = -210.4
    beta2 = 0

    d3 = -210.4
    theta3 = 0
    a3 = 0
    alpha3 = pi / 2
    b3 = -6.4
    beta3 = 0

    d4 = -6.4
    theta4 = 0
    a4 = 0
    alpha4 = -pi / 2
    b4 = -208.4
    beta4 = 0

    d5 = -105.9
    theta5 = 0
    a5 = 0
    alpha5 = pi / 2
    b5 = -0
    beta5 = 0

    d6 = 0
    theta6 = 0
    a6 = 0
    alpha6 = -pi / 2
    b6 = -105.9
    beta6 = 0

    d7 = -61.5
    theta7 = 0
    a7 = 0
    alpha7 = pi
    b7 = 0
    beta7 = 0

    m_LTR = TranZ(156.4) * RotZ(0) * TranX(0) * RotX(pi) * TranZ(0) * RotZ(0)
    m_0T1 = TranZ(d1) * RotZ(q1) * TranX(a1) * RotX(alpha1) * TranZ(b1) * RotZ(beta1)
    m_1T2 = TranZ(d2) * RotZ(q2) * TranX(a2) * RotX(alpha2) * TranZ(b2) * RotZ(beta2)
    m_2T3 = TranZ(d3) * RotZ(q3) * TranX(a3) * RotX(alpha3) * TranZ(b3) * RotZ(beta3)
    m_3T4 = TranZ(d4) * RotZ(q4) * TranX(a4) * RotX(alpha4) * TranZ(b4) * RotZ(beta4)
    m_4T5 = TranZ(d5) * RotZ(q5) * TranX(a5) * RotX(alpha5) * TranZ(b5) * RotZ(beta5)
    m_5T6 = TranZ(d6) * RotZ(q6) * TranX(a6) * RotX(alpha6) * TranZ(b6) * RotZ(beta6)
    m_6T7 = TranZ(d7) * RotZ(q7) * TranX(a7) * RotX(alpha7) * TranZ(b7) * RotZ(beta7)

    m_0T7Now = m_LTR * m_0T1 * m_1T2 * m_2T3 * m_3T4 * m_4T5 * m_5T6 * m_6T7

    m_0T1Now = m_LTR * m_0T1
    rot_0R1Now = m_0T1Now[0:3, 0: 3]
    v_0p1Now = m_0T1Now[0:3, 3]
    v_0z1Now = m_0T1Now[0:3, 2]

    m_0T2Now = m_0T1Now * m_1T2
    rot_0R2Now = m_0T2Now[0:3, 0: 3]
    v_0p2Now = m_0T2Now[0:3, 3]
    v_0z2Now = m_0T2Now[0:3, 2]

    m_0T3Now = m_0T2Now * m_2T3
    rot_0R3Now = m_0T3Now[0:3, 0: 3]
    v_0p3Now = m_0T3Now[0:3, 3]
    v_0z3Now = m_0T3Now[0:3, 2]

    m_0T4Now = m_0T3Now * m_3T4
    rot_0R4Now = m_0T4Now[0:3, 0: 3]
    v_0p4Now = m_0T4Now[0:3, 3]
    v_0z4Now = m_0T4Now[0:3, 2]

    m_0T5Now = m_0T4Now * m_4T5
    rot_0R5Now = m_0T5Now[0:3, 0: 3]
    v_0p5Now = m_0T5Now[0:3, 3]
    v_0z5Now = m_0T5Now[0:3, 2]

    m_0T6Now = m_0T5Now * m_5T6
    rot_0R6Now = m_0T6Now[0:3, 0: 3]
    v_0p6Now = m_0T6Now[0:3, 3]
    v_0z6Now = m_0T6Now[0:3, 2]

    rot_0R7Now = m_0T7Now[0:3, 0: 3]
    v_0p7Now = m_0T7Now[0:3, 3]
    v_0z7Now = m_0T7Now[0:3, 2]

    rot_BaseR0Now = m_LTR[0:3, 0: 3]
    v_Basep0Now = m_LTR[0:3, 3]
    v_Basez0Now = m_LTR[0:3, 2]

    # Jaco67 = Jacobian(v_0p7Now, v_Basep0Now, v_0p1Now, v_0p2Now, v_0p3Now, v_0p4Now, v_0p5Now, v_0p6Now, v_Basez0Now,
    #                   v_0z1Now, v_0z2Now, v_0z3Now, v_0z4Now, v_0z5Now, v_0z6Now)


    #return m_0T7Now,Jaco67

    return v_0p1Now, v_0p2Now, v_0p3Now, v_0p4Now, v_0p5Now, v_0p6Now, v_0p7Now

def transformPts(dirName, fileName):
    bunch3d_array = np.zeros((2, 1, 2))
    obstacle_3d = np.zeros((2, 1, 2))
    pt_cloud_file = os.path.join(dirName, fileName)
#read .ply file
    pcd_load = o3d.io.read_point_cloud(pt_cloud_file)
#save it as a numpy array
    xyz_load = np.asarray(pcd_load.points) #shape is (226995, 3)
#are all points useful? if not, segregate them, how?

    #sample and get obstacle arrays, output will be A bunch of 2D arrays
    # each array wil have cols as x,y,z and each 2d array is a configuration, a bunch of such arrays will be bunch of configurations
    # output of this function is a 3D array of n-size where n is the number of samples of 2d array, which is bunch3d_array.. here is formed, it has to be modified to appropiate size
    # segregate the obstacle 3D too..the structure will be same
    return bunch3d_array, obstacle_3d

def mid_point(x,y):
    x_mid = 0.5 * (x[0]+y[0])
    y_mid = 0.5 * (x[1] + y[1])
    z_mid = 0.5 * (x[2] + y[2])
    return [x_mid, y_mid, z_mid]


# if __name__ == "__main__":
def transform_centroids_read_config():
    files = '/home/pragna/Documents/Documents/collision/collision_model/main/octree_centroids.dat' #centroids after region_growing
    fw = open(files, 'r')
    lines = fw.readlines()
    centroids = []
    span = []
    radius = []
    # #fw = open('objects.txt', 'w')
    # allFiles = []

    for l in lines:
        split_lines = l.split(',')
        filename = split_lines[0]
        centroids.append(float(split_lines[1]))
        centroids.append(float(split_lines[2]))
        centroids.append(float(split_lines[3]))
        span.append(float(split_lines[4]))
        span.append(float(split_lines[5]))
        span.append(float(split_lines[6]))
        radius.append(float(split_lines[7])/2.0)
    centroids = np.asarray(centroids).reshape((-1, 3))

    span = np.asarray(span).reshape((-1, 3))
    translation = np.asarray([-0.80, 1.25, 1.80])
    translation = translation[np.newaxis,:]
    centroids_world = centroids.astype(np.float) - translation.repeat(centroids.shape[0], axis=0)
    translation_lr = np.asarray([-0.8777, -0.1488, 1.191])
    translation_rr = np.asarray([-0.42882, -0.1488, 1.191])
    centroids_lr = centroids_world + translation_lr
    centroids_rr = centroids_world + translation_rr
    # identify which are robots' part among the centroids

    # generate random translations
    mu, sigma = 0, 1  # mean and standard deviation
    t_x = np.random.normal(mu, sigma, 10)

    # mu, sigma = 0, 1  # mean and standard deviation
    t_y = np.random.normal(mu, sigma, 10)

    # mu, sigma = 0, 1  # mean and standard deviation
    t_z = np.random.normal(mu, sigma, 10)
    translation_random = [t_x, t_y, t_z]
    translated_centroids_rr =[]

    for i, x in enumerate(centroids_rr): #why centroids_lr is not having the same translation?
        translated_centroids_rr.extend(np.array(translation_random).T+x)
    translated_centroids_rr = np.array(translated_centroids_rr)
    f1 = open("octree_TranslatedCentroid_Data_rr.pkl", 'wb') #saving translated centroids_rr in pickle
    pickle.dump(translated_centroids_rr, f1)
    f1.close()
    translated_centroids_lr = []
    for i, x in enumerate(centroids_lr): #why centroids_lr is not having the same translation?
        translated_centroids_lr.extend(np.array(translation_random).T+x)
    translated_centroids_lr = np.array(translated_centroids_lr)
    f2 = open("octree_TranslatedCentroid_Data_lr.pkl", 'wb') #saving translated centroids_rr in pickle
    pickle.dump(translated_centroids_lr, f2)
    f2.close()

    # df = pd.read_csv('/home/pragna/Documents/Documents/collision/collision_model/Fk_ale/KinovaWorkSpaceInGloveBox.csv')
    #
    # print(len(df))
    #
    # df = np.array(df[['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7']])
    #
    # print(df.shape)

    # chain = kp.build_serial_chain_from_urdf(open("GEN3_URDF_V12.urdf").read(), "EndEffector_Link")
    # print(chain)
    # print(chain.get_joint_parameter_names())

    # dirName = "/home/pragna/Documents/Documents/collision/"
    # base = "Fwd_kin"
    # suffix = ".pkl"
    # list_midpts = []
    # list_jtPos = []
    # list_sum = []
    # list_TB7_np = []
    # f3 = open("TB7_np_Data.pkl", 'wb') #saving joint (x,y,z) co-ordinates
    #
    # for j in range(0, df.shape[0]):
    #     # print("No of rows:", j)
    #     q1 = math.radians(df[j][0])
    #     q2 = math.radians(df[j][1])
    #     q3 = math.radians(df[j][2])
    #     q4 = math.radians(df[j][3])
    #     q5 = math.radians(df[j][4])
    #     q6 = math.radians(df[j][5])
    #     q7 = math.radians(df[j][6])
    #
    #     TB7 = Gen3DirKin(q1, q2, q3, q4, q5, q6, q7)
    #     TB7_np = np.zeros((len(TB7), 3))
    #     for i, x in enumerate(TB7):
    #         TB7_np[i] = np.array(x.tolist()).astype(np.float64).squeeze()
    #
    #     pickle.dump(TB7_np, f3)
        # midpoints = np.zeros((len(TB7)-1, 3))
        # for i in range(len(TB7)-1):
        #     midpoints[i] = np.array(mid_point(TB7[i], TB7[i+1]))
        # eucleadeanDist = cdist(midpoints, translated_centroids)
        # sumDist = np.sum(eucleadeanDist, axis=0)
        # pickle.dump(sumDist, f3)
        #list_sum.append(sumDist)
    # f3.close()
    #f3.close()
    #print(len(list_sum))
    #print(list_sum[0].shape)

