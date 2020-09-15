import pandas as pd
import numpy as np
import math
import pickle
from sympy import *
import sys
sys.path.insert(0, "/home/pragna/kinpy")
import kinpy as kp
def RotX(angle):
    cti =cos(angle)
    sti =sin(angle)
    res = Matrix([[1, 0, 0, 0], [0 , cti,  -sti,   0],[ 0, sti, cti, 0],[0, 0, 0, 1]])
    return res

def RotY(angle):
    cti= cos(angle)
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

def get_input_joint_pos(q_sa):
    q1 = q_sa[0]
    q2 = q_sa[1]
    q3 = q_sa[2]
    q4 = q_sa[3]
    q5 = q_sa[4]
    q6 = q_sa[5]
    q7 = q_sa[6]
    joint_pos = Gen3DirKin(q1, q2, q3, q4, q5, q6, q7)
    joint_pos_np = np.zeros((len(joint_pos), 3))
    for i, x in enumerate(joint_pos):
        joint_pos_np[i] = np.array(x.tolist()).astype(np.float64).squeeze()

    return joint_pos_np