import math
import numpy as np
from scipy.spatial.transform import Rotation
from math import radians, sin, cos

import torch

def dh_matrix(a, alpha, d, theta):
# 传入四个DH参数，根据公式3-6，输出一个T矩阵。
    alpha = alpha / 180 * np.pi
    theta = theta / 180 * np.pi
    matrix = np.identity(4)
    matrix[0,0] = cos(theta)
    matrix[0,1] = -sin(theta)
    matrix[0,2] = 0
    matrix[0,3] = a
    matrix[1,0] = sin(theta)*cos(alpha)
    matrix[1,1] = cos(theta)*cos(alpha)
    matrix[1,2] = -sin(alpha)
    matrix[1,3] = -sin(alpha)*d
    matrix[2,0] = sin(theta)*sin(alpha)
    matrix[2,1] = cos(theta)*sin(alpha)
    matrix[2,2] = cos(alpha)
    matrix[2,3] = cos(alpha)*d
    matrix[3,0] = 0
    matrix[3,1] = 0
    matrix[3,2] = 0
    matrix[3,3] = 1
    return matrix


def dh_transform(alpha, beta, gamma, x, y, z):
# 传入四个DH参数，根据公式3-6，输出一个T矩阵。
    alpha = alpha / 180 * np.pi
    beta = beta / 180 * np.pi
    gamma = gamma / 180 * np.pi
    matrix = np.identity(4)
    matrix[0,0] = cos(alpha)*cos(beta)
    matrix[0,1] = cos(alpha)*sin(beta)*cos(gamma)-sin(alpha)*cos(gamma)
    matrix[0,2] = cos(alpha)*sin(beta)*cos(gamma)+sin(alpha)*sin(gamma)
    matrix[0,3] = x
    matrix[1,0] = sin(alpha)*cos(beta)
    matrix[1,1] = sin(alpha)*sin(beta)*sin(gamma)+cos(alpha)*cos(gamma)
    matrix[1,2] = sin(alpha)*sin(beta)*sin(gamma)-cos(alpha)*sin(gamma)
    matrix[1,3] = y
    matrix[2,0] = -sin(beta)
    matrix[2,1] = cos(beta)*sin(gamma)
    matrix[2,2] = cos(beta)*cos(gamma)
    matrix[2,3] = z
    matrix[3,0] = 0
    matrix[3,1] = 0
    matrix[3,2] = 0
    matrix[3,3] = 1
    return matrix

def getPoint_Transfor(theta1,theta2,theta3,l):
    Trans_matrix=[]
    # Trans_params=[[90,0,-90,1.7,0.2,-0.476],[-90,0,-90,l,-0.2,-0.476],[90,0,-90,l,0.2,-0.476],[-90,0,-90,1.7,-0.2,-0.476],[90,0,-90,-1.7,0.2,-0.476],[-90,0,-90,-1.7,-0.2,-0.476]]
    Trans_params =[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
    for i in range(6):
        # DH_params=[[0.,0,0,theta1[i]],[0.5,0,0,theta2[i]],[0,90,0,theta3[i]-90],[0,90,-0.165,90]]
        DH_params=[[0.,0,0,theta1[i]],[0.5,180,0,theta2[i]-90],[0,90,-0.165,0]]
        DH_matrix=np.matmul(dh_matrix(DH_params[0][0],DH_params[0][1],DH_params[0][2],DH_params[0][3]),dh_matrix(DH_params[1][0],DH_params[1][1],DH_params[1][2],DH_params[1][3]))
        DH_matrix = np.matmul(DH_matrix,dh_matrix(DH_params[2][0],DH_params[2][1],DH_params[2][2],DH_params[2][3]),)
        Trans_matrix.append(np.matmul(dh_transform(Trans_params[i][0],Trans_params[i][1],Trans_params[i][2],Trans_params[i][3],Trans_params[i][4],Trans_params[i][5]),DH_matrix))
    P_i_o = torch.tensor(np.matmul(Trans_matrix,np.array([0,0,0,1])))
    P_s_o = torch.tensor(np.matmul(dh_transform(0,0,0,l,0,-0.476),np.array([0,0,0,1])))
    print(torch.tensor(Trans_matrix))
    print(P_i_o)
    mass = {'slider_mass':10, 'wheel_mass':30}
    P_m_o = (mass['slider_mass']*P_s_o + torch.sum(mass['wheel_mass'] * P_i_o,0)) / 1146
    P_it=torch.tensor([0.463,0,0,1])
    P_ci_o=np.matmul(Trans_matrix,P_it)
    return P_m_o, P_ci_o

def stability_margin(actions,X,Y,Z):
    theta1=[actions[0],actions[4], actions[8], actions[12]+ actions[17]+ actions[21]]
    theta2 = [actions[1], actions[5], actions[9], actions[13] + actions[18] + actions[22]]
    theta3 = [actions[2], actions[6], actions[10], actions[14] + actions[19] + actions[23]]
    P_m_o, P_ci_o=getPoint_Transfor(theta1, theta2, theta3, actions[16])

    from itertools import combinations
    numbers=list(range(6))
    combination_list=list(combinations(numbers,3))
    for i,j,k in combination_list:
        Lij = torch.sqrt(torch.square(P_ci_o[:, i, 0] - P_ci_o[:, j, 0]) + torch.square(
            P_ci_o[:, i, 1] - P_ci_o[:, j, 1]) + torch.square(P_ci_o[:, i, 2] - P_ci_o[:, j, 2]))
        Lmi = torch.sqrt(torch.square(P_m_o[:, 0] - P_ci_o[:, i, 0]) + torch.square(P_m_o[:, 1] - P_ci_o[:, i, 1]) + torch.square(P_m_o[:, 2] - P_ci_o[:, i, 2]))
        Lmj = torch.sqrt(torch.square(P_m_o[:, 0] - P_ci_o[:, j, 0]) + torch.square(P_m_o[:, 1] - P_ci_o[:, j, 1]) + torch.square(P_m_o[:, 2] - P_ci_o[:, j, 2]))
        Hijmax=Lmi*torch.sqrt(1-torch.square((torch.square(Lmi)+torch.square(Lij)-torch.square(Lmj))/(2*Lmi*Lij)))

        Plane_A = (P_ci_o[:, j, 1] - P_ci_o[:, i, 1]) * (P_ci_o[:, k, 2] - P_ci_o[:, i, 2]) - (P_ci_o[:, k, 1] - P_ci_o[:, i, 1]) * (P_ci_o[:, j, 2] - P_ci_o[:, i, 2])
        Plane_B = (P_ci_o[:, j, 2] - P_ci_o[:, i, 2]) * (P_ci_o[:, k, 0] - P_ci_o[:, i, 0]) - (P_ci_o[:, k, 2] - P_ci_o[:, i, 2]) * (P_ci_o[:, j, 0] - P_ci_o[:, i, 0])
        Plane_C = (P_ci_o[:, j, 0] - P_ci_o[:, i, 0]) * (P_ci_o[:, k, 1] - P_ci_o[:, i, 1]) - (P_ci_o[:, k, 0] - P_ci_o[:, i, 0]) * (P_ci_o[:, j, 1] - P_ci_o[:, i, 1])
        Plane_D = -Plane_A*P_ci_o[:, i, 0]-Plane_B*P_ci_o[:, i, 1]-Plane_C*P_ci_o[:, i, 2]
        Hijcom=torch.abs(Plane_A*P_m_o[:, 0]+Plane_B*P_m_o[:, 1]+Plane_C*P_m_o[:, 2]+Plane_D)/torch.sqrt(torch.square(Plane_A)+torch.square(Plane_B)+torch.square(Plane_C))

        PLane_norm_o=torch.stack((Plane_A,Plane_B,Plane_C),1)
        R_o_w=torch.tensor(Rotation.from_euler('zyx',[Z,Y,X]))
        PLane_norm_w=R_o_w*PLane_norm_o
        cos_gamma=torch.abs(PLane_norm_w[2])/torch.sqrt(torch.square(PLane_norm_w[0])+torch.square(PLane_norm_w[1])+torch.square(PLane_norm_w[2]))
        Enesmij=(Hijmax-Hijcom)*cos_gamma
        return Enesmij
