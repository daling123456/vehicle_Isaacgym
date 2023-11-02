import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import time
import math
# import xlwt
# import xlrd

class CPG:
    def __init__(self):
        self.p0 = [1, -1, -1, 1, 0, 0, 0, 0]
        self.step_len = 0.1
        self.step_height = 0.3

    def get_param(self):
        alpha = 10          #收敛速度
        leg_num = 4         #腿的数量
        gait = 2            #步态选择
        mu = 1
        a = 50
        psai = [1, 1, 1, 1]             #关节形式 膝式-1 肘式 1
        omega_sw = 2*np.pi
        u1=0
        u2=0                            #误差 影响x,y平衡位置
        h = 0.1                         #抬腿高度
        v = 1                           #行走速度
        gait_t = 0.4                    #步态周期
        s = v * gait_t
        l = 0.21                        #腿节长度
        theta0 = math.radians(40)       #髋关节和膝关节平衡位置与垂直线夹角
        L = 2 * l * np.cos(theta0)          #髋关节与足端之间长度
        return [alpha, leg_num, gait, mu, a, psai, omega_sw, u1, u2]

    def get_gait(self, gait_m):
        #walk
        global beta, phase, v, t, h, ah, ak
        if gait_m == 1:
            beta = 0.75
            phase = [0, 0.5, 0.75, 0.25]
            v = 0.3
            t = 0.8
            h = 0.1
            self.p0 = [1, -1, 0, 0, 0, 0, -1, 1]
        #trot
        elif gait_m == 2:
            beta = 0.5
            phase = [0, 0.5, 0.5, 0]
            v = 1
            t = 0.4
            h = 0.1
            self.p0 = [1, -1, -1, 1, 0, 0, 0, 0]
        #pace
        elif gait_m == 3:
            beta = 0.5
            phase = [0, 0.5, 0, 0.5]
            v = 1.2
            t = 0.4
            h = 0.1
            self.p0 = [1, -1, 1, -1, 0, 0, 0, 0]
        #gallop
        elif gait_m == 4:
            beta = 0.5
            phase = [0, 0, 0.5, 0.5]
            v = 1.6
            t = 0.4
            h = 0.2
            self.p0 = [1, 1, -1, -1, 0, 0, 0, 0]
        return[beta, phase, v, t, h]

    #########计算θ(i,j)############
    def get_theta(self):
        theta = np.zeros([4, 4])
        for i in range(4):
            for j in range(4):
                theta[i, j] = (phase[i] - phase[j])
        return 2 * np.pi * theta

    #################计算耦合项##################
    def get_r(self, x, y):
        r_x = np.zeros([4, 4])
        r_y = np.zeros([4, 4])
        theta = self.get_theta()
        for i in range(4):
            for j in range(4):
                r_x[i, j] = np.cos(theta[i, j]) * x[j] - np.sin(theta[i, j]) * y[j]
                r_y[i, j] = np.sin(theta[i, j]) * x[j] + np.cos(theta[i, j]) * y[j]
        return np.sum(r_x, axis=1), np.sum(r_y, axis=1)

    ################霍普夫振荡器##################
    def hopf(self, gait_m, steps, time):
        beta, phase, v, t, h = self.get_gait(gait_m)
        alpha, leg_num, gait, mu, a, psai, omega_sw, u1, u2 = self.get_param()
        x1, x2, x3, x4, y1, y2, y3, y4 = self.p0
        omega_st = ((1 - beta) / beta) * omega_sw

        x = np.array([x1, x2, x3, x4])
        y = np.array([y1, y2, y3, y4])
        num = len(time)

        dx = np.zeros([4])
        dy = np.zeros([4])
        leg_h_Point_x = np.zeros([num, leg_num])
        leg_h_Point_y = np.zeros([num, leg_num])
        leg_k_Point_x = np.zeros([num, leg_num])
        leg_k_Point_y = np.zeros([num, leg_num])

        for i in range(num):
            r = (x - u1) ** 2 + (y - u2) ** 2
            omega = omega_st / (np.e ** (-100 * y) + 1) + omega_sw / (np.e ** (100 * y) + 1)
            r_x, r_y = self.get_r(x, y)
            dx = alpha * (mu - r) * x - omega * y + r_x
            dy = alpha * (mu - r) * y + omega * x + r_y
            x = x + dx * steps
            y = y + dy * steps

            # leg_h_Point_x[i] = x
            leg_h_Point_x[i] = self.step_len * x
            leg_h_Point_y[i] = h * y

            for j in range(4):
                if y[j] > 0:
                    # leg_k_Point_y[i, j] = 0
                    leg_k_Point_y[i, j] = self.step_height
                else:
                    leg_k_Point_y[i, j] = self.step_height + h * np.sign(psai[j]) * (y[j]**3)
                    # leg_k_Point_y[i, j] = -h * np.sign(psai[j]) * (y[j] ** 3)

        pos = np.hstack([leg_h_Point_x, leg_h_Point_y])
        date = np.hstack([pos, leg_k_Point_y])
        return date

def write_to_excel(data, gait_type, f):
    sheet1 = f.add_sheet(gait_type, cell_overwrite_ok=True)  # 创建sheet工作表
    sheet1.write(0, 0, "x1")
    sheet1.write(0, 1, "y1")
    sheet1.write(0, 2, "x2")
    sheet1.write(0, 3, "y2")
    sheet1.write(0, 4, "x3")
    sheet1.write(0, 5, "y3")
    sheet1.write(0, 6, "x4")
    sheet1.write(0, 7, "y4")
    for i in range(len(data)):
        sheet1.write(i+1, 0, data[i, 0])
        sheet1.write(i+1, 1, data[i, 8])
        sheet1.write(i+1, 2, data[i, 1])
        sheet1.write(i+1, 3, data[i, 9])
        sheet1.write(i+1, 4, data[i, 2])
        sheet1.write(i+1, 5, data[i, 10])
        sheet1.write(i+1, 6, data[i, 3])
        sheet1.write(i+1, 7, data[i, 11])


def train_CPG():
    pass

# time = np.arange(0, 10, 0.001)
# cpg = CPG()
# f = xlwt.Workbook('encoding = utf-8')  # 设置工作簿编码
#
# date = cpg.hopf(1, 0.001, time)
# write_to_excel(date, "walk", f)
# date = cpg.hopf(2, 0.001, time)
# write_to_excel(date, "trot", f)
# date = cpg.hopf(3, 0.001, time)
# write_to_excel(date, "pace", f)
# date = cpg.hopf(4, 0.001, time)
# write_to_excel(date, "gallop", f)
#
# f.save('cpg_data.xls')  # 保存.xls到当前工作目录
# print(date.shape)
# plt.subplot(4, 1, 1)
# plt.title('figure1')
# plt.plot(time, date[:, 0], "-r", label="x")
# plt.plot(time, date[:, 4], "-y", label="pre_y")
# plt.plot(time, date[:, 8], "-b", label="y")
# plt.plot(date[:, 0], date[:, 8])
# # print(date[:,0],date[:,4])
# plt.legend(loc="right")
# # plt.show()
# plt.subplot(4, 1, 2)
# # plt.title('figure2')
# plt.plot(time, date[:, 1], "-r", label="x")
# plt.plot(time, date[:, 5], "-y", label="pre_y")
# plt.plot(time, date[:, 9], "-b", label="y")
# plt.plot(date[:, 1], date[:, 9])
# # print(date[:, 0], date[:, 4])
# plt.legend(loc="right")
# # plt.show()
# plt.subplot(4, 1, 3)
# # plt.title('figure3')
# plt.plot(time, date[:, 2], "-r", label="x")
# plt.plot(time, date[:, 6], "-y", label="pre_y")
# plt.plot(time, date[:, 10], "-b", label="y")
# plt.plot(date[:, 2], date[:, 10])
# # print(date[:, 0], date[:, 4])
# plt.legend(loc="right")
# # plt.show()
# plt.subplot(4, 1, 4)
# # plt.title('figure4')
# plt.plot(time, date[:, 3], "-r", label="x")
# plt.plot(time, date[:, 7], "-y", label="pre_y")
# plt.plot(time, date[:, 11], "-b", label="y")
# plt.plot(date[:, 3], date[:, 11])
# # print(date[:, 0], date[:, 4])
# plt.legend(loc="right")
# plt.show()
#
# plt.title('fig')
# plt.plot(time, date[:, 0], "-r", label="x1")
# plt.plot(time, date[:, 1], "-b", label="x2")
# plt.plot(time, date[:, 2], "-y", label="x3")
# plt.plot(time, date[:, 3], "-g", label="x4")
#
# plt.legend(loc="right")
# plt.show()