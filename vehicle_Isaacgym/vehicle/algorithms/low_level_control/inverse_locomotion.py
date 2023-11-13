from math import *
import matplotlib.pyplot as plt
import numpy as np

class conventional_control:
	def __init__(self, f=100):
		############初始运动参数设置############
		self.max_run_time = 3
		self.f = f
		self.h = 0.25
		self.xf = 0.15
		self.xs = -0.15
		self.yf = 0
		self.ys = 0
		self.zs = 0.30
		self.Ts = 0.2
		self.fai = 0.5
		self.l1 = 0.21
		self.l2 = 0.21

	def run(self, env):
		t = 0
		env.reset()
		plotlist = []
		while t < self.max_run_time:
			foot_location_13 = self.foot_trajectory_generate(t, 0)
			foot_location_24 = self.foot_trajectory_generate(t, pi)
			motor_angle_13 = self.inverse_locomotion(foot_location_13[0], foot_location_13[2])
			motor_angle_24 = self.inverse_locomotion(foot_location_24[0], foot_location_24[2])
			motor_angle_list = motor_angle_13+motor_angle_24+motor_angle_24+motor_angle_13
			env.step(motor_angle_list)
			t += 0.01



	def inverse_locomotion(self, x, z, y=0):
		fai = acos((x**2+z**2+self.l1**2-self.l2**2)/(2*sqrt(x**2+z**2)*self.l1))
		theta1 = 0
		# if x >= 0:
		# 	theta2 = abs(atan(x/z))-fai
		# elif x < 0:
		# 	theta2 = atan(x/z) - fai
		theta2 = atan(x / z) - abs(fai)
		theta3 = pi - acos((x**2+z**2-self.l1**2-self.l2**2)/(-2*self.l1*self.l2))
		return [theta1, theta2, theta3]



#####################对角步态规划##########################
	def foot_trajectory_generate(self, time, xiangwei=0):
		###########仿真参数###################
		t = time+xiangwei/(2*pi)*self.Ts
		if t < 0:
			return [0.00001, 0, 0.3]
		t = t % self.Ts

		###########实时足端轨迹解算############
		xep = self.xs
		yep = self.ys
		zep = self.zs
		if t <= self.Ts*self.fai:
			sigma = 2 * pi * t / (self.fai * self.Ts)
			xep = (self.xf - self.xs) *((sigma - sin(sigma)) / (2 * pi)) + self.xs
			yep = (self.yf - self.ys) * ((sigma - sin(sigma)) / (2 * pi)) + self.ys
			zep = -self.h * (1 - cos(sigma)) / 2 + self.zs
		elif t > self.Ts * self.fai and t < self.Ts:
			sigma = 2 * pi * (t-self.fai*self.Ts) / (self.fai * self.Ts)
			xep = self.xf + (self.xs - self.xf) * ((sigma - sin(sigma)) / (2 * pi))
			yep = (self.ys - self.yf) * ((sigma - sin(sigma)) / (2 * pi)) + self.yf
			zep = self.zs

		return [xep, yep, zep]
