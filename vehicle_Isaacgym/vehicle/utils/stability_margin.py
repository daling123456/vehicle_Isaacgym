

def stability_margin(actions):
    theta1 = [actions[0], actions[4], actions[8], actions[12] + actions[17] + actions[21]]
    theta2 = [actions[1], actions[5], actions[9], actions[13] + actions[18] + actions[22]]
    theta3 = [actions[2], actions[6], actions[10], actions[14] + actions[19] + actions[23]]
    P_m_o, P_ci_o = getPoint_Transform(theta1, theta2, theta3, actions[16])


def getPoint_Transform(theta1, theta2, theta3, l):

    pass