import numpy as np
from math import pi, cos, sin, sqrt 

def actuation_middle(distance, q, current_pose, links_lenghts, integral, palm, R):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    # print(q1, q2, q3)
    dr_dot = 0

    offset_middle_x = links_lenghts[0]
    offset_middle_y = links_lenghts[1]
    offset_middle_z = links_lenghts[2]
    L1y_middle = links_lenghts[3]
    L1z_middle = links_lenghts[4]
    L2y_middle = links_lenghts[5]
    L2z_middle = links_lenghts[6]
    L3y_middle = links_lenghts[7]
    L3z_middle = links_lenghts[8]
    
    # print("LINKS: ",links_lenghts)
    #print(np.linalg.norm(current_pose - np.array([0, 0, 1])))

    py_j2 = - L1y_middle*cos(q1) - L1z_middle*sin(q1)
    pz_j2 = - L1y_middle*sin(q1) + L1z_middle*cos(q1)
        
    py_j3 = py_j2 - L2y_middle*cos(q1+q2) - L2z_middle*sin(q1+q2) 
    pz_j3 = pz_j2 - L2y_middle*sin(q1+q2) + L2z_middle*cos(q1+q2)

    py_tip = py_j3 - L3y_middle*cos(q1+q2+q3) - L3z_middle*sin(q1+q2+q3)
    pz_tip = pz_j3 - L3y_middle*sin(q1+q2+q3) + L3z_middle*cos(q1+q2+q3)


    pos = np.array([[0], [py_tip], [pz_tip]])  

    p = R.dot(pos) + np.array([[offset_middle_x], [offset_middle_y], [offset_middle_z]])
    
    # print("Position", p.T)

    # Task Jacobian: p_dot = Jp(q)q_dot
    Jp = np.array([[0, 0, 0],
                   [L1y_middle*sin(q1) - L1z_middle*cos(q1) + L2y_middle*sin(q1+q2) - L2z_middle*cos(q1+q2) + L3y_middle*sin(q1+q2+q3) - L3z_middle*cos(q1+q2+q3), 
                    L2y_middle*sin(q1+q2) - L2z_middle*cos(q1+q2) + L3y_middle*sin(q1+q2+q3) - L3z_middle*cos(q1+q2+q3), 
                    L3y_middle*sin(q1+q2+q3) - L3z_middle*cos(q1+q2+q3)], 
                   [-L1y_middle*cos(q1) - L1z_middle*sin(q1) - L2y_middle*cos(q1+q2) - L2z_middle*sin(q1+q2) - L3y_middle*cos(q1+q2+q3) - L3z_middle*sin(q1+q2+q3),
                    -L2y_middle*cos(q1+q2) - L2z_middle*sin(q1+q2) - L3y_middle*cos(q1+q2+q3) - L3z_middle*sin(q1+q2+q3),
                    -L3y_middle*cos(q1+q2+q3) - L3z_middle*sin(q1+q2+q3)]])

    Jp = R.dot(Jp)
    # Distance from origin d
    d = np.linalg.norm(p)
    

    W = [[10.5, 0,  0], 
         [0,  4.5,  0], 
         [0,  0, 7.5]]

    #print(d - distance)

    # Extended jacobian for distance: d_dot = Jd(q)q_dot
    Jd = 1/d*p.transpose().dot(Jp)
    W_inv = np.linalg.inv(W)

    Jd_trans = Jd.transpose()
    detJ = np.float64(Jd.dot(W_inv.dot(Jd_trans)))
    if detJ <= 1e-5:
        mu = (detJ + 1.0)/2000
    else:
        mu = 0
    Jinv = W_inv.dot(Jd_trans).dot(1/(detJ + mu**2))
    
    # Proportional Gain
    K = 100

    # Distance measured from WeArt 
    dr = distance
    dr_dot = 0

    if np.linalg.norm(current_pose - palm) >= 0.1:
        K_i = 0
        integral += 0
    else:
        integral += dr - d
        if abs(integral) < 1e-6:
            integral = 0
        K_i = 0.1
    K_i = 0
    
    w = np.array([(5*pi/16 - q1)/(3*pi/8), (3*pi/8 - q2)/(pi/4), (5*pi/16 - q3)/(3*pi/8)])
    u_vinc = (np.eye(3) - Jinv.dot(Jd))*w.transpose()
    qdot = Jinv.dot(dr_dot + K*(dr - d) + K_i*integral) + u_vinc

    return qdot, integral