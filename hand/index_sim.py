import numpy as np
from math import pi, cos, sin, sqrt 

def actuation_index(distance, q, current_pose, links_lenghts, integral, timestep):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2] 
    dr_dot = 0

    offset_index_y = links_lenghts[0]
    offset_index_z = links_lenghts[1]
    L1y_index = links_lenghts[2]
    L1z_index = links_lenghts[3]
    L2y_index = links_lenghts[4]
    L2z_index = links_lenghts[5]
    L3y_index = links_lenghts[6]
    L3z_index = links_lenghts[7]
    
    py_j2 = offset_index_y - L1y_index*cos(q1) - L1z_index*sin(q1)
    pz_j2 = offset_index_z - L1y_index*sin(q1) + L1z_index*cos(q1)
    
    py_j3 = py_j2 - L2y_index*cos(q1+q2) - L2z_index*sin(q1+q2) 
    pz_j3 = pz_j2 - L2y_index*sin(q1+q2) + L2z_index*cos(q1+q2)

    py_tip = py_j3 - L3y_index*cos(q1+q2+q3) - L3z_index*sin(q1+q2+q3) + 0.25
    pz_tip = pz_j3 - L3y_index*sin(q1+q2+q3) + L3z_index*cos(q1+q2+q3) - 1.1

    p = np.array([[py_tip], [pz_tip]])

    # Task Jacobian: p_dot = Jp(q)q_dot
    Jp = np.array([[L1y_index*sin(q1) - L1z_index*cos(q1) + L2y_index*sin(q1+q2) - L2z_index*cos(q1+q2) + L3y_index*sin(q1+q2+q3) - L3z_index*cos(q1+q2+q3), 
                    L2y_index*sin(q1+q2) - L2z_index*cos(q1+q2) + L3y_index*sin(q1+q2+q3) - L3z_index*cos(q1+q2+q3), 
                    L3y_index*sin(q1+q2+q3) - L3z_index*cos(q1+q2+q3)], 
                    [-L1y_index*cos(q1) - L1z_index*sin(q1) - L2y_index*cos(q1+q2) - L2z_index*sin(q1+q2) - L3y_index*cos(q1+q2+q3) - L3z_index*sin(q1+q2+q3),
                    -L2y_index*cos(q1+q2) - L2z_index*sin(q1+q2) - L3y_index*cos(q1+q2+q3) - L3z_index*sin(q1+q2+q3),
                    -L3y_index*cos(q1+q2+q3) - L3z_index*sin(q1+q2+q3)]])
    
    # Distance from origin d
    d = np.linalg.norm(p)

    W = [[7.5, 0,  0], 
         [0,  5.5,  0], 
         [0,  0, 6.5]]

    #print(d - distance)

    # Extended jacobian for distance: d_dot = Jd(q)q_dot
    Jd = 1/d*p.transpose().dot(Jp)

    W_inv = np.linalg.inv(W)

    Jd_trans = Jd.transpose()
    detJ = np.float64(Jd.dot(W_inv.dot(Jd_trans)))
    if detJ <= 1e-1:
        mu = (detJ + 1.0)/20
    else:
        mu = 0
    Jinv = W_inv.dot(Jd_trans).dot(1/(detJ + mu**2))
    
    # Proportional Gain
    K = 15

    K_d = 0

    # Distance measured from WeArt 
    dr = distance
    dr_dot = 0

    e = dr - d
    
    #e_dot = (e - e_old)/timestep
    e_dot = 0
    
    if np.linalg.norm(current_pose - np.array([0, 0, 1])) >= 0.7:
        K_i = 0
        integral += 0
    else:
        integral += dr - d
        if abs(integral) < 1e-4:
            integral = 0
        K_i = 0.1
    
    w = np.array([(5*pi/16 - q1)/(3*pi/8), (3*pi/8 - q2)/(pi/4), (5*pi/16 - q3)/(3*pi/8)])
    u_vinc = (np.eye(3) - Jinv.dot(Jd))*w.transpose()
    qdot = Jinv.dot(dr_dot + K*(dr - d) + K_i*integral + K_d*e_dot) + u_vinc

    return qdot, integral, d
