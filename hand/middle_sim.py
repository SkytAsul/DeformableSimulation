import numpy as np
from math import pi, cos, sin, sqrt 

def actuation_middle(distance, q, current_pose, links_lenghts, integral, timestep):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]    
    dr_dot = 0

    offset_middle_y = links_lenghts[0]
    offset_middle_z = links_lenghts[1]
    L1y_middle = links_lenghts[2]
    L1z_middle = links_lenghts[3]
    L2y_middle = links_lenghts[4]
    L2z_middle = links_lenghts[5]
    L3y_middle = links_lenghts[6]
    L3z_middle = links_lenghts[7]
    

    #print(np.linalg.norm(current_pose - np.array([0, 0, 1])))

    py_j2 = offset_middle_y - L1y_middle*cos(q1) - L1z_middle*sin(q1)
    pz_j2 = offset_middle_z - L1y_middle*sin(q1) + L1z_middle*cos(q1)
        
    py_j3 = py_j2 - L2y_middle*cos(q1+q2) - L2z_middle*sin(q1+q2) 
    pz_j3 = pz_j2 - L2y_middle*sin(q1+q2) + L2z_middle*cos(q1+q2)

    py_tip = py_j3 - L3y_middle*cos(q1+q2+q3) - L3z_middle*sin(q1+q2+q3) + 0.2
    pz_tip = pz_j3 - L3y_middle*sin(q1+q2+q3) + L3z_middle*cos(q1+q2+q3) - 1.1

    p = np.array([[py_tip], [pz_tip]])    
    
    # Task Jacobian: p_dot = Jp(q)q_dot
    Jp = np.array([[L1y_middle*sin(q1) - L1z_middle*cos(q1) + L2y_middle*sin(q1+q2) - L2z_middle*cos(q1+q2) + L3y_middle*sin(q1+q2+q3) - L3z_middle*cos(q1+q2+q3), 
                    L2y_middle*sin(q1+q2) - L2z_middle*cos(q1+q2) + L3y_middle*sin(q1+q2+q3) - L3z_middle*cos(q1+q2+q3), 
                    L3y_middle*sin(q1+q2+q3) - L3z_middle*cos(q1+q2+q3)], 
                    [-L1y_middle*cos(q1) - L1z_middle*sin(q1) - L2y_middle*cos(q1+q2) - L2z_middle*sin(q1+q2) - L3y_middle*cos(q1+q2+q3) - L3z_middle*sin(q1+q2+q3),
                    -L2y_middle*cos(q1+q2) - L2z_middle*sin(q1+q2) - L3y_middle*cos(q1+q2+q3) - L3z_middle*sin(q1+q2+q3),
                    -L3y_middle*cos(q1+q2+q3) - L3z_middle*sin(q1+q2+q3)]])

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
    if detJ <= 1e-1:
        mu = (detJ + 1.0)/20
    else:
        mu = 0
    Jinv = W_inv.dot(Jd_trans).dot(1/(detJ + mu**2))
    
    # Proportional Gain
    K = 10

    K_d = 0

    # Distance measured from WeArt 
    dr = distance
    dr_dot = 0

    e = dr - d
    
    #e_dot = (e - e_old)/timestep
    e_dot = 0

    if np.linalg.norm(current_pose - np.array([0, 0, 1])) >= 0.8:
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