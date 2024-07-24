import numpy as np
from math import pi, cos, sin, sqrt 

def actuation_thumb(distance, q, current_pose, links_lenghts, integral, timestep):
    q1 = q[0]
    q2 = q[1]
    #q3 = q[2] 
    dr_dot = 0

    offset_thumb_x = links_lenghts[0]
    offset_thumb_y = links_lenghts[1]
    L2x_thumb = links_lenghts[2]
    L2y_thumb = links_lenghts[3]
    L3x_thumb = links_lenghts[4]
    L3y_thumb = links_lenghts[5]
    
    px_j2 = offset_thumb_x + L2x_thumb*cos(q1) - L2y_thumb*sin(q1)
    py_j2 = offset_thumb_y - L2x_thumb*sin(q1) - L2y_thumb*cos(q1)
    
    px_tip = px_j2 + L3x_thumb*cos(q1+q2) - L3y_thumb*sin(q1+q2) + 0.2
    py_tip = py_j2 - L3x_thumb*sin(q1+q2) - L3y_thumb*cos(q1+q2) + 0.35

    p = np.array([[px_tip], [py_tip], [0.13]])

    
    # Task Jacobian: p_dot = Jp(q)q_dot
    Jp = np.array([[- L2x_thumb*sin(q1) - L2y_thumb*cos(q1) - L3x_thumb*sin(q1+q2) - L3y_thumb*cos(q1+q2),
                    - L3x_thumb*sin(q1+q2) - L3y_thumb*cos(q1+q2)],
                   [- L2x_thumb*cos(q1) - L2y_thumb*sin(q1) - L3x_thumb*cos(q1+q2) + L3y_thumb*sin(q1+q2),
                    - L3x_thumb*cos(q1+q2) + L3y_thumb*sin(q1+q2)], 
                   [0, 0]])
    
    # Distance from origin d
    d = np.linalg.norm(p)

    W = [[1,  0], 
         [0,  1]]

    #print(d, distance)

    # Extended jacobian for distance: d_dot = Jd(q)q_dot
    Jd = 1/d*p.transpose().dot(Jp)

    W_inv = np.linalg.inv(W)

    Jd_trans = Jd.transpose()
    detJ = np.float64(Jd.dot(W_inv.dot(Jd_trans)))
    if detJ <= 1e-1:
        mu = (detJ + 1.0)/10
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
    
    if np.linalg.norm(current_pose - np.array([0, 0, 1])) >= 0.775:
        K_i = 0
        integral += 0
    else:
        integral += dr - d
        if abs(integral) < 1e-4:
            integral = 0
        K_i = 0.1
    K_i  = 0
    
    w = np.array([(pi/2 - q1)/(2*pi/3), (pi/3 - q2)/(pi/3)])
    u_vinc = (np.eye(2) - Jinv.dot(Jd))*w.transpose()
    qdot = Jinv.dot(dr_dot + K*(dr - d) + K_i*integral + K_d*e_dot) + u_vinc

    return qdot, integral, d