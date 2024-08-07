import numpy as np
from math import pi, cos, sin 

def actuation_index(distance, q, links_lenghts,R):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2] 

    offset_index_x = links_lenghts[0]
    offset_index_y = links_lenghts[1]
    offset_index_z = links_lenghts[2]
    L1y_index = links_lenghts[3]
    L1z_index = links_lenghts[4]
    L2y_index = links_lenghts[5]
    L2z_index = links_lenghts[6]
    L3y_index = links_lenghts[7]
    L3z_index = links_lenghts[8]
    
    py_j2 = - L1y_index*cos(q1) - L1z_index*sin(q1)
    pz_j2 = - L1y_index*sin(q1) + L1z_index*cos(q1)
    
    py_j3 = py_j2 - L2y_index*cos(q1+q2) - L2z_index*sin(q1+q2) 
    pz_j3 = pz_j2 - L2y_index*sin(q1+q2) + L2z_index*cos(q1+q2)

    py_tip = py_j3 - L3y_index*cos(q1+q2+q3) - L3z_index*sin(q1+q2+q3) 
    pz_tip = pz_j3 - L3y_index*sin(q1+q2+q3) + L3z_index*cos(q1+q2+q3) 

    pos = np.array([[0], [py_tip], [pz_tip + 0.02]])  

    p = R.dot(pos) + np.array([[offset_index_x], [offset_index_y], [offset_index_z]])

    # Task Jacobian: p_dot = Jp(q)q_dot
    Jp = np.array([[0,0,0],
                    [L1y_index*sin(q1) - L1z_index*cos(q1) + L2y_index*sin(q1+q2) - L2z_index*cos(q1+q2) + L3y_index*sin(q1+q2+q3) - L3z_index*cos(q1+q2+q3), 
                    L2y_index*sin(q1+q2) - L2z_index*cos(q1+q2) + L3y_index*sin(q1+q2+q3) - L3z_index*cos(q1+q2+q3), 
                    L3y_index*sin(q1+q2+q3) - L3z_index*cos(q1+q2+q3)], 
                    [-L1y_index*cos(q1) - L1z_index*sin(q1) - L2y_index*cos(q1+q2) - L2z_index*sin(q1+q2) - L3y_index*cos(q1+q2+q3) - L3z_index*sin(q1+q2+q3),
                    -L2y_index*cos(q1+q2) - L2z_index*sin(q1+q2) - L3y_index*cos(q1+q2+q3) - L3z_index*sin(q1+q2+q3),
                    -L3y_index*cos(q1+q2+q3) - L3z_index*sin(q1+q2+q3)]])
    
    Jp = R.dot(Jp)

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
    #print(detJ)
    if detJ <= 1e-5:
        mu = (detJ + 1.0)/1000
    else:
        mu = 0
    Jinv = W_inv.dot(Jd_trans).dot(1/(detJ + mu**2))
    
    # Proportional Gain
    K = 100

    # Distance measured from WeArt 
    dr = distance
    

    w = np.array([(5*pi/16 - q1)/(3*pi/8), (3*pi/8 - q2)/(pi/4), (5*pi/16 - q3)/(3*pi/8)])
    u_vinc = np.reshape((np.eye(3) - Jinv.dot(Jd)).dot(w.T),[3,1])
    
    qdot = Jinv*K*(dr - d) + u_vinc
    return qdot


def move_index(distance,data,model,joint_ids,palm,links,R,hand):
    off = 7 if hand=="Right_" else 14
    q1 = data.qpos[joint_ids[hand+'Index_J1']+off]
    q2 = data.qpos[joint_ids[hand+'Index_J2']+off]
    q3 = data.qpos[joint_ids[hand+'Index_J3']+off]
    L1y_index = links[0] 
    L1z_index = links[1] 
    L2y_index = links[2] 
    L2z_index = links[3] 
    L3y_index = links[4] 
    L3z_index = links[5]

    #print(np.linalg.norm(current_pose_index - np.array([0, 0, 1])))

    offset_index =  data.xpos[model.body(hand+'Index_J1.stl').id] 
    offx_index = offset_index[0] - palm[0]
    offy_index = offset_index[1] - palm[1]
    offz_index = offset_index[2] - palm[2]

    links = [offx_index, offy_index, offz_index, L1y_index, L1z_index, L2y_index, L2z_index, L3y_index, L3z_index]

    qdot = actuation_index(distance, [q1, q2, q3], links,R)

    v1 = qdot[0][0] 
    v2 = qdot[1][0] 
    v3 = qdot[2][0] 

    data.ctrl[joint_ids[hand+'Index_J1']] = v1
    data.ctrl[joint_ids[hand+'Index_J2']] = v2
    data.ctrl[joint_ids[hand+'Index_J3']] = v3