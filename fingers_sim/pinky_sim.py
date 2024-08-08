import numpy as np
from math import pi, cos, sin 

def actuation_pinky(distance, q, links_lenghts, R):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]    

    offset_pinky_x = links_lenghts[0]
    offset_pinky_y = links_lenghts[1]
    offset_pinky_z = links_lenghts[2]
    L1y_pinky = links_lenghts[3]
    L1z_pinky = links_lenghts[4]
    L2y_pinky = links_lenghts[5]
    L2z_pinky = links_lenghts[6]
    L3y_pinky = links_lenghts[7]
    L3z_pinky = links_lenghts[8]
    
    #print(np.linalg.norm(current_pose - np.array([0, 0, 1])))

    py_j2 = - L1y_pinky*cos(q1) - L1z_pinky*sin(q1)
    pz_j2 = - L1y_pinky*sin(q1) + L1z_pinky*cos(q1)
        
    py_j3 = py_j2 - L2y_pinky*cos(q1+q2) - L2z_pinky*sin(q1+q2) 
    pz_j3 = pz_j2 - L2y_pinky*sin(q1+q2) + L2z_pinky*cos(q1+q2)

    py_tip = py_j3 - L3y_pinky*cos(q1+q2+q3) - L3z_pinky*sin(q1+q2+q3) 
    pz_tip = pz_j3 - L3y_pinky*sin(q1+q2+q3) + L3z_pinky*cos(q1+q2+q3)

    pos = np.array([[0], [py_tip], [pz_tip + 0.03]])    
    
    p = R.dot(pos) + np.array([[offset_pinky_x], [offset_pinky_y], [offset_pinky_z]])
    # Task Jacobian: p_dot = Jp(q)q_dot
    Jp = np.array([[0, 0, 0],
                   [L1y_pinky*sin(q1) - L1z_pinky*cos(q1) + L2y_pinky*sin(q1+q2) - L2z_pinky*cos(q1+q2) + L3y_pinky*sin(q1+q2+q3) - L3z_pinky*cos(q1+q2+q3), 
                    L2y_pinky*sin(q1+q2) - L2z_pinky*cos(q1+q2) + L3y_pinky*sin(q1+q2+q3) - L3z_pinky*cos(q1+q2+q3), 
                    L3y_pinky*sin(q1+q2+q3) - L3z_pinky*cos(q1+q2+q3)], 
                    [-L1y_pinky*cos(q1) - L1z_pinky*sin(q1) - L2y_pinky*cos(q1+q2) - L2z_pinky*sin(q1+q2) - L3y_pinky*cos(q1+q2+q3) - L3z_pinky*sin(q1+q2+q3),
                    -L2y_pinky*cos(q1+q2) - L2z_pinky*sin(q1+q2) - L3y_pinky*cos(q1+q2+q3) - L3z_pinky*sin(q1+q2+q3),
                    -L3y_pinky*cos(q1+q2+q3) - L3z_pinky*sin(q1+q2+q3)]])
    
    Jp = R.dot(Jp)
    # Distance from origin d
    d = np.linalg.norm(p)

    W = [[6.5, 0,  0], 
         [0,  5.5,  0], 
         [0,  0, 7.5]]

    #print(d - distance)

    # Extended jacobian for distance: d_dot = Jd(q)q_dot
    Jd = 1/d*p.transpose().dot(Jp)

    W_inv = np.linalg.inv(W)

    Jd_trans = Jd.transpose()
    detJ = np.float64(Jd.dot(W_inv.dot(Jd_trans)))
    if detJ <= 1e-4:
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

def move_pinky(distance,data,model,joint_ids,palm,links,R, hand):
    off = 7 if hand=="Right_" else 14
    q1 = data.qpos[joint_ids[hand+'Pinky_J1']+off]
    q2 = data.qpos[joint_ids[hand+'Pinky_J2']+off]
    q3 = data.qpos[joint_ids[hand+'Pinky_J3']+off]
    
    L1y_pinky = links[0] 
    L1z_pinky = links[1] 
    L2y_pinky = links[2] 
    L2z_pinky = links[3] 
    L3y_pinky = links[4] 
    L3z_pinky = links[5]
    
    offset_pinky =  data.xpos[model.body(hand+'Pinky_J1.stl').id] 
    offx_pinky = offset_pinky[0] - palm[0] 
    offy_pinky = offset_pinky[1] - palm[1] 
    offz_pinky = offset_pinky[2] - palm[2]

    links = [offx_pinky, offy_pinky, offz_pinky, L1y_pinky, L1z_pinky, L2y_pinky, L2z_pinky, L3y_pinky, L3z_pinky]
    
    qdot = actuation_pinky(distance, [q1, q2, q3], links, R)

    v1 = qdot[0][0] 
    v2 = qdot[1][0] 
    v3 = qdot[2][0] 

    data.ctrl[joint_ids[hand+'Pinky_J1']] = v1
    data.ctrl[joint_ids[hand+'Pinky_J2']] = v2
    data.ctrl[joint_ids[hand+'Pinky_J3']] = v3