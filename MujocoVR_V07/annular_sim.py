import numpy as np
from math import pi, cos, sin 

def actuation_annular(distance, q, links_lenghts,R):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]    

    offset_annular_x = links_lenghts[0]
    offset_annular_y = links_lenghts[1]
    offset_annular_z = links_lenghts[2]
    L1y_annular = links_lenghts[3]
    L1z_annular = links_lenghts[4]
    L2y_annular = links_lenghts[5]
    L2z_annular = links_lenghts[6]
    L3y_annular = links_lenghts[7]
    L3z_annular = links_lenghts[8]

    #print(np.linalg.norm(current_pose - np.array([0, 0, 1])))

    py_j2 =  -L1y_annular*cos(q1) - L1z_annular*sin(q1)
    pz_j2 =  -L1y_annular*sin(q1) + L1z_annular*cos(q1)
        
    py_j3 = py_j2 - L2y_annular*cos(q1+q2) - L2z_annular*sin(q1+q2) 
    pz_j3 = pz_j2 - L2y_annular*sin(q1+q2) + L2z_annular*cos(q1+q2)

    py_tip = py_j3 - L3y_annular*cos(q1+q2+q3) - L3z_annular*sin(q1+q2+q3)
    pz_tip = pz_j3 - L3y_annular*sin(q1+q2+q3) + L3z_annular*cos(q1+q2+q3)

    pos = np.array([[0], [py_tip], [pz_tip  + 0.03]])   
    p = R.dot(pos) + np.array([[offset_annular_x], [offset_annular_y], [offset_annular_z]])
    
    # Task Jacobian: p_dot = Jp(q)q_dot
    Jp = np.array([[0,0,0],
                   [L1y_annular*sin(q1) - L1z_annular*cos(q1) + L2y_annular*sin(q1+q2) - L2z_annular*cos(q1+q2) + L3y_annular*sin(q1+q2+q3) - L3z_annular*cos(q1+q2+q3), 
                    L2y_annular*sin(q1+q2) - L2z_annular*cos(q1+q2) + L3y_annular*sin(q1+q2+q3) - L3z_annular*cos(q1+q2+q3), 
                    L3y_annular*sin(q1+q2+q3) - L3z_annular*cos(q1+q2+q3)], 
                    [-L1y_annular*cos(q1) - L1z_annular*sin(q1) - L2y_annular*cos(q1+q2) - L2z_annular*sin(q1+q2) - L3y_annular*cos(q1+q2+q3) - L3z_annular*sin(q1+q2+q3),
                    -L2y_annular*cos(q1+q2) - L2z_annular*sin(q1+q2) - L3y_annular*cos(q1+q2+q3) - L3z_annular*sin(q1+q2+q3),
                    -L3y_annular*cos(q1+q2+q3) - L3z_annular*sin(q1+q2+q3)]])

    Jp = R.dot(Jp)
    
    # Distance from origin d
    d = np.linalg.norm(p)

    W = [[9.5, 0,  0], 
         [0,  5.5,  0], 
         [0,  0, 7.5]]

    #print(d - distance)

    # Extended jacobian for distance: d_dot = Jd(q)q_dot
    Jd = 1/d*p.transpose().dot(Jp)

    W_inv = np.linalg.inv(W)

    Jd_trans = Jd.transpose()
    detJ = np.float64(Jd.dot(W_inv.dot(Jd_trans)))
    if detJ <= 1e-6:
        mu = (detJ + 1.0)/2000
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

def move_annular(distance,data,model,joint_ids,palm,links,R, hand):
    off = 7 if hand=="Right_" else 14
    q1 = data.qpos[joint_ids[hand+'Annular_J1']+off]
    q2 = data.qpos[joint_ids[hand+'Annular_J2']+off]
    q3 = data.qpos[joint_ids[hand+'Annular_J3']+off]
    
    L1y_annular = links[0] 
    L1z_annular = links[1] 
    L2y_annular = links[2] 
    L2z_annular = links[3] 
    L3y_annular = links[4] 
    L3z_annular = links[5]
    
  
    offset_annular =  data.xpos[model.body(hand+'Annular_J1.stl').id] 
    offx_annular =  offset_annular[0] - palm[0]
    offy_annular = offset_annular[1] - palm[1] 
    offz_annular = offset_annular[2] - palm[2]

    links = [offx_annular,offy_annular, offz_annular, L1y_annular, L1z_annular, L2y_annular, L2z_annular, L3y_annular, L3z_annular]
    
    qdot = actuation_annular(distance, [q1, q2, q3], links,R)

    v1 = qdot[0][0] 
    v2 = qdot[1][0] 
    v3 = qdot[2][0] 

    data.ctrl[joint_ids[hand+'Annular_J1']] = v1
    data.ctrl[joint_ids[hand+'Annular_J2']] = v2
    data.ctrl[joint_ids[hand+'Annular_J3']] = v3