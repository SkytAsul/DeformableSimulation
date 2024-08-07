import numpy as np
from math import pi, cos, sin 

def actuation_middle(distance, q, links_lenghts, R):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    #print(q1, q2, q3)    
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
    
    #print(np.linalg.norm(current_pose - np.array([0, 0, 1])))

    py_j2 = - L1y_middle*cos(q1) - L1z_middle*sin(q1)
    pz_j2 = - L1y_middle*sin(q1) + L1z_middle*cos(q1)
        
    py_j3 = py_j2 - L2y_middle*cos(q1+q2) - L2z_middle*sin(q1+q2) 
    pz_j3 = pz_j2 - L2y_middle*sin(q1+q2) + L2z_middle*cos(q1+q2)

    py_tip = py_j3 - L3y_middle*cos(q1+q2+q3) - L3z_middle*sin(q1+q2+q3)
    pz_tip = pz_j3 - L3y_middle*sin(q1+q2+q3) + L3z_middle*cos(q1+q2+q3)


    pos = np.array([[0], [py_tip], [pz_tip + 0.03]])  

    p = R.dot(pos) + np.array([[offset_middle_x], [offset_middle_y], [offset_middle_z]])
    
    #print(p.T-current_pose.T)

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
    
    w = np.array([(5*pi/16 - q1)/(3*pi/8), (3*pi/8 - q2)/(pi/4), (5*pi/16 - q3)/(3*pi/8)])
    u_vinc = np.reshape((np.eye(3) - Jinv.dot(Jd)).dot(w.T),[3,1])
    
    qdot = Jinv*K*(dr - d) + u_vinc

    return qdot


def move_middle(distance,data,model,joint_ids,palm,links,R, hand):
    off = 7 if hand=="Right_" else 14
    q1 = data.qpos[joint_ids[hand+'Middle_J1']+off]
    q2 = data.qpos[joint_ids[hand+'Middle_J2']+off]
    q3 = data.qpos[joint_ids[hand+'Middle_J3']+off]
    
    L1y_middle = links[0] 
    L1z_middle = links[1] 
    L2y_middle = links[2] 
    L2z_middle = links[3] 
    L3y_middle = links[4] 
    L3z_middle = links[5]
    
    #end_effector_middle = model.site('forSensorMiddle_4.stl').id #"End-effector we wish to control.
    offset_middle =  data.xpos[model.body(hand+'Middle_J1.stl').id] 
    offx_middle = offset_middle[0] - palm[0]
    offy_middle = offset_middle[1] - palm[1] 
    offz_middle = offset_middle[2] - palm[2]

    #print(offset_middle, current_pose_middle)
    
    #lenght links [offsety, offsetz, l1y, l1z, l2y, l2z, l3y, l3z]
    links = [offx_middle, offy_middle, offz_middle, L1y_middle, L1z_middle, L2y_middle, L2z_middle, L3y_middle, L3z_middle]
    
    qdot = actuation_middle(distance, [q1, q2, q3], links, R)

    v1 = qdot[0][0] 
    v2 = qdot[1][0] 
    v3 = qdot[2][0] 
    #v1 = 15
    #v2 = 15
    #v3 = 15
    
    data.ctrl[joint_ids[hand+'Middle_J1']] = v1
    data.ctrl[joint_ids[hand+'Middle_J2']] = v2
    data.ctrl[joint_ids[hand+'Middle_J3']] = v3
    # print("qpos:" , data.qpos)
