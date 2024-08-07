import numpy as np
from math import pi, cos, sin 

def actuation_thumb(distance, q, links_lenghts, R, hand_side):
    q1 = q[0]
    q2 = q[1] 

    offset_thumb_x = links_lenghts[0]
    offset_thumb_y = links_lenghts[1]
    offset_thumb_z = links_lenghts[2]
    L2x_thumb = links_lenghts[3]
    L2y_thumb = links_lenghts[4]
    L3x_thumb = links_lenghts[5]
    L3y_thumb = links_lenghts[6]
    
    px_j2 =  L2x_thumb*cos(q1) - L2y_thumb*sin(q1)
    py_j2 = - L2x_thumb*sin(q1) - L2y_thumb*cos(q1)
    
    px_tip = px_j2 + L3x_thumb*cos(q1+q2) - L3y_thumb*sin(q1+q2) 
    py_tip = py_j2 - L3x_thumb*sin(q1+q2) - L3y_thumb*cos(q1+q2) 

    pos = np.array([[px_tip + 0.05], [py_tip], [0]])
    pos *= hand_side 

    p = R.dot(pos) + np.array([[offset_thumb_x], [offset_thumb_y], [offset_thumb_z]])
    
    # Task Jacobian: p_dot = Jp(q)q_dot
    Jp = np.array([[- L2x_thumb*sin(q1) - L2y_thumb*cos(q1) - L3x_thumb*sin(q1+q2) - L3y_thumb*cos(q1+q2),
                    - L3x_thumb*sin(q1+q2) - L3y_thumb*cos(q1+q2)],
                   [- L2x_thumb*cos(q1) - L2y_thumb*sin(q1) - L3x_thumb*cos(q1+q2) + L3y_thumb*sin(q1+q2),
                    - L3x_thumb*cos(q1+q2) + L3y_thumb*sin(q1+q2)],
                    [0,0]])
    
    Jp = R.dot(Jp)
    Jp *= hand_side 
    # Distance from origin d
    d = np.linalg.norm(p)

    W = [[0.5,  0], 
         [0,  0.5]]

    

    # Extended jacobian for distance: d_dot = Jd(q)q_dot
    Jd = 1/d*p.transpose().dot(Jp)

    W_inv = np.linalg.inv(W)

    Jd_trans = Jd.transpose()
    detJ = np.float64(Jd.dot(W_inv.dot(Jd_trans)))
    if detJ <= 1e-3:
        mu = (detJ + 1.0)/20
        # print("sing", detJ)
    else:
        mu = 0
    Jinv = W_inv.dot(Jd_trans)
    Jinv = Jinv * (1/(detJ + mu**2))
    
    #print(d, distance) 

    # Proportional Gain
    K = 100

    # Distance measured from WeArt 
    dr = distance

    w = np.array([(pi/4 - q1)/(pi/2), (pi/4 - q2)/(pi/2)])
    
    u_vinc = np.reshape((np.eye(2) - Jinv.dot(Jd)).dot(w.T),[2,1])
    qdot = Jinv*K*(dr - d) + u_vinc
    # print((qdot))
    return qdot

def actuation_thumb_abd(abduction, q1,Ly,Lz):

    z = - Ly*sin(q1) + Lz*cos(q1)-0.01
    Jz = -Ly*cos(q1)-Lz*sin(q1)

    detJ = Jz**2
    if detJ <= 1e-3:
        mu = (detJ + 1.0)/50
    else:
        mu = 0
    Jinv = Jz/(detJ + mu**2)
    
    # Proportional Gain
    K = 50

    # Distance measured from WeArt 
    dr_dot = 0
  
    qdot = Jinv*( dr_dot + K*(abduction - z))

    return qdot    

def move_thumb(distance,abduction,data,model,joint_ids,palm,links,R, hand_side):
    hand = 'Right_' if hand_side==1 else 'Left_'

    off = 7 if hand=="Right_" else 14
    q1 = data.qpos[joint_ids[hand+'Thumb_J2']+off]
    q2 = data.qpos[joint_ids[hand+'Thumb_J3']+off]

    L2x_thumb = links[0] 
    L2y_thumb = links[1] 
    L3x_thumb = links[2] 
    L3y_thumb = links[3]


    q = data.qpos[joint_ids[hand+'Thumb_J1']+off]
    Ly = links[4]
    Lz = links[5]

    offset_thumb =  data.xpos[model.body(hand+'Thumb_J1.stl').id] 
    offx_thumb = offset_thumb[0] - palm[0] 
    offy_thumb = offset_thumb[1] - palm[1]
    offz_thumb = offset_thumb[2] - palm[2]

    links = [offx_thumb, offy_thumb, offz_thumb, L2x_thumb, L2y_thumb, L3x_thumb, L3y_thumb]

    Rq = np.array([[1,0,0],[0, cos(q), -sin(q)],[0,sin(q),cos(q)]])
    R = R.dot(Rq)
    qdot = actuation_thumb(distance, [q1, q2], links,R, hand_side)

    v1 = qdot[0][0]
    v2 = qdot[1][0]

    #print(abduction)  
    data.ctrl[joint_ids[hand + 'Thumb_J1']] = actuation_thumb_abd(abduction, q,Ly,Lz)
    data.ctrl[joint_ids[hand + 'Thumb_J2']] = v1
    data.ctrl[joint_ids[hand + 'Thumb_J3']] = v2