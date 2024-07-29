import numpy as np
from math import pi, cos, sin, sqrt 

def actuation_thumb(distance, q, current_pose, links_lenghts, palm, R):
    q1 = q[0]
    q2 = q[1] 
    dr_dot = 0

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

    pos = np.array([[px_tip+0.05], [py_tip], [0]])  

    p = R.dot(pos) + np.array([[offset_thumb_x], [offset_thumb_y], [offset_thumb_z]])
    
    # Task Jacobian: p_dot = Jp(q)q_dot
    Jp = np.array([[- L2x_thumb*sin(q1) - L2y_thumb*cos(q1) - L3x_thumb*sin(q1+q2) - L3y_thumb*cos(q1+q2),
                    - L3x_thumb*sin(q1+q2) - L3y_thumb*cos(q1+q2)],
                   [- L2x_thumb*cos(q1) - L2y_thumb*sin(q1) - L3x_thumb*cos(q1+q2) + L3y_thumb*sin(q1+q2),
                    - L3x_thumb*cos(q1+q2) + L3y_thumb*sin(q1+q2)],
                    [0,0]])
    
    Jp = R.dot(Jp)
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
    if detJ <= 1e-3:
        mu = (detJ + 1.0)/50
    else:
        mu = 0
    Jinv = W_inv.dot(Jd_trans).dot(1/(detJ + mu**2))
    
    # Proportional Gain
    K = 70

    # Distance measured from WeArt 
    dr = distance
    dr_dot = 0

    
    if np.linalg.norm(current_pose - palm) >= 0.08:
        K_i = 0
    else:
        K_i = 0.1
    K_i = 0
    
    w = np.array([(pi/2 - q1)/(2*pi/3), (pi/3 - q2)/(pi/3)])
    u_vinc = (np.eye(2) - Jinv.dot(Jd))*w.transpose()
    qdot = Jinv.dot(dr_dot + K*(dr - d)) + u_vinc

    return qdot

def move_thumb(distance,data,model,joint_ids,palm,links,R):
    q1 = data.qpos[joint_ids['Thumb_J2']]
    q2 = data.qpos[joint_ids['Thumb_J3']]
    
    L2x_thumb = links[0] 
    L2y_thumb = links[1] 
    L3x_thumb = links[2] 
    L3y_thumb = links[3]

    offset_thumb =  data.xpos[model.body('Thumb_J1.stl').id] 
    offx_thumb = offset_thumb[0] - palm[0] 
    offy_thumb = offset_thumb[1] - palm[1]
    offz_thumb = offset_thumb[2] - palm[2]

    current_pose_thumb = data.site_xpos[model.site('forSensorThumb_3.stl').id] 

    links = [offx_thumb, offy_thumb, offz_thumb, L2x_thumb, L2y_thumb, L3x_thumb, L3y_thumb]

    qdot = actuation_thumb(distance, [q1, q2], current_pose_thumb, links, palm,R)

    v1 = qdot[0][0]
    v2 = qdot[1][0]

    data.qvel[joint_ids['Thumb_J2']] = v1
    data.qvel[joint_ids['Thumb_J3']] = v2