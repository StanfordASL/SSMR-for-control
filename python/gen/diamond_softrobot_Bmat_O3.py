import numpy as np

def diamond_softrobot_Bmat_O3(in1):
    #diamond_softrobot_Bmat_O3
    #    B_mat = diamond_softrobot_Bmat_O3(IN1)
    
    #    This function was generated by the Symbolic Math Toolbox version 9.0.
    #    07-Jun-2022 22:34:09
    
    mt1 = np.array([0.0,0.0,0.0,-6.438668309935568e+1,-8.020081222603156e+1,1.647850231855044e+1,0.0,0.0,0.0,-8.596353237564298e+1,5.007862554292316e+1,3.402583494640329e+1,0.0,0.0,0.0])
    mt2 = np.array([-8.844126815338683,7.261452954276311e+1,-5.262805850636806e+1,0.0,0.0,0.0,-7.52561091863776,-3.80219384164585e+1,-9.688298583928201e+1])
    B_mat = np.reshape(np.hstack([mt1,mt2]), (4, 6)).T
    return B_mat
