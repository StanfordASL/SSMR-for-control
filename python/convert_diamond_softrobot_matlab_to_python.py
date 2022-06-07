import os

import numpy as np

from florianworld.utils import convert_matlab_to_numpy


def main():
    print("Converting to matlab...")
    models_dir = os.path.dirname(__file__)
    matlab_dir = os.path.normpath(os.path.join(models_dir, "../ROM/gen/"))
    output_dir = os.path.normpath(os.path.join(models_dir, "gen"))

    diamond_softrobot_Amat_O3 = os.path.join(matlab_dir, "diamond_softrobot_Amat_O3.m")
    diamond_softrobot_f_reduced_O3 = os.path.join(matlab_dir, "diamond_softrobot_f_reduced_O3.m")
    diamond_softrobot_Bmat_O3 = os.path.join(matlab_dir, "diamond_softrobot_Bmat_O3.m")
    diamond_softrobot_W_O3 = os.path.join(matlab_dir, "diamond_softrobot_W_O3.m")
    diamond_softrobot_C_O3 = os.path.join(matlab_dir, "diamond_softrobot_C_O3.m")
    #
    # convert_matlab_to_numpy(diamond_softrobot_Amat_O3, output_dir)
    # convert_matlab_to_numpy(diamond_softrobot_f_reduced_O3, output_dir)
    # convert_matlab_to_numpy(diamond_softrobot_Bmat_O3, output_dir)
    # convert_matlab_to_numpy(diamond_softrobot_W_O3, output_dir)
    # convert_matlab_to_numpy(diamond_softrobot_C_O3, output_dir)
    print("Conversion completed")

    print("Testing converted functions")

    from gen.diamond_softrobot_Amat_O3 import diamond_softrobot_Amat_O3
    from gen.diamond_softrobot_f_reduced_O3 import diamond_softrobot_f_reduced_O3
    from gen.diamond_softrobot_Bmat_O3 import diamond_softrobot_Bmat_O3
    from gen.diamond_softrobot_W_O3 import diamond_softrobot_W_O3
    from gen.diamond_softrobot_C_O3 import diamond_softrobot_C_O3

    diamond_softrobot_Amat = diamond_softrobot_Amat_O3
    diamond_softrobot_f_reduced = diamond_softrobot_f_reduced_O3
    diamond_softrobot_Bmat = diamond_softrobot_Bmat_O3
    diamond_softrobot_W = diamond_softrobot_W_O3
    diamond_softrobot_C = diamond_softrobot_C_O3


    p = np.ones((6, 1))
    u = np.ones((4, 1))

    print("Amat(p=1) = ")
    print(diamond_softrobot_Amat(p))
    print("f_reduced(p=1, u=1) = ")
    print(diamond_softrobot_f_reduced(p, u))
    print("Bmat(p=1) = ")
    print(diamond_softrobot_Bmat(p))
    print("W(p=1) = ")
    print(diamond_softrobot_W(p))
    print("C(p=1) = ")
    print(diamond_softrobot_C(p))





if __name__ == '__main__':
    main()
