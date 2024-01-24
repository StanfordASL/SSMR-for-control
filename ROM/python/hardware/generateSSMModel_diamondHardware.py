import numpy as np
from copy import deepcopy
from os.path import join, exists, isdir
from os import listdir, mkdir
import matplotlib.pyplot as plt
from time import time
import yaml
import pickle
from tqdm.auto import tqdm

from os.path import dirname, abspath, join

import sys
from scipy.interpolate import interp1d
import ROM.python.utils as utils
import ROM.python.plot_utils as plot


path = dirname(abspath(__file__))
root = dirname(path)
sys.path.append(root)


np.set_printoptions(linewidth=300)


SETTINGS = {
    'observables': "delay-embedding", # "pos-vel", # "delay-embedding"
    'reduced_coordinates': "local", # "global" # "local"
    'use_ssmlearn': "py", # "matlab", "py"

    'robot_dir': "/home/jalora/soft-robot-control/examples/hardware",
    'tip_node': 1354,
    'n_nodes': 1628,
    'input_dim': 4,
    
    'dt': 0.01,
    'subsample': 1,

    'rDOF': 3,
    'oDOF': 3,
    'n_delay': 7,
    'SSMDim': 6,
    'SSMOrder': 3,
    'ROMOrder': 3,
    'RDType': "flow",
    'ridge_alpha': {
        'manifold': 0., # 1.,
        'reduced_dynamics': 0., # 100.,
        'B': 0. # 1.
    },
    'custom_delay': None, # Specify a custom delay embedding for the SSM

    'data_dir': "/home/jalora/SSMR-for-control/ROM/python/hardware",
    # 'data_dir': "/home/jalora/Desktop/diamond_origin",
    'data_subdirs': False,
    'decay_dir': "decay/",
    'rest_file': "rest_qv.pkl",
    'model_save_dir': "SSMmodelsPy/",

    't_decay': [1, 4],
    't_truncate': [0.05, np.inf],

    'decay_test_set': [18, 19, 33],
    'decay_holdout_set': [11],

    'traj_test_set': [],
    'traj_holdout_set': [0, 1, 2], #0, 1, 2

    'poly_u_order': 1,
    'input_train_data_dir': "open-loop",
    'input_test_data_dir': [],
    'input_train_ratio': 0.8 # 0.8
}
SETTINGS['data_subdirs'] = sorted([dir for dir in listdir(SETTINGS['data_dir']) if isdir(join(SETTINGS['data_dir'], dir)) and
                                   'decay' in listdir(join(SETTINGS['data_dir'], dir)) and
                                   'open-loop' in listdir(join(SETTINGS['data_dir'], dir))])
print(SETTINGS['data_subdirs'])

PLOTS = 'save' # 'save', 'show', False ('show' shows and saves plots, 'save' only saves plots, False does nothing)

# observables are position of tip + n_delay delay embeddings of the tip position
output_node = SETTINGS['tip_node']
assemble_observables = lambda oData: utils.delayEmbedding(oData, embed_coords=np.arange(3), up_to_delay=SETTINGS['n_delay'])
svd_data = 'yDataTrunc'


def generate_ssmr_model(data_dir, save_model_to_data_dir=False):
    """
    Generates a complete SSMR model from the system data in data_dir
    Note: data_dir is assumed to contain two subfolders, containing decay data and open-loop training data, respectively
    """
    print(f"###### Generate SSM model from data in {data_dir} ######")
    start_time = time()

    # create a new directory into which the new model will be saved and get the equilibrium state
    model_save_dir = path
    decay_data_dir = path
    q_eq = np.zeros(SETTINGS['oDOF'])
    u_eq = np.zeros(4)

    obsNodes = [12, 13, 14]
    outdofs = [3, 4, 5]
    idxTrajRegressC = 10

    # ====== Import decay trajectories -- oData ====== #
    print("====== Import decay trajectories ======")    
    Data = {}
    # For /home/jjalora/Desktop/Diamond
    Data['oData'] = utils.import_pos_data(decay_data_dir, 
                                          q_rest=q_eq, 
                                          output_node=obsNodes, 
                                          file_type='mat', 
                                          return_velocity=False,
                                          hardware=True)
    
    nTRAJ = len(Data['oData'])

    # ====== assemble observables for each decay trajectory -- yData ====== #
    Data['yData'] = deepcopy(Data['oData'])
    for i in range(nTRAJ):
        Data['yData'][i][1] = assemble_observables(Data['oData'][i][1])

    if PLOTS:
        utils.plotDecayXYZData(SETTINGS, model_save_dir, Data, outdofs, PLOTS=PLOTS)


    # ====== Truncate decay trajectories ====== #
    Data['oDataTrunc'] = utils.slice_trajectories(Data['oData'], SETTINGS['t_truncate'], t_shift=SETTINGS['n_delay'] * SETTINGS['dt'])
    Data['yDataTrunc'] = utils.slice_trajectories(Data['yData'], SETTINGS['t_truncate'], t_shift=SETTINGS['n_delay'] * SETTINGS['dt'])

    # ====== Get Tangent Space and include reduced coords in Data ====== #
    
    Vde, Data = utils.getChartandReducedCoords(SETTINGS, model_save_dir, Data, svd_data, PLOTS)
    # Vde = None

    # ====== Train/test split for decay data ====== #
    indTest = SETTINGS['decay_test_set']
    indHoldout = SETTINGS['decay_holdout_set']
    indTrain = [i for i in range(nTRAJ) if i not in [*indTest, *indHoldout]]

    # ====== Learn SSM geometry and reduced dynamics ====== #
    print("====== Learn SSM geometry and reduced dynamics ======")
    # SSM geometry and reduced dynamics
    IMInfo, RDInfo, Data = utils.learnIMandRD(SETTINGS, Data, indTrain, custom_delay=SETTINGS['custom_delay'])

    # Sets the new observables for plotting
    if SETTINGS['custom_delay']:
        outdofs = utils.mapInt2List(SETTINGS['custom_delay'])
    
    # ====== Analyze autonomous SSM: accuracy of geometry and reduced dynamics ====== #
    if PLOTS:
        print("====== Analyze autonomous SSM: accuracy of geometry and reduced dynamics ======")
        utils.analyzeSSMErrors(model_save_dir, IMInfo, RDInfo, Data, indTrain, indTest, outdofs, PLOTS)

    # ===== Influence of control ====== #
    print("====== Learn influence of control ======")
    # define mappings
    Rauton = lambda x: RDInfo['reducedDynamics']['coefficients'] @ utils.phi(x, RDInfo['reducedDynamics']['polynomialOrder'])
    Vauton = lambda x: IMInfo['parametrization']['H'] @ utils.phi(x, IMInfo['parametrization']['polynomialOrder'])
    
    if SETTINGS['custom_delay']:
        Wauton = lambda y: IMInfo['chart']['H'] @ utils.phi(y, IMInfo['chart']['polynomialOrder'])
    else:
        Wauton = lambda y: Vde.T @ y

    #### Import and process open loop training data ####
    outData, inputData = utils.import_traj_data(join(data_dir, SETTINGS['input_train_data_dir']), obsNodes)

    y = outData
    for i in range(len(outData)):
        if SETTINGS['custom_delay'] is not None:
            y[i][1] = utils.delayEmbedding(outData[i][1], embed_coords=np.arange(3), up_to_delay=SETTINGS['custom_delay'])
        else:
            y[i][1] = utils.delayEmbedding(outData[i][1], embed_coords=np.arange(3), up_to_delay=SETTINGS['n_delay'])
    
    indCntrlTest = SETTINGS['traj_test_set']
    indCntrlHoldout = SETTINGS['traj_holdout_set']
    indCntrlTrain = [i for i in range(np.shape(outData)[0]) if i not in [*indCntrlTest, *indCntrlHoldout]]

    # Consolidate Train Data
    zData, yData, uData = [], [], []
    for iTraj in indCntrlTrain:
        yData.append(y[iTraj][1])
        uData.append(inputData[iTraj][1])
        zData.append(outData[iTraj][1])
    
    yDataFull = np.concatenate(yData, axis=1)
    uDataFull = np.concatenate(uData, axis=1)
    zDataFull = np.concatenate(zData, axis=1)
    xFull = Wauton(yDataFull)
    tFull = np.arange(0, np.shape(yDataFull)[1]*SETTINGS['dt'], SETTINGS['dt'])

    # train/test split on input training data
    split_idx = int(SETTINGS['input_train_ratio'] * len(tFull))
    t_train, t_test = tFull[:split_idx], tFull[split_idx:]
    _, z_test = zDataFull[:, :split_idx], zDataFull[:, split_idx:]
    # y_train, y_test = y[:, :split_idx], y[:, split_idx:]
    u_train, u_test = uDataFull[:, :split_idx], uDataFull[:, split_idx:]
    x_train, x_test = xFull[:, :split_idx], xFull[:, split_idx:]

    controlData = {'t_train': t_train, 't_test': t_test, 'z_test': z_test, 'u_train': u_train, 'u_test': u_test, 
            'x_train': x_train, 'x_test': x_test, 'dxdt': np.gradient(x_train, SETTINGS['dt'], axis=1), 'dxdt_ROM': Rauton(x_train)}

    Br = utils.fitControlMatrix(Rauton, x_train, u_train, SETTINGS['dt'], alpha=SETTINGS['ridge_alpha']['B'])
    # Br = np.array([[-36.2267, -15.8509, -12.2239, -16.6699],
    #                [24.9702, 4.3417, -8.8862, 4.3134],
    #                [-2.4812, 0.6296, 1.4031, 0.3656],
    #                [51.971, 29.0565, -42.8865, -18.6257],
    #                [12.5348, -48.0646, -43.7983, 15.0153],
    #                [-35.9809, -16.2992, -37.931, -52.1558]])

    R = lambda x, u: Rauton(np.atleast_2d(x)) + Br @ utils.phi(u, 1)

    
    # ====== Test open-loop prediction capabalities of SSM model ====== #

    print("====== Test open-loop prediction capabalities of SSM model ======")
    # Consolidate Test Data
    zDataTest, yDataTest, uDataTest, tDataTest = [], [], [], []
    for iTraj in indCntrlTest:
        tDataTest.append(y[iTraj][0])
        zDataTest.append(outData[iTraj][1])
        yDataTest.append(y[iTraj][1])
        uDataTest.append(inputData[iTraj][1])
    
    test_data = (tDataTest, zDataTest, yDataTest, uDataTest)
    traj_outDofs = [0, 1, 2]
    test_results = utils.analyze_open_loop(test_data, model_save_dir, controlData, Wauton, R, Vauton, embed_coords=traj_outDofs, 
                            traj_coords=[0, 1, 2], PLOTS='save')

    if PLOTS:
        # # plot training data
        # plot.traj_xyz_txyz(t=t,
        #                 x=z[traj_outDofs[0], :], y=z[traj_outDofs[1], :], z=z[traj_outDofs[2], :],
        #                 show=(PLOTS == 'show'))
        # plt.savefig(join(model_save_dir, "plots", f"input_training_data_z.png"), bbox_inches='tight')
        # plot.inputs(t, u, show=(PLOTS == 'show'))
        # if PLOTS == 'save':
        #     plt.savefig(join(model_save_dir, "plots", f"input_training_data_u.png"), bbox_inches='tight')
            
        # plot gradients (numerical, predicted autonomous, predicted with inputs)
        dxDt_ROM_with_B = R(controlData['x_train'], controlData['u_train'])
        utils.plot_gradients(controlData['t_train'], controlData['dxdt'], controlData['dxdt_ROM'], dxDt_ROM_with_B, PLOTS, model_save_dir)


    # # ====== Save SSM model to folder, along with settings and test results ====== #
    utils.save_ssm_model(model_save_dir, RDInfo, IMInfo, Br, Vde, q_eq, u_eq, SETTINGS, test_results)
    print(f"====== Saved SSM model to folder {model_save_dir} ======")

    # Close all plots for this model
    plt.close('all')
    # Display total runtime
    end_time = time()
    print(f"====== Finished after total runtime: {(end_time - start_time):.2f} sec ======")


if __name__ == "__main__":
    data_dir = path
    generate_ssmr_model(data_dir, save_model_to_data_dir=True)