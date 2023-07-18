import numpy as np
from copy import deepcopy
from os.path import join, exists, isdir
from os import listdir, mkdir
import utils as utils
import plot_utils as plot
import matplotlib.pyplot as plt
from time import time
import yaml
import pickle
from tqdm.auto import tqdm

np.set_printoptions(linewidth=300)


SETTINGS = {
    'observables': "delay-embedding", # "delay-embedding", # "pos-vel", #
    'reduced_coordinates': "local", # "global" # "local"

    'use_ssmlearn': "py", # "matlab", "py"

    'robot_dir': "/home/jalora/soft-robot-control/examples/trunk",
    'tip_node': 51,
    'n_nodes': 709,
    'input_dim': 8,
    
    'dt': 0.01,

    'rDOF': 3,
    'oDOF': 3,
    'n_delay': 4,
    'SSMDim': 6,
    'SSMOrder': 3,
    'ROMOrder': 3,
    'RDType': "flow",
    'ridge_alpha': {
        'manifold': 1., # 1.,
        'reduced_dynamics': 100., # 100.,
        'B': 1. # 1.
    },
    'custom_delay': False, # Specify a custom delay embedding for the SSM (False or Integer)


    # 'data_dir': "/media/jonas/Backup Plus/jonas_soft_robot_data/trunk_adiabatic_10ms_N=100_sparsity=0.95", # 33_handcrafted/",
    'data_dir': "/home/jalora/Desktop/trunk_origin",
    # 'data_subdirs': [
    #     "origin",
    #     "north",
    #     "west",
    #     "south",
    #     "east",
    #     "northwest",
    #     "southwest",
    #     "southeast",
    #     "northeast"
    # ],
    'decay_dir': "decay/",
    'rest_file': "rest_qv.pkl",
    'model_save_dir': "SSMmodels/",

    't_decay': [1, 4],
    't_truncate': [0.0, 3],

    # 'decay_test_set': [0, 8, 16, 24],
    'decay_test_set': [0, 4],

    'poly_u_order': 1,
    'input_train_data_dir': "open-loop",
    'input_test_data_dir': [],
    'input_train_ratio': 0.8
}
SETTINGS['data_subdirs'] = sorted([dir for dir in listdir(SETTINGS['data_dir']) if isdir(join(SETTINGS['data_dir'], dir)) and
                                   'decay' in listdir(join(SETTINGS['data_dir'], dir)) and
                                   'open-loop' in listdir(join(SETTINGS['data_dir'], dir))])
print(SETTINGS['data_subdirs'])

PLOTS = 'save' # 'save', 'show', False ('show' shows and saves plots, 'save' only saves plots, False does nothing)

if SETTINGS['observables'] == "delay-embedding":
    # observables are position of tip + n_delay delay embeddings of the tip position
    output_node = SETTINGS['tip_node']
    assemble_observables = lambda oData: utils.delayEmbedding(oData, up_to_delay=SETTINGS['n_delay'])
    svd_data = 'yDataTrunc'
elif SETTINGS['observables'] == "pos-vel":
    # observables is position and velocity of tip node
    output_node = 'all'
    def assemble_observables(oData):
        if oData.shape[0] > 6:
            tip_node_slice = np.s_[3*SETTINGS['tip_node']:3*SETTINGS['tip_node']+3]
        else:
            tip_node_slice = np.s_[:3]
        return np.vstack((oData[tip_node_slice, :], np.gradient(oData[tip_node_slice, :], SETTINGS['dt'], axis=1)))
    svd_data = 'oDataTrunc'
else:
    raise RuntimeError("Unknown type of observables, should be ['delay-embedding', 'pos-vel']")


def generate_ssmr_model(data_dir, save_model_to_data_dir=False):
    """
    Generates a complete SSMR model from the system data in data_dir
    Note: data_dir is assumed to contain two subfolders, containing decay data and open-loop training data, respectively
    """
    print(f"###### Generate SSM model from data in {data_dir} ######")
    start_time = time()

    # create a new directory into which the new model will be saved and get the equilibrium state
    model_save_dir, decay_data_dir, q_eq, u_eq = utils.setupDirandRestParams(SETTINGS, data_dir, save_model_to_data_dir=save_model_to_data_dir, PLOTS=PLOTS)
    outdofs = [0, 1, 2]

    # ====== Import decay trajectories -- oData ====== #
    print("====== Import decay trajectories ======")    
    Data = {}

    # For /home/jjalora/Desktop/trunk_origin
    Data['oData'] = utils.import_pos_data(decay_data_dir,
                                        rest_file=None, # join(SETTINGS['robot_dir'], SETTINGS['rest_file']),
                                        q_rest=q_eq,
                                        output_node=output_node,
                                        t_in=SETTINGS['t_decay'][0], t_out=SETTINGS['t_decay'][1])
    
    nTRAJ = len(Data['oData'])

    # ====== assemble observables for each decay trajectory -- yData ====== #
    Data['yData'] = deepcopy(Data['oData'])
    for i in range(nTRAJ):
        Data['yData'][i][1] = assemble_observables(Data['oData'][i][1])

    if PLOTS:
        utils.plotDecayXYZData(SETTINGS, model_save_dir, Data, outdofs, PLOTS=PLOTS)


    # ====== Truncate decay trajectories ====== #
    Data['oDataTrunc'] = utils.slice_trajectories(Data['oData'], SETTINGS['t_truncate'])
    Data['yDataTrunc'] = utils.slice_trajectories(Data['yData'], SETTINGS['t_truncate'])


    # ====== Get Tangent Space and include reduced coords in Data ====== #
    Vde, Data = utils.getChartandReducedCoords(SETTINGS, model_save_dir, Data, svd_data, PLOTS)

    # ====== Train/test split for decay data ====== #
    indTest = SETTINGS['decay_test_set']
    indTrain = [i for i in range(nTRAJ) if i not in indTest]

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


    # ====== Visualize dominant displacement modes ====== #
    if PLOTS and SETTINGS['observables'] == "pos-vel":
        modesDir, modesFreq = utils.dominant_displacement_modes(RDInfo, Vde, SSMDim=SETTINGS['SSMDim'], tip_node=SETTINGS['tip_node'], n_nodes=SETTINGS['n_nodes'])
        plot.mode_direction(modesDir, modesFreq, show=(PLOTS == 'show'))
        if PLOTS == 'save':
            plt.savefig(join(model_save_dir, "plots", f"displacement_modes.png"), bbox_inches='tight') 


    # ===== Influence of control ====== #
    print("====== Learn influence of control ======")
    # define mappings
    Rauton = lambda x: RDInfo['reducedDynamics']['coefficients'] @ utils.phi(x, RDInfo['reducedDynamics']['polynomialOrder'])
    Vauton = lambda x: IMInfo['parametrization']['H'] @ utils.phi(x, IMInfo['parametrization']['polynomialOrder'])
    if SETTINGS['observables'] == "pos-vel":
        Wauton = lambda y: IMInfo['chart']['H'] @ utils.phi(y, IMInfo['chart']['polynomialOrder'])
    elif SETTINGS['observables'] == "delay-embedding":
        if SETTINGS['custom_delay']:
            Wauton = lambda y: IMInfo['chart']['H'] @ utils.phi(y, IMInfo['chart']['polynomialOrder'])
        else:
            Wauton = lambda y: Vde.T @ y
    else:
        raise RuntimeError("Unknown type of observables, should be ['delay-embedding', 'pos-vel']")

    # For /home/jjalora/Desktop/trunk_origin/
    (t, z), u = utils.import_pos_data(data_dir=join(data_dir, SETTINGS['input_train_data_dir']),
                                    rest_file=None, # join(SETTINGS['robot_dir'], SETTINGS['rest_file']),
                                    q_rest = q_eq,
                                    output_node=SETTINGS['tip_node'], 
                                    return_inputs=True)

    # Conduct regression and get R (controlled reduced dynamics)
    controlData = utils.learnBmatrix(SETTINGS, Wauton, Rauton, z, u, t, u_eq)
    R = controlData['R']
    
    # ====== Test open-loop prediction capabalities of SSM model ====== #

    print("====== Test open-loop prediction capabalities of SSM model ======")
    test_results = utils.analyzeOLControlPredict(SETTINGS, model_save_dir, controlData, q_eq, u_eq, Wauton, R, Vauton, embed_coords=outdofs,
                                                 PLOTS=PLOTS)

    traj_outDofs = [0, 1, 2]
    if PLOTS:
        # plot training data
        plot.traj_xyz_txyz(t=t,
                        x=z[traj_outDofs[0], :], y=z[traj_outDofs[1], :], z=z[traj_outDofs[2], :],
                        show=(PLOTS == 'show'))
        plt.savefig(join(model_save_dir, "plots", f"input_training_data_z.png"), bbox_inches='tight')
        plot.inputs(t, u, show=(PLOTS == 'show'))
        if PLOTS == 'save':
            plt.savefig(join(model_save_dir, "plots", f"input_training_data_u.png"), bbox_inches='tight')
            
        # plot gradients (numerical, predicted autonomous, predicted with inputs)
        dxDt_ROM_with_B = R(controlData['x_train'], controlData['u_train'])
        utils.plot_gradients(controlData['t_train'], controlData['dxdt'], controlData['dxdt_ROM'], dxDt_ROM_with_B, PLOTS, model_save_dir)


    # ====== Save SSM model to folder, along with settings and test results ====== #
    utils.save_ssm_model(model_save_dir, RDInfo, IMInfo, controlData['B_learn'], Vde, q_eq, u_eq, SETTINGS, test_results)
    print(f"====== Saved SSM model to folder {model_save_dir} ======")

    # Close all plots for this model
    plt.close('all')
    # Display total runtime
    end_time = time()
    print(f"====== Finished after total runtime: {(end_time - start_time):.2f} sec ======")


if __name__ == "__main__":
    if SETTINGS['data_subdirs'] is None:
        data_dir = SETTINGS['data_dir']
        generate_ssmr_model(data_dir, save_model_to_data_dir=False)
    else:
        for subdir in tqdm(SETTINGS['data_subdirs']):
            data_dir = join(SETTINGS['data_dir'], subdir)
            generate_ssmr_model(data_dir, save_model_to_data_dir=True)