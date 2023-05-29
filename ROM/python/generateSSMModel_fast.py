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
    'observables': "delay-embedding", # "pos-vel", #
    'reduced_coordinates': "local",

    'use_ssmlearn': "py", # "matlab"

    'robot_dir': "/home/jonas/Projects/stanford/soft-robot-control/examples/trunk",
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

    'data_dir': "/media/jonas/Backup Plus/jonas_soft_robot_data/trunk_adiabatic_N=100",
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
    't_truncate': [0.1, 3],

    'decay_test_set': [0, 8, 16, 24],

    'poly_u_order': 1,
    'input_train_data_dir': "open-loop",
    'input_test_data_dir': ["open-loop_circle", ],
    'input_train_ratio': 0.8
}
SETTINGS['data_subdirs'] = sorted([dir for dir in listdir(SETTINGS['data_dir']) if isdir(join(SETTINGS['data_dir'], dir))])[8:]
print(SETTINGS['data_subdirs'])

PLOTS = False # 'save', 'show', False ('show' shows and saves plots, 'save' only saves plots, False does nothing)

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
    # create a new directory into which the new model will be saved
    if not save_model_to_data_dir:
        # create a new folder in robot_dir/SSMmodels named model_XXX
        model_dir = join(SETTINGS['robot_dir'], SETTINGS['model_save_dir'])
        models = sorted(listdir(model_dir))
        if not models:
            model_save_dir = join(model_dir, "model_00")
        else:
            model_save_dir = join(model_dir, f"model_{int(models[-1].split('_')[-1])+1:02}")
    else:
        # save model to a new dir called SSMmodel_{observable} inside the data_dir
        model_save_dir = join(data_dir, f"SSMmodel_{SETTINGS['observables']}")
    if not exists(model_save_dir):
        mkdir(model_save_dir)
    if PLOTS and not exists(join(model_save_dir, "plots")):
        mkdir(join(model_save_dir, "plots"))

    start_time = time()

    # ====== Import decay trajectories -- oData ====== #
    print("====== Import decay trajectories ======")
    Data = {}
    decay_data_dir = join(data_dir, SETTINGS['decay_dir'])
    Data['oData'] = utils.import_pos_data(decay_data_dir,
                                        rest_file=join(SETTINGS['robot_dir'], SETTINGS['rest_file']),
                                        output_node=output_node,
                                        t_in=SETTINGS['t_decay'][0], t_out=SETTINGS['t_decay'][1])
    nTRAJ = len(Data['oData'])

    # ====== Change of coordinates: shift oData to the offset equilibrium position ====== #
    # pre-tensioned equilibrium position
    q_eq = np.mean([Data['oData'][i][1][:, -1] for i in range(nTRAJ)], axis=0)    
    for i in range(nTRAJ):
        Data['oData'][i][1] = (Data['oData'][i][1].T - q_eq).T
    with open(join(model_save_dir, "pre-tensioned_rest_q.pkl"), "wb") as f:
        pickle.dump(q_eq, f)
    # pre-tensioning, mean control input
    with open(join(decay_data_dir, "pre_tensioning.pkl"), "rb") as f:
        pre_tensioning = pickle.load(f)
    u_eq = np.array(pre_tensioning)

    # ====== assemble observables for each decay trajectory -- yData ====== #
    Data['yData'] = deepcopy(Data['oData'])
    for i in range(nTRAJ):
        Data['yData'][i][1] = assemble_observables(Data['oData'][i][1])

    if PLOTS:
        outdofs = [0, 1, 2]
        # plot trajectories in 3D [x, y, z] space
        plot.traj_3D(Data,
                    xyz_idx=[('yData', outdofs[0]), ('yData', outdofs[1]), ('yData', outdofs[2])],
                    xyz_names=[r'$x$ [mm]', r'$y$ [mm]', r'$z$ [mm]'], show=(PLOTS == 'show'))
        if PLOTS == 'save':
            plt.savefig(join(model_save_dir, "plots", f"decay_3D_xyz.png"), bbox_inches='tight')
        # plot evolution of x, y and z in time, separately in 3 subplots
        plot.traj_xyz(Data,
                    xyz_idx=[('yData', outdofs[0]), ('yData', outdofs[1]), ('yData', outdofs[2])],
                    xyz_names=[r'$x$ [mm]', r'$y$ [mm]', r'$z$ [mm]'], show=(PLOTS == 'show'))
        if PLOTS == 'save':
            plt.savefig(join(model_save_dir, "plots", f"decay_t_xyz.png"), bbox_inches='tight')
        # plot trajectories in 3D [x, x_dot, z] space / NB: x_dot = v_x
        plot.traj_3D(Data,
                    xyz_idx=[('yData', outdofs[0]), ('yData', outdofs[0]+SETTINGS['oDOF']), ('yData', outdofs[2])],
                    xyz_names=[r'$x$ [mm]', r'$\dot{x}$ [mm/s]', r'$z$ [mm]'], show=(PLOTS == 'show'))
        if PLOTS == 'save':
            plt.savefig(join(model_save_dir, "plots", f"decay_3D_xxdotz.png"), bbox_inches='tight')


    # ====== Truncate decay trajectories ====== #
    Data['oDataTrunc'] = utils.slice_trajectories(Data['oData'], SETTINGS['t_truncate'])
    Data['yDataTrunc'] = utils.slice_trajectories(Data['yData'], SETTINGS['t_truncate'])


    # ====== Perform SVD on displacement field ====== #
    Data['etaDataTrunc'] = deepcopy(Data['oDataTrunc'])
    if SETTINGS['reduced_coordinates'] == "global":
        with open(join(SETTINGS['data_dir'], "SSM_model_origin.pkl"), "rb") as f:
            Vde = pickle.load(f)['model']['V']
        for i in range(nTRAJ):
                Data['etaDataTrunc'][i][1] = Vde.T @ Data[svd_data][i][1]
    else:
        print("====== Perform SVD on displacement field ======")
        show_modes = 9
        Xsnapshots = np.hstack([DataTrunc[1] for DataTrunc in Data[svd_data]])
        v, s = utils.sparse_svd(Xsnapshots, up_to_mode=max(SETTINGS['SSMDim'], show_modes))
        if SETTINGS['observables'] == "delay-embedding":
            Vde = v[:, :SETTINGS['SSMDim']]
            for i in range(nTRAJ):
                Data['etaDataTrunc'][i][1] = Vde.T @ Data[svd_data][i][1]
        elif SETTINGS['observables'] == "pos-vel":
            assert SETTINGS['SSMDim'] % 2 == 0
            Vde = np.kron(np.eye(2), v[:, :SETTINGS['SSMDim']//2])
            for i in range(nTRAJ):
                Data['etaDataTrunc'][i][1] = Vde.T @ np.vstack((Data[svd_data][i][1], np.gradient(Data[svd_data][i][1], SETTINGS['dt'], axis=1)))
        else:
            raise RuntimeError("Unknown type of observables, should be ['delay-embedding', 'pos-vel']")

        if PLOTS:
            # Plot variance description: we expect the first couple of modes to capture almost all variance.
            # Note we assume data centered around the origin, which is the fixed point of our system.
            plot.pca_modes(s**2, up_to_mode=show_modes, show=(PLOTS == 'show'))
            if PLOTS == 'save':
                plt.savefig(join(model_save_dir, "plots", f"pca_modes.png"), bbox_inches='tight')
            # plot first three reduced coordinates
            plot.traj_xyz(Data,
                        xyz_idx=[('etaDataTrunc', 0), ('etaDataTrunc', 1), ('etaDataTrunc', 2)],
                        xyz_names=[r'$x_1$', r'$x_2$', r'$x_3$'], show=(PLOTS == 'show'))
            if PLOTS == 'save':
                plt.savefig(join(model_save_dir, "plots", f"decay_reduced_coords_123.png"), bbox_inches='tight')

    # ====== Train/test split for decay data ====== #
    indTest = SETTINGS['decay_test_set']
    indTrain = [i for i in range(nTRAJ) if i not in indTest]



    print("====== Learn SSM geometry and reduced dynamics ======")
    # SSM geometry
    if SETTINGS['use_ssmlearn'] == "matlab":
        # ====== Start Matlab engine and SSMLearn ====== #
        print("====== Start Matlab engine and SSMLearn ======")
        ssm = utils.start_matlab_ssmlearn("/home/jonas/Projects/stanford/SSMR-for-control")
        # make data ready for Matlab
        yDataTruncTrain_matlab = utils.numpy_to_matlab([Data['yDataTrunc'][i] for i in indTrain])
        etaDataTruncTrain_matlab = utils.numpy_to_matlab([Data['etaDataTrunc'][i] for i in indTrain])
        IMInfo = ssm.IMGeometry(yDataTruncTrain_matlab, SETTINGS['SSMDim'], SETTINGS['SSMOrder'],
                                'reducedCoordinates', etaDataTruncTrain_matlab, 'l_vals', 0.)
        if SETTINGS['observables'] == "pos-vel":
            IMInfoInv = ssm.IMGeometry(etaDataTruncTrain_matlab, SETTINGS['SSMDim'], SETTINGS['SSMOrder'],
                                    'reducedCoordinates', yDataTruncTrain_matlab)
            for key in ['map', 'polynomialOrder', 'dimension', 'nonlinearCoefficients', 'phi', 'exponents', 'H']:
                IMInfo['chart'][key] = IMInfoInv['parametrization'][key]
        # SSM reduced dynamics
        RDInfo = ssm.IMDynamicsFlow(etaDataTruncTrain_matlab, 'R_PolyOrd', SETTINGS['ROMOrder'], 'style', 'default', 'l_vals', 0.)
        # quit matlab engine
        ssm.quit()
        # convert matlab double arrays to numpy arrays
        utils.matlab_info_dict_to_numpy(IMInfo)
        utils.matlab_info_dict_to_numpy(RDInfo)
    elif SETTINGS['use_ssmlearn'] == "py":
        print("====== Using SSMLearnPy ======")
        from ssmlearnpy import SSMLearn
        ssm = SSMLearn(
            t=[Data['oDataTrunc'][i][0] for i in indTrain], 
            x=[Data['yDataTrunc'][i][1] for i in indTrain], 
            reduced_coordinates=[Data['etaDataTrunc'][i][1] for i in indTrain],
            ssm_dim=SETTINGS['SSMDim'], 
            dynamics_type=SETTINGS['RDType']
        )
        # find parametrization of SSM and reduced dynamics on SSM
        ssm.get_parametrization(poly_degree=SETTINGS['SSMOrder'])    
        ssm.get_reduced_dynamics(poly_degree=SETTINGS['ROMOrder'])
        # Save relevant coeffs and params into dictss which resemble the outputs of the Matlab SSMLearn package
        IMInfo = {'parametrization': {
            'polynomialOrder': SETTINGS['SSMOrder'],
            'H': ssm.decoder.map_info['coefficients']
        }, 'chart': {}}
        RDInfo = {'reducedDynamics': {
            'polynomialOrder': SETTINGS['ROMOrder'],
            'coefficients': ssm.reduced_dynamics.map_info['coefficients'],
            'eigenvaluesLinPartFlow': ssm.reduced_dynamics.map_info['eigenvalues_lin_part']
        }, 'dynamicsType': SETTINGS['RDType']}

    # ====== Analyze autonomous SSM: accuracy of geometry and reduced dynamics ====== #
    if PLOTS:
        print("====== Analyze autonomous SSM: accuracy of geometry and reduced dynamics ======")
        trajRec = {}
        # geometry error
        meanErrorGeo = {}
        trajRec['geo'] = utils.lift_trajectories(IMInfo, Data['etaDataTrunc'])
        normedTrajDist = utils.compute_trajectory_errors(trajRec['geo'], Data['yDataTrunc'])[0] * 100
        meanErrorGeo['Train'] = np.mean(normedTrajDist[indTrain])
        meanErrorGeo['Test'] = np.mean(normedTrajDist[indTest])
        print(f"Average parametrization train error: {meanErrorGeo['Train']:.4e}")
        print(f"Average parametrization test error: {meanErrorGeo['Test']:.4e}")
        # plot comparison of SSM-predicted vs. actual test trajectories
        axs = plot.traj_xyz(Data,
                            xyz_idx=[('yData', 0), ('yData', 1), ('yData', 2)],
                            xyz_names=[r'$x$ [mm]', r'$y$ [mm]', r'$z$ [mm]'],
                            traj_idx=indTest,
                            show=False)
        plot.traj_xyz(trajRec,
                    xyz_idx=[('geo', 0), ('geo', 1), ('geo', 2)],
                    xyz_names=[r'$x$ [mm]', r'$y$ [mm]', r'$z$ [mm]'],
                    traj_idx=indTest,
                    axs=axs, ls=':', color='darkblue', show=(PLOTS == 'show'))
        if PLOTS == 'save':
            plt.savefig(join(model_save_dir, "plots", f"geometry_error.png"), bbox_inches='tight')
        # reduced dynamics error
        meanErrorDyn = {}
        trajRec['rd'] = utils.advectRD(RDInfo, Data['etaDataTrunc'])[0]
        normedTrajDist = utils.compute_trajectory_errors(trajRec['rd'], Data['etaDataTrunc'])[0] * 100
        meanErrorDyn['Train'] = np.mean(normedTrajDist[indTrain])
        meanErrorDyn['Test'] = np.mean(normedTrajDist[indTest])
        print(f"Average dynamics train error: {meanErrorDyn['Train']:.4f}")
        print(f"Average dynamics test error: {meanErrorDyn['Test']:.4f}")
        axs = plot.traj_xyz(Data,
                            xyz_idx=[('etaDataTrunc', 0), ('etaDataTrunc', 1), ('etaDataTrunc', 2)],
                            xyz_names=[r'$x_1$', r'$x_2$', r'$x_3$'],
                            traj_idx=indTest,
                            show=False)
        plot.traj_xyz(trajRec,
                    xyz_idx=[('rd', 0), ('rd', 1), ('rd', 2)],
                    xyz_names=[r'$x_1$', r'$x_2$', r'$x_3$'],
                    traj_idx=indTest,
                    axs=axs, ls=':', color='darkblue', show=(PLOTS == 'show'))
        if PLOTS == 'save':
            plt.savefig(join(model_save_dir, "plots", f"reduced_dynamics_error.png"), bbox_inches='tight')
        # global error
        meanErrorGlo = {}
        trajRec['glob'] = utils.lift_trajectories(IMInfo, trajRec['rd'])
        normedTrajDist = utils.compute_trajectory_errors(trajRec['glob'], Data['yDataTrunc'])[0] * 100
        meanErrorGlo['Train'] = np.mean(normedTrajDist[indTrain])
        meanErrorGlo['Test'] = np.mean(normedTrajDist[indTest])
        print(f"Average global train error: {meanErrorGlo['Train']:.4f}")
        print(f"Average global test error: {meanErrorGlo['Test']:.4f}")
        axs = plot.traj_xyz(Data,
                            xyz_idx=[('yData', 0), ('yData', 1), ('yData', 2)],
                            xyz_names=[r'$x$', r'$y$', r'$z$'],
                            traj_idx=indTest,
                            show=False)
        plot.traj_xyz(trajRec,
                    xyz_idx=[('glob', 0), ('glob', 1), ('glob', 2)],
                    xyz_names=[r'$x$', r'$y$', r'$z$'],
                    traj_idx=indTest,
                    axs=axs, ls=':', color='darkblue', show=(PLOTS == 'show'))
        if PLOTS == 'save':
            plt.savefig(join(model_save_dir, "plots", f"global_error.png"), bbox_inches='tight')


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
        Wauton = lambda y: Vde.T @ y
    else:
        raise RuntimeError("Unknown type of observables, should be ['delay-embedding', 'pos-vel']")
    # training data
    (t, z), u = utils.import_pos_data(data_dir=join(data_dir, SETTINGS['input_train_data_dir']),
                                    rest_file=join(SETTINGS['robot_dir'], SETTINGS['rest_file']),
                                    output_node=SETTINGS['tip_node'], return_inputs=True)
    if SETTINGS['observables'] == "pos-vel":
        z = (z.T - q_eq[3*SETTINGS['tip_node']:3*SETTINGS['tip_node']+3]).T
    else:
        print(z)
        print(q_eq)
        z = (z.T - q_eq).T    
    u = (u.T - u_eq).T
    y = assemble_observables(z)
    x = Wauton(y)
    # train/test split on input training data
    split_idx = int(SETTINGS['input_train_ratio'] * len(t))
    t_train, t_test = t[:split_idx], t[split_idx:]
    _, z_test = z[:, :split_idx], z[:, split_idx:]
    # y_train, y_test = y[:, :split_idx], y[:, split_idx:]
    u_train, u_test = u[:, :split_idx], u[:, split_idx:]
    x_train, x_test = x[:, :split_idx], x[:, split_idx:]

    if PLOTS:
        # plot training data
        plot.traj_xyz_txyz(t=t,
                        x=z[0, :], y=z[1, :], z=z[2, :],
                        show=(PLOTS == 'show'))
        plt.savefig(join(model_save_dir, "plots", f"input_training_data_z.png"), bbox_inches='tight')
        plot.inputs(t, u, show=(PLOTS == 'show'))
        if PLOTS == 'save':
            plt.savefig(join(model_save_dir, "plots", f"input_training_data_u.png"), bbox_inches='tight')

    # autonomous reduced dynamics vs. numerical derivative
    dxdt = np.gradient(x_train, SETTINGS['dt'], axis=1)
    dxdt_ROM = Rauton(x_train)

    # ====== regress B matrix ====== #
    assemble_features = lambda u, x: utils.phi(u, order=SETTINGS['poly_u_order']) # utils.phi(np.vstack([u, x]), order=SETTINGS['poly_u_order']) # 
    X = assemble_features(u_train, x_train)
    B_learn = utils.regress_B(X, dxdt, dxdt_ROM, alpha=0, method='ridge')
    print(f"Frobenius norm of B_learn: {np.linalg.norm(B_learn, ord='fro'):.4f}")

    R = lambda x, u: Rauton(np.atleast_2d(x)) + B_learn @ assemble_features(u, x)
    if PLOTS:
        # plot gradients (numerical, predicted autonomous, predicted with inputs)
        plot_reduced_coords = np.s_[:] # [3, 4, 5]
        dxDt_ROM_with_B = R(x_train, u_train)
        plot.reduced_coordinates_gradient(t_train, [dxdt[plot_reduced_coords, :], dxdt_ROM[plot_reduced_coords, :], dxDt_ROM_with_B[plot_reduced_coords, :]],
                                        labels=["true numerical", "predicted autonomous", "predicted with inputs"], how="norm", show=(PLOTS == 'show'))
        if PLOTS == "save":
            plt.savefig(join(model_save_dir, "plots", f"reduced_coordinates_gradient.png"), bbox_inches='tight')


    # ====== Test open-loop prediction capabalities of SSM model ====== #
    print("====== Test open-loop prediction capabalities of SSM model ======")
    test_results = {}
    test_trajectories = [{
            'name': "like training data",
            't': t_test,
            'z': z_test,
            'u': u_test,
            'x': x_test
        }]
    for test_traj in SETTINGS['input_test_data_dir']:
        traj_dir = join(SETTINGS['robot_dir'], "dataCollection", test_traj)
        (t, z), u = utils.import_pos_data(data_dir=traj_dir,
                                        rest_file=join(SETTINGS['robot_dir'], SETTINGS['rest_file']),
                                        output_node=SETTINGS['tip_node'], return_inputs=True, traj_index=0)
        if SETTINGS['observables'] == "pos-vel":
            z = (z.T - q_eq[3*SETTINGS['tip_node']:3*SETTINGS['tip_node']+3]).T
        else:
            z = (z.T - q_eq).T
        u = (u.T - u_eq).T
        y = assemble_observables(z)
        x = Wauton(y)
        test_trajectories.append({
                'name': test_traj,
                't': t,
                'z': z,
                'u': u,
                'x': x
            })
    for traj in test_trajectories:
        try:
            z_pred = utils.predict_open_loop(R, Vauton, traj['t'], traj['u'], x0=traj['x'][:, 0])
        except Exception as e:
            z_pred = np.nan * np.ones_like(traj['z'])
        rmse = float(np.sum(np.sqrt(np.mean((z_pred[:3, :] - traj['z'][:3])**2, axis=0))) / len(traj['t']))
        print(f"({traj['name']}): RMSE = {rmse:.4f}")
        test_results[traj['name']] = {
            'RMSE': rmse
        }
        if PLOTS:
            axs = plot.traj_xyz_txyz(traj['t'],
                                    z_pred[0, :], z_pred[1, :], z_pred[2, :],
                                    show=False)
            axs = plot.traj_xyz_txyz(traj['t'],
                                    traj['z'][0, :], traj['z'][1, :], traj['z'][2, :],
                                    color="tab:orange", axs=axs, show=(PLOTS == 'show'))
            axs[-1].legend(["Predicted trajectory", "Actual trajectory"])
            if PLOTS == 'save':
                plt.savefig(join(model_save_dir, "plots", f"open-loop-prediction_{traj['name']}.png"), bbox_inches='tight')
    print(f"(overall): RMSE = {np.mean([test_results[traj]['RMSE'] for traj in test_results]):.4f}")


    # ====== Save SSM model to folder, along with settings and test results ====== #
    utils.save_ssm_model(model_save_dir, RDInfo, IMInfo, B_learn, Vde, q_eq, u_eq, SETTINGS, test_results)
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