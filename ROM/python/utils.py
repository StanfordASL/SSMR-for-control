import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from scipy.integrate import solve_ivp
from scipy.linalg import orth
from copy import deepcopy
import pickle
from os.path import join, exists, isdir
from os import listdir, mkdir
from copy import deepcopy
from scipy.sparse.linalg import svds
import matlab.engine
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import yaml
from scipy.io import loadmat
import ROM.python.plot_utils as plot
import matplotlib.pyplot as plt

# from sympy.polys.monomials import itermonomials
# from sympy.polys.orderings import monomial_key
# import sympy as sp

# TODO: The shifting here is to account for the fact that we are doing delay embedding wrong.
# Fix the delay embedding and remove this shifting
def slice_trajectories(data, interval: list, t_shift=0.0):
    dataTrunc = deepcopy(data)
    ntraj = len(data)
    for traj in range(ntraj):
        data[traj][0] += t_shift
        # truncate trajectory array
        dataTrunc[traj][0] = data[traj][0][(data[traj][0] >= interval[0]) & (data[traj][0] <= interval[1])]
        # truncate time array
        dataTrunc[traj][1] = data[traj][1][:, (data[traj][0] >= interval[0]) & (data[traj][0] <= interval[1])]
    return dataTrunc

def mapInt2List(n):
    return [n * 3, n * 3 + 1, n * 3 + 2]


def phi(x, order: int):
    if not isinstance(order, int):
        order = int(order)
    return PolynomialFeatures(degree=order, include_bias=False).fit_transform(x.T).T

""""
    Constructs separate features for u and x
"""
def phi_control(u, x, order_u: int, order_x: int):
    if not isinstance(order_u, int):
        order_u = int(order_u)
    if not isinstance(order_x, int):
        order_x = int(order_x)
    poly_u = PolynomialFeatures(degree=order_u, include_bias=False)
    poly_x = PolynomialFeatures(degree=order_x, include_bias=False)
    u_features = poly_u.fit_transform(u.T).T
    x_features = poly_x.fit_transform(x.T).T
    
    cross = np.array([uf * xf for uf in u_features for xf in x_features])

    # Reshape to combine the last two dimensions
    features = np.concatenate((u_features, cross), axis=0)

    return features

def lift_trajectories(IMInfo: dict, etaData):
    H = np.array(IMInfo['parametrization']['H'])
    SSMOrder = IMInfo['parametrization']['polynomialOrder']
    map = lambda x: H @ phi(x, SSMOrder)
    return transform_trajectories(map, etaData)


def transform_trajectories(map, inData):
    nTraj = len(inData)
    outData = deepcopy(inData)
    for i in range(nTraj):
        # outData[0][i] = inData[0][i];
        outData[i][1] = map(inData[i][1])
    return outData


def compute_trajectory_errors(yData1, yData2, inds='all'):
    
    nTraj = len(yData1)
    trajDim = yData1[0][1].shape[0]
    trajErrors = np.zeros(nTraj)
    ampErrors = np.zeros(nTraj)

    if inds == 'all':
        inds = list(range(trajDim))
        # check if trajectories have same dimensionality (i.e. the same number of observables)
        assert yData1[0][1].shape[0] == yData2[0][1].shape[0], 'Dimensionalities of trajectories are inconsistent'

    for i in range(nTraj):
        # check if trajectories have same length
        if yData1[i][1].shape[1] == yData2[i][1].shape[1]:
            # print(yData1[i][1][inds, :].shape, yData2[i][1][inds, :].shape)
            trajErrors[i] = np.mean(np.linalg.norm(yData1[i][1][inds, :] - yData2[i][1][inds, :], axis=0)) / np.amax(np.linalg.norm(yData2[i][1][inds, :], axis=0))
            ampErrors[i] = np.mean(np.abs(np.linalg.norm(yData1[i][1][inds, :], axis=0) - np.linalg.norm(yData2[i][1][inds, :]))) / np.mean(np.linalg.norm(yData2[i][1][inds, :], axis=0))
        else:
            # integration has failed
            trajErrors[i] = np.inf
            ampErrors[i] = np.inf
    return trajErrors, ampErrors


def advectRD(RDInfo, etaData):
    assert RDInfo['dynamicsType'] == 'flow', "Only flow type dynamics implemented"
    invT = lambda x: x
    T = lambda x: x
    W_r = np.array(RDInfo['reducedDynamics']['coefficients'])
    polynomialOrder = int(RDInfo['reducedDynamics']['polynomialOrder'])
    N = lambda t, y: W_r @ phi(np.atleast_2d(y), polynomialOrder)
    zData = transform_trajectories(invT, etaData)
    zRec = integrateFlows(N, zData)
    etaRec = transform_trajectories(T, zRec)
    return etaRec, zRec, zData


def integrateFlows(flow, etaData):
    nTraj = len(etaData)
    etaRec = etaData.copy()
    for i in range(nTraj):
        tStart = etaData[i][0][0]
        tEnd = etaData[i][0][-1]
        nSamp = len(etaData[i][0])
        sol = solve_ivp(flow,
                        t_span=[tStart, tEnd],
                        t_eval=np.linspace(tStart, tEnd, nSamp),
                        y0=etaData[i][1][:, 0],
                        method='RK45',
                        vectorized=True,
                        rtol=1e-3,
                        atol=1e-3)
        etaRec[i][0] = sol.t
        etaRec[i][1] = sol.y
    return etaRec

# Newest observations stored at the bottom of vector
def delayEmbedding(undelayedData, embed_coords=[0, 1, 2], up_to_delay=4):    
    undelayed = undelayedData[embed_coords, :]
    buf = [undelayed]
    for delta in list(range(1, up_to_delay+1)):
        delayed_by_delta = np.roll(undelayed, -delta)
        delayed_by_delta[:, -delta:] = 0
        buf.append(delayed_by_delta)
    delayedData = np.vstack(buf)
    return delayedData

# Assume we import from mat file and that the data is already centered
def import_traj_data(data_dir, outdofs):

    trajData = []
    inputData = []
    files = sorted([f for f in listdir(data_dir) if 'snapshots.mat' in f])
    if not isinstance(files, list):
        files = [files]
    
    for i, traj in enumerate(files):
        with open(join(data_dir, traj), 'rb') as file:
            data = loadmat(file)

            # Extract decay trajectories from the various initial conditions
            for i in range(data['oData'].shape[0]):
                t = data['oData'][i][0][0] #[traj idx][time (0) or traj (1)][single array (0)]
                yData = data['oData'][i][1][outdofs, :]
                trajData.append([t, yData])

                uData = data['oData'][i][3]
                inputData.append([t, uData])

    if len(trajData) == 1:
        trajData, inputData = trajData[0], inputData[0]
    return np.array(trajData, dtype=object), np.array(inputData, dtype=object)

    

# TODO: We should generalize this to more than just the tip
def import_pos_data(data_dir, rest_file=None, q_rest=None, output_node=None, 
                    t_in=0, t_out=None, return_inputs=False, return_velocity=False, 
                    traj_index=np.s_[:], file_type='pkl', subsample=1, shift=False, return_reduced_coords=False,
                    hardware=False):
    if q_rest is not None:
        q_rest = np.array(q_rest)
    elif rest_file is not None:
        with open(rest_file, 'rb') as file:
            rest_file = pickle.load(file)
            try:
                q_rest = rest_file['q'][0]
            except:
                q_rest = rest_file['rest']
    else:
        raise ValueError('No rest file or rest position provided')

    if file_type == 'pkl':
        files = sorted([f for f in listdir(data_dir) if 'snapshots.pkl' in f])
    elif file_type == 'mat':
        files = sorted([f for f in listdir(data_dir) if 'snapshots.mat' in f])
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    files = files[traj_index]
    if not isinstance(files, list):
        files = [files]
    out_data = []
    reduced_data = []
    u = []
    for i, traj in enumerate(files):
        # Setup output nodes
        if output_node == 'all':
            node_slice = np.s_[:]
        elif type(output_node) == int:
            node_slice = np.s_[3*output_node:3*output_node+3]
        else:
            node_slice = output_node
        
        with open(join(data_dir, traj), 'rb') as file:
            # Assumes files are NOT centered
            if file_type == 'pkl':
                data = pickle.load(file)
                t = np.array(data['t'])
                ind = (t_in <= t)
                if t_out is not None:
                    ind = ind & (t <= t_out)
                t = t[ind] - t_in
                t = t[::subsample]

                q_node = ()
                if len(data['q']) > 0:
                    q_node = (np.array(data['q'])[:, node_slice] - q_rest[node_slice]).T
                elif len(data['z']) > 0:
                    q_node = (np.array(data['z'])[:, 3:] - q_rest[node_slice]).T
                else:
                    raise RuntimeError("Cannot find data for desired node")
                
                # Truncate to specific times
                q_node_trunc = q_node[:, ind]
                # then subsample
                q_node = q_node_trunc[:, ::subsample]
                if return_velocity:
                    v_node = np.array(data['v'])[ind, node_slice].T
                    v_node = v_node[:, ::subsample]
                    data_traj = np.vstack((q_node, v_node))
                else:
                    data_traj = q_node
                out_data.append([t, data_traj])
                uTrunc = np.array(data['u'])[ind, :].T
                u.append(uTrunc[:, ::subsample])
            
            # Mat files only expect observables and not full nodes
            # oData contains all of the decay for each initial condition
            # Assumes we only load one file
            # Assumes files ARE centered. If uncentered, set shift = True
            elif file_type == 'mat':
                data = loadmat(file)

                # Extract decay trajectories from the various initial conditions
                for i in range(data['oData'].shape[0]):
                    t = data['oData'][i][0][0] #[traj idx][time (0) or traj (1)][single array (0)]
                    ind = (t_in <= t)
                    if t_out is not None:
                        ind = ind & (t <= t_out)
                    t = t[ind] - t_in
                    t = t[::subsample]

                    # Assume the format (pos, vel)
                    if return_velocity:
                        if shift:
                            data_traj_trunc = data['oData'][i][1][:, ind] - np.hstack((q_rest[node_slice], np.zeros(3))).reshape((6,1))
                        else:
                            data_traj_trunc = data['oData'][i][1][:, ind]
                    else:
                        if shift:
                            if type(node_slice) == list:
                                data_traj_trunc = data['oData'][i][1][:, ind] - q_rest[node_slice].reshape((3,1))
                                data_traj_trunc = data_traj_trunc[node_slice, :]
                            else:
                                data_traj_trunc = data['oData'][i][1][0:3, ind] - q_rest[node_slice].reshape((3,1))
                        else:
                            if type(node_slice) == list:
                                data_traj_trunc = data['oData'][i][1][:, ind]
                                data_traj_trunc = data_traj_trunc[node_slice, :]
                            else:
                                data_traj_trunc = data['oData'][i][1][0:3, ind]

                    data_traj = data_traj_trunc[:, ::subsample]
                    out_data.append([t, data_traj])
                    
                    if return_inputs:
                        if hardware:
                            uTrunc = np.array(data['u'])[i][0][:, ind]
                        else:
                            uTrunc = np.array(data['u'])[ind, :].T
                        u.append(uTrunc[:, ::subsample])
                    elif return_reduced_coords:
                        eta_traj_trunc = data['yData'][i][1][:, ind]
                        eta_traj = eta_traj_trunc[:, ::subsample]
                        reduced_data.append([t, eta_traj])


    if len(out_data) == 1:
        out_data, u = out_data[0], u[0]
    if return_inputs:
        return out_data, u
    elif return_reduced_coords:
        return out_data, reduced_data
    else:
        return out_data


def sparse_svd(X, up_to_mode):
    """compute X = USV^T and order columns of V, S by magnitude of singular values
    returns S, V"""
    v, s, _ = svds(X, k=up_to_mode, which="LM")
    ind = np.argsort(s)[::-1]
    s = s[ind]
    v = v[:, ind]
    return v, s


def numpy_to_matlab(array_in):
    array_out = [[], []]
    for i in range(len(array_in)):
        array_out[0].append(matlab.double(initializer=array_in[i][0].tolist()))
        array_out[1].append(matlab.double(initializer=array_in[i][1].tolist()))
    return array_out


def start_matlab_ssmlearn(ssmlearn_dir):
    eng = matlab.engine.start_matlab()
    eng.cd(ssmlearn_dir, nargout=0)
    eng.install(nargout=0)
    eng.cd("ROM/", nargout=0)
    return eng


def matlab_info_dict_to_numpy(info_dict):
  for k,v in info_dict.items():        
     if isinstance(v, dict):
         matlab_info_dict_to_numpy(v)
     else:            
        if isinstance(v, matlab.double):
           info_dict[k] = np.array(v)


def dominant_displacement_modes(RDInfo, Vde, SSMDim, tip_node, n_nodes):
    nDOF = 3 * n_nodes
    outdofsMat = np.zeros((3, 2*nDOF))
    outdofsPS = 3 * tip_node + np.array([0, 1, 2])
    for i in range(3):
        outdofsMat[i, outdofsPS[i]] = 1
    rDOF = SSMDim // 2
    redDynLinearPart = np.array(RDInfo['reducedDynamics']['coefficients'])[:, :2*rDOF]
    redStiffnessMat = -redDynLinearPart[rDOF:, :rDOF]
    w2, UconsRedDyn = np.linalg.eig(redStiffnessMat)
    print(w2)
    sorted_idx = np.argsort(w2)
    _, UconsRedDyn = np.diag(np.sqrt(w2[sorted_idx])), UconsRedDyn[:, sorted_idx]
    Ucons = np.kron(np.eye(2), UconsRedDyn)
    Vcons = Vde @ Ucons
    modesDir = outdofsMat @ Vcons[:, :rDOF]
    modesFreq = np.imag(RDInfo['eigenvaluesLinPartFlow']).flatten()[:rDOF]
    return modesDir, modesFreq


def regress_B(X, dxdt, dxdt_ROM, poly_u_order=1, alpha=0, method='ridge'):
    if method == 'ridge':
        ridgeModel = Ridge(alpha=alpha, fit_intercept=False)
        ridgeModel.fit(X.T, (dxdt - dxdt_ROM).T)
        B_learn = ridgeModel.coef_
    else:
        raise RuntimeError("Desired regression method not yet implemented")
    return B_learn


def predict_open_loop(R, Vauton, t, u, x0, method='RK45'):
    uInterpFun = interp1d(t, u, axis=1, fill_value="extrapolate")
    uFun = lambda t: uInterpFun(t).reshape(-1, 1)
    R_t = lambda t, x: R(x, uFun(t))
    # solve IVP of reduced dynamics using open-loop inputs
    sol = solve_ivp(R_t,
                    t_span=[t[0], t[-1]],
                    t_eval=t,
                    y0=x0,
                    method=method,
                    vectorized=True,
                    rtol=1e-2,
                    atol=1e-2)
    # resulting (predicted) open-loop trajectory in reduced coordinates
    xTraj = sol.y
    yTraj = Vauton(xTraj)
    zTraj = yTraj
    # zTraj = yTraj[:3, :]
    # if integration unsuccessful, fill up solution with nans
    if zTraj.shape[1] < len(t):
        zTraj = np.hstack((zTraj, np.tile(np.nan, (zTraj.shape[0], len(t) - zTraj.shape[1]))))
    return zTraj

def multivariate_polynomial(x, order: int):
    # poly = PolynomialFeatures(degree=SSMOrder, include_bias=False).fit(x.T)
    # # print(poly.get_feature_names_out(['a', 'b', 'c', 'd', 'e', 'f']))
    # features = poly.transform(x.T)
    if not isinstance(order, int):
        order = int(order)
    return PolynomialFeatures(degree=order, include_bias=False).fit_transform(x.T).T

def Rauton(RDInfo):
    W_r = np.array(RDInfo['reducedDynamics']['coefficients'])
    polynomialOrder = RDInfo['reducedDynamics']['polynomialOrder']
    phi = lambda x: multivariate_polynomial(x, polynomialOrder)
    Rauton = lambda y: W_r @ phi(y) # / 10
    return Rauton

def save_ssm_model(model_save_dir, RDInfo, IMInfo, B_learn, Vde, q_eq, u_eq, settings, test_results, custom_obs=False):
    SSM_model = {'model': {}, 'params': {}}
    SSM_model['model']['w_coeff'] = IMInfo['parametrization']['H']
    SSM_model['model']['r_coeff'] = RDInfo['reducedDynamics']['coefficients']
    SSM_model['model']['B'] = B_learn
    SSM_model['params']['SSM_order'] = settings['SSMOrder']
    SSM_model['params']['ROM_order'] = settings['ROMOrder']
    SSM_model['params']['state_dim'] = settings['SSMDim']
    SSM_model['params']['u_order'] = settings['poly_u_order']
    SSM_model['params']['input_dim'] = settings['input_dim']
    SSM_model['model']['q_eq'] = q_eq
    SSM_model['model']['u_eq'] = u_eq
    if settings['observables'] == "delay-embedding":
        SSM_model['params']['delay_embedding'] = True
        SSM_model['model']['V'] = Vde
        SSM_model['params']['output_dim'] = settings['oDOF']
        if settings['custom_delay']:
            SSM_model['params']['obs_dim'] = settings['oDOF'] * (1 + settings['custom_delay'])
            SSM_model['model']['v_coeff'] = IMInfo['chart']['H']
            SSM_model['params']['delays'] = settings['custom_delay']
        elif custom_obs: # Switching to position and velocity observables
            SSM_model['params']['delay_embedding'] = False
            SSM_model['params']['obs_dim'] = 6
            SSM_model['model']['v_coeff'] = IMInfo['chart']['H']
            SSM_model['params']['delays'] = 0
            SSM_model['params']['output_dim'] = 2 * settings['oDOF']
        else:
            SSM_model['params']['delays'] = settings['n_delay']
            SSM_model['params']['obs_dim'] = settings['oDOF'] * (1 + settings['n_delay'])
            SSM_model['model']['v_coeff'] = None
    elif settings['observables'] == "pos-vel":
        SSM_model['model']['V'] = Vde
        SSM_model['model']['v_coeff'] = IMInfo['chart']['H']
        SSM_model['params']['delay_embedding'] = False
        SSM_model['params']['delays'] = 0
        SSM_model['params']['obs_dim'] = 6
        SSM_model['params']['output_dim'] = 2 * settings['oDOF']
    with open(join(model_save_dir, "SSM_model.pkl"), 'wb') as f:
        pickle.dump(SSM_model, f)
    with open(join(model_save_dir, 'settings.yaml'), 'w') as f:
        yaml.dump(settings, f)
    with open(join(model_save_dir, 'test_results.yaml'), 'w') as f:
        yaml.dump(test_results, f)

def advect_adiabaticRD_with_inputs(t, y0, u, y_target, interpolator, ROMOrder=3, SSMDim=6, know_target=True):
    dt = 0.01
    N = len(t)-1
    x = np.full((SSMDim, N+1), np.nan)
    y_pred = np.full((len(y0), N+1), np.nan)
    y_pred[:, 0] = y0
    y_bar = np.full((len(y0), N+1), np.nan)
    u_bar = np.full((u.shape[0], N+1), np.nan)
    xdot = np.full((SSMDim, N+1), np.nan)
    weights = np.full((1, N+1), np.nan)

    transform = interpolator.transform  # timed_transform

    for i in range(N):
        # compute the weights used at this timestep
        try:
            xy = y0[:3] # y_target[:2, i]
            # simplex = tri.find_simplex(y[-3:-1, i])
            # b = tri.transform[simplex, :2] @ (y[-3:-1, i] - tri.transform[simplex, 2])
            # c = np.r_[b, 1 - b.sum()]
            # point_idx = tri.simplices[simplex]
            # weights[point_idx, i] = c
            # compute and integrate the dynamics at this timestep
            y_bar[:, i] = np.tile(transform(xy, 'q_bar'), 1 + 4) # np.concatenate([transform(xy, 'q_bar'), np.zeros(3)]) # y0 # y_target[:, i] #
            u_bar[:, i] = transform(xy, 'u_bar')
            # x[i] = V[i]^T @ (y[i] - y_bar[i])
            if i == 0:
                # print(transform(xy, 'v_coeff').shape)
                x[:, i] = transform(xy, 'V').T @ (y0 - y_bar[:, i]) # (transform(xy, 'v_coeff') @ phi((y0 - y_bar[:, i]).reshape(-1, 1), 3)).flatten() # y[:, i]) # 
            # xdot[i] = R(x[i]) + B[i] @ (u[i] - u_bar[i])
            xdot[:, i] = (transform(xy, 'r_coeff') @ phi(x[:, i].reshape(-1, 1), ROMOrder)).flatten() + transform(xy, 'B_r') @ (u[:, i] - u_bar[:, i]) # -0.5 * np.eye(6) @ x[:, i] # -0.001 * np.ones(6) # 
            # forward Euler: x[i+1] = x[i] + dt * xdot[i]
            x[:, i+1] = x[:, i] + dt * xdot[:, i]
            # y[i+1] = W(x[i+1]) + y_bar[i]
            y_pred[:, i+1] = (transform(xy, 'w_coeff') @ phi(x[:, i+1].reshape(-1, 1), 3)).T + y_bar[:, i]
            if not know_target:
                y_target[:, i+1] = y_pred[:, i+1]
        except Exception as e:
            # print("i =", i)
            # print("y_pred[i-1]:", y_pred[:, i-1])
            # print("y[i-1]:", y_target[:, i-1])
            # print("x[i-1]:", x[:, i-1])
            # print("=======================================")
            # print("y_pred[i]:", y_pred[:, i])
            # print("y[i]:", y_target[:, i])
            # print("x[i]:", x[:, i])
            # print("=======================================")
            # print("y_pred[i+1]:", y_pred[:, i+1])
            # print("y[i+1]:", y_target[:, i+1])
            # print("x[i+1]:", x[:, i+1])
            # print("=======================================")
            # print(e)
            # break
            raise e
    return t, x, y_pred, xdot, y_bar, u_bar, weights
    

'''
setupDirAndRestParams: sets up the directory to save the model to and saves the settings and model to that directory
    SETTINGS: dict of settings
    data_dir: directory to save the model to
    save_model_to_data_dir: if True, save the model to the data_dir, else save it to the robot_dir/SSMmodels
    PLOTS: if True, plot the results
'''
def setupDirandRestParams(SETTINGS, data_dir, save_model_to_data_dir=False, PLOTS=False):
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
        model_save_dir = join(data_dir, f"SSMmodel_{SETTINGS['observables']}_ROMOrder={SETTINGS['ROMOrder']}_{SETTINGS['reduced_coordinates']}V")
    if not exists(model_save_dir):
        mkdir(model_save_dir)
    if PLOTS and not exists(join(model_save_dir, "plots")):
        mkdir(join(model_save_dir, "plots"))

    # ====== Change of coordinates: need to shift oData to the offset equilibrium position with respective pre-tensioning ====== #
    # pre-tensioned equilibrium position
    with open(join(data_dir, "rest_q.pkl"), "rb") as f:
        q_eq = np.array(pickle.load(f))
        
    # pre-tensioning, mean control input
    with open(join(data_dir, "pre_tensioning.pkl"), "rb") as f:
        u_eq = np.array(pickle.load(f))
    print(f"Pre-tensioning: {u_eq}")
    print(f"Tip equilibrium position: {q_eq[3*SETTINGS['tip_node']:3*SETTINGS['tip_node']+3]}")

    decay_data_dir = join(data_dir, SETTINGS['decay_dir'])

    return model_save_dir, decay_data_dir, q_eq, u_eq

def plotDecayXYZData(SETTINGS, model_save_dir, Data, outdofs, PLOTS):
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

def getChartandReducedCoords(SETTINGS, model_save_dir, Data, svd_data, PLOTS):
    nTRAJ = len(Data['oData'])
    newData = deepcopy(Data)
    newData['etaDataTrunc'] = deepcopy(newData['oDataTrunc'])
    if SETTINGS['reduced_coordinates'] == "global":
        with open(join(SETTINGS['data_dir'], f"SSM_model_origin_{SETTINGS['observables']}.pkl"), "rb") as f:
            Vde = pickle.load(f)['model']['V']
        if SETTINGS['observables'] == "delay-embedding":
            for i in range(nTRAJ):
                newData['etaDataTrunc'][i][1] = Vde.T @ newData[svd_data][i][1]
        elif SETTINGS['observables'] == "pos-vel":
            for i in range(nTRAJ):
                newData['etaDataTrunc'][i][1] = Vde.T @ np.vstack((newData[svd_data][i][1], np.gradient(newData[svd_data][i][1], SETTINGS['dt'], axis=1)))
    else:
        print("====== Perform SVD on displacement field ======")
        show_modes = 9
        Xsnapshots = np.hstack([DataTrunc[1] for DataTrunc in newData[svd_data]])
        v, s = sparse_svd(Xsnapshots, up_to_mode=max(SETTINGS['SSMDim'], show_modes))
        if SETTINGS['observables'] == "delay-embedding":
            Vde = v[:, :SETTINGS['SSMDim']]
            for i in range(nTRAJ):
                newData['etaDataTrunc'][i][1] = Vde.T @ newData[svd_data][i][1]
        elif SETTINGS['observables'] == "pos-vel":
            assert SETTINGS['SSMDim'] % 2 == 0
            Vde = np.kron(np.eye(2), v[:, :SETTINGS['SSMDim']//2])
            for i in range(nTRAJ):
                newData['etaDataTrunc'][i][1] = Vde.T @ np.vstack((newData[svd_data][i][1], np.gradient(newData[svd_data][i][1], SETTINGS['dt'], axis=1)))
        else:
            raise RuntimeError("Unknown type of observables, should be ['delay-embedding', 'pos-vel']")
        
        if PLOTS:
            # Plot variance description: we expect the first couple of modes to capture almost all variance.
            # Note we assume data centered around the origin, which is the fixed point of our system.
            plot.pca_modes(s**2, up_to_mode=show_modes, show=(PLOTS == 'show'))
            if PLOTS == 'save':
                plt.savefig(join(model_save_dir, "plots", f"pca_modes.png"), bbox_inches='tight')
            # plot first three reduced coordinates
            plot.traj_xyz(newData,
                        xyz_idx=[('etaDataTrunc', 0), ('etaDataTrunc', 1), ('etaDataTrunc', 2)],
                        xyz_names=[r'$x_1$', r'$x_2$', r'$x_3$'], show=(PLOTS == 'show'))
            if PLOTS == 'save':
                plt.savefig(join(model_save_dir, "plots", f"decay_reduced_coords_123.png"), bbox_inches='tight')
    
    return Vde, newData

# TODO: Ensure hardcoded values are not present
# TODO: If not custom observables, use yDataTrunc. Otherwise, use oDataTrunc (which should be assembled prior)
def learnIMandRD(SETTINGS, Data, indTrain, custom_delay=False, custom_outdofs=None):
    if SETTINGS['use_ssmlearn'] == "matlab":
        # ====== Start Matlab engine and SSMLearn ====== #
        print("====== Start Matlab engine and SSMLearn ======")
        ssm = start_matlab_ssmlearn("/home/jalora/SSMR-for-control")
        # make data ready for Matlab
        yDataTruncTrain_matlab = numpy_to_matlab([Data['yDataTrunc'][i] for i in indTrain])
        etaDataTruncTrain_matlab = numpy_to_matlab([Data['etaDataTrunc'][i] for i in indTrain])
        IMInfo = ssm.IMGeometry(yDataTruncTrain_matlab, SETTINGS['SSMDim'], SETTINGS['SSMOrder'],
                                'reducedCoordinates', etaDataTruncTrain_matlab, 'l', SETTINGS['ridge_alpha']['manifold'])
        if SETTINGS['observables'] == "pos-vel":
            # IMInfo_paramonly = ssm.IMGeometry(yDataTruncTrain_matlab, SETTINGS['SSMDim'], SETTINGS['SSMOrder'],
            #             'reducedCoordinates', etaDataTruncTrain_matlab, 'style', 'custom', 'l', SETTINGS['ridge_alpha']['manifold'])
            
            # tanspace0_not_orth = IMInfo_paramonly['parametrization']['tangentSpaceAtOrigin']
            # tanspace0 = orth(tanspace0_not_orth)

            # Data['etaDataTruncNew'] = deepcopy(Data['etaDataTrunc'])
            # for iTraj in range(len(Data['etaDataTruncNew'])):
            #     Data['etaDataTruncNew'][iTraj][1] = np.transpose(tanspace0) @ tanspace0_not_orth @ Data['etaDataTrunc'][iTraj][1]
            # Data['etaDataTrunc'] = deepcopy(Data['etaDataTruncNew'])
            # Data.pop('etaDataTruncNew', None)
            # etaDataTruncTrain_matlab = numpy_to_matlab([Data['etaDataTrunc'][i] for i in indTrain])

            # IMInfo_paramonly = ssm.IMGeometry(yDataTruncTrain_matlab, SETTINGS['SSMDim'], SETTINGS['SSMOrder'],
            #             'reducedCoordinates', etaDataTruncTrain_matlab, 'style', 'custom', 'l', SETTINGS['ridge_alpha']['manifold'])
            
            # IMInfo_chartonly = ssm.IMGeometry(etaDataTruncTrain_matlab, SETTINGS['SSMDim'], SETTINGS['SSMOrder'],
            #             'reducedCoordinates', yDataTruncTrain_matlab, 'style', 'custom', 'l', SETTINGS['ridge_alpha']['manifold'])

            # # Define new Geometry
            # IMInfo = {'chart': IMInfo_chartonly['parametrization'], 'parametrization': IMInfo_paramonly['parametrization']}
            
            IMInfoInv = ssm.IMGeometry(etaDataTruncTrain_matlab, SETTINGS['SSMDim'], SETTINGS['SSMOrder'],
                                    'reducedCoordinates', yDataTruncTrain_matlab, 'l', SETTINGS['ridge_alpha']['manifold'])
            for key in ['map', 'polynomialOrder', 'dimension', 'nonlinearCoefficients', 'phi', 'exponents', 'H']:
                IMInfo['chart'][key] = IMInfoInv['parametrization'][key]
        
        # Rotate and regress IM and RD based on new delay coordinates
        if custom_delay:
            assert isinstance(custom_delay, int), 'Custom delay must take an integer value'

            nd = (custom_delay + 1) * SETTINGS['oDOF']

            # Modify yDataTrunc to only include last nd data points
            Data['yDataTruncNew'] = deepcopy(Data['yDataTrunc'])
            for iTraj in range(len(Data['yDataTrunc'])):
                Data['yDataTruncNew'][iTraj][1] = Data['yDataTrunc'][iTraj][1][-nd:, :]
            Data['yDataTrunc'] = deepcopy(Data['yDataTruncNew'])
            Data.pop('yDataTruncNew', None)

            # Setup regression between current reduced coordinates and new observable
            yDataTruncTrain_matlab = numpy_to_matlab([Data['yDataTrunc'][i] for i in indTrain])
            etaDataTruncTrain_matlab = numpy_to_matlab([Data['etaDataTrunc'][i] for i in indTrain])

            # Regress current reduced coordinates with new observable (used to infer the linear part of manifold, i.e., tangent space)
            IMInfo_paramonly = ssm.IMGeometry(yDataTruncTrain_matlab, SETTINGS['SSMDim'], SETTINGS['SSMOrder'],
                        'reducedCoordinates', etaDataTruncTrain_matlab, 'style', 'custom', 'l', SETTINGS['ridge_alpha']['manifold'])
            
            # Calculate tangent space at origin and orthogonalize
            tanspace0_not_orth = IMInfo_paramonly['parametrization']['tangentSpaceAtOrigin']
            tanspace0 = orth(tanspace0_not_orth)

            # Change reduced coordinates and repopulate eta
            Data['etaDataTruncNew'] = deepcopy(Data['etaDataTrunc'])
            for iTraj in range(len(Data['etaDataTruncNew'])):
                Data['etaDataTruncNew'][iTraj][1] = np.transpose(tanspace0) @ tanspace0_not_orth @ Data['etaDataTrunc'][iTraj][1]
            Data['etaDataTrunc'] = deepcopy(Data['etaDataTruncNew'])
            Data.pop('etaDataTruncNew', None)
            etaDataTruncTrain_matlab = numpy_to_matlab([Data['etaDataTrunc'][i] for i in indTrain])

            # Regress the new reduced coordinates with the new observable (used to infer the nonlinear part of manifold)
            IMInfo_paramonly = ssm.IMGeometry(yDataTruncTrain_matlab, SETTINGS['SSMDim'], SETTINGS['SSMOrder'],
                        'reducedCoordinates', etaDataTruncTrain_matlab, 'style', 'custom', 'l', SETTINGS['ridge_alpha']['manifold'])
            # IMInfo_chartonly = ssm.IMGeometry(etaDataTruncTrain_matlab, SETTINGS['SSMDim'], SETTINGS['SSMOrder'],
            #             'reducedCoordinates', yDataTruncTrain_matlab, 'style', 'custom', 'Ve', np.transpose(tanspace0))
            IMInfo_chartonly = ssm.IMGeometry(etaDataTruncTrain_matlab, nd, SETTINGS['SSMOrder'],
                        'reducedCoordinates', yDataTruncTrain_matlab, 'style', 'custom', 'l', SETTINGS['ridge_alpha']['manifold'])
            
            # Define new Geometry
            IMInfo = {'chart': IMInfo_chartonly['parametrization'], 'parametrization': IMInfo_paramonly['parametrization']}

        # SSM reduced dynamics
        RDInfo = ssm.IMDynamicsFlow(etaDataTruncTrain_matlab, 'R_PolyOrd', SETTINGS['ROMOrder'], 'style', 'default', 'l', SETTINGS['ridge_alpha']['reduced_dynamics'])
        # quit matlab engine
        ssm.quit()
        # convert matlab double arrays to numpy arrays
        matlab_info_dict_to_numpy(IMInfo)
        matlab_info_dict_to_numpy(RDInfo)
    elif SETTINGS['use_ssmlearn'] == "py":
        print("====== Using SSMLearnPy ======")
        from ssmlearnpy import SSMLearn
        ssm = SSMLearn(
            t=[Data['yDataTrunc'][i][0] for i in indTrain], 
            x=[Data['yDataTrunc'][i][1] for i in indTrain], 
            reduced_coordinates=[Data['etaDataTrunc'][i][1] for i in indTrain],
            ssm_dim=SETTINGS['SSMDim'], 
            dynamics_type=SETTINGS['RDType']
        )
        # find parametrization of SSM and reduced dynamics on SSM
        ssm.get_parametrization(poly_degree=SETTINGS['SSMOrder'], alpha=SETTINGS['ridge_alpha']['manifold'])    

        # Save relevant coeffs and params into dictss which resemble the outputs of the Matlab SSMLearn package
        IMInfo = {'parametrization': {
            'polynomialOrder': SETTINGS['SSMOrder'],
            'H': ssm.decoder.map_info['coefficients']
        }, 'chart': {}}

        if SETTINGS['observables'] == "pos-vel":
            ssm_inv = SSMLearn(
                t=[Data['etaDataTrunc'][i][0] for i in indTrain], 
                x=[Data['etaDataTrunc'][i][1] for i in indTrain], 
                reduced_coordinates=[Data['yDataTrunc'][i][1] for i in indTrain],
                ssm_dim=SETTINGS['SSMDim'], 
                dynamics_type=SETTINGS['RDType']
            )
            ssm_inv.get_parametrization(poly_degree=SETTINGS['SSMOrder'], alpha=SETTINGS['ridge_alpha']['manifold'])
            IMInfo['chart'] = {
                'polynomialOrder': SETTINGS['SSMOrder'],
                'H': ssm_inv.decoder.map_info['coefficients']
            }

            # # Regress current reduced coordinates with new observable (used to infer the linear part of manifold, i.e., tangent space)
            # ssm_paramonly = SSMLearn(
            # t=[Data['yDataTrunc'][i][0] for i in indTrain], 
            # x=[Data['yDataTrunc'][i][1] for i in indTrain], 
            # reduced_coordinates=[Data['etaDataTrunc'][i][1] for i in indTrain],
            # ssm_dim=SETTINGS['SSMDim'], 
            # dynamics_type=SETTINGS['RDType']
            # )
            # ssm_paramonly.get_parametrization(poly_degree=SETTINGS['SSMOrder'], alpha=SETTINGS['ridge_alpha']['manifold'])

            # # Calculate tangent space at origin and orthogonalize
            # tanspace0_not_orth = ssm_paramonly.decoder.map_info['coefficients'][:SETTINGS['SSMDim'], :SETTINGS['SSMDim']]
            # tanspace0 = orth(tanspace0_not_orth)

            # # Change reduced coordinates and repopulate eta
            # Data['etaDataTruncNew'] = deepcopy(Data['etaDataTrunc'])
            # for iTraj in range(len(Data['etaDataTruncNew'])):
            #     Data['etaDataTruncNew'][iTraj][1] = np.transpose(tanspace0) @ tanspace0_not_orth @ Data['etaDataTrunc'][iTraj][1]
            # Data['etaDataTrunc'] = deepcopy(Data['etaDataTruncNew'])
            # Data.pop('etaDataTruncNew', None)

            # # Regress the new reduced coordinates with the new observable (used to infer the nonlinear part of manifold)
            # # Get the parameterization
            # ssm_paramonly = SSMLearn(
            # t=[Data['yDataTrunc'][i][0] for i in indTrain], 
            # x=[Data['yDataTrunc'][i][1] for i in indTrain], 
            # reduced_coordinates=[Data['etaDataTrunc'][i][1] for i in indTrain],
            # ssm_dim=SETTINGS['SSMDim'], 
            # dynamics_type=SETTINGS['RDType']
            # )
            # ssm_paramonly.get_parametrization(poly_degree=SETTINGS['SSMOrder'], alpha=SETTINGS['ridge_alpha']['manifold'])

            # # Construct the chart
            # ssm_chartonly = SSMLearn(
            #     t=[Data['etaDataTrunc'][i][0] for i in indTrain], 
            #     x=[Data['etaDataTrunc'][i][1] for i in indTrain], 
            #     reduced_coordinates=[Data['yDataTrunc'][i][1] for i in indTrain],
            #     ssm_dim=SETTINGS['SSMDim'], 
            #     dynamics_type=SETTINGS['RDType']
            # )
            # ssm_chartonly.get_parametrization(poly_degree=SETTINGS['SSMOrder'], alpha=SETTINGS['ridge_alpha']['manifold'])

            # IMInfo = {
            #     'parametrization': {
            #     'polynomialOrder': SETTINGS['SSMOrder'],
            #     'H': ssm_paramonly.decoder.map_info['coefficients']
            #     }, 
            #     'chart': {
            #     'polynomialOrder': SETTINGS['SSMOrder'],
            #     'H': ssm_chartonly.decoder.map_info['coefficients']
            #     }
            # }
            
            # # Reassign to get new rotated reduced dynamics
            # ssm = ssm_paramonly
        
        if custom_delay:
            assert isinstance(custom_delay, int), 'Custom delay must take an integer value'
            if custom_outdofs is None:
                nd = (custom_delay + 1) * SETTINGS['oDOF']
            else:
                # TODO: Assume we have the observables were trying to delay in here
                nd = (custom_delay + 1) * len(Data['oDataTrunc'][0][1][:, 0])

            # Get new delayed observables and rotate reduced coordinates
            Data['yDataTruncNew'] = deepcopy(Data['yDataTrunc'])
            for iTraj in range(len(Data['yDataTrunc'])):
                if custom_outdofs is None:
                    Data['yDataTruncNew'][iTraj][1] = Data['yDataTrunc'][iTraj][1][-nd:, :]
                else:
                    Data['yDataTruncNew'][iTraj][1] = Data['yDataTrunc'][iTraj][1][custom_outdofs, :]
            Data['yDataTrunc'] = deepcopy(Data['yDataTruncNew'])
            Data.pop('yDataTruncNew', None)

            # Regress current reduced coordinates with new observable (used to infer the linear part of manifold, i.e., tangent space)
            ssm_paramonly = SSMLearn(
            t=[Data['yDataTrunc'][i][0] for i in indTrain], 
            x=[Data['yDataTrunc'][i][1] for i in indTrain], 
            reduced_coordinates=[Data['etaDataTrunc'][i][1] for i in indTrain],
            ssm_dim=SETTINGS['SSMDim'], 
            dynamics_type=SETTINGS['RDType']
            )
            ssm_paramonly.get_parametrization(poly_degree=SETTINGS['SSMOrder'], alpha=SETTINGS['ridge_alpha']['manifold'])

            # Calculate tangent space at origin and orthogonalize
            tanspace0_not_orth = ssm_paramonly.decoder.map_info['coefficients'][:SETTINGS['SSMDim'], :SETTINGS['SSMDim']]
            tanspace0 = orth(tanspace0_not_orth)

            # Change reduced coordinates and repopulate eta
            Data['etaDataTruncNew'] = deepcopy(Data['etaDataTrunc'])
            for iTraj in range(len(Data['etaDataTruncNew'])):
                Data['etaDataTruncNew'][iTraj][1] = np.transpose(tanspace0) @ tanspace0_not_orth @ Data['etaDataTrunc'][iTraj][1]
            Data['etaDataTrunc'] = deepcopy(Data['etaDataTruncNew'])
            Data.pop('etaDataTruncNew', None)

            # Regress the new reduced coordinates with the new observable (used to infer the nonlinear part of manifold)
            # Get the parameterization
            ssm_paramonly = SSMLearn(
            t=[Data['yDataTrunc'][i][0] for i in indTrain], 
            x=[Data['yDataTrunc'][i][1] for i in indTrain], 
            reduced_coordinates=[Data['etaDataTrunc'][i][1] for i in indTrain],
            ssm_dim=SETTINGS['SSMDim'], 
            dynamics_type=SETTINGS['RDType']
            )
            ssm_paramonly.get_parametrization(poly_degree=SETTINGS['SSMOrder'], alpha=SETTINGS['ridge_alpha']['manifold'])

            # Construct the chart
            ssm_chartonly = SSMLearn(
                t=[Data['etaDataTrunc'][i][0] for i in indTrain], 
                x=[Data['etaDataTrunc'][i][1] for i in indTrain], 
                reduced_coordinates=[Data['yDataTrunc'][i][1] for i in indTrain],
                ssm_dim=SETTINGS['SSMDim'], 
                dynamics_type=SETTINGS['RDType']
            )
            ssm_chartonly.get_parametrization(poly_degree=SETTINGS['SSMOrder'], alpha=SETTINGS['ridge_alpha']['manifold'])

            IMInfo = {
                'parametrization': {
                'polynomialOrder': SETTINGS['SSMOrder'],
                'H': ssm_paramonly.decoder.map_info['coefficients']
                }, 
                'chart': {
                'polynomialOrder': SETTINGS['SSMOrder'],
                'H': ssm_chartonly.decoder.map_info['coefficients']
                }
            }
            
            # Reassign to get new rotated reduced dynamics
            ssm = ssm_paramonly
        
        # Construct reduced dynamics
        ssm.get_reduced_dynamics(poly_degree=SETTINGS['ROMOrder'], alpha=SETTINGS['ridge_alpha']['reduced_dynamics'])
        RDInfo = {
            'reducedDynamics': {
                'polynomialOrder': SETTINGS['ROMOrder'],
                'coefficients': ssm.reduced_dynamics.map_info['coefficients'],
            },
            'eigenvaluesLinPartFlow': ssm.reduced_dynamics.map_info['eigenvalues_lin_part'],
            'dynamicsType': SETTINGS['RDType']
        }

    return IMInfo, RDInfo, Data

def analyzeSSMErrors(model_save_dir, IMInfo, RDInfo, Data, indTrain, indTest, outdofs, PLOTS):
    trajRec = {}
    # geometry error
    meanErrorGeo = {}
    trajRec['geo'] = lift_trajectories(IMInfo, Data['etaDataTrunc'])
    normedTrajDist = compute_trajectory_errors(trajRec['geo'], Data['yDataTrunc'])[0] * 100
    meanErrorGeo['Train'] = np.mean(normedTrajDist[indTrain])
    meanErrorGeo['Test'] = np.mean(normedTrajDist[indTest])
    print(f"Average parametrization train error: {meanErrorGeo['Train']:.4e}")
    print(f"Average parametrization test error: {meanErrorGeo['Test']:.4e}")
    # plot comparison of SSM-predicted vs. actual test trajectories
    axs = plot.traj_xyz(Data,
                        xyz_idx=[('yData', outdofs[0]), ('yData', outdofs[1]), ('yData', outdofs[2])],
                        xyz_names=[r'$x$ [mm]', r'$y$ [mm]', r'$z$ [mm]'],
                        traj_idx=indTest,
                        show=False, t_shift=Data['yData'][0][0][0])
    plot.traj_xyz(trajRec,
                xyz_idx=[('geo', outdofs[0]), ('geo', outdofs[1]), ('geo', outdofs[2])],
                xyz_names=[r'$x$ [mm]', r'$y$ [mm]', r'$z$ [mm]'],
                traj_idx=indTest,
                axs=axs, ls=':', color='darkblue', show=(PLOTS == 'show'))
    if PLOTS == 'save':
        plt.savefig(join(model_save_dir, "plots", f"geometry_error.png"), bbox_inches='tight')
    # reduced dynamics error
    meanErrorDyn = {}
    trajRec['rd'] = advectRD(RDInfo, Data['etaDataTrunc'])[0]
    normedTrajDist = compute_trajectory_errors(trajRec['rd'], Data['etaDataTrunc'])[0] * 100
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
    trajRec['glob'] = lift_trajectories(IMInfo, trajRec['rd'])
    normedTrajDist = compute_trajectory_errors(trajRec['glob'], Data['yDataTrunc'])[0] * 100
    meanErrorGlo['Train'] = np.mean(normedTrajDist[indTrain])
    meanErrorGlo['Test'] = np.mean(normedTrajDist[indTest])
    print(f"Average global train error: {meanErrorGlo['Train']:.4f}")
    print(f"Average global test error: {meanErrorGlo['Test']:.4f}")
    axs = plot.traj_xyz(Data,
                        xyz_idx=[('yData', outdofs[0]), ('yData', outdofs[1]), ('yData', outdofs[2])],
                        xyz_names=[r'$x$', r'$y$', r'$z$'],
                        traj_idx=indTest,
                        show=False, t_shift=Data['yData'][0][0][0])
    plot.traj_xyz(trajRec,
                xyz_idx=[('glob', outdofs[0]), ('glob', outdofs[1]), ('glob', outdofs[2])],
                xyz_names=[r'$x$', r'$y$', r'$z$'],
                traj_idx=indTest,
                axs=axs, ls=':', color='darkblue', show=(PLOTS == 'show'))
    if PLOTS == 'save':
        plt.savefig(join(model_save_dir, "plots", f"global_error.png"), bbox_inches='tight')

def learnBmatrix(SETTINGS, Wauton, Rauton, z, u, t, u_eq, embed_coords=[0, 1, 2], hardware=False):
    u = (u.T - u_eq).T
    if SETTINGS['observables'] == "delay-embedding":
        y = None
        # TODO: This is trying to organize batch data for hardware; currently not used
        if hardware:
            for i in range(z.reshape(-1,1).shape[1]):
                yi = delayEmbedding(z[:, i], embed_coords=embed_coords,
                                up_to_delay=SETTINGS['custom_delay'] if SETTINGS['custom_delay'] else SETTINGS['n_delay'])
                if y is None:
                    y = yi
                else:
                    y = np.hstack([y, yi])
        else:
            y = delayEmbedding(z, embed_coords=embed_coords, 
                          up_to_delay=SETTINGS['custom_delay'] if SETTINGS['custom_delay'] else SETTINGS['n_delay'])
    else:
        y = np.vstack([z, np.gradient(z, SETTINGS['dt'], axis=1)])
    x = Wauton(y)
    # train/test split on input training data
    split_idx = int(SETTINGS['input_train_ratio'] * len(t))
    t_train, t_test = t[:split_idx], t[split_idx:]
    _, z_test = z[:, :split_idx], z[:, split_idx:]
    # y_train, y_test = y[:, :split_idx], y[:, split_idx:]
    u_train, u_test = u[:, :split_idx], u[:, split_idx:]
    x_train, x_test = x[:, :split_idx], x[:, split_idx:]

    # autonomous reduced dynamics vs. numerical derivative
    dxdt = np.gradient(x_train, SETTINGS['dt'], axis=1)
    dxdt_ROM = Rauton(x_train)

    # ====== regress B matrix ====== #
    # Regress using linear features
    assemble_features = lambda u, x: phi(u, order=SETTINGS['poly_u_order']) # utils.phi(np.vstack([u, x]), order=SETTINGS['poly_u_order']) # 
    
    # Regress using nonlinear features
    # assemble_features = lambda u, x: phi_control(u, x, SETTINGS['poly_u_order'], 1)

    X = assemble_features(u_train, x_train)
    B_learn = regress_B(X, dxdt, dxdt_ROM, alpha=SETTINGS['ridge_alpha']['B'], method='ridge')
    print(f"Frobenius norm of B_learn: {np.linalg.norm(B_learn, ord='fro'):.4f}")

    R = lambda x, u: Rauton(np.atleast_2d(x)) + B_learn @ assemble_features(u, x)

    return {'t_train': t_train, 't_test': t_test, 'z_test': z_test, 'u_train': u_train, 'u_test': u_test, 
            'x_train': x_train, 'x_test': x_test, 'dxdt': dxdt, 'dxdt_ROM': dxdt_ROM,'B_learn': B_learn, 'R': R}

def fitControlMatrix(Rauton, x_train, u_train, dt, alpha=0.0):
    # autonomous reduced dynamics vs. numerical derivative
    dxdt = np.gradient(x_train, dt, axis=1)
    dxdt_ROM = Rauton(x_train)

    # ====== regress B matrix ====== #
    X = phi(u_train, 1)
    B_learn = regress_B(X, dxdt, dxdt_ROM, alpha=alpha, method='ridge')
    print(f"Frobenius norm of B_learn: {np.linalg.norm(B_learn, ord='fro'):.4f}")

    return B_learn

def plot_gradients(t_train, dxdt, dxdt_ROM, dxDt_ROM_with_B, PLOTS, model_save_dir):
    plot_reduced_coords = np.s_[:] # [3, 4, 5]
    plot.reduced_coordinates_gradient(t_train, [dxdt[plot_reduced_coords, :], dxdt_ROM[plot_reduced_coords, :], dxDt_ROM_with_B[plot_reduced_coords, :]],
                                    labels=["true numerical", "predicted autonomous", "predicted with inputs"], how="norm", show=(PLOTS == 'show'))
    if PLOTS == "save":
        plt.savefig(join(model_save_dir, "plots", f"reduced_coordinates_gradient.png"), bbox_inches='tight')

def analyzeOLControlPredict(SETTINGS, model_save_dir, controlData, q_eq, u_eq, Wauton, R, Vauton, embed_coords,
                            traj_coords=[0, 1, 2], PLOTS=False, file_type='pkl', hardware='False'):
    test_results = {}
    test_trajectories = [{
            'name': "like training data",
            't': controlData['t_test'],
            'z': controlData['z_test'],
            'u': controlData['u_test'],
            'x': controlData['x_test']
        }]
    for test_traj in SETTINGS['input_test_data_dir']:
        traj_dir = join(SETTINGS['data_dir'], test_traj)
        (t, z), u = import_pos_data(data_dir=traj_dir,
                                    rest_file=None, # join(SETTINGS['robot_dir'], SETTINGS['rest_file']),
                                    q_rest = q_eq,
                                    output_node=SETTINGS['tip_node'], 
                                    return_inputs=True, 
                                    traj_index=0, file_type=file_type, hardware=hardware)

        u = (u.T - u_eq).T
        if SETTINGS['observables'] == "delay-embedding":
            y = delayEmbedding(z, embed_coords=embed_coords, 
                              up_to_delay=SETTINGS['custom_delay'] if SETTINGS['custom_delay'] else SETTINGS['n_delay'])
        else:
            y = np.vstack([z, np.gradient(z, SETTINGS['dt'], axis=1)])
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
            z_pred = predict_open_loop(R, Vauton, traj['t'], traj['u'], x0=traj['x'][:, 0], method='Radau')
        except Exception as e:
            z_pred = np.nan * np.ones_like(traj['z'])
        idx_end = np.where(np.isnan(z_pred))[1][0] if np.any(np.isnan(z_pred)) else len(traj['t'])
        rmse = float(np.sum(np.sqrt(np.mean((z_pred[:3, :idx_end] - traj['z'][:3, :idx_end])**2, axis=0))) / len(traj['t']))
        print(f"({traj['name']}): RMSE = {rmse:.4f}")
        test_results[traj['name']] = {
            'RMSE': rmse
        }
        if PLOTS:
            axs = plot.traj_xyz_txyz(traj['t'][:idx_end],
                                    z_pred[embed_coords[0], :idx_end], z_pred[embed_coords[1], :idx_end], z_pred[embed_coords[2], :idx_end],
                                    show=False)
            axs = plot.traj_xyz_txyz(traj['t'][:idx_end],
                                    traj['z'][traj_coords[0], :idx_end], traj['z'][traj_coords[1], :idx_end], traj['z'][traj_coords[2], :idx_end],
                                    color="tab:orange", axs=axs, show=(PLOTS == 'show'))
            axs[-1].legend(["Predicted trajectory", "Actual trajectory"])
            if PLOTS == 'save':
                plt.savefig(join(model_save_dir, "plots", f"open-loop-prediction_{traj['name']}.png"), bbox_inches='tight')
    print(f"(overall): RMSE = {np.mean([test_results[traj]['RMSE'] for traj in test_results]):.4f}")

    return test_results

def analyze_open_loop(test_data, model_save_dir, controlData, Wauton, R, Vauton, embed_coords, 
                            traj_coords=[0, 1, 2], PLOTS=False):
    
    t_test, y_test, z_test, u_test = test_data
    n_test = len(y_test)
    test_results = {}
    test_trajectories = [{
            'name': "like training data",
            't': controlData['t_test'],
            'z': controlData['z_test'],
            'u': controlData['u_test'],
            'x': controlData['x_test']
        }]
    for idx_test in range(n_test):
        t, z, y, u = t_test[idx_test], z_test[idx_test], y_test[idx_test], u_test[idx_test]

        test_trajectories.append({
                'name': idx_test,
                't': t,
                'z': z,
                'u': u,
                'x': Wauton(y)
            })
    for traj in test_trajectories:
        try:
            z_pred = predict_open_loop(R, Vauton, traj['t'], traj['u'], x0=traj['x'][:, 0], method='Radau')
        except Exception as e:
            z_pred = np.nan * np.ones_like(traj['z'])
        idx_end = np.where(np.isnan(z_pred))[1][0] if np.any(np.isnan(z_pred)) else len(traj['t'])
        rmse = float(np.sum(np.sqrt(np.mean((z_pred[:3, :idx_end] - traj['z'][:3, :idx_end])**2, axis=0))) / len(traj['t']))
        print(f"({traj['name']}): RMSE = {rmse:.4f}")
        test_results[traj['name']] = {
            'RMSE': rmse
        }
        if PLOTS:
            axs = plot.traj_xyz_txyz(traj['t'][:idx_end],
                                    z_pred[embed_coords[0], :idx_end], z_pred[embed_coords[1], :idx_end], z_pred[embed_coords[2], :idx_end],
                                    show=False)
            axs = plot.traj_xyz_txyz(traj['t'][:idx_end],
                                    traj['z'][traj_coords[0], :idx_end], traj['z'][traj_coords[1], :idx_end], traj['z'][traj_coords[2], :idx_end],
                                    color="tab:orange", axs=axs, show=(PLOTS == 'show'))
            axs[-1].legend(["Predicted trajectory", "Actual trajectory"])
            if PLOTS == 'save':
                plt.savefig(join(model_save_dir, "plots", f"open-loop-prediction_{traj['name']}.png"), bbox_inches='tight')
    print(f"(overall): RMSE = {np.mean([test_results[traj]['RMSE'] for traj in test_results]):.4f}")

    return test_results