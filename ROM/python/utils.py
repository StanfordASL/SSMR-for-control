import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from scipy.integrate import solve_ivp
from copy import deepcopy
import pickle
from os.path import join
from os import listdir
from scipy.sparse.linalg import svds
# import matlab.engine
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import yaml
from scipy.io import loadmat

# from sympy.polys.monomials import itermonomials
# from sympy.polys.orderings import monomial_key
# import sympy as sp


def slice_trajectories(data, interval: list):
    dataTrunc = deepcopy(data)
    ntraj = len(data)
    for traj in range(ntraj):
        # truncate time array
        dataTrunc[traj][1] = data[traj][1][:, (data[traj][0] >= interval[0]) & (data[traj][0] <= interval[1])]
        # truncate trajectory array
        dataTrunc[traj][0] = data[traj][0][(data[traj][0] >= interval[0]) & (data[traj][0] <= interval[1])]
    return dataTrunc


def phi(x, order: int):
    # poly = PolynomialFeatures(degree=SSMOrder, include_bias=False).fit(x.T)
    # # print(poly.get_feature_names_out(['a', 'b', 'c', 'd', 'e', 'f']))
    # features = poly.transform(x.T)
    if not isinstance(order, int):
        order = int(order)
    return PolynomialFeatures(degree=order, include_bias=False).fit_transform(x.T).T
    # dim = x.shape[0]
    # zeta = sp.Matrix(sp.symbols('x1:{}'.format(dim + 1)))
    # polynoms = sorted(itermonomials(list(zeta), order),
    #                     key=monomial_key('grevlex', list(reversed(zeta))))
    # polynoms = polynoms[1:]

    # return sp.lambdify(zeta, polynoms)(*x[:, 0]) # , modules=[jnp, jsp.special])


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


def delayEmbedding(undelayedData, embed_coords=[0, 1, 2], up_to_delay=4):    
    undelayed = undelayedData[embed_coords, :]
    buf = [undelayed]
    for delta in list(range(1, up_to_delay+1)):
        delayed_by_delta = np.roll(undelayed, delta)
        delayed_by_delta[:, :delta] = np.tile(undelayed[:, 0], (delta, 1)).T
        buf.append(delayed_by_delta)
    delayedData = np.vstack(buf[::-1])
    return delayedData

# TODO: We should generalize this to more than just the tip
def import_pos_data(data_dir, rest_file=None, q_rest=None, output_node=None, 
                    t_in=0, t_out=None, return_inputs=False, return_velocity=False, 
                    traj_index=np.s_[:], file_type='pkl', subsample=1, shift=False):
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
        files = sorted([f for f in listdir(data_dir) if 'snapshots.pkl' in f or 'sim.pkl' in f])
    elif file_type == 'mat':
        files = sorted([f for f in listdir(data_dir) if 'snapshots.mat' in f])
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    files = files[traj_index]
    if not isinstance(files, list):
        files = [files]
    out_data = []
    u = []
    for i, traj in enumerate(files):
        # Setup output nodes
        if output_node == 'all':
            node_slice = np.s_[:]
        else:
            node_slice = np.s_[3*output_node:3*output_node+3]
        
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
                            data_traj_trunc = data['oData'][i][1][0:3, ind] - q_rest[node_slice].reshape((3,1))
                        else:
                            data_traj_trunc = data['oData'][i][1][0:3, ind]

                    data_traj = data_traj_trunc[:, ::subsample]
                    out_data.append([t, data_traj])
                    
                    if return_inputs:
                        uTrunc = np.array(data['u'])[ind, :].T
                        u.append(uTrunc[:, ::subsample])
    if len(out_data) == 1:
        out_data, u = out_data[0], u[0]
    if return_inputs:
        return out_data, u
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
    zTraj = yTraj[-3:, :]
    # if integration unsuccessful, fill up solution with nans
    if zTraj.shape[1] < len(t):
        zTraj = np.hstack((zTraj, np.tile(np.nan, (3, len(t) - zTraj.shape[1]))))
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

def save_ssm_model(model_save_dir, RDInfo, IMInfo, B_learn, Vde, q_eq, u_eq, settings, test_results):
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
        SSM_model['model']['V'] = Vde
        SSM_model['model']['v_coeff'] = None
        SSM_model['params']['delays'] = settings['n_delay']
        SSM_model['params']['delay_embedding'] = True
        SSM_model['params']['obs_dim'] = 3 * (1 + settings['n_delay'])
        SSM_model['params']['output_dim'] = settings['oDOF']
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

def advect_adiabaticRD_with_inputs(t, y0, u, y_target, interpolator, know_target, ROMOrder=3, SSMDim=6, interpolate="xyz", interp_slice=None):
    transform = interpolator.transform  # timed_transform
    if interp_slice is None:
        if interpolate == "xyz":
            interp_slice = np.s_[:3]
        elif interpolate == "xy":
            interp_slice = np.s_[:2]
        elif interpolate == "reduced_coords":
            interp_slice = np.s_[:SSMDim]
    dt = 0.01
    interpolant_len = len(np.zeros(10)[interp_slice])
    N = len(t)-1
    y_pred = np.full((len(y0), N+1), np.nan)
    y_pred[:, 0] = y0
    x = np.full((SSMDim, N+1), np.nan)
    # x[0] = V_0^T @ (y[0] - y_bar[0])
    x[:, 0] = interpolator.transform(np.zeros(interpolant_len), "V").T @ y0
    y_bar = np.full((len(y0), N+1), np.nan)
    x_bar = np.full((SSMDim, N+1), np.nan)
    u_bar = np.full((u.shape[0], N+1), np.nan)
    xdot = np.full((SSMDim, N+1), np.nan)
    weights = np.full((1, N+1), np.nan)

    for i in range(N):
        # compute the weights used at this timestep
        try:
            if interpolate in ["xyz", "xy"]:
                if know_target:
                    psi = y_target[:, i]
                else:
                    psi = y0
            elif interpolate == "reduced_coords":
                if know_target:
                    psi = interpolator.transform(np.zeros(interpolant_len), "V").T @ y_target[:, i]
                else:
                    psi = x[:, i]
            else:
                raise ValueError("interpolate must be one of 'xyz', 'xy', or 'reduced_coords'")
            psi = psi[interp_slice]
            # compute and integrate the dynamics at this timestep
            u_bar[:, i] = transform(psi, 'u_bar')
            if interpolate in ["xyz", "xy"]:
                y_bar[:, i] = np.tile(transform(psi, 'q_bar'), 1 + 4)
                # xdot[i] = R(x[i]) + B[i] @ (u[i] - u_bar[i])
                xdot[:, i] = (transform(psi, 'r_coeff') @ phi(x[:, i].reshape(-1, 1), ROMOrder)).flatten() + transform(psi, 'B_r') @ (u[:, i] - u_bar[:, i])
                # y[i+1] = W(x[i+1]) + y_bar[i]
                # forward Euler: x[i+1] = x[i] + dt * xdot[i]
                x[:, i+1] = x[:, i] + dt * xdot[:, i]
                y_pred[:, i+1] = (transform(psi, 'w_coeff') @ phi(x[:, i+1].reshape(-1, 1), 3)).T + y_bar[:, i]
            elif interpolate == "reduced_coords":
                x_bar[:, i] = transform(psi, 'x_bar')
                y_bar[:, i] = np.tile(transform(psi, 'q_bar'), 1 + 4)
                # xdot[i] = R(x[i] - x_bar[i]) + B[i] @ (u[i] - u_bar[i])
                xdot[:, i] = (transform(psi, 'r_coeff') @ phi((x[:, i] - x_bar[:, i]).reshape(-1, 1), ROMOrder)).flatten() + transform(psi, 'B_r') @ (u[:, i] - u_bar[:, i])
                # forward Euler: x[i+1] = x[i] + dt * xdot[i]
                x[:, i+1] = x[:, i] + dt * xdot[:, i]
                # y[i+1] = W(x[i+1] - x_bar[i])
                y_pred[:, i+1] = (transform(psi, 'w_coeff') @ phi((x[:, i+1] - x_bar[:, i]).reshape(-1, 1), ROMOrder)).T + y_bar[:, i]

        except Exception as e:
            # break
            raise e
    return t, x, y_pred, xdot, y_bar if interpolate in ["xy", "xyz"] else x_bar, u_bar, weights


def cart2sph(x, y, z):
    # z_shift = z - 195
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    sph = [az, el, r]
    return sph