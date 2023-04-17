import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.integrate import solve_ivp
from copy import deepcopy
import pickle
import os

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


def multivariate_polynomial(x, order: int):
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
    phi = lambda x: multivariate_polynomial(x, SSMOrder)
    map = lambda x: H @ phi(x)
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
    polynomialOrder = int(RDInfo['conjugateDynamics']['polynomialOrder'])
    phi = lambda x: multivariate_polynomial(x, polynomialOrder)
    N = lambda t, y: W_r @ phi(np.atleast_2d(y))
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
                        method='DOP853',
                        vectorized=True,
                        rtol=1e-3,
                        atol=1e-3)
        etaRec[i][0] = sol.t
        etaRec[i][1] = sol.y
    return etaRec


def Rauton(RDInfo):
    W_r = np.array(RDInfo['reducedDynamics']['coefficients'])
    polynomialOrder = RDInfo['reducedDynamics']['polynomialOrder']
    phi = lambda x: multivariate_polynomial(x, polynomialOrder)
    Rauton = lambda x: W_r @ phi(x) # / 10
    return Rauton


def delayEmbedding(undelayedData, embed_coords=[0, 1, 2], up_to_delay=4):    
    undelayed = undelayedData[embed_coords, :]
    buf = [undelayed]
    for delta in list(range(1, up_to_delay+1)):
        delayed_by_delta = np.roll(undelayed, -delta)
        delayed_by_delta[:, -delta:] = 0
        buf.append(delayed_by_delta)
    delayedData = np.vstack(buf)
    return delayedData


def import_pos_data(data_dir, rest_file, output_node, t_in=0, t_out=None, return_inputs=False, return_velocity=False):
    with open(rest_file, 'rb') as file:
        rest_file = pickle.load(file)
        try:
            q_rest = rest_file['q'][0]
        except:
            q_rest = rest_file['rest']
    files = sorted([f for f in os.listdir(data_dir) if '.pkl' in f]) # [0:1]
    out_data = []
    u = []
    for i, traj in enumerate(files):
        # if i % 3 != 0:
        #     continue
        with open(os.path.join(data_dir, traj), 'rb') as file:
            data = pickle.load(file)
        t = np.array(data['t'])        
        if output_node == 'all':
            node_slice = np.s_[:]
        else:
            node_slice = np.s_[3*output_node:3*output_node+3]
        if len(data['q']) > 0:
            q_node = (np.array(data['q'])[:, node_slice] - q_rest[node_slice]).T
        elif len(data['z']) > 0:
            q_node = (np.array(data['z'])[:, 3:] - q_rest[node_slice]).T
        else:
            raise RuntimeError("Cannot find data for desired node")
        # remove time until t_in
        ind = (t_in <= t)
        if t_out is not None:
            ind = ind & (t <= t_out)
        t = t[ind] - t_in
        q_node = q_node[:, ind]
        if return_velocity:
            v_node = np.array(data['v'])[:, node_slice].T
            data_traj = np.hstack((q_node, v_node))
        else:
            data_traj = q_node
        out_data.append([t, data_traj])
        if return_inputs:
            u.append(np.array(data['u']).T)
    if len(out_data) == 1:
        # print("Only one trajectory found in folder")
        out_data, u = out_data[0], u[0]
    if return_inputs:
        return out_data, u
    else:
        return out_data
