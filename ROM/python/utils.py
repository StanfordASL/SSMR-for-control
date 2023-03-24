import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.integrate import solve_ivp


def slice_trajectories(data, interval: list):
    dataTrunc = data.copy()
    ntraj = len(data)
    for traj in range(ntraj):
        # truncate time array
        dataTrunc[traj, 1] = data[traj, 1][:, (data[traj, 0] >= interval[0]) & (data[traj, 0] <= interval[1])]
        # truncate trajectory array
        dataTrunc[traj, 0] = data[traj, 0][(data[traj, 0] >= interval[0]) & (data[traj, 0] <= interval[1])]
    return dataTrunc


def multivariate_polynomial(x, order: int):
    # poly = PolynomialFeatures(degree=SSMOrder, include_bias=False).fit(x.T)
    # # print(poly.get_feature_names_out(['a', 'b', 'c', 'd', 'e', 'f']))
    # features = poly.transform(x.T)
    if not isinstance(order, int):
        order = int(order)
    return PolynomialFeatures(degree=order, include_bias=False).fit_transform(x.T).T


def lift_trajectories(IMInfo: dict, etaData):
    H = np.array(IMInfo['parametrization']['H'])
    SSMOrder = IMInfo['parametrization']['polynomialOrder']
    phi = lambda x: multivariate_polynomial(x, SSMOrder)
    map = lambda x: H @ phi(x)
    return transform_trajectories(map, etaData)


def transform_trajectories(map, inData):
    nTraj = inData.shape[0]
    outData = inData.copy()
    for i in range(nTraj):
        # outData[0][i] = inData[0][i];
        outData[i, 1] = map(inData[i, 1])
    return outData


def compute_trajectory_errors(yData1, yData2, inds='all'):
    
    nTraj = yData1.shape[0]
    trajDim = yData1[0, 1].shape[0]
    trajErrors = np.zeros(nTraj)
    ampErrors = np.zeros(nTraj)

    if inds == 'all':
        inds = list(range(trajDim))
        # check if trajectories have same dimensionality (i.e. the same number of observables)
        assert yData1[0, 1].shape[0] == yData2[0, 1].shape[0], 'Dimensionalities of trajectories are inconsistent'

    for i in range(nTraj):
        # check if trajectories have same length
        if yData1[0, 1].shape[1] == yData1[0, 1].shape[1]:
            trajErrors[i] = np.mean(np.linalg.norm(yData1[i, 1][inds, :] - yData2[i, 1][inds, :], axis=0)) / np.amax(np.linalg.norm(yData2[i, 1][inds, :], axis=0))
            ampErrors[i] = np.mean(np.abs(np.linalg.norm(yData1[i, 1][inds, :], axis=0) - np.linalg.norm(yData2[i, 1][inds, :]))) / np.mean(np.linalg.norm(yData2[i, 1][inds, :], axis=0))
        else:
            # integration has failed
            trajErrors[i] = np.inf
            ampErrors[i] = np.inf
    return trajErrors, ampErrors


def advectRD(RDInfo, etaData):
    assert RDInfo['dynamicsType'] == 'flow', "Issue: only flow type dynamics implemented"
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
    nTraj = etaData.shape[0]
    etaRec = etaData.copy()
    for i in range(nTraj):
        tStart = etaData[i, 0][0]
        tEnd = etaData[i, 0][-1]
        nSamp = len(etaData[i, 0])
        sol = solve_ivp(flow,
                        t_span=[tStart, tEnd],
                        t_eval=np.linspace(tStart, tEnd, nSamp),
                        y0=etaData[i, 1][:, 0],
                        method='DOP853',
                        vectorized=True,
                        rtol=1e-3,
                        atol=1e-3)
        etaRec[i, 0] = sol.t
        etaRec[i, 1] = sol.y
    return etaRec


def Rauton(RDInfo):
    W_r = np.array(RDInfo['reducedDynamics']['coefficients'])
    polynomialOrder = RDInfo['reducedDynamics']['polynomialOrder']
    phi = lambda x: multivariate_polynomial(x, polynomialOrder)
    Rauton = lambda y: W_r @ phi(y)
    return Rauton