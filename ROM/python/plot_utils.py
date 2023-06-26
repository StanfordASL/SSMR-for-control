import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': 'FreeSerif'})
plt.rcParams.update({'mathtext.fontset': 'cm'})

FONTSCALE = 1.2

plt.rc('font', size=12*FONTSCALE)          # controls default text sizes
plt.rc('axes', titlesize=15*FONTSCALE)     # fontsize of the axes title
plt.rc('axes', labelsize=13*FONTSCALE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12*FONTSCALE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12*FONTSCALE)    # fontsize of the tick labels
plt.rc('legend', fontsize=11*FONTSCALE)    # legend fontsize
plt.rc('figure', titlesize=15*FONTSCALE)   # fontsize of the figure title
suptitlesize = 20*FONTSCALE

plt.rc('figure', autolayout=True)
# plt.rc('axes', xmargin=0)

# PADDING = {
#     'w_pad': 1.0,
#     'h_pad': 1.0
# }

COLORS = ['tab:blue', 'tab:orange', 'tab:green', "tab:red", "tab:purple", "tab:brown", "black", "cyan"]

TRAJ_COLORMAP = plt.cm.cool
TRAJ_LINEWIDTH = 1


def traj_3D(Data, xyz_idx, xyz_names, show=True):
    assert len(xyz_idx) == 3
    fig = plt.figure(figsize=(9, 4))
    ax = plt.axes(projection='3d')
    ntraj = len(Data['oData'])
    colors = TRAJ_COLORMAP(np.linspace(0, 1, ntraj))
    for traj in range(ntraj):
        ax.plot3D(Data[xyz_idx[0][0]][traj][1][xyz_idx[0][1], :],
                  Data[xyz_idx[1][0]][traj][1][xyz_idx[1][1], :],
                  Data[xyz_idx[2][0]][traj][1][xyz_idx[2][1], :],
                  color=colors[traj], lw=TRAJ_LINEWIDTH)
    ax.set_xlabel(xyz_names[0])
    ax.set_ylabel(xyz_names[1])
    ax.set_zlabel(xyz_names[2])
    # ax.set_aspect('equal', 'box')
    if show:
        plt.show()


def traj_3D_xyz(x, y, z, ax=None, color='tab:blue', show=True):
    if ax is None:
        fig = plt.figure(figsize=(9, 4))
        ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, color=color, lw=TRAJ_LINEWIDTH)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')
    # ax.set_aspect('equal', 'box')
    if show:
        plt.show()
    else:
        return ax
    

def traj_2D_xy(x, y, ax=None, color="tab:blue", ls="-", show=True, label=""):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    ax.plot(x, y, color=color, ls=ls, lw=TRAJ_LINEWIDTH, label=label)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_aspect('equal', 'box')
    if show:
        plt.show()
    else:
        return ax


def traj_xyz(Data, xyz_idx, xyz_names, traj_idx=None, axs=None, ls='-', color=None, show=True, highlight_idx=[]):
    if axs is None:
        fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    if traj_idx is None:
        traj_idx = list(range(len(Data[xyz_idx[0][0]])))
    if color is None:
        colors = TRAJ_COLORMAP(np.linspace(0, 1, len(traj_idx)))
    else:
        colors = [color] * len(traj_idx)
    for coord, ax in enumerate(axs):
        for i, traj in enumerate(traj_idx):
            # plot(t, x/y/z)
            ax.plot(Data[xyz_idx[coord][0]][traj][0],
                    Data[xyz_idx[coord][0]][traj][1][xyz_idx[coord][1], :],
                    color=colors[i], ls=ls, lw=TRAJ_LINEWIDTH)
        for idx in highlight_idx:
            ax.plot(Data[xyz_idx[coord][0]][idx][0],
                    Data[xyz_idx[coord][0]][idx][1][xyz_idx[coord][1], :],
                    color='tab:green', ls=ls, lw=TRAJ_LINEWIDTH*1.2)
        ax.set_ylabel(xyz_names[coord])
        ax.set_xmargin(0)
    axs[-1].set_xlabel(r"$t$")    
    if show:
        plt.show()
    else:
        return axs
    

def traj_xyz_txyz(t, x, y, z, axs=None, xyz_names=None, color=None, ls="-", show=True, label="", rotate_yticks=False):
    if axs is None:
        fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    if xyz_names is None:
        xyz_names = [r'$x$ [mm]', r'$y$ [mm]', r'$z$ [mm]']
    if rotate_yticks:
        y_tick_rotation = 0
    else:
        y_tick_rotation = 90
    for i, coord in enumerate([x, y, z]):
        ax = axs[i]
        ax.plot(t,
                coord,
                color=color, ls=ls, lw=TRAJ_LINEWIDTH, label=label)
        ax.set_ylabel(xyz_names[i], rotation=y_tick_rotation, ha="right")
        ax.set_xmargin(0)
    axs[-1].set_xlabel(r"$t$")
    if show:
        plt.show()
    else:
        return axs

def pca_modes(l2vals, up_to_mode=10, show=True):
    fig = plt.figure(figsize=(9, 4))
    ax = plt.axes()
    ax.plot(np.arange(1, len(l2vals)+1, step=1), np.cumsum(l2vals)/np.sum(l2vals)*100, color="tab:blue", marker="o")
    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Variance [%]")
    ax.set_xlim([1, up_to_mode])
    ax.set_ylim(40, 100)
    if show:
        plt.show()


def mode_direction(modeDir, modeFreq, show=True):
    fig = plt.figure(figsize=(9, 4))
    ax = plt.axes(projection='3d')
    ls = [':', '--', '-.']
    for i in range(modeDir.shape[1]):
        ax.plot3D([0, modeDir[0, i]],
                  [0, modeDir[1, i]],
                  [0, modeDir[2, i]],
                  lw=2, color=COLORS[i], ls = ls[i],
                  label=fr"Mode {i+1} - $\omega$ = {modeFreq[i]:.2f} rad/s")
        ax.plot3D(modeDir[0, i],
                  modeDir[1, i],
                  modeDir[2, i],
                  marker='o', color=COLORS[i])
    ax.legend()
    ax.set_xlabel(r"$x$ [mm]")
    ax.set_ylabel(r"$y$ [mm]")
    ax.set_zlabel(r"$z$ [mm]")
    ax.set_aspect('equal','box')
    if show:
        plt.show()


def inputs(t, u, ax=None, show=True):
    # Align dimensionality of axes
    if t.shape[0] != u.shape[0]:
        u = u.T

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    ax.plot(t, u[:, :], lw=TRAJ_LINEWIDTH)
    ax.set_ylabel(r"$u$")
    ax.set_xlabel(r"$t$")
    ax.set_xmargin(0)
    ax.legend([rf"$u_{i}$" for i in range(1, u.shape[1]+1)])
    if show:
        plt.show()
    else:
        return ax
    
def reduced_coordinates_gradient(t, gradients, labels=None, how="norm", ax=None, show=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    if labels is None:
        labels = ["unknown"] * len(gradients)
    cmaps = [plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens]
    for i, gradient in enumerate(gradients):
        if how == "norm":
            y = np.linalg.norm(gradient, axis=0).reshape(-1, 1)
            colors = [COLORS[i]]
        elif how == "all":
            y = gradient.T
            colors = cmaps[i](np.linspace(0.5, 1, gradient.shape[0]))
        else:
            y = np.array([])
            colors = []
            print("How to plot gradients??")
        for j in range(y.shape[1]):
            ax.plot(t, y[:, j], lw=TRAJ_LINEWIDTH, color=colors[j], label=labels[i])
    ax.set_ylabel(r"$\dot{x}$")
    ax.set_xlabel(r"$t$")
    ax.set_xmargin(0)
    h, l = ax.get_legend_handles_labels()
    by_label = dict(zip(l, h))
    ax.legend(by_label.values(), by_label.keys())
    if show:
        plt.show()
    else:
        return ax
    

def dependence_of_xdot_on_inputs(xdot, u):
    fig, axs = plt.subplots(1, u.shape[0], figsize=(10, 3))
    for i, ax in enumerate(axs):
        x = u[i, :]
        # y = np.linalg.norm(xdot, axis=0).reshape(-1, 1)
        y = xdot.T
        for j in range(y.shape[1]):
            ax.scatter(x, y[:, j], color=COLORS[j])
        ax.set_xlabel(rf"$u_{i+1}$")
        ax.set_ylabel(r"$|\dot{x}|$")
    plt.show()


def adiabatic_model_weights(t, weights, model_names):
    fig, ax = plt.subplots(1, 1, figsize=(9, 3))
    # for i, name in enumerate(model_names):
    #     ax.plot(t, weights[i, :].T, color=COLORS[i], label=name)
    # ax.legend(loc="lower right")
    ax.plot(t, weights.T)
    ax.legend(model_names, ncol=3)
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$g_p(t)$")
    ax.set_xmargin(0)
    ax.set_ylim(0, 1)
    ax.set_xlim(t[0], t[-1])
    fig.suptitle("Model weights")
    plt.show()


def prediction_accuracy_map(q_samples, rmse_samples, vmax=None, colorbar=True, cax=None, ax=None, show=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('rg',["forestgreen", "gold", "firebrick"], N=256) 
    # cmap = mpl.cm.RdYlGn_r
    if not np.all(np.isnan(rmse_samples)):
        sc = ax.scatter(q_samples[:, 0], q_samples[:, 1], s=30, c=rmse_samples, cmap=cmap, vmax=vmax, alpha=0.8)
    nan_idx = np.isnan(rmse_samples)
    ax.scatter(q_samples[nan_idx, 0], q_samples[nan_idx, 1], s=30, c="grey", alpha=0.8)
    ax.set_xlabel(r"$x$ [mm]")
    ax.set_ylabel(r"$y$ [mm]")
    ax.set_aspect('equal','box')
    if colorbar and not np.all(np.isnan(rmse_samples)):
        plt.colorbar(sc, aspect=20, ax=cax, fraction=1, pad=0.02, shrink=0.9, label="RMSE [mm]")
    if show:
        plt.show()
    else:
        return ax
    

def boxplot(samples, vmax=None, xlabel="", ax=None, show=True):
    samples = samples[~np.isnan(samples)]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 2))
    ax.boxplot(samples, vert=False, sym="", widths=0.5, medianprops=dict(color="tab:blue", linewidth=1.5))
    ax.set_xlabel(xlabel)
    if vmax is not None:
        ax.set_xlim(0, vmax)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_yticks([])
    if show:
        plt.show()
    else:
        return ax