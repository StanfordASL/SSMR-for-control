import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.sans-serif': 'Times New Roman'})
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

COLORS = ['tab:blue', 'tab:orange', 'tab:green']

TRAJ_COLORMAP = plt.cm.cool
TRAJ_LINEWIDTH = 1


def traj_3D(Data, xyz_idx, xyz_names):
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

    fig.show()


def traj_3D_xyz(x, y, z, ax=None, color='tab:blue', show=True):

    if ax is None:
        fig = plt.figure(figsize=(9, 4))
        ax = plt.axes(projection='3d')

    ax.plot3D(x, y, z, color=color, lw=TRAJ_LINEWIDTH)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')

    if show:
        plt.show()
    else:
        return ax


def traj_xyz(Data, xyz_idx, xyz_names, traj_idx=None, axs=None, ls='-', color=None, show=True):

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
        ax.set_ylabel(xyz_names[coord])
        ax.set_xlabel(r"$t$")
        ax.set_xmargin(0)
    
    if show:
        plt.show()
    else:
        return axs
    

def traj_xyz_txyz(t, x, y, z, axs=None, color='tab:blue', show=True):
    if axs is None:
        fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    xyz_names = [r'$x$', r'$y$', r'$z$']
    for i, coord in enumerate([x, y, z]):
        ax = axs[i]
        ax.plot(t,
                coord,
                color=color, lw=TRAJ_LINEWIDTH)
        ax.set_ylabel(xyz_names[i])
        ax.set_xlabel(r"$t$")
        ax.set_xmargin(0)
    if show:
        plt.show()
    else:
        return axs

def pca_modes(l2vals, up_to_mode=10):

    fig = plt.figure(figsize=(9, 4))
    ax = plt.axes()

    ax.plot(np.arange(1, len(l2vals)+1, step=1), np.cumsum(l2vals)/np.sum(l2vals)*100, color="tab:blue", marker="o")

    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Variance [%]")
    ax.set_xlim([1, up_to_mode])
    ax.set_ylim(40, 100)

    fig.show()


def mode_direction(modeDir, modeFreq):

    fig = plt.figure(figsize=(9, 4))
    ax = plt.axes(projection='3d')
    ls = [':', '--', '-.']
    for i in range(3):
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
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")

    plt.show()