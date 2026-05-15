"""
@author: Maziar Raissi
PyTorch conversion
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

np.random.seed(1234)
torch.manual_seed(1234)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def newfig(width, ratio):
    fig = plt.figure(figsize=(8 * width, 8 * width * ratio))
    ax = fig.add_subplot(111)
    return fig, ax


class PhysicsInformedNN:
    def __init__(self, x, y, t, u, v, layers):

        X = np.concatenate([x, y, t], 1)

        self.lb = torch.tensor(X.min(0), dtype=torch.float32, device=device)
        self.ub = torch.tensor(X.max(0), dtype=torch.float32, device=device)

        self.x = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=True)
        self.y = torch.tensor(y, dtype=torch.float32, device=device, requires_grad=True)
        self.t = torch.tensor(t, dtype=torch.float32, device=device, requires_grad=True)

        self.u = torch.tensor(u, dtype=torch.float32, device=device)
        self.v = torch.tensor(v, dtype=torch.float32, device=device)

        self.layers = layers

        self.weights, self.biases = self.initialize_NN(layers)

        self.lambda_1 = nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device))
        self.lambda_2 = nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device))

        self.params = list(self.weights) + list(self.biases) + [self.lambda_1, self.lambda_2]

        self.optimizer_Adam = torch.optim.Adam(self.params)
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.params,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn='strong_wolfe'
        )

    def initialize_NN(self, layers):
        weights = []
        biases = []
        for l in range(len(layers) - 1):
            W = self.xavier_init([layers[l], layers[l + 1]])
            b = nn.Parameter(torch.zeros(1, layers[l + 1], dtype=torch.float32, device=device))
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        W = nn.Parameter(torch.empty(size[0], size[1], dtype=torch.float32, device=device))
        nn.init.xavier_normal_(W)
        return W

    def neural_net(self, X):
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(len(self.weights) - 1):
            H = torch.tanh(H @ self.weights[l] + self.biases[l])
        Y = H @ self.weights[-1] + self.biases[-1]
        return Y

    def net_NS(self, x, y, t):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2

        psi_and_p = self.neural_net(torch.cat([x, y, t], dim=1))
        psi = psi_and_p[:, 0:1]
        p   = psi_and_p[:, 1:2]

        u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        v = -torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]

        u_t  = torch.autograd.grad(u,   t, grad_outputs=torch.ones_like(u),   create_graph=True)[0]
        u_x  = torch.autograd.grad(u,   x, grad_outputs=torch.ones_like(u),   create_graph=True)[0]
        u_y  = torch.autograd.grad(u,   y, grad_outputs=torch.ones_like(u),   create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

        v_t  = torch.autograd.grad(v,   t, grad_outputs=torch.ones_like(v),   create_graph=True)[0]
        v_x  = torch.autograd.grad(v,   x, grad_outputs=torch.ones_like(v),   create_graph=True)[0]
        v_y  = torch.autograd.grad(v,   y, grad_outputs=torch.ones_like(v),   create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        f_u = u_t + lambda_1 * (u * u_x + v * u_y) + p_x - lambda_2 * (u_xx + u_yy)
        f_v = v_t + lambda_1 * (u * v_x + v * v_y) + p_y - lambda_2 * (v_xx + v_yy)

        return u, v, p, f_u, f_v

    def loss_fn(self):
        u_pred, v_pred, _, f_u_pred, f_v_pred = self.net_NS(self.x, self.y, self.t)
        loss = (torch.sum((self.u - u_pred) ** 2) +
                torch.sum((self.v - v_pred) ** 2) +
                torch.sum(f_u_pred ** 2) +
                torch.sum(f_v_pred ** 2))
        return loss

    def callback(self, loss):
        print('Loss: %.3e, l1: %.3f, l2: %.5f' % (
            loss.item(), self.lambda_1.item(), self.lambda_2.item()))

    def train(self, nIter):
        start_time = time.time()
        for it in range(nIter):
            self.optimizer_Adam.zero_grad()
            loss = self.loss_fn()
            loss.backward()
            self.optimizer_Adam.step()

            if it % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, l1: %.3f, l2: %.5f, Time: %.2f' %
                      (it, loss.item(), self.lambda_1.item(), self.lambda_2.item(), elapsed))
                start_time = time.time()

        def closure():
            self.optimizer_LBFGS.zero_grad()
            loss = self.loss_fn()
            loss.backward()
            self.callback(loss)
            return loss

        self.optimizer_LBFGS.step(closure)

    def predict(self, x_star, y_star, t_star):
        x = torch.tensor(x_star, dtype=torch.float32, device=device, requires_grad=True)
        y = torch.tensor(y_star, dtype=torch.float32, device=device, requires_grad=True)
        t = torch.tensor(t_star, dtype=torch.float32, device=device, requires_grad=True)

        u, v, p, _, _ = self.net_NS(x, y, t)

        return u.detach().cpu().numpy(), v.detach().cpu().numpy(), p.detach().cpu().numpy()


def plot_solution(X_star, u_star, index):

    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x, y)

    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')

    plt.figure(index)
    plt.pcolor(X, Y, U_star, cmap='jet')
    plt.colorbar()


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


if __name__ == "__main__":

    N_train = 5000

    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]

    # Load Data
    data = scipy.io.loadmat('../Data/cylinder_nektar_wake.mat')

    U_star = data['U_star']  # N x 2 x T
    P_star = data['p_star']  # N x T
    t_star = data['t']       # T x 1
    X_star = data['X_star']  # N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]

    # Rearrange Data
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T        # N x T

    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star            # N x T

    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1

    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    p = PP.flatten()[:, None]  # NT x 1

    ######################################################################
    ######################## Noiseless Data ##############################
    ######################################################################
    idx = np.random.choice(N * T, N_train, replace=False)
    x_train = x[idx, :]
    y_train = y[idx, :]
    t_train = t[idx, :]
    u_train = u[idx, :]
    v_train = v[idx, :]

    # Training
    model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers)
    model.train(200000)

    # Test Data
    snap = np.array([100])
    x_star = X_star[:, 0:1]
    y_star = X_star[:, 1:2]
    t_star = TT[:, snap]

    u_star = U_star[:, 0, snap]
    v_star = U_star[:, 1, snap]
    p_star = P_star[:, snap]

    # Prediction
    u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)
    lambda_1_value = model.lambda_1.item()
    lambda_2_value = model.lambda_2.item()

    # Error
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)

    error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
    error_lambda_2 = np.abs(lambda_2_value - 0.01) / 0.01 * 100

    print('Error u: %e' % error_u)
    print('Error v: %e' % error_v)
    print('Error p: %e' % error_p)
    print('Error l1: %.5f%%' % error_lambda_1)
    print('Error l2: %.5f%%' % error_lambda_2)

    # Predict for plotting
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x, y)

    UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
    VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
    PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
    P_exact = griddata(X_star, p_star.flatten(), (X, Y), method='cubic')

    ######################################################################
    ########################### Noisy Data ###############################
    ######################################################################
    noise = 0.01
    u_train = u_train + noise * np.std(u_train) * np.random.randn(u_train.shape[0], u_train.shape[1])
    v_train = v_train + noise * np.std(v_train) * np.random.randn(v_train.shape[0], v_train.shape[1])

    # Training
    model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers)
    model.train(200000)

    lambda_1_value_noisy = model.lambda_1.item()
    lambda_2_value_noisy = model.lambda_2.item()

    error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0) * 100
    error_lambda_2_noisy = np.abs(lambda_2_value_noisy - 0.01) / 0.01 * 100

    print('Error l1: %.5f%%' % error_lambda_1_noisy)
    print('Error l2: %.5f%%' % error_lambda_2_noisy)

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################
    data_vort = scipy.io.loadmat('../Data/cylinder_nektar_t0_vorticity.mat')

    x_vort = data_vort['x']
    y_vort = data_vort['y']
    w_vort = data_vort['w']
    modes = data_vort['modes'].item()
    nel   = data_vort['nel'].item()

    xx_vort = np.reshape(x_vort, (modes + 1, modes + 1, nel), order='F')
    yy_vort = np.reshape(y_vort, (modes + 1, modes + 1, nel), order='F')
    ww_vort = np.reshape(w_vort, (modes + 1, modes + 1, nel), order='F')

    box_lb = np.array([1.0, -2.0])
    box_ub = np.array([8.0,  2.0])

    fig, ax = newfig(1.0, 1.2)
    ax.axis('off')

    ####### Row 0: Vorticity ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 2 / 4 + 0.12, left=0.0, right=1.0, wspace=0)
    ax = plt.subplot(gs0[:, :])

    for i in range(0, nel):
        h = ax.pcolormesh(xx_vort[:, :, i], yy_vort[:, :, i], ww_vort[:, :, i],
                          cmap='seismic', shading='gouraud', vmin=-3, vmax=3)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot([box_lb[0], box_lb[0]], [box_lb[1], box_ub[1]], 'k', linewidth=1)
    ax.plot([box_ub[0], box_ub[0]], [box_lb[1], box_ub[1]], 'k', linewidth=1)
    ax.plot([box_lb[0], box_ub[0]], [box_lb[1], box_lb[1]], 'k', linewidth=1)
    ax.plot([box_lb[0], box_ub[0]], [box_ub[1], box_ub[1]], 'k', linewidth=1)

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Vorticity', fontsize=10)

    ####### Row 1: Training data ##################
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=1 - 2 / 4, bottom=0.0, left=0.01, right=0.99, wspace=0)
    ax = plt.subplot(gs1[:, 0], projection='3d')
    ax.axis('off')

    r1 = [x_star.min(), x_star.max()]
    r2 = [data['t'].min(), data['t'].max()]
    r3 = [y_star.min(), y_star.max()]

    for s, e in combinations(np.array(list(product(r1, r2, r3))), 2):
        if (np.sum(np.abs(s - e)) == r1[1] - r1[0] or
                np.sum(np.abs(s - e)) == r2[1] - r2[0] or
                np.sum(np.abs(s - e)) == r3[1] - r3[0]):
            ax.plot3D(*zip(s, e), color="k", linewidth=0.5)

    ax.scatter(x_train, t_train, y_train, s=0.1)
    ax.contourf(X, UU_star, Y, zdir='y', offset=t_star.mean(), cmap='rainbow', alpha=0.8)

    ax.text(x_star.mean(), data['t'].min() - 1, y_star.min() - 1, '$x$')
    ax.text(x_star.max() + 1, data['t'].mean(), y_star.min() - 1, '$t$')
    ax.text(x_star.min() - 1, data['t'].min() - 0.5, y_star.mean(), '$y$')
    ax.text(x_star.min() - 3, data['t'].mean(), y_star.max() + 1, '$u(t,x,y)$')
    ax.set_xlim3d(r1)
    ax.set_ylim3d(r2)
    ax.set_zlim3d(r3)
    axisEqual3D(ax)

    ax = plt.subplot(gs1[:, 1], projection='3d')
    ax.axis('off')

    for s, e in combinations(np.array(list(product(r1, r2, r3))), 2):
        if (np.sum(np.abs(s - e)) == r1[1] - r1[0] or
                np.sum(np.abs(s - e)) == r2[1] - r2[0] or
                np.sum(np.abs(s - e)) == r3[1] - r3[0]):
            ax.plot3D(*zip(s, e), color="k", linewidth=0.5)

    ax.scatter(x_train, t_train, y_train, s=0.1)
    ax.contourf(X, VV_star, Y, zdir='y', offset=t_star.mean(), cmap='rainbow', alpha=0.8)

    ax.text(x_star.mean(), data['t'].min() - 1, y_star.min() - 1, '$x$')
    ax.text(x_star.max() + 1, data['t'].mean(), y_star.min() - 1, '$t$')
    ax.text(x_star.min() - 1, data['t'].min() - 0.5, y_star.mean(), '$y$')
    ax.text(x_star.min() - 3, data['t'].mean(), y_star.max() + 1, '$v(t,x,y)$')
    ax.set_xlim3d(r1)
    ax.set_ylim3d(r2)
    ax.set_zlim3d(r3)
    axisEqual3D(ax)

    # savefig('./figures/NavierStokes_data')

    fig, ax = newfig(1.015, 0.8)
    ax.axis('off')

    ######## Row 2: Pressure #######################
    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(top=1, bottom=1 - 1 / 2, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs2[:, 0])
    h = ax.imshow(PP_star, interpolation='nearest', cmap='rainbow',
                  extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal', 'box')
    ax.set_title('Predicted pressure', fontsize=10)

    ax = plt.subplot(gs2[:, 1])
    h = ax.imshow(P_exact, interpolation='nearest', cmap='rainbow',
                  extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal', 'box')
    ax.set_title('Exact pressure', fontsize=10)

    ######## Row 3: Table #######################
    gs3 = gridspec.GridSpec(1, 2)
    gs3.update(top=1 - 1 / 2, bottom=0.0, left=0.0, right=1.0, wspace=0)
    ax = plt.subplot(gs3[:, :])
    ax.axis('off')

    s = r'$\begin{tabular}{|c|c|}'
    s += r' \hline'
    s += r' Correct PDE & $\begin{array}{c}'
    s += r' u_t + (u u_x + v u_y) = -p_x + 0.01 (u_{xx} + u_{yy})\\'
    s += r' v_t + (u v_x + v v_y) = -p_y + 0.01 (v_{xx} + v_{yy})'
    s += r' \end{array}$ \\ '
    s += r' \hline'
    s += r' Identified PDE (clean data) & $\begin{array}{c}'
    s += r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_1_value, lambda_2_value)
    s += r' \\'
    s += r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_1_value, lambda_2_value)
    s += r' \end{array}$ \\ '
    s += r' \hline'
    s += r' Identified PDE (1\% noise) & $\begin{array}{c}'
    s += r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_1_value_noisy, lambda_2_value_noisy)
    s += r' \\'
    s += r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_1_value_noisy, lambda_2_value_noisy)
    s += r' \end{array}$ \\ '
    s += r' \hline'
    s += r' \end{tabular}$'

    ax.text(0.015, 0.0, s)

    # savefig('./figures/NavierStokes_prediction')
