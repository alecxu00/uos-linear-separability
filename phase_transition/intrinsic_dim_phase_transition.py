from typing import List, cast

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ortho_group

from sklearn.linear_model import LinearRegression
import os


def generate_mats(D, d, r):
    W = np.random.randn(D, d)
    U1 = ortho_group.rvs(d)[:, :r]
    U2 = ortho_group.rvs(d)[:, :r]

    # U1 = U[:, :r]
    # U2 = U[:, r:]

    X = W @ U1
    Y = W @ U2
    
    return X, Y


def test(X, Y):
    D, r = X.shape
    
    U = np.block(
        [
            [X, np.zeros(shape=(D, r))], 
            [np.zeros(shape=(D, r)), Y]
        ]
    )

    a = cp.Variable(D)
    Phi = cp.bmat(
        [
            [cp.diag(a), np.zeros(shape=(D, D))],
            [np.zeros(shape=(D, D)), -cp.diag(a)],
        ]
    )

    mat = U.T @ Phi @ U
    objective = cp.Maximize(cp.lambda_min(mat))
    constraints = []
    constraints = cast(List[cp.Constraint], constraints)
    problem = cp.Problem(objective, constraints)
    result = problem.solve()
    return result == np.inf



def plot_results(r_vals, D_vals, d, save_path, results=None, load_path=None):
    assert not (results is None and load_path is None)

    if results is None and load_path is not None:
        results = np.load(os.path.join(load_path, 'results_r.npy'))
        results = results[:len(D_vals), :len(r_vals)]
    
    _, ax = plt.subplots()
    im = ax.imshow(results)
    ax.invert_yaxis()

    r_tick_freq = 2
    ax.set_xticks(ticks=range(0, len(r_vals), r_tick_freq), labels=r_vals[::r_tick_freq])
    ax.set_xlabel("Intrinsic Dimension")

    D_tick_freq = 2
    ax.set_yticks(ticks=range(0, len(D_vals), D_tick_freq), labels=D_vals[::D_tick_freq])
    ax.set_ylabel("Network Width")


    figure_fname = os.path.join(save_path, 'separability_r.png')
    ax.set_title("Linear Separability (Ambient Dim = " + str(d) + ")", y=1.02)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(figure_fname, dpi=500)
    plt.close()


def plot_dependence(r_vals, D_vals, save_path, results=None, load_path=None):
    assert not (results is None and load_path is None)

    if results is None and load_path is not None:
        results = np.load(os.path.join(load_path, 'results_r.npy'))
    
    boundary = []

    for i in range(len(D_vals)):
        a = np.flatnonzero(results[i, :] == 1.0)
        if len(a) == 0:
            continue
        j = np.flatnonzero(results[i, :] == 1.0)[-1]
        boundary.append((D_vals[i], r_vals[j]))

    D_boundary, r_boundary = zip(*boundary)
    D_boundary = np.array(D_boundary)
    r_boundary = np.array(r_boundary)

    log_D_boundary = np.log(D_boundary)
    log_r_boundary = np.log(r_boundary)

    plt.figure()
    plt.plot(log_r_boundary, log_D_boundary, "o")

    # Power scaling fit
    reg = LinearRegression().fit(log_r_boundary.reshape(-1, 1), log_D_boundary)
    print(reg.coef_, reg.intercept_)
    log_r_vals = np.linspace(min(log_r_boundary), max(log_r_boundary), 100)

    save_path = os.path.join(save_path, 'log_D_vs_log_r.png')
    plt.plot(log_r_vals, reg.predict(log_r_vals.reshape(-1, 1)), "--")
    plt.title(r'$\mathrm{ln}(D)$ vs. $\mathrm{ln}(r)$')
    plt.xlabel(r'$\mathrm{ln}(r)$')
    plt.ylabel(r'$\mathrm{ln}(D)$')
    plt.tight_layout()
    plt.savefig(save_path, dpi=500)
    plt.close()


def D_vs_r(r_vals, D_vals, d, num_trials, root_save_dir):
    results = np.zeros((len(D_vals), len(r_vals)))
    
    for i, D in enumerate(D_vals):
        for j, r in enumerate(r_vals):
            num_successes = 0
            for k in range(num_trials):
                np.random.seed(k)

                # Generate random weight matrix and subspaces
                X, Y = generate_mats(D, d, r)

                # Determine linear separability
                num_successes += test(X, Y)
            
            results[i, j] = num_successes / num_trials
            print("(D =", D, "r =", r, "):", num_successes, "/", num_trials, "trials successful")


    # Save results
    save_path = os.path.join(root_save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_fname = os.path.join(save_path, 'results_r.npy')
    with open(save_fname, "wb") as f:
        np.save(f, results)
    
    # Plot results
    plot_results(r_vals, D_vals, d, save_path, results=results)
    plot_dependence(r_vals, D_vals, save_path, results=results)

    print("Finished D vs. r")


def compute_pearson(D_vals, r_vals, results=None, load_path=None):
    assert not (results is None and load_path is None)  
    
    if load_path is not None:
        results = np.load(os.path.join(load_path, 'results_r.npy'))

    boundary = []

    for i in range(len(D_vals)):
        a = np.flatnonzero(results[i, :] == 1.0)
        if len(a) == 0:
            continue
        j = np.flatnonzero(results[i, :] == 1.0)[-1]
        boundary.append((D_vals[i], r_vals[j]))

    D_boundary, r_boundary = zip(*boundary)
    D_boundary = np.array(D_boundary)
    r_boundary = np.array(r_boundary)

    log_D_boundary = np.log(D_boundary)
    log_r_boundary = np.log(r_boundary)

    log_D_mean = np.mean(log_D_boundary)
    log_r_mean = np.mean(log_r_boundary)

    log_D_centered = log_D_boundary - log_D_mean
    log_r_centered = log_r_boundary - log_r_mean

    numr = np.sum(log_D_centered * log_r_centered)
    denr1 = np.sqrt( np.sum( log_r_centered**2 ) )
    denr2 = np.sqrt( np.sum( log_D_centered**2 ) )

    return numr / (denr1 * denr2)

    

if __name__ == "__main__":
    r_vals = np.arange(1, 21, 1) # Rank of subspaces
    D_vals = np.arange(40, 100, 3) # Random network width
    d = 40 # Ambient dimension
    num_trials = 25
    root_save_dir = './figures'
    load_path = './figures'

    
    D_vs_r(r_vals, D_vals, d, num_trials, root_save_dir)

    plot_results(r_vals, D_vals, d, root_save_dir, load_path=load_path)
    plot_dependence(r_vals, D_vals, root_save_dir, load_path=load_path)
    rho = compute_pearson(D_vals, r_vals, load_path=load_path)
    print(rho)
    

