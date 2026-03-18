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



def plot_results(d_vals, D_vals, r, save_path, results=None, load_path=None):
    assert not (results is None and load_path is None)

    if results is None and load_path is not None:
        results = np.load(os.path.join(load_path, 'results_d.npy'))
        results = results[:len(D_vals), :len(d_vals)]

    _, ax = plt.subplots()
    im = ax.imshow(results)
    ax.invert_yaxis()

    d_tick_freq = 2
    ax.set_xticks(ticks=range(0, len(d_vals), d_tick_freq), labels=d_vals[::d_tick_freq])
    ax.set_xlabel("Ambient Dimension")

    D_tick_freq = 2
    ax.set_yticks(ticks=range(0, len(D_vals), D_tick_freq), labels=D_vals[::D_tick_freq])
    ax.set_ylabel("Network Width")


    figure_fname = os.path.join(save_path, 'separability_d.png')
    ax.set_title("Linear Separability (Intrinsic Dim = " + str(r) + ")", y=1.02)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(figure_fname, dpi=500)
    plt.close()


def D_vs_d(d_vals, D_vals, r, num_trials, root_save_dir):
    results = np.zeros((len(D_vals), len(d_vals)))
    
    for i, D in enumerate(D_vals):
        for j, d in enumerate(d_vals):
            num_successes = 0
            for k in range(num_trials):
                np.random.seed(k)

                # Generate random weight matrix and subspaces
                X, Y = generate_mats(D, d, r)

                # Determine linear separability
                num_successes += test(X, Y)
            
            results[i, j] = num_successes / num_trials
            print("(D =", D, "d =", d, "):", num_successes, "/", num_trials, "trials successful")


    # Save results
    save_path = os.path.join(root_save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_fname = os.path.join(save_path, 'results_d.npy')
    with open(save_fname, "wb") as f:
        np.save(f, results)
    
    # Plot results
    plot_results(d_vals, D_vals, r, save_path, results=results)
    
    print("Finished D vs. d")

    
                
if __name__ == "__main__":
    d_vals = np.arange(20, 41, 1) # Ambient dimensions
    D_vals = np.arange(40, 100, 3) # Network width
    r = 10 # Subspace dimension
    num_trials = 25
    root_save_dir = './figures/'
    load_path = './figures/'

    
    D_vs_d(d_vals, D_vals, r, num_trials, root_save_dir)

    plot_results(d_vals, D_vals, r, root_save_dir, load_path=load_path)
    

