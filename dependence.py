from typing import List, Tuple, cast

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm

from typing import Tuple

import numpy as np
from scipy.stats import ortho_group
import os


def generate_UV(
    r: int,
    d: int,
    rng: np.random.Generator,
    orth: bool = True
) -> Tuple[np.ndarray, np.ndarray]:

    if orth:
        assert d >= 2*r
        UV = ortho_group(dim=d, seed=rng).rvs()[:, : 2 * r]
        U, V = np.split(UV, 2, axis=-1)
    else:
        assert d > r
        U = ortho_group(dim=d, seed=rng).rvs()[:, :r]
        V = ortho_group(dim=d, seed=rng).rvs()[:, :r]

    return U, V


def generate_W(
    D: int,
    d: int,
    rng: np.random.Generator,
    orth: bool = False,
) -> np.ndarray:

    if orth:
        assert D >= d, "D must be greater than d for orthogonality"
        W = ortho_group(dim=D, seed=rng).rvs()[:, :d]
    else:
        W = rng.normal(size=(D, d))
    return W



def is_linearly_separable(
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray
) -> Tuple[bool, np.ndarray]:

    r = U.shape[1]
    D = W.shape[0]

    a = cp.Variable(D)
    Phi = W.T @ cp.diag(a) @ W
    mat = cp.bmat(
        [
            [U.T @ Phi @ U, np.zeros(shape=(r, r))],
            [np.zeros(shape=(r, r)), -V.T @ Phi @ V],
        ],
    )
    objective = cp.Maximize(cp.lambda_min(mat))
    constraints = []
    constraints = cast(List[cp.Constraint], constraints)
    problem = cp.Problem(objective, constraints)
    result = problem.solve()
    return result == np.inf, a.value


def test_separability(
    d: int,
    r: int,
    D: int,
    rng: np.random.Generator,
    n_trials: int = 3,
    W_orth: bool = False,
    UV_orth: bool = True
) -> Tuple[float, float, List]:

    results = []
    classifiers = []
    # for _ in (pbar := tqdm(range(n_trials))):
    for i in range(n_trials):
        # pbar.set_description(f"d = {d}, D = {D}, r = {r}")
        U, V = generate_UV(r, d, rng, UV_orth)
        W = generate_W(D, d, rng, W_orth)
        result = is_linearly_separable(U, V, W)
        results.append(result[0])
        if result[0]:
            classifiers.append(result[1])

    return np.mean(results).astype(float), np.std(results).astype(float), classifiers


def D_vs_d(
    D_vals: np.ndarray,
    d_vals: np.ndarray,
    r: int,
    n_trials: int = 10,
    seed: int = 0,
    W_orth: bool = False,
    UV_orth: bool = True,
    save_dir: str = "./figures/dependence"
):

    rng = np.random.default_rng(seed=seed)
    result = np.zeros((len(D_vals), len(d_vals)))

    for i, D in enumerate(tqdm(D_vals)):
        for j, d in enumerate(tqdm(d_vals)):
            result[i, j] = test_separability(d, r, D, rng, n_trials, W_orth=W_orth, UV_orth=UV_orth)[0]

    #_, ax = plt.subplots()
    #im = ax.imshow(result)
    #ax.invert_yaxis()

    #tick_freq = 2
    #ax.set_xticks(ticks=range(0, len(d_vals), tick_freq), labels=d_vals[::tick_freq])
    #ax.set_xlabel("d")

    #ax.set_yticks(ticks=range(0, len(D_vals), tick_freq), labels=D_vals[::tick_freq])
    #ax.set_ylabel("D")

    #ax.set_title("Linear Separability, r = ", str(r))

    save_name = "D_vs_d_orth" if UV_orth else "D_vs_d_random"
    save_name = save_name + "_rank_" + str(r) + "_widths_" + str(D_vals[0]) + "-" + str(D_vals[-1]) + "_dims_" + str(d_vals[0]) + "-" + str(d_vals[-1])
    save_path = os.path.join(save_dir, save_name)

    with open(save_path + ".npy", "wb") as f:
        np.save(f, result)

    #plt.colorbar(im, ax=ax)
    #plt.tight_layout()
    #plt.savefig(save_path, dpi=500)
    #plt.show()


def D_vs_r(
    D_vals: np.ndarray,
    r_vals: np.ndarray,
    d: int,
    load_path: str,
    n_trials: int = 10,
    seed: int = 0,
    W_orth: bool = False,
    UV_orth: bool = True,
    save_dir: str = "./figures/dependence"
) -> str:

    rng = np.random.default_rng(seed=seed)
    result = np.zeros((len(D_vals), len(r_vals)))

    for i, D in enumerate(tqdm(D_vals)):
        for j, r in enumerate(tqdm(r_vals)):
            result[i, j] = test_separability(d, r, D, rng, n_trials, W_orth=W_orth, UV_orth=UV_orth)[0]


    save_name = "D_vs_r_orth" if UV_orth else "D_vs_r_random"
    save_name = save_name + "_dim_" + str(d) + "_widths_" + str(D_vals[0]) + "-" + str(D_vals[-1]) + "_ranks_" + str(r_vals[0]) + "-" + str(r_vals[-1])
    save_path = os.path.join(save_dir, save_name)
    with open(save_path + ".npy", "wb") as f:
        np.save(f, result)

    _, ax = plt.subplots()
    im = ax.imshow(result)
    ax.invert_yaxis()

    tick_freq = 2
    ax.set_xticks(ticks=range(0, len(r_vals), tick_freq), labels=r_vals[::tick_freq])
    ax.set_xlabel("r")

    ax.set_yticks(ticks=range(0, len(D_vals), tick_freq), labels=D_vals[::tick_freq])
    ax.set_ylabel("D")

    ax.set_title("Linear Separability, d = " + str(d))

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(load_path + ".png", dpi=500)
    plt.show()

    return load_path #save_path


def D_vs_r_plot(
    D_vals: np.ndarray,
    r_vals: np.ndarray,
    d: int,
    load_path: str,
    UV_orth: bool = True,
    save_dir: str = "./figures/dependence"
):

    results = np.load(load_path + ".npy")

    boundary = []

    for i in range(len(D_vals)):
        a = np.flatnonzero(results[i, :] == 1.0)
        if len(a) == 0:
            continue
        j = np.flatnonzero(results[i, :] == 1.0)[-1]
        boundary.append((D_vals[i], r_vals[j]))

    D_boundary, r_boundary = zip(*boundary[5:])
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

    save_fname = "log_D_vs_log_r_orth" if UV_orth else "log_D_vs_log_r_random"
    save_fname = save_fname + "_dim_" + str(d) + "_widths_" + str(D_vals[0]) + "-" + str(D_vals[-1]) + "_ranks_" + str(r_vals[0]) + "-" + str(r_vals[-1])
    save_path = os.path.join(save_dir, save_fname)
    plt.plot(log_r_vals, reg.predict(log_r_vals.reshape(-1, 1)), "--")
    plt.xlabel("log(r)")
    plt.ylabel("log(D)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=500)
    plt.show()


# def classifier_weights(
#     D_vals: np.ndarray,
#     r_vals: np.ndarray,
#     d: int,
#     n_trials: int = 10,
#     seed: int = 0,
#     W_orth: bool = False,
#     UV_orth: bool = True,
#     save_dir: str = "./figures"
# ):

#     rng = np.random.default_rng(seed=seed)
#     result = np.zeros((len(D_vals), len(r_vals)))

#     for i, D in enumerate(tqdm(D_vals)):
#         for j, r in enumerate(tqdm(r_vals)):
#             result[i, j] = test_separability(d, r, D, rng, n_trials, W_orth=W_orth, UV_orth=UV_orth)[0]

#     save_name = "cls_weights_orth" if UV_orth else "cls_weights_random"
#     save_name = save_name + "_dim_" + str(d) + "_widths_" + str(D_vals[0]) + "-" + str(D_vals[-1]) + "_ranks_" + str(r_vals[0]) + "-" + str(r_vals[-1])
#     save_path = os.path.join(save_dir, save_name)
#     with open(save_path + ".npy", "wb") as f:
#         np.save(f, result)

#     _, ax = plt.subplots()
#     im = ax.imshow(result)
#     ax.invert_yaxis()

#     tick_freq = 2
#     ax.set_xticks(ticks=range(0, len(r_vals), tick_freq), labels=r_vals[::tick_freq])
#     ax.set_xlabel("r")

#     ax.set_yticks(ticks=range(0, len(D_vals), tick_freq), labels=D_vals[::tick_freq])
#     ax.set_ylabel("D")

#     ax.set_title("Linear Separability, d = ", str(d))

#     plt.colorbar(im, ax=ax)
#     plt.tight_layout()
#     plt.savefig(save_name + ".png", dpi=500)
#     plt.show()

#     return save_path


if __name__ == "__main__":
    # Network width vs data dimension
    D_vals = np.linspace(1, 256, 32, dtype=int)
    d_vals = np.linspace(128, 192, 16, dtype=int)
    r = 96
    D_vs_d(D_vals, d_vals, r, UV_orth=False)

    # Network width vs subspace rank
    #D_vals = np.linspace(1, 256, 32, dtype=int)
    #r_vals = np.linspace(1, 96, 32, dtype=int)
    #d = 128
    #path_ = D_vs_r(D_vals, r_vals, d, UV_orth=False)
    #D_vs_r_plot(D_vals, r_vals, d, path_, UV_orth=False)
