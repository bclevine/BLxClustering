import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
import emcee
import astropy.cosmology
import astropy.units as u
from chainconsumer import ChainConsumer
import pickle
import pandas as pd
import sacc
from scipy.optimize import minimize

# Turns off an annoying warning message :p
np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# SCALE CUTS
astropy_cosmo = astropy.cosmology.FlatLambdaCDM(
    H0=71, Om0=0.2648, Ob0=0.0448
)  # cosmoDC2 params
z_arr = np.array([0.3 + 0.2 * i for i in range(5)])  # 5 bins
in_angles = astropy_cosmo.arcsec_per_kpc_comoving(z_arr)


def compute_scale_cuts(physical_scale):
    return (in_angles * physical_scale * u.kpc).to(u.arcmin).value


# LOAD DATA VECTOR FROM SACC
def load_sacc(filename):
    s = sacc.Sacc.load_fits(filename)
    if (
        s.get_tracer_combinations("galaxy_density_xi")[0][0] == "lens0"
    ):  # USE BEN FORMAT
        covariances = []
        df = pd.DataFrame()
        nz_df = pd.DataFrame()
        for i in range(5):
            tx = s.get_theta_xi(
                "galaxy_density_xi", f"lens{i}", f"lens{i}", return_cov=True
            )
            df[f"theta_{i}"] = tx[0]
            df[f"w_{i}"] = tx[1]
            df[f"werr_{i}"] = np.sqrt(tx[2].diagonal())
            covariances.append(tx[2])
            nz_df[f"z_{i}"] = s.tracers[f"lens{i}"].z
            nz_df[f"Nz_{i}"] = s.tracers[f"lens{i}"].nz
    elif (
        s.get_tracer_combinations("galaxy_density_xi")[0][0] == "lens_0"
    ):  # USE JUDIT FORMAT:
        covariances = []
        df = pd.DataFrame()
        nz_df = pd.DataFrame()
        for i in range(5):
            tx = s.get_theta_xi(
                "galaxy_density_xi", f"lens_{i}", f"lens_{i}", return_cov=True
            )
            df[f"theta_{i}"] = tx[0]
            df[f"w_{i}"] = tx[1]
            df[f"werr_{i}"] = s.get_tag(
                "error", "galaxy_density_xi", (f"lens_{i}", f"lens_{i}")
            )
            covariances.append(tx[2])
            nz_df[f"z_{i}"] = s.tracers[f"lens_{i}"].z
            nz_df[f"Nz_{i}"] = s.tracers[f"lens_{i}"].nz
    else:
        raise ValueError("Check names of data vectors?")
    return df, nz_df, covariances


# LOAD DATA FROM PICKLE
def load_pkl(filename):
    with open(filename, "rb") as infile:
        data_vec, cov, nz_data_vec = pickle.load(infile)
    df_list = []
    nz_list = []
    for i in range(5):
        df_list.append(
            pd.DataFrame(data_vec[i])
            .rename(index={0: f"theta_{i}", 1: f"w_{i}", 2: f"werr_{i}"})
            .T
        )
        nz_list.append(
            pd.DataFrame(nz_data_vec[i])
            .rename(index={0: f"Nz_{i}", 1: f"z_{i}"})
            .T.reindex(columns=[f"z_{i}", f"Nz_{i}"])
        )
    return pd.concat(df_list, axis=1), pd.concat(nz_list, axis=1), cov


# PREPROCESS DATA VECTOR
def preprocess(data_vector, angles):
    masks = [
        np.where(data_vector[0][f"theta_{i}"] > angles[i])[0]
        for i in range(len(angles))
    ]
    inv_cov = np.array(
        [
            np.linalg.inv(
                data_vector[2][i][
                    masks[i][0] : masks[i][-1] + 1, masks[i][0] : masks[i][-1] + 1
                ]
            )
            for i in range(5)
        ]
    )
    return masks, inv_cov


# HELPER FUNCTION TO CLEAN UP CODE IN MODEL FUNCTION
def model_ccl_helper(data, b, nz_data, cosmo, mag_bias=None, has_rsd=False):
    mb = (
        [None for i in range(5)]
        if mag_bias is None
        else [
            (nz_data[f"z_{i}"], mag_bias[i] * np.ones_like(nz_data[f"z_{i}"]))
            for i in range(5)
        ]
    )
    tracers = [
        ccl.NumberCountsTracer(
            cosmo,
            has_rsd=has_rsd,
            dndz=(nz_data[f"z_{i}"], nz_data[f"Nz_{i}"]),
            bias=(nz_data[f"z_{i}"], b[i] * np.ones_like(nz_data[f"z_{i}"])),
            mag_bias=mb[i],
        )
        for i in range(5)
    ]
    ell = np.unique(np.geomspace(1, 30000, 250).astype(np.int32))
    cls_all = [ccl.angular_cl(cosmo, tracers[i], tracers[i], ell) for i in range(5)]
    return [
        ccl.correlation(cosmo, ell, cls_all[i], data[f"theta_{i}"] / 60)
        for i in range(5)
    ]


# CHI2
def chi2(data, w_model, inv_cov, masks):
    return [
        np.einsum(
            "i, ij, j",
            data[f"w_{i}"][masks[i]] - w_model[i][masks[i]],
            inv_cov[i],
            data[f"w_{i}"][masks[i]] - w_model[i][masks[i]],
        )
        for i in range(5)
    ]


# LIKELIHOOD FUNCTION
def log_like(pars, data, nz_data, inv_cov, masks, model, prior):
    pri = prior(pars)
    if np.isfinite(pri):
        try:
            w_model = model(pars, data, nz_data)
            resid = -0.5 * np.sum(chi2(data, w_model, inv_cov, masks))
            return resid + pri
        except Exception as e:
            print(e)
            return -np.inf
    else:
        return pri


# FUNCTION TO MINIMIZE LOG-LIKE
def min_func(*args):
    return -1 * log_like(*args)


# FUNCTION TO RUN THE EMCEE CHAIN
def run_emcee(
    data_vector,
    filename,
    initial,
    angle_list,
    model,
    prior,
    nsteps=700,
    scatter=0.05,
    posteriors=False,
):
    # SET UP VARIABLES AND DATA
    masks, inv_cov = preprocess(data_vector, angle_list)
    pos = initial + scatter * np.random.randn(32, len(initial))
    nwalkers, ndim = pos.shape
    backend_name = "samples/chain_Apr28.h5"
    backend = emcee.backends.HDFBackend(backend_name)
    backend.reset(nwalkers, ndim)

    # RUN EMCEE model(pars, data, nz_data, inv_cov, masks, model, prior):
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_like,
        args=(data_vector[0], data_vector[1], inv_cov, masks, model, prior),
        backend=backend,
    )
    sampler.run_mcmc(pos, nsteps, progress=True)
    flat_samples = sampler.get_chain(flat=True)
    if posteriors:
        np.save(filename, [flat_samples, sampler.get_log_prob(flat=True)])
    else:
        np.save(filename, flat_samples)


# PLOT WALKER CONVERGENCE
def plot_convergence(chain_list, parnames):
    try:
        for i in range(len(chain_list)):
            ndim = int(np.shape(chain_list[0])[1])
            samples = chain_list[i].reshape(
                (int(np.shape(chain_list[i])[0] / 32), 32, ndim)
            )
            fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
            labels = parnames
            for i in range(ndim):
                ax = axes[i]
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                plot_convergence_axis_formatter(ax, samples, labels, i)
            axes[-1].set_xlabel("step number")
    except Exception:
        for i in range(len(chain_list)):
            ndim = int(np.shape(chain_list[0])[1])
            samples = chain_list[i]
            fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
            labels = parnames
            for i in range(ndim):
                ax = axes[i]
                ax.plot(samples[:, i])
                plot_convergence_axis_formatter(ax, samples, labels, i)
            axes[-1].set_xlabel("step number")


def plot_convergence_axis_formatter(ax, samples, labels, i):
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)


# KILL STUCK WALKERS
def trim_chains(chain, n_steps):
    ndim = int(np.shape(chain)[1])
    samples = chain.reshape((int(np.shape(chain)[0] / 32), 32, ndim))[n_steps:]
    return samples.reshape((int(np.shape(chain)[0] - (n_steps * 32)), ndim))


def remove_stuck_walkers(samples, ndim):
    chain = samples.reshape(np.shape(samples)[0] * np.shape(samples)[1], ndim)
    remove_idxs = []
    for i in range(ndim):
        vals, idxs, counts = np.unique(
            chain[:, i], return_inverse=True, return_counts=True
        )
        remove_idxs.append(np.where(np.isin(idxs, np.where(counts > 1))))
    remove_mask = np.ones(len(chain))
    remove_mask[np.unique(remove_idxs)] = 0
    return remove_mask.astype(bool)


def process_stuck_chain(chain, trim_length):
    ndim = int(np.shape(chain)[1])
    samples = chain.reshape((int(np.shape(chain)[0] / 32), 32, ndim))[trim_length:]
    chain_thinned = samples.reshape(np.shape(samples)[0] * np.shape(samples)[1], ndim)
    chain_thinned = chain_thinned[remove_stuck_walkers(samples, ndim)]
    return chain_thinned


# DATA PROCESSOR CLASS
class Data_Processor:
    def __init__(self, data_vector, chain, model, parnames, fit_vals=None):
        # LOAD AND ANALYZE POSTERIOR
        if (chain is not None) and (fit_vals is None):
            c = ChainConsumer()
            c.add_chain(
                chain,
                parameters=parnames,
            )
            post_dict = c.analysis.get_summary()
            best_fits = [post_dict[key][1] for key in post_dict.keys()]
            self.best_fits = np.array(best_fits)
        else:
            # print("Fitting custom parameters.")
            self.best_fits = np.array(fit_vals)
        # print("Best Fits:", self.best_fits)

        # GENERATE MODEL CORRELATION
        self.ccl_guess = model(self.best_fits, data_vector[0], data_vector[1])


# MODEL PLOTTER
def plot_model(
    data_vector, chain, model, label, angle_list, parnames, fit_vals=None, color="C1"
):
    data = data_vector[0]
    processor = Data_Processor(data_vector, chain, model, parnames, fit_vals)

    # Calculate Chi2
    masks, inv_cov = preprocess(data_vector, angle_list)
    w_model = processor.ccl_guess
    resid = chi2(data, w_model, inv_cov, masks)
    n_dof = [len(data[f"w_{i}"][masks[i]]) for i in range(len(masks))]
    n_pars = len(parnames)

    # PLOT
    fig, axs = plt.subplots(1, 5, sharey=True, figsize=(15, 3))
    for i in range(len(axs)):
        axs[i].set_title(f"${np.round((.2*i)+.2, 2)} < z < {np.round((.2*i)+.4, 2)}$")
        axs[i].axvspan(0, angle_list[i], alpha=0.2)
        axs[i].errorbar(
            data[f"theta_{i}"],
            data[f"w_{i}"],
            data[f"werr_{i}"],
            label=label,
            color=color,
        )
        axs[i].plot(
            data[f"theta_{i}"][len(data[f"theta_{i}"]) - len(processor.ccl_guess[i]) :],
            processor.ccl_guess[i],
            label="Best Fit Model",
            color="red",
        )
        axs[i].set_xscale("log")
        axs[i].set_yscale("log")
        axs[i].set_xlim(2, 250)
        axs[i].set_ylim(4e-5, 0.1)
        axs[i].legend(loc="lower left")
        axs[i].set_xlabel("$\\theta$ [arcmin]")
        axs[i].text(
            0.95,
            0.9,
            f"$\\chi^2/\\nu$ = {resid[i]:.1f}/({n_dof[i]:.0f}",
            horizontalalignment="right",
            verticalalignment="center",
            transform=axs[i].transAxes,
        )
        if i == 0:
            axs[i].set_ylabel("$w(\\theta)$")
    fig.suptitle(
        f"{label}; pars = {np.round(processor.best_fits, 3)}; $\\chi^2/\\nu$ = {np.sum(resid):.1f}/{np.sum(n_dof)}-{n_pars}",
        y=1.1,
    )
    plt.show()


# MODEL+RESIDUAL PLOTTER
def plot_model_residual(
    data_vector,
    chain,
    model,
    label,
    angle_list,
    parnames,
    fit_vals=None,
    color="C1",
    use_cov=True,
    resid_limits=(-3, 2),
    delta_or_ratio="delta",
    yticks=None,
):
    # STYLES
    marker = "."
    capsize = 2
    ls = "-"
    markersize = 7
    lw = 1.5
    model_color = "red"

    # CODE BEGINS
    data, nz, cov = data_vector
    processor = Data_Processor(data_vector, chain, model, parnames, fit_vals)

    # Calculate Chi2
    masks, inv_cov = preprocess(data_vector, angle_list)
    w_model = processor.ccl_guess
    resid = chi2(data, w_model, inv_cov, masks)
    n_dof = [len(data[f"w_{i}"][masks[i]]) for i in range(len(masks))]
    n_pars = len(parnames)
    resid_label = (
        "$\\Delta w/\\sigma_{w}$" if delta_or_ratio == "delta" else "$w_{obs}/w_{CCL}$"
    )

    # PLOT
    fig, axs = plt.subplots(2, 5, figsize=(15, 3), height_ratios=[4, 1])
    for i in range(len(axs[1])):
        err = np.sqrt(cov[i].diagonal()) if use_cov else data[f"werr_{i}"]
        axs[0][i].set_title(
            f"${np.round((.2*i)+.2, 2)} < z < {np.round((.2*i)+.4, 2)}$"
        )
        axs[0][i].axvspan(
            0, angle_list[i], alpha=0.1, edgecolor="black", facecolor="gray"
        )
        axs[0][i].errorbar(
            data[f"theta_{i}"],
            data[f"w_{i}"],
            err,
            label=label,
            color=color,
            marker=marker,
            capsize=capsize,
            ls=ls,
            markersize=markersize,
            lw=lw,
        )
        axs[0][i].plot(
            data[f"theta_{i}"][len(data[f"theta_{i}"]) - len(processor.ccl_guess[i]) :],
            processor.ccl_guess[i],
            label="Best Fit Model",
            color=model_color,
            lw=lw,
        )
        residual_plot_helper(axs, 0, i, "log")
        axs[0][i].set_ylim(4e-5, 0.1)
        axs[0][i].text(
            0.95,
            0.9,
            f"$\\chi^2/\\nu$ = {resid[i]:.1f}/{n_dof[i]:.0f}",
            horizontalalignment="right",
            verticalalignment="center",
            transform=axs[0][i].transAxes,
        )
        axs[0][i].axes.xaxis.set_ticklabels([])
        # RESIDUALS
        axs[1][i].axvspan(
            0, angle_list[i], alpha=0.1, edgecolor="black", facecolor="gray"
        )
        if delta_or_ratio == "ratio":
            axs[1][i].axhline(1, 0, 500, color="black", ls=":")
            axs[1][i].errorbar(
                data[f"theta_{i}"],
                (data[f"w_{i}"] / processor.ccl_guess[i]),
                (err / np.abs(processor.ccl_guess[i])),
                color=color,
                marker=marker,
                capsize=capsize,
                ls=ls,
                markersize=markersize,
                lw=lw,
            )
        else:
            axs[1][i].axhline(0, 0, 500, color="black", ls=":")
            axs[1][i].plot(
                data[f"theta_{i}"],
                (data[f"w_{i}"] - processor.ccl_guess[i]) / err,
                color=color,
                marker=marker,
                ls=ls,
                markersize=markersize,
                lw=lw,
            )
        residual_plot_helper(axs, 1, i, "linear")
        axs[1][i].set_ylim(resid_limits[0], resid_limits[1])
        axs[1][i].set_xlabel("$\\theta$ [arcmin]")
        if yticks is not None:
            axs[1][i].set_yticks(yticks)
        if i == 0:
            axs[0][i].legend(loc="lower left")
            axs[0][i].set_ylabel("$w(\\theta)$")
            axs[1][i].set_ylabel(resid_label)
        else:
            axs[0][i].axes.yaxis.set_ticklabels([])
            axs[1][i].axes.yaxis.set_ticklabels([])
    fig.suptitle(
        f"{label}; pars = {np.round(processor.best_fits, 3)}; $\\chi^2/\\nu$ = {np.sum(resid):.1f}/({np.sum(n_dof)}-{n_pars})",
        y=1.1,
    )
    fig.subplots_adjust(hspace=0, wspace=0.1)
    plt.show()


def residual_plot_helper(axs, idx, i, yscale):
    axs[idx][i].set_xscale("log")
    axs[idx][i].set_yscale(yscale)
    axs[idx][i].set_xlim(1, 250)


# GRADIENT MINIMIZATION FUNCTION
def gradient_minimize(pars, data_vector, angles, model, prior):
    masks, inv_cov = preprocess(data_vector, angles)
    return minimize(
        min_func,
        pars,
        args=(data_vector[0], data_vector[1], inv_cov, masks, model, prior),
        method="Nelder-Mead",
    )
