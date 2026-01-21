from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

EV_TO_THz = 241.79893  # 1 eV = 241.79893 THz (E = h nu)
THz_TO_CM1 = 33.35641  # 1 THz ≈ 33.35641 cm^-1


def plot_dispersion_with_dos(
    band_structure,
    dos_energies_ev: np.ndarray,
    dos_weights_per_ev: np.ndarray,
    material: str,
    outdir: str,
    dpi: int = 300,
):
    """
    Plot band structure with DOS side panel.

    Inputs match scripts/phonons.py outputs:
      - band_structure from compute_band_structure()
      - dos_energies_ev, dos_weights_per_ev from compute_phonon_dos()
    """
    dos_energies_ev = np.asarray(dos_energies_ev)
    dos_weights_per_ev = np.asarray(dos_weights_per_ev)

    emax_ev = max(
        float(np.max(dos_energies_ev)),
        float(np.max(np.asarray(band_structure.energies))),
    ) * 1.1

    fig = plt.figure(figsize=(9, 5))

    ax = fig.add_axes([0.12, 0.1, 0.60, 0.8])
    band_structure.plot(ax=ax, emin=0.0, emax=emax_ev)

    ticks = ax.get_yticks()
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t * EV_TO_THz:.0f}" for t in ticks])
    ax.set_ylabel("Frequency (THz)", fontsize=14)

    dosax = fig.add_axes([0.75, 0.1, 0.20, 0.8])
    dosax.fill_between(
        dos_weights_per_ev,
        dos_energies_ev * EV_TO_THz,
        y2=0.0,
        color="grey",
        alpha=0.5,
    )

    dosax.set_ylim(0.0, emax_ev * EV_TO_THz)
    dosax.set_yticks([])
    dosax.set_xlabel("DOS", fontsize=14)
    fig.suptitle(f"{material} Phonon Dispersion and DOS", fontsize=16, y=0.98)
    fig.savefig(outdir, dpi=dpi)



def plot_phonon_dos(
    dos_energies_ev: np.ndarray,
    dos_weights_per_ev: np.ndarray,
    outdir: str,
    title: str | None = None,
    emax_thz: float | None = None,
    dpi: int = 450,
):
    """
    Single-panel DOS plot: DOS on x-axis, frequency on y-axis (THz),
    with a secondary y-axis in cm^-1.

    Inputs match scripts/phonons.py outputs.
    """
    energies_ev = np.asarray(dos_energies_ev)
    weights_per_ev = np.asarray(dos_weights_per_ev)

    freqs_thz = energies_ev * EV_TO_THz
    weights_per_thz = weights_per_ev / EV_TO_THz

    mask = freqs_thz >= -1e-6
    freqs_thz = np.clip(freqs_thz[mask], 0.0, None)
    weights_per_thz = weights_per_thz[mask]

    if emax_thz is None:
        emax_thz = float(np.max(freqs_thz)) * 1.02

    fig, ax = plt.subplots(figsize=(4.6, 6.2), constrained_layout=True)
    ax.plot(weights_per_thz, freqs_thz, linewidth=1.2)
    ax.fill_betweenx(freqs_thz, 0.0, weights_per_thz, alpha=0.25)

    ax.set_xlabel("Phonon DOS (states / THz)")
    ax.set_ylabel("Frequency (THz)")
    ax.set_ylim(0.0, emax_thz)
    ax.set_xlim(left=0.0)

    ax.grid(True, which="both", alpha=0.25)
    ax.tick_params(direction="in", top=True, right=True)

    if title:
        ax.set_title(title)

    ax2 = ax.twinx()
    y0, y1 = ax.get_ylim()
    ax2.set_ylim(y0 * THz_TO_CM1, y1 * THz_TO_CM1)
    ax2.set_ylabel(r"Frequency (cm$^{-1}$)")
    ax2.tick_params(direction="in", top=True, right=True)

    fig.savefig(outdir, dpi=dpi)
