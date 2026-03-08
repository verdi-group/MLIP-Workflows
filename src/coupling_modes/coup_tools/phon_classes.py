from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy import ndarray
import yaml

ndarray_realFloats = ndarray  # expected float dtype
ndarray_complex = ndarray  # expected complex dtype

eps = 1e-12


@dataclass(frozen=True)
class Structure:
    """Container for crystal structure data.

    Args:
        lattice (ndarray_realFloats): (3,3) lattice vectors in Å.
        frac (ndarray_realFloats): (N,3) fractional coordinates.
        elements (List[str]): Element symbols in order.
        counts (List[int]): Atom counts per element.

    Returns:
        None
    """

    lattice: ndarray_realFloats  # (3,3) rows are lattice vectors in Å
    frac: ndarray_realFloats  # (N,3) fractional coordinates
    elements: List[str]
    counts: List[int]


@dataclass(frozen=True)
class BandData:
    """Phonon band structure data container.

    Args:
        natom (int): Number of atoms in the unit cell.
        masses (ndarray_realFloats): (N,) atomic masses.
        q_positions (ndarray_realFloats): (nq,3) q-point positions.
        frequencies (ndarray_realFloats): (nq,3N) phonon frequencies.
        eigenvectors (ndarray_complex): (nq,3N,N,3) phonon eigenvectors.

    Returns:
        None
    """

    natom: int
    masses: ndarray_realFloats  # (N,)
    q_positions: ndarray_realFloats  # (nq,3)
    frequencies: ndarray_realFloats  # (nq,3N)
    eigenvectors: ndarray_complex  # (nq,3N,N,3) complex128

    @property
    def nmodes(self) -> int:
        """Return number of phonon modes per q-point.

        Args:
            None

        Returns:
            int: Number of modes (second axis of frequencies).
        """

        return int(self.frequencies.shape[1])
    # frequencies are shaped like (number of q points, number of modes) (should be 3N haha)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> BandData:
        """Load phonon band data from a phonopy-style band.yaml.

        Args:
            path (Union[str, Path]): Path to band.yaml file.

        Returns:
            BandData: Parsed phonon band data.
        """

        from .phon_parsers import _extract_masses, _parse_eigenvector

        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        # loads yaml file like a dictionary
        # would probably be better to use the phonopy.load however i stuck with this.

        phonon = data.get("phonon", None)
        if not phonon: # error catching to be precise about the phonopy structure requirement.
            hint = ""
            if isinstance(data, dict):
                # Common concern is when one passes a phonopy.yaml / phonons.yaml-style file (with no `phonon:` list)
                # instead of a phonopy band.yaml file.
                if "phonopy" in data and ("primitive_cell" in data or "unit_cell" in data):
                    hint = (
                        " (this looks like a phonopy.yaml/phonons.yaml file, not a phonopy band.yaml; "
                        "for this project the ML band file is typically under "
                        "`results/<model>/<structure>/raw/Plumipy_Files/band.yaml`)"
                    )
            raise ValueError(f"band.yaml missing/invalid 'phonon' list: {p}{hint}")

        if "natom" not in data:
            raise ValueError(f"band.yaml missing 'natom': {p}")
        natom = int(data["natom"])

        masses_list = _extract_masses(data, natom) # masses are stored per point (key) which has three values
        # one being its mass other being its symbol and the other being its coordinate.
        masses = np.asarray(masses_list, dtype=float) # preparing it to be diagonalised for mass 
        #weighted displacement vector

        # Accumulate per-q data; shapes after stacking:
        # q_positions -> (nq,3), freqs_all -> (nq,3N), eigs_all -> (nq,3N,N,3)
        q_positions: List[List[float]] = []
        freqs_all: List[List[float]] = []
        eigs_all: List[List[List[List[complex]]]] = [] # highly embodied since eig vectors themselves are list[list[complex]] 
        # and are per frequency which is per q point 

        nmodes: Optional[int] = None
        for ph in phonon:
            q = ph.get("q-position", None)
            band = ph.get("band", None)
            if q is None or band is None:
                raise ValueError(f"Invalid phonon entry (missing q-position/band) in {p}")
            if not isinstance(band, list) or not band:
                raise ValueError(f"Invalid band list at q-position in {p}")

            if nmodes is None:
                nmodes = len(band)
            elif len(band) != nmodes:
                raise ValueError(f"Inconsistent number of modes across q-points in {p}")

            # q-position is length-3 vector
            q_positions.append([float(q[0]), float(q[1]), float(q[2])])
            # convert these to floats as later we will be using l1 euclidean norm to qpoint match the mlip phonopy.yml
            # and the dft.

            freqs_q: List[float] = []  # length = nmodes (3N)
            eigs_q: List[List[List[complex]]] = []  # length = nmodes, each (N,3)
            for b in band:
                f = b.get("frequency", None)
                if f is None:
                    raise ValueError(f"Missing frequency in {p}")
                freqs_q.append(float(f))

                ev = b.get("eigenvector", None)
                if ev is None:
                    raise ValueError(f"Missing eigenvector in {p}")
                eigs_q.append(_parse_eigenvector(ev, natom))

            freqs_all.append(freqs_q)
            eigs_all.append(eigs_q)

        if nmodes is None:
            raise ValueError(f"No modes parsed from {p}")
        if nmodes != 3 * natom:
            raise ValueError(f"Expected nmodes==3N ({3*natom}) but got {nmodes} in {p}")

        # Final stacked arrays with fixed shapes
        q_positions_a = np.asarray(q_positions, dtype=float)  # (nq,3)
        freqs_a = np.asarray(freqs_all, dtype=float)  # (nq,3N)

        # Build eigenvector array explicitly to ensure shape (nq,3N,N,3)
        eigs = np.empty((len(phonon), nmodes, natom, 3), dtype=np.complex128)
        for iq in range(len(phonon)):
            for im in range(nmodes):
                eigs[iq, im, :, :] = np.asarray(eigs_all[iq][im], dtype=np.complex128)

        return cls( # cls is default class method is called on, we pass this info
            # through to return an instance of the BandData class.
            natom=natom,
            masses=masses,
            q_positions=q_positions_a,
            frequencies=freqs_a,
            eigenvectors=eigs,
        )

    def E(self, q_idx: int, normalize: bool = True) -> ndarray_complex:
        """Return eigenvector matrix at the given q-point.

        Args:
            q_idx (int): Index of the q-point.
            normalize (bool): Whether to normalize eigenvectors.

        Returns:
            ndarray_complex: (3N,3N) eigenvector matrix with columns as modes.
        """

        # Bounds check for q-point index
        if not (0 <= q_idx < self.q_positions.shape[0]):
            raise IndexError(f"q_idx out of range: {q_idx}")
        ev = self.eigenvectors[q_idx]  # (3N,N,3)
        # Flatten (N,3) per mode into (3N,) and stack as columns
        E = ev.reshape(self.nmodes, 3 * self.natom).T.copy()  # (3N,3N), cols are eigenvectors
        if normalize:
            norms = np.linalg.norm(E, axis=0)
            if np.any(norms <= eps):
                bad = np.where(norms <= eps)[0][:10].tolist()
                raise ValueError(f"Zero/near-zero eigenvector norm at modes {bad} (q_idx={q_idx})")
            E /= norms
        return E


@dataclass
class DFTCache:
    """Cached DFT-derived quantities for comparisons.

    Args:
        dq_flat (ndarray_realFloats): (3N,) mass-weighted displacement vector.
        dq_norm2 (float): Squared norm of dq_flat.
        masses (ndarray_realFloats): (N,) atomic masses.
        dft_path (str): Source band.yaml path.
        q_indices (List[int]): Selected DFT q-point indices.
        q_positions (ndarray_realFloats): (len(q_indices),3) q-point positions.
        freqs_by_q (List[ndarray_realFloats]): Per-q frequencies arrays.
        E_by_q (List[ndarray_complex]): Per-q eigenvector matrices.
        AvgProjPowX_by_q (List[Dict[str, Any]]): AvgProjPowX selection artifacts per q.
        clusters_by_q (List[List[List[int]]]): Per-q clusters of mode indices.
        cluster_ranges_by_q (List[List[Tuple[float, float]]]): Per-q (fmin,fmax) per cluster.
        w_dft_by_q (List[ndarray_realFloats]): Per-q DFT cluster weights.
        Q_cluster_by_q (List[List[ndarray_complex]]): Per-q orth bases per cluster.

    Returns:
        None
    """

    dq_flat: ndarray_realFloats  # (3N,)
    dq_norm2: float
    masses: ndarray_realFloats  # (N,)
    dft_path: str
    q_indices: List[int]
    q_positions: ndarray_realFloats  # (len(q_indices),3)

    freqs_by_q: List[ndarray_realFloats]  # each (3N,)
    E_by_q: List[ndarray_complex]  # each (3N,3N)
    AvgProjPowX_by_q: List[Dict[str, Any]]  # per-q artifact

    clusters_by_q: List[List[List[int]]]  # per-q clusters over all modes
    cluster_ranges_by_q: List[List[Tuple[float, float]]]  # per-q (fmin,fmax)
    w_dft_by_q: List[ndarray_realFloats]  # per-q (nclusters,)
    Q_cluster_by_q: List[List[ndarray_complex]]  # per-q list of orth bases per cluster

    w_mode_by_q: List[ndarray_realFloats]  # per-q (3N,) GS→ES per-mode weights (DFT-defined)
    valid_modes_by_q: List[np.ndarray]  # per-q indices used in GS→ES score


@dataclass
class ComparisonOutput:
    """Container for DFT cache and ML comparison results.

    Args:
        dft_cache (DFTCache): Cached DFT data.
        results_per_ml (Dict[str, Dict[str, Any]]): Results keyed by ML band path.

    Returns:
        None
    """

    dft_cache: DFTCache
    results_per_ml: Dict[str, Dict[str, Any]]
