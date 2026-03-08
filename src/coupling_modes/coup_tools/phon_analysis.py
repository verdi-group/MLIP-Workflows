from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .phon_classes import (
    BandData,
    ComparisonOutput,
    DFTCache,
    Structure,
    eps,
    ndarray_complex,
    ndarray_realFloats,
)
from .phon_parsers import read_poscar


 
def compute_dq_flat(
    gs: Structure,
    es: Structure,
    masses: ndarray_realFloats,
    lattice_tol: float,
    wrap_minimum_image: bool = True,
    remove_mass_weighted_com: bool = True,
) -> ndarray_realFloats:
    """Compute mass-weighted displacement vector between structures.

    Args:
        gs (Structure): Ground-state structure.
        es (Structure): Excited-state structure.
        masses (ndarray_realFloats): (N,) atomic masses.
        lattice_tol (float): Tolerance for lattice mismatch.
        wrap_minimum_image (bool): Whether to wrap displacements to minimum image.
        remove_mass_weighted_com (bool): Remove mass-weighted center-of-mass shift.

    Returns:
        ndarray_realFloats: (3N,) flattened mass-weighted displacement vector.
    """

    # Sanity checks: same atom count and similar lattice
    if gs.frac.shape != es.frac.shape:
        raise ValueError(f"GS/ES atom count mismatch: {gs.frac.shape} vs {es.frac.shape}")
    if np.max(np.abs(gs.lattice - es.lattice)) > lattice_tol:
        raise ValueError(f"GS/ES lattice mismatch beyond tol={lattice_tol}")

    # Mass array must match number of atoms
    n = int(gs.frac.shape[0])
    masses = np.asarray(masses, dtype=float).reshape(-1)
    if masses.shape != (n,):
        raise ValueError(f"Masses length mismatch: expected {n}, got {masses.shape}")

    # Fractional displacement (ES - GS)
    df = es.frac - gs.frac
    if wrap_minimum_image: # keep displacements relative to the nearest
        # two versions of the atoms, since we are dealing with PBC.
        df = (df + 0.5) % 1.0 - 0.5

    # Convert to Cartesian using GS lattice
    dr = df @ gs.lattice  # (N,3)

    if remove_mass_weighted_com: # needs to be applied before it is mass weighted.
        # basic removal of centre of mass, stops the acoustic phonons from dominating
        w = masses[:, None]
        com = np.sum(w * dr, axis=0) / float(np.sum(masses))
        dr = dr - com[None, :]
    # Apply mass-weighting (equivalent to diagonal mass matrix, but broadcasted)
    dq = (np.sqrt(masses)[:, None] * dr).reshape(3 * n)  # (3N,)
    return dq.astype(float, copy=False)


 
def choose_q_indices(band: BandData, gamma_only: bool, q_tol: float, select_unique_qpts: bool = True) -> List[int]:
    """Select q-point indices to use from band data.
    Robust against duplicate qpoints in DFT band.yaml if gamma_only is true. Now robust against duplicates 
    if gamma_only is not true. we use equality, so if the qpointsa are slightly different (float) then 
    they will not be removed. This can be fixed by supplementing a rounding tool prior. I thought 
    it best to keep to equality for now.
    
    Args:
        band (BandData): Phonon band data.
        gamma_only (bool): If True, select only the q-point closest to Gamma.
        q_tol (float): Tolerance for selecting Gamma.

    Returns:
        List[int]: Indices of selected q-points.
    """

    if gamma_only:
        # Select only the q-point closest to Gamma
        norms = np.linalg.norm(band.q_positions, axis=1)  # (nq,)
        idx = int(np.argmin(norms)) # argmin returns a single index everytime. 
        if float(norms[idx]) > q_tol:
            raise ValueError(
                f"gamma_only=True but nearest q is {band.q_positions[idx].tolist()} (||q||={float(norms[idx])})"
            )
        return [idx]
    
    elif select_unique_qpts: # this ensures there are no qpoint duplicates kept.
        qpts = band.q_positions
        #rounded = np.round(qpts / q_tol).astype(int) 
        _, keep = np.unique(qpts, axis=0, return_index=True)
        # np.unique returns (sorted array, unique indicies)
        keep = np.sort(keep)  # preserve original order
        return keep.tolist()

    # Otherwise use all q-points: indices 0..nq-1
    return list(range(int(band.q_positions.shape[0])))


 
def match_q_indices(dft_qpos: ndarray_realFloats, ml_band: BandData, q_tol: float) -> List[int]:
    """Match DFT q-points to nearest ML q-points.
    This is for robustness, most of the time this will be functionless. 
    In the case of several q points however this is useful. For instance, 
    if for whatever reason the band.yaml contains two identical qpoints (for ML) 
    with data duplication, this returns a singular index, and the remaining qpoint 
    is left alone. 

    This robustness is also implmeneted for DFT.

    Args:
        dft_qpos (ndarray_realFloats): (nq,3) DFT q-point positions.
        ml_band (BandData): ML phonon band data.
        q_tol (float): Maximum distance allowed for a match.

    Returns:
        List[int]: Indices in ml_band matching each DFT q-point.
    """

    # dft_qpos: (nq,3), ml_q: (nq_ml,3)
    ml_q = ml_band.q_positions
    out: List[int] = []
    for q in dft_qpos:
        # Nearest-neighbor match for high symmetry qpoint in reciprocal space.
        d = np.linalg.norm(ml_q - q[None, :], axis=1)  # (nq_ml,)
        j = int(np.argmin(d)) # returns one index. 
        if float(d[j]) > q_tol:
            raise ValueError(
                f"No ML q-point within q_tol={q_tol} of DFT q={q.tolist()} "
                f"(best={ml_q[j].tolist()}, dist={float(d[j])})"
            )
        out.append(j)
    return out


 
def AvgProjPowX_artifact_for_q(
    E_full: ndarray_complex,
    freqs: ndarray_realFloats,
    dq_flat: ndarray_realFloats,
    threshold: float,
    freq_cluster_tol: float,
    top_k_preview: int = 10,
) -> Dict[str, Any]:
    """Compute AvgProjPowX selection artifacts for a single q-point.

    Args:
        E_full (ndarray_complex): (3N,3N) eigenvector matrix.
        freqs (ndarray_realFloats): (3N,) frequencies for the q-point.
        dq_flat (ndarray_realFloats): (3N,) mass-weighted displacement vector (flattened to (3N,) rather than (N,3)
        threshold (float): Cumulative weight threshold for selection (tau)
        freq_cluster_tol (float): Frequency clustering tolerance.
        top_k_preview (int): Number of coupling modes to preview. the details of the first 10 will be displayed.

    Returns:
        Dict[str, Any]: Selection indices, clusters, and summary stats to be printed in the render report
    """

    # Convert to complex to align with complex eigenvectors
    dq = dq_flat.astype(np.complex128, copy=False)
    dq_norm2 = float(np.vdot(dq, dq).real)
    if dq_norm2 <= eps:
        raise ValueError("dq is (near) zero; cannot form projections")

    # Projection per mode: p_m = |e_m^{\dagger} dq|^2 / ||dq||^2
    proj = E_full.conj().T @ dq  # (3N,)
    p = (np.abs(proj) ** 2) / dq_norm2

    # Sort descending and take smallest prefix that reaches threshold
    order = np.argsort(-p, kind="stable")  # indices sorted by descending p. Use
    # a stable sorter  (ends up being mergesort) just because that is resiliant to 
    # mixing p index if p is equivalent. 

    csum = np.cumsum(p[order])  # cumulative contribution
    # np.cumsum is a generator for the sum of X indicies. = Sum top 1, Sum top2 , top 3 ...
    k = int(np.searchsorted(csum, threshold, side="left")) + 1 
    # searchsorted gives you the index of where to put a value to keep an array sorted. 
    # we use it for threshold as it tells us where tau should be put to maintain ordered ness 
    # of csum, and therefore, the index + 1 would have a cumulative sum greater than tau, and index -1
    # would have a cumulative sum less than tau. We want \geq \tau so use +1. 
    # sum top i maps to sum up to and including the ith index in p so these align fine. 
    k = min(k, int(p.size)) # in case np.searchsorted returns len(csum) which would mean
    # we want the len(csum)+1 term in p, but p is only len(csum) long.

    selected = order[:k].tolist()  #couplign modes

    # Group selected modes into near-degenerate frequency clusters
    clusters_selected = cluster_mode_indices_by_frequency(freqs, selected, freq_cluster_tol)

    sum_p = float(np.sum(p))
    warn = None
    if not (0.98 <= sum_p <= 1.02):
        warn = f"sum(p)={sum_p:.6f} outside [0.98,1.02]"

    # Small preview list for logging/inspection
    preview = []
    for idx in order[: min(top_k_preview, int(p.size))]:
        preview.append((int(idx), float(freqs[idx]), float(p[idx])))
    # above will thus show the index, frequency and projection power of the top coupling mdoes
    
    # eigen vectors could be appended but they are of shape (3N,). Can be retrieved from band.yaml with:
    # yam = BandData.from_yaml(path) #use the .eigenvector method
    # evs = yam.eigenvector 
    # print(evs[idx-of-interest])

    return {
        "selected_indices": selected,
        "selected_cumsum_last": float(csum[k - 1]) if k > 0 else 0.0,
        "clusters_selected": clusters_selected,
        "sum_p": sum_p,
        "sum_p_warning": warn,
        "top_contrib_preview": preview,  # (mode, freq, p) small summary only
    }


 
def cluster_mode_indices_by_frequency(
    freqs: ndarray_realFloats,
    indices: Optional[Sequence[int]],
    freq_tol: float,
) -> List[List[int]]:
    """Cluster mode indices by proximity in frequency. This is:
    group by L1 norm \leq freq_tol, otherwise:

    |\omega_{m_{i+1}}(\mathbf q) - \omega_{m_i}(\mathbf q)| > freq_tol → start a new cluster
    
    this gives our eigen spaces.

    Args:
        freqs (ndarray_realFloats): (3N,) frequencies.
        indices (Optional[Sequence[int]]): Mode indices to cluster, or None for all.
        freq_tol (float): Maximum frequency gap within a cluster.

    Returns:
        List[List[int]]: Clusters of mode indices.
    """

    # Use all indices if none provided
    # Gets an array of the indices by the arange generator or by what is specified
    # (the coupling indicies)

    if indices is None:
        idx = np.arange(freqs.size, dtype=int)
    else:
        idx = np.asarray(list(indices), dtype=int)

    # Sort selected indices by frequency (ascending)
    f = freqs[idx]
    order = np.argsort(f, kind="stable") # again we use stable to ensure that index is 
    # conserved if two members of the array are equivalent.
    idx_sorted = idx[order].tolist() # indicies of where frequencies appear such that 
    # they are in ascending order. idx_sorted[0] lowest frequency index etc... 
    f_sorted = f[order] # the sorted frequencies themselves. 

    clusters: List[List[int]] = []
    cur: List[int] = [idx_sorted[0]]  # initialise our cluster. cur holds 
    # the indices of the clustered modes. 
    for i in range(1, len(idx_sorted)):
        # Start new cluster when frequency gap exceeds tolerance
        if float(f_sorted[i] - f_sorted[i - 1]) > freq_tol:
            clusters.append(cur) # stash cur
            cur = [idx_sorted[i]] # reset cur (the beginning of a new cluster.)
        else:
            cur.append(idx_sorted[i])
    clusters.append(cur)
    return clusters


 
def orth(A: ndarray_complex, rcond: float = eps) -> ndarray_complex:
    """Compute an orthonormal basis for the column space of A.

    Args:
        A (ndarray_complex): Input matrix.
        rcond (float): Relative tolerance for rank determination.

    Returns:
        ndarray_complex: Orthonormal basis matrix with shape (m, r).
    """

    # QR-based orthonormal basis with rank cutoff
    if A.size == 0:
        return A[:, :0].copy()
    Q, R = np.linalg.qr(A, mode="reduced")  # Q: (m,k), R: (k,n), k=min(m,n)
    # "reduced" gives just the orthonormal columns spanning col(A) (no extra basis vectors)
    if R.size == 0:
        return Q[:, :0].copy()
    d = np.abs(np.diag(R))
    if d.size == 0:
        return Q[:, :0].copy()
    # Keep columns with diagonal(R) above tolerance. We want to get rid of any nearly 
    # dependent columns, by nearly we mean vanishingly small, rcond is 1e-12 (eps). 
    tol = float(d.max()) * float(rcond)
    r = int(np.sum(d > tol))
    return Q[:, :r]


 
def principal_angles(Qd: ndarray_complex, Qm: ndarray_complex) -> Tuple[ndarray_realFloats, ndarray_realFloats, float]:
    """Compute principal angles between two subspaces.

    Args:
        Qd (ndarray_complex): Orthonormal basis for DFT subspace.
        Qm (ndarray_complex): Orthonormal basis for ML subspace.

    Returns:
        Tuple[ndarray_realFloats, ndarray_realFloats, float]: Singular values, angles (deg), and X score. 
        (avg projection power)
    """

    # Qd: (3N,kd), Qm: (3N,km)
    kd = int(Qd.shape[1])
    if kd == 0: # these cases need to be handled as it is possible that there are 
        # no modes in the frequency of the coupling mode under study. 
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float), float("nan")
    km = int(Qm.shape[1])
    if km == 0:
        return np.zeros((0,), dtype=float), np.full((0,), 90.0, dtype=float), 0.0

    # Overlap matrix and its singular values define principal angles
    C = Qd.conj().T @ Qm  # (kd,km)
    sig = np.linalg.svd(C, compute_uv=False)
    sig = np.clip(sig.real.astype(float, copy=False), 0.0, 1.0) # should already be in [0,1] 
    # but we are dealing with floats... 
    theta = np.degrees(np.arccos(sig))
    X = float(np.sum(sig**2) / kd)
    return sig, theta, X


 
def acoustic_mode_indices(
    freqs: ndarray_realFloats, n_acoustic: int = 3, abs_tol: float = 1e-4
) -> np.ndarray:
    """Return indices of acoustic translational modes (Gamma) robustly. 
    (just takes the top 3 coupling modes. this is irrelevant if the center of mass 
    displacement has been removed from the displacement vector (default is that it has).)"""
    f = np.asarray(freqs, dtype=float)  # (3N,)
    af = np.abs(f)
    near0 = np.where(af <= abs_tol)[0]
    if near0.size >= n_acoustic:
        order = np.argsort(af[near0], kind="stable")[:n_acoustic]
        return near0[order]
    return np.argsort(af, kind="stable")[:n_acoustic]


 
def dft_mode_weights_for_q(
    E_full: ndarray_complex,
    freqs: ndarray_realFloats,
    dq_flat: ndarray_realFloats,
    *,
    kind: str = "p",  # "p", "S", "lambda"
    n_acoustic: int = 3,
    acoustic_abs_tol: float = 1e-4,
) -> Tuple[ndarray_realFloats, np.ndarray]:
    
    """
    DFT-defined per-mode weights for DeltaQ scores and MLIP ranking.

    This builds a per-mode importance weight using the DFT eigenvectors and
    the GS to ES mass-weighted displacement 'dq'. These weights are then used to
    compute weighted RMS errors (frequency + eigenvector mismatch) when matching
    ML modes to DFT modes via the assignment step. In other words, this function
    decides which DFT modes "matter" most for the ranking of MLIPs.
    
    a measures projection of weighted displacement vector onto eigenmode.

    The returned 'w' and 'valid' are consumed in 'gses_score_from_assignment':
      - 'w' supplies the weights in the RMS metrics (E_freq, E_vec, Score).
      - 'valid' selects the subset of modes that actually contribute (w > 0).

    use to rank mlips by a) p => how well the mode describes the geometry change 
    b) S or Lambda, they decrease weight of soft modes and increase the 
    weight of higher frequency modes -- typically use lambda for this because in the harmonid 
    approx energy goes with w^2 a^2. 
     
     so b) used to answer: does the mlip reproduce the energetically 
    important eignemodes? 

    use a) to answer: does the mlip reprorduce the geometrically importat eigenodes? 


    kind:
      - "p": |a|^2
      - "S":  \omega|a|^2
      - "lambda": \omega^2|a|^2
    """
    # Project the displacement onto eigenvectors to get per-mode amplitudes.
    # These amplitudes quantify how strongly each DFT mode participates
    # in the structural change (GS→ES) encoded in dq.
    dq = dq_flat.astype(np.complex128, copy=False)
    proj = E_full.conj().T @ dq # (3N,) 
    a2 = (np.abs(proj) ** 2).astype(float) # |a|^2
    # Frequency magnitudes (non-negative), used to scale weights by energy.
    omega = np.abs(np.asarray(freqs, dtype=float))  # |\omega|

    if kind == "p":
        # Pure projection power: importance is just how much dq lies in each mode.
        w = a2
    elif kind == "S":
        # weight by |\omega| to emphasize higher-frequency contributions mildly.
        w = omega * a2
    elif kind == "lambda":
        # Weight by |\omega|^2 to emphasize higher-frequency contributions  strongly.
        w = (omega ** 2) * a2
    else:
        raise ValueError(f"Unknown weight kind='{kind}' (use 'p','S','lambda')")

    # Remove acoustic translations at Gamma (zero their weights)
    # commented out because of the implemented COM removal.
    # ac = acoustic_mode_indices(freqs, n_acoustic=n_acoustic, abs_tol=acoustic_abs_tol)
    # w[ac] = 0.0

    # Only modes with nonzero weight are considered "important" for the GS to ES score (called dQ score).
    # This avoids polluting the weighted RMS with modes that dq does not excite.
    valid = np.where(w > 0.0)[0]  # indices with nonzero weight
    return w, valid


 
def overlap_sq(E_dft: ndarray_complex, E_ml: ndarray_complex) -> ndarray_realFloats:
    """Compute phase-invariant overlaps between DFT and ML eigenvector bases.

    Feed this into the Hungarian assignment in the GS->ES dQ score. We maximize
    the diagonal overlap between DFT and ML modes to get a one-to-one
    matching before scoring frequency/eigenvector errors.

    Shapes:
      - E_dft:(3N,3N) DFT eigenvectors (columns are modes).
      - E_ml: (3N,3N) ML eigenvectors (columns are modes).
      - O: (3N,3N) overlaps in [0,1], O_ij = |e_i^H e_j|^2.

    gives the cost matrix for hungarian_min. 
    """
    # Overlap between DFT and ML eigenvector bases (phase invariant).
    C = E_dft.conj().T @ E_ml  # (3N,3N)
    return (np.abs(C) ** 2).astype(float, copy=False)  # elementwise |.|^2


 
def hungarian_min(cost: ndarray_realFloats) -> np.ndarray:
    """
    Solve min-cost assignment for a square cost matrix (n,n).
    Returns assignment a where a[i] = j chosen for row i.

    hungarian minimisatino algo.
    used by `hungarian_maximize`, which is used to match DFT modes to ML modes 
    for the GS->ES dQ score.

    cost: (n,n) array.
    return: (n,) int array mapping row i to chosen column a[i].
    """
    a = np.asarray(cost, dtype=float)  # (n,n) cost matrix
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"hungarian_min requires square matrix, got {a.shape}")
    n = int(a.shape[0])

    # Use 1-based indexing internally to match the classic Hungarian pseudocode.
    u = np.zeros(n + 1, dtype=float) # row potentials
    v = np.zeros(n + 1, dtype=float) # column potentials
    p = np.zeros(n + 1, dtype=int) # matching for columns: p[j] = i
    way = np.zeros(n + 1, dtype=int) # predecessor tracking

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf, dtype=float)
        used = np.zeros(n + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0

            for j in range(1, n + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assign = np.empty(n, dtype=int)  # row to column assignment
    for j in range(1, n + 1):
        i = p[j]
        if i != 0:
            assign[i - 1] = j - 1
    return assign


 
def hungarian_maximize(weight: ndarray_realFloats) -> np.ndarray:
    """Maximize sum weight[i, assign[i]] for square matrix.

    Shapes:
      - weight: (n,n) overlap/score matrix.
      - return: (n,) assignment array.
    """
    W = np.asarray(weight, dtype=float)  # (n,n) weight/overlap matrix
    wmax = float(np.max(W)) if W.size else 0.0
    cost = wmax - W
    return hungarian_min(cost)


 
def gses_score_from_assignment(
    freqs_dft: ndarray_realFloats,
    freqs_ml: ndarray_realFloats,
    w_mode: ndarray_realFloats,
    valid_modes: np.ndarray,
    assign: np.ndarray,
    O: ndarray_realFloats,
    *,
    omega_floor: float = 1e-4,
    alpha: float = 0.5,
) -> Dict[str, float]:
    """
    Compute dQ score from a DFT→ML assignment:
      - E_freq (weighted RMS abs error)
      - E_freq_rel (weighted RMS relative error with omega_floor)
      - E_vec (weighted RMS eigenvector mismatch)
      - Score = E_freq + alpha * E_vec

    This is the GS->ES score ranking channel. It is called from 'compare_one_ml'
    after the optimal DFT->ML assignment is found, and it uses DFT-defined
    per-mode weights (from 'dft_mode_weights_for_q') to emphasize the
    displacement-relevant modes.

    Shapes:
    freqs_dft, freqs_ml: (3N,) arrays.
    w_mode: (3N,) weights.
    valid_modes: (n_valid,) indices where w_mode > 0.
    assign: (3N,) DFT->ML mapping.
    O: (3N,3N) overlap matrix.
    """
    # Shapes: freqs_* are (3N,), w_mode is (3N,), idx is list of valid mode indices.
    f_d = np.asarray(freqs_dft, dtype=float)
    f_m = np.asarray(freqs_ml, dtype=float)
    w = np.asarray(w_mode, dtype=float)
    idx = np.asarray(valid_modes, dtype=int)

    if idx.size == 0:
        return {
            "E_freq": float("nan"),
            "E_freq_rel": float("nan"),
            "E_vec": float("nan"),
            "Score": float("nan"),
        }

    # Assignment: pi[i] is the ML mode index matched to DFT mode i.
    pi = np.asarray(assign, dtype=int)  # length 3N, maps DFT -> ML
    j = pi[idx]  # ML mode indices matched to each DFT mode in idx

    # Frequency errors on valid modes (matched DFT->ML pairs).
    dw = f_m[j] - f_d[idx]
    denom = float(np.sum(w[idx]))
    if denom <= eps:
        # Fallback to uniform weights on idx if weights vanish.
        wf = np.ones(idx.size, dtype=float)
        denom = float(np.sum(wf))
    else:
        wf = w[idx]

    E_freq = float(np.sqrt(np.sum(wf * (dw ** 2)) / denom))

    # Relative errors with floor to avoid div-by-zero near acoustic modes
    rel = dw / (np.abs(f_d[idx]) + float(omega_floor))
    E_freq_rel = float(np.sqrt(np.sum(wf * (rel ** 2)) / denom))

    # Overlaps for matched modes (values in [0,1]) quantify eigenvector similarity.
    odiag = O[idx, j]
    E_vec = float(np.sqrt(np.sum(wf * (1.0 - odiag)) / denom))

    Score = float(E_freq + float(alpha) * E_vec)

    return {
        "E_freq": E_freq,
        "E_freq_rel": E_freq_rel,
        "E_vec": E_vec,
        "Score": Score,
    }


 
def cluster_basis_and_weight(
    E_full: ndarray_complex,
    dq: ndarray_complex,
    dq_norm2: float,
    cluster: Sequence[int],
) -> Tuple[ndarray_complex, float]:
    """Compute cluster basis and its weight on dq.
    Used in `build_dft_cache` to convert each DFT frequency cluster into
    a subspace basis and a scalar coupling weight. These are later used
    to identify relevant clusters 

    Shapes:
    E_full: (3N,3N) eigenvector matrix.
    cluster: list of mode indices, length k_cluster.
    Q: (3N,k_cluster) orthonormal basis.
    w: scalar in [0,1], fraction of dq captured by this cluster.

    Args:
        E_full (ndarray_complex): (3N,3N) eigenvector matrix.
        dq (ndarray_complex): (3N,) mass-weighted displacement vector.
        dq_norm2 (float): Squared norm of dq.
        cluster (Sequence[int]): Mode indices in the cluster.

    Returns:
        Tuple[ndarray_complex, float]: Orthonormal basis Q and weight w.
    """

    # Build cluster subspace from selected eigenvectors.
    A = E_full[:, list(cluster)]  # (3N, k_cluster)
    Q = orth(A)
    if Q.shape[1] == 0:
        return Q, 0.0
    # Weight = projection of dq onto cluster subspace.
    v = Q.conj().T @ dq
    w = float(np.sum(np.abs(v) ** 2) / dq_norm2)
    return Q, w


  
def top_clusters_by_weight(weights: ndarray_realFloats, threshold: float) -> List[int]:
    """Select cluster indices by cumulative weight threshold. (\tau)

    Args:
        weights (ndarray_realFloats): Cluster weights.
        threshold (float): Cumulative weight threshold.

    Returns:
        List[int]: Indices of selected clusters.
    """

    # select highest-weight clusters until cumulative threshold reached
    # same as what was done to the modes. 
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return []
    order = np.argsort(-w, kind="stable")
    c = np.cumsum(w[order])
    k = int(np.searchsorted(c, threshold, side="left")) + 1
    k = min(k, int(w.size))
    return order[:k].tolist()


 
def build_dft_cache(
    contcar_gs: Union[str, Path],
    contcar_es: Union[str, Path],
    band_dft_path: Union[str, Path],
    q_tol: float,
    lattice_tol: float,
    threshold: float,
    freq_cluster_tol: float,
    freq_window: float,
    remove_mass_weighted_com: bool,
    gamma_only: bool,
    weight_type: str = "p", # "p", "S", or "lambda"
) -> DFTCache:
    """Build cached DFT data and derived quantities.

    One-time preprocessing step. Computes the GS->ES displacement (dq),
    loads DFT phonons, selects q-points, and derives all DFT-side        
    quantities needed to compare *multiple* MLIPs efficiently.
    The resulting cache contains per-q eigenvectors, per-mode weights,
    frequency clusters, and cluster subspace weights used in ranking.

    Shapes (key cached fields):
      dq_flat: (3N,) mass-weighted displacement.
     freqs_by_q: list of (3N,) arrays.
      E_by_q: list of (3N,3N) eigenvector matrices.
      clusters_by_q: list of list-of-mode-index clusters.
      Q_cluster_by_q: list of list of (3N,k_cluster) bases.
      w_dft_by_q: list of (ncluster,) DFT cluster weights.

    Args:
        contcar_gs (Union[str, Path]): Ground-state CONTCAR path.
        contcar_es (Union[str, Path]): Excited-state CONTCAR path.
        band_dft_path (Union[str, Path]): DFT band.yaml path.
        q_tol (float): Q-point match tolerance.
        lattice_tol (float): Lattice mismatch tolerance.
        threshold (float): Cumulative weight threshold.
        freq_cluster_tol (float): Frequency cluster tolerance.
        freq_window (float): Frequency window for ML clustering.
        remove_mass_weighted_com (bool): Remove COM shift if True.
        gamma_only (bool): Only use Gamma if True.

    Returns:
        DFTCache: Cached DFT-derived data.
    """

    # Load ground/excited structures (same atom ordering and similar lattice).
    gs = read_poscar(contcar_gs)
    es = read_poscar(contcar_es)

    # Load DFT band data (phonopy band.yaml).
    dft = BandData.from_yaml(band_dft_path)

    if dft.natom != int(gs.frac.shape[0]):
        raise ValueError(f"DFT natom ({dft.natom}) != CONTCAR natom ({int(gs.frac.shape[0])})")

    # Mass-weighted displacement vector between GS and ES.
    dq_flat = compute_dq_flat(
        gs=gs,
        es=es,
        masses=dft.masses,
        lattice_tol=lattice_tol,
        wrap_minimum_image=True,
        remove_mass_weighted_com=remove_mass_weighted_com,
    )
    # Complex dtype for projections with complex eigenvectors.
    dq = dq_flat.astype(np.complex128, copy=False)
    dq_norm2 = float(np.vdot(dq, dq).real)
    if dq_norm2 <= eps:
        raise ValueError("dq is (near) zero; check CONTCAR_gs/CONTCAR_es consistency")

    # Select which q-points to use (Gamma-only or all), and de-duplicate if needed.
    q_indices = choose_q_indices(dft, gamma_only=gamma_only, q_tol=q_tol)
    q_positions = dft.q_positions[q_indices].copy()

    # Per-q storage for frequencies, eigenvectors, and selection artifacts.
    freqs_by_q: List[ndarray_realFloats] = [] # ndarray_realFloats for arryas expected to be floats
    E_by_q: List[ndarray_complex] = [] #ndarray_complex for arrays expected to be complex
    AvgProjPowX_by_q: List[Dict[str, Any]] = []

    # Per-q frequency clusters and their subspace information (used in research metrics).
    clusters_by_q: List[List[List[int]]] = []
    cluster_ranges_by_q: List[List[Tuple[float, float]]] = []
    w_dft_by_q: List[ndarray_realFloats] = []
    Q_cluster_by_q: List[List[ndarray_complex]] = []

    w_mode_by_q: List[ndarray_realFloats] = []
    valid_modes_by_q: List[np.ndarray] = []

    for qi in q_indices:
        # Pull frequencies and eigenvectors for each selected q-point.
        freqs = dft.frequencies[qi].astype(float, copy=False)
        if freqs.shape[0] != 3 * dft.natom:
            raise ValueError(f"DFT nmodes mismatch at q_idx={qi}: got {freqs.shape[0]}")
        E_full = dft.E(qi, normalize=True)

        # --- GS→ES per-mode weights (DFT-defined) ---
        # These weights determine which modes matter most for the dQ score
        # used in the final MLIP ranking.
        w_mode, valid_modes = dft_mode_weights_for_q(
            E_full=E_full,
            freqs=freqs,
            dq_flat=dq_flat,
            kind=weight_type,
            n_acoustic=3,
            acoustic_abs_tol=1e-4,
        )

        # AvgProjPowX is the average projected power of the mlip
        # coupling modes onto DFT's coupling modes. X is just kept
        # because that is what i initially called it mathematically haha ...
        AvgProjPowX = AvgProjPowX_artifact_for_q(
            E_full=E_full,
            freqs=freqs,
            dq_flat=dq_flat,
            threshold=threshold,
            freq_cluster_tol=freq_cluster_tol,
        )

        # Cluster all modes by frequency proximity (degeneracy groups).
        clusters = cluster_mode_indices_by_frequency(freqs, indices=None, freq_tol=freq_cluster_tol)

        Qs: List[ndarray_complex] = []
        ws: List[float] = []
        ranges: List[Tuple[float, float]] = []
        for cl in clusters:
            # Cluster frequency range and its projection weight.
            fcl = freqs[list(cl)]
            fmin = float(np.min(fcl))
            fmax = float(np.max(fcl))
            Qc, wc = cluster_basis_and_weight(E_full, dq, dq_norm2, cl)
            Qs.append(Qc)
            ws.append(wc)
            ranges.append((fmin, fmax))

        freqs_by_q.append(freqs.copy())
        E_by_q.append(E_full)
        AvgProjPowX_by_q.append(AvgProjPowX)

        # Cache DFT cluster data for later ML comparisons.
        clusters_by_q.append(clusters)
        cluster_ranges_by_q.append(ranges)
        w_dft_by_q.append(np.asarray(ws, dtype=float))
        Q_cluster_by_q.append(Qs)

        # Cache the per-mode weights and valid indices for the dQ score.
        w_mode_by_q.append(w_mode.copy())  # (3N,)
        valid_modes_by_q.append(valid_modes.copy())  # (n_valid,)

    # DFTCache is reused for every ML model (avoids recomputing DFT-side quantities).
    return DFTCache(
        dq_flat=dq_flat,
        dq_norm2=dq_norm2,
        masses=dft.masses.copy(),
        dft_path=str(Path(band_dft_path)),
        q_indices=q_indices,
        q_positions=q_positions,
        freqs_by_q=freqs_by_q,
        E_by_q=E_by_q,
        AvgProjPowX_by_q=AvgProjPowX_by_q,
        clusters_by_q=clusters_by_q,
        cluster_ranges_by_q=cluster_ranges_by_q,
        w_dft_by_q=w_dft_by_q,
        Q_cluster_by_q=Q_cluster_by_q,
        w_mode_by_q=w_mode_by_q,
        valid_modes_by_q=valid_modes_by_q,
    )


   
def compare_one_ml(
    cache: DFTCache,
    ml_path: Union[str, Path],
    q_tol: float,
    threshold: float,
    freq_cluster_tol: float,
    freq_window: float,
    alpha: float = 0.5,
) -> Dict[str, Any]:
    """Compare one ML band.yaml against cached DFT data.
        Core per-model evaluation. Computes: AvgProjPowX subspace agreement, cluster-window
        agreement, and GS->ES dQ score used for final ranking.

    Shapes:
      cache.E_by_q[iq]: (3N,3N) DFT eigenvector matrix.
      freqs_ml: (3N,) ML frequencies at matched q.
      AvgProjPowX_per_q: list of dicts with X, sigma, theta, etc.
      research_per_q: list of dicts with cluster-window metrics.
      gses_per_q: list of dicts with E_freq/E_vec/Score.

    Args:
        cache (DFTCache): Cached DFT data.
        ml_path (Union[str, Path]): ML band.yaml path.
        q_tol (float): Q-point match tolerance.
        threshold (float): Cumulative weight threshold.
        freq_cluster_tol (float): Frequency cluster tolerance.
        freq_window (float): Frequency window for ML clustering.

    Returns:
        Dict[str, Any]: Comparison results for AvgProjPowX and coupling cluster analysis.
    """
    ml = BandData.from_yaml(ml_path)

    n = int(cache.masses.shape[0])
    if ml.natom != n:
        raise ValueError(f"ML natom ({ml.natom}) != DFT natom ({n}) for {ml_path}")

    # match DFT q-points to nearest ML q-points.
    ml_q_indices = match_q_indices(cache.q_positions, ml, q_tol=q_tol)

    # reuse cached dq for projections.
    dq = cache.dq_flat.astype(np.complex128, copy=False)
    dq_norm2 = cache.dq_norm2

    # (AvgProjPowX) per-q and aggregated.
    AvgProjPowX_per_q: List[Dict[str, Any]] = []
    Xs: List[float] = []

    research_per_q: List[Dict[str, Any]] = []
    l1s: List[float] = []
    angle_scores: List[float] = []
    sigma2_scores: List[float] = []

    # GS->ES targeted (dQ score) metrics per-qm=, aggregated
    gses_per_q: List[Dict[str, Any]] = []
    gses_scores: List[float] = []
    gses_Efreq: List[float] = []
    gses_Evec: List[float] = []
    gses_EfreqRel: List[float] = []

    for iq, (dft_qi, ml_qi) in enumerate(zip(cache.q_indices, ml_q_indices)):
        # Pull ML data for matched q-point.
        freqs_ml = ml.frequencies[ml_qi].astype(float, copy=False)
        if freqs_ml.shape[0] != 3 * ml.natom:
            raise ValueError(f"ML nmodes mismatch at q_idx={ml_qi} for {ml_path}")

        E_ml = ml.E(ml_qi, normalize=True)  # (3N,3N)

        # GS→ES targeted: DFT-weighted optimal mode matching + score 
        E_dft = cache.E_by_q[iq]  # (3N,3N)
        O = overlap_sq(E_dft, E_ml)  # (3N,3N) overlaps
        assign = hungarian_maximize(O)  # length 3N, DFT->ML assignment

        w_mode = cache.w_mode_by_q[iq]  # (3N,)
        valid_modes = cache.valid_modes_by_q[iq]  # subset of indices

        metrics = gses_score_from_assignment(
            freqs_dft=cache.freqs_by_q[iq],
            freqs_ml=freqs_ml,
            w_mode=w_mode,
            valid_modes=valid_modes,
            assign=assign,
            O=O,
            omega_floor=1e-4,
            alpha=alpha,
        )

        gses_per_q.append(
            {
                "dft_q_index": int(dft_qi),
                "ml_q_index": int(ml_qi),
                "q_position": cache.q_positions[iq].astype(float),
                **metrics,
            }
        )
        gses_scores.append(float(metrics["Score"]))
        gses_Efreq.append(float(metrics["E_freq"]))
        gses_EfreqRel.append(float(metrics["E_freq_rel"]))
        gses_Evec.append(float(metrics["E_vec"]))

        # Compute AvgProjPowX selection on ML side.
        AvgProjPowX_ml = AvgProjPowX_artifact_for_q(
            E_full=E_ml,
            freqs=freqs_ml,
            dq_flat=cache.dq_flat,
            threshold=threshold,
            freq_cluster_tol=freq_cluster_tol,
        )

        # Compare subspaces spanned by selected modes.
        dft_AvgProjPowX = cache.AvgProjPowX_by_q[iq]
        dft_sel = dft_AvgProjPowX["selected_indices"]
        ml_sel = AvgProjPowX_ml["selected_indices"]

        # Build orthonormal bases for the selected coupling modes.
        Qd = orth(cache.E_by_q[iq][:, dft_sel])  # (3N,k_dft)
        Qm = orth(E_ml[:, ml_sel])  # (3N,k_ml)
        sig, theta, X = principal_angles(Qd, Qm)

        AvgProjPowX_per_q.append(
            {
                "dft_q_index": int(dft_qi),
                "ml_q_index": int(ml_qi),
                "q_position": cache.q_positions[iq].astype(float),
                "k_dft": int(Qd.shape[1]),
                "k_ml": int(Qm.shape[1]),
                "sigma": sig,
                "theta_deg": theta,
                "X": float(X),
                "dft_sum_p_warning": dft_AvgProjPowX.get("sum_p_warning", None),
                "ml_sum_p_warning": AvgProjPowX_ml.get("sum_p_warning", None),
            }
        )
        Xs.append(float(X))

        # Pull data from DFT cache (so it is not computed every time).
        clusters = cache.clusters_by_q[iq]
        ranges = cache.cluster_ranges_by_q[iq]
        w_dft = cache.w_dft_by_q[iq]
        Qd_clusters = cache.Q_cluster_by_q[iq]

        # obtain the "coupling clusters" (highest DFT-weighted clusters).
        relevant = top_clusters_by_weight(w_dft, threshold=threshold)

        cluster_rows: List[Dict[str, Any]] = []
        w_ml_list: List[float] = []
        mean_theta_list: List[float] = []
        mean_sigma2_list: List[float] = []

        for cid, (cl, (fmin, fmax), QdC, wd) in enumerate(zip(clusters, ranges, Qd_clusters, w_dft)):
            # Window ML frequencies around each DFT cluster. These are the frequency bins to form
            # ml clusters. Aim is to test whether ML captures the the same modes in teh same frequency domains.
            lo = fmin - freq_window
            hi = fmax + freq_window
            window_idx = np.where((freqs_ml >= lo) & (freqs_ml <= hi))[0].astype(int).tolist()
            # window_idx gives the indices for where the ML freqs are in the respetive bin
            # (we loop over the dft clusters)

            if window_idx:
                # Compare cluster subspaces and weights within the window
                QmW = orth(E_ml[:, window_idx])
                v = QmW.conj().T @ dq
                wml = float(np.sum(np.abs(v) ** 2) / dq_norm2)
                sigC, thetaC, _ = principal_angles(QdC, QmW)
                mean_theta = float(np.mean(thetaC)) if thetaC.size else 90.0
                max_theta = float(np.max(thetaC)) if thetaC.size else 90.0
                sigma2 = float(np.sum(sigC**2) / max(int(QdC.shape[1]), 1)) if sigC.size else 0.0
            else:
                wml = 0.0
                sigC = np.zeros((0,), dtype=float)
                thetaC = np.zeros((0,), dtype=float)
                mean_theta = 90.0
                max_theta = 90.0
                sigma2 = 0.0

            w_ml_list.append(wml)
            mean_theta_list.append(mean_theta)
            mean_sigma2_list.append(sigma2)

            if cid in set(relevant):
                cluster_rows.append(
                    {
                        "cluster_id": int(cid),
                        "size_dft": int(len(cl)),
                        "freq_range_dft": (float(fmin), float(fmax)),
                        "w_dft": float(wd),
                        "w_ml_window": float(wml),
                        "theta_mean_deg": float(mean_theta),
                        "theta_max_deg": float(max_theta),
                        "sigma": sigC,
                    }
                )

        # Aggregate per-q measurement  for relevant clusters (coupling clusters)
        w_ml = np.asarray(w_ml_list, dtype=float)
        l1 = float(np.sum(np.abs(w_ml[relevant] - w_dft[relevant]))) if relevant else 0.0
        angle_score = float(np.sum(w_dft[relevant] * np.asarray(mean_theta_list, dtype=float)[relevant])) if relevant else 0.0
        sigma2_score = float(
            np.sum(w_dft[relevant] * (1.0 - np.asarray(mean_sigma2_list, dtype=float)[relevant]))
        ) if relevant else 0.0

        research_per_q.append(
            {
                "dft_q_index": int(dft_qi),
                "ml_q_index": int(ml_qi),
                "q_position": cache.q_positions[iq].astype(float),
                "relevant_clusters": relevant,
                "clusters_relevant": cluster_rows,
                "summary": {
                    "L1_weights_relevant": l1,
                    "weighted_mean_theta_deg_relevant": angle_score,
                    "weighted_1_minus_sigma2_relevant": sigma2_score,
                },
            }
        )
        l1s.append(l1)
        angle_scores.append(angle_score)
        sigma2_scores.append(sigma2_score)

    # Summaries across q-points (per-MLIP aggregates used in the final ranking).
    X_arr = np.asarray(Xs, dtype=float)
    AvgProjPowX_summary = {
        "X_mean": float(np.mean(X_arr)) if X_arr.size else float("nan"),
        "X_min": float(np.min(X_arr)) if X_arr.size else float("nan"),
        "X_max": float(np.max(X_arr)) if X_arr.size else float("nan"),
        "n_q": int(len(Xs)),
    }

    research_summary = {
        "L1_weights_mean": float(np.mean(l1s)) if l1s else float("nan"),
        "weighted_mean_theta_deg_mean": float(np.mean(angle_scores)) if angle_scores else float("nan"),
        "weighted_1_minus_sigma2_mean": float(np.mean(sigma2_scores)) if sigma2_scores else float("nan"),
        "n_q": int(len(l1s)),
    }

    gses_summary = {
        "Score_mean": float(np.mean(gses_scores)) if gses_scores else float("nan"),
        "Score_min": float(np.min(gses_scores)) if gses_scores else float("nan"),
        "E_freq_mean": float(np.mean(gses_Efreq)) if gses_Efreq else float("nan"),
        "E_freq_rel_mean": float(np.mean(gses_EfreqRel)) if gses_EfreqRel else float("nan"),
        "E_vec_mean": float(np.mean(gses_Evec)) if gses_Evec else float("nan"),
        "n_q": int(len(gses_scores)),
    }

    return {
        "AvgProjPowX": {"per_q": AvgProjPowX_per_q, "summary": AvgProjPowX_summary},
        "research": {"per_q": research_per_q, "summary": research_summary},
        "gses": {"per_q": gses_per_q, "summary": gses_summary},
    }


 
def run(
    contcar_gs: str,
    contcar_es: str,
    band_dft_path: str,
    band_ml_paths: List[str],
    q_tol: float,
    lattice_tol: float,
    threshold: float,
    freq_cluster_tol: float,
    freq_window: float,
    remove_mass_weighted_com: bool,
    gamma_only: bool,
    alpha: float, 
    weight_kind: str,
) -> ComparisonOutput:
    """Run DFT/ML band comparison and return results. build DFT cache once, then evaluate
        each MLIP band.yaml against it. This is the programmatic entry point
        used by the CLI and downstream scripts.

    Args:
        contcar_gs (str): Ground-state CONTCAR path.
        contcar_es (str): Excited-state CONTCAR path.
        band_dft_path (str): DFT band.yaml path.
        band_ml_paths (List[str]): ML band.yaml paths.
        q_tol (float): Q-point match tolerance.
        lattice_tol (float): Lattice mismatch tolerance.
        threshold (float): Cumulative weight threshold.
        freq_cluster_tol (float): Frequency cluster tolerance.
        freq_window (float): Frequency window for ML clustering.
        remove_mass_weighted_com (bool): Remove COM shift if True.
        gamma_only (bool): Only use Gamma if True.

    Returns:
        ComparisonOutput: Cached DFT data and ML comparison results.
    """

    # Build DFT cache once, then compare against each ML model
    cache = build_dft_cache(
        contcar_gs=contcar_gs,
        contcar_es=contcar_es,
        band_dft_path=band_dft_path,
        q_tol=q_tol,
        lattice_tol=lattice_tol,
        threshold=threshold,
        freq_cluster_tol=freq_cluster_tol,
        freq_window=freq_window,
        remove_mass_weighted_com=remove_mass_weighted_com,
        gamma_only=gamma_only,
        weight_type=weight_kind,
    )

    results: Dict[str, Dict[str, Any]] = {}
    for mlp in band_ml_paths:
        # Compute comparison for each ML band.yaml
        results[str(mlp)] = compare_one_ml(
            cache=cache,
            ml_path=mlp,
            q_tol=q_tol,
            threshold=threshold,
            freq_cluster_tol=freq_cluster_tol,
            freq_window=freq_window,
            alpha = alpha
        )

    return ComparisonOutput(dft_cache=cache, results_per_ml=results)
