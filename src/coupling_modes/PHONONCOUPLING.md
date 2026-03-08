# Phonon mode--coupling comparison: DFT vs MLIPs (CBVN)


## Overview
This script compares the phonon coupling between a variety of instances of MLIPs to the gold standard DFT. for a single defect supercell.

Inputs:
- `CONTCAR_GS` and `CONTCAR_ES` (same atom ordering).
- One DFT `band.yaml` and a set of MLIP `band.yaml` files (Phonopy format).

Output: a text report that (i) selects DFT ``coupling'' modes from the GS$\to$ES displacement, (ii) scores each MLIP on several metrics, and (iii) produces a ranked list.

## What the script computes (the metrics)
**(1) GS$\to$ES mass-weighted displacement.**  
From the two CONTCARs we compute the minimum-image displacement in Cartesian coordinates, optionally remove the mass-weighted center-of-mass shift, and form the mass-weighted displacement vector:
$$
\Delta \mathbf r_i = \mathbf r_i^{\mathrm{ES}} - \mathbf r_i^{\mathrm{GS}},\qquad
\Delta \mathbf q = \bigl(\sqrt{m_1}\Delta \mathbf r_1, \ldots, \sqrt{m_N}\Delta \mathbf r_N\bigr)\in\mathbb{R}^{3N}.
$$

**(2) Coupling-mode weights from DFT eigenvectors.**  
Let $E(\mathbf q)\in\mathbb{C}^{3N\times 3N}$ be the DFT eigenvector matrix at a chosen $\mathbf q$ (columns normalized).  
Define per-mode amplitudes and projection power:
$$
a_m = \mathbf e_m^\dagger \Delta \mathbf q,\qquad
p_m = \frac{|a_m|^2}{\|\Delta \mathbf q\|^2}.
$$
The ``coupling modes'' are the smallest set of modes whose cumulative $p_m$ exceeds the threshold $\tau$.

For MLIP ranking we also define DFT per-mode weights
$$
w_m =
\begin{cases}
|a_m|^2, & \text{kind = `p`}\\
|\omega_m|\,|a_m|^2, & \text{kind = `S`}\\
|\omega_m|^2\,|a_m|^2, & \text{kind = `lambda`}
\end{cases}
$$
where $\omega_m$ is the DFT mode frequency. These weights decide which DFT modes ``matter'' most when scoring an MLIP.

**(3) GS$\to$ES score**  
For a matched DFT/MLIP $\mathbf q$-point, form the overlap matrix between eigenvector bases:
$$
O_{ij} = \bigl|\mathbf e_i^\dagger \tilde{\mathbf e}_j\bigr|^2\in[0,1].
$$
We compute the one-to-one assignment $\pi$ that maximizes $\sum_i O_{i,\pi(i)}$ (Hungarian algorithm).  
Using the DFT-defined weights $w_m$ (restricted to modes with $w_m>0$), we compute:
$$
E_{\mathrm{freq}} =
\sqrt{\frac{\sum_{m\in\mathcal V} w_m\,(\tilde\omega_{\pi(m)}-\omega_m)^2}{\sum_{m\in\mathcal V} w_m}},
\qquad
E_{\mathrm{vec}} =
\sqrt{\frac{\sum_{m\in\mathcal V} w_m\,(1-O_{m,\pi(m)})}{\sum_{m\in\mathcal V} w_m}},
$$
and the combined score
$$
\mathrm{Score} = E_{\mathrm{freq}} + \alpha\,E_{\mathrm{vec}}.
$$
Lower Score is better. I thought this would answer the question: `which MLIP best matches the DFT coupling modes eigenvalues and eigenvectors, and how well does it do that?`  
The different kinds of weights allow for different phrasings, for `kind=p` then the question is the above (where coupling modes refer to the vibrational modes containing the largest proportion of the mass weighted displacement vector).  
for `kind=lambda` the question is `How well does the MLIP reproduce the DFT modes that dominate the harmonic energetic cost of the GS to ES displacement?' and for `kind=S` the question is a mix. I need to confirm but I think it would be best suited for evaluating how accurate the PL spectra would be.

**(4) Secondary diagnostics (subspace agreement).**  
The report also prints:
- *AvgProjPowX*: compares the subspace spanned by the selected DFT coupling modes against the subspace spanned by the selected ML coupling modes, using principal angles. If $Q_d$ and $Q_m$ are orthonormal bases, with singular values $\sigma_\ell$ of $Q_d^\dagger Q_m$:
  $$
  X = \frac{1}{k_d}\sum_\ell \sigma_\ell^2,\quad (0\le X\le 1).
  $$
  I thought this score would answer the question: `How well do the strongly coupled phonons of <MLIP> match those of DFT?'
- *Coupling clusters*: clusters modes by frequency proximity (gap $>\Delta f_{\mathrm{cluster}}$ starts a new cluster), assigns each DFT cluster a weight (fraction of $\Delta\mathbf q$ captured), and checks whether ML frequencies in a window around each DFT cluster recover a similar subspace/weight. i thought this would answer the question `How well does the eigenspace of an <MLIP> frequency range match the displacements predicted by DFT in the same range?'


# Dictionary for `phonon_coupling_report_{i}.txt`
This is appended in pdf form as 'symbols.pdf'. Below i provide interpretation of the couplign_report.txt file. 

Below is a compact, ordered walkthrough of the most important report sections. Each section follows the same pattern: title, one-sentence essence, an example copied from `phonon_coupling_report_4.txt`, and a terse glossary of every key term that appears in that example.

**FINAL RANKING (lower dQ Score is better)**
Essence: A sorted scoreboard of models where smaller `Score_mean` means closer agreement with DFT coupling modes.
Example:
```text
FINAL RANKING (lower dQ Score is better)
----------------------------------------
rank  Score_mean    E_freq      E_vec       E_freq_rel   X_mean      model
   1    0.969945   0.430380   0.415050     0.048464   0.765116  orb-v3-direct-omat
   2    0.973523   0.271346   0.540136     0.026675   0.660771  pet-omat-m-v1.0.0
   3    0.975993   0.508419   0.359672     0.045925   0.553110  mace-matpes-r2scan-omat-ft
   4    0.999108   0.491352   0.390582     0.057103   0.538476  mace-omat-0-medium
   5    1.087501   0.631274   0.350944     0.069844   0.662776  mattersim-v1.0.0-1M
   6    1.087599   0.631273   0.351020     0.069844   0.662705  mattersim-v1.0.0-5M
   7    1.174055   0.628945   0.419315     0.065520   0.458567  small-omat-0
   8    1.421401   0.895533   0.404514     0.104516   0.553788  pet-omat-l-v1.0.0
   9    1.476293   0.833172   0.494708     0.043442   0.436489  TensorNetDGL-MatPES-r2SCAN-v2025.1-PES
  10    1.528758   0.974075   0.426679     0.093157   0.544434  mace-matpes-pbe-omat-ft
  11    1.774417   1.105344   0.514672     0.135703   0.289872  pet-omat-xl-v1.0.0
  12    1.802689   1.176082   0.482005     0.141564   0.468044  orb-v3-conservative-inf-omat
  13    1.812293   1.047904   0.587991     0.104396   0.235390  TensorNetDGL-MatPES-PBE-v2025.1-PES
  14    2.504070   1.792995   0.546980     0.213416   0.498711  pet-omad-s-v1.0.0
  15    5.862976   4.842523   0.784964     0.379993   0.425238  CHGNet-MatPES-PBE-2025.2.10-2.7M-PES
  16    5.925198   5.127700   0.613460     0.447990   0.317461  M3GNet-MatPES-r2SCAN-v2025.1-PES
  17    6.535556   5.732048   0.618083     0.356611   0.536645  CHGNet-MPtrj-2023.12.1-2.7M-PES
  18    6.778403   5.982676   0.612097     0.288561   0.326966  mace-mpa-0-medium
  19    7.415465   6.436802   0.752818     0.471850   0.134123  CHGNet-MPtrj-2024.2.13-11M-PES
  20    9.408333   8.550760   0.659671     0.782129   0.373738  pet-mad-s-v1.1.0
  21    9.596367   8.392233   0.926257     0.874040   0.306365  orb-d3-sm-v2
  22    9.936405   8.903370   0.794642     0.659815   0.559477  M3GNet-MP-2021.2.8-PES
  23   10.927271  10.038709   0.683509     0.632255   0.466293  CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES
  24   13.071580  12.047831   0.787500     0.742900   0.384124  pet-omad-l-v0.1.0
  25   13.872716  12.897500   0.750167     1.079009   0.456099  M3GNet-MP-2021.2.8-DIRECT-PES
```
Key terms:
- `rank`: ordering by best (lowest) `Score_mean`.
- `Score_mean`: Mean dQ score across q-points. $\mathrm{Score\_mean}=\langle \mathrm{Score}(\mathbf q)\rangle_{\mathbf q}$ with $\mathrm{Score}(\mathbf q)=E_{\mathrm{freq}}(\mathbf q)+\alpha\,E_{\mathrm{vec}}(\mathbf q)$.
- `E_freq`: Weighted RMS frequency error after Hungarian matching. $E_{\mathrm{freq}}(\mathbf q)=\sqrt{\sum_i \tilde w_i\left(\omega_{\pi(i)}^{\mathrm{ML}}-\omega_i^{\mathrm{DFT}}\right)^2}$, $\tilde w_i=w_i/\sum_j w_j$, $\pi$ = Hungarian assignment.
- `E_vec`: Weighted RMS eigenvector mismatch after Hungarian matching. $E_{\mathrm{vec}}(\mathbf q)=\sqrt{\sum_i \tilde w_i\left(1-O_{i,\pi(i)}\right)^2}$, $O_{ij}=|\mathbf e_i^{\mathrm{DFT}\,\dagger}\mathbf e_j^{\mathrm{ML}}|^2$.
- `E_freq_rel`: Weighted RMS relative frequency error. $E_{\mathrm{freq,rel}}(\mathbf q)=\sqrt{\sum_i \tilde w_i\left(\frac{\omega_{\pi(i)}^{\mathrm{ML}}-\omega_i^{\mathrm{DFT}}}{\max(|\omega_i^{\mathrm{DFT}}|,\varepsilon)}\right)^2}$.
- `X_mean`: Mean coupling-subspace agreement across q-points. $X(\mathbf q)=\frac{1}{k_{\mathrm{DFT}}}\sum_{\ell=1}^{\min(k_{\mathrm{DFT}},k_{\mathrm{ML}})}\sigma_\ell^2$ and $X_{\mathrm{mean}}=\langle X(\mathbf q)\rangle_{\mathbf q}$ with $\sigma_\ell=\mathrm{svd}(Q_{\mathrm{DFT}}^\dagger Q_{\mathrm{ML}})$.
- `model`: model identifier string.

**Model-dQ-AvgProjPowerX-couplingcluster-cid**
Essence: The per-model block that reports its dQ score, coupling-mode subspace overlap, and coupling-cluster statistics.
Example:
```text
Model: /home/rnpla/projects/mlip_phonons/results/small-omat-0/cbvn/raw/Plumipy_Files/band.yaml
  dQ score: mean=1.174055  min=1.174055  E_freq=0.628945  E_freq_rel=0.065520  E_vec=0.419315
  AvgProjPowX coupling modes subspace scores X: mean=0.458567  min=0.458567  max=0.458567
  Coupling cluster (eigenspaces) subspace scores: L1w_mean=0.244902  wθ_mean=5.690790  w(1-σ^2)_mean=0.041833
  Per-q dQ score (summary):
    q=[ 0.000000,  0.000000,  0.000000]  Score=1.174055  E_freq=0.628945  E_freq_rel=0.065520  E_vec=0.419315
  Per-q AvgProjPowX angles (summary):
    q=[ 0.000000,  0.000000,  0.000000]  X=0.458567  k_dft=13  k_ml=10  σ_min=0.011010  σ_mean=0.638864  θ_mean=37.992°  θ_max=89.369°
  Per-q coupling cluster stats:
    q=[ 0.000000,  0.000000,  0.000000]  L1w=0.244902  wθ=5.690790  w(1-σ^2)=0.041833
      cid  size  fmin        fmax        w_dft     w_ml      θ_mean    θ_max     θ_min
        4    35   6.550634   9.632476   0.540667   0.660824    2.425°   18.388°   0.582541
        5     3   10.942293   11.367780   0.264235   0.169903   10.154°   13.756°   0.564211
        6    16   12.418793   13.775959   0.027630   0.018534    6.351°   61.638°   0.889272
        7     7   15.656533   16.109659   0.029404   0.025030   18.243°   89.995°   1.000000
        9    49   18.030717   20.682009   0.034739   0.044923   21.927°   89.873°   0.999998
       16    17   31.176752   35.012994   0.023934   0.017175    9.325°   21.037°   0.595147
```
Key terms:
- `Model`: path to the MLIP `band.yaml` used for this block.
- `dQ score`: Combined frequency + eigenvector error. $\mathrm{Score}(\mathbf q)=E_{\mathrm{freq}}(\mathbf q)+\alpha\,E_{\mathrm{vec}}(\mathbf q)$, with `mean/min` taken across q-points.
- `mean`: average across q-points.
- `min`: minimum across q-points.
- `E_freq`: Weighted RMS frequency error after Hungarian matching. $E_{\mathrm{freq}}(\mathbf q)=\sqrt{\sum_i \tilde w_i\left(\omega_{\pi(i)}^{\mathrm{ML}}-\omega_i^{\mathrm{DFT}}\right)^2}$, with $\tilde w_i=w_i/\sum_j w_j$ and $\pi$ the Hungarian assignment.
- `E_freq_rel`: Weighted RMS relative frequency error. $E_{\mathrm{freq,rel}}(\mathbf q)=\sqrt{\sum_i \tilde w_i\left(\frac{\omega_{\pi(i)}^{\mathrm{ML}}-\omega_i^{\mathrm{DFT}}}{\max(|\omega_i^{\mathrm{DFT}}|,\varepsilon)}\right)^2}$ (dimensionless).
- `E_vec`: Weighted RMS eigenvector mismatch after Hungarian matching. $E_{\mathrm{vec}}(\mathbf q)=\sqrt{\sum_i \tilde w_i\left(1-O_{i,\pi(i)}\right)^2}$, $O_{ij}=|\mathbf e_i^{\mathrm{DFT}\,\dagger}\mathbf e_j^{\mathrm{ML}}|^2$.
- `AvgProjPowX`: Coupling-subspace agreement at each q-point. $X(\mathbf q)=\frac{1}{k_{\mathrm{DFT}}}\sum_{\ell=1}^{\min(k_{\mathrm{DFT}},k_{\mathrm{ML}})}\sigma_\ell^2$ (mean/min/max over q are reported).
- `coupling modes subspace scores`: summary of `X` across q-points.
- `X`: Coupling-subspace agreement score. $X(\mathbf q)$ as above, with $\sigma_\ell=\mathrm{svd}(Q_{\mathrm{DFT}}^\dagger Q_{\mathrm{ML}})$ and $Q_{\mathrm{DFT}}$, $Q_{\mathrm{ML}}$ orthonormal coupling-mode bases.
- `angles`: Principal angles between coupling subspaces. $\theta_\ell=\arccos(\sigma_\ell)$, summarized as `θ_*`.
- `max`: maximum across q-points.
- `Coupling cluster (eigenspaces) subspace scores`: summary of cluster-level subspace agreement.
- `L1w_mean`: Average cluster-weight mismatch. $\langle \sum_{C\in\mathcal R_\tau(\mathbf q)}|w_C^{\mathrm{ML}}-w_C^{\mathrm{DFT}}|\rangle_{\mathbf q}$, $\mathcal R_\tau$ = DFT-important clusters.
- `wθ_mean`: Average DFT-weighted cluster misalignment angle. $\langle \sum_{C\in\mathcal R_\tau(\mathbf q)} w_C^{\mathrm{DFT}}\,\overline{\theta}_C\rangle_{\mathbf q}$, $\overline{\theta}_C$ = mean principal angle in cluster $C$ (deg).
- `w(1-σ^2)_mean`: Average DFT-weighted subspace mismatch. $\langle \sum_{C\in\mathcal R_\tau(\mathbf q)} w_C^{\mathrm{DFT}}\left(1-\sigma_C^2\right)\rangle_{\mathbf q}$, $\sigma_C^2$ = mean squared singular value for cluster $C$.
- `Per-q`: values for each individual q-point.
- `q=[ ... ]`: the q-point vector used.
- `Score`: Per-q dQ score at the listed q-point. $\mathrm{Score}(\mathbf q)=E_{\mathrm{freq}}(\mathbf q)+\alpha\,E_{\mathrm{vec}}(\mathbf q)$.
- `k_dft`: dimension of the DFT coupling-mode subspace.
- `k_ml`: dimension of the ML coupling-mode subspace.
- `σ_min`: Smallest singular value between coupling subspaces. $\min_\ell \sigma_\ell$ for $Q_{\mathrm{DFT}}^\dagger Q_{\mathrm{ML}}$.
- `σ_mean`: Mean singular value between coupling subspaces. $\langle \sigma_\ell\rangle_\ell$ for $Q_{\mathrm{DFT}}^\dagger Q_{\mathrm{ML}}$.
- `θ_mean`: Mean principal angle between coupling subspaces. $\langle \theta_\ell\rangle_\ell$.
- `θ_max`: Largest principal angle between coupling subspaces. $\max_\ell \theta_\ell$.
- `L1w`: Per-q cluster-weight mismatch. $\sum_{C\in\mathcal R_\tau(\mathbf q)}|w_C^{\mathrm{ML}}-w_C^{\mathrm{DFT}}|$.
- `wθ`: Per-q DFT-weighted cluster misalignment angle. $\sum_{C\in\mathcal R_\tau(\mathbf q)} w_C^{\mathrm{DFT}}\,\overline{\theta}_C$.
- `w(1-σ^2)`: Per-q DFT-weighted subspace mismatch. $\sum_{C\in\mathcal R_\tau(\mathbf q)} w_C^{\mathrm{DFT}}\left(1-\sigma_C^2\right)$.
- `cid`: cluster ID.
- `size`: number of modes in the cluster.
- `fmin`: minimum frequency in the cluster.
- `fmax`: maximum frequency in the cluster.
- `w_dft`: DFT weight assigned to the cluster.
- `w_ml`: MLIP weight assigned to the cluster.
- `θ_min`: minimum principal angle for the cluster subspace.

**Mode–coupling comparison report**
Essence: The run header that records the inputs and global settings for this report.
Example:
```text
Mode–coupling comparison report
-------------------------------
DFT: /home/rnpla/projects/mlip_phonons/test/CBVN/band.yaml
N atoms: 97    N modes: 291
Settings: threshold=0.900, freq_cluster_tol=0.500, freq_window=0.500, alpha=1.300, weight_kind=S
q-points used: 1
  q[0] index=0  pos=[ 0.000000,  0.000000,  0.000000]
```
Key terms:
- `DFT`: the reference `band.yaml` path.
- `N atoms`: atom count in the supercell.
- `N modes`: total phonon modes (`3N`).
- `Settings`: run configuration summary line.
- `threshold`: cumulative `p` cutoff used to select coupling modes.
- `freq_cluster_tol`: frequency gap that starts a new cluster.
- `freq_window`: window around a DFT cluster used for ML comparisons.
- `alpha`: weight on eigenvector mismatch in the dQ score.
- `weight_kind`: which DFT weighting scheme is used (`p`, `S`, or `lambda`).
- `q-points used`: number of q-points included in the report.
- `q[0]`: q-point index label.
- `index`: the q-point index inside the `band.yaml`.
- `pos`: the q-point coordinates.

**DFT Coupling modes (per mode)**
Essence: Lists the individual DFT modes that capture the largest share of the GS→ES displacement.
Example:
```text
DFT Coupling modes  (per mode) 
-------------------------------
q[0] index=0  sum(p)=1.000000
  selected modes: k=13  cumsum_last=0.906262
  Coupling modes: mode | freq | p):
      31   7.079503   5.384810e-01
      58   11.367780   1.906097e-01
      57   11.125993   7.362567e-02
      75   15.656533   2.673579e-02
      84   17.151593   1.270706e-02
     177   29.123638   9.271332e-03
     111   19.452359   8.767571e-03
      82   16.753087   8.660799e-03
     179   30.224084   8.554050e-03
     195   33.687570   7.468104e-03
```
Key terms:
- `q[0]`: q-point index label.
- `index`: the q-point index inside the `band.yaml`.
- `sum(p)`: total projection power summed over all modes (should be ~1).
- `selected modes`: the subset kept after thresholding.
- `k`: number of selected modes.
- `cumsum_last`: cumulative `p` of the selected set (≥ threshold).
- `Coupling modes`: the per-mode table header.
- `mode`: mode index.
- `freq`: mode frequency.
- `p`: projection power of the mode (fraction of |Δq|^2).

**DFT Coupling clusters (eigenspace)**
Essence: Groups DFT modes into frequency clusters and reports each cluster's DFT weight.
Example:
```text
DFT Coupling clusters (eigenspace)
----------------------------------
q[0] index=0  pos=[ 0.000000,  0.000000,  0.000000]
  cid  size  fmin        fmax        w_dft
    4    35   6.550634   9.632476   0.540667
    5     3   10.942293   11.367780   0.264235
    9    49   18.030717   20.682009   0.034739
    7     7   15.656533   16.109659   0.029404
    6    16   12.418793   13.775959   0.027630
   16    17   31.176752   35.012994   0.023934
```
Key terms:
- `q[0]`: q-point index label.
- `index`: the q-point index inside the `band.yaml`.
- `pos`: the q-point coordinates.
- `cid`: cluster ID. (after clustering frequencies)
- `size`: number of modes in the cluster.
- `fmin`: minimum cluster frequency.
- `fmax`: maximum cluster frequency.
- `w_dft`: DFT weight assigned to the cluster. (approximately how much of the displacement that that cluster of modes explains)


## Scores used in the report

- **Singular values $\sigma_i$ (report fields: $\sigma_{\min}$, $\sigma_{\mathrm{mean}}$)**  
  Given two subspaces with orthonormal bases $Q_A,Q_B$,
  $$
  C=Q_A\Herm Q_B,\qquad \{\sigma_i\}=\mathrm{svd}(C),\qquad 0\le\sigma_i\le 1.
  $$
  Interpretation: $\sigma_i\approx 1$ means aligned; $\sigma_i\approx 0$ means orthogonal.

- **Principal angles $\theta_i$ (report fields: $\theta_{\mathrm{mean}}$, $\theta_{\max}$)**  
  $$
  \theta_i=\arccos(\sigma_i)\times\frac{180}{\pi}\quad\text{(degrees)}.
  $$
  Interpretation: $0^\circ$ is aligned; $90^\circ$ is orthogonal.

- **$k_{\mathrm{DFT}}$ (`k_dft`)**  
  $k_{\mathrm{DFT}}=\dim\big(\mathrm{span}\{\mathbf e_m^{\mathrm{DFT}}:m\in S_\tau^{\mathrm{DFT}}\}\big)$.  
  Interpretation: dimension of the DFT coupling-mode subspace.

- **$k_{\mathrm{ML}}$ (`k_ml`)**  
  analogous dimension for the MLIP coupling-mode subspace.  
  Interpretation: if $k_{\mathrm{ML}}<k_{\mathrm{DFT}}$, the  subspace score \(X\) is capped.

- **$X$ (`AvgProjPowX`, printed as `X`)**  
  For coupling-mode subspaces $Q_{\mathrm{DFT}}$ and $Q_{\mathrm{ML}}$,
  $$
  X(\mathbf q)=\frac{1}{k_{\mathrm{DFT}}}\sum_{i=1}^{\min(k_{\mathrm{DFT}},k_{\mathrm{ML}})}\sigma_i^2\in[0,1].
  $$
  Interpretation: coupling-subspace agreement; closer to $1$ is better.

- **`X: mean/min/max`**  
  For multiple q-points: `mean`$=\langle X(\mathbf q)\rangle_{\mathbf q}$ and `min/max` are extrema across analyzed $\mathbf q$.  
  Interpretation: variability of subspace agreement over q (with $\Gamma$-only, these often coincide).

- **$X_{\mathrm{mean}}$ (`X_mean`)**  
  $X_{\mathrm{mean}}=\langle X(\mathbf q)\rangle_{\mathbf q}$.

- **`dQ score: mean/min`**  
  In the per-model header, (see below for `Score()`)
  $$
  \texttt{mean}=\langle \mathrm{Score}(\mathbf q)\rangle_{\mathbf q}=\mathrm{Score\_mean},
  \qquad
  \texttt{min}=\min_{\mathbf q}\mathrm{Score}(\mathbf q).
  $$

- **`L1w` and `L1w_mean`**
  $$
  \mathrm{L1w}(\mathbf q)=\sum_{C\in\mathcal R_\tau(\mathbf q)}\big|w_C^{\mathrm{ML}}(\mathbf q)-w_C^{\mathrm{DFT}}(\mathbf q)\big|.
  $$
  Interpretation: cluster-weight mismatch in DFT-important spectral regions; $0$ is perfect.

- **`L1w_mean`**  
  $\mathrm{L1w\_mean}=\langle \mathrm{L1w}(\mathbf q)\rangle_{\mathbf q}$.  
  Interpretation: average L1 cluster-weight mismatch across analyzed q-points.

- **`wtheta_mean`**  
  $\mathrm{w\theta\_mean}=\langle w\theta(\mathbf q)\rangle_{\mathbf q}$.  
  Interpretation: average DFT-weighted cluster misalignment angle (degrees).

- **`w(1-sigma2)_mean`**  
  $\langle w(1-\sigma^2)(\mathbf q)\rangle_{\mathbf q}$.  
  Interpretation: average cluster-level $\sigma^2$ misalignment score across q-points.

- **`wtheta` and `wtheta_mean`**  
  For each relevant cluster $C$, compute principal angles between $Q_C^{\mathrm{DFT}}$ and $Q_{W(C)}^{\mathrm{ML}}$ and take the mean angle $\overline{\theta}_C(\mathbf q)$.  
  Then
  $$
  w\theta(\mathbf q)=\sum_{C\in\mathcal R_\tau(\mathbf q)} w_C^{\mathrm{DFT}}(\mathbf q)\,\overline{\theta}_C(\mathbf q)\quad (\text{degrees}).
  $$
  Interpretation: DFT-weighted subspace misalignment inside the important frequency windows; smaller is better.

- **`w(1-sigma2)` and `w(1-sigma2)_mean`**  
  For each relevant cluster $C$, define
  $$
  \sigma_C^2(\mathbf q)=\frac{1}{\dim(Q_C^{\mathrm{DFT}})}\sum_i \sigma_i(C;\mathbf q)^2,\qquad
  w(1-\sigma^2)(\mathbf q)=\sum_{C\in\mathcal R_\tau(\mathbf q)} w_C^{\mathrm{DFT}}(\mathbf q)\big(1-\sigma_C^2(\mathbf q)\big).
  $$
  Interpretation: another cluster-level misalignment score; $0$ is best.

- **$O_{ij}$ (used inside `E_vec`)**
  $$
  O_{ij}(\mathbf q)=\left|\mathbf e_i^{\mathrm{DFT}}(\mathbf q)\Herm\mathbf e_j^{\mathrm{ML}}(\mathbf q)\right|^2\in[0,1].
  $$
  Interpretation: phase/sign-invariant similarity between individual modes.

- **$\pi$ (Hungarian assignment, used inside `E_freq`, `E_vec`)**
  $$
  \pi=\arg\max_{\text{one-to-one maps}}\sum_{i=1}^{3N} O_{i,\pi(i)}.
  $$
  Interpretation: matches MLIP modes to DFT modes without trusting mode indices (or frequencies)

- **`weight_kind`**  
  Selects nonnegative DFT importance weights $w_i$ for the targeted (mode-by-mode) RMS scores.
  $$
  w_i \propto |a_i|^2
  $$
  Interpretation: emphasizes modes that contribute to the GS$\to$ES displacement, or, depending on kind, emphasises higher frequency modes.

- **$\alpha$ (`alpha`)**  
  Tradeoff scalar in the final targeted score.  
  Interpretation: larger $\alpha$ penalizes eigenvector mismatch more strongly relative to frequency error. I settled for around 1.3.

- **`E_freq`**  
  With normalized weights $\tilde w_i=w_i/\sum_j w_j$ and assigned partner $\pi(i)$,
  $$
  E_{\mathrm{freq}}(\mathbf q)=\sqrt{\sum_i \tilde w_i\left(\omega_{\pi(i)}^{\mathrm{ML}}(\mathbf q)-\omega_i^{\mathrm{DFT}}(\mathbf q)\right)^2}.
  $$
  Interpretation: weighted RMS frequency error on GS$\to$ES- DFT coupling modes; $0$ is best.

- **`E_freq_rel`**
  $$
  E_{\mathrm{freq,rel}}(\mathbf q)=\sqrt{\sum_i \tilde w_i\left(\frac{\omega_{\pi(i)}^{\mathrm{ML}}-\omega_i^{\mathrm{DFT}}}{\max(|\omega_i^{\mathrm{DFT}}|,\varepsilon)}\right)^2}.
  $$
  Interpretation: weighted RMS *relative* frequency error (dimensionless).

- **`E_vec`**  
  Define $d_i(\mathbf q)=1-O_{i,\pi(i)}(\mathbf q)\in[0,1]$.  
  Then
  $$
  E_{\mathrm{vec}}(\mathbf q)=\sqrt{\sum_i \tilde w_i\,d_i(\mathbf q)^2}.
  $$
  Interpretation: weighted RMS displacement-pattern mismatch; $0$ is best.

- **`Score` and `Score_mean`**
  $$
  \mathrm{Score}(\mathbf q)=E_{\mathrm{freq}}(\mathbf q)+\alpha\,E_{\mathrm{vec}}(\mathbf q),
  \qquad
  \mathrm{Score\_mean}=\langle \mathrm{Score}(\mathbf q)\rangle_{\mathbf q}.
  $$
  Interpretation: used for ranking MLIPs (smaller is better).

- **`q-points used`**  
  $n_q=\text{number of q-points included in the report}$.  
  Interpretation: with `Gamma_only`=`True`, $n_q=1$, so ``mean/min/max'' coincide.
