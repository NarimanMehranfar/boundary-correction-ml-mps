# Feature description

This document describes the columns of the CSV files consumed by the pipeline
(`dataForTrainingAndValidation.csv`, `test_near_wall.csv`,
`test_not_near_wall.csv`) and maps them to the feature definitions in the paper
*"A Machine Learning-based Solid Boundary Treatment for Meshfree Particle
Methods."*

The network predicts the **boundary correction term** $\hat{B}_i$ for an MPS
operator $\mathcal{L}$ at a near-wall particle $i$. Following the paper, the
operator value is split as

$$
C_i = \langle \mathcal{L}\phi \rangle_i = A_i + B_i,
$$

where $A_i$ is the contribution of the surrounding fluid and wall particles and
$B_i$ is the contribution the ghost particles would have supplied. The model
learns $\hat{B}_i = \mathrm{NN}_\theta(\mathbf{X}_i)$, and the total field is
reconstructed as $A_i + \hat{B}_i$.

## Naming conventions

| Suffix in a column name | Meaning | Paper symbol |
|--------------------------|---------|--------------|
| `WG` | the *without-ghost* baseline — operator value from fluid + wall particles only | $A_i$ |
| `OG` | the *only-ghost* contribution — the boundary term (this is the **prediction target**) | $B_i$ |
| unsuffixed (`n0`, `laplacianScalar`, …) | the *total* field, with ghosts | $C_i = A_i + B_i$ |

So for every operator, `(unsuffixed) = WG + OG`, i.e. `total = baseline + boundary
contribution`. The inference script predicts the `OG` term and adds the `WG`
baseline back to recover the total field.

The nine nearest wall particles are indexed `(0)…(8)` in each stencil column
group, reordered to follow the boundary so they form a coherent 1-D sequence for
the CNN. Original (intentional) spellings such as `temprature`, `lamda`, and
`Neghbs` are kept as-is to match the data files.

## Notation

| Symbol | Meaning |
|--------|---------|
| $\mathbf{r}_i = (X, Y)$ | position of the target near-wall particle |
| $\mathbf{r}_k$ | position of the $k$-th nearest wall particle, $k \in \{0..8\}$ |
| $\mathbf{r}_{ik} = \mathbf{r}_k - \mathbf{r}_i$ | relative position (the `(dX)`/`(dY)` columns) |
| $\lVert \mathbf{r}_{ik}\rVert$ | distance to the wall neighbour |
| $\phi$, $\boldsymbol{\phi}$ | scalar field (temperature) / vector field (velocity) |
| $\phi_{ik} = \phi_k - \phi_i$ | field difference to the wall neighbour |
| $d_p$, $r_e$ | particle diameter, kernel support radius |
| $n_i$, $\lambda_i$ | particle number density, Laplacian normalisation coefficient |
| $\mathrm{BC}_k$ | boundary-condition type of wall neighbour $k$ (Dirichlet / Neumann) |

---

## 1. Common input columns (all operators)

These are present for every operator and make up the *geometric* and
*kernel/boundary* parts of $\mathbf{X}_i$.

### Scalar (target-particle) features

| Column | Description | Paper symbol |
|--------|-------------|--------------|
| `X`, `Y` | target particle position | $\mathbf{r}_i$ |
| `particleDiameter` | particle diameter | $d_p$ |
| `re` | kernel support radius | $r_e$ |
| `numberOfWallParticlesInNeghbs` | count of wall particles within the support of $i$ | $Nw_i$ |
| `lamda` | Laplacian normalisation coefficient | $\lambda_i$ |

### Per-neighbour geometric stencils (9 columns each, indices `(0)…(8)`)

| Column group | Description | Paper symbol |
|--------------|-------------|--------------|
| `wallParticleNeighbours(X)(k)` | wall-neighbour position, $x$ | $\mathbf{r}_k$ (x) |
| `wallParticleNeighbours(Y)(k)` | wall-neighbour position, $y$ | $\mathbf{r}_k$ (y) |
| `wallParticleNeighbours(dX)(k)` | relative position, $x$ | $\mathbf{r}_{ik}$ (x) |
| `wallParticleNeighbours(dY)(k)` | relative position, $y$ | $\mathbf{r}_{ik}$ (y) |
| `wallParticleNeighboursDistance(k)` | distance to wall neighbour | $\lVert \mathbf{r}_{ik}\rVert$ |
| `wallParticleNeighboursDistance(pow2)(k)` | squared distance | $\lVert \mathbf{r}_{ik}\rVert^2$ |

The `(dX)`, `(dY)`, and `Distance(pow2)` columns are computed during feature
engineering (`add_features`); the rest come directly from the CSV. The squared
distance is kept as an explicit input because the redundancy provides a useful
nonlinearity to the network, as noted in the paper.

---

## 2. Per-operator field-variable columns and targets

Each operator adds its own field-variable inputs, a `WG` baseline ($A_i$), and a
target output (`OG`, the boundary contribution $B_i$). Number density `n0`
depends on geometry only, so it carries **no** field-variable features — only
$A_i$ (its `WG` baseline), exactly as stated in the paper.

### `n0` — particle number density $n$

| Column | Role | Description |
|--------|------|-------------|
| `n0WG` | input | $A_i$: number density from fluid + wall particles |
| `n0OG` | **target** | $B_i$: boundary contribution to number density |
| `n0` | total | reconstructed $n0 = n0WG + n0OG$ |

### `gradientScalar` — scalar gradient $\nabla\phi$

| Column | Role | Description |
|--------|------|-------------|
| `n0` | input | particle number density $n_i$ |
| `temprature` | input | scalar field at target, $\phi_i$ |
| `gradientScalarWG(X)`, `gradientScalarWG(Y)` | input | $A_i$: fluid + wall gradient baseline |
| `wallParticleNeighboursBCTypes(k)` | input stencil | boundary-condition type $\mathrm{BC}_k$ |
| `wallParticleNeighboursTemprature(k)` | input stencil | wall-neighbour field $\phi_k$ |
| `wallParticleNeighboursTempratureDiff(k)` | input stencil | $\phi_{ik} = \phi_k - \phi_i$ |
| `wallParticleNeighboursTempratureEij(X)(k)` | input stencil | $\phi_{ik}\,\Delta x_{ik}$ (direction product, $x$) |
| `wallParticleNeighboursTempratureEij(Y)(k)` | input stencil | $\phi_{ik}\,\Delta y_{ik}$ (direction product, $y$) |
| `gradientScalarOG(X)`, `gradientScalarOG(Y)` | **target** | $B_i$: boundary contribution to the gradient |

### `laplacianScalar` — scalar Laplacian $\nabla^2\phi$

| Column | Role | Description |
|--------|------|-------------|
| `n0` | input | particle number density $n_i$ |
| `temprature` | input | scalar field at target, $\phi_i$ |
| `laplacianScalarWG` | input | $A_i$: fluid + wall Laplacian baseline |
| `wallParticleNeighboursBCTypes(k)` | input stencil | boundary-condition type $\mathrm{BC}_k$ |
| `wallParticleNeighboursTemprature(k)` | input stencil | wall-neighbour field $\phi_k$ |
| `wallParticleNeighboursTempratureDiff(k)` | input stencil | $\phi_{ik}$ |
| `wallParticleNeighboursTempratureEij(X)(k)` | input stencil | $\phi_{ik}\,\Delta x_{ik}$ |
| `wallParticleNeighboursTempratureEij(Y)(k)` | input stencil | $\phi_{ik}\,\Delta y_{ik}$ |
| `laplacianScalarOG` | **target** | $B_i$: boundary contribution to the Laplacian |

### `divergenceVector` — vector divergence $\nabla\cdot\boldsymbol{\phi}$

| Column | Role | Description |
|--------|------|-------------|
| `n0` | input | particle number density $n_i$ |
| `velocity(X)`, `velocity(Y)` | input | vector field at target, $\boldsymbol{\phi}_i$ |
| `divergenceVectorWG` | input | $A_i$: fluid + wall divergence baseline |
| `wallParticleNeighboursBCTypes(k)` | input stencil | boundary-condition type $\mathrm{BC}_k$ |
| `wallParticleNeighboursVelocity(X)(k)`, `…(Y)(k)` | input stencil | wall-neighbour field $\boldsymbol{\phi}_k$ |
| `wallParticleNeighboursVelocity(dX)(k)`, `…(dY)(k)` | input stencil | field difference $\boldsymbol{\phi}_{ik} = \boldsymbol{\phi}_k - \boldsymbol{\phi}_i$ |
| `wallParticleNeighboursVelocityEij(k)` | input stencil | directional product of $\boldsymbol{\phi}_{ik}$ and $\mathbf{r}_{ik}$ |
| `divergenceVectorOG` | **target** | $B_i$: boundary contribution to the divergence |

### `laplacianVector` — vector Laplacian $\nabla^2\boldsymbol{\phi}$

| Column | Role | Description |
|--------|------|-------------|
| `n0` | input | particle number density $n_i$ |
| `velocity(X)`, `velocity(Y)` | input | vector field at target, $\boldsymbol{\phi}_i$ |
| `laplacianVectorWG(X)`, `laplacianVectorWG(Y)` | input | $A_i$: fluid + wall vector-Laplacian baseline |
| `wallParticleNeighboursBCTypes(k)` | input stencil | boundary-condition type $\mathrm{BC}_k$ |
| `wallParticleNeighboursVelocity(X)(k)`, `…(Y)(k)` | input stencil | wall-neighbour field $\boldsymbol{\phi}_k$ |
| `wallParticleNeighboursVelocity(dX)(k)`, `…(dY)(k)` | input stencil | $\boldsymbol{\phi}_{ik}$ |
| `wallParticleNeighboursVelocityEij(k)` | input stencil | directional product of $\boldsymbol{\phi}_{ik}$ and $\mathbf{r}_{ik}$ |
| `laplacianVectorOG(X)`, `laplacianVectorOG(Y)` | **target** | $B_i$: boundary contribution to the vector Laplacian |

---

## 3. How the columns feed the network

The columns are split into two inputs (see the paper's architecture and the
README):

* **Wide vector $\mathbf{X}_i$** — all input columns above, flattened, fed to the
  MLP.
* **CNN stencils $\mathbf{X}_{w,i}$** — the per-neighbour `(0)…(8)` column groups,
  each forming one length-9 sequence (one "channel") for the CNN.

The number of features per wall neighbour ($f$) and the resulting input lengths
match the paper exactly:

| Operator | CNN stencil groups ($f$) | $\mathbf{X}_{w,i}$ length ($f \times 9$) | $\mathbf{X}_i$ length |
|----------|--------------------------|------------------------------------------|-----------------------|
| `n0` | 6 (geometry only) | 54 | 61 |
| `gradientScalar` | 11 | 99 | 109 |
| `laplacianScalar` | 11 | 99 | 108 |
| `divergenceVector` | 12 | 108 | 118 |
| `laplacianVector` | 12 | 108 | 119 |

The scalar operators add 5 temperature/BC stencil groups (`BCTypes`,
`Temprature`, `TempratureDiff`, `TempratureEij(X)`, `TempratureEij(Y)`) on top of
the 6 geometric groups; the vector operators add 6 velocity/BC groups
(`BCTypes`, `Velocity(X)`, `Velocity(Y)`, `Velocity(dX)`, `Velocity(dY)`,
`VelocityEij`); `n0` uses only the 6 geometric groups.

## 4. Other columns

* `*WithSumOfScalar*` (e.g. `gradientScalarWithSumOfScalarOG(X)`) — values for an
  alternative gradient summation form. When `psum = True`, these are copied into
  the corresponding `gradientScalar*` columns before feature engineering.
* `test_not_near_wall.csv` — background (non-near-wall) particles. These are not
  predicted; they only provide spatial context in the comparison plots.
