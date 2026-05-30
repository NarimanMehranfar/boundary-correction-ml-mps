# Dataset feature description

Each row is one fluid particle near a wall in a 2D SPH (smoothed-particle
hydrodynamics) simulation. The neural network learns the **ghost-particle
contribution** (the `*OG` fields) to a differential operator, which is then
added to the with-ghost baseline (`*WG`) to reconstruct the total field.

Three CSVs share the same 146-column schema:

| file | contents |
|------|----------|
| `dataForTrainingAndValidation.csv` | training + validation particles (you supply this) |
| `test_near_wall.csv`               | test particles **near** the wall (the ones predicted/compared) |
| `test_not_near_wall.csv`           | test particles **away** from the wall (used only as background points in the spatial plots) |

Every neighbour quantity is stored as 9 columns indexed `(0)`..`(8)` — the 9
wall-particle neighbours used by the CNN stencils.

## Particle state

| column | meaning |
|--------|---------|
| `X`, `Y`, `Z` | particle position |
| `particleTypes` | particle type id |
| `BCTypeTemprature`, `BCTypePressure`, `BCTypeVelocity` | boundary-condition type flags |
| `temprature` | scalar field value (temperature) at the particle |
| `pressure` | pressure at the particle |
| `velocity(X)`, `velocity(Y)`, `velocity(Z)` | velocity components |
| `particleDiameter` | particle diameter (sets the plot margin / point size) |
| `n`, `lamda` | particle number density and the MPS/SPH lambda parameter |
| `re` | kernel interaction radius |
| `numberOfWallParticlesInNeghbs` | number of wall particles among the neighbours |
| `nearWall` | flag: is the particle near a wall |

## Operator fields

For each differential operator there are three variants:

* **(no suffix)** — the *full / ground-truth* operator (all particles).
* **`WG`** — the *with-ghost* baseline (computed with ghost particles present).
* **`OG`** — the *ghost-only contribution* = full − WG. This is what the network predicts.

The total field is reconstructed at inference time as `OG_predicted + WG`.

| operator | scalar/vector | columns |
|----------|---------------|---------|
| `n0`               | scalar | `n0`, `n0WG`, `n0OG` |
| `lamda`            | scalar | `lamda`, `lamdaWG`, `lamdaOG` |
| `laplacianScalar`  | scalar | `laplacianScalar`, `laplacianScalarWG`, `laplacianScalarOG` |
| `divergenceVector` | scalar | `divergenceVector`, `divergenceVectorWG`, `divergenceVectorOG` |
| `gradientScalar`   | vector | `gradientScalar(X/Y/Z)`, `gradientScalarWG(X/Y/Z)`, `gradientScalarOG(X/Y/Z)` |
| `laplacianVector`  | vector | `laplacianVector(X/Y/Z)`, `laplacianVectorWG(X/Y/Z)`, `laplacianVectorOG(X/Y/Z)` |

There are also `gradientScalarWithScalarHat*` and `gradientScalarWithSumOfScalar*`
variants. When `psum = True` the scripts copy the `WithSumOfScalar` columns onto
the plain `gradientScalar*` names (this matches the original pipeline).

## Neighbour stencils (9 columns each, indices `(0)`..`(8)`)

| base name | meaning |
|-----------|---------|
| `wallParticleNeighbours(X)`, `(Y)`, `(Z)` | neighbour positions |
| `wallParticleNeighboursDistance` | neighbour distance from the particle |
| `wallParticleNeighboursBCTypes` | neighbour boundary-condition type |
| `wallParticleNeighboursTemprature` | neighbour scalar field value |
| `wallParticleNeighboursVelocity(X)`, `(Y)`, `(Z)` | neighbour velocity components |

## Features derived by the scripts (not in the CSV)

`common_pipeline.add_features` builds these per neighbour `i` before training and
inference, so the raw CSVs do **not** need to contain them:

* `wallParticleNeighbours(dX)(i)`, `(dY)(i)` — neighbour offset from the particle.
* `wallParticleNeighboursDistance(pow2)(i)` — squared distance.
* `wallParticleNeighboursTempratureDiff(i)`, `...Eij(X)(i)`, `...Eij(Y)(i)` — temperature differences and their products with the offsets (non-`n0` models).
* `wallParticleNeighboursVelocity(dX)(i)`, `(dY)(i)`, `...VelocityEij(i)` — velocity differences and the velocity Eij term (non-`n0` models).
* `OG(i)`, `ALL(i)` — distance-weighted magnitudes used only for the train/test range filter.
