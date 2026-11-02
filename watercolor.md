# Watercolour simulation: research notes

Working notes for the simulator in `watercolour-sketchpad.html`. The aim is to
replace guesswork in the step shader with models from the literature, in
particular to get fast, directional wet-in-wet bleeding, which the current
diffusion-only model cannot produce.

Last updated: 2026-11-03.

## 1. Sources and how much of each was actually read

| Source | Status | Value for us |
|---|---|---|
| Curtis, Anderson, Seims, Fleischer, Salesin. *Computer-Generated Watercolor*. SIGGRAPH 1997. | **Read in full** (PDF text extracted and method pages checked as images). | The reference model. Section 2 below is a faithful transcription. |
| Chu, Tai. *MoXi: Real-Time Ink Dispersion in Absorbent Paper*. SIGGRAPH 2005. | Only the 1-page implementation sketch was readable. The full paper is behind a TLS-broken project page and an ACM paywall. | Section 3 is the sketch plus **my recollection** of the main paper, marked as such. |
| Van Laerhoven, Van Reeth. *Real-time simulation of watery paint*. CAVW 2005. | **Not read.** Closed access; the one open mirror returned an empty body. | Known to be a GPU port of Curtis. Nothing below comes from it. |
| Lei, Chang. *Real-Time Rendering of Watercolor Effects for Virtual Environments*. PCM 2004. | Read in full. | Post-process only (Sobel edges + paper texture). Not a fluid model. Not useful to us beyond rendering tricks. |
| Sun et al. *Fluid computational model for mineral and vegetal pigments diffusing on Xuan paper*. i3D 2018 poster. | Read; equations did not survive extraction. | Fick's-law diffusion plus advection. Nothing we don't already do. |
| Deegan et al. *Contact line deposits in an evaporating drop* (the "coffee ring" paper), cited by Curtis. | Not read. | Physical basis for edge darkening: evaporation at a pinned edge pulls fluid outward. |

PDF copies of the read papers are in this session's tool-results folder, not in the repo.

## 2. The Curtis 1997 model (transcribed)

Curtis is explicit that it is "empirically-based", not a strict physical
simulation. The goal is salient effects, predictable and controllable. It was an
offline simulator: one glaze at 640x480 took minutes to hours in 1997, and the
interactive painter only painted *initial conditions* and then ran the sim.

### 2.1 Layers and quantities

Three layers over a 2D grid:

1. **Shallow-water layer**: water and pigment flow above the paper.
   - `M` wet-area mask, 1 wet / 0 dry. Flow is confined to it.
   - `u, v` water velocity (stored on a **staggered grid**: velocities on cell faces, everything else at cell centres).
   - `p` water pressure.
   - `g_k` concentration of pigment k in the water.
   - `grad h` slope of the paper height field.
   - Medium constants: viscosity `mu = 0.1`, viscous drag `kappa = 0.01` (all their examples).
2. **Pigment-deposition layer**: `d_k`, pigment adsorbed onto the paper. Per pigment: density `rho`, staining power `omega`, granularity `gamma`.
3. **Capillary layer**: water inside the paper pores. `s` saturation, `c` capacity. Only used for backruns.

Paper: height field `h` in (0,1) from Perlin/Worley-type noise.
Capacity `c = h * (c_max - c_min) + c_min`.

### 2.2 Main loop

```
for each time step:
    MoveWater(M, u, v, p)
    MovePigment(M, u, v, g_1..g_n)
    TransferPigment(g_1..g_n, d_1..d_n)
    SimulateCapillaryFlow(M, s)
```

### 2.3 MoveWater = UpdateVelocities, RelaxDivergence, FlowOutward

Design conditions they list: (1) water stays inside the wet mask, (2) surplus
water flows outward, (3) damped to avoid waves, (4) perturbed by paper texture
to give streaks along the flow, (5) local changes have global effect, (6)
outward flow toward edges for edge darkening.

Shallow-water equations (1),(2):

```
du/dt = -( d(u^2)/dx + d(uv)/dy ) + mu * laplacian(u) - dp/dx
dv/dt = -( d(v^2)/dy + d(uv)/dx ) + mu * laplacian(v) - dp/dy
```

**UpdateVelocities** (Euler, adaptive sub-steps so no velocity exceeds one cell per sub-step):

```
(u, v) <- (u, v) - grad h                      # paper slope nudges the flow
dt <- 1 / ceil( max_ij { |u|, |v| } )
for t = 0 to 1 by dt:
    for all cells (i,j):
        A <- u[i,j]^2 - u[i+1,j]^2 + (uv)[i+.5,j-.5] - (uv)[i+.5,j+.5]
        B <- u[i+1.5,j] + u[i-.5,j] + u[i+.5,j+1] + u[i+.5,j-1] - 4 u[i+.5,j]
        u'[i+.5,j] <- u[i+.5,j] + dt * ( A - mu*B + p[i,j] - p[i+1,j] - kappa*u[i+.5,j] )
        A <- v[i,j]^2 - v[i,j+1]^2 + (uv)[i-.5,j+.5] - (uv)[i+.5,j+.5]
        B <- v[i+1,j+.5] + v[i-1,j+.5] + v[i,j+1.5] + v[i,j-.5] - 4 v[i,j+.5]
        v'[i,j+.5] <- v[i,j+.5] + dt * ( A - mu*B + p[i,j] - p[i,j+1] - kappa*v[i,j+.5] )
    (u, v) <- (u', v')
    EnforceBoundaryConditions(M, u, v)     # velocity on any face of a dry cell = 0
```

Note: `A - mu*B` as printed. With `B` being the 5-point Laplacian the viscous
term has the sign of a diffusion only if `mu` is negative or the sign is a typo;
treat it as `+ mu * laplacian` per equations (1)-(2).

**RelaxDivergence** (Foster-style pressure relaxation; N = 50, tau = 0.01, xi = 0.1):

```
repeat up to N times:
    (u', v') <- (u, v); dmax <- 0
    for all cells:
        delta <- xi * ( u[i+.5,j] - u[i-.5,j] + v[i,j+.5] - v[i,j-.5] )   # xi * divergence
        p[i,j] <- p[i,j] + delta
        u'[i+.5,j] <- u'[i+.5,j] + delta
        u'[i-.5,j] <- u'[i-.5,j] - delta
        v'[i,j+.5] <- v'[i,j+.5] + delta
        v'[i,j-.5] <- v'[i,j-.5] - delta
        dmax <- max(|delta|, dmax)
    (u, v) <- (u', v')
until dmax <= tau
```

This is what makes "adding water here" push water everywhere (condition 5) and
is the expensive part.

**FlowOutward** (edge darkening). Gaussian-blur the mask with a K x K kernel to
get `M'`, then lower pressure near the edge:

```
p <- p - eta * (1 - M') * M        # K = 10, eta in [0.01, 0.05]
```

Lower pressure at the rim pulls water, and therefore pigment, from the interior
toward the edge as it dries. Deegan's pinned-contact-line mechanism.

### 2.4 MovePigment (advection by the water velocity)

Upwind, mass-conserving, with the same sub-stepping so nothing moves more than one cell:

```
dt <- 1 / ceil( max_ij { |u|, |v| } )
for each pigment k, for t = 0 to 1 by dt:
    g' <- g <- g_k
    for all cells:
        g'[i+1,j] += max(0,  u[i+.5,j] * g[i,j])
        g'[i-1,j] += max(0, -u[i-.5,j] * g[i,j])
        g'[i,j+1] += max(0,  v[i,j+.5] * g[i,j])
        g'[i,j-1] += max(0, -v[i,j-.5] * g[i,j])
        g'[i,j]   -= (sum of the four amounts above)
    g_k <- g'
```

There is **no pigment diffusion term** in Curtis. All pigment movement is advection.

### 2.5 TransferPigment (adsorption / desorption)

Per pigment k, for wet cells only:

```
down <- g[i,j] * (1 - h[i,j]*gamma) * rho            # water -> paper
up   <- d[i,j] * (1 + (h[i,j]-1)*gamma) * rho / omega  # paper -> water
clamp so d + down <= 1 and g + up <= 1
d += down - up
g += up - down
```

So: high `rho` (dense pigment) settles fast; high `omega` (staining) resists
lifting; `gamma` makes settling favour paper valleys (low h) and lifting favour
peaks, which is granulation. Pigment values per Curtis Figure 5:

| Pigment | K rgb | S rgb | rho | omega | gamma |
|---|---|---|---|---|---|
| Quinacridone Rose | 0.22 1.47 0.57 | 0.05 0.003 0.03 | 0.02 | 5.5 | 0.81 |
| Indian Red | 0.46 1.07 1.50 | 1.28 0.38 0.21 | 0.05 | 7.0 | 0.40 |
| Cadmium Yellow | 0.10 0.36 3.45 | 0.97 0.65 0.007 | 0.05 | 3.4 | 0.81 |
| Hookers Green | 1.62 0.61 1.64 | 0.01 0.012 0.003 | 0.09 | 1.0 | 0.41 |
| Cerulean Blue | 1.52 0.32 0.25 | 0.06 0.26 0.40 | 0.01 | 1.0 | 0.31 |
| Burnt Umber | 0.74 1.54 2.10 | 0.09 0.09 0.004 | 0.09 | 9.3 | 0.90 |
| Cadmium Red | 0.14 1.08 1.68 | 0.77 0.015 0.018 | 0.02 | 1.0 | 0.63 |
| Brilliant Orange | 0.13 0.81 3.45 | 0.005 0.009 0.007 | 0.01 | 1.0 | 0.14 |
| Hansa Yellow | 0.06 0.21 1.78 | 0.50 0.88 0.009 | 0.06 | 1.0 | 0.08 |
| Phthalo Green | 1.55 0.47 0.63 | 0.01 0.05 0.035 | 0.02 | 1.0 | 0.12 |
| French Ultramarine | 0.86 0.86 0.06 | 0.005 0.005 0.09 | 0.01 | 3.1 | 0.91 |
| Interference Lilac | 0.08 0.11 0.07 | 1.25 0.42 1.43 | 0.06 | 1.0 | 0.08 |

### 2.6 SimulateCapillaryFlow (backruns, wet mask growth)

Water soaks from the shallow layer into the pores at rate `alpha`, diffuses
cell to cell until neighbours are full, and any cell whose saturation passes
`sigma` joins the wet mask. Thresholds: `epsilon` minimum saturation to give,
`delta` minimum saturation to receive.

```
for all cells: if M > 0: s += max(0, min(alpha, c - s))
s' <- s
for all cells (i,j), each 4-neighbour (k,l):
    if s[i,j] > epsilon and s[i,j] > s[k,l] and s[k,l] > delta:
        ds <- max(0, min(s[i,j] - s[k,l], c[k,l] - s[k,l]) / 4)
        s'[i,j] -= ds;  s'[k,l] += ds
s <- s'
for all cells: if s > sigma: M <- 1
```

Capacity varies with paper height, so the mask grows with a ragged, branching
front. Curtis treats backruns (capillary, damp) and wet-in-wet flow (momentum,
wet) as two separate regimes and names unifying them by wetness as future work.
Our current code already has that unification in spirit (damp vs pooled cells).

### 2.7 Dry brush

Exclude from the wet mask any cell whose paper height is below a threshold.
(Ours does the same via `dryMask` in the brush shader.)

### 2.8 Rendering: Kubelka-Munk

Each pigment has absorption `K` and scattering `S` per RGB channel. Derived
from two user colours, reflectance over white `Rw` and over black `Rb`, with
`0 < Rb < Rw < 1`:

```
a = 0.5 * ( Rw + (Rb - Rw + 1) / Rb )
b = sqrt(a^2 - 1)
S = (1/b) * arccoth( (b^2 - (a - Rw)(a - 1)) / (b (1 - Rw)) )
K = S (a - 1)
```

Layer of thickness x (x = g + d summed over pigments, S and K thickness-weighted):

```
c = a sinh(bSx) + b cosh(bSx)
R = sinh(bSx) / c
T = b / c
```

Composite layer 1 over layer 2:

```
R = R1 + T1^2 R2 / (1 - R1 R2)
T = T1 T2 / (1 - R1 R2)
```

Opaque paints: S high in own hue; transparent: S low everywhere, K high in
complement; interference: S high, K low. Ours uses only K (Beer-Lambert), so we
cannot do opaque or interference pigments. Low priority.

### 2.9 Which mechanism gives which effect

| Effect | Mechanism in Curtis |
|---|---|
| Dry brush | Mask excludes low paper cells. |
| Edge darkening | FlowOutward lowers rim pressure; outward velocity advects pigment to the edge; it is deposited there. |
| Backruns | Capillary layer grows the mask into damp paper; water pushes pigment ahead of it. |
| Granulation | `gamma` in TransferPigment biases deposition into valleys. |
| Flow patterns / wet-in-wet | Momentum flow in the shallow layer, perturbed by `grad h`, gives streaks along the flow. |
| Glazing | Separate glazes, KM compositing. |

## 3. MoXi (Chu and Tai 2005)

**From the sketch (read):** everything runs on the GPU as RGBA float textures.
Four simulation textures hold 16 fields: `VelDen = [u, v, wf, seep]`,
`Misc = [blk, f0, lwf, ws]`, `Dist1 = f[N,E,W,S]`, `Dist2 = f[NE,SE,NW,SW]`.
So it is a **D2Q9 lattice Boltzmann** solver: f0 is the rest particle, eight
directional distributions. Six passes per step: (1) derive f0 and the blocking
factor `blk`, deposit or update surface water `ws`, save `wf` to `lwf`;
(2),(3) collide; (4),(5) stream; (6) derive `u, v, wf, seep` and receive water
from the surface layer. Pinning sites get blocking factor = infinity, and the
"interpolate with infinity gives infinity" trick lets a single texture sample
detect a cross-boundary source point in the semi-Lagrangian pigment advection.

**From memory of the main paper (not verified this session):**
- Two water layers: a **surface layer** (`ws`, where the brush deposits) that
  seeps into a **flow layer** (`wf`, the LBE fluid inside the paper), plus a
  fixture layer for ink that has bonded to fibres.
- BGK collision `f_i <- f_i - (f_i - f_i^eq)/tau`, with the equilibrium scaled
  by a per-cell **blocking factor** derived from a fibre-structure texture. The
  blocking factor damps movement, so the flow percolates unevenly, which is
  what produces feathery, branching edges.
- **Pinning**: when the water density at a wet/dry boundary cell drops below a
  threshold the cell is pinned (block = infinity) and the front stops there.
  Boundary trimming removes thin tendrils.
- Ink is advected semi-Lagrangian along the LBE velocity and fixed to the paper
  at a rate; evaporation lowers `wf` over time.
- Ran at interactive rates on 2005 hardware at a few hundred pixels square.

Why it matters: ink in MoXi spreads fast because the LBE carries momentum and a
real pressure (density) gradient, not because of a diffusion coefficient.

## 4. What our shader does today, versus Curtis

| Curtis | Ours (`FS_STEP`) |
|---|---|
| `u, v` velocity, momentum, advection | **No velocity.** Water moves by neighbour level difference (diffusion-like). |
| RelaxDivergence makes local changes global | Nothing equivalent. A blob only affects its neighbours per step. |
| FlowOutward lowers rim pressure | `uEdge` raises rim evaporation; same intent, gradient only one cell deep. |
| Pigment moves only by advection | Pigment moves with water flux plus an explicit diffusion term `uPD`. Diffusion is what we rely on for bleeding, and it is slow by nature. |
| Capillary layer grows the mask | `front()` creeps pooled water into dry neighbours. Similar. |
| TransferPigment with rho, omega, gamma | `uDep`, `uLift`, `uGran`. Same shape, fewer parameters. |
| Wet mask painted, then sim run | Continuous interactive sim; mask is implicit (`W > WET`). |
| 50 relaxation iterations per step | We run up to 16 steps per frame, one pass each. |

The missing ingredient for the reported problem (paint blob next to a clean
wet circle barely moves) is **momentum**. With only level-difference flow,
two pools at the same height exchange nothing, and the pigment is left to
diffuse. With a velocity field, the blob's excess pushes a flow that keeps
going and carries pigment for many cells.

## 5. Implementation plan: add a velocity layer

Two candidate schemes. Recommendation is B.

### A. Curtis directly (staggered velocity + pressure relaxation)

- New texture `UV` with `u, v` (collocated is acceptable; MAC staggering can be
  faked by storing face velocities in the cell's own texel, u for its +x face
  and v for its +y face).
- Pass 1: UpdateVelocities (advection, viscosity, pressure gradient, drag, paper slope).
- Pass 2..N: Jacobi divergence relaxation. 50 iterations per sim step is far
  too many at 240 steps/s; 8 to 16 per *frame* is realistic, giving a water
  that is only approximately incompressible. Acceptable for painting.
- Pass N+1: MovePigment upwind along `u, v`, then TransferPigment, then
  evaporation as now.
- Risk: the pressure solve is the cost and the tuning headache. Curtis's own
  numbers (`mu`, `kappa`, `xi`, `eta`) assume their grid scale.

### B. Virtual-pipe shallow water (recommended)

Mei, Decaudin, Hu 2007 "Fast hydraulic erosion simulation and visualization on
GPU" (not re-read this session; well known). Each cell stores four outgoing
fluxes `F = [fL, fR, fT, fB]` to its neighbours. Per step:

```
# 1. flux update (one pass, writes F)
dh_dir = (W + b) - (W_nb + b_nb)              # water surface height difference; b = paper height * small weight
f_dir  = max(0, f_dir * damp + dt * A * g * dh_dir / L)
scale  = min(1, W * L*L / ((fL+fR+fT+fB) * dt))   # never send more than the cell holds
F *= scale
# 2. water update (one pass, reads F of self and 4 neighbours)
dV = dt * (inflow from neighbours - outflow)
W += dV / (L*L)
# 3. velocity for advection
u = (fR - fL + fL_nb_in - fR_nb_in) / 2, v likewise
# 4. pigment: semi-Lagrangian or upwind advection along (u, v), as Curtis MovePigment
```

Why it fits us:
- Flux has memory, so water keeps moving after the level difference is gone.
  That is the directional rush we are missing.
- Mass is conserved exactly without any iterative solve. One extra texture,
  two extra passes.
- Damping `damp` (about 0.98 to 0.995 per step) plays Curtis's `kappa`.
  Paper slope enters via `b`, playing `grad h`.
- Our existing pool/damp split survives: only run the pipe model where both
  cells are pooled (`W > uSpread`), keep the current slow capillary exchange
  for damp paper. That is Curtis's "unify by wetness" future work for free.
- Edge darkening: keep `uEdge` evaporation, and optionally lower the effective
  surface height at the rim by `eta * (1 - M')` so flow runs outward, which is
  FlowOutward in pipe form.

Stability: `dt * A * g / L` with the flux cap above is stable for any reasonable
value; the usual choice is `A = L = 1`, `g` tuned (about 1 to 10 in cell units),
`dt = 1`. Tuning sliders to expose: gravity (flow speed), damping, pigment
carry factor.

Textures after the change (all RGBA32F, ping-ponged):
- `WP` = wet pigment rgb + water (as now)
- `D`  = dried pigment rgb + edge distance (as now)
- `F`  = four outgoing fluxes (new)

Passes per step: flux update, then the existing step shader extended to read
`F` for water and pigment transport. Two draws per step instead of one.

### C. Keep in mind

- All per-step constants must be scaled by `sim.scale` so Detail does not
  change the physics (done for the diffusion constants on 2026-11-03; the pipe
  model needs `L` = cell size in CSS px, so it scales itself).
- Curtis has no pigment diffusion. Once advection works, `uPDPool` should be
  cut back toward the damp value or edges will go muddy.
- Brush: a loaded brush should deposit water with a non-zero initial flux
  outward (or simply a higher level) so a blob immediately drives flow.
- Pigment settling in Curtis is per-pigment (rho, omega, gamma). We have one
  global set. The pigment table in 2.5 is a good starting point if we go that way.
- KM rendering (opaque pigments) is independent of the fluid work and can wait.

## 6. What happened when scheme B was built (2026-11-03, branch `flow-momentum`)

Lessons from implementing and measuring it with scripted circle-and-blob tests
(clean-water circle, paint blob on its edge, simulation stepped synchronously
from script so the browser frame rate does not matter):

1. **Explicit stability budget.** Every per-step lateral operator on pigment
   (pipe advection, diffusion, speck smoothing, bleed) is a fraction of a cell's
   content moved per neighbour per step. Their sum over four neighbours must stay
   well under 1 or pigment grows without bound. The diffusion boost committed on
   main earlier the same day (pool diffusion up to 0.15 per neighbour) broke this
   and blows up to 1e38 within a simulated second. Current caps: pipe outflow
   0.25 of water per cell, pool diffusion 0.03, damp diffusion 0.05, speck 0.1,
   bleed 0.1 per neighbour.
2. **Pipe model checkerboard.** With four neighbours the checkerboard mode feels
   eight times the per-pipe gravity, so `g` must be far below 0.25 per step. At
   0.32 the grid locked into a period-2 pattern with each cell swapping its whole
   surplus with all four neighbours every step. Fixes: `g` capped at 0.1 (default
   0.03 times grid factor) and opposing pipes through one wall cancel so only net
   flow keeps momentum.
3. **Momentum alone does not bleed.** A local water surplus spreads as a wave,
   but the water parcels, and the pigment on them, move only a few cells. Curtis
   gets away with it because one of his steps is about one of our seconds (a wash
   dries in ~250 of his steps, ~10,000 of ours), so his 1% drag is enormous
   momentum by our standards, and his brush adds a pressure kick of 1.0 against
   pigment increments of 0.01 to 0.2.
4. **Wet-in-wet bleeding is surface tension, not inertia.** Paint binder lowers
   surface tension, so the surface flows from paint toward clean water
   (Marangoni spreading). Modelled as a pigment velocity between pooled
   neighbours proportional to the concentration difference, `v = k (c_c - c_n)`
   capped at 0.1 cells per step, moving the sender's pigment. Two forms that do
   not work: a concentration-dependent diffusion (still diffusion, square-root
   of time, capped by the stability budget) and a speed scaled by the relative
   difference (vanishes on any smooth ramp, and a brush edge is a ramp). The
   plain gradient form with `k` about 0.2 per unit concentration at Ultra moves a
   fresh blob's edge about 7 px in the first second, slowing as the edge softens.
   `k` scales with the square of the grid factor.
5. **Pipes feed the creep.** With water able to flow freely, the edge cells of a
   wash are always full, so the old creep rate (tuned for a diffusion-starved
   edge) doubled the wet area in 8 s. Creep default cut from 0.06 to 0.004.
   Lowering the rim surface ("flow toward edge") made this worse and dragged
   pigment outward, so it defaults to 0.
6. **Browser pane caveat.** A hidden browser pane throttles requestAnimationFrame
   to about 1 per second, so timed observations there are meaningless. Step the
   simulation from script instead.

## 7. Open questions

- Van Laerhoven 2005 would tell us exactly how they cut Curtis down for the
  GPU (iteration counts, time step). Worth obtaining through a library if the
  pipe model disappoints.
- MoXi full text for the blocking-factor equilibrium and pinning thresholds,
  if we want feathered Eastern-ink edges rather than watercolour edges.
