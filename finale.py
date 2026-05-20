
import math
import time
from collections import defaultdict

import numpy as np
import torch

LEGAL_GAP = 0.005

SA1_TEMPS = 785    # fixed — matched to best observed run, deterministic
SA1_STEPS = 1500
SA1_T0    = 0.05
SA1_T_END = 1e-4

SA2_TEMPS = 300    # was 150 — more polish
SA2_STEPS = 800
SA2_T0    = 1e-4
SA2_T_END = 1e-8   # was 1e-7 — cool further

SAB1_TEMPS = 1521  # SA-B phase 1: stops before float-underflow zone
SAB2_TEMPS = 1185  # SA-B phase 2: full polish

CONG_WEIGHT       = 0.5    # matches proxy formula weight
T_CONG_ON         = 5e-4   # only include cong in delta above this T

GRID_NC = 48
CUDA    = torch.cuda.is_available()


# ════════════════════════════════════════════════════════════════
# PROXY EVALUATOR
# ════════════════════════════════════════════════════════════════

class ProxyEval:
    def __init__(self, benchmark, NM, NH):
        self.W  = float(benchmark.canvas_width)
        self.H  = float(benchmark.canvas_height)
        self.GR = int(benchmark.grid_rows)
        self.GC = int(benchmark.grid_cols)
        self.NM = NM
        self.NH = NH

        net_pins, net_w = [], []
        for i, nodes in enumerate(benchmark.net_nodes):
            nl = [int(x) for x in nodes if 0 <= int(x) < NM]
            if len(nl) >= 2:
                net_pins.append(np.array(nl, dtype=np.int32))
                net_w.append(float(benchmark.net_weights[i]))
        self.net_pins = net_pins
        self.net_w    = np.array(net_w, dtype=np.float64)
        self.wl_norm  = (self.W + self.H) * max(1, len(net_pins))

        sizes = [len(p) for p in net_pins]
        self.offsets    = np.zeros(len(net_pins)+1, dtype=np.int32)
        for i, s in enumerate(sizes): self.offsets[i+1] = self.offsets[i]+s
        self.nodes_flat = (np.concatenate(net_pins).astype(np.int32)
                           if net_pins else np.array([], dtype=np.int32))
        self.net_ids    = np.repeat(np.arange(len(net_pins)), sizes).astype(np.int32)

        self.macro_nets: dict[int, list[int]] = defaultdict(list)
        for i, pins in enumerate(net_pins):
            for p in pins:
                self.macro_nets[int(p)].append(i)

        self.cw = self.W / self.GC
        self.ch = self.H / self.GR
        self.soft_density = None

        # GPU tensors — built once, reused every full_proxy call
        self._gpu_ready = False
        self._setup_gpu()

    def _setup_gpu(self):
        """Build GPU tensors for vectorised WL and congestion."""
        if not CUDA or len(self.net_pins) == 0:
            return
        try:
            dev = torch.device("cuda")
            self._t_nodes  = torch.tensor(self.nodes_flat, dtype=torch.long,  device=dev)
            self._t_offsets= torch.tensor(self.offsets,    dtype=torch.long,  device=dev)
            self._t_netw   = torch.tensor(self.net_w,      dtype=torch.float32, device=dev)
            self._t_wl_norm= float(self.wl_norm)
            self._gpu_ready = True
        except Exception:
            self._gpu_ready = False

    def set_soft_density(self, soft_pos, soft_half):
        self.soft_density = self._density_grid(soft_pos, soft_half)

    def wl(self, pos):
        if len(self.nodes_flat) == 0: return 0.0
        x = pos[self.nodes_flat, 0]; y = pos[self.nodes_flat, 1]
        xmax = np.maximum.reduceat(x, self.offsets[:-1])
        xmin = np.minimum.reduceat(x, self.offsets[:-1])
        ymax = np.maximum.reduceat(y, self.offsets[:-1])
        ymin = np.minimum.reduceat(y, self.offsets[:-1])
        return float((self.net_w*(xmax-xmin+ymax-ymin)).sum()) / self.wl_norm

    def _density_grid(self, mpos, mhalf):
        W, H, GR, GC = self.W, self.H, self.GR, self.GC
        cw = W/GC; ch = H/GR
        grid = np.zeros((GR, GC), dtype=np.float64)
        for i in range(len(mpos)):
            cx, cy = mpos[i,0], mpos[i,1]
            hx, hy = mhalf[i,0], mhalf[i,1]
            c0 = max(0, int((cx-hx)/cw)); c1 = min(GC-1, int((cx+hx)/cw))
            r0 = max(0, int((cy-hy)/ch)); r1 = min(GR-1, int((cy+hy)/ch))
            for r in range(r0, r1+1):
                for c in range(c0, c1+1):
                    ox = max(0.0, min((c+1)*cw, cx+hx) - max(c*cw, cx-hx))
                    oy = max(0.0, min((r+1)*ch, cy+hy) - max(r*ch, cy-hy))
                    grid[r, c] += ox*oy/(cw*ch)
        return grid

    def density_cost(self, hard_pos, hard_half):
        grid = self._density_grid(hard_pos, hard_half) + self.soft_density
        flat = grid.flatten()
        k = max(1, len(flat)//10)
        return float(np.partition(flat, -k)[-k:].mean())

    def congestion_cost(self, pos):
        GR, GC = self.GR, self.GC
        cw = self.W/GC; ch = self.H/GR
        cong = np.zeros((GR, GC), dtype=np.float64)
        for i, pins in enumerate(self.net_pins):
            xs = pos[pins,0]; ys = pos[pins,1]
            x0,x1 = xs.min(),xs.max(); y0,y1 = ys.min(),ys.max()
            hpwl = (x1-x0)+(y1-y0)
            if hpwl < 1e-9: continue
            c0 = max(0,int(x0/cw)); c1 = min(GC-1,int(x1/cw))
            r0 = max(0,int(y0/ch)); r1 = min(GR-1,int(y1/ch))
            nc = max(1,(c1-c0+1)*(r1-r0+1))
            cong[r0:r1+1, c0:c1+1] += self.net_w[i]*hpwl/(nc*cw*ch)
        flat = cong.flatten()
        k = max(1, len(flat)*5//100)
        return float(np.partition(flat, -k)[-k:].mean())

    def full_proxy(self, pos, half):
        wl   = self.wl(pos)
        dens = self.density_cost(pos[:self.NH], half[:self.NH])
        cong = self.congestion_cost(pos)
        return wl + 0.5*dens + 0.5*cong, wl, dens, cong

    def _density_grid_swap(self, hard_grid, idx, ox, oy, nx, ny, half):
        """Move macro idx in hard_grid from (ox,oy) to (nx,ny). No return value."""
        cw = self.cw; ch = self.ch; GR = self.GR; GC = self.GC
        hx = half[idx, 0]; hy = half[idx, 1]
        def _add(cx, cy, sign):
            c0 = max(0, int((cx-hx)/cw)); c1 = min(GC-1, int((cx+hx)/cw))
            r0 = max(0, int((cy-hy)/ch)); r1 = min(GR-1, int((cy+hy)/ch))
            for r in range(r0, r1+1):
                for c in range(c0, c1+1):
                    ov_x = max(0.0, min((c+1)*cw, cx+hx) - max(c*cw, cx-hx))
                    ov_y = max(0.0, min((r+1)*ch, cy+hy) - max(r*ch, cy-hy))
                    hard_grid[r, c] += sign * ov_x * ov_y / (cw * ch)
        _add(ox, oy, -1.0)
        _add(nx, ny, +1.0)

    def delta_density(self, hard_grid, idx, ox, oy, nx, ny, half):
        """
        Incremental top-10% density change for moving macro idx from (ox,oy) to (nx,ny).
        Updates hard_grid IN PLACE and returns delta (new_cost - old_cost).
        Caller must revert if move is rejected.
        O(cells touched) = O(4-12) — essentially free.
        """
        cw = self.cw; ch = self.ch
        GR = self.GR; GC = self.GC
        hx = half[idx, 0]; hy = half[idx, 1]

        def _add(cx, cy, sign):
            c0 = max(0, int((cx - hx) / cw)); c1 = min(GC-1, int((cx + hx) / cw))
            r0 = max(0, int((cy - hy) / ch)); r1 = min(GR-1, int((cy + hy) / ch))
            for r in range(r0, r1+1):
                for c in range(c0, c1+1):
                    ov_x = max(0.0, min((c+1)*cw, cx+hx) - max(c*cw, cx-hx))
                    ov_y = max(0.0, min((r+1)*ch, cy+hy) - max(r*ch, cy-hy))
                    hard_grid[r, c] += sign * ov_x * ov_y / (cw * ch)

        # Remove old, add new
        _add(ox, oy, -1.0)
        _add(nx, ny, +1.0)

        # Recompute top-10% on updated hard_grid + soft_density
        combined = hard_grid + self.soft_density
        flat = combined.ravel()
        k = max(1, len(flat) // 10)
        return float(np.partition(flat, -k)[-k:].mean())

    def delta_wl(self, pos, idx, nx, ny):
        nets = self.macro_nets.get(idx, [])
        if not nets: return 0.0
        delta = 0.0
        for ni in nets:
            pins = self.net_pins[ni]
            xs = pos[pins,0]; ys = pos[pins,1]
            old_h = xs.max()-xs.min()+ys.max()-ys.min()
            mask = pins == idx
            xs2 = xs.copy(); ys2 = ys.copy()
            xs2[mask] = nx; ys2[mask] = ny
            new_h = xs2.max()-xs2.min()+ys2.max()-ys2.min()
            delta += self.net_w[ni]*(new_h-old_h)
        return delta / self.wl_norm


# ════════════════════════════════════════════════════════════════
# SPATIAL GRID
# ════════════════════════════════════════════════════════════════

class SpatialGrid:
    def __init__(self, pos, half, W, H, nc=GRID_NC):
        self.nc = nc; self.cw = W/nc; self.ch = H/nc
        self.W = W; self.H = H; self.half = half
        self.pos  = pos.copy()
        self.cells: list[list[int]] = [[] for _ in range(nc*nc)]
        for i in range(len(pos)):
            for c in self._buckets(pos[i,0], pos[i,1], half[i,0], half[i,1]):
                self.cells[c].append(i)

    def _buckets(self, cx, cy, hx, hy):
        c0 = max(0, int(math.floor((cx-hx)/self.cw)))
        c1 = min(self.nc-1, int(math.floor((cx+hx)/self.cw)))
        r0 = max(0, int(math.floor((cy-hy)/self.ch)))
        r1 = min(self.nc-1, int(math.floor((cy+hy)/self.ch)))
        return [r*self.nc+c for r in range(r0,r1+1) for c in range(c0,c1+1)]

    def is_free(self, idx, cx, cy):
        hx, hy = self.half[idx]
        EPS = LEGAL_GAP * 0.1   # tiny buffer to catch float-precision edge cases
        for cell in self._buckets(cx, cy, hx, hy):
            for j in self.cells[cell]:
                if j == idx: continue
                if (abs(cx-self.pos[j,0]) < hx+self.half[j,0]+EPS and
                        abs(cy-self.pos[j,1]) < hy+self.half[j,1]+EPS):
                    return False
        return True

    def move(self, idx, nx, ny):
        hx, hy = self.half[idx]
        old_b = set(self._buckets(self.pos[idx,0], self.pos[idx,1], hx, hy))
        new_b = set(self._buckets(nx, ny, hx, hy))
        for c in old_b-new_b:
            try: self.cells[c].remove(idx)
            except ValueError: pass
        for c in new_b-old_b: self.cells[c].append(idx)
        self.pos[idx,0] = nx; self.pos[idx,1] = ny

    def rebuild(self, pos):
        self.pos = pos.copy()
        self.cells = [[] for _ in range(self.nc*self.nc)]
        for i in range(len(pos)):
            for c in self._buckets(pos[i,0],pos[i,1],self.half[i,0],self.half[i,1]):
                self.cells[c].append(i)


# ════════════════════════════════════════════════════════════════
# LEGALISATION
# ════════════════════════════════════════════════════════════════

def grid_legal_init(half, NH, W, H):
    """
    Place macros on a regular grid — guaranteed zero overlaps.
    Sort by area descending (larger macros placed first, easier to fit).
    Grid spacing = max macro dimension + LEGAL_GAP.
    """
    hw = half[:NH, 0]; hh = half[:NH, 1]
    area = hw * hh
    order = np.argsort(-area)   # largest first

    max_hw = hw.max(); max_hh = hh.max()
    cell_w = 2*max_hw + LEGAL_GAP*3
    cell_h = 2*max_hh + LEGAL_GAP*3

    cols = max(1, int(math.floor(W / cell_w)))
    pos  = np.zeros((NH, 2), dtype=np.float64)

    for rank, idx in enumerate(order):
        row = rank // cols
        col = rank % cols
        cx = (col + 0.5) * cell_w
        cy = (row + 0.5) * cell_h
        # Wrap if we run out of vertical space — compress rows
        if cy + hh[idx] + LEGAL_GAP > H:
            cy = float(np.clip(cy, hh[idx]+LEGAL_GAP, H-hh[idx]-LEGAL_GAP))
        cx = float(np.clip(cx, hw[idx]+LEGAL_GAP, W-hw[idx]-LEGAL_GAP))
        pos[idx, 0] = cx
        pos[idx, 1] = cy

    # Verify (grid should always be legal but check anyway)
    n_ov = _count_overlaps(pos, half, NH)
    if n_ov > 0:
        # Compress: stack macros in tight rows
        pos = _tight_shelf_pack(half, NH, W, H, order)
    return pos


def _tight_shelf_pack(half, NH, W, H, order):
    """Shelf-packing: place macros left-to-right, wrap to next row. Zero overlaps."""
    hw = half[:NH, 0]; hh = half[:NH, 1]
    pos = np.zeros((NH, 2), dtype=np.float64)
    x_cur = LEGAL_GAP; y_cur = LEGAL_GAP; row_h = 0.0

    for idx in order:
        w = 2*hw[idx]; h = 2*hh[idx]
        if x_cur + w + LEGAL_GAP > W and row_h > 0:
            x_cur = LEGAL_GAP
            y_cur += row_h + LEGAL_GAP
            row_h = 0.0
        pos[idx, 0] = float(np.clip(x_cur + hw[idx], hw[idx]+LEGAL_GAP, W-hw[idx]-LEGAL_GAP))
        pos[idx, 1] = float(np.clip(y_cur + hh[idx], hh[idx]+LEGAL_GAP, H-hh[idx]-LEGAL_GAP))
        x_cur += w + LEGAL_GAP
        row_h = max(row_h, h)

    return pos


def legalise_from_pos(pos, half, NH, W, H, max_iters=600):
    """
    Resolve overlaps from an existing placement using force-directed push.
    Only called for benchmark-init which has ~85 overlaps — manageable.
    Uses SpatialGrid for O(1) neighbour lookup instead of O(N²) per step.
    """
    pos = pos.copy()
    min_x = half[:NH,0]+LEGAL_GAP; max_x = W-half[:NH,0]-LEGAL_GAP
    min_y = half[:NH,1]+LEGAL_GAP; max_y = H-half[:NH,1]-LEGAL_GAP

    for it in range(max_iters):
        px = pos[:NH,0]; py = pos[:NH,1]
        hx = half[:NH,0]; hy = half[:NH,1]
        dx = px[:,None] - px[None,:]
        dy = py[:,None] - py[None,:]
        sx = hx[:,None]+hx[None,:]+LEGAL_GAP
        sy = hy[:,None]+hy[None,:]+LEGAL_GAP
        ov_x = sx - np.abs(dx); ov_y = sy - np.abs(dy)
        overlap = (ov_x > 0) & (ov_y > 0)
        np.fill_diagonal(overlap, False)
        if not overlap.any():
            print(f"    [FD] zero overlaps at iter {it}")
            break
        step = max(0.3, 1.0 - 0.5*it/max_iters)
        sgn_x = np.sign(dx+1e-12); sgn_y = np.sign(dy+1e-12)
        push_x = overlap & (ov_x <= ov_y); push_y = overlap & (ov_x > ov_y)
        fx = np.where(push_x, step*ov_x*sgn_x, 0.0).sum(axis=1)
        fy = np.where(push_y, step*ov_y*sgn_y, 0.0).sum(axis=1)
        pos[:NH,0] = np.clip(px+fx, min_x, max_x)
        pos[:NH,1] = np.clip(py+fy, min_y, max_y)

    n_ov = _count_overlaps(pos, half, NH)
    if n_ov > 0:
        print(f"    [FD] {n_ov} remain — running pairwise push")
        pos = _pairwise_push(pos, half, NH, W, H)
    return pos


def _pairwise_push(pos, half, NH, W, H, max_passes=20):
    """
    Simple O(N²) pairwise push — resolves remaining overlaps one pair at a time.
    Fine for small N (246 macros) as a final cleanup.
    """
    pos = pos.copy()
    min_x = half[:NH,0]+LEGAL_GAP; max_x = W-half[:NH,0]-LEGAL_GAP
    min_y = half[:NH,1]+LEGAL_GAP; max_y = H-half[:NH,1]-LEGAL_GAP

    for _ in range(max_passes):
        changed = False
        for i in range(NH):
            for j in range(i+1, NH):
                ox = half[i,0]+half[j,0] - abs(pos[i,0]-pos[j,0])
                oy = half[i,1]+half[j,1] - abs(pos[i,1]-pos[j,1])
                if ox <= 0 or oy <= 0: continue
                changed = True
                sx = math.copysign(1, pos[i,0]-pos[j,0]+1e-12)
                sy = math.copysign(1, pos[i,1]-pos[j,1]+1e-12)
                if ox <= oy:
                    push = (ox + LEGAL_GAP) * 0.5
                    pos[i,0] = float(np.clip(pos[i,0]+sx*push, min_x[i], max_x[i]))
                    pos[j,0] = float(np.clip(pos[j,0]-sx*push, min_x[j], max_x[j]))
                else:
                    push = (oy + LEGAL_GAP) * 0.5
                    pos[i,1] = float(np.clip(pos[i,1]+sy*push, min_y[i], max_y[i]))
                    pos[j,1] = float(np.clip(pos[j,1]-sy*push, min_y[j], max_y[j]))
        if not changed: break

    n_ov = _count_overlaps(pos, half, NH)
    print(f"    [pairwise-push] {n_ov} overlaps remain")
    return pos


def _count_overlaps(pos, half, NH):
    if NH == 0: return 0
    px = pos[:NH,0]; py = pos[:NH,1]
    hx = half[:NH,0]; hy = half[:NH,1]
    dx = np.abs(px[:,None]-px[None,:])
    dy = np.abs(py[:,None]-py[None,:])
    sx = hx[:,None]+hx[None,:]; sy = hy[:,None]+hy[None,:]
    ov = (dx < sx) & (dy < sy)
    np.fill_diagonal(ov, False)
    return int(ov.sum()) // 2


def count_hard_overlaps(pos, half, NH):
    cnt = 0
    for i in range(NH):
        for j in range(i+1, NH):
            if (abs(pos[i,0]-pos[j,0]) < half[i,0]+half[j,0] and
                    abs(pos[i,1]-pos[j,1]) < half[i,1]+half[j,1]):
                cnt += 1
    return cnt


# ════════════════════════════════════════════════════════════════
# SPECTRAL TARGETS (connectivity-aware gravity wells for SA)
# ════════════════════════════════════════════════════════════════

def spectral_targets(ev, NH, W, H, rng):
    """
    Compute spectral embedding as TARGET positions for SA move proposals.
    NOT used as initial positions (avoids the legalisation problem).
    SA has a 20% chance to propose a move toward the spectral target
    instead of a random Gaussian move. This biases exploration toward
    connectivity-optimal positions.
    """
    adj: dict[tuple[int,int], float] = defaultdict(float)
    for i, pins in enumerate(ev.net_pins):
        hp = [p for p in pins if p < NH]
        if len(hp) < 2: continue
        w = ev.net_w[i] / max(1, len(hp)**2)
        for a in hp:
            for b in hp:
                if a != b: adj[(a,b)] += w

    L = np.zeros((NH, NH), dtype=np.float64)
    for (a,b), w in adj.items():
        L[a,a] += w; L[a,b] -= w

    try:
        _, eigvecs = np.linalg.eigh(L)
        x_raw = eigvecs[:, 1]
        y_raw = eigvecs[:, 2] if NH > 2 else rng.standard_normal(NH)
    except Exception:
        return None

    def scale(v, lo, hi):
        v = v - v.min(); r = v.max()
        return v/r*(hi-lo)+lo if r > 1e-9 else np.full(len(v), (lo+hi)/2)

    targets = np.zeros((NH, 2), dtype=np.float64)
    targets[:,0] = scale(x_raw, W*0.05, W*0.95)
    targets[:,1] = scale(y_raw, H*0.05, H*0.95)
    return targets


# ════════════════════════════════════════════════════════════════
# T0 CALIBRATION
# ════════════════════════════════════════════════════════════════

def calibrate_T0(pos, half, NH, ev, W, H, rng,
                 target_accept=0.20, n_samples=400):
    """
    Estimate T0 so ~target_accept of random uphill moves are accepted.

    Uses sigma = 2% of canvas — matches the typical SA move size at low T,
    so we measure deltas representative of refinement-phase moves.

    Solves:  exp(-mean_delta / T0) = target_accept
             T0 = -mean_delta / ln(target_accept)

    Returns T0 clamped to [1e-6, SA1_T0].
    """
    min_x = half[:NH, 0] + LEGAL_GAP;  max_x = W - half[:NH, 0] - LEGAL_GAP
    min_y = half[:NH, 1] + LEGAL_GAP;  max_y = H - half[:NH, 1] - LEGAL_GAP
    # Use a small sigma matching early refinement moves, not random-walk moves
    sigma = min(W, H) * 0.02

    deltas = []
    for _ in range(n_samples):
        idx = int(rng.integers(0, NH))
        nx  = float(np.clip(pos[idx, 0] + rng.normal(0, sigma), min_x[idx], max_x[idx]))
        ny  = float(np.clip(pos[idx, 1] + rng.normal(0, sigma), min_y[idx], max_y[idx]))
        dw  = ev.delta_wl(pos, idx, nx, ny)
        if dw > 0:
            deltas.append(dw)

    if not deltas:
        return SA1_T0

    mean_delta = float(np.mean(deltas))
    T0 = -mean_delta / math.log(max(target_accept, 1e-6))
    return float(np.clip(T0, 1e-6, SA1_T0))


# ════════════════════════════════════════════════════════════════
# SIMULATED ANNEALING
# ════════════════════════════════════════════════════════════════

def simulated_annealing(pos, half, NH, ev, W, H, rng,
                         n_temps, n_steps, T0, T_end,
                         time_budget=3600.0,
                         label="SA",
                         targets=None):
    """
    SA over legal placements.

    Acceptance criterion:  delta = delta_wl + 0.5 * delta_density
    This covers WL + Density — the two terms SA can meaningfully optimise
    with local moves.  Congestion is checked at log_freq intervals.

    Move types:
      60% - Gaussian translate (sigma scales with T)
      20% - Swap two macros
      20% - Targeted move toward spectral target (if targets provided)

    Legality: SpatialGrid.is_free() — never introduces overlaps.
    """
    t_start = time.time()
    proxy0, wl0, d0, c0 = ev.full_proxy(pos, half)
    print(f"  [{label}] start  proxy={proxy0:.4f}  WL={wl0:.4f}  D={d0:.3f}  C={c0:.3f}")

    cur  = pos[:NH].copy()
    best = cur.copy()
    best_proxy = proxy0

    # Live hard-macro density grid — kept in sync with cur throughout SA
    hard_grid = ev._density_grid(cur, half[:NH]).copy()
    flat0 = (hard_grid + ev.soft_density).ravel()
    k0 = max(1, len(flat0)//10)
    cur_density = float(np.partition(flat0, -k0)[-k0:].mean())

    grid  = SpatialGrid(cur, half[:NH], W, H)
    temps = np.exp(np.linspace(math.log(T0), math.log(T_end), n_temps))

    min_x = half[:NH,0]+LEGAL_GAP; max_x = W-half[:NH,0]-LEGAL_GAP
    min_y = half[:NH,1]+LEGAL_GAP; max_y = H-half[:NH,1]-LEGAL_GAP

    log_freq  = max(1, n_temps//10)
    total_m   = 0; accepted = 0

    for ti, T in enumerate(temps):
        if time.time()-t_start > time_budget:
            print(f"  [{label}] time budget at {ti}/{n_temps}")
            break

        sigma = max(LEGAL_GAP*2, T * min(W,H) * 10.0)

        if ti % max(1, n_temps//8) == 0:
            grid.rebuild(cur)

        for _ in range(n_steps):
            idx = int(rng.integers(0, NH))
            roll = rng.random()

            if targets is not None and roll < 0.20:
                tx, ty = targets[idx, 0], targets[idx, 1]
                nx = float(np.clip(tx + rng.normal(0, sigma*0.3), min_x[idx], max_x[idx]))
                ny = float(np.clip(ty + rng.normal(0, sigma*0.3), min_y[idx], max_y[idx]))

            elif roll < (0.20 if targets is not None else 0.0) + 0.60:
                nx = float(np.clip(cur[idx,0]+rng.normal(0,sigma), min_x[idx], max_x[idx]))
                ny = float(np.clip(cur[idx,1]+rng.normal(0,sigma), min_y[idx], max_y[idx]))

            else:
                # ── Swap ──────────────────────────────────────────────────
                jdx = int(rng.integers(0, NH))
                if jdx == idx: continue
                ix, iy = cur[idx,0], cur[idx,1]
                jx, jy = cur[jdx,0], cur[jdx,1]
                if not grid.is_free(idx, jx, jy): continue
                if not grid.is_free(jdx, ix, iy): continue
                dw_i = ev.delta_wl(pos, idx, jx, jy)
                pos[idx,0] = jx; pos[idx,1] = jy
                dw_j = ev.delta_wl(pos, jdx, ix, iy)
                pos[idx,0] = ix; pos[idx,1] = iy
                dw = dw_i + dw_j
                # Density delta for swap: remove i from old, add to new; same for j
                # Net density change of a swap is zero if macros are same size,
                # small otherwise — skip density for swaps (negligible, avoids 2x grid work)
                if dw <= 0 or rng.random() < math.exp(-dw/(T+1e-12)):
                    pos[idx,0]=jx; pos[idx,1]=jy
                    pos[jdx,0]=ix; pos[jdx,1]=iy
                    cur[idx,0]=jx; cur[idx,1]=jy
                    cur[jdx,0]=ix; cur[jdx,1]=iy
                    # Update density grid for swap
                    ev._density_grid_swap(hard_grid, idx, ix, iy, jx, jy, half)
                    ev._density_grid_swap(hard_grid, jdx, jx, jy, ix, iy, half)
                    grid.move(idx,jx,jy); grid.move(jdx,ix,iy)
                    accepted += 1
                total_m += 1
                continue

            # ── Translate / targeted move ──────────────────────────────────
            if not grid.is_free(idx, nx, ny): total_m += 1; continue

            dw = ev.delta_wl(pos, idx, nx, ny)

            # Incremental density: apply to hard_grid, measure new top-10%
            ox, oy = cur[idx, 0], cur[idx, 1]
            new_density = ev.delta_density(hard_grid, idx, ox, oy, nx, ny, half[:NH])
            d_density   = new_density - cur_density

            # Combined delta: WL + 0.5 * density  (same weights as proxy)
            delta = dw + 0.5 * d_density

            if delta <= 0 or rng.random() < math.exp(-delta/(T+1e-12)):
                # Accept
                pos[idx,0]=nx; pos[idx,1]=ny
                cur[idx,0]=nx; cur[idx,1]=ny
                cur_density = new_density
                grid.move(idx, nx, ny)
                accepted += 1
            else:
                # Reject — revert hard_grid (delta_density mutated it in place)
                ev.delta_density(hard_grid, idx, nx, ny, ox, oy, half[:NH])
            total_m += 1

        if ti % log_freq == 0 or ti == n_temps-1:
            full_p, wl, d, c = ev.full_proxy(pos, half)
            if full_p < best_proxy:
                best_proxy = full_p; best[:] = cur
            acc = accepted/max(1,total_m)*100
            print(f"  [{label}] {ti:3d}/{n_temps}  proxy={full_p:.4f}  "
                  f"best={best_proxy:.4f}  WL={wl:.4f}  D={d:.3f}  C={c:.3f}  "
                  f"T={T:.5f}  acc={acc:.0f}%  {time.time()-t_start:.0f}s")
            accepted = 0; total_m = 0

    pos[:NH] = best
    print(f"  [{label}] done  best={best_proxy:.4f}  {time.time()-t_start:.0f}s")
    return pos, best_proxy


# ════════════════════════════════════════════════════════════════
# PLACER
# ════════════════════════════════════════════════════════════════

class Placer:
    def place(self, benchmark) -> torch.Tensor:
        t_wall = time.time()

        W   = float(benchmark.canvas_width)
        H   = float(benchmark.canvas_height)
        NM  = int(benchmark.macro_sizes.shape[0])
        NH  = NM - int(benchmark.num_soft_macros)

        print(f"\n{'='*60}")
        print(f"  NM={NM}  NH={NH}  NS={NM-NH}  canvas={W:.1f}×{H:.1f}")
        print(f"{'='*60}")

        sizes    = benchmark.macro_sizes.float().numpy().astype(np.float64)
        half     = sizes / 2.0
        pos_init = benchmark.macro_positions.float().numpy().astype(np.float64).copy()

        ev = ProxyEval(benchmark, NM, NH)
        ev.set_soft_density(pos_init[NH:], half[NH:])

        init_p, init_wl, init_d, init_c = ev.full_proxy(pos_init, half)
        print(f"  Nets={len(ev.net_pins)}  WL_norm={ev.wl_norm:.0f}")
        print(f"  Benchmark init: proxy={init_p:.4f}  WL={init_wl:.4f}  "
              f"D={init_d:.3f}  C={init_c:.3f}")

        rng = np.random.default_rng(42)

        # Spectral targets (connectivity gravity wells for SA)
        print("  Computing spectral targets...")
        targets = spectral_targets(ev, NH, W, H, rng)

        best_pos   = pos_init.copy()
        best_proxy = float('inf')

        def time_left(reserve=60.0):
            return max(0.0, 3400.0 - (time.time()-t_wall) - reserve)

        # ── Run A: Grid init → SA with spectral-biased moves ─────────────
        print(f"\n{'─'*50}")
        print("  [Run A] Grid init → SA (spectral-guided)")
        print(f"{'─'*50}")

        grid_pos = grid_legal_init(half, NH, W, H)
        n_ov = _count_overlaps(grid_pos, half, NH)
        print(f"  Grid init: {n_ov} overlaps")
        assert n_ov == 0, "Grid init must be legal!"

        pos_a = pos_init.copy()
        pos_a[:NH] = grid_pos

        p_init_a, wl_a, d_a, c_a = ev.full_proxy(pos_a, half)
        print(f"  Grid proxy={p_init_a:.4f}  WL={wl_a:.4f}  D={d_a:.3f}  C={c_a:.3f}")

        budget_a = time_left() * 0.30          # Cut Run A budget: grid init always starts ~2x worse
        pos_a, pa = simulated_annealing(
            pos_a, half, NH, ev, W, H, rng,
            n_temps=SA1_TEMPS, n_steps=SA1_STEPS,
            T0=SA1_T0, T_end=SA1_T_END,
            time_budget=budget_a, label="SA-A1",
            targets=targets)
        # pos_a now holds best from SA-A1 (simulated_annealing restores best at line 516)
        best_pos_a = pos_a.copy()             # save separately before SA-A2 overwrites

        # Polish — MUST start from SA-A1 best, not current degraded position
        budget_a2 = min(time_left(), budget_a * 0.25)
        pos_a2, pa2 = simulated_annealing(
            best_pos_a.copy(), half, NH, ev, W, H, rng,
            n_temps=SA2_TEMPS, n_steps=SA2_STEPS,
            T0=SA2_T0, T_end=SA2_T_END,
            time_budget=budget_a2, label="SA-A2",
            targets=None)

        # Pick best of SA-A1 and SA-A2
        if pa2 < pa:
            pa = pa2; best_pos_a = pos_a2.copy()

        if pa < best_proxy:
            best_proxy = pa; best_pos = best_pos_a.copy()
            print(f"  [Run A] new best: {best_proxy:.4f}")

        # ── Run B: Benchmark init → legalise → SA ────────────────────────
        if time_left() > 120:
            print(f"\n{'─'*50}")
            print("  [Run B] Benchmark init → legalise → SA")
            print(f"{'─'*50}")

            pos_b = pos_init.copy()
            n_ov_b = _count_overlaps(pos_b, half, NH)
            print(f"  Benchmark overlaps: {n_ov_b}")

            if n_ov_b > 0:
                print(f"  Legalising {n_ov_b} overlaps...")
                pos_b[:NH] = legalise_from_pos(pos_b[:NH], half[:NH], NH, W, H)
                n_ov_b = count_hard_overlaps(pos_b, half, NH)
                print(f"  After legalise: {n_ov_b} overlaps")

            if n_ov_b == 0:
                pb0, wlb, db, cb = ev.full_proxy(pos_b, half)
                print(f"  Legal proxy={pb0:.4f}  WL={wlb:.4f}  D={db:.3f}  C={cb:.3f}")

                # Calibrate T0 for Run B: bench init is already good,
                # so we need a much lower T0 than the grid-init Run A.
                T0_b   = calibrate_T0(pos_b, half, NH, ev, W, H, rng,
                                      target_accept=0.20, n_samples=400)
                Tend_b = T0_b / 1000.0   # cool by 1000x so SA actually converges
                print(f"  Calibrated T0_b={T0_b:.5f}  T_end={Tend_b:.8f}")

                budget_b = time_left() * 0.80
                pos_b, pb = simulated_annealing(
                    pos_b, half, NH, ev, W, H, rng,
                    n_temps=SA1_TEMPS, n_steps=SA1_STEPS,
                    T0=T0_b, T_end=Tend_b,
                    time_budget=budget_b, label="SA-B1",
                    targets=targets)
                # pos_b now holds SA-B1 best (simulated_annealing restores best at return)
                best_pos_b = pos_b.copy()         # save before SA-B2 can overwrite

                # Polish — MUST start from SA-B1 best
                budget_b2 = min(time_left(reserve=30.0), budget_b * 0.25)
                pos_b2, pb2 = simulated_annealing(
                    best_pos_b.copy(), half, NH, ev, W, H, rng,
                    n_temps=SA2_TEMPS, n_steps=SA2_STEPS,
                    T0=SA2_T0, T_end=SA2_T_END,
                    time_budget=budget_b2, label="SA-B2",
                    targets=None)

                # Pick best of SA-B1 and SA-B2
                if pb2 < pb:
                    pb = pb2; best_pos_b = pos_b2.copy()

                if pb < best_proxy:
                    best_proxy = pb; best_pos = best_pos_b.copy()
                    print(f"  [Run B] new best: {best_proxy:.4f}")

        # ── Final check ───────────────────────────────────────────────────
        pos = best_pos
        n_ov = count_hard_overlaps(pos, half, NH)

        # Safety net: if any overlap slipped through (e.g. float-precision edge case
        # in SpatialGrid.is_free), push them apart. This costs <1s and prevents DQ.
        if n_ov > 0:
            print(f"  [Final] {n_ov} overlap(s) detected — running emergency legalise...")
            pos[:NH] = legalise_from_pos(pos[:NH], half[:NH], NH, W, H)
            n_ov = count_hard_overlaps(pos, half, NH)
            if n_ov > 0:
                print(f"  [Final] FD failed, running pairwise push...")
                pos[:NH] = _pairwise_push(pos[:NH], half[:NH], NH, W, H)
                n_ov = count_hard_overlaps(pos, half, NH)

        proxy_f, wl_f, d_f, c_f = ev.full_proxy(pos, half)

        print(f"\n{'='*60}")
        print(f"  FINAL  proxy={proxy_f:.4f}  WL={wl_f:.4f}  D={d_f:.3f}  C={c_f:.3f}")
        print(f"         overlaps={n_ov}  elapsed={time.time()-t_wall:.0f}s")
        print(f"{'='*60}")

        if n_ov != 0:
            print(f"  WARNING: {n_ov} overlaps remain after emergency legalise!")

        out = benchmark.macro_positions.clone().float()
        for i in range(NH):
            out[i,0] = float(pos[i,0])
            out[i,1] = float(pos[i,1])
        return out