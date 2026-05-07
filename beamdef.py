# beam_rays.py
# Beam composed of rays with strength; blockages (axis-aligned rectangles)
# reduce strength only AFTER the ray first hits them. No masks, no cars.

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable

import pygame
import numpy as np
from functools import lru_cache


# ----------------------------- helpers -----------------------------

# ---------- array/HPBW helpers ----------
def _hamming_weights(N: int) -> np.ndarray:
    if N <= 1:
        return np.ones(1)
    n = np.arange(N, dtype=float)
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))

def _array_factor(phi_deg: np.ndarray, N: int, d_lambda: float,
                  steer_deg: float = 0.0, weights: Optional[np.ndarray] = None) -> np.ndarray:
    phi = np.deg2rad(phi_deg)
    k = 2 * np.pi
    d = d_lambda
    if weights is None:
        weights = np.ones(N, dtype=complex)
    weights = np.asarray(weights, dtype=complex)
    phi0 = math.radians(steer_deg)
    beta = -k * d * math.sin(phi0)
    n = np.arange(N)
    psi = k * d * np.sin(phi)[:, None] + beta
    AF = np.sum(weights[None, :] * np.exp(1j * n[None, :] * psi), axis=1)
    return AF

def _find_hpbw(phi_deg: np.ndarray, P_norm: np.ndarray):
    PdB = 10 * np.log10(np.maximum(P_norm, 1e-12))
    i0 = np.argmin(np.abs(phi_deg))
    target = PdB[i0] - 3.0
    # right
    xr = None
    for i in range(i0, len(PdB) - 1):
        if (PdB[i] - target) * (PdB[i + 1] - target) <= 0:
            x1, y1 = phi_deg[i], PdB[i]
            x2, y2 = phi_deg[i + 1], PdB[i + 1]
            xr = x1 + (target - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x1
            break
    # left
    xl = None
    for i in range(i0, 0, -1):
        if (PdB[i] - target) * (PdB[i - 1] - target) <= 0:
            x1, y1 = phi_deg[i], PdB[i]
            x2, y2 = phi_deg[i - 1], PdB[i - 1]
            xl = x1 + (target - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x1
            break
    if xl is None or xr is None:
        return None
    return xr - xl, xl, xr

@lru_cache(maxsize=None)
def _tune_array_for_hpbw(target_deg: float):
    phi_deg = np.linspace(-180, 180, 8001)
    best_err = None
    best = None
    for N in range(2, 13):
        w = _hamming_weights(N)
        for d in np.linspace(0.15, 0.60, 226):
            AF = _array_factor(phi_deg, N, d, 0.0, w)
            P = np.abs(AF) ** 2
            P /= P.max()
            res = _find_hpbw(phi_deg, P)
            if res is None:
                continue
            h, L, R = res
            err = abs(h - target_deg)
            if (best_err is None) or (err < best_err):
                best_err = err
                best = (N, d, h, L, R)
            if err < 0.2:
                return best
    return best

def _sector_radius_km_for_area(theta_deg: float, area_km2: float) -> float:
    # A = (theta/360) * pi * R^2  ->  R = sqrt(A*360/(pi*theta))
    if theta_deg <= 0:
        return 0.0
    return math.sqrt(area_km2 * 360.0 / (math.pi * theta_deg))

def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0



# ----------------------------- model ------------------------------

# ---------- Strength-based Beam (uses your AF helpers) ----------
from typing import List, Tuple, Optional, Dict, Any
import math
import numpy as np
import pygame

class Beam:
    """
    Strength-rendered main-lobe sector shaped by ULA array factor.

    Geometry:
      - HPBW bounds from measured AF crossings (L..R degrees)
      - Range along a ray: r0(φ) = R_max * P(φ), where R_max derives from (sector_deg, area_km2)

    Strength:
      - Base per-angle power P(φ) from AF (normalized 0..1)
      - Blockages are axis-aligned rectangles with attenuation in [0,1] (or loss_db)
      - A ray’s strength at depth d is:  P(φ) * ∏ atten_i  for all rectangles whose
        FIRST hit distance t_i <= d  (i.e., dim only after obstacle is met)

    Key API:
      - strength_at_point(px, py) -> 0..1
      - has_strength(px, py, min_strength=0.1) -> bool
      - set_blockages([...]) / add_blockage(...)
      - render(surface, ppm, ...) draws by strength (no masks)
    """

    def __init__(self,
                 x: float,
                 y: float,
                 dir_deg: float = 0.0,
                 sector_deg: float = 90.0,
                 area_km2: float = 0.0491,
                 color_rgba: Tuple[int, int, int, int] = (255, 105, 180, 190),  # base pink
                 outline_color: Tuple[int, int, int, int] = (255, 120, 0, 255),
                 name: str = "Beam"):
        self.x = float(x)
        self.y = float(y)
        self.dir_deg = float(dir_deg)
        self.sector_deg = float(sector_deg)
        self.area_km2 = float(area_km2)
        self.color_rgba = color_rgba
        self.outline_color = outline_color
        self.name = name
        # shaping options
        self.radius_mode = "af"      # "af" (current, r0 ~ P) or "constant" (r0 = R_max)
        self.strength_mode = "gamma"    # "af" (current), "flat", or "gamma"
        self.strength_gamma = 0.6    # used when strength_mode == "gamma" (P**gamma lifts edges)
        self.strength_floor = 0.3    # clamp base strength >= this (e.g., 0.3 keeps edges from dropping too low)


        # AF profile over [-180, 180] deg
        self._phi_all = np.linspace(-180.0, 180.0, 4001)
        self._P = None                         # normalized array-factor power vs angle
        self._Lhpbw = None; self._Rhpbw = None # HPBW bounds (deg, relative)
        self._R_m = None; self._R_km = None    # max radius scale (meters, km)

        # tuned AF params for HUD (optional)
        self._N = None; self._dlam = None; self._h_meas = None

        # Blockages: list of dicts {x1,y1,x2,y2,atten}
        self._blockages: List[Dict[str, float]] = []

        # cached fonts
        self._font = None

        self._recompute_profile()

    # ---- configuration -------------------------------------------------
    def set_direction_deg(self, deg: float):
        self.dir_deg = float(_wrap_deg(deg))

    def rotate_deg(self, delta: float):
        self.dir_deg = float(_wrap_deg(self.dir_deg + delta))

    def set_sector_deg(self, sector_deg: float):
        self.sector_deg = float(sector_deg)
        self._recompute_profile()

    def set_area_km2(self, area_km2: float):
        self.area_km2 = float(area_km2)
        self._recompute_profile()

    def move_to(self, x: float, y: float):
        self.x = float(x); self.y = float(y)

    # ---- blockages -----------------------------------------------------
    def set_blockages(self, rects: List[Any]):
        """
        Accepts:
          - tuples (x1,y1,x2,y2, atten) with atten in [0,1] or >1 interpreted as dB loss
          - dicts {"x1":...,"y1":...,"x2":...,"y2":...,"atten":...} or {"loss_db":...}
        """
        out: List[Dict[str, float]] = []
        for r in rects or []:
            if isinstance(r, (list, tuple)) and len(r) >= 5:
                x1,y1,x2,y2,a = r[:5]
                fac = float(a if a <= 1.0 else 10.0**(-a/10.0))
            elif isinstance(r, dict):
                x1=float(r["x1"]); y1=float(r["y1"]); x2=float(r["x2"]); y2=float(r["y2"])
                if "atten" in r: fac = float(r["atten"])
                elif "loss_db" in r: fac = 10.0**(-float(r["loss_db"])/10.0)
                else: fac = 0.0
            else:
                continue
            xa, xb = (x1, x2) if x1 <= x2 else (x2, x1)
            ya, yb = (y1, y2) if y1 <= y2 else (y2, y1)
            out.append({"x1":xa, "y1":ya, "x2":xb, "y2":yb,
                        "atten": max(0.0, min(1.0, fac))})
        self._blockages = out

    def add_blockage(self, x1, y1, x2, y2, atten: Optional[float]=None, loss_db: Optional[float]=None):
        if atten is None and loss_db is None:
            atten = 0.0
        if loss_db is not None:
            atten = 10.0**(-float(loss_db)/10.0)
        atten = max(0.0, min(1.0, float(atten)))
        xa, xb = (x1, x2) if x1 <= x2 else (x2, x1)
        ya, yb = (y1, y2) if y1 <= y2 else (y2, y1)
        self._blockages.append({"x1":xa, "y1":ya, "x2":xb, "y2":yb, "atten":atten})

    # ---- internals -----------------------------------------------------
    def _recompute_profile(self):
        tuned = _tune_array_for_hpbw(self.sector_deg)
        if tuned is None:
            self._Lhpbw, self._Rhpbw = -self.sector_deg*0.5, self.sector_deg*0.5
            self._P = np.ones_like(self._phi_all)
            self._N, self._dlam, self._h_meas = 1, 0.5, self.sector_deg
        else:
            N, dlam, h_meas, L, R = tuned
            self._N, self._dlam, self._h_meas = N, dlam, h_meas
            w = _hamming_weights(N)
            AF = _array_factor(self._phi_all, N, dlam, 0.0, w)
            P = np.abs(AF)**2
            P /= P.max()
            self._P = P
            self._Lhpbw, self._Rhpbw = L, R

        R_km = _sector_radius_km_for_area(self.sector_deg, self.area_km2)
        self._R_km = R_km
        self._R_m  = R_km * 1000.0

    def _power_at_rel_angle(self, rel_deg: float) -> float:
        rel_deg = _wrap_deg(rel_deg) - 180
        return float(np.interp(rel_deg, self._phi_all, self._P))

    # --- ray/rect intersection helpers ---
    def _ray_rect_first_hit(self, a_rad: float, rect: Dict[str, float]) -> Optional[float]:
        """
        Return distance t>=0 from (x0,y0) along ray angle a_rad to first hit with rect edges.
        Return 0 if the origin starts inside the rect. None if miss.
        """
        x0,y0 = self.x, self.y
        dx, dy = math.cos(a_rad), math.sin(a_rad)
        x1,y1,x2,y2 = rect["x1"], rect["y1"], rect["x2"], rect["y2"]

        if x1 <= x0 <= x2 and y1 <= y0 <= y2:
            return 0.0

        ts: List[float] = []
        if abs(dx) > 1e-9:
            t = (x1 - x0)/dx;  y = y0 + t*dy
            if t >= 0 and y1-1e-9 <= y <= y2+1e-9: ts.append(t)
            t = (x2 - x0)/dx;  y = y0 + t*dy
            if t >= 0 and y1-1e-9 <= y <= y2+1e-9: ts.append(t)
        if abs(dy) > 1e-9:
            t = (y1 - y0)/dy;  x = x0 + t*dx
            if t >= 0 and x1-1e-9 <= x <= x2+1e-9: ts.append(t)
            t = (y2 - y0)/dy;  x = x0 + t*dx
            if t >= 0 and x1-1e-9 <= x <= x2+1e-9: ts.append(t)
        if not ts:
            return None
        return min(ts)

    def _ray_hits_sorted(self, abs_deg: float) -> List[Tuple[float, float]]:
        """Sorted [(t_hit_m, atten_factor), ...] for this absolute direction (degrees)."""
        a = math.radians(abs_deg)
        hits: List[Tuple[float, float]] = []
        for r in self._blockages:
            t = self._ray_rect_first_hit(a, r)
            if t is not None:
                hits.append((t, r["atten"]))
        hits.sort(key=lambda p: p[0])
        return hits

    # ---- strength queries ---------------------------------------------
    def strength_at_point(self, px: float, py: float) -> float:
        dx, dy = px - self.x, py - self.y
        theta_deg = math.degrees(math.atan2(dy, dx))
        rel = _wrap_deg(theta_deg - self.dir_deg) - 180

        # angular gate (still HPBW bounded)
        if rel < self._Lhpbw or rel > self._Rhpbw:
            return 0.0

        P = self._power_at_rel_angle(rel)  # 0..1 from AF

        # --- radius shaping ---
        if self.radius_mode == "constant":
            r0 = self._R_m
        else:  # "af"
            r0 = self._R_m * P

        r = math.hypot(dx, dy)
        if r > r0 + 1e-6:
            return 0.0

        # --- strength shaping (decoupled from geometry if you want) ---
        if self.strength_mode == "flat":
            base = 1.0
        elif self.strength_mode == "gamma":
            base = max(P, 1e-6) ** float(self.strength_gamma)
        else:  # "af"
            base = P
        base = max(base, float(self.strength_floor))

        # attenuation AFTER first hits up to the point (unchanged)
        hits = self._ray_hits_sorted(self.dir_deg + rel)
        att = 1.0
        for t, fac in hits:
            if t <= r + 1e-6:
                att *= fac
            else:
                break
        return float(base * att)


    def has_strength(self, px: float, py: float, min_strength: float = 0.1) -> bool:
        """Convenience boolean: True iff strength_at_point(px,py) >= min_strength."""
        return self.strength_at_point(px, py) >= float(min_strength)

    # ---- rendering (strength, no masks) --------------------------------
    def render(self,
               surface: pygame.Surface,
               ppm: float,
               edge_steps: int = 256,
               font: Optional[pygame.font.Font] = None,
               show_panel: bool = False,
               draw_blockages: bool = False):
        """
        Draw the beam by strength:
          - Split sector into small quads between adjacent rays (edge_steps controls angular resolution)
          - Subdivide each quad at all first-hit depths from either bordering ray
          - Fill each strip with alpha = base_alpha * strength at the NEAR edge (so dimming starts after obstacles)
          - Blockages do NOT shorten geometry; they only dim alpha beyond their hit depths
        """
        W, H = surface.get_size()
        cx, cy = int(self.x * ppm), int(self.y * ppm)
        base_r, base_g, base_b, base_a = self.color_rgba

        # rays across HPBW (relative deg) and corresponding absolute angles
        rels = np.linspace(self._Lhpbw, self._Rhpbw, max(3, edge_steps))
        abs_list = [self.dir_deg + float(r) for r in rels]
        # AF base power and AF-limited radius on each ray
        Pvals = np.interp(rels, self._phi_all, self._P)
        if self.radius_mode == "constant":
            r0s_m = [self._R_m for _ in Pvals]
        else:
            r0s_m = [self._R_m * float(P) for P in Pvals]


        # precompute blockage first-hit lists for all rays
        hitlists = [self._ray_hits_sorted(a) for a in abs_list]

        overlay = pygame.Surface((W, H), pygame.SRCALPHA)

        # helper: cumulative attenuation up to depth d along a given ray hitlist
        def att_at_depth(hits, d):
            a = 1.0
            for t, f in hits:
                if t <= d + 1e-6: a *= f
                else: break
            return a

        # iterate adjacent ray pairs (make quads)
        for i in range(len(rels) - 1):
            rel1, rel2 = float(rels[i]), float(rels[i+1])
            a1_deg, a2_deg = abs_list[i], abs_list[i+1]
            a1, a2 = math.radians(a1_deg), math.radians(a2_deg)
            P1, P2 = float(Pvals[i]), float(Pvals[i+1])
            r1max, r2max = r0s_m[i], r0s_m[i+1]
            r_end = min(r1max, r2max)
            if r_end <= 1e-6:
                continue

            hits1, hits2 = hitlists[i], hitlists[i+1]

            # breakpoints along depth where dimming changes (first hits on either ray)
            B: List[float] = [0.0]
            for t,_ in hits1:
                if t <= r_end + 1e-6: B.append(max(0.0, t))
            for t,_ in hits2:
                if t <= r_end + 1e-6: B.append(max(0.0, t))
            B.append(r_end)
            B = sorted(set(round(b, 6) for b in B))
            if len(B) < 2:
                continue

            for k in range(len(B)-1):
                d0, d1 = B[k], B[k+1]
                if d1 - d0 <= 1e-6:
                    continue

                # strength at NEAR depth (piecewise constant until next breakpoint)
                s1 = P1 * att_at_depth(hits1, d0)
                s2 = P2 * att_at_depth(hits2, d0)
                s  = 0.5 * (s1 + s2)
                if s <= 1e-6:
                    continue
                alpha = int(max(0, min(255, base_a * s)))

                # quad corners (pixels)
                x11 = cx + (d0 * ppm) * math.cos(a1); y11 = cy + (d0 * ppm) * math.sin(a1)
                x12 = cx + (d0 * ppm) * math.cos(a2); y12 = cy + (d0 * ppm) * math.sin(a2)
                x21 = cx + (d1 * ppm) * math.cos(a2); y21 = cy + (d1 * ppm) * math.sin(a2)
                x22 = cx + (d1 * ppm) * math.cos(a1); y22 = cy + (d1 * ppm) * math.sin(a1)

                pygame.draw.polygon(
                    overlay,
                    (base_r, base_g, base_b, alpha),
                    [(int(x11), int(y11)), (int(x12), int(y12)),
                     (int(x21), int(y21)), (int(x22), int(y22))],
                    0
                )

        # draw HPBW boundary rays (use unattenuated r on those rays)
        aL = math.radians(self.dir_deg + self._Lhpbw)
        aR = math.radians(self.dir_deg + self._Rhpbw)
        rL = r0s_m[0]; rR = r0s_m[-1]
        pL = (int(cx + (rL*ppm)*math.cos(aL)), int(cy + (rL*ppm)*math.sin(aL)))
        pR = (int(cx + (rR*ppm)*math.cos(aR)), int(cy + (rR*ppm)*math.sin(aR)))
        pygame.draw.line(overlay, self.outline_color, (cx, cy), pL, 2)
        pygame.draw.line(overlay, self.outline_color, (cx, cy), pR, 2)

        # optional: visualize blockage rectangles
        if draw_blockages and self._blockages:
            for r in self._blockages:
                x1, y1 = int(r["x1"] * ppm), int(r["y1"] * ppm)
                x2, y2 = int(r["x2"] * ppm), int(r["y2"] * ppm)
                rect = pygame.Rect(min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1))
                pygame.draw.rect(overlay, (40,40,40,120), rect)
                pygame.draw.rect(overlay, (235,235,235,200), rect, 1)

        surface.blit(overlay, (0, 0))

        if font is None:
            if self._font is None:
                self._font = pygame.font.SysFont("Arial", 16)
            font = self._font
        beam_name = getattr(self, 'name', 'Beam')
        info = f"{beam_name}"
        pad = 8
        rendered = font.render(info, True, (240,240,240))
        w = rendered.get_width() + 2*pad
        h = rendered.get_height() + 2*pad
        # render at the start of the beam (near the origin)
        Rpx_vis = max(20, int(40 * ppm))  # small offset from beam origin
        aC = math.radians(self.dir_deg)
        ax = int(cx + Rpx_vis * math.cos(aC)); ay = int(cy + Rpx_vis * math.sin(aC))
        W, H = surface.get_size()
        if ax > W-500:
            ax -= 150
        ax = max(4, min(W - w - 4, ax)); ay = max(4, min(H - h - 4, ay))
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(panel, (0,0,0,120), (0,0,w,h), border_radius=10)
        panel.blit(rendered, (pad, pad))
        surface.blit(panel, (ax, ay))
