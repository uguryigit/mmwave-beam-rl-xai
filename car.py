from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
import math
import numpy as np
import pygame
from beam import Beam
from pathloss import CIPathloss

# Assumes these helpers exist in your module:
#   fspl_db, noise_dbm, dbm_to_mw, mw_to_dbm, shannon_capacity_bps

@dataclass
class HSeg:
    y: float
    x1: float
    x2: float
    speed: float  # m/s

    def contains(self, x: float, y: float, tol: float = 2.0) -> bool:
        if abs(y - self.y) > tol:
            return False
        lo, hi = sorted((self.x1, self.x2))
        return lo - tol <= x <= hi + tol

    def direction_unit(self, prefer: int) -> Tuple[int, int]:
        # prefer +1 (east) or -1 (west)
        return (1, 0) if prefer >= 0 else (-1, 0)

    def endpoints(self) -> List[Tuple[float, float]]:
        return [(self.x1, self.y), (self.x2, self.y)]


@dataclass
class VSeg:
    x: float
    y1: float
    y2: float
    speed: float  # m/s

    def contains(self, x: float, y: float, tol: float = 2.0) -> bool:
        if abs(x - self.x) > tol:
            return False
        lo, hi = sorted((self.y1, self.y2))
        return lo - tol <= y <= hi + tol

    def direction_unit(self, prefer: int) -> Tuple[int, int]:
        # prefer +1 (south) or -1 (north)
        return (0, 1) if prefer >= 0 else (0, -1)

    def endpoints(self) -> List[Tuple[float, float]]:
        return [(self.x, self.y1), (self.x, self.y2)]


class Car:
    def __init__(self,
                 x: float,
                 y: float,
                 heading: Tuple[int, int],
                 name: str = "V-1",
                 beams: Optional[List["Beam"]] = None,
                 pathloss_model: Optional[CIPathloss] = None):
        self.x = x
        self.y = y
        self.hx, self.hy = heading
        self.speed = 8.0
        self.on_roundabout = False
        self.round_theta: Optional[float] = None
        self.round_dir = 1
        self.segment: Optional[Union[HSeg, VSeg]] = None
        self._round_progress = 0.0
        self.reentry_cd = 0

        # --- Interference label knobs (used by UI "score" helpers only) ---
        self.interf_mode = "ratio"   # "share" or "ratio"
        self.interf_beta = 2.2
        self.interf_topk = 3
        self.interf_penalty = 3.0
        self.interf_alpha = 1.3
        self.noise_floor = 0.02

        self.name = name
        self.roundebout_exit = None

        # Path loss model (mmWave-appropriate)
        self.pathloss_model = CIPathloss(26000.0)

        # --- Beam lock + label settings ---
        self._beams: List["Beam"] = list(beams) if beams else []
        self.current_beam_idx: Optional[int] = None  # sticky assignment

        # Hysteresis thresholds for lock (on visual strengths)
        self.entry_threshold: float = 0.12
        self.exit_threshold: float = 0.06

        # Label styling
        self.label_offset_px: int = -18
        self.label_bg_rgba = (0, 0, 0, 150)
        self.label_text_rgb = (240, 240, 240)

        # Last computed RF metrics (for debugging/HUD)
        self.last_sinr_db: float = float("-inf")
        self.capacity_mbps: float = 0.0
        self.last_serving_idx: Optional[int] = None
        self.last_noise_dbm: float = float("-inf")
        self.last_interf_dbm: float = float("-inf")
        self._last_metrics = None  # per-beam dicts

    # --- Beams wiring ---
    def set_beams(self, beams: List["Beam"]):
        self._beams = list(beams or [])

    # --- RF math: CI path loss + AF pattern + penetration + interference ---
    def compute_capacity_mbps(self, beams, radio=None, serving='locked_or_strongest'):
        """
        Returns (sinr_db, capacity_mbps) using CI path loss:
          PR = EIRP_boresight + G_tx_rel(θ) - PL_CI(d, LOS/NLOS) - misc_losses + G_rx - penetration
          I  = sum other PR (linear)
          N  = -174 dBm/Hz + 10*log10(BW) + NF
          SINR = PR_serv - 10*log10(N_lin + I_lin)
          C = BW * log2(1 + 10^(SINR/10))  (Mbps)
        """
        if radio is None:
            radio = {
                'freq_mhz': 26000.0,        # MHz
                'eirp_dbm': 55.0,           # boresight EIRP (incl. peak TX gain)
                'g_rx_dbi': 8.0,
                'misc_losses_db': 15.0,     # body/orientation/rain/etc.
                'nf_db': 7.0,
                'bw_hz': 100e6,             # start with 100e6; try 400e6 later
                'sidelobe_floor_db': 1.0,  # clamp AF to ≥ -X dB (deeper = less interference)
                'interf_topk': 3,           # consider strongest K interferers (None = all)
                'n_los': 2.0,               # used only if we construct CI here
                'n_nlos': 3.4,
                'sigma_los': 100.0,
                'sigma_nlos': 200.0,
            }

        if not beams:
            # Car outside all beams - return 0 capacity
            self.last_sinr_db = float("-inf")
            self.capacity_mbps = 0.0
            self.last_serving_idx = None
            return float("-inf"), 0.0

        # Choose CI model: prefer self.pathloss_model, else build one from radio
        ci = self.pathloss_model 

        px, py = float(self.x), float(self.y)

        def pattern_rel_db(b: "Beam", rel_deg: float, floor_db: float) -> float:
            """AF power (linear 0..1) → relative dB, clamped to -floor."""
            P = float(b._power_at_rel_angle(rel_deg))  # no visual gamma/floor
            g_rel = 10.0 * math.log10(max(P, 1e-12))
            return max(g_rel, -float(floor_db))

        pr_dbm_list: List[float] = []
        per_beam: List[dict] = []

        for b in beams:
            # Check if car is in beam coverage first
            if hasattr(b, 'is_in_coverage'):
                if not b.is_in_coverage(px, py):
                    pr_dbm_list.append(float("-inf"))
                    per_beam.append({
                        'beam': b, 'rel_deg': 0, 'd_m': float('inf'), 'los': False,
                        'g_tx_rel_db': float("-inf"), 'pl_db': float('inf'), 
                        'pen_db': 0, 'pr_dbm': float("-inf")
                    })
                    continue

            # Geometry
            dx = px - float(b.x)
            dy = py - float(b.y)
            d_m = max(1.0, math.hypot(dx, dy))  # clamp ≥1 m
            abs_deg = math.degrees(math.atan2(dy, dx))
            rel_deg = ((abs_deg - float(b.dir_deg) + 180.0) % 360.0) - 180.0

            # Check if within beam coverage using existing beam method
            if hasattr(b, 'strength_at_point'):
                strength = b.strength_at_point(px, py)
                if strength <= 0.0:  # Outside beam coverage
                    pr_dbm_list.append(float("-inf"))
                    per_beam.append({
                        'beam': b, 'rel_deg': rel_deg, 'd_m': d_m, 'los': False,
                        'g_tx_rel_db': float("-inf"), 'pl_db': float('inf'), 
                        'pen_db': 0, 'pr_dbm': float("-inf")
                    })
                    continue

            # TX pattern (relative dB)
            g_tx_rel_db = pattern_rel_db(b, rel_deg, radio['sidelobe_floor_db'])

            # Penetration & LOS via ray/rect hits
            hits = b._ray_hits_sorted(abs_deg)  # [(t_m, atten_factor), ...] sorted
            pen_fac = 1.0
            los = True
            for t, fac in hits:
                if t <= d_m + 1e-6:
                    pen_fac *= float(fac)
                    los = False  # first obstacle before UE → non-LOS / penetrated
                else:
                    break
            pen_db = -10.0 * math.log10(max(pen_fac, 1e-12))

            # CI path loss (dB)
            try:
                pl_db = float(ci.get(d_m, los=los))   # CIPathloss signature
            except TypeError:
                pl_db = float(ci.get(d_m))            # generic _Pathloss fallback

            # Received power
            pr_dbm = (float(radio['eirp_dbm']) + g_tx_rel_db
                      - float(pl_db) - float(radio['misc_losses_db'])
                      + float(radio['g_rx_dbi']) - float(pen_db))

            pr_dbm_list.append(pr_dbm)
            per_beam.append({
                'beam': b, 'rel_deg': rel_deg, 'd_m': d_m, 'los': los,
                'g_tx_rel_db': g_tx_rel_db, 'pl_db': pl_db, 'pen_db': pen_db, 'pr_dbm': pr_dbm
            })

        # Check if any beam provides coverage
        valid_powers = [p for p in pr_dbm_list if p != float("-inf")]
        if not valid_powers:
            # Car outside all beam coverage
            self.last_sinr_db = float("-inf")
            self.capacity_mbps = 0.0
            self.last_serving_idx = None
            return float("-inf"), 0.0

        # Serving beam: locked → current → strongest PR
        if serving == 'locked_or_strongest' and getattr(self, 'locked_beam_id', None) is not None:
            srv_idx = min(int(self.locked_beam_id), len(pr_dbm_list) - 1)
        elif serving == 'locked_or_strongest' and getattr(self, 'current_beam_idx', None) is not None:
            srv_idx = min(int(self.current_beam_idx), len(pr_dbm_list) - 1)
        else:
            srv_idx = int(np.argmax(pr_dbm_list))

        s_dbm = pr_dbm_list[srv_idx]
        
        # If serving beam has no coverage, return 0
        if s_dbm == float("-inf"):
            self.last_sinr_db = float("-inf")
            self.capacity_mbps = 0.0
            self.last_serving_idx = None
            return float("-inf"), 0.0

        # Interference (optionally top-K)
        others = [p for i, p in enumerate(pr_dbm_list) if i != srv_idx and p != float("-inf")]
        k = radio.get('interf_topk')
        if k:
            others = sorted(others, reverse=True)[:int(k)]
        i_lin = sum(10.0 ** (p / 10.0) for p in others)

        # Thermal noise
        n_dbm = -174.0 + 10.0 * math.log10(float(radio['bw_hz'])) + float(radio['nf_db'])
        n_lin = 10.0 ** (n_dbm / 10.0)

        # SINR & Capacity
        s_lin = 10.0 ** (s_dbm / 10.0)
        sinr_lin = s_lin / (n_lin + i_lin)
        sinr_db = 10.0 * math.log10(max(sinr_lin, 1e-15))

        cap_bps = float(radio['bw_hz']) * math.log2(1.0 + max(sinr_lin, 1e-15))
        cap_mbps = cap_bps / 1e6

        # Save for HUD / debugging
        self.last_sinr_db = float(sinr_db)
        self.capacity_mbps = float(cap_mbps)
        self.last_serving_idx = srv_idx
        self.last_noise_dbm = float(n_dbm)
        self.last_interf_dbm = (10.0 * math.log10(i_lin) if i_lin > 0.0 else -1e9)
        self._last_metrics = per_beam

        return sinr_db, cap_mbps

    # --- Internal helpers for label-only strength (visual, not RF) ---
    def _punished_strength(self, s_list: List[float], idx: int) -> float:
        """Visual-only: S_eff (0..1) = S reduced by an interference score."""
        if idx is None or idx < 0 or idx >= len(s_list):
            return 0.0
        S_raw = max(0.0, min(1.0, s_list[idx]))
        score = self._interference_score(s_list, idx)

        if self.interf_mode == "share":
            factor = max(0.0, min(1.0, score))
        else:  # "ratio"
            factor = score / (1.0 + score) if score > 0.0 else 0.0
        return S_raw * factor

    def _strengths_linear(self) -> List[float]:
        """Raw per-beam visual strengths at the car (0..1)."""
        if not self._beams:
            return []
        vals = []
        for b in self._beams:
            try:
                s = float(b.strength_at_point(self.x, self.y))
            except Exception:
                s = 0.0
            vals.append(max(0.0, min(1.0, s)))
        return vals

    def _sinr_like(self, s_list: List[float], idx: int) -> float:
        """Visual-only SINR-like score for labels."""
        if idx < 0 or idx >= len(s_list):
            return 0.0
        S = s_list[idx]
        I = sum(s_list) - S
        return S / (self.noise_floor + self.interf_penalty * I)

    def _interference_score(self, s_list: List[float], idx: int) -> float:
        """Visual-only interference score."""
        if idx < 0 or idx >= len(s_list):
            return 0.0

        S = max(0.0, min(1.0, s_list[idx]))

        if self.interf_mode == "ratio":
            others = sorted([v for i, v in enumerate(s_list) if i != idx], reverse=True)[:self.interf_topk]
            I = sum(others)
            I_eff = (max(I, 0.0)) ** float(self.interf_alpha)
            denom = self.noise_floor + float(self.interf_penalty) * I_eff
            return S / max(1e-9, denom)

        # "share"
        beta = float(self.interf_beta)
        num = (S ** beta)
        denom = self.noise_floor + sum((max(0.0, min(1.0, v)) ** beta) for v in s_list)
        return num / max(1e-9, denom)

    # --- Beam lock logic (visual) ---
    def update_beam_lock(self):
        s = self._strengths_linear()
        if not s:
            self.current_beam_idx = None
            return

        # keep lock until we exit the beam
        if self.current_beam_idx is not None:
            cur = self.current_beam_idx
            if cur < len(s) and s[cur] >= self.exit_threshold:
                return
            else:
                self.current_beam_idx = None

        # acquire best if above entry threshold
        best_idx = max(range(len(s)), key=lambda i: s[i])
        self.current_beam_idx = best_idx if s[best_idx] >= self.entry_threshold else None

    def current_effective_strength(self) -> float:
        """Visual-only: effective strength of the locked beam (0..1)."""
        if not self._beams:
            return 0.0
        self.update_beam_lock()
        if self.current_beam_idx is None:
            return 0.0
        s_list = self._strengths_linear()
        return self._punished_strength(s_list, self.current_beam_idx)

    # --- Label rendering helpers ---
    def get_stregth_label(self,
                          beam_names: Optional[List[str]] = None,
                          show_db: bool = True) -> Tuple[str, float]:
        """Show RF SINR for the locked beam (uses compute_capacity_mbps)."""
        self.update_beam_lock()
        idx = self.current_beam_idx
        if idx is None:
            return ("SINR: 0.0 dB", 0.0)

        sinr_db, _ = self.compute_capacity_mbps(self._beams)
        name = (beam_names[idx] if beam_names and idx < len(beam_names) else f"B{idx}")
        if show_db:
            label = f"SINR:{sinr_db:.1f} dB {name}"
            return (label, sinr_db)
        else:
            label = f"Score:{sinr_db:.2f} {name}"
            return (label, sinr_db)

    def render_beam_label(self, surface, ppm: float, font,
                          beam_names: Optional[List[str]] = None,
                          show_db: bool = False):
        import pygame
        if font is None or not self._beams:
            return

        # draw pill above the car
        cx, cy = int(self.x * ppm), int(self.y * ppm)
        tx, ty = cx, cy + self.label_offset_px
        text = font.render(self.name, True, self.label_text_rgb)
        w, h = text.get_width(), text.get_height()
        pad = 6
        pill = pygame.Surface((w + 2 * pad, h + 2 * pad), pygame.SRCALPHA)
        pygame.draw.rect(pill, self.label_bg_rgba, (0, 0, pill.get_width(), pill.get_height()), border_radius=10)
        pill.blit(text, (pad, pad))

        W, H = surface.get_size()
        px = max(2, min(W - pill.get_width() - 2, tx - pill.get_width() // 2))
        py = max(2, min(H - pill.get_height() - 2, ty - pill.get_height() // 2))
        surface.blit(pill, (px, py))
