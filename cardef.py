from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
import pygame
from beam import Beam
import math

# ---- Radio math helpers (mmWave / n258) -----------------------------------------
import math
import numpy as np

def fspl_db(f_mhz: float, d_km: float | np.ndarray) -> float | np.ndarray:
    """Free-space path loss in dB. d_km is clamped to >= 0.001 km to avoid log(0)."""
    d_km = np.maximum(d_km, 0.001)
    return 32.45 + 20*np.log10(f_mhz) + 20*np.log10(d_km)

def noise_dbm(bw_hz: float, nf_db: float) -> float:
    """Thermal noise in dBm for bandwidth and receiver NF (dB)."""
    return -174 + 10*math.log10(bw_hz) + nf_db

def dbm_to_mw(x_dbm: float | np.ndarray) -> float | np.ndarray:
    return 10**(np.asarray(x_dbm)/10.0)

def mw_to_dbm(x_mw: float | np.ndarray) -> float | np.ndarray:
    return 10*np.log10(np.maximum(np.asarray(x_mw), 1e-15))

def shannon_capacity_bps(bw_hz: float, sinr_db: float) -> float:
    sinr_lin = 10**(sinr_db/10.0)
    return bw_hz * math.log2(1.0 + sinr_lin)


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


from typing import Optional, Tuple, List

class Car:
    def __init__(self, x: float, y: float, heading: Tuple[int, int], name: str = "V-1",
                 beams: Optional[List["Beam"]] = None,):
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
        # --- Interference severity knobs ---
        self.interf_mode    = "ratio"   # "share" (strong punishment) or "ratio"
        self.interf_beta    = 2.2       # for "share": exponent >1 penalizes overlaps a lot
        self.interf_topk    = 3         # only strongest K interferers matter
        self.interf_penalty = 3.0       # for "ratio": scales interference
        self.interf_alpha   = 1.3       # for "ratio": I -> I^alpha (alpha>1 = harsher)
        self.noise_floor    = 0.02      # small baseline so denom>0
        self.name = name
        self.roundebout_exit = None


        # --- NEW: beam lock + label settings ---
        self._beams: List["Beam"] = list(beams) if beams else []
        self.current_beam_idx: Optional[int] = None  # sticky assignment

        # Hysteresis: acquire when above entry, drop only when below exit
        self.entry_threshold: float = 0.12   # acquire if S >= this
        self.exit_threshold: float  = 0.06   # release if S < this

        # Interference model for label strength (SINR-like)
        self.noise_floor: float = 0.02       # small floor so denom never 0
        self.interf_penalty: float = 2.0     # scales how much other beams hurt (>=1)

        # Label rendering
        self.label_offset_px: int = -18
        self.label_bg_rgba = (0, 0, 0, 150)
        self.label_text_rgb = (240, 240, 240)

    # --- Beams wiring ---
    def set_beams(self, beams: List["Beam"]):
        self._beams = list(beams or [])

    def compute_capacity_mbps(self, beams, radio=None, serving='locked_or_strongest'):
        """
        Compute this car's downlink capacity (Mbps) from given beams.
        beams: list like [main_beam, *interference_beams]
        radio: dict with radio params (defaults below)
        serving: 'locked_or_strongest' (use locked beam if any, else strongest)
        """
        # ---- Defaults for n258 (edit to your scenario) ----
        if radio is None:
            radio = {
                'freq_mhz': 26000.0,   # ~26 GHz
                'eirp_dbm': 55.0,
                'g_rx_dbi': 0.0,
                'misc_losses_db': 15.0,  # rain/body/orientation etc.
                'nf_db': 7.0,
                'bw_hz': 100e6          # choose 100e6 or 400e6
            }

        px, py = float(self.x), float(self.y)

        # Measure strength from each beam at the car and build Pr for each
        strengths = []
        pr_dbm_list = []
        for b in beams:
            # Distance (meters→km) from beam origin to car
            dx = px - float(b.x)
            dy = py - float(b.y)
            d_km = math.hypot(dx, dy) / 1000.0

            # Array-factor / blockage aware normalized strength in [0,1]
            s = float(b.strength_at_point(px, py))
            # Convert normalized power to dB offset (s=1 → 0 dB, s=0.5 → -3 dB, clamp tiny)
            s_db = 10.0 * math.log10(max(s, 1e-9))

            # Path loss and received power
            l_fspl = float(fspl_db(radio['freq_mhz'], d_km))
            pr_dbm = radio['eirp_dbm'] + radio['g_rx_dbi'] - l_fspl - radio['misc_losses_db'] + s_db

            strengths.append(s)
            pr_dbm_list.append(pr_dbm)

        # Choose serving beam
        if len(beams) == 0:
            self.capacity_mbps = 0.0
            return self.capacity_mbps

        if serving == 'locked_or_strongest' and getattr(self, 'locked_beam_id', None) is not None:
            srv_idx = min(self.locked_beam_id, len(beams)-1)
        else:
            srv_idx = int(np.argmax(strengths)) 

        # Signal, Interference, Noise
        s_dbm = pr_dbm_list[srv_idx]
        i_mw = 0.0
        for j, pr_dbm in enumerate(pr_dbm_list):
            if j == srv_idx:
                continue
            i_mw += dbm_to_mw(pr_dbm)

        n_dbm = noise_dbm(radio['bw_hz'], radio['nf_db'])
        denom_mw = dbm_to_mw(n_dbm) + i_mw
        sinr_db = s_dbm - mw_to_dbm(denom_mw)

        # Capacity
        cap_bps = shannon_capacity_bps(radio['bw_hz'], sinr_db)
        self.capacity_mbps = cap_bps / 1e6
        return self.capacity_mbps


    def _punished_strength(self, s_list: List[float], idx: int) -> float:
        """
        Returns S_eff (0..1): raw S reduced by interference, using the same
        interference model as _interference_score().
        - share:   S_eff = S_raw * Score        (Score in [0,1])
        - ratio:   S_eff = S_raw * SINR/(1+SINR)  (maps SINR∈[0,∞) -> [0,1))
        """
        if idx is None or idx < 0 or idx >= len(s_list):
            return 0.0
        S_raw = max(0.0, min(1.0, s_list[idx]))
        score = self._interference_score(s_list, idx)

        if self.interf_mode == "share":
            factor = max(0.0, min(1.0, score))
        else:  # "ratio" (SINR-like)
            # Normalize SINR to [0,1): f(s)=s/(1+s) is stable & monotone
            factor = score / (1.0 + score) if score > 0.0 else 0.0

        return S_raw * factor


    # --- Internal helpers ---
    def _strengths_linear(self) -> List[float]:
        """Raw per-beam strengths at the car (0..1)."""
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
        """Interference-aware score for beam idx: S / (noise + penalty * sum(other S))."""
        if idx < 0 or idx >= len(s_list):
            return 0.0
        S = s_list[idx]
        I = sum(s_list) - S
        return S / (self.noise_floor + self.interf_penalty * I)

    def _interference_score(self, s_list: List[float], idx: int) -> float:
        """
        Returns a punishment-heavy, interference-aware score in [0, 1] by default.
        - "share": S^beta / (noise + sum_j S_j^beta)          (strong, stable, bounded)
        - "ratio": S / (noise + penalty * (sum_topk others)^alpha)  (classic SINR-like)
        """
        if idx < 0 or idx >= len(s_list):
            return 0.0

        S = max(0.0, min(1.0, s_list[idx]))

        if self.interf_mode == "ratio":
            others = sorted([v for i,v in enumerate(s_list) if i != idx], reverse=True)[:self.interf_topk]
            I = sum(others)
            I_eff = (max(I, 0.0)) ** float(self.interf_alpha)
            denom = self.noise_floor + float(self.interf_penalty) * I_eff
            return S / max(1e-9, denom)

        # default: "share"
        beta = float(self.interf_beta)
        num   = (S ** beta)
        denom = self.noise_floor + sum((max(0.0, min(1.0, v)) ** beta) for v in s_list)
        return num / max(1e-9, denom)


    # --- Beam lock logic (call this once per frame) ---
    def update_beam_lock(self):
        s = self._strengths_linear()
        if not s:
            self.current_beam_idx = None
            return

        # If we already have a lock, keep it as long as we're still inside it
        if self.current_beam_idx is not None:
            cur = self.current_beam_idx
            if s[cur] >= self.exit_threshold:
                # stay locked; do NOT switch even if others are stronger
                return
            else:
                # fell below exit threshold -> release
                self.current_beam_idx = None

        # No lock: acquire the beam with largest raw strength above entry threshold
        best_idx = max(range(len(s)), key=lambda i: s[i])
        if s[best_idx] >= self.entry_threshold:
            self.current_beam_idx = best_idx
        else:
            self.current_beam_idx = None

    def current_effective_strength(self) -> float:
        """Effective strength of the locked beam at this car (0..1)."""
        if not self._beams:
            return 0.0
        self.update_beam_lock()
        if self.current_beam_idx is None:
            return 0.0
        s_list = self._strengths_linear()
        return self._punished_strength(s_list, self.current_beam_idx)

    # --- Label rendering: show ONLY the locked beam, with interference-aware strength ---

    def get_stregth_label(self,beam_names: Optional[List[str]] = None,show_db: bool = True) -> Tuple[str,float]:
        self.update_beam_lock()
        idx = self.current_beam_idx
        if idx is None:
            return f" S={0}->{0}  SINR={0}",0.0

        # strengths at this car
        s_list = self._strengths_linear()
        S_raw  = s_list[idx]
        score  = self._interference_score(s_list, idx)   # Score (share) or SINR (ratio)
        S_eff  = self._punished_strength(s_list, idx)    # <-- reduced S

        # Label text
        name = (beam_names[idx] if beam_names and idx < len(beam_names) else f"B{idx}")
        if self.interf_mode == "ratio" and show_db:
            score_disp = f"{10.0 * math.log10(max(score, 1e-12)):.1f} dB"
            label = f"{self.name}:{name}  S={S_raw:.2f}->{S_eff:.2f}  SINR={score_disp}"
        else:
            label = f"{self.name}:{name}  S={S_raw:.2f}->{S_eff:.2f}  Score={score:.2f}"


        return (label,10.0 * math.log10(max(score, 1e-12)) if self.interf_mode == "ratio" else score)

    def render_beam_label(self, surface, ppm: float, font,
                        beam_names: Optional[List[str]] = None,
                        show_db: bool = False):
        if font is None or not self._beams:
            return

        # keep sticky lock
        # self.update_beam_lock()
        # idx = self.current_beam_idx
        # if idx is None:
        #     return

        # strengths at this car
        # s_list = self._strengths_linear()
        # S_raw  = s_list[idx]
        # score  = self._interference_score(s_list, idx)   # Score (share) or SINR (ratio)
        # S_eff  = self._punished_strength(s_list, idx)    # <-- reduced S

        # # Label text
        # name = (beam_names[idx] if beam_names and idx < len(beam_names) else f"B{idx}")
        # if self.interf_mode == "ratio" and show_db:
        #     score_disp = f"{10.0 * math.log10(max(score, 1e-12)):.1f} dB"
        #     label = f"{name}  S={S_raw:.2f}->{S_eff:.2f}  SINR={score_disp}"
        # else:
        #     label = f"{name}  S={S_raw:.2f}->{S_eff:.2f}  Score={score:.2f}"

        # draw pill above the car
        cx, cy = int(self.x * ppm), int(self.y * ppm)
        tx, ty = cx, cy + self.label_offset_px
        text = font.render(self.name, True, self.label_text_rgb)
        w, h = text.get_width(), text.get_height()
        pad = 6
        pill = pygame.Surface((w + 2*pad, h + 2*pad), pygame.SRCALPHA)
        pygame.draw.rect(pill, self.label_bg_rgba, (0, 0, pill.get_width(), pill.get_height()), border_radius=10)
        pill.blit(text, (pad, pad))

        W, H = surface.get_size()
        px = max(2, min(W - pill.get_width() - 2, tx - pill.get_width() // 2))
        py = max(2, min(H - pill.get_height() - 2, ty - pill.get_height() // 2))
        surface.blit(pill, (px, py))


