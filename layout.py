import importlib
import os
import sys
import subprocess
import venv
from pathlib import Path
import platform

# List of required packages
REQUIRED_PACKAGES = ["pygame", "numpy", "gymnasium"]
ENV_FLAG = "VENV_BOOTSTRAPPED"


def kmh(v): 
    return v / 3.6

SPEED_ROAD_SLOW   = kmh(30)   # 8.333... m/s
SPEED_ROAD_FAST   = kmh(50)   # 13.889... m/s
SPEED_RAB         = kmh(20)   # 5.556... m/s


def in_venv() -> bool:
    # Check if we are already inside a virtual environment
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)

def venv_python(venv_dir: Path) -> str:
    # Return the correct python path inside the virtual environment
    if platform.system() == "Windows":
        return str(venv_dir / "Scripts" / "python.exe")
    return str(venv_dir / "bin" / "python")

def ensure_packages():
    missing = []
    # Check which packages are missing
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)

    if not missing:
        return  # All required packages are already installed

    # If we are already in a venv, install missing packages there
    if in_venv():
        try:
            print(f"[INFO] Installing missing packages into active venv: {missing}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            return
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Installation into active venv failed: {e}")

    # Otherwise, create a new .venv in the project directory
    proj_dir = Path(__file__).resolve().parent
    vdir = proj_dir / ".venv"

    if not vdir.exists():
        print("[INFO] .venv not found, creating one...")
        venv.create(str(vdir), with_pip=True)

    py = venv_python(vdir)

    # Install missing packages into the new venv
    try:
        print(f"[INFO] Installing {missing} into {vdir.name}...")
        subprocess.check_call([py, "-m", "pip", "install", *missing])
    except subprocess.CalledProcessError as e:
        msg = (
            f"[ERROR] Installation failed for {missing}: {e}\n"
            "Try manually:\n"
            f"  {py} -m pip install {' '.join(missing)}\n"
            "Or create and activate a venv manually:\n"
            "  python3 -m venv .venv && source .venv/bin/activate && pip install pygame numpy gymnasium\n"
        )
        print(msg)
        raise

    # Restart the script using the new venv Python (to ensure imports succeed)
    if os.environ.get(ENV_FLAG) != "1":
        print("[INFO] Restarting script with virtual environment...")
        env = os.environ.copy()
        env[ENV_FLAG] = "1"
        os.execvpe(py, [py, *sys.argv], env)


ensure_packages()
import pygame
import numpy as np
import gymnasium as gym

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Union
from beam import Beam
from gymnasium import spaces

# --- Colors (Google Maps native theme) ---
ASPHALT = (255, 255, 255)        # white roads
LANE_MARK = (210, 210, 210)      # subtle road border
GRID_BG = (242, 240, 234)        # off-white land background
CAR_COLOR = (66, 133, 244)       # Google blue
ISLAND = (220, 220, 215)         # roundabout island (slightly darker than bg)
TEXT = (220, 220, 220)           # HUD text (HUD panel stays black)
TRANSITION_CHANCE = 0.25
BIN_PER_PIXEL = 20


# --- Beamforming-based Beam class with in-beam info panel ---
import math
from functools import lru_cache
from typing import List, Tuple, Optional

import numpy as np
import pygame

# --- Beamforming-based Beam class with interference rendering/counting ---
import math
from functools import lru_cache
from typing import List, Tuple, Optional

import numpy as np
import pygame

from car import Car, HSeg, VSeg



class RoundaboutSegmentedEnv(gym.Env):
    """Sparse *segmented* road network + roundabout with per-road speeds.

    - World: 500m x 500m
    - Step: 0.1s
    - Roads: lists of horizontal (HSeg) and vertical (VSeg) segments, each with its own speed
    - Roundabout at (round_x, round_y). Roundabout speed is configurable
    - 5 auto-controlled cars; no acceleration; speed = current road/roundabout speed
    - Pygame renderer
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self,
                 render_mode: Optional[str] = None,
                 width_m: int = 500,
                 height_m: int = 500,
                 dt: float = 0.1,
                 pixels_per_meter: float = 1.5,
                 car_count: int = 20,
                 round_x: float = 125.0,
                 round_y: float = 125.0,
                 round_radius: float = 25.0,
                 round_lane: float = 6.0,
                 round_speed_mps: float = SPEED_RAB,
                 seed: Optional[int] = None):
        super().__init__()
        self._init_seed = seed
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            random.seed(int(seed))
        self.W = float(width_m)
        self.H = float(height_m)
        self.dt = dt
        self.ppm = pixels_per_meter
        self.N = car_count
        self.beam = None

        self.turn_cost_per_step = float(200)   # Mbps per 45° hop
        self.sector_switch_cost = float(100)   # Mbps one-time when sector changes
        self.prev_action = None

        self.step_count = 0
        self.max_step = 1_800
        self.interference_beams = []
        self.directions = [0,45,90,135,180,225,270,315]
        self.angles = [65,90,120]
        
        # Left sector (0–100 range, 5 values for x and y)
        v = np.linspace(0, 100, 5, dtype=int)   # [0, 25, 50, 75, 100]
        pairsLeft = np.array([[x, y] for x in v for y in v])  # 25 pairs

        # Center sector (200–300 range, 5 values for x and y)
        v = np.linspace(200, 300, 5, dtype=int)  # [200, 225, 250, 275, 300]
        pairsCenter = np.array([[x, y] for x in v for y in v])  # 25 pairs

        # Right sector (x in 400–500, y in 0–100, 5 values each)
        v1 = np.linspace(400, 500, 5, dtype=int)  # [400, 425, 450, 475, 500]
        v2 = np.linspace(0, 100, 5, dtype=int)    # [0, 25, 50, 75, 100]
        pairsRight = np.array([[x, y] for x in v1 for y in v2])  # 25 pairs
        # Combine all
        self.allPairs = np.vstack([pairsLeft, pairsCenter, pairsRight])
        self.pos_index = { (int(x), int(y)): i for i, (x, y) in enumerate(self.allPairs.tolist()) }
        self.dir_index = { int(d): i for i, d in enumerate(self.directions) }
        self.ang_index = { int(a): i for i, a in enumerate(self.angles) }

        # --- Build default segmented network (matches your snippet) ---
        self.hroads: List[HSeg] = [
            # y, x1, x2, speed (m/s)
            HSeg(125, 0, 100, SPEED_ROAD_SLOW),
            HSeg(125, 150, 250, SPEED_ROAD_SLOW),
            HSeg(125, 250, 500, SPEED_ROAD_FAST),
            HSeg(350, 0, 500, SPEED_ROAD_FAST),
        ]
        self.vroads: List[VSeg] = [
            # x, y1, y2, speed
            VSeg(125, 0, 100, SPEED_ROAD_SLOW),
            VSeg(125, 150, 250, SPEED_ROAD_SLOW),
            VSeg(125, 250, 500, SPEED_ROAD_FAST),
            VSeg(200, 350, 500, SPEED_ROAD_FAST),
            VSeg(300, 0, 125, SPEED_ROAD_FAST),
            VSeg(375, 0, 500, SPEED_ROAD_FAST),
        ]

        # --- Roundabout ---
        self.round_x = round_x
        self.round_y = round_y
        self.round_radius = round_radius
        self.round_lane = round_lane
        self.round_speed = round_speed_mps

        # cars
        self.cars: List[Car] = []
        # gym api

        self.action_space = spaces.MultiDiscrete([8, 3])
        # Define bin size and calculate number of bins
        bins_x = int(self.W / BIN_PER_PIXEL)
        bins_y = int(self.H / BIN_PER_PIXEL)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=0,
                high=self.N,
                shape=(bins_x, bins_y),
                dtype=np.int32
            ),
            "beam": spaces.MultiDiscrete([8, 3])  # direction, angle
        })

        # rendering
        self.render_mode = render_mode
        self._screen = None
        self._clock = None
        self._font = None

        # RNG
        self._rng = np.random.default_rng()

        # build node index for intersections (segment endpoints and crossings)
        self.nodes = self._build_nodes()
        self.show_markers = False

    # ------------------------- helpers -------------------------
    def _perpendicular_available(self, x: float, y: float, for_seg) -> bool:
        """Is there any perpendicular segment at (x,y)?"""
        segs = self._segments_at_point(x, y, tol=3.0)
        if isinstance(for_seg, HSeg):
            return any(t == "V" for t, _ in segs)
        else:
            return any(t == "H" for t, _ in segs)

    def _force_return(self, car: Car, nudge: float = 0.8):
        """Flip heading and nudge inside the current segment so we don't re-trigger the tip."""
        car.hx, car.hy = -car.hx, -car.hy
        seg = car.segment
        if isinstance(seg, HSeg):
            lo, hi = sorted((seg.x1, seg.x2))
            # put car slightly inside from whichever tip we're at
            if abs(car.x - lo) < abs(car.x - hi):
                car.x = min(lo + nudge, hi - 0.1)
            else:
                car.x = max(hi - nudge, lo + 0.1)
            car.y = seg.y
        elif isinstance(seg, VSeg):
            lo, hi = sorted((seg.y1, seg.y2))
            if abs(car.y - lo) < abs(car.y - hi):
                car.y = min(lo + nudge, hi - 0.1)
            else:
                car.y = max(hi - nudge, lo + 0.1)
            car.x = seg.x
        car.speed = self._current_speed(car)
        
    def _find_straight_neighbor(self, seg, heading: Tuple[int, int], tol: float = 3.0):
        """
        If there's a same-orientation segment that abuts 'seg' exactly at the tip
        in the given heading, return that neighbor; else None.
        """
        hx, hy = heading
        if isinstance(seg, HSeg):
            y = seg.y
            lo, hi = sorted((seg.x1, seg.x2))
            if hx == 1:  # going east: next segment should start at x ~= hi
                for h in self.hroads:
                    if h is seg: 
                        continue
                    if abs(h.y - y) <= tol:
                        h_lo, h_hi = sorted((h.x1, h.x2))
                        if abs(h_lo - hi) <= tol:   # abuts on the right
                            return h
            elif hx == -1:  # going west: next segment should end at x ~= lo
                for h in self.hroads:
                    if h is seg:
                        continue
                    if abs(h.y - y) <= tol:
                        h_lo, h_hi = sorted((h.x1, h.x2))
                        if abs(h_hi - lo) <= tol:   # abuts on the left
                            return h
            return None

        if isinstance(seg, VSeg):
            x = seg.x
            lo, hi = sorted((seg.y1, seg.y2))
            if hy == 1:  # going south: neighbor should start at y ~= hi
                for v in self.vroads:
                    if v is seg:
                        continue
                    if abs(v.x - x) <= tol:
                        v_lo, v_hi = sorted((v.y1, v.y2))
                        if abs(v_lo - hi) <= tol:   # abuts below
                            return v
            elif hy == -1:  # going north: neighbor should end at y ~= lo
                for v in self.vroads:
                    if v is seg:
                        continue
                    if abs(v.x - x) <= tol:
                        v_lo, v_hi = sorted((v.y1, v.y2))
                        if abs(v_hi - lo) <= tol:   # abuts above
                            return v
            return None

        return None

    def _force_turn_from_tip(self, car: Car, options: List[Tuple[str, int]], eps: float = 0.8) -> int:
        """
        Pick a perpendicular segment and force a turn away from the tip.
        Chooses the direction with more available length on the chosen segment.
        Returns 1 (transitioned).
        """
        # Determine the exact tip point we are at
        seg = car.segment
        if isinstance(seg, HSeg):
            lo, hi = sorted((seg.x1, seg.x2))
            x_tip = hi if car.hx == 1 else lo
            y_tip = seg.y
        else:  # VSeg
            lo, hi = sorted((seg.y1, seg.y2))
            x_tip = seg.x
            y_tip = hi if car.hy == 1 else lo

        # Pick a perpendicular segment (random among available)
        tag, idx = random.choice(options)
        new_seg = self._seg_object((tag, idx))

        if isinstance(new_seg, HSeg):
            h_lo, h_hi = sorted((new_seg.x1, new_seg.x2))
            # available length to each side
            right_len = max(0.0, h_hi - x_tip)
            left_len  = max(0.0, x_tip - h_lo)
            # pick direction with more room (break ties to the right)
            car.hx, car.hy = (random.choice([1, -1]), 0)
            car.segment = new_seg
            car.y = new_seg.y
            car.x = x_tip + car.hx * eps  # nudge onto the road

        else:  # VSeg
            v_lo, v_hi = sorted((new_seg.y1, new_seg.y2))
            down_len = max(0.0, v_hi - y_tip)
            up_len   = max(0.0, y_tip - v_lo)
            car.hx, car.hy = (0, random.choice([1, -1]))
            car.segment = new_seg
            car.x = new_seg.x
            car.y = y_tip + car.hy * eps  # nudge onto the road

        car.speed = self._current_speed(car)
        return 1

    def _build_nodes(self, tol: float = 1e-6):
        nodes: Dict[Tuple[int, int], List[Tuple[str, int]]] = {}
        # index endpoints (tag with type and index)
        for i, h in enumerate(self.hroads):
            for (x, y) in h.endpoints():
                key = (int(round(x)), int(round(y)))
                nodes.setdefault(key, []).append(("H", i))
        for i, v in enumerate(self.vroads):
            for (x, y) in v.endpoints():
                key = (int(round(x)), int(round(y)))
                nodes.setdefault(key, []).append(("V", i))
        # also index crossings where a horizontal and vertical segment overlap
        for hi, h in enumerate(self.hroads):
            lo, hi_x = sorted((h.x1, h.x2))
            for vi, v in enumerate(self.vroads):
                lo_y, hi_y = sorted((v.y1, v.y2))
                if lo - tol <= v.x <= hi_x + tol and lo_y - tol <= h.y <= hi_y + tol:
                    key = (int(round(v.x)), int(round(h.y)))
                    nodes.setdefault(key, []).extend([("H", hi), ("V", vi)])
        # add roundabout attach points (on cardinal axes)
        for (x, y) in [
            (self.round_x - self.round_radius - self.round_lane, self.round_y),
            (self.round_x + self.round_radius + self.round_lane, self.round_y),
            (self.round_x, self.round_y - self.round_radius - self.round_lane),
            (self.round_x, self.round_y + self.round_radius + self.round_lane),
        ]:
            key = (int(round(x)), int(round(y)))
            nodes.setdefault(key, []).append(("R", -1))

        return nodes

    def _segments_at_point(self, x: float, y: float, tol: float = 5.0):
        segs: List[Tuple[str, int]] = []
        for i, h in enumerate(self.hroads):
            if h.contains(x, y, tol):
                segs.append(("H", i))
        for i, v in enumerate(self.vroads):
            if v.contains(x, y, tol):
                segs.append(("V", i))
        return segs

    def _current_speed(self, car: Car) -> float:
        if car.on_roundabout:
            return self.round_speed
        seg = car.segment
        if seg is None:
            return 8.0
        return seg.speed

    def _all_endpoints(self) -> List[Tuple[float, float, Tuple[int,int], object]]:
        endpoints: List[Tuple[float, float, Tuple[int, int], object]] = []
        for h in self.hroads:
            (x1, y1), (x2, y2) = h.endpoints()
            endpoints.append((x1, y1, (1, 0), h))
            endpoints.append((x2, y2, (-1, 0), h))
        for v in self.vroads:
            (xv, y1), (_, y2) = v.endpoints()
            endpoints.append((xv, y1, (0, 1), v))
            endpoints.append((xv, y2, (0, -1), v))
        return endpoints

    def _random_spawn(self, car: Optional[Car] = None, exclude: Optional[Tuple[float,float]] = None):
        starts = self._edge_entry_points()
        if not starts:
            raise RuntimeError("No start points to spawn cars. Define some HSeg/VSeg first.")
        # avoid same point and avoid overlapping with other cars at spawn
        taken = {(round(c.x,1), round(c.y,1)) for c in self.cars}
        if exclude is not None:
            candidates = [s for s in starts if (round(s[0],1), round(s[1],1)) != (round(exclude[0],1), round(exclude[1],1))]
            candidates = candidates or starts
        else:
            candidates = list(starts)
        self._rng.shuffle(candidates)
        # choose first not taken; if all taken, allow with small forward offset
        pick = None
        for s in candidates:
            key = (round(s[0],1), round(s[1],1))
            if key not in taken:
                pick = s
                break
        if pick is None:
            pick = candidates[0]
        x, y, hd, seg = pick
        hx, hy = hd
        # If location is taken, push forward by 6m increments until free (max 5 tries)
        step = 6.0
        tries = 0
        while (round(x,1), round(y,1)) in taken and tries < 5:
            x += hx * step
            y += hy * step
            tries += 1
        # clamp inside segment bounds
        if isinstance(seg, HSeg):
            lo, hi = sorted((seg.x1, seg.x2))
            x = max(lo, min(hi, x))
            y = seg.y
        else:
            lo, hi = sorted((seg.y1, seg.y2))
            y = max(lo, min(hi, y))
            x = seg.x
        if car is None:
            car = Car(x, y, (hx, hy))
            car.segment = seg
            car.speed = self._current_speed(car)
            return car
        else:
            car.x, car.y = x, y
            car.hx, car.hy = hx, hy
            car.segment = seg
            car.on_roundabout = False
            car.round_theta = None
            car._round_progress = 0.0
            car.speed = self._current_speed(car)
            return car

    def _start_points(self) -> List[Tuple[float, float, Tuple[int,int], object]]:
        """
        Spawn ONLY from the first 3 horizontal segments and first 3 vertical segments.
        We still only use tips that touch the world borders (x=0/W for H, y=0/H for V).
        """
        starts: List[Tuple[float, float, Tuple[int,int], object]] = []
        tol = 1e-3

        # First 3 horizontals
        for h in self.hroads[:3]:
            lo, hi = sorted((h.x1, h.x2))
            if lo <= 0.0 + tol:
                starts.append((0.0, h.y, (1, 0), h))          # enter from left edge → eastbound
            if hi >= self.W - tol:
                starts.append((self.W, h.y, (-1, 0), h))      # enter from right edge → westbound

        # First 3 verticals
        for v in self.vroads[:3]:
            lo, hi = sorted((v.y1, v.y2))
            if lo <= 0.0 + tol:
                starts.append((v.x, 0.0, (0, 1), v))          # enter from top edge → southbound
            if hi >= self.H - tol:
                starts.append((v.x, self.H, (0, -1), v))      # enter from bottom edge → northbound

        return starts


    def _edge_entry_points(self) -> List[Tuple[float, float, Tuple[int,int], object]]:
        """
        Return entry points ONLY at world edges so cars never spawn mid-road,
        even if a physically straight road is split into multiple segments.
        - For HSeg: use x==0 (eastbound) and x==W (westbound)
        - For VSeg: use y==0 (southbound) and y==H (northbound)
        """
        starts: List[Tuple[float, float, Tuple[int,int], object]] = []
        tol = 1e-3

        # all horizontals
        for h in self.hroads:
            lo, hi = sorted((h.x1, h.x2))
            if lo <= 0.0 + tol:
                starts.append((0.0, h.y, (1, 0), h))            # enter from left edge → eastbound
            if hi >= self.W - tol:
                starts.append((self.W, h.y, (-1, 0), h))        # enter from right edge → westbound

        # all verticals
        for v in self.vroads:
            lo, hi = sorted((v.y1, v.y2))
            if lo <= 0.0 + tol:
                starts.append((v.x, 0.0, (0, 1), v))            # enter from top edge → southbound
            if hi >= self.H - tol:
                starts.append((v.x, self.H, (0, -1), v))        # enter from bottom edge → northbound

        return starts


    def _start_points_init(self) -> List[Tuple[float, float, Tuple[int,int], object]]:
        """Return canonical *starts* of each road: the minimum coordinate end pointing inward.
        HSeg -> (min(x1,x2), y) heading east. VSeg -> (x, min(y1,y2)) heading south.
        """
        starts: List[Tuple[float, float, Tuple[int,int], object]] = []
        for h in self.hroads:
            lo, hi = sorted((h.x1, h.x2))
            if self.round_x - self.round_radius - self.round_lane < lo < self.round_x + self.round_radius + self.round_lane:
                continue
            starts.append((lo, h.y, (1, 0), h))
        for v in self.vroads:
            lo, hi = sorted((v.y1, v.y2))
            if self.round_y - self.round_radius - self.round_lane < lo < self.round_y + self.round_radius + self.round_lane:
                continue
            starts.append((v.x, lo, (0, 1), v))
        return starts


    # ------------------------- Gym API -------------------------
    def reset(self, pairIndex: int, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.step_count = 0
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.cars.clear()

        # x_val,y_val = self.allPairs[random.randint(0,74)]
        x_val,y_val = self.allPairs[pairIndex]
        x_val=100;y_val=100    # "dir_deg": 45, "ang_deg": 120
        #x_val=425;y_val=75     # "dir_deg": 180, "ang_deg": 120
        #x_val=250;y_val=250    # "dir_deg": 225, "ang_deg": 120
        dir_val=self.directions[random.randint(0,7)]
        sector_val=self.angles[random.randint(0,2)]
        self.beam = Beam(x=x_val, y=y_val, dir_deg=dir_val, sector_deg=sector_val, name="BS:0")
        blockages = [
            # atten as multiplicative factor (e.g., 0.5 → 3 dB extra one-way loss)
            {"x1": 40, "y1": 135, "x2": 80, "y2": 160, "atten": 0.5},
            {"x1": 300, "y1": 190, "x2": 365, "y2": 135, "atten": 0.2},
            {"x1": 135, "y1": 40, "x2": 190, "y2": 80, "atten": 0.5},
        ]
        self.beam.set_blockages(blockages)
        self.interference_beams.clear()
        self.interference_beams.append(Beam(x=0, y=500, dir_deg=315.0, sector_deg=90.0, color_rgba=(88, 79, 247, 128), name="BS:1"))
        self.interference_beams.append(Beam(x=500, y=500, dir_deg=225.0, sector_deg=90.0, color_rgba=(88, 79, 247, 128), name="BS:2"))
        starts = self._edge_entry_points()
        if not starts:
            raise RuntimeError("No start points to spawn cars. Define some HSeg/VSeg first.")
        self._rng.shuffle(starts)
        # place unique first, then reuse with spacing if needed
        for i in range(self.N):
            if i < len(starts):
                x, y, hd, seg = starts[i]
                car = Car(x, y, hd, name=f"V-{i}")
                car.segment = seg
                car.speed = self._current_speed(car)
                car.set_beams([self.beam] + self.interference_beams)
                self.cars.append(car)
            else:
                # reuse starts cyclically but offset forward to avoid overlap
                base = starts[i % len(starts)]
                bx, by, (hx, hy), seg = base
                offset = 6.0 * ((i // len(starts)) + 1)
                x = bx + hx * offset
                y = by + hy * offset
                if isinstance(seg, HSeg):
                    lo, hi = sorted((seg.x1, seg.x2))
                    x = max(lo, min(hi, x))
                    y = seg.y
                else:
                    lo, hi = sorted((seg.y1, seg.y2))
                    y = max(lo, min(hi, y))
                    x = seg.x
                car = Car(x, y, (hx, hy), name=f"V-{i}")
                car.segment = seg
                car.speed = self._current_speed(car)
                car.set_beams([self.beam] + self.interference_beams)
                self.cars.append(car)

        self.prev_action = None
        return self._get_obs(), {}

    def _total_capacity_mbps(self, beams):
        tot = 0.0
        for c in self.cars:
            _, cap = c.compute_capacity_mbps(beams)
            tot += cap
        return tot

    def _movement_penalty_mbps(self, action):
        """
        action = (direction_idx, sector_idx)
        - direction_idx in {0..7}  (each step = 45 degrees, circular)
        - sector_idx    in {0,1,2} (angles [65, 90, 120])
        Returns a penalty in Mbps.
        """
        if self.prev_action is None:
            return 0.0

        prev_dir, prev_sec = self.prev_action
        dir_now,  sec_now  = action

        # circular distance on 8 directions
        d = (dir_now - prev_dir) % 8
        dir_steps = min(d, 8 - d)             # number of 45° hops
        penalty = self.turn_cost_per_step * dir_steps

        # sector switch penalty (one-time)
        if sec_now != prev_sec:
            penalty += self.sector_switch_cost

        return penalty

    def step(self, action):
        self.step_count += 1
        for car in self.cars:
            self._update_car(car)

        direction, sector = action
        
        self.beam.set_direction_deg(direction * 45 + 180)
        self.beam.set_sector_deg([65, 90, 120][sector])

        if self.render_mode == "human":
            self.render()

        cap_base = self._total_capacity_mbps(self.interference_beams)
        move_penalty = self._movement_penalty_mbps((direction, sector))

        # --- with agent: capacity with all 3 beams (agent ON)
        beams_all = [self.beam] + self.interference_beams
        cap_with = self._total_capacity_mbps(beams_all)

        # --- marginal gain is the reward
        delta = cap_with - cap_base

        # scale (pick a stable normalizer for your units)
        # e.g., Mbps → Gbps per user, or empirical constant
        normalizer = max(1.0, 1000.0 * self.N)  # keep reward ~ O(0.1–1)
        reward = float((delta - move_penalty) / normalizer)
        self.prev_action = (direction, sector)
        # (optional) proportional-fairness utility instead of raw sum:
        # eps = 1e-3
        # util = sum(math.log(eps + c_i) for each car i)
        # reward = (util_with - util_base)
        
        return self._get_obs(), reward, self.step_count >= self.max_step, self.step_count >= self.max_step, {}


    # ------------------------- Dynamics -------------------------
    def _update_car(self, car: Car):
        # step down roundabout re-entry cooldown
        if car.reentry_cd > 0:
            car.reentry_cd -= 1
        car.speed = self._current_speed(car)
        if car.on_roundabout:
            R = self.round_radius
            omega = car.speed / max(1e-3, R)
            adv = abs(omega * self.dt)
            car.round_theta = (car.round_theta + car.round_dir * omega * self.dt) % (2 * math.pi)
            car.x = self.round_x + R * math.cos(car.round_theta)
            car.y = self.round_y + R * math.sin(car.round_theta)
            car._round_progress += adv
            # try to exit when aligned to an attached road and not immediately after entry
            tol = 0.06
            exits = [  # directions point AWAY from the roundabout
                (0.0, (1, 0), (self.round_x + self.round_radius + self.round_lane, self.round_y)),      # RIGHT -> east
                (math.pi, (-1, 0), (self.round_x - self.round_radius - self.round_lane, self.round_y)), # LEFT  -> west
                (-math.pi/2, (0, -1), (self.round_x, self.round_y - self.round_radius - self.round_lane)), # TOP   -> north
                ( math.pi/2, (0,  1), (self.round_x, self.round_y + self.round_radius + self.round_lane)), # BOTTOM-> south
            ][car.roundebout_exit]
            for theta_target, dir_vec, attach_point in [exits]:
                if abs(((car.round_theta - theta_target + math.pi) % (2*math.pi)) - math.pi) < tol:
                    segs = self._segments_at_point(*attach_point)
                    seg = self._pick_segment_for_dir(segs, dir_vec)
                    if seg and (car._round_progress > 0.5):
                        car.on_roundabout = False
                        car.round_theta = None
                        car.roundebout_exit = None
                        car._round_progress = 0.0
                        car.hx, car.hy = dir_vec
                        car.x, car.y = attach_point
                        car.segment = self._seg_object(seg)
                        car.speed = self._current_speed(car)
                        car.reentry_cd = max(car.reentry_cd, int(max(1, round(0.8 / self.dt))))  # ~0.8s cooldown
                        return
            return

        # Move along current segment
        car.x += car.hx * car.speed * self.dt
        car.y += car.hy * car.speed * self.dt

        # If we overshot outside current segment bounds (due to large dt*speed),
        # try to handoff to a contiguous same-orientation neighbor first.
        # --- inside _update_car, replace the overshoot handling with this ---

        seg = car.segment
        if isinstance(seg, HSeg):
            lo, hi = sorted((seg.x1, seg.x2))
            if car.x > hi + 0.2 and car.hx == 1:
                over = car.x - hi
                nb = self._find_straight_neighbor(seg, (1, 0))
                if nb:
                    n_lo, n_hi = sorted((nb.x1, nb.x2))
                    car.segment = nb
                    car.y = nb.y
                    car.x = min(n_hi, n_lo + max(0.0, over))
                    car.speed = self._current_speed(car)
                    return
                # no straight neighbor; if perpendicular exists at the tip -> U-turn
                if self._perpendicular_available(hi, seg.y, seg):
                    car.x = hi  # snap to tip
                    self._force_return(car)
                    return
                prev = (round(car.x,1), round(car.y,1))
                self._random_spawn(car, exclude=prev)
                return

            if car.x < lo - 0.2 and car.hx == -1:
                over = lo - car.x
                nb = self._find_straight_neighbor(seg, (-1, 0))
                if nb:
                    n_lo, n_hi = sorted((nb.x1, nb.x2))
                    car.segment = nb
                    car.y = nb.y
                    car.x = max(n_lo, n_hi - max(0.0, over))
                    car.speed = self._current_speed(car)
                    return
                if self._perpendicular_available(lo, seg.y, seg):
                    car.x = lo
                    self._force_return(car)
                    return
                prev = (round(car.x,1), round(car.y,1))
                self._random_spawn(car, exclude=prev)
                return

        elif isinstance(seg, VSeg):
            lo, hi = sorted((seg.y1, seg.y2))
            if car.y > hi + 0.2 and car.hy == 1:
                over = car.y - hi
                nb = self._find_straight_neighbor(seg, (0, 1))
                if nb:
                    n_lo, n_hi = sorted((nb.y1, nb.y2))
                    car.segment = nb
                    car.x = nb.x
                    car.y = min(n_hi, n_lo + max(0.0, over))
                    car.speed = self._current_speed(car)
                    return
                if self._perpendicular_available(seg.x, hi, seg):
                    car.y = hi
                    self._force_return(car)
                    return
                prev = (round(car.x,1), round(car.y,1))
                self._random_spawn(car, exclude=prev)
                return

            if car.y < lo - 0.2 and car.hy == -1:
                over = lo - car.y
                nb = self._find_straight_neighbor(seg, (0, -1))
                if nb:
                    n_lo, n_hi = sorted((nb.y1, nb.y2))
                    car.segment = nb
                    car.x = nb.x
                    car.y = max(n_lo, n_hi - max(0.0, over))
                    car.speed = self._current_speed(car)
                    return
                if self._perpendicular_available(seg.x, lo, seg):
                    car.y = lo
                    self._force_return(car)
                    return
                prev = (round(car.x,1), round(car.y,1))
                self._random_spawn(car, exclude=prev)
                return


        # keep inside current segment bounds; if near an endpoint, try to transition
        state = self._maybe_transition(car)
        if state == -1:  # DEAD END -> respawn at a different start point
            prev = (round(car.x,1), round(car.y,1))
            self._random_spawn(car, exclude=prev)
            return
        elif state == 0:
            # clamp to bounds to avoid drifting off the road
            if isinstance(car.segment, HSeg):
                lo, hi = sorted((car.segment.x1, car.segment.x2))
                car.y = car.segment.y
                car.x = max(lo, min(hi, car.x))
            elif isinstance(car.segment, VSeg):
                lo, hi = sorted((car.segment.y1, car.segment.y2))
                car.x = car.segment.x
                car.y = max(lo, min(hi, car.y))

        # attempt to enter roundabout if endpoint aligns with its attach point
        if self._near_roundabout_entry(car):
            self._enter_roundabout(car)

    def _maybe_transition(self, car: Car, tol: float = 3.0) -> int:
        """
        Return 1 if transitioned, 0 if stayed, -1 if dead-end.
        Force a perpendicular turn when there's no straight option but turn options exist.
        Never re-selects the same segment.
        """
        def at_seg_end(c: Car, eps: float = 1.0) -> bool:
            tol_conn = max(eps, 40.0)
            if isinstance(c.segment, HSeg):
                lo, hi = sorted((c.segment.x1, c.segment.x2))
                y = c.segment.y
                at_lo = abs(c.x - lo) < eps
                at_hi = abs(c.x - hi) < eps
                if not (at_lo or at_hi):
                    return False
                x_tip = lo if at_lo else hi
                for h in self.hroads:
                    if h is c.segment: 
                        continue
                    if abs(h.y - y) <= tol_conn:
                        h_lo, h_hi = sorted((h.x1, h.x2))
                        if (abs(h_lo - x_tip) <= tol_conn or abs(h_hi - x_tip) <= tol_conn or
                            (h_lo - tol_conn <= x_tip <= h_hi + tol_conn)):
                            return False
                for v in self.vroads:
                    if (abs(v.x - x_tip) <= tol_conn and
                        min(v.y1, v.y2) - tol_conn <= y <= max(v.y1, v.y2) + tol_conn):
                        return False
                return True

            if isinstance(c.segment, VSeg):
                lo, hi = sorted((c.segment.y1, c.segment.y2))
                x = c.segment.x
                at_lo = abs(c.y - lo) < eps
                at_hi = abs(c.y - hi) < eps
                if not (at_lo or at_hi):
                    return False
                y_tip = lo if at_lo else hi
                for v in self.vroads:
                    if v is c.segment:
                        continue
                    if abs(v.x - x) <= tol_conn:
                        v_lo, v_hi = sorted((v.y1, v.y2))
                        if (abs(v_lo - y_tip) <= tol_conn or abs(v_hi - y_tip) <= tol_conn or
                            (v_lo - tol_conn <= y_tip <= v_hi + tol_conn)):
                            return False
                for h in self.hroads:
                    if (abs(h.y - y_tip) <= tol_conn and
                        min(h.x1, h.x2) - tol_conn <= x <= max(h.x1, h.x2) + tol_conn):
                        return False
                return True
            return False

        # Don't early-return on missing node; we still want to try transitions.
        nx, ny = int(round(car.x)), int(round(car.y))
        key = (nx, ny)
        if key not in self.nodes and at_seg_end(car):
            # Truly isolated tip — no connections known within generous tolerance
            return -1

        segs = self._segments_at_point(car.x, car.y)
        heading = (car.hx, car.hy)
        cur_id = self._seg_id(car.segment)

        # Straight-through candidates (same orientation, not the same segment)
        if heading in [(1,0), (-1,0)]:
            straight_opts = [(t,i) for (t,i) in segs if t == "H" and (t,i) != cur_id]
        else:
            straight_opts = [(t,i) for (t,i) in segs if t == "V" and (t,i) != cur_id]

        # Perpendicular options (turns)
        turn_opts: List[Tuple[str, int]] = []
        for tag, idx in segs:
            if tag == "H" and heading in [(0,1), (0,-1)] and (tag,idx) != cur_id:
                turn_opts.append((tag, idx))
            if tag == "V" and heading in [(1,0), (-1,0)] and (tag,idx) != cur_id:
                turn_opts.append((tag, idx))


    
        # If a straight option exists, take it stochastically (your previous behavior)
        if straight_opts and len(turn_opts) == 0:
            def in_front(seg_obj) -> bool:
                if isinstance(seg_obj, HSeg):
                    lo, hi = sorted((seg_obj.x1, seg_obj.x2))
                    return (heading == (1,0) and hi >= car.x) or (heading == (-1,0) and lo <= car.x)
                if isinstance(seg_obj, VSeg):
                    lo, hi = sorted((seg_obj.y1, seg_obj.y2))
                    return (heading == (0,1) and hi >= car.y) or (heading == (0,-1) and lo <= car.y)
                return False
            candidates = [(t,i) for (t,i) in straight_opts if in_front(self._seg_object((t,i)))]
            if candidates and self.np_random.random() < TRANSITION_CHANCE:
                tag, idx = random.choice(candidates)
                new_seg = self._seg_object((tag, idx))
                car.segment = new_seg
                if isinstance(new_seg, HSeg):
                    car.y = new_seg.y
                else:
                    car.x = new_seg.x
                car.speed = self._current_speed(car)
                # heading unchanged
                return 1

        # If there is NO straight option but there ARE turn options, we FORCE a turn.
        if not straight_opts and turn_opts:
            # If we're at/near the segment end or cannot continue, force turn immediately.
            at_end_now = at_seg_end(car, eps=1.0)
            if isinstance(car.segment, HSeg):
                lo, hi = sorted((car.segment.x1, car.segment.x2))
                can_continue = (heading == (1,0) and car.x < hi - 0.5) or (heading == (-1,0) and car.x > lo + 0.5)
            else:
                lo, hi = sorted((car.segment.y1, car.segment.y2))
                can_continue = (heading == (0,1) and car.y < hi - 0.5) or (heading == (0,-1) and car.y > lo + 0.5)

            if at_end_now or not can_continue:
                return self._force_turn_from_tip(car, turn_opts)  # <- forced turn, no randomness

            # Not at the very tip yet — keep your stochastic behavior to occasionally turn early
            if self.np_random.random() < TRANSITION_CHANCE:
                return self._force_turn_from_tip(car, turn_opts)

            return 0  # keep going until we hit the tip

        # If neither straight nor turn options exist, either keep going (if possible) or dead-end
        if isinstance(car.segment, HSeg):
            lo, hi = sorted((car.segment.x1, car.segment.x2))
            if (heading == (1,0) and car.x < hi - 0.5) or (heading == (-1,0) and car.x > lo - 0.5):
                return 0
        else:
            lo, hi = sorted((car.segment.y1, car.segment.y2))
            if (heading == (0,1) and car.y < hi - 0.5) or (heading == (0,-1) and car.y > lo - 0.5):
                return 0

        return -1

    def _seg_id(self, seg) -> Optional[Tuple[str, int]]:
        if seg is None:
            return None
        if isinstance(seg, HSeg):
            for i, s in enumerate(self.hroads):
                if s is seg:
                    return ("H", i)
        if isinstance(seg, VSeg):
            for i, s in enumerate(self.vroads):
                if s is seg:
                    return ("V", i)
        return None

    def _seg_object(self, seg_id: Tuple[str, int]):
        tag, idx = seg_id
        return self.hroads[idx] if tag == "H" else self.vroads[idx]

    def _pick_segment_for_dir(self, segs: List[Tuple[str, int]], dir_vec: Tuple[int, int]):
        hx, hy = dir_vec
        for tag, idx in segs:
            if tag == "H" and hy == 0:
                return (tag, idx)
            if tag == "V" and hx == 0:
                return (tag, idx)
        return None

    def _near_roundabout_entry(self, car: Car, tol: float = 3.0) -> bool:
        # Cooldown after exiting to prevent instant re-entry
        if getattr(car, 'reentry_cd', 0) > 0:
            return False
        # Check if car is near a roundabout attach point and heading towards it
        attach_points = {
            (1,0): (self.round_x - self.round_radius - self.round_lane, self.round_y),   # from west -> heading east
            (-1,0): (self.round_x + self.round_radius + self.round_lane, self.round_y),  # from east -> heading west
            (0,1): (self.round_x, self.round_y - self.round_radius - self.round_lane),   # from north -> heading south
            (0,-1): (self.round_x, self.round_y + self.round_radius + self.round_lane),  # from south -> heading north
        }
        target = attach_points.get((car.hx, car.hy))
        if not target:
            return False
        tx, ty = target
        if abs(car.x - tx) < tol and abs(car.y - ty) < tol:
            # ensure there is a segment at this attach point
            segs = self._segments_at_point(tx, ty)
            return len(segs) > 0
        return False

    def _enter_roundabout(self, car: Car):
        # set theta so the tangent matches current heading (CCW)
        if (car.hx, car.hy) == (1,0):
            theta = math.pi  # left point
        elif (car.hx, car.hy) == (-1,0):
            theta = 0.0      # right point
        elif (car.hx, car.hy) == (0,1):
            theta = -math.pi/2  # top
        else:
            theta = math.pi/2   # bottom
        car.on_roundabout = True
        car.round_theta = theta
        car.round_dir = 1
        car.roundebout_exit = random.randint(0, 3)
        car._round_progress = 0.0
        car.x = self.round_x + self.round_radius * math.cos(theta)
        car.y = self.round_y + self.round_radius * math.sin(theta)

    # ------------------------- Observations -------------------------
    def _get_obs(self):
        bins_x = int(self.W / BIN_PER_PIXEL)
        bins_y = int(self.H / BIN_PER_PIXEL)
        obs = np.zeros((bins_x, bins_y), dtype=np.int32)
        
        for c in self.cars:
            # Convert car position to bin indices
            bin_x = int(c.x / BIN_PER_PIXEL)
            bin_y = int(c.y / BIN_PER_PIXEL)
            
            # Clamp to valid bin range
            bin_x = max(0, min(bins_x - 1, bin_x))
            bin_y = max(0, min(bins_y - 1, bin_y))
            
            # Increment count in this bin
            obs[bin_x, bin_y] += 1

        pos_idx = self.pos_index[(int(self.beam.x), int(self.beam.y))]
        dir_idx = self.dir_index[int(self.beam.dir_deg)]
        ang_idx = self.ang_index[int(self.beam.sector_deg)]
        return {
            "grid": obs,
            "beam": np.array([dir_idx, ang_idx], dtype=np.int32)
        }
    
    def _placements_75(self):
        left   = [(x, y) for x in range(0,   101, 25) for y in range(0,   101, 25)]   # 5x5
        center = [(x, y) for x in range(200, 301, 25) for y in range(200, 301, 25)]   # 5x5
        right  = [(x, y) for x in range(400, 501, 25) for y in range(0,   101, 25)]   # 5x5
        return left + center + right  # toplam 75

    def _world_to_screen(self, x, y, origin=None, scale=None):
        if origin is None:
            origin = (0, 0)
        if scale is None:
            scale = getattr(self, "ppm", 1.0)
        ox, oy = origin
        return int(ox + x * scale), int(oy + y * scale)

    def draw_runid_markers(self, surface, active_run_id=None, font=None, idx_font=None):
        import pygame
        if idx_font is None:
            idx_font = getattr(self, "_idx_font", None) or font

        pts = self._placements_75()
        color_all, color_curr = (170,170,170), (255,210,0)
        text_all, text_curr   = (220,220,220), (255,255,255)
        r_all, r_curr = 3, 5

        right_shift_px = 7   
        left_shift_px  = 7   

        for idx, (mx, my) in enumerate(pts):
            sx, sy = self._world_to_screen(mx, my)

            if idx == active_run_id:
                pygame.draw.circle(surface, color_curr, (sx, sy), r_curr, 0)
                pygame.draw.circle(surface, (40,40,40), (sx, sy), r_curr, 1)
                r = r_curr
                txt_color = text_curr
            else:
                pygame.draw.circle(surface, color_all, (sx, sy), r_all, 0)
                r = r_all
                txt_color = text_all

            if idx_font is not None:
                txt = str(idx)
                label = idx_font.render(txt, True, txt_color)
                lw, lh = label.get_size()

                x_pos = sx - lw // 2
                y_pos = sy + r + 2

                if 0 <= idx <= 4:
                    x_pos = sx - lw // 2 + right_shift_px
                elif 70 <= idx <= 74:
                    x_pos = sx - lw // 2 - left_shift_px

                surface.blit(label, (x_pos, y_pos))

    # ------------------------- Rendering -------------------------
    def render(self):
        if self.render_mode is None:
            return
        if self._screen is None:
            pygame.init()
            size = (int(self.W * self.ppm + 450), int(self.H * self.ppm + 50))
            self._screen = pygame.display.set_mode(size)
            pygame.display.set_caption("Segmented Roads + Roundabout (per-road speeds)")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("Arial", 14)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        self._screen.fill(GRID_BG)
        self._draw_roads()
        self._draw_roundabout()

        self._draw_cars()
        for beam in self.interference_beams:
            beam.render(self._screen, self.ppm, font=self._font, show_panel=False)

        self.beam.render(self._screen, self.ppm, font=self._font, show_panel=False)


        for blockage in self.beam._blockages:
            pygame.draw.rect(self._screen, (255, 0, 0), pygame.Rect(blockage["x1"] * self.ppm, blockage["y1"] * self.ppm, (blockage["x2"] - blockage["x1"]) * self.ppm, (blockage["y2"] - blockage["y1"]) * self.ppm))
        active_id = getattr(self, "active_run_id", None)  

        if not hasattr(self, "_idx_font"):
            pygame.font.init()
            self._idx_font = pygame.font.SysFont("Arial", 14, bold=True)

        if self.show_markers:
            self.draw_runid_markers(
                surface=self._screen,
                active_run_id=getattr(self, "active_run_id", None),
                font=self._font,
                idx_font=self._idx_font
            )

        self._draw_hud()

        pygame.display.flip()
        self._clock.tick(int(1.0 / self.dt))

    def _draw_roads(self):
        rw = int(14.0 * self.ppm)
        # horizontals
        for h in self.hroads:
            y = int(h.y * self.ppm)
            x1 = int(min(h.x1, h.x2) * self.ppm)
            x2 = int(max(h.x1, h.x2) * self.ppm)
            road_rect = pygame.Rect(x1, y - rw//2, x2 - x1, rw)
            pygame.draw.rect(self._screen, ASPHALT, road_rect)
            pygame.draw.rect(self._screen, LANE_MARK, road_rect, 1)
        # verticals
        for v in self.vroads:
            x = int(v.x * self.ppm)
            y1 = int(min(v.y1, v.y2) * self.ppm)
            y2 = int(max(v.y1, v.y2) * self.ppm)
            road_rect = pygame.Rect(x - rw//2, y1, rw, y2 - y1)
            pygame.draw.rect(self._screen, ASPHALT, road_rect)
            pygame.draw.rect(self._screen, LANE_MARK, road_rect, 1)

    def _draw_roundabout(self):
        cx, cy = int(self.round_x * self.ppm), int(self.round_y * self.ppm)
        R = int(self.round_radius * self.ppm)
        lane = int(self.round_lane * self.ppm)
        pygame.draw.circle(self._screen, ASPHALT, (cx, cy), R + lane, 0)
        pygame.draw.circle(self._screen, ISLAND, (cx, cy), max(2, R - lane//2))
        pygame.draw.circle(self._screen, (170, 170, 170), (cx, cy), R, 2)

    def _draw_cars(self):
        for c in self.cars:
            x = int(c.x * self.ppm)
            y = int(c.y * self.ppm)
            size = max(4, int(4 * self.ppm))
            ang = math.atan2(c.hy, c.hx)
            pts = [
                (x + int(math.cos(ang) * size), y + int(math.sin(ang) * size)),
                (x + int(math.cos(ang + 2.4) * size), y + int(math.sin(ang + 2.4) * size)),
                (x + int(math.cos(ang - 2.4) * size), y + int(math.sin(ang - 2.4) * size)),
            ]
            pygame.draw.polygon(self._screen, CAR_COLOR, pts)
            beam_names = getattr(self, "beam_names", None)  # e.g., ["N-90", "E-65", "SW-120"]
            c.render_beam_label(self._screen, self.ppm, self._font,
                                beam_names=beam_names,
                                )         


    def _draw_hud(self):
        if not self._font:
            return

        W, H = self._screen.get_size()
        PANEL_W = 450                     # fixed width
        x = W - PANEL_W                   # right-aligned (0 margin)
        y = 0
        pad = 10
        line_gap = 4
        bg_rgba = (0, 0, 0)
        fg_rgb = TEXT  # uses your global TEXT color

        # --------- gather lines ----------
        lines = []
        lines.append(f"Cars: {self.N}   dt: {self.dt:.2f}s")
        lines.append(f"Round v: {self.round_speed:.1f} m/s")

        if hasattr(self, "beam") and self.beam is not None:
            sector_deg = getattr(self.beam, "sector_deg", 90.0)
            dir_deg = (getattr(self.beam, "dir_deg", 0.0) + 360.0) % 360.0
            N = getattr(self.beam, "_N", None) or 1
            dlam = getattr(self.beam, "_dlam", None)
            dlam = dlam if dlam is not None else 0.50
            h_meas = getattr(self.beam, "_h_meas", None)
            h_meas = h_meas if h_meas is not None else sector_deg
            Rkm = getattr(self.beam, "_R_km", None)
            if Rkm is None:
                R_m = getattr(self.beam, "_R_m", 0.0)
                Rkm = (R_m or 0.0) / 1000.0
            area_km2 = getattr(self.beam, "area_km2", 0.0491)

            lines.append("")  # spacer
            lines.append(f"Sector: {sector_deg:.0f}°  |  A={area_km2:.4f} km²")
            lines.append(f"Beam Diameter R≈{Rkm:.3f} km")
            lines.append(f"Beam Direction: {dir_deg:>3.0f}°")
            lines.append(f"ULA: N={N}, d≈{dlam:.2f}λ")
            lines.append(f"Measured HPBW≈{h_meas:.1f}°")
            lines.append("")
            lines.append(f"Beam x,y: ({self.beam.x:.0f}, {self.beam.y:.0f})")
            lines.append("Keys : [1]=65°, [2]=90°, [3]=120°")
            lines.append("Key [m] : Show / Hide BS Placements")
            lines.append("←/→ : Change Beam  Direction")
            lines.append("ESC : Pause, ENTER : Continue")
        else:
            lines.append("")  # spacer
            lines.append("In-Beam (Cars): 0")

        # --------- wrap to panel width ----------
        max_text_w = PANEL_W - 2 * pad

        def wrap_line(text: str) -> list[str]:
            words = text.split(" ")
            out, cur = [], ""
            for w in words:
                test = (cur + " " + w).strip()
                if self._font.size(test)[0] <= max_text_w or not cur:
                    cur = test
                else:
                    out.append(cur)
                    cur = w
            if cur:
                out.append(cur)
            return out

        wrapped_lines: list[str] = []
        for t in lines:
            if t == "":
                wrapped_lines.append("")  # blank spacer
            else:
                wrapped_lines.extend(wrap_line(t))

        # --------- draw panel (full height) ----------
        panel = pygame.Surface((PANEL_W, H), pygame.SRCALPHA)
        pygame.draw.rect(panel, bg_rgba, (0, 0, PANEL_W, H), border_radius=0)

        yy = pad
        for t in wrapped_lines:
            if t == "":
                yy += self._font.get_height() // 2  # smaller spacer
                continue
            surf = self._font.render(t, True, fg_rgb)
            if yy + surf.get_height() > H - pad:
                break  # out of vertical space
            panel.blit(surf, (pad, yy))
            yy += surf.get_height() + line_gap

        self._screen.blit(panel, (x, y))

        y += 300
        beam_names = [i.name for i in [self.beam] + self.interference_beams] # e.g., ["N-90", "E-65", "SW-120"]
        total_strength = 0.0
        total_db_strength = 0.0
        
        for c in self.cars:
            label,db_st = c.get_stregth_label(beam_names=beam_names,show_db=True)
            strength, caps = c.compute_capacity_mbps([self.beam] + self.interference_beams)
            surf = self._font.render(f"{c.name} C:{caps:.2f} Mbps {label}", True, fg_rgb)
            self._screen.blit(surf, (x + pad, y + pad))
            y += surf.get_height() + line_gap

            total_strength += caps
            total_db_strength += db_st
        surf = self._font.render(f"Sum-rate: {total_strength:.2f} Mbps", True, fg_rgb)
        self._screen.blit(surf, (x + pad, y + pad))
        y += surf.get_height() + line_gap   


    def close(self):
        if self._screen is not None:
            pygame.display.quit()
            pygame.quit()
        self._screen = None
        self._clock = None
        self._font = None

if __name__ == "__main__":
    env = RoundaboutSegmentedEnv(render_mode="human")
    obs, info = env.reset(0)
    last_action = (0, 0)
    directions = [0, 45, 90, 135, 180, 225, 270, 315]
    pause = False

    # Keep window responsive during pause
    clock = pygame.time.Clock()

    try:
        for i in range(3000):
            # If paused, stay in a small loop that processes pygame events
            if pause:
                # Keep rendering so window doesn't freeze; wait until ENTER
                while pause:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_RETURN:   # resume on Enter
                                pause = False
                            elif event.key == pygame.K_ESCAPE: # still paused on Esc
                                pause = True

                    # Optional: keep the frame updated while paused
                    if hasattr(env, "render"):
                        env.render()
                    clock.tick(30)  # prevent busy loop
                # After unpausing, continue to normal step cycle
                continue

            # Normal running step
            obs, reward, term, trunc, info = env.step(last_action)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # Pause on Esc (only Enter will resume)
                        pause = True
                    elif event.unicode in ('1', '2', '3'):
                        sector_key = int(event.unicode)
                        last_action = (last_action[0], sector_key - 1)
                    elif event.key == pygame.K_LEFT:
                        dir_idx = (last_action[0] - 1) % len(directions)
                        last_action = (dir_idx, last_action[1])
                    elif event.key == pygame.K_RIGHT:
                        dir_idx = (last_action[0] + 1) % len(directions)
                        last_action = (dir_idx, last_action[1])
                    elif event.key == pygame.K_m:
                        env.show_markers = not env.show_markers                       
    except KeyboardInterrupt:
        pass
    finally:
        env.close()