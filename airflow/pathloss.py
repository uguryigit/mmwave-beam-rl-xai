
import math
import numpy as np

class CIPathloss():
    """
    Close-In (CI) path-loss model with 1 m reference:
        PL(d) = FSPL(1 m) + 10 * n * log10(d / 1 m) + X_sigma

    - Frequency in MHz (f_GHz = freq_mhz / 1000)
    - Distance in meters (clamped to >= 1 m)
    - n = n_los for LOS, n_nlos for NLOS (select via .get(..., los=True/False))
    - Optional log-normal shadowing (sigma_los / sigma_nlos) in dB

    Parameters
    ----------
    freq_mhz : float
        Carrier frequency in MHz (e.g., 26000.0 for 26 GHz).
    n_los : float
        Path-loss exponent for LOS (default 2.0).
    n_nlos : float
        Path-loss exponent for NLOS (default 3.4).
    sigma_los : float
        Shadowing std-dev in dB for LOS (default 0.0 i.e., disabled).
    sigma_nlos : float
        Shadowing std-dev in dB for NLOS (default 0.0).
    seed : int | None
        RNG seed for deterministic shadowing (optional).
    """

    def __init__(self,
                 freq_mhz: float,
                 n_los: float = 2.0,
                 n_nlos: float = 3.4,
                 sigma_los: float = 0.0,
                 sigma_nlos: float = 0.0,
                 seed: int | None = None):
        self.freq_mhz = float(freq_mhz)
        self.n_los = float(n_los)
        self.n_nlos = float(n_nlos)
        self.sigma_los = float(sigma_los)
        self.sigma_nlos = float(sigma_nlos)
        self._rng = np.random.default_rng(seed)

        # FSPL at 1 m: 32.4 + 20*log10(f_GHz)
        f_ghz = self.freq_mhz / 1000.0
        self._fspl_1m_db = 32.4 + 20.0 * math.log10(max(f_ghz, 1e-9))

    def set_shadowing(self, sigma_los: float | None = None, sigma_nlos: float | None = None):
        if sigma_los is not None:
            self.sigma_los = float(sigma_los)
        if sigma_nlos is not None:
            self.sigma_nlos = float(sigma_nlos)

    def get(self, distance: float, los: bool | None = True) -> float:
        """
        Compute PL(d) in dB.

        Parameters
        ----------
        distance : float
            TX–RX separation in meters (clamped to >= 1 m).
        los : bool | None
            If True use n_los; if False use n_nlos.
            If None, defaults to LOS.

        Returns
        -------
        float : path loss in dB
        """
        d = max(1.0, float(distance))
        n = self.n_los if (los is None or los) else self.n_nlos
        # CI model: PL(d) = FSPL(d0=1m) + 10*n*log10(d/d0) + Xσ
        # (matches paper Eq. (1) and 3GPP TR 38.901 UMi mmWave references)
        pl = self._fspl_1m_db + 10.0 * n * math.log10(d)

        # shadowing
        sigma = self.sigma_los if (los is None or los) else self.sigma_nlos
        if sigma > 0.0:
            pl += float(self._rng.normal(0.0, sigma))
        return pl

    def __repr__(self) -> str:
        return (f"CIPathloss(freq_mhz={self.freq_mhz}, n_los={self.n_los}, "
                f"n_nlos={self.n_nlos}, sigma_los={self.sigma_los}, sigma_nlos={self.sigma_nlos})")
