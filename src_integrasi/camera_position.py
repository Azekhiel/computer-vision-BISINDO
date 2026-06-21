"""Acuan posisi kamera/user dari bahu MediaPipe, untuk publish topic CAMPOS.

Posisi diukur dari titik tengah bahu (rata-rata x/y bahu kiri & kanan, ternormalisasi
0..1 terhadap frame) plus lebar bahu (jarak x). Saat runtime user menekan tombol
kalibrasi untuk menyimpan posisi "pas" sebagai acuan ke camera_position.json, lengkap
dengan acceptable error (toleransi). evaluate() membandingkan posisi sekarang dengan
acuan dan mengembalikan OK / UP / DOWN / LEFT / RIGHT.

Koordinat gambar: x naik ke kanan, y naik ke bawah.
- Bahu terlalu ke bawah (mid_y > acuan) -> user perlu naik -> "UP".
- Bahu terlalu ke atas (mid_y < acuan) -> "DOWN".
- Arah horizontal mengikuti flip_horizontal (set true kalau preview mirror/selfie).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "camera_position.json"

# Toleransi default (fraksi frame). Tidak harus pas banget supaya gampang dapat OK.
DEFAULT_TOL_X = 0.10
DEFAULT_TOL_Y = 0.10


@dataclass
class CameraPositionConfig:
    ref_mid_x: float = 0.5
    ref_mid_y: float = 0.45
    ref_width: float = 0.28
    tol_x: float = DEFAULT_TOL_X
    tol_y: float = DEFAULT_TOL_Y
    flip_horizontal: bool = False
    calibrated: bool = False

    @classmethod
    def load(cls, path: Path | None = None) -> "CameraPositionConfig":
        target = Path(path) if path is not None else CONFIG_PATH
        if not target.exists():
            return cls()
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        known = {k: data[k] for k in cls().__dict__ if k in data}
        return cls(**known)

    def save(self, path: Path | None = None) -> None:
        target = Path(path) if path is not None else CONFIG_PATH
        target.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def calibrate(self, mid_x: float, mid_y: float, width: float) -> None:
        self.ref_mid_x = float(mid_x)
        self.ref_mid_y = float(mid_y)
        self.ref_width = float(width)
        self.calibrated = True

    def evaluate(self, mid_x: float | None, mid_y: float | None) -> str | None:
        """Bandingkan posisi sekarang dengan acuan.

        Return None kalau bahu tidak terdeteksi (mid_x/mid_y None/NaN) -> pemanggil
        bisa pilih tidak publish CAMPOS.
        """
        if mid_x is None or mid_y is None:
            return None
        if isinstance(mid_x, float) and math.isnan(mid_x):
            return None
        if isinstance(mid_y, float) and math.isnan(mid_y):
            return None

        dx = float(mid_x) - self.ref_mid_x
        dy = float(mid_y) - self.ref_mid_y
        tol_x = max(self.tol_x, 1e-4)
        tol_y = max(self.tol_y, 1e-4)

        within_x = abs(dx) <= tol_x
        within_y = abs(dy) <= tol_y
        if within_x and within_y:
            return "OK"

        # Pilih sumbu dengan deviasi (ternormalisasi terhadap toleransi) terbesar.
        nx = abs(dx) / tol_x
        ny = abs(dy) / tol_y
        if ny >= nx:
            return "UP" if dy > 0 else "DOWN"
        # Horizontal: too-far-right -> geser kiri (LEFT), kecuali preview mirror.
        too_right = dx > 0
        if self.flip_horizontal:
            too_right = not too_right
        return "LEFT" if too_right else "RIGHT"


def shoulder_midpoint(shoulders) -> tuple[float | None, float | None, float | None]:
    """Hitung (mid_x, mid_y, width) dari array bahu (2,4) [L,R]x[x,y,z,vis].

    Mengembalikan (None, None, None) kalau data tidak valid.
    """
    try:
        lx, ly = float(shoulders[0][0]), float(shoulders[0][1])
        rx, ry = float(shoulders[1][0]), float(shoulders[1][1])
    except (TypeError, IndexError, ValueError):
        return None, None, None
    for v in (lx, ly, rx, ry):
        if math.isnan(v):
            return None, None, None
    mid_x = (lx + rx) * 0.5
    mid_y = (ly + ry) * 0.5
    width = abs(lx - rx)
    return mid_x, mid_y, width
