"""Small Zeroconf broadcaster for the Sherpa UDP audio target."""

from __future__ import annotations

from dataclasses import dataclass
import socket
from typing import Callable


HOTSPOT_IP = "10.42.0.1"
HOTSPOT_PORT = 8080
SERVICE_TYPE = "_jetsonaudio._udp.local."
SERVICE_NAME = "JetsonAudioTarget._jetsonaudio._udp.local."


class BroadcastError(RuntimeError):
    pass


@dataclass(frozen=True)
class BroadcastTarget:
    ip: str
    port: int = HOTSPOT_PORT
    source: str = "hotspot"


def can_bind_ip(ip: str) -> bool:
    """Return true when the local machine owns the IPv4 address."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((ip, 0))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def get_auto_local_ip() -> str:
    """Detect the primary LAN IP without sending any packets."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("10.255.255.255", 1))
        return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def choose_broadcast_target(
    *,
    preferred_ip: str = HOTSPOT_IP,
    port: int = HOTSPOT_PORT,
    ip_checker: Callable[[str], bool] = can_bind_ip,
    auto_ip_getter: Callable[[], str] = get_auto_local_ip,
) -> BroadcastTarget:
    if ip_checker(preferred_ip):
        return BroadcastTarget(ip=preferred_ip, port=int(port), source="hotspot")
    return BroadcastTarget(ip=auto_ip_getter(), port=int(port), source="auto")


class HotspotBroadcaster:
    def __init__(
        self,
        *,
        preferred_ip: str = HOTSPOT_IP,
        port: int = HOTSPOT_PORT,
        service_name: str = SERVICE_NAME,
        service_type: str = SERVICE_TYPE,
    ) -> None:
        self.preferred_ip = preferred_ip
        self.port = int(port)
        self.service_name = service_name
        self.service_type = service_type
        self.target: BroadcastTarget | None = None
        self._zeroconf = None
        self._service_info = None

    @property
    def running(self) -> bool:
        return self._zeroconf is not None and self._service_info is not None

    def start(self) -> BroadcastTarget:
        if self.running and self.target is not None:
            return self.target
        try:
            from zeroconf import IPVersion, ServiceInfo, Zeroconf
        except Exception as exc:
            raise BroadcastError(
                "Dependency zeroconf belum tersedia. Install optional Sherpa dependency dulu."
            ) from exc

        target = choose_broadcast_target(preferred_ip=self.preferred_ip, port=self.port)
        props = {
            b"status": b"ready",
            b"type": b"audio_streaming",
            b"engine": b"sherpa-onnx",
        }
        try:
            info = ServiceInfo(
                self.service_type,
                self.service_name,
                addresses=[socket.inet_aton(target.ip)],
                port=target.port,
                properties=props,
            )
            zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
            zeroconf.register_service(info)
        except Exception as exc:
            raise BroadcastError(f"Gagal broadcast service UDP {target.ip}:{target.port}: {exc}") from exc

        self.target = target
        self._service_info = info
        self._zeroconf = zeroconf
        return target

    def stop(self) -> None:
        if self._zeroconf is None:
            return
        try:
            if self._service_info is not None:
                self._zeroconf.unregister_service(self._service_info)
        finally:
            self._zeroconf.close()
            self._zeroconf = None
            self._service_info = None

    def status_text(self) -> str:
        if not self.running or self.target is None:
            return "Broadcast: mati"
        mode = "hotspot" if self.target.source == "hotspot" else "auto-detect"
        return f"Broadcast: {self.target.ip}:{self.target.port} ({mode})"

