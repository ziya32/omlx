"""mDNS service discovery for omlx server."""

import logging
import socket

from zeroconf import ServiceInfo
from zeroconf.asyncio import AsyncZeroconf

# IMPORTANT: must match nanobot's myemee/gateway/discovery.py:OMLX_SERVICE_TYPE
OMLX_SERVICE_TYPE = "_myemee-omlx._tcp.local."

log = logging.getLogger(__name__)


class OMLXBeacon:
    """Publishes omlx as a Bonjour/mDNS service for LAN discovery."""

    def __init__(self, port: int, display_name: str = "") -> None:
        self._port = port
        self._display_name = display_name or socket.gethostname()
        self._zc: AsyncZeroconf | None = None
        self._info: ServiceInfo | None = None

    async def start(self) -> None:
        hostname = socket.gethostname()
        self._info = ServiceInfo(
            OMLX_SERVICE_TYPE,
            name=f"omlx-{hostname}.{OMLX_SERVICE_TYPE}",
            port=self._port,
            properties={
                b"displayName": self._display_name.encode(),
                b"version": _get_version().encode(),
            },
            server=f"{hostname}.local.",
        )
        self._zc = AsyncZeroconf()
        await self._zc.async_register_service(self._info)
        log.info("Published %s on port %d", OMLX_SERVICE_TYPE, self._port)

    async def stop(self) -> None:
        if self._zc and self._info:
            await self._zc.async_unregister_service(self._info)
            await self._zc.async_close()
            self._zc = None
            self._info = None


def _get_version() -> str:
    try:
        from omlx import __version__

        return __version__
    except Exception:
        return "unknown"
