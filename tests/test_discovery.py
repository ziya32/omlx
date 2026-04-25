"""Tests for omlx.discovery – OMLXBeacon mDNS service publishing."""

import socket
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.discovery import OMLX_SERVICE_TYPE, OMLXBeacon


# ---------------------------------------------------------------------------
# 1. Service-type contract
# ---------------------------------------------------------------------------

def test_omlx_service_type_matches_contract():
    assert OMLX_SERVICE_TYPE == "_myemee-omlx._tcp.local."


# ---------------------------------------------------------------------------
# 2. Constructor defaults
# ---------------------------------------------------------------------------

def test_init_defaults():
    beacon = OMLXBeacon(port=9999)
    assert beacon._port == 9999
    assert beacon._display_name == socket.gethostname()


def test_init_custom_display_name():
    beacon = OMLXBeacon(port=8080, display_name="mybox")
    assert beacon._display_name == "mybox"


# ---------------------------------------------------------------------------
# 3. start() registers the service
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_start_registers_service():
    beacon = OMLXBeacon(port=5050, display_name="test-host")

    mock_zc_instance = MagicMock()
    mock_zc_instance.async_register_service = AsyncMock()

    with patch("omlx.discovery.AsyncZeroconf", return_value=mock_zc_instance):
        await beacon.start()

    mock_zc_instance.async_register_service.assert_called_once()
    info = mock_zc_instance.async_register_service.call_args[0][0]
    assert info.type == OMLX_SERVICE_TYPE
    assert info.port == 5050


# ---------------------------------------------------------------------------
# 4. start() publishes displayName in properties
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_start_publishes_display_name():
    beacon = OMLXBeacon(port=5050, display_name="test-host")

    mock_zc_instance = MagicMock()
    mock_zc_instance.async_register_service = AsyncMock()

    with patch("omlx.discovery.AsyncZeroconf", return_value=mock_zc_instance):
        await beacon.start()

    info = mock_zc_instance.async_register_service.call_args[0][0]
    assert b"displayName" in info.properties
    assert info.properties[b"displayName"] == b"test-host"


# ---------------------------------------------------------------------------
# 5. stop() unregisters and closes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_unregisters_and_closes():
    beacon = OMLXBeacon(port=5050, display_name="test-host")

    mock_zc_instance = MagicMock()
    mock_zc_instance.async_register_service = AsyncMock()
    mock_zc_instance.async_unregister_service = AsyncMock()
    mock_zc_instance.async_close = AsyncMock()

    with patch("omlx.discovery.AsyncZeroconf", return_value=mock_zc_instance):
        await beacon.start()

    await beacon.stop()

    mock_zc_instance.async_unregister_service.assert_called_once()
    mock_zc_instance.async_close.assert_called_once()
    assert beacon._zc is None
    assert beacon._info is None


# ---------------------------------------------------------------------------
# 6. stop() is idempotent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_idempotent():
    beacon = OMLXBeacon(port=5050)
    # Never started – _zc and _info are None.
    await beacon.stop()  # should not raise
