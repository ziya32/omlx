# SPDX-License-Identifier: Apache-2.0
"""
MCP (Model Context Protocol) API routes.

This module provides FastAPI routes for MCP tool management:
- GET /v1/mcp/tools - List available MCP tools
- GET /v1/mcp/servers - List MCP server status
- POST /v1/mcp/execute - Execute an MCP tool
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .openai_models import (
    MCPExecuteRequest,
    MCPExecuteResponse,
    MCPServerInfo,
    MCPServersResponse,
    MCPToolInfo,
    MCPToolsResponse,
)

router = APIRouter(prefix="/v1/mcp", tags=["mcp"])

# Auth dependency — wired by server.py via set_auth_dependency()
_auth_dependency = None
_security = HTTPBearer(auto_error=False)


def set_auth_dependency(dep):
    """Set the auth dependency function (called by server.py after verify_api_key is defined)."""
    global _auth_dependency
    _auth_dependency = dep


async def _verify_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> bool:
    """Forward to server's verify_api_key if wired, otherwise reject."""
    if _auth_dependency is None:
        raise HTTPException(status_code=401, detail="Auth not configured")
    return await _auth_dependency(request=request, credentials=credentials)


# Callback function to get MCP manager (set by server.py)
_get_mcp_manager = None


def set_mcp_manager_getter(getter):
    """
    Set the callback function to get MCP manager.

    Args:
        getter: A callable that returns the MCP manager instance or None
    """
    global _get_mcp_manager
    _get_mcp_manager = getter


def _get_manager():
    """Get the MCP manager instance."""
    if _get_mcp_manager is None:
        return None
    return _get_mcp_manager()


@router.get("/tools")
async def list_mcp_tools(_: bool = Depends(_verify_auth)) -> MCPToolsResponse:
    """List all available MCP tools."""
    manager = _get_manager()
    if manager is None:
        return MCPToolsResponse(tools=[], count=0)

    tools = []
    for tool in manager.get_all_tools():
        tools.append(MCPToolInfo(
            name=tool.full_name,
            description=tool.description,
            server=tool.server_name,
            parameters=tool.input_schema,
        ))

    return MCPToolsResponse(tools=tools, count=len(tools))


@router.get("/servers")
async def list_mcp_servers(_: bool = Depends(_verify_auth)) -> MCPServersResponse:
    """Get status of all MCP servers."""
    manager = _get_manager()
    if manager is None:
        return MCPServersResponse(servers=[])

    servers = []
    for status in manager.get_server_status():
        servers.append(MCPServerInfo(
            name=status.name,
            state=status.state.value,
            transport=status.transport.value,
            tools_count=status.tools_count,
            error=status.error,
        ))

    return MCPServersResponse(servers=servers)


@router.post("/execute")
async def execute_mcp_tool(request: MCPExecuteRequest, _: bool = Depends(_verify_auth)) -> MCPExecuteResponse:
    """Execute an MCP tool."""
    manager = _get_manager()
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="MCP not configured. Start server with --mcp-config"
        )

    result = await manager.execute_tool(
        request.tool_name,
        request.arguments,
    )

    return MCPExecuteResponse(
        tool_name=result.tool_name,
        content=result.content,
        is_error=result.is_error,
        error_message=result.error_message,
    )
