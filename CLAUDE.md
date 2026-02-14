# MCP Server: mcp-manifold

## Overview
MCP server for Manifold Markets prediction platform. Provides search and detailed
market data including community probability history and comments with full
backtesting support.

## Key Files
- `server.py` - Main MCP server with tool definitions
- `rate_limits.json` - Rate limiting configuration (8 req/s)
- `pyproject.toml` - Dependencies

## Running the Server
```bash
uv sync
uv run fastmcp run server.py
```

## Available Tools

- **manifold_search_markets** - Search markets by query with filtering (supports backtesting)
- **manifold_get_market** - Get full market details with probability history and comments (supports backtesting)

## Backtesting

Both tools support `cutoff_date` for backtesting:
- Markets created after cutoff are hidden
- Resolution is hidden for markets resolved after cutoff
- Probability history is filtered to before cutoff
- Comments are filtered to before cutoff
- Volume is only shown when NOT backtesting
