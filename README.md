# mcp-manifold

MCP server for Manifold Markets prediction platform.

## Quick Start

```bash
uv sync
uv run python server.py
```

## Tools

- `manifold_search_markets` - Search markets by keyword with sort/filter
- `manifold_get_market` - Get market details with probability history and comments

## Key Details

- **No auth needed** - public API
- **Rate limits** - 500 requests/minute/IP
- **Timestamps** - Unix milliseconds
- **Backtesting** - Full support via cutoff_date parameter

## Testing

```bash
uv sync --extra test
uv run pytest tests/ -v
```
