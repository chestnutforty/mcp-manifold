"""Tests for Manifold Markets MCP server backtesting and live tools.

Uses the manifold-sdk for all API interactions.
Run with: uv run pytest tests/test_manifold_server.py -v
"""

import asyncio
import inspect

import pytest

from server import (
    mcp,
    manifold_search_markets,
    manifold_get_market,
    _get_cutoff_timestamp_ms,
    _sample_probability_history,
    _extract_tiptap_text,
    _window_prob_history,
)


async def _get_tools_dict():
    """Helper to get tools dict from FastMCP v3."""
    tools = await mcp.list_tools()
    return {t.name: t for t in tools}


class TestBacktestingConfiguration:
    """Tests to ensure all tools have correct backtesting configuration."""

    @pytest.mark.asyncio
    async def test_all_backtest_tools_have_cutoff_date_parameter(self):
        tools = await _get_tools_dict()
        for tool_name, tool in tools.items():
            tags = getattr(tool, "tags", set()) or set()
            if "backtesting_supported" in tags:
                sig = inspect.signature(tool.fn)
                params = list(sig.parameters.keys())
                assert "cutoff_date" in params, (
                    f"Tool '{tool_name}' missing cutoff_date parameter"
                )

    @pytest.mark.asyncio
    async def test_all_backtest_tools_hide_cutoff_date_from_schema(self):
        tools = await _get_tools_dict()
        for tool_name, tool in tools.items():
            tags = getattr(tool, "tags", set()) or set()
            if "backtesting_supported" in tags:
                schema_params = tool.parameters.get("properties", {}).keys()
                assert "cutoff_date" not in schema_params, (
                    f"Tool '{tool_name}' exposes cutoff_date in schema"
                )

    @pytest.mark.asyncio
    async def test_all_backtest_tools_have_cutoff_date_default(self):
        tools = await _get_tools_dict()
        for tool_name, tool in tools.items():
            sig = inspect.signature(tool.fn)
            cutoff_param = sig.parameters.get("cutoff_date")
            if cutoff_param is not None:
                assert cutoff_param.default != inspect.Parameter.empty, (
                    f"Tool '{tool_name}' has cutoff_date but no default"
                )

    @pytest.mark.asyncio
    async def test_server_has_tools(self):
        tools = await _get_tools_dict()
        assert len(tools) > 0

    @pytest.mark.asyncio
    async def test_all_tools_have_descriptions(self):
        tools = await _get_tools_dict()
        for tool_name, tool in tools.items():
            assert tool.description, f"Tool '{tool_name}' has no description"

    @pytest.mark.asyncio
    async def test_expected_tools_exist(self):
        tools = await _get_tools_dict()
        expected = ["manifold_search_markets", "manifold_get_market"]
        for name in expected:
            assert name in tools, f"Expected tool '{name}' not found"


class TestHelperFunctions:
    """Tests for helper/utility functions."""

    def test_cutoff_timestamp_ms_basic(self):
        ts = _get_cutoff_timestamp_ms("2024-01-01")
        assert ts == 1704153600000

    def test_cutoff_timestamp_ms_end_of_year(self):
        ts = _get_cutoff_timestamp_ms("2024-12-31")
        assert ts == 1735689600000

    def test_sample_probability_history_short(self):
        points = [(1000, 0.5), (2000, 0.6), (3000, 0.7)]
        result = _sample_probability_history(points, max_points=50)
        assert result == points

    def test_sample_probability_history_long(self):
        points = [(i * 1000, 0.5 + i * 0.001) for i in range(200)]
        result = _sample_probability_history(points, max_points=20)
        assert len(result) <= 21
        assert result[0] == points[0]
        assert result[-1] == points[-1]

    def test_extract_tiptap_text_simple(self):
        node = {"type": "text", "text": "Hello world"}
        assert _extract_tiptap_text(node) == "Hello world"

    def test_extract_tiptap_text_nested(self):
        node = {
            "type": "doc",
            "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": "Hello"}]},
                {"type": "paragraph", "content": [{"type": "text", "text": "World"}]},
            ],
        }
        result = _extract_tiptap_text(node)
        assert "Hello" in result
        assert "World" in result

    def test_extract_tiptap_text_empty(self):
        assert _extract_tiptap_text({}) == ""
        assert _extract_tiptap_text(None) == ""

    def test_window_prob_history_basic(self):
        points = [
            (1000, 0.3),
            (2000, 0.4),
            (3000, 0.5),
            (4000, 0.6),
            (5000, 0.7),
        ]
        result = _window_prob_history(points, cutoff_ms=5000, window_ms=2000)
        assert len(result) >= 2
        assert result[-1][0] == 4000

    def test_window_prob_history_empty(self):
        result = _window_prob_history([], cutoff_ms=5000, window_ms=2000)
        assert result == []


class TestLiveTools:
    """Live tests that call Manifold Markets API via SDK."""

    @staticmethod
    def is_network_error(error: Exception) -> bool:
        error_str = str(error).lower()
        return any(
            ind in error_str
            for ind in ["connect", "timeout", "network", "connection", "refused", "dns", "proxy"]
        )

    @pytest.mark.asyncio
    async def test_manifold_search_markets(self):
        try:
            result = await manifold_search_markets(
                query="AI",
                limit=5,
                sort="score",
                filter_="all",
                cutoff_date="2025-01-01",
            )
            assert result is not None
            assert isinstance(result, str)
            assert "AI" in result.upper() or "No markets found" in result
            print(f"\n--- Search 'AI' result (first 500 chars) ---\n{result[:500]}")
        except Exception as e:
            if self.is_network_error(e):
                pytest.skip(f"Network error: {e}")
            raise

    @pytest.mark.asyncio
    async def test_manifold_search_markets_empty(self):
        try:
            result = await manifold_search_markets(
                query="xyznonexistent12345qwerty99999",
                limit=5,
                cutoff_date="2025-01-01",
            )
            assert "No markets found" in result
            print(f"\n--- Empty search result ---\n{result}")
        except Exception as e:
            if self.is_network_error(e):
                pytest.skip(f"Network error: {e}")
            raise

    @pytest.mark.asyncio
    async def test_manifold_search_markets_current(self):
        """Test search without backtesting (current date)."""
        try:
            result = await manifold_search_markets(
                query="election",
                limit=3,
                sort="score",
            )
            assert result is not None
            assert isinstance(result, str)
            print(f"\n--- Search 'election' (current) ---\n{result[:500]}")
        except Exception as e:
            if self.is_network_error(e):
                pytest.skip(f"Network error: {e}")
            raise

    @pytest.mark.asyncio
    async def test_manifold_get_market_by_slug(self):
        try:
            result = await manifold_get_market(
                slug="will-ai-be-able-to-generate-mass",
                include_history=False,
                include_comments=False,
                cutoff_date="2025-01-01",
            )
            assert result is not None
            assert isinstance(result, str)
            assert "Question:" in result or "No market found" in result or "did not exist" in result
            print(f"\n--- Get market by slug ---\n{result[:500]}")
        except Exception as e:
            if self.is_network_error(e):
                pytest.skip(f"Network error: {e}")
            if "404" in str(e) or "not found" in str(e).lower():
                pytest.skip(f"Market not found: {e}")
            raise

    @pytest.mark.asyncio
    async def test_manifold_get_market_with_history_and_comments(self):
        """Test getting market with full probability history and comments."""
        try:
            # First search for a known active binary market
            search_result = await manifold_search_markets(
                query="AI",
                limit=1,
                sort="liquidity",
                filter_="open",
                cutoff_date="2025-06-01",
            )
            # Extract a slug from search results
            slug = None
            for line in search_result.split("\n"):
                if line.startswith("Slug: "):
                    slug = line.split("Slug: ")[1].split(" ")[0]
                    break

            if not slug:
                pytest.skip("No market slug found in search results")

            result = await manifold_get_market(
                slug=slug,
                include_history=True,
                include_comments=True,
                cutoff_date="2025-06-01",
            )
            assert result is not None
            assert "Question:" in result
            print(f"\n--- Get market with history ---\n{result[:1000]}")
        except Exception as e:
            if self.is_network_error(e):
                pytest.skip(f"Network error: {e}")
            raise

    @pytest.mark.asyncio
    async def test_manifold_get_market_invalid_slug(self):
        try:
            result = await manifold_get_market(
                slug="invalid-slug-that-does-not-exist-12345",
                include_history=False,
                include_comments=False,
                cutoff_date="2025-01-01",
            )
            assert "No market found" in result
            print(f"\n--- Invalid slug result ---\n{result}")
        except Exception as e:
            if self.is_network_error(e):
                pytest.skip(f"Network error: {e}")
            raise

    @pytest.mark.asyncio
    async def test_manifold_get_market_no_args(self):
        result = await manifold_get_market(
            cutoff_date="2025-01-01",
        )
        assert "provide either" in result.lower()


class TestMCPConfiguration:
    """Tests for MCP server configuration."""

    def test_mcp_name(self):
        assert mcp.name == "manifold"

    def test_mcp_instructions_format(self):
        instructions = mcp.instructions
        assert len(instructions) > 0
        assert "manifold" in instructions.lower()
        assert "###" not in instructions
