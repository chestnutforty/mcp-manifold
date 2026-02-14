"""Tests for Manifold Markets MCP server backtesting and live tools."""

import inspect

import pytest

from server import (
    mcp,
    _get_cutoff_timestamp_ms,
    _sample_probability_history,
    _extract_tiptap_text,
)


@pytest.fixture(autouse=True)
async def reset_clients():
    """Reset global HTTP clients before each test."""
    import server

    if server._async_client is not None:
        await server._async_client.aclose()
        server._async_client = None
    yield
    if server._async_client is not None:
        await server._async_client.aclose()
        server._async_client = None


class TestBacktestingConfiguration:
    """Tests to ensure all tools have correct backtesting configuration."""

    def get_all_tools(self):
        return mcp._tool_manager._tools

    def test_all_backtest_tools_have_cutoff_date_parameter(self):
        tools = self.get_all_tools()
        for tool_name, tool in tools.items():
            tags = getattr(tool, "tags", set()) or set()
            if "backtesting_supported" in tags:
                sig = inspect.signature(tool.fn)
                params = list(sig.parameters.keys())
                assert "cutoff_date" in params, (
                    f"Tool '{tool_name}' missing cutoff_date parameter"
                )

    def test_all_backtest_tools_hide_cutoff_date_from_schema(self):
        tools = self.get_all_tools()
        for tool_name, tool in tools.items():
            tags = getattr(tool, "tags", set()) or set()
            if "backtesting_supported" in tags:
                schema_params = tool.parameters.get("properties", {}).keys()
                assert "cutoff_date" not in schema_params, (
                    f"Tool '{tool_name}' exposes cutoff_date in schema"
                )

    def test_all_backtest_tools_have_cutoff_date_default(self):
        tools = self.get_all_tools()
        for tool_name, tool in tools.items():
            sig = inspect.signature(tool.fn)
            cutoff_param = sig.parameters.get("cutoff_date")
            if cutoff_param is not None:
                assert cutoff_param.default != inspect.Parameter.empty, (
                    f"Tool '{tool_name}' has cutoff_date but no default"
                )

    def test_server_has_tools(self):
        tools = self.get_all_tools()
        assert len(tools) > 0

    def test_all_tools_have_descriptions(self):
        tools = self.get_all_tools()
        for tool_name, tool in tools.items():
            assert tool.description, f"Tool '{tool_name}' has no description"

    def test_expected_tools_exist(self):
        tools = self.get_all_tools()
        expected = ["manifold_search_markets", "manifold_get_market"]
        for name in expected:
            assert name in tools, f"Expected tool '{name}' not found"


class TestHelperFunctions:
    """Tests for helper/utility functions."""

    def test_cutoff_timestamp_ms_basic(self):
        ts = _get_cutoff_timestamp_ms("2024-01-01")
        # Should be start of 2024-01-02 in ms (1704067200 + 86400 = 1704153600)
        assert ts == 1704153600000

    def test_cutoff_timestamp_ms_end_of_year(self):
        ts = _get_cutoff_timestamp_ms("2024-12-31")
        # Should be start of 2025-01-01 in ms
        assert ts == 1735689600000

    def test_sample_probability_history_short(self):
        points = [(1000, 0.5), (2000, 0.6), (3000, 0.7)]
        result = _sample_probability_history(points, max_points=50)
        assert result == points  # No sampling needed

    def test_sample_probability_history_long(self):
        points = [(i * 1000, 0.5 + i * 0.001) for i in range(200)]
        result = _sample_probability_history(points, max_points=20)
        assert len(result) <= 21  # max_points + possibly the last point
        # First point should be included
        assert result[0] == points[0]
        # Last point should be included
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


class TestLiveTools:
    """Live tests that call Manifold Markets API."""

    def get_all_tools(self):
        return mcp._tool_manager._tools

    @staticmethod
    def is_network_error(error: Exception) -> bool:
        error_str = str(error).lower()
        return any(
            ind in error_str
            for ind in ["connect", "timeout", "network", "connection", "refused", "dns"]
        )

    @pytest.mark.asyncio
    async def test_manifold_search_markets(self):
        tools = self.get_all_tools()
        tool = tools["manifold_search_markets"]
        try:
            result = await tool.fn(
                query="AI",
                limit=5,
                sort="score",
                filter_="all",
                cutoff_date="2025-01-01",
            )
            assert result is not None
            assert isinstance(result, str)
            assert "AI" in result.upper() or "No markets found" in result
        except Exception as e:
            if self.is_network_error(e):
                pytest.skip(f"Network error: {e}")
            raise

    @pytest.mark.asyncio
    async def test_manifold_search_markets_empty(self):
        tools = self.get_all_tools()
        tool = tools["manifold_search_markets"]
        try:
            result = await tool.fn(
                query="xyznonexistent12345qwerty99999",
                limit=5,
                cutoff_date="2025-01-01",
            )
            assert "No markets found" in result
        except Exception as e:
            if self.is_network_error(e):
                pytest.skip(f"Network error: {e}")
            raise

    @pytest.mark.asyncio
    async def test_manifold_get_market_by_slug(self):
        tools = self.get_all_tools()
        tool = tools["manifold_get_market"]
        try:
            # Use a well-known market
            result = await tool.fn(
                slug="will-ai-be-able-to-generate-mass",
                include_history=False,
                include_comments=False,
                cutoff_date="2025-01-01",
            )
            assert result is not None
            assert isinstance(result, str)
            # Should contain market info or not found
            assert "Question:" in result or "No market found" in result or "did not exist" in result
        except Exception as e:
            if self.is_network_error(e):
                pytest.skip(f"Network error: {e}")
            if "404" in str(e):
                pytest.skip(f"Market not found: {e}")
            raise

    @pytest.mark.asyncio
    async def test_manifold_get_market_invalid_slug(self):
        tools = self.get_all_tools()
        tool = tools["manifold_get_market"]
        try:
            result = await tool.fn(
                slug="invalid-slug-that-does-not-exist-12345",
                include_history=False,
                include_comments=False,
                cutoff_date="2025-01-01",
            )
            assert result is not None
        except Exception as e:
            if self.is_network_error(e):
                pytest.skip(f"Network error: {e}")
            # 404 is expected
            if "404" in str(e):
                pass
            else:
                raise

    @pytest.mark.asyncio
    async def test_manifold_get_market_no_args(self):
        tools = self.get_all_tools()
        tool = tools["manifold_get_market"]
        result = await tool.fn(
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
