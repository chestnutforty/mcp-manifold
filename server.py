"""MCP server for Manifold Markets prediction platform.

Provides tools for forecasting agents with full backtesting support.
Access prediction market data including market details, community probability
history, and comments with proper cutoff date handling.

Uses the manifold-sdk for all API interactions.
"""

import asyncio
import os
import traceback
from datetime import datetime, timezone
from functools import wraps
from typing import Annotated

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP
from manifold_sdk import AsyncClient, NotFoundError
from manifold_sdk.types import Comment, Market, ProbabilityPoint

load_dotenv()

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
MCP_NAME = "manifold"

# Proxy support - Manifold API is unauthenticated, rate-limits by IP
def _build_proxy_url() -> str | None:
    username = os.getenv("OXYLABS_USERNAME")
    password = os.getenv("OXYLABS_PASSWORD")
    if not username or not password:
        return None
    return f"http://{username}:{password}@pr.oxylabs.io:7777"

PROXY_URL = _build_proxy_url()

# SDK client - lazily initialized per event loop
_sdk_client: AsyncClient | None = None


def _get_sdk_client() -> AsyncClient:
    """Get or create the SDK async client."""
    global _sdk_client
    if _sdk_client is None:
        _sdk_client = AsyncClient(proxy=PROXY_URL, timeout=30.0)
    return _sdk_client


def send_slack_error(
    tool_name: str, error: Exception, args: tuple, kwargs: dict
) -> None:
    if not SLACK_WEBHOOK_URL:
        return
    try:
        error_message = {
            "text": f"MCP Tool Error in `{MCP_NAME}`",
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "MCP Tool Error", "emoji": True},
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*MCP Server:*\n{MCP_NAME}"},
                        {"type": "mrkdwn", "text": f"*Tool:*\n{tool_name}"},
                    ],
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Error:*\n```{str(error)[:500]}```"},
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Traceback:*\n```{traceback.format_exc()[:1000]}```"},
                },
            ],
        }
        httpx.post(SLACK_WEBHOOK_URL, json=error_message, timeout=5)
    except Exception:
        pass


def notify_on_error(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            send_slack_error(func.__name__, e, args, kwargs)
            raise
    return wrapper


mcp = FastMCP(
    name="manifold",
    instructions=r"""
Manifold Markets is a prediction market platform where users create and trade on
questions about real-world events. Supports binary (yes/no), multiple choice, and
numeric markets. Uses play money (mana) but serves as a strong signal for crowd
forecasting.

**Data available:**
- Market question text, full description, and resolution criteria
- Current and historical community probability (reconstructed from bet history)
- Community comments with reasoning and analysis
- Market creator info, close dates, and resolution details
- Multiple choice outcomes with individual probabilities
""".strip(),
)


# =============================================================================
# HELPERS (kept from original)
# =============================================================================


def _is_backtesting(cutoff_date: str) -> bool:
    today = datetime.now().strftime("%Y-%m-%d")
    return cutoff_date < today


def _get_cutoff_timestamp_ms(cutoff_date: str) -> int:
    """Convert cutoff_date to Unix ms at END of that day (start of next day)."""
    from datetime import timedelta
    cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    next_day = cutoff_dt + timedelta(days=1)
    return int(next_day.timestamp() * 1000)


def _ts_ms_to_date(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _ts_ms_to_datetime(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _dt_to_date(dt: datetime) -> str:
    """Convert a datetime to YYYY-MM-DD string."""
    return dt.strftime("%Y-%m-%d")


def _dt_to_ms(dt: datetime) -> int:
    """Convert a datetime to Unix ms."""
    return int(dt.timestamp() * 1000)


def _sample_probability_history(prob_points: list[tuple[int, float]], max_points: int = 50) -> list[tuple[int, float]]:
    """Downsample probability history to at most max_points evenly spaced entries."""
    if len(prob_points) <= max_points:
        return prob_points
    step = len(prob_points) / max_points
    sampled = []
    for i in range(max_points):
        idx = int(i * step)
        sampled.append(prob_points[idx])
    # Always include the last point
    if sampled[-1] != prob_points[-1]:
        sampled.append(prob_points[-1])
    return sampled


def _extract_tiptap_text(node: dict) -> str:
    """Extract plain text from TipTap JSON content."""
    if not isinstance(node, dict):
        return str(node) if node else ""
    texts = []
    if node.get("type") == "text":
        texts.append(node.get("text", ""))
    for child in node.get("content", []):
        texts.append(_extract_tiptap_text(child))
    return " ".join(t for t in texts if t).strip()


def _prob_points_to_tuples(points: list[ProbabilityPoint]) -> list[tuple[int, float]]:
    """Convert SDK ProbabilityPoint list to (timestamp_ms, probability) tuples."""
    return [(_dt_to_ms(p.timestamp), p.probability) for p in points]


def _window_prob_history(
    points: list[tuple[int, float]], cutoff_ms: int, window_ms: int
) -> list[tuple[int, float]]:
    """Window probability points to only include those within window before cutoff.

    Returns points in [cutoff_ms - window_ms, cutoff_ms) with one additional
    point just before the window start for trend calculation.
    """
    window_start = cutoff_ms - window_ms
    in_window = []
    last_before_window: tuple[int, float] | None = None

    for ts, prob in points:
        if ts >= cutoff_ms:
            break
        if ts >= window_start:
            in_window.append((ts, prob))
        else:
            last_before_window = (ts, prob)

    # Include one point before window for trend
    if last_before_window and (not in_window or in_window[0][0] > window_start):
        in_window.insert(0, last_before_window)
    return in_window


# =============================================================================
# FORMATTING (adapted from original to work with SDK models)
# =============================================================================


def _format_market_summary(
    market: Market, cutoff_date: str, recent_history: list[tuple[int, float]] | None = None
) -> str:
    """Format a market into a concise summary line for search results."""
    lines = []
    lines.append(f"Question: {market.question}")

    if market.slug:
        lines.append(f"Slug: {market.slug} (use with manifold_get_market)")

    if market.id:
        lines.append(f"ID: {market.id}")

    lines.append(f"Type: {market.outcome_type}")

    backtesting = _is_backtesting(cutoff_date)

    # Probability for binary markets
    # CRITICAL: API probability is CURRENT, not historical. Use bet history when backtesting.
    if market.outcome_type == "BINARY":
        if backtesting:
            if recent_history:
                current_prob = recent_history[-1][1]
                lines.append(f"Probability (as of {cutoff_date}): {current_prob * 100:.1f}%")
                # Show trend
                if len(recent_history) >= 2:
                    start_prob = recent_history[0][1]
                    delta = current_prob - start_prob
                    sign = "+" if delta >= 0 else ""
                    days_span = (recent_history[-1][0] - recent_history[0][0]) / (1000 * 86400)
                    lines.append(f"  Trend ({int(days_span)}d): {sign}{delta*100:.1f}pp ({start_prob*100:.1f}% -> {current_prob*100:.1f}%)")
            else:
                lines.append("Probability: N/A (no bet history available)")
        else:
            if market.probability is not None:
                lines.append(f"Probability: {market.probability * 100:.1f}%")

    # Multiple choice answers
    # CRITICAL: Answer probabilities are CURRENT, not historical. Hide when backtesting.
    if market.outcome_type == "MULTIPLE_CHOICE":
        answers = market.answers or []
        if answers:
            if backtesting:
                lines.append(f"Options ({len(answers)}):")
                for ans in answers[:10]:
                    lines.append(f"  {ans.text}: N/A")
                if len(answers) > 10:
                    lines.append(f"  ... and {len(answers) - 10} more options")
                lines.append("  (use manifold_get_market for historical probabilities)")
            else:
                lines.append(f"Options ({len(answers)}):")
                sorted_answers = sorted(answers, key=lambda a: a.probability or 0, reverse=True)
                for ans in sorted_answers[:10]:
                    prob = ans.probability or 0
                    lines.append(f"  {ans.text}: {prob * 100:.1f}%")
                if len(answers) > 10:
                    lines.append(f"  ... and {len(answers) - 10} more options")

    # Status
    cutoff_ms = _get_cutoff_timestamp_ms(cutoff_date)

    if market.is_resolved and market.resolution_time and _dt_to_ms(market.resolution_time) < cutoff_ms:
        lines.append(f"Status: RESOLVED ({market.resolution})")
    elif market.is_resolved and market.resolution_time and _dt_to_ms(market.resolution_time) >= cutoff_ms and backtesting:
        lines.append("Status: Open (as of cutoff)")
    else:
        if market.close_time:
            lines.append(f"Closes: {_dt_to_date(market.close_time)}")
        lines.append("Status: Open")

    # Created date
    lines.append(f"Created: {_dt_to_date(market.created_time)}")

    # Volume (only when not backtesting)
    if not backtesting:
        if market.volume:
            lines.append(f"Volume: {market.volume:,.0f} mana")

    return "\n".join(lines)


def _format_market_detail(
    market: Market,
    prob_history: list[tuple[int, float]] | None,
    comments: list[Comment] | None,
    cutoff_date: str,
) -> str:
    """Format full market details including probability history and comments."""
    lines = []
    lines.append(f"Question: {market.question}")
    lines.append(f"ID: {market.id}")
    lines.append(f"Slug: {market.slug}")
    lines.append(f"Type: {market.outcome_type}")

    cutoff_ms = _get_cutoff_timestamp_ms(cutoff_date)
    backtesting = _is_backtesting(cutoff_date)

    # Probability for binary markets
    # CRITICAL: API probability is CURRENT. When backtesting, use last bet prob instead.
    if market.outcome_type == "BINARY":
        if backtesting and prob_history:
            last_prob = prob_history[-1][1]
            lines.append(f"Probability (as of {cutoff_date}): {last_prob * 100:.1f}%")
        elif backtesting:
            lines.append("Probability: N/A (no historical bet data available)")
        else:
            if market.probability is not None:
                lines.append(f"Probability: {market.probability * 100:.1f}%")

    # Multiple choice
    # CRITICAL: Answer probabilities are CURRENT. Hide when backtesting.
    if market.outcome_type == "MULTIPLE_CHOICE":
        answers = market.answers or []
        if answers:
            if backtesting:
                lines.append(f"Options ({len(answers)}):")
                for ans in answers:
                    lines.append(f"  {ans.text}: N/A (historical probabilities not available)")
            else:
                sorted_answers = sorted(answers, key=lambda a: a.probability or 0, reverse=True)
                lines.append(f"Options ({len(answers)}):")
                for ans in sorted_answers:
                    prob = ans.probability or 0
                    lines.append(f"  {ans.text}: {prob * 100:.1f}%")

    # Resolution status
    if market.is_resolved and market.resolution_time and _dt_to_ms(market.resolution_time) < cutoff_ms:
        lines.append(f"Status: RESOLVED ({market.resolution})")
        lines.append(f"Resolved: {_dt_to_date(market.resolution_time)}")
    elif market.is_resolved and backtesting:
        lines.append("Status: Open (as of cutoff; resolution hidden)")
    else:
        if market.close_time:
            lines.append(f"Closes: {_dt_to_date(market.close_time)}")

    # Dates
    lines.append(f"Created: {_dt_to_date(market.created_time)}")

    # Volume (only when not backtesting)
    if not backtesting:
        if market.volume:
            lines.append(f"Volume: {market.volume:,.0f} mana")

    # Description
    text_desc = market.get_description_text()
    if text_desc:
        lines.append(f"\nDescription:\n{text_desc}")

    # Probability history
    if prob_history:
        sampled = _sample_probability_history(prob_history)
        if sampled:
            prices = [p for _, p in sampled]
            lines.append(f"\n=== Community Probability History ({len(sampled)} points) ===")
            lines.append(f"Range: {min(prices)*100:.1f}% - {max(prices)*100:.1f}%")
            lines.append(f"Period: {_ts_ms_to_date(sampled[0][0])} to {_ts_ms_to_date(sampled[-1][0])}")
            start_p = sampled[0][1]
            end_p = sampled[-1][1]
            trend = end_p - start_p
            sign = "+" if trend >= 0 else ""
            lines.append(f"Movement: {sign}{trend*100:.1f}% (from {start_p*100:.1f}% to {end_p*100:.1f}%)")
            lines.append("")
            for ts_ms, prob in sampled:
                lines.append(f"  {_ts_ms_to_datetime(ts_ms)}: {prob*100:.1f}%")

    # Comments
    if comments:
        lines.append(f"\n=== Community Comments ({len(comments)} most recent) ===")
        for c in comments:
            user = c.user_name
            content = c.get_text()
            if not content:
                continue
            date_str = _dt_to_date(c.created_time)
            lines.append(f"\n[{user} - {date_str}]")
            lines.append(content)

    return "\n".join(lines)


# =============================================================================
# TOOLS
# =============================================================================


@mcp.tool(
    name="manifold_search_markets",
    title="Search Manifold Markets",
    description="""Search for prediction markets on Manifold Markets by text query.
Returns market metadata including questions, probabilities, and recent trend.

When backtesting, fetches recent bet history for each binary market to show the
probability at cutoff and trend over the last N days. Results include market
slugs - use these with manifold_get_market for full probability history and comments.""",
    meta={
        "when_to_use": """
Use to find prediction markets on Manifold related to a topic. After finding
markets, use the slug from results to fetch detailed data with manifold_get_market.

Forecast: "Will AI pass the Turing test by 2030?"
-> manifold_search_markets(query="AI Turing test")

Forecast: "Will there be a US recession in 2026?"
-> manifold_search_markets(query="US recession 2026")

Forecast: "Who will win the next UK election?"
-> manifold_search_markets(query="UK election")
"""
    },
    tags={"backtesting_supported", "output:medium", "format:list"},
    exclude_args=["cutoff_date"],
)
@notify_on_error
async def manifold_search_markets(
    query: Annotated[str, "Search text (e.g., 'AI risk', 'election 2028', 'bitcoin')"],
    limit: Annotated[int | None, "Maximum results to return (default 20)"] = 20,
    sort: Annotated[str | None, "Sort order: 'score' (relevance), 'newest', 'liquidity' (default 'score')"] = "score",
    filter_: Annotated[str | None, "Filter: 'all', 'open', 'closed', 'resolved' (default 'all')"] = "all",
    history_days: Annotated[int | None, "Days of probability history to show in results (default 30)"] = 30,
    cutoff_date: Annotated[str, "YYYY-MM-DD format"] = datetime.now().strftime("%Y-%m-%d"),
) -> str:
    sdk_client = _get_sdk_client()
    cutoff_ms = _get_cutoff_timestamp_ms(cutoff_date)

    # SDK handles search + cutoff_date filtering (creation time + resolution masking)
    market_list = await sdk_client.markets.search(
        query,
        limit=limit or 20,
        sort=sort or "score",
        filter=filter_ if filter_ and filter_ != "all" else None,
        cutoff_date=cutoff_date,
    )
    filtered = market_list.items

    if not filtered:
        return f"No markets found matching '{query}'."

    # Fetch recent bet history for binary markets when backtesting
    backtesting = _is_backtesting(cutoff_date)
    history_map: dict[str, list[tuple[int, float]]] = {}
    if backtesting:
        window_ms = (history_days or 30) * 86400 * 1000
        binary_markets = [m for m in filtered if m.outcome_type == "BINARY"]
        if binary_markets:
            tasks = [
                sdk_client.bets.get_probability_history(
                    contract_id=m.id, cutoff_date=cutoff_date
                )
                for m in binary_markets
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for m, result in zip(binary_markets, results):
                if not isinstance(result, BaseException) and result.points:
                    all_points = _prob_points_to_tuples(result.points)
                    windowed = _window_prob_history(all_points, cutoff_ms, window_ms)
                    if windowed:
                        history_map[m.id] = windowed

    lines = [f"Search results for: {query}", f"Found {len(filtered)} markets", ""]
    for m in filtered:
        recent = history_map.get(m.id)
        lines.append(_format_market_summary(m, cutoff_date, recent_history=recent))
        lines.append("-" * 40)

    return "\n".join(lines)


@mcp.tool(
    name="manifold_get_market",
    title="Get Manifold Market Details",
    description="""Get full market details including community probability history and comments.

Fetches the market by slug or ID, reconstructs probability history from bet data,
and includes recent community comments with reasoning. This gives you the full
picture of how community sentiment evolved over time.

The probability history shows how the crowd prediction changed as new information
emerged - useful for understanding market dynamics and current consensus.""",
    meta={
        "when_to_use": """
Use after manifold_search_markets to get detailed data for a specific market.

Forecast: "Will GPT-5 be released in 2025?"
-> First search: manifold_search_markets(query="GPT-5 release 2025")
-> Then detail: manifold_get_market(slug="will-gpt5-be-released-in-2025")

Forecast: "What does the Manifold community think about X?"
-> Get the market with comments to see community reasoning

The slug is found in manifold_search_markets results (Slug: field) or in
Manifold URLs: manifold.markets/username/market-slug
"""
    },
    tags={"backtesting_supported", "output:high", "format:structured"},
    exclude_args=["cutoff_date"],
)
@notify_on_error
async def manifold_get_market(
    slug: Annotated[str | None, "Market slug from search results or Manifold URL"] = None,
    market_id: Annotated[str | None, "Market ID (alternative to slug)"] = None,
    include_history: Annotated[bool, "Include probability history from bets (default true)"] = True,
    include_comments: Annotated[bool, "Include community comments (default true)"] = True,
    cutoff_date: Annotated[str, "YYYY-MM-DD format"] = datetime.now().strftime("%Y-%m-%d"),
) -> str:
    if not slug and not market_id:
        return "Please provide either a slug or market_id."

    sdk_client = _get_sdk_client()
    cutoff_ms = _get_cutoff_timestamp_ms(cutoff_date)

    # Fetch market via SDK (handles 404 -> NotFoundError, resolution masking)
    try:
        market = await sdk_client.markets.get(
            slug=slug, market_id=market_id, cutoff_date=cutoff_date
        )
    except NotFoundError:
        identifier = slug or market_id
        return f"No market found with {'slug' if slug else 'ID'} '{identifier}'."

    # Check if market existed before cutoff
    created_ms = _dt_to_ms(market.created_time)
    if created_ms >= cutoff_ms:
        return f"Market did not exist as of {cutoff_date} (created {_dt_to_date(market.created_time)})."

    # Fetch probability history and comments concurrently
    prob_history: list[tuple[int, float]] | None = None
    comments: list[Comment] | None = None

    tasks = []
    fetch_history = include_history and market.outcome_type == "BINARY"
    if fetch_history:
        tasks.append(
            sdk_client.bets.get_probability_history(
                contract_id=market.id, cutoff_date=cutoff_date
            )
        )
    if include_comments:
        tasks.append(
            sdk_client.comments.list(
                contract_id=market.id, cutoff_date=cutoff_date
            )
        )

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        idx = 0

        if fetch_history:
            if not isinstance(results[idx], BaseException):
                history_result = results[idx]
                prob_history = _prob_points_to_tuples(history_result.points)
            idx += 1

        if include_comments:
            if idx < len(results) and not isinstance(results[idx], BaseException):
                comment_list = results[idx]
                # SDK returns sorted newest first; limit to 20
                comments = comment_list.items[:20] if comment_list.items else None

    return _format_market_detail(market, prob_history, comments, cutoff_date)


if __name__ == "__main__":
    mcp.run()
