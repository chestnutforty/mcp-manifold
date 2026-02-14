"""MCP server for Manifold Markets prediction platform.

Provides tools for forecasting agents with full backtesting support.
Access prediction market data including market details, community probability
history, and comments with proper cutoff date handling.
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

load_dotenv()

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
MCP_NAME = "manifold"

BASE_URL = "https://api.manifold.markets"

# Concurrency control
MAX_CONCURRENCY = 10
_async_client: httpx.AsyncClient | None = None
_semaphore: asyncio.Semaphore | None = None
_semaphore_loop: asyncio.AbstractEventLoop | None = None


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


def get_async_client() -> httpx.AsyncClient:
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(timeout=30.0)
    return _async_client


def get_semaphore() -> asyncio.Semaphore:
    global _semaphore, _semaphore_loop
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None
    if _semaphore is None or _semaphore_loop is not current_loop:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        _semaphore_loop = current_loop
    return _semaphore


async def fetch_with_retry(
    client: httpx.AsyncClient,
    url: str,
    params: dict | None = None,
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> httpx.Response:
    last_error = None
    for attempt in range(max_retries):
        try:
            async with get_semaphore():
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
    raise last_error


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


async def _fetch_bets(
    client: httpx.AsyncClient, contract_id: str, cutoff_ms: int
) -> list[tuple[int, float]]:
    """Fetch full bet history and extract (timestamp_ms, probAfter) pairs before cutoff."""
    prob_points = []
    after_id = None

    while True:
        params = {"contractId": contract_id, "limit": 1000, "order": "asc"}
        if after_id:
            params["after"] = after_id

        response = await fetch_with_retry(client, f"{BASE_URL}/v0/bets", params=params)
        bets = response.json()
        if not bets or not isinstance(bets, list):
            break

        for bet in bets:
            ts = bet.get("createdTime", 0)
            if ts >= cutoff_ms:
                return prob_points
            prob_after = bet.get("probAfter")
            if prob_after is not None:
                prob_points.append((ts, prob_after))

        if len(bets) < 1000:
            break
        after_id = bets[-1].get("id")

    return prob_points


async def _fetch_bets_for_search(
    client: httpx.AsyncClient, contract_id: str, cutoff_ms: int, window_ms: int
) -> list[tuple[int, float]]:
    """Fetch bet history for search results - returns points within window before cutoff.

    Uses ascending order and pages through all bets up to cutoff, keeping only
    points within the window. Stops at cutoff or after max_pages to limit cost.
    """
    prob_points = []
    window_start = cutoff_ms - window_ms
    last_point_before_window: tuple[int, float] | None = None
    after_id = None

    for _ in range(10):  # max 10 pages = 10k bets
        params = {"contractId": contract_id, "limit": 1000, "order": "asc"}
        if after_id:
            params["after"] = after_id

        try:
            response = await fetch_with_retry(client, f"{BASE_URL}/v0/bets", params=params)
        except Exception:
            break
        bets = response.json()
        if not bets or not isinstance(bets, list):
            break

        for bet in bets:
            ts = bet.get("createdTime", 0)
            if ts >= cutoff_ms:
                # Past cutoff, return what we have
                # Include one point before window for trend calculation
                if last_point_before_window and (not prob_points or prob_points[0][0] > window_start):
                    prob_points.insert(0, last_point_before_window)
                return prob_points
            prob_after = bet.get("probAfter")
            if prob_after is not None:
                if ts >= window_start:
                    prob_points.append((ts, prob_after))
                else:
                    last_point_before_window = (ts, prob_after)

        if len(bets) < 1000:
            break
        after_id = bets[-1].get("id")

    # Include one point before window for trend
    if last_point_before_window and (not prob_points or prob_points[0][0] > window_start):
        prob_points.insert(0, last_point_before_window)
    return prob_points


async def _fetch_comments(
    client: httpx.AsyncClient, contract_id: str, cutoff_ms: int
) -> list[dict]:
    """Fetch comments for a market before cutoff date."""
    params = {"contractId": contract_id}
    response = await fetch_with_retry(client, f"{BASE_URL}/v0/comments", params=params)
    all_comments = response.json() or []

    filtered = []
    for c in all_comments:
        ts = c.get("createdTime", 0)
        if ts < cutoff_ms:
            filtered.append(c)

    # Sort by time, return most recent 20
    filtered.sort(key=lambda x: x.get("createdTime", 0), reverse=True)
    return filtered[:20]


def _format_market_summary(
    market: dict, cutoff_date: str, recent_history: list[tuple[int, float]] | None = None
) -> str:
    """Format a market into a concise summary line for search results."""
    lines = []
    question = market.get("question", "Unknown")
    lines.append(f"Question: {question}")

    slug = market.get("slug", "")
    if slug:
        lines.append(f"Slug: {slug} (use with manifold_get_market)")

    market_id = market.get("id", "")
    if market_id:
        lines.append(f"ID: {market_id}")

    outcome_type = market.get("outcomeType", "BINARY")
    lines.append(f"Type: {outcome_type}")

    backtesting = _is_backtesting(cutoff_date)

    # Probability for binary markets
    # CRITICAL: API probability is CURRENT, not historical. Use bet history when backtesting.
    if outcome_type == "BINARY":
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
            prob = market.get("probability")
            if prob is not None:
                lines.append(f"Probability: {prob * 100:.1f}%")

    # Multiple choice answers
    # CRITICAL: Answer probabilities are CURRENT, not historical. Hide when backtesting.
    if outcome_type == "MULTIPLE_CHOICE":
        answers = market.get("answers", [])
        if answers:
            if backtesting:
                lines.append(f"Options ({len(answers)}):")
                for ans in answers[:10]:
                    text = ans.get("text", "Unknown")
                    lines.append(f"  {text}: N/A")
                if len(answers) > 10:
                    lines.append(f"  ... and {len(answers) - 10} more options")
                lines.append("  (use manifold_get_market for historical probabilities)")
            else:
                lines.append(f"Options ({len(answers)}):")
                sorted_answers = sorted(answers, key=lambda a: a.get("probability", 0), reverse=True)
                for ans in sorted_answers[:10]:
                    prob = ans.get("probability", 0)
                    text = ans.get("text", "Unknown")
                    lines.append(f"  {text}: {prob * 100:.1f}%")
                if len(answers) > 10:
                    lines.append(f"  ... and {len(answers) - 10} more options")

    # Status
    is_resolved = market.get("isResolved", False)
    cutoff_ms = _get_cutoff_timestamp_ms(cutoff_date)
    resolution_time = market.get("resolutionTime")

    if is_resolved and resolution_time and resolution_time < cutoff_ms:
        resolution = market.get("resolution", "")
        lines.append(f"Status: RESOLVED ({resolution})")
    elif is_resolved and resolution_time and resolution_time >= cutoff_ms and _is_backtesting(cutoff_date):
        lines.append("Status: Open (as of cutoff)")
    else:
        close_time = market.get("closeTime")
        if close_time:
            lines.append(f"Closes: {_ts_ms_to_date(close_time)}")
        lines.append("Status: Open")

    # Created date
    created = market.get("createdTime")
    if created:
        lines.append(f"Created: {_ts_ms_to_date(created)}")

    # Volume (only when not backtesting)
    if not _is_backtesting(cutoff_date):
        volume = market.get("volume")
        if volume:
            lines.append(f"Volume: {volume:,.0f} mana")

    return "\n".join(lines)


def _format_market_detail(
    market: dict,
    prob_history: list[tuple[int, float]] | None,
    comments: list[dict] | None,
    cutoff_date: str,
) -> str:
    """Format full market details including probability history and comments."""
    lines = []
    question = market.get("question", "Unknown")
    lines.append(f"Question: {question}")

    slug = market.get("slug", "")
    market_id = market.get("id", "")
    outcome_type = market.get("outcomeType", "BINARY")
    lines.append(f"ID: {market_id}")
    lines.append(f"Slug: {slug}")
    lines.append(f"Type: {outcome_type}")

    cutoff_ms = _get_cutoff_timestamp_ms(cutoff_date)
    backtesting = _is_backtesting(cutoff_date)

    # Probability for binary markets
    # CRITICAL: API probability is CURRENT. When backtesting, use last bet prob instead.
    if outcome_type == "BINARY":
        if backtesting and prob_history:
            # Use last probability from bet history (already filtered to before cutoff)
            last_prob = prob_history[-1][1]
            lines.append(f"Probability (as of {cutoff_date}): {last_prob * 100:.1f}%")
        elif backtesting:
            lines.append("Probability: N/A (no historical bet data available)")
        else:
            prob = market.get("probability")
            if prob is not None:
                lines.append(f"Probability: {prob * 100:.1f}%")

    # Multiple choice
    # CRITICAL: Answer probabilities are CURRENT. Hide when backtesting.
    if outcome_type == "MULTIPLE_CHOICE":
        answers = market.get("answers", [])
        if answers:
            if backtesting:
                lines.append(f"Options ({len(answers)}):")
                for ans in answers:
                    text = ans.get("text", "Unknown")
                    lines.append(f"  {text}: N/A (historical probabilities not available)")
            else:
                sorted_answers = sorted(answers, key=lambda a: a.get("probability", 0), reverse=True)
                lines.append(f"Options ({len(answers)}):")
                for ans in sorted_answers:
                    prob = ans.get("probability", 0)
                    text = ans.get("text", "Unknown")
                    lines.append(f"  {text}: {prob * 100:.1f}%")

    # Resolution status
    is_resolved = market.get("isResolved", False)
    resolution_time = market.get("resolutionTime")

    if is_resolved and resolution_time and resolution_time < cutoff_ms:
        resolution = market.get("resolution", "")
        lines.append(f"Status: RESOLVED ({resolution})")
        lines.append(f"Resolved: {_ts_ms_to_date(resolution_time)}")
    elif is_resolved and _is_backtesting(cutoff_date):
        lines.append("Status: Open (as of cutoff; resolution hidden)")
    else:
        close_time = market.get("closeTime")
        if close_time:
            lines.append(f"Closes: {_ts_ms_to_date(close_time)}")

    # Dates
    created = market.get("createdTime")
    if created:
        lines.append(f"Created: {_ts_ms_to_date(created)}")

    # Volume (only when not backtesting)
    if not _is_backtesting(cutoff_date):
        volume = market.get("volume")
        if volume:
            lines.append(f"Volume: {volume:,.0f} mana")

    # Description
    text_desc = market.get("textDescription", "")
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
            user = c.get("userName", "Unknown")
            ts = c.get("createdTime", 0)
            content = c.get("content", c.get("text", ""))
            # Extract text from TipTap JSON content if needed
            if isinstance(content, dict):
                content = _extract_tiptap_text(content)
            if not content:
                continue
            date_str = _ts_ms_to_date(ts) if ts else ""
            lines.append(f"\n[{user} - {date_str}]")
            lines.append(content)

    return "\n".join(lines)


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
    client = get_async_client()
    cutoff_ms = _get_cutoff_timestamp_ms(cutoff_date)

    params = {
        "term": query,
        "limit": min(limit or 20, 50),
        "sort": sort or "score",
    }
    if filter_ and filter_ != "all":
        params["filter"] = filter_

    response = await fetch_with_retry(client, f"{BASE_URL}/v0/search-markets", params=params)
    markets = response.json() or []

    # Filter: only markets that existed before cutoff
    filtered = []
    for m in markets:
        created = m.get("createdTime", 0)
        if created < cutoff_ms:
            filtered.append(m)

    # Hide resolution for markets resolved after cutoff (backtesting)
    backtesting = _is_backtesting(cutoff_date)
    if backtesting:
        for m in filtered:
            resolution_time = m.get("resolutionTime")
            if resolution_time and resolution_time >= cutoff_ms:
                m.pop("resolution", None)
                m.pop("resolutionTime", None)
                m["isResolved"] = False

    if not filtered:
        return f"No markets found matching '{query}'."

    # Fetch recent bet history for binary markets when backtesting
    history_map: dict[str, list[tuple[int, float]]] = {}
    if backtesting:
        window_ms = (history_days or 30) * 86400 * 1000
        binary_markets = [m for m in filtered if m.get("outcomeType") == "BINARY"]
        if binary_markets:
            tasks = [
                _fetch_bets_for_search(client, m["id"], cutoff_ms, window_ms)
                for m in binary_markets
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for m, result in zip(binary_markets, results):
                if not isinstance(result, BaseException) and result:
                    history_map[m["id"]] = result

    lines = [f"Search results for: {query}", f"Found {len(filtered)} markets", ""]
    for m in filtered:
        recent = history_map.get(m.get("id", ""))
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

    client = get_async_client()
    cutoff_ms = _get_cutoff_timestamp_ms(cutoff_date)

    # Fetch market (handle 404 gracefully)
    try:
        if slug:
            response = await fetch_with_retry(client, f"{BASE_URL}/v0/slug/{slug}")
        else:
            response = await fetch_with_retry(client, f"{BASE_URL}/v0/market/{market_id}")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            identifier = slug or market_id
            return f"No market found with {'slug' if slug else 'ID'} '{identifier}'."
        raise

    market = response.json()
    if not market or not market.get("id"):
        identifier = slug or market_id
        return f"No market found with {'slug' if slug else 'ID'} '{identifier}'."

    # Check if market existed before cutoff
    created = market.get("createdTime", 0)
    if created >= cutoff_ms:
        return f"Market did not exist as of {cutoff_date} (created {_ts_ms_to_date(created)})."

    contract_id = market["id"]

    # Hide resolution if after cutoff (backtesting)
    if _is_backtesting(cutoff_date):
        resolution_time = market.get("resolutionTime")
        if resolution_time and resolution_time >= cutoff_ms:
            market.pop("resolution", None)
            market.pop("resolutionTime", None)
            market["isResolved"] = False

    # Fetch probability history and comments concurrently
    prob_history = None
    comments = None

    tasks = []
    if include_history and market.get("outcomeType") == "BINARY":
        tasks.append(_fetch_bets(client, contract_id, cutoff_ms))
    else:
        tasks.append(asyncio.coroutine(lambda: None)() if False else asyncio.sleep(0))

    if include_comments:
        tasks.append(_fetch_comments(client, contract_id, cutoff_ms))
    else:
        tasks.append(asyncio.sleep(0))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    if include_history and market.get("outcomeType") == "BINARY":
        if not isinstance(results[0], BaseException):
            prob_history = results[0]

    if include_comments:
        comment_idx = 1 if (include_history and market.get("outcomeType") == "BINARY") else 0
        if comment_idx < len(results) and not isinstance(results[comment_idx], BaseException):
            comments = results[comment_idx]
            if not isinstance(comments, list):
                comments = None

    return _format_market_detail(market, prob_history, comments, cutoff_date)


if __name__ == "__main__":
    mcp.run()
