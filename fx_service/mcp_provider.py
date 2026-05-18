"""
MCP-based Rate Provider (GoF Strategy pattern).

Concrete Strategy + Adapter: calls the remote MCP server deployed on
Cloudflare Workers via the MCP Python SDK's Streamable HTTP transport.

Instead of hitting the Bank of Canada Valet API directly, this provider
delegates to the MCP server's ``get_rate`` tool, which encapsulates all
API interaction and data normalisation on the server side.

GoF Patterns:
- Strategy: implements the RateProvider interface, interchangeable with
  BankOfCanadaProvider without modifying any client code.
- Adapter: adapts the MCP tool-calling protocol to the RateProvider interface
  that the service layer depends on.
"""

import asyncio
import logging
from datetime import date
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from .rate_provider import RateProvider

logger = logging.getLogger(__name__)

# Default MCP server endpoint (Cloudflare Worker)
DEFAULT_MCP_URL = (
    "https://mcp-server-bank-of-canada-valet.bill-ying.workers.dev/mcp"
)


class McpProvider(RateProvider):
    """
    Concrete Strategy + Adapter: MCP server for Bank of Canada rates.

    Adapts the MCP tool-calling protocol (Streamable HTTP) to the
    RateProvider interface. Each call opens a short-lived MCP session,
    invokes the ``get_rate`` tool, and parses the structured response.

    This mirrors BankOfCanadaProvider's role but replaces the direct
    HTTP/REST call with an MCP tool call, gaining schema validation
    and structured output from the server.
    """

    def __init__(self, mcp_url: str = DEFAULT_MCP_URL):
        """
        Initialize the MCP provider.

        Args:
            mcp_url: URL of the MCP server endpoint (must support Streamable HTTP).
        """
        self._mcp_url = mcp_url

    # ------------------------------------------------------------------
    # RateProvider interface
    # ------------------------------------------------------------------

    def get_rate(
        self, series_name: str, lookup_date: date
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single rate observation via the MCP server's ``get_rate`` tool.

        The *series_name* (e.g. ``FXUSDCAD``) is decomposed into the
        ``from_currency`` / ``to_currency`` pair expected by the MCP tool.

        Args:
            series_name: The rate series identifier (e.g., 'FXUSDCAD')
            lookup_date: The date to look up

        Returns:
            Observation dictionary compatible with the existing service layer
            (keys: ``d``, ``<series_name>`` → ``{v: <rate>}``), or *None*
            if no data was found.
        """
        from_currency, to_currency = self._parse_series(series_name)

        result = self._call_get_rate(
            from_currency=from_currency,
            to_currency=to_currency,
            date_str=lookup_date.isoformat(),
        )

        if result is None or not result.get("success"):
            return None

        # Re-pack into the observation format that FxRateService._extract_rate expects
        return {
            "d": result.get("rateDate", lookup_date.isoformat()),
            series_name: {"v": str(result["rate"])},
        }

    def get_rates(
        self,
        series_name: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        """
        Fetch rate observations for a date range.

        The MCP server's ``get_rate`` tool only supports single-date lookups,
        so this method iterates over each business day in the range.

        Args:
            series_name: The rate series identifier
            start_date: Start of the date range (inclusive)
            end_date: End of the date range (inclusive)

        Returns:
            List of observation dictionaries for the range
        """
        from datetime import timedelta

        from_currency, to_currency = self._parse_series(series_name)
        observations: List[Dict[str, Any]] = []

        current = start_date
        while current <= end_date:
            # Skip weekends (no rate data expected)
            if current.weekday() < 5:
                result = self._call_get_rate(
                    from_currency=from_currency,
                    to_currency=to_currency,
                    date_str=current.isoformat(),
                )
                if result and result.get("success"):
                    observations.append(
                        {
                            "d": result.get("rateDate", current.isoformat()),
                            series_name: {"v": str(result["rate"])},
                        }
                    )
            current += timedelta(days=1)

        return observations

    def close(self) -> None:
        """No persistent resources to release (sessions are per-call)."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_get_rate(
        self,
        from_currency: str,
        to_currency: str,
        date_str: str,
        amount: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Call the MCP server's ``get_rate`` tool and return the parsed response.

        Opens a short-lived Streamable HTTP session, invokes the tool,
        and extracts the structured content from the response.

        Args:
            from_currency: Source currency code ('USD' or 'CAD')
            to_currency:   Target currency code ('USD' or 'CAD')
            date_str:      Date in YYYY-MM-DD format
            amount:        Optional amount to convert

        Returns:
            Parsed structured content dict, or None on failure.
        """
        arguments: Dict[str, Any] = {
            "from_currency": from_currency,
            "to_currency": to_currency,
            "date": date_str,
        }
        if amount is not None:
            arguments["amount"] = amount

        try:
            return asyncio.run(self._async_call_tool(arguments))
        except Exception:
            logger.exception("MCP tool call failed")
            return None

    async def _async_call_tool(
        self, arguments: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Async helper: open an MCP session and call ``get_rate``.

        Args:
            arguments: Tool arguments to pass

        Returns:
            Structured content dict from the tool response, or None
        """
        async with streamablehttp_client(self._mcp_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool(
                    name="get_rate", arguments=arguments
                )

                # Extract structured content or fall back to text parsing
                if hasattr(result, "structuredContent") and result.structuredContent:
                    return dict(result.structuredContent)

                # Fallback: parse from text content
                for content in result.content:
                    if hasattr(content, "text"):
                        logger.debug("MCP tool text response: %s", content.text)

                return None

    @staticmethod
    def _parse_series(series_name: str) -> tuple:
        """
        Parse a Bank of Canada series name into (from_currency, to_currency).

        Args:
            series_name: e.g. 'FXUSDCAD' or 'FXCADUSD'

        Returns:
            Tuple of (from_currency, to_currency)

        Raises:
            ValueError: If the series name format is unrecognised
        """
        series_map = {
            "FXUSDCAD": ("USD", "CAD"),
            "FXCADUSD": ("CAD", "USD"),
        }
        if series_name not in series_map:
            raise ValueError(f"Unsupported series: {series_name}")
        return series_map[series_name]
