"""
AI Assistant for natural language FX rate queries.
Uses Command R via Ollama with native tool calling for FX rate lookups.
"""

import json
import logging
from datetime import datetime, date
from typing import Optional, Dict, Any, List

import ollama

from fx_service import FxRateService, ExchangeResult

logger = logging.getLogger(__name__)


class FxAIAssistant:
    """
    AI-powered assistant for USD/CAD exchange rate queries.
    
    Uses Command R model via Ollama with native tool calling to:
    1. Parse natural language queries
    2. Call the FX rate service for actual rates from Bank of Canada
    3. Generate natural language responses
    """
    
    MODEL_NAME = "gemma3:12b"
    
    # Native tool definition for Ollama
    FX_RATE_TOOL = {
        "type": "function",
        "function": {
            "name": "get_fx_rate",
            "description": "Get the exchange rate between USD and CAD for a specific date from Bank of Canada. Use this tool whenever the user asks about exchange rates, currency conversion, or how much one currency is worth in another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_currency": {
                        "type": "string",
                        "enum": ["USD", "CAD"],
                        "description": "The source currency to convert from"
                    },
                    "to_currency": {
                        "type": "string",
                        "enum": ["USD", "CAD"],
                        "description": "The target currency to convert to"
                    },
                    "date": {
                        "type": "string",
                        "description": "The date for the exchange rate in YYYY-MM-DD format"
                    },
                    "amount": {
                        "type": "number",
                        "description": "Optional amount to convert. If not specified, returns the rate for 1 unit of source currency."
                    }
                },
                "required": ["from_currency", "to_currency", "date"]
            }
        }
    }
    
    SYSTEM_PROMPT = """You are a helpful AI assistant. 
You have access to real-time and historical exchange rates from the Bank of Canada, but you can also answer general knowledge questions.

To use a tool, you MUST use this format and then STOP: [TOOL: tool_name(arg1=val1, arg2=val2)]

Available tools:
- get_fx_rate(from_currency: str, to_currency: str, date: str, amount: float = None): Get the exchange rate between USD and CAD for a specific date.

When a user asks about exchange rates or currency conversion between USD and CAD:
1. You MUST use the get_fx_rate tool to fetch the actual rate. Do not guess the rate.
2. Output ONLY the tool call tag and nothing else.
3. Once you output the tag, you will receive an "Observation" with the tool result.
4. Then, provide a clear, friendly response to the user based on that observation.

Important notes for FX queries:
- Only USD and CAD currencies are supported
- Rates are from the Bank of Canada
- If the user doesn't specify a date, ask them for the date
- If the user asks for a date in the future, explain that you can only provide historical rates
- The Bank of Canada may not have rates for weekends or holidays

For all other questions (e.g. general knowledge, history, facts), answer them directly using your own knowledge. Do not use the get_fx_rate tool for these queries."""

    def __init__(self, fx_service: Optional[FxRateService] = None):
        """
        Initialize the AI assistant.
        
        Args:
            fx_service: Optional FxRateService instance. Creates a new one if not provided.
        """
        self._fx_service = fx_service or FxRateService()
        self._owns_service = fx_service is None
        self._conversation_history: List[Dict[str, Any]] = []
        
    def chat(self, user_message: str) -> str:
        """
        Process a user message and return the assistant's response.
        
        Args:
            user_message: The user's natural language query
            
        Returns:
            The assistant's response as a string
        """
        logger.info(f"User message: {user_message}")
        
        # Add user message to history
        self._conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            # Call Ollama
            try:
                response = ollama.chat(
                    model=self.MODEL_NAME,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        *self._conversation_history
                    ],
                    tools=[self.FX_RATE_TOOL]
                )
                assistant_message = response["message"]
            except Exception as e:
                # If model doesn't support tools (like Gemma 3 in some Ollama versions), 
                # retry without tools and parse manually
                if "does not support tools" in str(e):
                    logger.info(f"Model {self.MODEL_NAME} does not support native tools, falling back to manual parsing.")
                    response = ollama.chat(
                        model=self.MODEL_NAME,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            *self._conversation_history
                        ]
                    )
                    assistant_message = response["message"]
                else:
                    raise e

            logger.debug(f"Model response: {assistant_message}")
            
            # Check for tool calls (native or manual)
            tool_calls = assistant_message.get("tool_calls")
            if not tool_calls:
                # Try manual parsing
                tool_calls = self._parse_manual_tool_calls(assistant_message.get("content", ""))
            
            if tool_calls:
                tool_result = self._handle_tool_calls(tool_calls)
                
                # Add assistant message with tool call to history
                self._conversation_history.append(assistant_message)
                
                # Add tool result to history
                # When falling back to manual parsing, we use 'user' role with 'Observation' prefix
                # as many models handled through Ollama don't support the 'tool' role natively
                self._conversation_history.append({
                    "role": "user",
                    "content": f"Observation: {tool_result}"
                })
                
                # Get final response from model
                final_response = ollama.chat(
                    model=self.MODEL_NAME,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        *self._conversation_history
                    ]
                )
                
                final_content = final_response["message"]["content"]
                logger.debug(f"Final model response: {final_content}")
                self._conversation_history.append({
                    "role": "assistant",
                    "content": final_content
                })
                
                return final_content
            else:
                # No tool call, just return the response
                content = assistant_message.get("content", "")
                self._conversation_history.append({
                    "role": "assistant",
                    "content": content
                })
                return content
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_msg = f"I encountered an error processing your request: {str(e)}"
            self._conversation_history.append({
                "role": "assistant",
                "content": error_msg
            })
            return error_msg
                
    def _parse_manual_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Parse manual tool calls like [TOOL: tool_name(arg1=val1)] from text."""
        import re
        
        tool_calls = []
        # Pattern to match [TOOL: tool_name(...)] or <call:tool_name(...)> (fallback)
        patterns = [
            r"\[TOOL:\s*(\w+)\((.*?)\)\]",
            r"<call:(\w+)\((.*?)\)>"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                func_name = match.group(1)
                args_str = match.group(2)
                
                # More robust argument parsing: match key:val or key=val
                args = {}
                if args_str.strip():
                    # Match key names followed by : or = and then a value
                    # Values can be quoted or unquoted
                    arg_pattern = r"(\w+)\s*[:=]\s*(\"[^\"]*\"|'[^']*'|[^,\)]+)"
                    arg_matches = re.finditer(arg_pattern, args_str)
                    for arg_match in arg_matches:
                        k = arg_match.group(1).strip()
                        v = arg_match.group(2).strip().strip("'").strip('"')
                        
                        # Try to convert to numeric if possible
                        try:
                            if '.' in v:
                                args[k] = float(v)
                            else:
                                args[k] = int(v)
                        except ValueError:
                            args[k] = v
                
                tool_calls.append({
                    "function": {
                        "name": func_name,
                        "arguments": args
                    }
                })
            
        return tool_calls

    def _handle_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> str:
        """
        Handle tool calls from the model.
        
        Args:
            tool_calls: List of tool call dictionaries from the model
            
        Returns:
            Tool result as a string
        """
        results = []
        
        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            func_name = func.get("name")
            
            if func_name == "get_fx_rate":
                args = func.get("arguments", {})
                result = self._execute_fx_rate_lookup(args)
                results.append(result)
            else:
                results.append(f"Unknown tool: {func_name}")
        
        return "\n".join(results)
    
    def _execute_fx_rate_lookup(self, args: Dict[str, Any]) -> str:
        """
        Execute the FX rate lookup tool.
        
        Args:
            args: Tool arguments containing from_currency, to_currency, date, and optional amount
            
        Returns:
            Result string with exchange rate information
        """
        from_currency = args.get("from_currency", "").upper()
        to_currency = args.get("to_currency", "").upper()
        date_str = args.get("date", "")
        amount = args.get("amount")
        
        logger.info(f"Executing FX lookup: {from_currency} -> {to_currency} on {date_str}, amount={amount}")
        
        try:
            result = self._fx_service.get_rate_for_date(
                from_currency=from_currency,
                to_currency=to_currency,
                lookup_date=date_str,
                amount=amount
            )
            
            if result:
                return self._format_result(result)
            else:
                return f"No exchange rate data available for {date_str}. This might be a weekend, holiday, or a date outside the available data range."
                
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error fetching rate: {e}")
            return f"Error fetching exchange rate: {str(e)}"
    
    def _format_result(self, result: ExchangeResult) -> str:
        """Format an ExchangeResult for the model to use in its response."""
        if result.amount is not None and result.converted_amount is not None:
            return (f"Exchange rate on {result.rate_date}: "
                    f"1 {result.from_currency.value} = {result.rate} {result.to_currency.value}. "
                    f"{result.amount} {result.from_currency.value} = {result.converted_amount} {result.to_currency.value}")
        return f"Exchange rate on {result.rate_date}: 1 {result.from_currency.value} = {result.rate} {result.to_currency.value}"
    
    def clear_history(self):
        """Clear the conversation history."""
        self._conversation_history = []
    
    def close(self):
        """Clean up resources."""
        if self._owns_service:
            self._fx_service.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
