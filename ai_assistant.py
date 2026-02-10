"""
AI Assistant for natural language FX rate queries.
Uses LangChain with Ollama and Google Gemma 3:12b for tool calling and FX rate lookups.
"""

import logging
from typing import Optional, List, Dict, Any

from langchain_community.chat_models import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from fx_service import FxRateService, ExchangeResult

logger = logging.getLogger(__name__)


class FxAIAssistant:
    """
    AI-powered assistant for USD/CAD exchange rate queries.
    
    Uses LangChain with Google Gemma 3 model via Ollama to:
    1. Parse natural language queries
    2. Call the FX rate service for actual rates from Bank of Canada
    3. Generate natural language responses
    """
    
    MODEL_NAME = "gemma3:12b"
    
    SYSTEM_PROMPT = """You are a helpful AI assistant. 
You have access to real-time and historical exchange rates from the Bank of Canada, but you can also answer general knowledge questions.

When a user asks about exchange rates or currency conversion between USD and CAD:
1. You MUST use the get_fx_rate tool to fetch the actual rate. Do not guess the rate.
2. Once you receive the tool result, provide a clear, friendly response to the user.

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
        self._chat_history: List[Any] = []
        
        # Initialize LangChain components
        self._llm = ChatOllama(
            model=self.MODEL_NAME,
            temperature=0
        )
        
        # Create the tool
        self._tools = [self._create_fx_tool()]
    
    def _create_fx_tool(self):
        """Create the FX rate lookup tool using LangChain's @tool decorator."""
        fx_service = self._fx_service
        
        @tool
        def get_fx_rate(
            from_currency: str,
            to_currency: str,
            date: str,
            amount: Optional[float] = None
        ) -> str:
            """Get the exchange rate between USD and CAD for a specific date from Bank of Canada.
            
            Use this tool whenever the user asks about exchange rates, currency conversion, 
            or how much one currency is worth in another.
            
            Args:
                from_currency: The source currency to convert from (USD or CAD)
                to_currency: The target currency to convert to (USD or CAD)
                date: The date for the exchange rate in YYYY-MM-DD format
                amount: Optional amount to convert. If not specified, returns the rate for 1 unit
            
            Returns:
                Exchange rate information as a formatted string
            """
            logger.info(f"Tool called: {from_currency} -> {to_currency} on {date}, amount={amount}")
            
            try:
                result = fx_service.get_rate_for_date(
                    from_currency=from_currency.upper(),
                    to_currency=to_currency.upper(),
                    lookup_date=date,
                    amount=amount
                )
                
                if result:
                    return _format_result(result)
                else:
                    return f"No exchange rate data available for {date}. This might be a weekend, holiday, or a date outside the available data range."
                    
            except ValueError as e:
                return f"Error: {str(e)}"
            except Exception as e:
                logger.error(f"Error fetching rate: {e}")
                return f"Error fetching exchange rate: {str(e)}"
        
        return get_fx_rate
    
    def chat(self, user_message: str) -> str:
        """
        Process a user message and return the assistant's response.
        
        Args:
            user_message: The user's natural language query
            
        Returns:
            The assistant's response as a string
        """
        logger.info(f"User message: {user_message}")
        
        try:
            # Manual tool call handling with LangChain
            self._chat_history.append(HumanMessage(content=user_message))
            
            messages = [SystemMessage(content=self.SYSTEM_PROMPT)] + self._chat_history
            response = self._llm.invoke(messages)
            
            # Check if response contains a tool call
            tool_result = self._parse_and_execute_tool(response.content)
            
            if tool_result:
                # Add the tool call to history
                self._chat_history.append(AIMessage(content=response.content))
                
                # Add tool result as a user message (observation)
                self._chat_history.append(HumanMessage(content=f"Observation: {tool_result}"))
                
                # Get final response after tool execution
                messages = [SystemMessage(content=self.SYSTEM_PROMPT)] + self._chat_history
                final_response = self._llm.invoke(messages)
                
                self._chat_history.append(AIMessage(content=final_response.content))
                return final_response.content
            else:
                # No tool call, return response directly
                self._chat_history.append(AIMessage(content=response.content))
                return response.content
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_msg = f"I encountered an error processing your request: {str(e)}"
            return error_msg
    
    def _parse_and_execute_tool(self, content: str) -> Optional[str]:
        """Parse tool calls from model output and execute them."""
        import re
        
        # Try to find tool calls in various formats
        # Format 1: ```tool_code\nget_fx_rate(...)\n```
        tool_code_pattern = r"```tool_code\s*\n(get_fx_rate\([^)]*\))"
        # Format 2: get_fx_rate(...)
        direct_pattern = r"get_fx_rate\(([^)]*)\)"
        
        match = re.search(tool_code_pattern, content) or re.search(direct_pattern, content)
        
        if match:
            # Extract the function call
            if 'tool_code' in content:
                func_call = match.group(1)
            else:
                func_call = match.group(0)
            
            logger.info(f"Detected tool call: {func_call}")
            
            # Parse arguments
            args = {}
            arg_pattern = r'(\w+)\s*=\s*["\']([^"\']*)["\']|(\w+)\s*=\s*(\d+\.?\d*)'
            for arg_match in re.finditer(arg_pattern, func_call):
                if arg_match.group(1):  # string argument
                    args[arg_match.group(1)] = arg_match.group(2)
                elif arg_match.group(3):  # numeric argument
                    key = arg_match.group(3)
                    val = arg_match.group(4)
                    args[key] = float(val) if '.' in val else int(val)
            
            # Execute the tool
            if args:
                tool = self._tools[0]
                try:
                    result = tool.func(**args)
                    return result
                except Exception as e:
                    logger.error(f"Error executing tool: {e}")
                    return f"Error executing tool: {str(e)}"
        
        return None
    
    def clear_history(self):
        """Clear the conversation history."""
        self._chat_history = []
    
    def close(self):
        """Clean up resources."""
        if self._owns_service:
            self._fx_service.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def _format_result(result: ExchangeResult) -> str:
    """Format an ExchangeResult for the model to use in its response."""
    if result.amount is not None and result.converted_amount is not None:
        return (f"Exchange rate on {result.rate_date}: "
                f"1 {result.from_currency.value} = {result.rate} {result.to_currency.value}. "
                f"{result.amount} {result.from_currency.value} = {result.converted_amount} {result.to_currency.value}")
    return f"Exchange rate on {result.rate_date}: 1 {result.from_currency.value} = {result.rate} {result.to_currency.value}"
