#!/usr/bin/env python3
"""
AI-Assisted Foreign Exchange Rate Lookup

Interactive CLI application for querying USD/CAD exchange rates
using natural language. Powered by Command R7B via Ollama.
"""

import argparse
import logging
import sys

from ai_assistant import FxAIAssistant


def setup_logging(debug: bool = False):
    """Configure logging based on debug flag."""
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def print_welcome():
    """Print welcome message and usage hints."""
    print("\n" + "=" * 60)
    print("  AI-Assisted USD/CAD Exchange Rate Lookup")
    print("  Powered by Bank of Canada data & Gemma 3")
    print("=" * 60)
    print("\nAsk me about USD/CAD exchange rates in natural language!")
    print("\nExample questions:")
    print('  • "How much CAD was for USD$1 on 2024-01-15?"')
    print('  • "What was the exchange rate from CAD to USD on March 31, 2024?"')
    print('  • "Convert 100 USD to CAD on 2024-06-15"')
    print("\nCommands:")
    print("  • Type 'quit' or 'exit' to end the session")
    print("  • Type 'clear' to reset conversation history")
    print("-" * 60 + "\n")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="AI-assisted USD/CAD exchange rate lookup using Bank of Canada data"
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Single query mode: ask a question and exit"
    )
    
    args = parser.parse_args()
    setup_logging(args.debug)
    
    # Single query mode
    if args.query:
        with FxAIAssistant() as assistant:
            response = assistant.chat(args.query)
            print(response)
        return 0
    
    # Interactive mode
    print_welcome()
    
    try:
        with FxAIAssistant() as assistant:
            while True:
                try:
                    user_input = input("You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    if user_input.lower() in ('quit', 'exit', 'q'):
                        print("\nGoodbye! Have a great day!")
                        break
                    
                    if user_input.lower() == 'clear':
                        assistant.clear_history()
                        print("Conversation history cleared.\n")
                        continue
                    
                    print("\nAssistant: ", end="", flush=True)
                    response = assistant.chat(user_input)
                    print(response)
                    print()
                    
                except KeyboardInterrupt:
                    print("\n\nInterrupted. Type 'quit' to exit or continue asking questions.")
                    
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.debug:
            raise
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
