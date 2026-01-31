# AI-Assisted Foreign Exchange Rate Lookup

## Proof of Concept & Regulatory Compliance

> [!IMPORTANT]
> **This project is a Proof of Concept (POC) demonstrating AI as a natural language interface.**

In regulated financial environments, core business logic and financial calculations must be **repeatable, deterministic, and auditable**. 

- **Repeatability**: Processes must be capable of being re-run directly to verify results.
- **Determinism**: The same inputs must always produce the exact same outputs. AI models are probabilistic by nature and may not satisfy this requirement.
- **Auditability**: It must be possible to trace exactly how a value was derived. "The model said so" is not an acceptable audit trail.

**This project demonstrates how AI can serve as a user-friendly INTERFACE layer.** 
However, the *actual* heavyweight financial logic (ledgers, rate application, risk calculations, etc.) should remain in deterministic, rule-based systems where code can be statically analyzed and behaviour is guaranteed. The AI should tool-call into these deterministic systems rather than attempting to perform the calculations itself.

## Comparison with Previous Work

The original [fx-rate](https://github.com/bill-ying/fx-rate) project is a traditional command-line utility. It relies on strict, deterministic inputs where users must provide specific flags and formats (e.g., `--date 2024-01-01`) to get a result. This ensures reliability but offers a rigid user experience.

In contrast, this **AI-assisted project** acts as a Proof of Concept for a modern, conversational interface. It leverages **Gemma 3:12b** to understand natural language queries (e.g., "What was the rate last Friday?"), prioritizing an intuitive and user-friendly experience while using the same underlying data source.

## Features

- **Natural Language Queries**: Ask about exchange rates in plain English
- **Bank of Canada Data**: Official exchange rates from the Bank of Canada Valet API
- **AI-Powered**: Uses Google's Gemma 3:12b via Ollama with native function calling
- **Bidirectional**: Supports both USD→CAD and CAD→USD conversions
- **Amount Conversion**: Specify an amount to convert, not just the rate

## Prerequisites

- Python 3.9 or higher
- [Ollama](https://ollama.ai/) installed and running
- **Gemma 3:12b** model pulled in Ollama (`ollama pull gemma3:12b`)


## Usage

### Interactive Mode

Start an interactive chat session:

```bash
python main.py
```

Example conversation:

```
You: How much CAD was for USD$1 on 2024-01-15?
Assistant: On January 15, 2024, 1 USD was worth 1.3456 CAD according to the Bank of Canada.

You: What about converting 500 USD to CAD on that date?
Assistant: Using the exchange rate from January 15, 2024 (1 USD = 1.3456 CAD), 
500 USD would be equal to 672.80 CAD.

You: Now show me CAD to USD for March 31, 2024
Assistant: On March 31, 2024, 1 CAD was worth 0.7385 USD according to the Bank of Canada.
```

### Single Query Mode

Ask a single question and exit:

```bash
python main.py --query "How much CAD was for USD$1 on 2024-01-15?"
```

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.