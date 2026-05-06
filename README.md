# AI-Assisted Foreign Exchange Rate Lookup

## Proof of Concept & Regulatory Compliance

> [!IMPORTANT]
> **This project is a Proof of Concept (POC) in which AI is demonstrated as a natural language interface.**

In regulated financial environments, systems must be **repeatable, deterministic, and auditable**. When incorporating AI, **explainability, transparency, and accountability** must also be ensured.

- **Repeatability**: Processes must be capable of being re-run directly to verify results.
- **Determinism**: The same inputs must always produce the exact same outputs. AI models are probabilistic by nature and may not satisfy this requirement.
- **Auditability**: It must be possible to trace exactly how a value was derived. "The AI model said so" is not an acceptable audit trail.
- **Explainability**: The system must be able to explain how a result was reached (e.g., "The rate of 1.35 was fetched from the Bank of Canada API for date X").
- **Transparency**: The boundary between probabilistic AI logic and deterministic business logic must be clearly defined and visible.
- **Accountability**: The system must have clear ownership of decisions, ensuring that AI tool-calls are logged and verifiable against the underlying deterministic systems.

**It is demonstrated in this project how AI can serve as a user-friendly INTERFACE layer.** 
However, the *actual* heavyweight financial logic (ledgers, rate application, risk calculations, etc.) should remain in deterministic, rule-based systems where code can be statically analyzed and behaviour is guaranteed. Tool-calls should be performed by the AI into these deterministic systems rather than attempting to perform the calculations itself.

## Comparison with Previous Work

The original [fx-rate](https://github.com/bill-ying/fx-rate) project is a traditional command-line utility. It relies on strict, deterministic inputs where users must provide specific flags and formats (e.g., `--date 2024-01-01`) to get a result. This ensures reliability but offers a rigid user experience.

In contrast, this **AI-assisted project** acts as a Proof of Concept for a modern, conversational interface. **Gemma 4 26B A4B** is leveraged to understand natural language queries (e.g., "What was the rate on 2024-01-15?"), whereby an intuitive and user-friendly experience is prioritized while the same underlying data source is used.

## Architecture

The codebase follows **Gang of Four (GoF) design patterns** for extensibility and clean separation of concerns:

```
ai-assisted-foreign-exchange/
├── main.py                          # CLI entry point
├── ai_assistant.py                  # Backward-compat shim
├── ai/                              # AI layer (GoF patterns)
│   ├── fx_assistant.py              # Facade + Factory Method
│   ├── events.py                    # Observer (audit events)
│   ├── tools.py                     # Template Method + Registry
│   ├── result_formatter.py          # Strategy (output formatting)
│   └── chat_history.py             # Strategy (message storage)
└── fx_service/                      # FX domain layer
    ├── fx_rate_service.py           # Service orchestrator
    ├── rate_provider.py             # Strategy (data sources)
    ├── bank_of_canada_client.py     # HTTP client
    └── currency.py                  # Domain model
```

### Design Patterns Used

| Pattern | Location | Purpose |
|---------|----------|---------|
| **Facade** | `FxAssistant` | Hides LLM, tools, history, and events behind `chat()` |
| **Factory Method** | `FxAssistant.create()` | Assembles components with sensible defaults |
| **Strategy** | `RateProvider`, `ResultFormatter`, `ChatHistory` | Swap implementations without modifying clients |
| **Template Method** | `BaseFxTool` → `FxRateTool` | validate → execute → format pipeline |
| **Observer** | `EventBus`, `AuditLogger` | Decoupled audit logging for compliance |
| **Adapter** | `BankOfCanadaProvider` | Adapts HTTP client to `RateProvider` interface |

## Features

- **Natural Language Queries**: Exchange rates can be queried in plain English
- **Bank of Canada Data**: Official exchange rates are provided from the Bank of Canada Valet API
- **AI-Powered**: **Gemma 4 26B** is used via Ollama with native function calling (LangChain)
- **Bidirectional**: Both USD→CAD and CAD→USD conversions are supported
- **Amount Conversion**: An amount can be specified to be converted, not just the rate
- **Audit Logging**: All tool calls and responses are logged via the Observer pattern
- **IDE Debugging**: Includes VS Code debug configurations

## Prerequisites

- Python 3.9 or higher
- [Ollama](https://ollama.ai/) installed and running
- **Gemma 4 26B** model pulled in Ollama (`ollama pull gemma4:26b`)


## Usage

### Interactive Mode

An interactive chat session can be started:

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

A single question can be asked and exited:

```bash
python main.py --query "How much CAD was for USD$1 on 2024-01-15?"
```

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.