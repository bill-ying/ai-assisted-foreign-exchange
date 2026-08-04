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

## Architecture

Exchange rate data is fetched via a dedicated [MCP server](https://github.com/bill-ying/mcp-server-ts-bank-of-canada-valet) deployed as a Cloudflare Worker. The AI assistant calls the MCP server's `get_rate` tool over the Streamable HTTP transport — it **never** calls the Bank of Canada Valet API directly.

```
┌───────────────────────────────────────────────────────────┐
│                  AI-Assisted FX Rate Lookup               │
│                                                           │
│  User ─► OpenRouter (Cohere/Poolside/Google) ─► LangChain │
│                          Tool Call │                      │
│                                    ▼                      │
│                          McpProvider (Strategy)           │
│                       MCP Python SDK (Streamable HTTP)    │
└────────────────────────────────┬──────────────────────────┘
                                 │  HTTPS / JSON-RPC
                                 ▼
┌───────────────────────────────────────────────────────────┐
│       MCP Server (Cloudflare Worker)                      │
│       get_rate tool → Bank of Canada Valet API            │
│  https://mcp-server-bank-of-canada-valet.                 │
│         bill-ying.workers.dev/mcp                         │
└───────────────────────────────────────────────────────────┘
```

This decoupled architecture cleanly separates concerns:
- **MCP server** owns the data source integration, schema validation, and structured responses
- **AI client** owns the conversational interface, tool orchestration, and compliance validation
- **Strategy pattern** allows swapping between `McpProvider` and `BankOfCanadaProvider` with zero code changes

## Comparison with Previous Work

The original [fx-rate](https://github.com/bill-ying/fx-rate) project is a traditional command-line utility. It relies on strict, deterministic inputs where users must provide specific flags and formats (e.g., `--date 2024-01-01`) to get a result. This ensures reliability but offers a rigid user experience.

In contrast, this **AI-assisted project** acts as a Proof of Concept for a modern, conversational interface. Free-tier models on OpenRouter (led by **Cohere North Mini Code**) are leveraged to understand natural language queries (e.g., "What was the rate on 2024-01-15?"), whereby an intuitive and user-friendly experience is prioritized while the same underlying data source is used.

## Features

- **Natural Language Queries**: Exchange rates can be queried in plain English
- **MCP Server Integration**: Exchange rates are fetched via the [Bank of Canada Valet MCP server](https://github.com/bill-ying/mcp-server-ts-bank-of-canada-valet) using the MCP Python SDK (Streamable HTTP transport)
- **AI-Powered**: Multiple free-tier OpenRouter models are used with automatic fallback (see [Model Fallback](#model-fallback)); native function calling via LangChain
- **Bidirectional**: Both USD→CAD and CAD→USD conversions are supported
- **Amount Conversion**: An amount can be specified to be converted, not just the rate
- **Audit Logging**: All tool calls and responses are logged via the Observer pattern
- **Compliance Validation** *(opt-in)*: LangGraph graph verifies every LLM response against raw tool data, auto-corrects on failure, and appends a disclaimer if correction is exhausted
- **IDE Debugging**: Includes VS Code debug configurations

## Model Fallback

The assistant uses a prioritised list of free-tier OpenRouter models. If the primary model returns an **HTTP 429 (rate-limited)** response, the next model in the list is tried automatically. If all three models are rate-limited simultaneously, the user receives a friendly message asking them to try again later.

| Priority | Model ID | Provider |
|----------|----------|----------|
| 1 (primary) | `cohere/north-mini-code:free` | Cohere |
| 2 (fallback) | `poolside/laguna-s-2.1:free` | Poolside |
| 3 (fallback) | `google/gemma-4-31b-it:free` | Google |

The fallback list is defined in `FxAssistant.MODELS` and can be updated without touching any other code.

## Compliance Validation (LangGraph)

Enabled with `--compliance` / `-c`. When active, every `chat()` call routes through a **LangGraph `StateGraph`** instead of the standard direct loop:

```
invoke_llm
    │
    ├─ tool calls? ──yes──► execute_tools ──► invoke_llm_final
    │                                              │
    └─ no ─────────────────────────────────────────┤
                                                   ▼
                                              validate
                                                   │
                                    ┌──────────────┴──────────────┐
                                 passed?                       failed?
                                    │                             │
                                  emit                         correct
                                 [END]                (inject correction prompt)
                                                                  │
                                                            max retries?
                                                                  ├─ yes ──► emit + disclaimer
                                                                  └─ no  ──► invoke_llm_final
```

### Compliance Rules (Chain of Responsibility)

| Rule | Severity | What it checks |
|------|----------|----------------|
| `RateValuePresentRule` | **ERROR** | Exact numeric rate from tool result appears in response |
| `SourceAttributionRule` | WARNING | "Bank of Canada" is cited as the data source |
| `DateConsistencyRule` | WARNING | Queried year appears in the response |
| `CurrencyConsistencyRule` | WARNING | Queried currencies (USD/CAD) are mentioned |

### Validation Strategies (Strategy pattern)

| Strategy | Behaviour |
|----------|-----------|
| `LenientValidationStrategy` *(default)* | Only ERROR violations trigger correction |
| `StrictValidationStrategy` | WARNING violations also trigger correction |

## Prerequisites

- Python 3.9 or higher
- An OpenRouter API Key configured in `.env` file (`OPENROUTER_API_KEY="sk-or-..."`)
- Internet access (for calling OpenRouter API and the MCP server on Cloudflare Workers)


## Usage

### Interactive Mode

An interactive chat session can be started:

```bash
# Standard mode
python main.py

# With LangGraph compliance validation
python main.py --compliance
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
# Standard
python main.py --query "How much CAD was for USD$1 on 2024-01-15?"

# With compliance validation
python main.py --compliance --query "How much CAD was for USD$1 on 2024-01-15?"
```

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.