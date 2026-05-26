# MLE Hiring Challenge — Support Triage Agent

A terminal-based AI support triage agent that classifies and responds to customer support tickets for three product ecosystems: **DevPlatform**, **Claude** (by Anthropic), and **Visa**.

## Quick Start

```bash
# 1. Clone and navigate
cd code/

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp ../.env.example ../.env
# Edit ../.env with your API keys (see below)

# 5. Run the agent
python main.py
```

## Environment Variables

Create a `.env` file in the project root with at least one LLM provider key:

```env
# Primary LLM: Groq (fastest inference)
GROQ_API_KEY=gsk_your_key_here

# Fallback LLM Option A: Anthropic Claude (direct API)
ANTHROPIC_API_KEY=sk-ant-your_key_here

# Fallback LLM Option B: AWS Bedrock (Claude via Bedrock)
A_BEARER_TOKEN_ANTHROPIC=ABSK_your_base64_token_here
```

The agent uses **Groq** as the primary provider (fastest) and **Claude** as a fallback. At least one key must be set.

### Windows PowerShell UTF-16 Fix

If your `.env` was created via PowerShell's `echo` command, it may be UTF-16 encoded. The agent automatically detects and handles this — no manual fix needed.

## CLI Usage

```bash
# Fresh run — process all tickets
python main.py

# Resume from last checkpoint (if interrupted)
python main.py --resume

# Merge — reprocess only fallback rows from a previous run
python main.py --merge ../support_tickets/output_old.csv

# Dry run — process first 3 tickets only (for testing)
python main.py --dry-run

# Custom input/output paths
python main.py --input ../support_tickets/custom_input.csv --output ../support_tickets/custom_output.csv
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for full documentation including:
- Pipeline diagram and component design
- Retrieval strategy (hybrid BM25 + TF-IDF + RRF)
- Adversarial safety layer (14+ injection patterns)
- Escalation decision logic
- Tool calling prerequisites
- Self-assessment scores per evaluation dimension

## File Structure

```
code/
├── main.py           # CLI entry point (resume, merge, rate limits)
├── agent.py          # Core triage agent (LLM pipeline, prompt, post-processing)
├── retriever.py      # Hybrid BM25 + TF-IDF retrieval (stdlib only)
├── safety.py         # Pre-LLM adversarial scanner (regex, base64, PII)
├── validate_output.py # Output CSV validation script (provided)
├── requirements.txt  # Python dependencies
├── README.md         # This file
└── ARCHITECTURE.md   # Architecture documentation
```

## Dependencies

| Package | Purpose |
|---|---|
| `groq>=0.9.0` | Groq API client (llama-3.3-70b-versatile) |
| `anthropic>=0.40.0` | Anthropic Claude API client |
| `boto3>=1.34.0` | AWS Bedrock support (Claude via Bedrock) |
| `python-dotenv>=1.0.0` | Environment variable loading |

**Note**: The retrieval module (`retriever.py`) and safety scanner (`safety.py`) use only Python stdlib — no external NLP libraries.

## Performance

| Metric | Value |
|---|---|
| Tickets processed | 89/89 |
| Total time | ~120s |
| Per-ticket latency | ~1.4s |
| Errors | 0 |
| Rate limit waits | 0 |
| Validation | ✅ PASS (14/14 columns) |
| Adversarial detection | 14 injections + 8 PII catches |

## How It Works

1. **Parse** each ticket's JSON conversation history
2. **Detect** language (Unicode ranges + keyword heuristics)
3. **Scan** for adversarial patterns and PII (pre-LLM)
4. **Infer** product from company field + content
5. **Retrieve** top-5 relevant corpus documents via hybrid search
6. **Call** LLM with hardened prompt + corpus context + safety context
7. **Parse** structured JSON response
8. **Validate** schema, paths, PII, enums
9. **Write** output CSV with checkpointing

## Validation

After processing, run the provided validator:

```bash
python validate_output.py
```

Expected output:
```
✅ PASS: Output format is valid.
```

## Troubleshooting

| Issue | Solution |
|---|---|
| "No LLM API key found" | Set `GROQ_API_KEY` or `ANTHROPIC_API_KEY` in `.env` |
| Rate limit errors | Use `--resume` to continue from checkpoint |
| Partial output | Use `--resume` to process remaining tickets |
| Low-quality rows | Use `--merge output.csv` to reprocess fallback rows |
| Windows encoding issues | `.env` UTF-16 is auto-detected |
