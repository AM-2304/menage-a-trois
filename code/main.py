#!/usr/bin/env python3
"""
MLE Hiring Challenge — CLI Entry Point

Processes support tickets through the triage agent pipeline.

Usage:
    python code/main.py                        # Fresh run
    python code/main.py --resume               # Resume from last checkpoint
    python code/main.py --merge OLD_CSV        # Merge: keep good rows from OLD_CSV, reprocess fallback rows
    python code/main.py --input PATH           # Use custom input CSV
    python code/main.py --output PATH          # Write to custom output CSV

Features:
    - Saves progress after every ticket (crash-safe)
    - Resume from where it stopped on restart
    - Merge mode: replace fallback/low-quality rows from a previous run
    - Rate limit handling with exponential backoff
    - Windows PowerShell UTF-16 .env file auto-detection
    - Deterministic output (same input → same output)
"""

import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# ─── Fix Windows PowerShell UTF-16 .env issue ────────────────────────────────


def load_env_file(env_path: str) -> None:
    """
    Load .env file with automatic encoding detection.

    PowerShell's echo/Out-File writes UTF-16 LE with BOM by default.
    Python's dotenv library can't read this. We detect and handle it.
    """
    if not os.path.exists(env_path):
        return

    # Try reading with different encodings
    content = None
    for encoding in ('utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'ascii'):
        try:
            with open(env_path, 'r', encoding=encoding) as f:
                content = f.read()
            # If we successfully read it and it has content, use this encoding
            if content and '=' in content:
                break
        except (UnicodeDecodeError, UnicodeError):
            continue

    if not content:
        return

    # Parse key=value pairs
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' in line:
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                os.environ.setdefault(key, value)


def find_env_file() -> Optional[str]:
    """Find .env file, searching up from current directory."""
    # Check common locations
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'),
        os.path.join(os.getcwd(), '.env'),
        os.path.join(Path.home(), '.env'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# ─── Progress tracking ───────────────────────────────────────────────────────

CHECKPOINT_SUFFIX = '.checkpoint.json'


def save_checkpoint(output_path: str, completed_indices: List[int], results: Dict[int, Dict]) -> None:
    """Save progress checkpoint."""
    ckpt_path = output_path + CHECKPOINT_SUFFIX
    data = {
        'completed': completed_indices,
        'results': {str(k): v for k, v in results.items()},
    }
    with open(ckpt_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def load_checkpoint(output_path: str) -> Optional[Dict]:
    """Load progress checkpoint if it exists."""
    ckpt_path = output_path + CHECKPOINT_SUFFIX
    if not os.path.exists(ckpt_path):
        return None
    try:
        with open(ckpt_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Convert string keys back to int
        data['results'] = {int(k): v for k, v in data['results'].items()}
        return data
    except Exception:
        return None


def remove_checkpoint(output_path: str) -> None:
    """Remove checkpoint file after successful completion."""
    ckpt_path = output_path + CHECKPOINT_SUFFIX
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


# ─── Fallback row detection (for --merge mode) ───────────────────────────────


def is_fallback_row(row: Dict) -> bool:
    """Detect if a row was produced by rule-based fallback (low quality)."""
    justification = (row.get('justification') or '').lower()
    response = (row.get('response') or '').lower()

    indicators = [
        'rule-based fallback' in justification,
        'fallback' in justification,
        'llm unavailability' in justification,
        response == '',
        'routing it to a human support agent' in response and len(response) < 150,
    ]
    return any(indicators)


# ─── CSV I/O ─────────────────────────────────────────────────────────────────

OUTPUT_HEADERS = [
    'issue', 'subject', 'company', 'response', 'product_area',
    'status', 'request_type', 'justification', 'confidence_score',
    'source_documents', 'risk_level', 'pii_detected', 'language',
    'actions_taken',
]


def read_input_csv(path: str) -> List[Dict]:
    """Read input CSV and return list of ticket dicts."""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize header case
            normalized = {}
            for k, v in row.items():
                normalized[k.strip().lower().replace(' ', '_')] = v
            rows.append(normalized)
    return rows


def write_output_csv(path: str, rows: List[Dict]) -> None:
    """Write output CSV with all required columns."""
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADERS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def merge_output_csvs(old_path: str, new_results: Dict[int, Dict], tickets: List[Dict]) -> List[Dict]:
    """Merge old output with new results, replacing fallback rows."""
    old_rows = read_input_csv(old_path)
    merged = []

    for i, old_row in enumerate(old_rows):
        if i in new_results:
            merged.append(new_results[i])
        else:
            merged.append(old_row)

    return merged


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='MLE Hiring Challenge — Support Ticket Triage Agent',
    )
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Input CSV path (default: support_tickets/support_tickets.csv)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output CSV path (default: support_tickets/output.csv)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    parser.add_argument('--merge', type=str, metavar='OLD_CSV',
                        help='Merge mode: reprocess fallback rows from OLD_CSV')
    parser.add_argument('--dry-run', action='store_true',
                        help='Process first 3 tickets only (for testing)')

    args = parser.parse_args()

    # Resolve paths
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = args.input or os.path.join(repo_root, 'support_tickets', 'support_tickets.csv')
    output_path = args.output or os.path.join(repo_root, 'support_tickets', 'output.csv')

    # Load environment
    env_path = find_env_file()
    if env_path:
        print(f"Loading .env from: {env_path}")
        load_env_file(env_path)
    else:
        print("No .env file found — using existing environment variables")

    # Check for API keys
    has_groq = bool(os.environ.get('GROQ_API_KEY', '').strip())
    has_claude = bool(os.environ.get('ANTHROPIC_API_KEY', '').strip())
    has_bedrock = bool(os.environ.get('A_BEARER_TOKEN_ANTHROPIC', '').strip())

    if not has_groq and not has_claude and not has_bedrock:
        print("❌ ERROR: No LLM API key found. Set GROQ_API_KEY, ANTHROPIC_API_KEY, or A_BEARER_TOKEN_ANTHROPIC in .env")
        sys.exit(1)

    print(f"LLM providers: Groq={'✅' if has_groq else '❌'}, Claude={'✅' if has_claude else '❌'}, Bedrock={'✅' if has_bedrock else '❌'}")

    # Read input tickets
    print(f"\nReading tickets from: {input_path}")
    tickets = read_input_csv(input_path)
    total = len(tickets)
    print(f"Found {total} tickets")

    if args.dry_run:
        tickets = tickets[:3]
        total = len(tickets)
        print(f"[DRY RUN] Processing first {total} tickets only")

    # Initialize agent
    print("\nInitializing triage agent...")

    # Add code/ to path for imports
    code_dir = os.path.dirname(os.path.abspath(__file__))
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    from agent import TriageAgent
    agent = TriageAgent(repo_root)

    # Determine which tickets to process
    completed = {}
    results = {}

    if args.resume:
        checkpoint = load_checkpoint(output_path)
        if checkpoint:
            results = checkpoint['results']
            completed_set = set(checkpoint['completed'])
            print(f"Resuming: {len(completed_set)}/{total} tickets already completed")
        else:
            completed_set = set()
            print("No checkpoint found, starting fresh")
    elif args.merge:
        # In merge mode, reprocess only fallback rows
        old_rows = read_input_csv(args.merge)
        completed_set = set()
        for i, row in enumerate(old_rows):
            if not is_fallback_row(row):
                results[i] = row
                completed_set.add(i)
        print(f"Merge mode: keeping {len(completed_set)}/{total} good rows, reprocessing {total - len(completed_set)}")
    else:
        completed_set = set()

    # Process tickets
    print(f"\nProcessing {total - len(completed_set)} remaining tickets...\n")
    start_time = time.time()
    errors = 0
    rate_limit_waits = 0

    for i, ticket in enumerate(tickets):
        if i in completed_set:
            continue

        issue = ticket.get('issue', '')
        subject = ticket.get('subject', '')
        company = ticket.get('company', '')

        # Show progress
        elapsed = time.time() - start_time
        eta = (elapsed / max(1, i - len(completed_set) + 1)) * (total - i - 1) if i > 0 else 0
        print(f"[{i+1}/{total}] Subject: {(subject or '(none)')[:50]}... (ETA: {eta:.0f}s)")

        # Process with retry on rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = agent.process_ticket(issue, subject, company)

                # Add input columns to result
                result['issue'] = issue
                result['subject'] = subject
                result['company'] = company

                results[i] = result
                completed_set.add(i)

                # Save checkpoint after every ticket
                save_checkpoint(output_path, list(completed_set), results)
                break

            except Exception as e:
                error_str = str(e)
                if '429' in error_str or 'rate_limit' in error_str.lower():
                    wait_time = min(60, 2 ** (attempt + 1))
                    rate_limit_waits += 1
                    print(f"  [RATE LIMIT] Waiting {wait_time}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                elif attempt == max_retries - 1:
                    print(f"  [ERROR] Failed after {max_retries} attempts: {error_str}")
                    errors += 1
                    # Use emergency fallback
                    result = {
                        'issue': issue, 'subject': subject, 'company': company,
                        'status': 'escalated', 'product_area': 'general_support',
                        'response': 'Your request has been routed to a human support agent for assistance.',
                        'justification': f'Processing error, escalating for safety. Error: {error_str[:100]}',
                        'request_type': 'product_issue',
                        'confidence_score': 0.1,
                        'source_documents': '',
                        'risk_level': 'medium',
                        'pii_detected': 'false',
                        'language': 'en',
                        'actions_taken': json.dumps([{"action": "escalate_to_human", "parameters": {"priority": "normal", "department": "general", "summary": "Processing error"}}]),
                    }
                    results[i] = result
                    completed_set.add(i)
                    save_checkpoint(output_path, list(completed_set), results)
                else:
                    print(f"  [RETRY] Attempt {attempt+1} failed: {error_str}")
                    time.sleep(1)

    # Write final output
    print(f"\nWriting output to: {output_path}")
    output_rows = [results[i] for i in range(total) if i in results]
    write_output_csv(output_path, output_rows)

    # Clean up checkpoint
    remove_checkpoint(output_path)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Tickets processed: {len(output_rows)}/{total}")
    print(f"  Errors: {errors}")
    print(f"  Rate limit waits: {rate_limit_waits}")
    print(f"  Time elapsed: {elapsed:.1f}s ({elapsed/max(1,len(output_rows)):.1f}s/ticket)")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")

    # Run validation
    print(f"\nRunning validation...")
    validate_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'validate_output.py')
    if os.path.exists(validate_path):
        os.system(f'{sys.executable} "{validate_path}"')


if __name__ == '__main__':
    main()
