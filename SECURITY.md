# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.9.x   | :white_check_mark: |
| < 0.9   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in NEURO, please report it by emailing **elvizekaj02@gmail.com**.

**Please do not:**
- Open a public GitHub issue for security vulnerabilities
- Disclose the vulnerability publicly before it's fixed

**What to include:**
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Response Timeline

- **Acknowledgment:** Within 48 hours
- **Initial assessment:** Within 1 week
- **Fix timeline:** Depends on severity, typically 2-4 weeks

## Security Best Practices

When using NEURO:

1. **Keep Ollama updated** - NEURO depends on Ollama for LLM inference
2. **Run locally** - NEURO is designed for local use; don't expose to public networks
3. **Review tool permissions** - Be cautious with file/shell operations
4. **Protect your data** - Knowledge bases contain your code patterns

## Scope

This policy applies to:
- The main NEURO repository
- Official NEURO modules (neuro-*)
- The elo-agi PyPI package

Third-party integrations are not covered.
