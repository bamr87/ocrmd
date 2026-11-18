# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| The latest release (or the default branch, for repos that do not cut releases) | ✅ |
| Anything older | ❌ |

Security fixes ship as patch releases through release-please.

## Reporting a vulnerability

**Do not open a public issue.** Report privately through GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) on this repository (**Security → Report a vulnerability**).

Please include the affected version or commit, what an attacker can do, and the steps to reproduce.

## Response window

You will get an acknowledgement **within 7 days**. We then agree a fix timeline with you, and credit you in the advisory unless you ask us not to.

## ocrmd notes

- The local web UI (`ocrmd serve`) binds to `127.0.0.1` by default and has no authentication. It is not designed for untrusted networks. `--host` exists only for a trusted LAN; do not expose it to the internet.
- Page images are sent to the configured vision provider (Claude by default, or xAI) unless you run with `--no-llm`. Nothing else is supposed to leave the machine.
- Credentials live in `.env` or the environment (`ANTHROPIC_API_KEY`, `CLAUDE_CODE_OAUTH_TOKEN`, `XAI_API_KEY`, and destination `bearer` / `api_key` in `ocrmd.toml`). Do not paste those values into issues, logs or pull requests.
