# Repository Agent Instructions

## Git Workflow (Mandatory)
- Never merge directly into `main`.
- Always create a working branch with prefix `codex/`.
- Always push the branch and open a Pull Request to `main`.
- Stop after creating the PR and wait for user approval before any merge.
- If a merge is explicitly requested, reconfirm once before merging.

## GitHub Auth (Mandatory)
- If GitHub CLI or git HTTPS auth fails, ask the user for a one-time `GH_TOKEN` and proceed with that token.
- Use the token only as an environment variable for the current command/session (do not store it in files or git config).
- After push/PR/issue operations are done, treat the token as expired and ask again next time if needed.

## Codex Review Trigger (Mandatory)
- Do not manually comment `@codex review` after push; GitHub Actions handles the trigger automatically.
- Add a manual `@codex review` comment only when the user explicitly requests it.
- If GitHub Actions auto-trigger is unavailable or failing, confirm with the user before manual triggering.

## Default Delivery Style
- Prefer small, reviewable commits.
- Include clear test steps in PR description.
- Keep README and docs updated when behavior changes.
