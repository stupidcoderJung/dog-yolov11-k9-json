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
- After each push to an open PR branch, request review by commenting `@codex review`.
- Do this by default without waiting for a separate user instruction.
- Skip only when the user explicitly asks not to trigger Codex review.

## Default Delivery Style
- Prefer small, reviewable commits.
- Include clear test steps in PR description.
- Keep README and docs updated when behavior changes.
