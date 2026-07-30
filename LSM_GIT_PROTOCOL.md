# LSM goal checkpoint protocol

This repository uses one verified Git commit per completed goal.

## Definition of a completed goal

A goal is complete only when it has a concrete acceptance criterion, the relevant implementation is finished, Ruff passes, the full pytest suite passes, and the result has reproducible evidence. A progress report without verification is not a completed goal.

## Required workflow

1. Read the current repository state and define one atomic goal.
2. Implement only that goal.
3. Run any goal-specific validation in addition to the standard checks.
4. List every file that belongs to the goal. Never use `git add -A` or `git add .`.
5. Run `scripts\goal_checkpoint.ps1` with the goal description and that explicit file list.
6. Record the returned commit hash in `PROGRESS.md`.
7. Continue with the next goal without waiting for human confirmation.

Example:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\goal_checkpoint.ps1 `
  -Goal "validate phase-routed evaluation" `
  -Paths "scripts/evaluate_phase_routed.py","tests/test_phase_routing.py","PROGRESS.md","README.md"
```

The script refuses to proceed when:

- another change is already staged;
- a declared path is ignored, outside the repository, missing, or resembles a secret;
- Ruff or pytest fails;
- the declared files contain no changes;
- Git is in detached-HEAD state;
- commit or push fails.

By default, a successful checkpoint is committed and pushed to the current branch on `origin`. Use a separate branch or worktree for every simultaneous agent. Do not let multiple agents write and commit from the same working tree.

## Root instruction for Chat + LSM

At the beginning of the session, read this file. Break the project into atomic, independently verifiable goals. After each goal passes its acceptance checks, invoke `scripts\goal_checkpoint.ps1` with an explicit list of only that goal's files. A goal is not complete until the script returns a commit hash and confirms the push. If the script refuses or a check fails, fix the issue instead of bypassing the guard. Never stage or commit unrelated work, generated models, logs, credentials, API keys, or another agent's changes. After a successful checkpoint, record the hash in `PROGRESS.md` and immediately continue to the next goal.
