# Contributing

## Commit policy

- One logical goal produces one verified commit.
- Do not create a follow-up commit solely to write the previous commit hash into a tracked file. The Git commit and the hash printed by `scripts/goal_checkpoint.ps1` are the provenance record.
- Use Conventional Commit types that describe the change: `feat`, `fix`, `test`, `docs`, `refactor`, `perf`, `build`, `ci`, or `chore`.
- Keep subjects readable rather than optimizing for commit count.

Examples:

```text
fix(training): reject boolean evaluation episode counts
test(controller): cover request integer boundaries
docs(progress): record desktop migration results
ci: run Ruff and pytest on pull requests
```

Create a checkpoint with an explicit type and optional scope:

```powershell
scripts/goal_checkpoint.ps1 `
  -Type fix `
  -Scope training `
  -Goal "reject invalid evaluation episode counts" `
  -Paths src/stickman_rl/training.py,tests/test_training_smoke.py,PROGRESS.md
```

## Validation

Before committing:

1. Reproduce the problem with a failing regression when applicable.
2. Make the smallest scoped change that fixes it.
3. Run focused Ruff and pytest checks.
4. Run the complete Ruff and pytest suites.
5. Scan changed text files for sensitive information.

Do not commit when tests fail, the index contains unrelated files, secrets are detected, merge state is unclear, or the change cannot be explained precisely.

## Branches and pull requests

Use a short-lived branch and pull request for changes that alter behavior, span multiple subsystems, change experiment methodology, or require design discussion. A pull request should state:

- the reproducible problem;
- the failing regression or measured baseline;
- the minimal implementation change;
- the focused and full verification results.

Small documentation or repository-maintenance changes may be committed directly only when the scope is obvious and the complete checks pass.

## AI-assisted work

Agent assistance is allowed, but the repository owner should be able to explain the changed code, the regression, and the verification evidence. Commit volume is not a project metric; understandable decisions and reproducible results are.
