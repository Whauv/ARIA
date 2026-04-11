# Contributing

## Workflow

1. Create a branch from the current working branch.
2. Make focused, non-destructive changes.
3. Run:

```powershell
python scripts\run_checks.py
```

4. Open a pull request with a clear summary and testing notes.

## Guidelines

- Preserve runtime behavior unless the change explicitly targets behavior.
- Keep environment-specific values out of source control.
- Prefer tests for deterministic logic changes.
- Document any ambiguous file placement in the pull request.
