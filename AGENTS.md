# AGENTS.md

## Branches
- `develop` is the base for every PR. `main` only receives tested releases from `develop`.
- No force push or history rewriting on `develop` or `main`.

## Automatic merge (PRs into `develop` only)
An agent may merge a PR into `develop` without a human when all of these hold:
1. A reviewer agent, separate from the author, approved it with no critical, high or medium findings.
2. The `test` workflow is green on the PR head.
3. The PR does not conflict with `develop`.
4. The PR does not weaken tests (no deleted, loosened or skipped tests).

Merges into `main` are always done by a human.

## Checks before pushing
- `python3 -m pytest -q test` (no GPU or weights needed).
- Changes to nodes or the image: build with `docker compose build` on a CUDA machine and run the
  node against a real image or rosbag.
- Never publish to drive topics (`/cmd_vel`, `/set_position`, ...) from tests or tools in this repo.
