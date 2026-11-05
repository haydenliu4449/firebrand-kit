---
name: deploy
description: Ship code changes to Cloud Batch - test, commit, push, and rebuild the container image so the next job runs the new code.
disable-model-invocation: true
---

Deploy the current code so Batch jobs use it.

1. `git status` and `git diff --stat`: list what changed. Nothing changed and
   no unpushed commits → say so and stop.
2. Tests, stop on any failure:
   - `python3 tests/test_negatives.py && python3 tests/test_reallabels.py`
   - `python3 tests/benchmark.py`: "mirrored" must equal "shipped defaults";
     compare precision and track recall with the numbers in
     `docs/notes/approach.md` and flag any drop.
   - `python3 -m py_compile` on every changed `.py` file.
3. If the change alters a workflow or flag, update `README.md` (and
   `CLAUDE.md` if a rule changed) in the same commit.
4. Commit with a message that says what changed and why; push.
5. Ask, then run `bash scripts/01-build_and_push.sh` (takes ~10 min). Report
   when the image is pushed.
6. Remind: jobs submitted before the build used the old image. If the change
   touches torch code (`train.py`, `infer.py`, `model.py`), suggest a `smoke`
   job (~$0.10) before a full train.
