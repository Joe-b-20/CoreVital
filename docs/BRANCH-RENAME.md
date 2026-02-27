# Push a new branch, then merge to main (best practice)

Do **not** set this branch (or a renamed copy) as `main` directly. Push a **new branch**, open a PR, then merge into `main`.

```bash
# 1. Rename current branch to a clear feature name (optional but recommended)
git branch -m feature/docs-dashboard-cleanup

# 2. Push the new branch (creates it on remote)
git push -u origin feature/docs-dashboard-cleanup

# 3. On GitHub: open a Pull Request from feature/docs-dashboard-cleanup → main, review, then merge.
```

After the PR is merged, you can delete the feature branch locally and on remote:

```bash
git checkout main
git pull origin main
git branch -d feature/docs-dashboard-cleanup
git push origin --delete feature/docs-dashboard-cleanup   # optional: remove remote branch
```
