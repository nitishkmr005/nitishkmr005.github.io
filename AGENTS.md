# Repository Guidelines

## Project Structure & Module Organization
- Hugo site with PaperMod theme. Core content lives in `content/posts/` and `content/projects/`.
- Layout overrides live in `layouts/`, theme source in `themes/PaperMod/`.
- Static assets (images, files) go in `static/`, typically `static/images/`.
- Generated outputs: `public/` (build) and `resources/` (Hugo cache).
- Custom styling is in `assets/css/extended/` (see `README.md` for structure map).

## Build, Test, and Development Commands
- `make dev` or `hugo server -D`: run local dev server with drafts.
- `make build`: production build (`hugo --gc --minify`).
- `make test`: build-only check to ensure the site compiles.
- `make clean`: remove `public/` and `resources/`.
- `make new-post TITLE="My Post"`: create a new post in `content/posts/`.
- `make new-project TITLE="My Project"`: create a new project in `content/projects/`.
- `make setup`: init/update the theme submodule.

## Coding Style & Naming Conventions
- Markdown content files use Hugo front matter (YAML/TOML). Keep keys consistent: `title`, `date`, `draft`, `tags`, `categories`, `description`.
- Image references should be absolute from `static/`, e.g. `![Alt](/images/posts/foo.png)`.
- Filenames are lowercase with hyphens (e.g., `my-new-post.md`).
- CSS uses standard formatting; keep changes scoped in `assets/css/extended/`.

## Testing Guidelines
- No dedicated test framework. Use `make test` or `hugo --gc --minify` to validate builds.
- Check for broken links manually or via `make check` (builds + reminder to verify).

## Commit & Pull Request Guidelines
- Recent commits use short, lowercase, descriptive messages (e.g., “updated article…”, “added new images…”). Follow this style.
- PRs should include a brief summary, list of affected sections (e.g., `content/posts/...`), and screenshots for layout/UI changes.

## Configuration Tips
- Site settings live in `hugo.toml` (base URL, social links, analytics).
- Keep generated folders (`public/`, `resources/`) out of source edits unless explicitly required.
