.PHONY: help install dev build serve clean deploy

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install Hugo (macOS with Homebrew)
	@echo "Installing Hugo..."
	@brew install hugo

dev: ## Start local development server
	@echo "Starting Hugo development server..."
	@hugo server -D --bind 0.0.0.0 --baseURL http://localhost

serve: dev ## Alias for dev

build: ## Build the site for production
	@echo "Building site..."
	@hugo --gc --minify

clean: ## Clean generated files
	@echo "Cleaning up..."
	@rm -rf public resources

new-post: ## Create a new blog post (usage: make new-post TITLE="My Post Title")
	@if [ -z "$(TITLE)" ]; then \
		echo "Usage: make new-post TITLE=\"Your Post Title\""; \
		exit 1; \
	fi
	@hugo new content/posts/$(shell echo "$(TITLE)" | tr '[:upper:]' '[:lower:]' | tr ' ' '-').md
	@echo "Created new post!"

new-project: ## Create a new project (usage: make new-project TITLE="My Project")
	@if [ -z "$(TITLE)" ]; then \
		echo "Usage: make new-project TITLE=\"Your Project Title\""; \
		exit 1; \
	fi
	@hugo new content/projects/$(shell echo "$(TITLE)" | tr '[:upper:]' '[:lower:]' | tr ' ' '-').md
	@echo "Created new project!"

test: ## Test the build locally
	@echo "Testing build..."
	@hugo --gc --minify
	@echo "Build successful!"

check: ## Check for broken links
	@echo "Checking for broken links..."
	@hugo --gc --minify
	@echo "Build complete. Manually check links or use htmltest."

deploy: build ## Deploy to GitHub Pages (via git push)
	@echo "Committing and pushing changes..."
	@git add .
	@git commit -m "Update site" || true
	@git push origin main
	@echo "Pushed to GitHub. GitHub Actions will deploy automatically."

submodule-update: ## Update theme submodule
	@echo "Updating theme submodule..."
	@git submodule update --remote --merge

setup: ## Initial setup for the project
	@echo "Setting up project..."
	@git submodule update --init --recursive
	@echo "Setup complete!"

stats: ## Show site statistics
	@echo "Counting posts and projects..."
	@echo "Posts: $$(find content/posts -name '*.md' -not -name '_index.md' | wc -l | tr -d ' ')"
	@echo "Projects: $$(find content/projects -name '*.md' -not -name '_index.md' | wc -l | tr -d ' ')"
	@echo "Total words: $$(find content -name '*.md' -not -name '_index.md' -exec cat {} \; | wc -w | tr -d ' ')"

preview: ## Build and open site in browser
	@make build
	@open public/index.html
