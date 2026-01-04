# Codebase Overview

This is a **Hugo-based static portfolio website** for data science and ML work, deployed on GitHub Pages.

## Tech Stack
- **Framework**: Hugo v0.153.2 (static site generator)
- **Theme**: PaperMod (as git submodule) with extensive custom styling
- **Content**: Markdown files
- **Deployment**: GitHub Actions (auto-deploy on push to master)

## Structure

### Core Directories
- `content/` - All site content
  - `posts/` - 5 blog posts on ML/data science topics
  - `projects/` - 3 project case studies
  - `about.md` - Professional resume page
  - `contact.md`, `search.md`
- `layouts/` - Custom template overrides
  - Three-column homepage layout
  - Custom partials (profile sidebar, widgets)
- `assets/css/extended/custom.css` - 1738 lines of custom styling
- `static/` - Images and static assets
- `themes/PaperMod/` - Base theme (submodule)

## Key Features

### Design
- Three-column responsive layout (profile sidebar | main content | widgets)
- Dark/light theme toggle
- Comprehensive CSS variable system for theming
- Smooth animations and hover effects
- Mobile-optimized with breakpoints at 1200px and 768px

### Functionality
- Client-side search (Fuse.js)
- Reading time & word count
- Table of contents
- Share buttons
- Tag cloud with trending tags
- Timeline layout support

### Automation
- `Makefile` with commands for dev, build, new posts/projects
- GitHub Actions workflow for automated deployment
- Minified output for production

## Configuration
`hugo.toml` - Main config with site metadata, theme settings, SEO options, social links (LinkedIn, GitHub)

## Summary
This is a well-structured, production-ready portfolio showcasing data science work with professional styling and automated deployment.

---

## Detailed Directory Structure

```
portfolio/
├── .github/workflows/           # GitHub Actions automation
│   ├── deploy.yml             # Auto-deploy to GitHub Pages on push to master
│   └── test.yml               # Build testing workflow
│
├── archetypes/                # Hugo content templates (defaults for new posts/projects)
│
├── assets/css/extended/       # Custom CSS overrides
│   └── custom.css            # 1738 lines of comprehensive styling
│
├── content/                   # Main content directory
│   ├── posts/                # Blog articles (~5 posts on ML/Data Science topics)
│   ├── projects/             # Project showcases (~3 project case studies)
│   ├── about.md              # Professional resume & about page
│   ├── contact.md            # Contact information page
│   └── search.md             # Search functionality page
│
├── data/                      # Data files for templates
│
├── layouts/                   # Custom Hugo template overrides
│   ├── index.html            # Three-column homepage layout
│   ├── _default/single.html  # Single post/page template
│   ├── posts/list.html       # Posts listing page
│   ├── projects/list.html    # Projects listing page
│   └── partials/             # Reusable template components
│       ├── profile-sidebar.html        # Left sidebar with profile card
│       ├── sidebar-widgets.html        # Right sidebar with trending tags & recent updates
│       └── header.html                 # Navigation header
│
├── static/                    # Static assets
│   ├── images/               # Image files for content
│   └── profile.JPG           # Profile picture
│
├── themes/PaperMod/          # Hugo theme (git submodule)
│
├── public/                    # Generated static site output (build artifact)
│
├── hugo.toml                  # Main configuration file
├── Makefile                   # Development automation commands
├── README.md                  # Comprehensive documentation
└── LICENSE                    # MIT License
```

## Site Configuration (hugo.toml)

- **Base URL**: `https://nitishkmr005.github.io/`
- **Title**: "Data Science Portfolio"
- **Language**: English (en-us)
- **Copyright**: 2025 Nitishkumar Harsoor
- **Theme**: PaperMod
- **Default theme**: Dark mode with toggle enabled
- **Features**:
  - Reading time display
  - Share buttons
  - Table of contents
  - Code copy buttons
  - Word count display
  - RSS feed
  - Robots.txt and sitemaps for SEO
- **Minification**: Enabled
- **Main sections**: Posts and Projects
- **Social links**: LinkedIn, GitHub
- **Search**: Fuse.js configuration
- **Syntax highlighting**: Monokai style

## Custom Styling System (custom.css)

### CSS Variables - Theming
- Typography scale (from xs to 4xl)
- Spacing scale (xs to 2xl)
- Shadow system (4 levels)
- Transition durations (fast, base, slow)
- Color palette (primary: #007bff, accent: #28a745, etc.)
- Dark mode color overrides

### Three-Column Layout (Homepage)
- Left sidebar (300px): Profile card with sticky positioning
- Main content (1fr): Posts listing with pagination
- Right sidebar (320px): Widgets (recently updated, trending tags)
- Responsive: Collapses to single column on screens < 1200px
- Mobile optimizations for touch devices

### UI Components
- **Profile Card**: Circular image, bio, social icons, navigation menu
- **Post Cards**: Featured images, title, summary, hover effects with gradient animation
- **Sidebar Widgets**: Styled with gradient backgrounds, hover effects
- **Tag Cloud**: Responsive, interactive tags with hover states
- **Navigation**: Active states, smooth transitions, icon support
- **Timeline Layout**: Vertical timeline for chronological content display

### Interactive Features
- Smooth hover effects with transforms and shadows
- Gradient text animations
- Pulsing timeline dots
- Backdrop blur on navigation
- Image zoom on hover
- Smooth scrolling behavior

### Responsive Design
- Mobile-first approach
- Breakpoints: 1200px, 768px
- Flexible layouts that adapt to screen size
- Touch-optimized navigation

## Content Structure

### Posts (content/posts/)
1. data-science-project-workflow.md
2. feature-engineering-techniques.md
3. getting-started-with-mlops.md
4. ml-model-deployment-best-practices.md
5. python-data-science-tips.md

### Projects (content/projects/)
1. customer-churn-prediction.md
2. computer-vision-defect-detection.md
3. sentiment-analysis-nlp.md

### Special Pages
- about.md - Professional resume with 12+ years experience
- contact.md - Contact information
- search.md - Client-side search page

## Build & Deployment

### GitHub Actions Workflow (deploy.yml)
- **Trigger**: Push to `master` branch or manual workflow dispatch
- **Steps**:
  1. Checkout code with git submodules
  2. Setup Hugo v0.153.2 (extended)
  3. Configure GitHub Pages settings
  4. Build site with Hugo (minified, garbage collected)
  5. Upload artifact to GitHub Pages
  6. Deploy to GitHub Pages
- **Permissions**: Read contents, write pages, token auth
- **Concurrency**: Only one deployment at a time

### Makefile Commands
- `make dev` - Start local development server
- `make build` - Production build
- `make clean` - Remove generated files
- `make new-post TITLE="..."` - Create new blog post
- `make new-project TITLE="..."` - Create new project
- `make test` - Test build locally
- `make deploy` - Push to GitHub (auto-deploys via Actions)
- `make setup` - Initialize git submodules
- `make stats` - Show content statistics

## Performance & SEO Features

- Minified CSS and HTML output
- Responsive images with lazy loading support
- Meta tags and description
- Robots.txt and sitemap generation
- RSS feeds for all sections
- Social media card support
- Code syntax highlighting
- Open Graph support
- Expected Lighthouse Scores: 95+ (Performance, Accessibility, Best Practices, SEO)

## Architecture Patterns

1. **Modular CSS**: CSS variables for theming, separated concerns, responsive utilities
2. **Template Inheritance**: Base PaperMod theme with custom overrides
3. **Component-Based**: Reusable partials for sidebar, profile, widgets
4. **Static Generation**: All pages pre-rendered to static HTML at build time
5. **Git-Driven Workflow**: Content updates trigger automatic deployments
