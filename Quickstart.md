# Quickstart Guide

Get your portfolio website up and running in 10 minutes!

## 🎨 Design Status

Your portfolio is pre-configured with a modern, dark-themed design using the PaperMod theme:
- ✅ Dark mode as default
- ✅ Cover images for all posts and projects
- ✅ Custom CSS styling with smooth animations
- ✅ Profile mode enabled
- ✅ Search functionality

For detailed design information, see [DESIGN_SETUP.md](DESIGN_SETUP.md).

## Prerequisites Checklist

- [ ] Hugo installed (v0.153.2+)
- [ ] Git installed
- [ ] GitHub account (for deployment)
- [ ] Text editor (VS Code, Sublime, etc.)

## Step 1: Setup (2 minutes)

### Install Hugo (if not installed)

**macOS:**
```bash
brew install hugo
```

**Windows:**
```bash
choco install hugo-extended
```

**Linux:**
```bash
snap install hugo --channel=extended
```

Verify installation:
```bash
hugo version
# Should show: hugo v0.153.2+extended
```

### Clone and Initialize

```bash
# Clone the repository
git clone https://github.com/yourusername/portfolio.git
cd portfolio

# Initialize theme
git submodule update --init --recursive

# Or use Make
make setup
```

## Step 2: Personalize (5 minutes)

### 1. Update Site Configuration

Edit `hugo.toml`:

```toml
baseURL = 'https://yourusername.github.io/'
title = 'Your Name - Data Scientist'

[params]
  author = "Your Name"
  description = "Your tagline here"
```

### 2. Update About Page

Edit `content/about.md`:

- Replace placeholder name
- Add your experience
- Update skills and education
- Add your contact information

### 3. Update Social Links

In `hugo.toml`, find `[[params.socialIcons]]` and update:

```toml
[[params.socialIcons]]
  name = "linkedin"
  url = "https://linkedin.com/in/YOURPROFILE"

[[params.socialIcons]]
  name = "github"
  url = "https://github.com/YOURUSERNAME"

[[params.socialIcons]]
  name = "email"
  url = "mailto:YOUR.EMAIL@example.com"
```

### 4. Add Your Profile Picture

- Create or find a professional photo (400x400px)
- Save as `static/profile.jpg`
- Or update path in `hugo.toml` if using different name

### 5. Generate Favicons

1. Go to [RealFaviconGenerator](https://realfavicongenerator.net/)
2. Upload your logo/profile pic
3. Download favicon package
4. Extract to `static/` folder

## Step 3: Preview Locally (1 minute)

Start development server:

```bash
hugo server -D
# Or: make dev
```

Open browser to [http://localhost:1313](http://localhost:1313)

You should see your portfolio with sample content!

## Step 4: Deploy to GitHub Pages (2 minutes)

### Create GitHub Repository

1. Go to [GitHub](https://github.com) and create new repository
2. Name it: `yourusername.github.io` (replace with your GitHub username)
3. Don't initialize with README

### Enable GitHub Pages

1. Go to repository Settings → Pages
2. Under "Build and deployment":
   - Source: **GitHub Actions**

### Push Your Code

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit"

# Add remote and push
git remote add origin https://github.com/yourusername/yourusername.github.io.git
git branch -M main
git push -u origin main
```

### Wait for Deployment

1. Go to your repository's "Actions" tab
2. Watch the "Deploy Hugo Site to GitHub Pages" workflow
3. Once complete (1-2 minutes), your site is live!

Visit: `https://yourusername.github.io`

## What's Next?

### Content Creation

#### Write Your First Blog Post

```bash
make new-post TITLE="My First Post"
# Or: hugo new content/posts/my-first-post.md
```

Edit `content/posts/my-first-post.md`:

```yaml
---
title: "My First Post"
date: 2025-12-23
draft: false
tags: ["Machine Learning"]
description: "My first blog post"
---

## Introduction

Your content here...
```

#### Add Your First Project

```bash
make new-project TITLE="My Project"
# Or: hugo new content/projects/my-project.md
```

### Customize Further

#### Add Google Analytics

1. Create Google Analytics account
2. Get tracking ID (G-XXXXXXXXXX)
3. Update `hugo.toml`:

```toml
[params.analytics.google]
  GoogleAnalyticsID = "G-XXXXXXXXXX"
```

#### Setup Contact Form

1. Sign up at [Formspree](https://formspree.io/)
2. Create new form
3. Get form ID
4. Update `content/contact.md`:

```html
<form action="https://formspree.io/f/YOUR_FORM_ID" method="POST">
```

#### Custom Styling

Edit `assets/css/extended/custom.css` to add your custom styles.

## Common Tasks

### Add a New Post

```bash
make new-post TITLE="Post Title"
```

### Add a New Project

```bash
make new-project TITLE="Project Name"
```

### Test Build

```bash
make test
```

### Deploy Updates

```bash
git add .
git commit -m "Update content"
git push origin main
# GitHub Actions automatically deploys
```

### View Statistics

```bash
make stats
```

## Troubleshooting

### Issue: Theme not loading

**Solution:**
```bash
git submodule update --init --recursive
```

### Issue: Site not building

**Solution:**
```bash
make clean
make build
```

### Issue: Images not showing

**Solution:**
- Ensure images are in `static/` folder
- Use absolute paths: `/images/pic.jpg`
- Check file names (case-sensitive)

### Issue: Changes not showing on live site

**Solution:**
1. Check GitHub Actions workflow completed successfully
2. Clear browser cache (Cmd+Shift+R / Ctrl+Shift+R)
3. Wait 1-2 minutes for CDN to update

## Tips for Success

1. **Start Small**
   - Begin with 2-3 quality posts
   - Add projects as you complete them
   - Expand over time

2. **Be Consistent**
   - Write at least 1 post per month
   - Keep projects updated
   - Regular commits look good!

3. **Optimize Content**
   - Use descriptive titles
   - Add relevant tags
   - Include code examples
   - Show real results

4. **Share Your Work**
   - Post on LinkedIn
   - Share in data science communities
   - Include in job applications
   - Add to email signature

## Quick Reference

### Useful Commands

```bash
make dev              # Start dev server
make build            # Build for production
make new-post         # New blog post
make new-project      # New project
make clean            # Clean build files
make stats            # Show statistics
```

### Folder Structure

```
content/
├── posts/           # Your blog posts
├── projects/        # Your projects
├── about.md         # About page
└── contact.md       # Contact page

static/
├── images/          # Your images
└── profile.jpg      # Profile picture
```

### Front Matter Template

**For Posts:**
```yaml
---
title: "Title"
date: 2025-12-23
tags: ["Tag1", "Tag2"]
categories: ["Category"]
description: "Description"
---
```

**For Projects:**
```yaml
---
title: "Project Title"
date: 2025-12-23
tags: ["ML", "Python"]
description: "Project description"
weight: 1
---
```

## Resources

- [Hugo Documentation](https://gohugo.io/documentation/)
- [PaperMod Wiki](https://github.com/adityatelange/hugo-PaperMod/wiki)
- [Markdown Guide](https://www.markdownguide.org/)
- [GitHub Pages Docs](https://docs.github.com/en/pages)

## Need Help?

- 📖 Check [README.md](README.md) for detailed docs
- 🐛 [Open an issue](https://github.com/yourusername/portfolio/issues)
- 💬 Ask in discussions

---

**You're all set! Start creating amazing content! 🚀**

