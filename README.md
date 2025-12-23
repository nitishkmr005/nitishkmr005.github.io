# Data Science Portfolio Website

A modern, fast, and beautiful portfolio website built with Hugo and the PaperMod theme. Perfect for data scientists, ML engineers, and AI researchers to showcase their work.

## 🌟 Features

- ✅ **3-Column Layout** - Professional home page with profile sidebar, main content, and widgets
- ✅ **Blog Posts** - Technical articles with code highlighting and math support
- ✅ **Projects Showcase** - Detailed project case studies with metrics and impact
- ✅ **Cover Images** - Beautiful featured images for all posts and projects
- ✅ **Sticky Sidebars** - Profile and widgets stay visible while scrolling
- ✅ **Recently Updated** - Widget showing latest content
- ✅ **Trending Tags** - Quick navigation to popular topics
- ✅ **Search Functionality** - Fast client-side search powered by Fuse.js
- ✅ **Tags & Categories** - Organize content for easy discovery
- ✅ **Contact Form** - Integrated with Formspree
- ✅ **Analytics** - Google Analytics support
- ✅ **Dark Theme** - Professional dark mode as default
- ✅ **Responsive Design** - Mobile-first and looks great on all devices
- ✅ **Fast Loading** - Optimized for performance
- ✅ **SEO Optimized** - Meta tags, sitemap, and robots.txt

## 📸 Demo

Visit the live site: [https://yourusername.github.io](https://yourusername.github.io)

## 🚀 Quick Start

### Prerequisites

- [Hugo](https://gohugo.io/) v0.153.2 or later (extended version)
- Git
- (Optional) Make for automation

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/portfolio.git
cd portfolio
```

2. **Initialize theme submodule**

```bash
git submodule update --init --recursive
```

Or using Make:

```bash
make setup
```

3. **Start development server**

```bash
hugo server -D
```

Or using Make:

```bash
make dev
```

4. **Open in browser**

Navigate to [http://localhost:1313](http://localhost:1313)

## 📝 Configuration

### Basic Settings

Edit `hugo.toml` to customize:

```toml
baseURL = 'https://yourusername.github.io/'
title = 'Your Name - Data Scientist'
[params]
  author = "Your Name"
  description = "Your portfolio description"
```

### Social Links

Update social links in `hugo.toml`:

```toml
[[params.socialIcons]]
  name = "linkedin"
  url = "https://linkedin.com/in/yourprofile"

[[params.socialIcons]]
  name = "github"
  url = "https://github.com/yourusername"

[[params.socialIcons]]
  name = "email"
  url = "mailto:your.email@example.com"
```

### Analytics

Add your Google Analytics ID:

```toml
[params.analytics.google]
  GoogleAnalyticsID = "G-XXXXXXXXXX"
```

### Contact Form

1. Sign up at [Formspree](https://formspree.io/)
2. Create a new form
3. Update `content/contact.md` with your form ID

## 📁 Project Structure

```
portfolio/
├── .github/
│   └── workflows/          # GitHub Actions for deployment
├── archetypes/            # Content templates
├── assets/
│   └── css/
│       └── extended/       # Custom CSS
├── content/
│   ├── posts/             # Blog posts
│   ├── projects/          # Project showcases
│   ├── about.md          # About page
│   ├── contact.md        # Contact page
│   └── search.md         # Search page
├── data/                  # Data files
├── layouts/              # Custom layout overrides
├── static/               # Static assets (images, files)
│   └── images/           # Image files
├── themes/
│   └── PaperMod/         # Hugo theme (submodule)
├── hugo.toml             # Site configuration
├── Makefile              # Automation commands
└── README.md
```

## 📚 Content Management

### Create a New Blog Post

```bash
hugo new content/posts/my-new-post.md
```

Or using Make:

```bash
make new-post TITLE="My New Post"
```

Edit the generated file in `content/posts/` and update the front matter:

```yaml
---
title: "My New Post"
date: 2025-12-23
draft: false
tags: ["Machine Learning", "Python"]
categories: ["Tutorial"]
description: "A brief description"
---

Your content here...
```

### Create a New Project

```bash
hugo new content/projects/my-project.md
```

Or using Make:

```bash
make new-project TITLE="My Project"
```

### Add Images

1. Place images in `static/images/`
2. Reference in markdown:

```markdown
![Alt text](/images/my-image.png)
```

For blog post covers, add to front matter:

```yaml
cover:
    image: "/images/posts/cover.jpg"
    alt: "Cover image"
    caption: "Image caption"
```

## 🎨 Customization

### Custom CSS

Add custom styles to `assets/css/extended/custom.css`

### Profile Picture

Add your profile picture as `static/profile.jpg` (400x400px recommended)

### Favicons

Generate favicons at [RealFaviconGenerator](https://realfavicongenerator.net/) and place in `static/`

Required files:
- `favicon.ico`
- `favicon-16x16.png`
- `favicon-32x32.png`
- `apple-touch-icon.png`

## 🚢 Deployment

### Deploy to GitHub Pages

1. **Create a GitHub repository** named `username.github.io`

2. **Enable GitHub Pages**
   - Go to repository Settings → Pages
   - Source: GitHub Actions

3. **Push your code**

```bash
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/username.github.io.git
git push -u origin main
```

4. **Automatic deployment**
   - GitHub Actions will automatically build and deploy
   - Site will be live at `https://yourusername.github.io`

### Deploy to Netlify

1. Push code to GitHub
2. Connect repository to [Netlify](https://netlify.com)
3. Configure build settings:
   - Build command: `hugo --gc --minify`
   - Publish directory: `public`
   - Hugo version: `0.153.2`

### Deploy to Vercel

1. Push code to GitHub
2. Import repository in [Vercel](https://vercel.com)
3. Configure:
   - Framework: Hugo
   - Build command: `hugo --gc --minify`
   - Output directory: `public`

## 🛠️ Makefile Commands

```bash
make help           # Show all available commands
make dev            # Start development server
make build          # Build for production
make clean          # Clean generated files
make new-post       # Create new blog post
make new-project    # Create new project
make test           # Test build
make deploy         # Deploy to GitHub Pages
make stats          # Show site statistics
```

## 📊 Performance

This site is optimized for:
- ⚡ **Lighthouse Score**: 95+ (Performance, Accessibility, Best Practices, SEO)
- 📦 **Bundle Size**: < 100KB (minified)
- 🚀 **First Contentful Paint**: < 1s
- 🎯 **Time to Interactive**: < 2s

## 🔧 Troubleshooting

### Theme not loading

```bash
git submodule update --init --recursive
```

### Hugo version mismatch

Install Hugo extended v0.153.2+:

```bash
brew install hugo
hugo version
```

### Build fails

1. Check Hugo version: `hugo version`
2. Clean generated files: `make clean` or `rm -rf public resources`
3. Rebuild: `make build`

## 📝 Best Practices

1. **Regular Updates**
   - Write consistently (1-2 posts per month)
   - Keep projects updated with latest work
   - Update About page with new skills/experience

2. **SEO**
   - Write descriptive titles and descriptions
   - Use relevant tags and categories
   - Add alt text to images
   - Keep URLs clean and descriptive

3. **Content Quality**
   - Focus on depth over breadth
   - Include code examples and visualizations
   - Share real results and learnings
   - Proofread before publishing

4. **Performance**
   - Optimize images before uploading (< 500KB)
   - Use code syntax highlighting sparingly
   - Test on multiple devices

## 🤝 Contributing

Found a bug or want to suggest an improvement?

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Hugo](https://gohugo.io/) - Static site generator
- [PaperMod](https://github.com/adityatelange/hugo-PaperMod) - Hugo theme
- [Formspree](https://formspree.io/) - Contact form
- [Fuse.js](https://fusejs.io/) - Search functionality

## 📧 Contact

- **Website**: [https://yourusername.github.io](https://yourusername.github.io)
- **Email**: your.email@example.com
- **LinkedIn**: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- **GitHub**: [github.com/yourusername](https://github.com/yourusername)

---

**Built with ❤️ for the data science community**

