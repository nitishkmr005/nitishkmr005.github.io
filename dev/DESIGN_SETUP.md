# Portfolio Design Configuration

This document outlines the design setup for your Hugo portfolio using the PaperMod theme, configured to match the modern, dark-themed design.

## ✅ Design Features Implemented

### 1. **Dark Theme as Default**
- Set `defaultTheme = "dark"` in `hugo.toml`
- Enhanced dark mode colors for better contrast
- Background: `#141414` (darker than default)
- Card background: `#1e1e1e`
- Border color: `#2d2d2d`

### 2. **3-Column Home Page Layout**
- **Left Sidebar**: Profile card with photo, bio, social icons, and navigation menu
- **Main Content**: List of posts and projects with cover images
- **Right Sidebar**: "Recently Updated" and "Trending Tags" widgets
- Sticky sidebars for better UX
- Fully responsive design (adapts to mobile/tablet)
- See [HOME_LAYOUT_GUIDE.md](HOME_LAYOUT_GUIDE.md) for details

### 3. **Cover Images for Posts**
All blog posts now have beautiful cover images from Unsplash:
- **Python Tips**: Code on screen imagery
- **MLOps Guide**: Infrastructure and pipeline visuals
- **ML Deployment**: Cloud infrastructure
- **Feature Engineering**: Data analytics charts
- **Project Workflow**: Project planning visuals

### 4. **Cover Images for Projects**
Project pages feature relevant imagery:
- **Sentiment Analysis**: NLP visualization
- **Customer Churn**: Analytics dashboard
- **Computer Vision**: Manufacturing quality control

### 5. **Enhanced CSS Styling**
Custom CSS in `assets/css/extended/custom.css` includes:
- Modern card hover effects with smooth transitions
- Enhanced dark mode styling
- Cover image styling with zoom effect on hover
- Sidebar widget styling (for "Recently Updated" and "Trending Tags")
- Tag cloud styling with hover effects
- Better typography and spacing
- Smooth scrolling and animations

### 6. **Menu Structure**
Clean navigation menu:
1. Home
2. Posts
3. Projects
4. About
5. Search
6. Contact

### 7. **Features Enabled**
- ✅ Reading time display
- ✅ Share buttons
- ✅ Post navigation links
- ✅ Breadcrumbs
- ✅ Code copy buttons
- ✅ Word count
- ✅ Table of contents
- ✅ Search functionality (Fuse.js)
- ✅ Syntax highlighting (Monokai theme)

## 📁 File Structure

```
portfolio/
├── hugo.toml                     # Main configuration
├── assets/css/extended/
│   └── custom.css               # Custom styling
├── content/
│   ├── posts/                   # Blog posts with cover images
│   ├── projects/                # Project pages with cover images
│   ├── about.md
│   ├── contact.md
│   └── search.md
├── static/
│   ├── profile.jpg              # Your profile picture (ADD THIS)
│   ├── images/
│   │   ├── posts/              # Optional local post images
│   │   └── projects/           # Optional local project images
│   ├── favicon.ico              # ADD FAVICONS
│   ├── favicon-16x16.png
│   ├── favicon-32x32.png
│   └── apple-touch-icon.png
└── themes/PaperMod/             # Theme (git submodule)
```

## 🎨 Color Scheme

### Dark Mode (Default)
- Primary: `#4da3ff` (Blue)
- Secondary: `#adb5bd` (Gray)
- Accent: `#51cf66` (Green)
- Background: `#141414` (Very Dark Gray)
- Card Background: `#1e1e1e` (Dark Gray)
- Border: `#2d2d2d` (Medium Gray)
- Text: `#e4e6eb` (Light Gray)

### Light Mode
- Primary: `#007bff` (Blue)
- Background: `#ffffff` (White)
- Card Background: `#f8f9fa` (Light Gray)
- Text: `#333` (Dark Gray)

## 📝 Next Steps

### 1. Add Your Profile Picture
Place your profile picture in the `static/` folder as `profile.jpg`:
- Recommended size: 400x400px minimum
- Format: JPG or PNG
- Should be optimized (< 500KB)

### 2. Generate Favicons
Use [Favicon Generator](https://realfavicongenerator.net/) to create:
- `favicon.ico`
- `favicon-16x16.png`
- `favicon-32x32.png`
- `apple-touch-icon.png`

Place these in the `static/` folder.

### 3. Update Social Links
Edit `hugo.toml` lines 82-92 to add your social media profiles:
```toml
[[params.socialIcons]]
  name = "linkedin"
  url = "https://linkedin.com/in/yourprofile"  # UPDATE THIS

[[params.socialIcons]]
  name = "github"
  url = "https://github.com/nitishkmr005"     # ALREADY SET

[[params.socialIcons]]
  name = "email"
  url = "mailto:your.email@example.com"       # UPDATE THIS
```

### 4. Optional: Replace Cover Images
Currently using Unsplash images for posts/projects. You can replace with your own:
1. Add images to `static/images/posts/` or `static/images/projects/`
2. Update the `cover.image` field in each post's frontmatter
3. Example:
   ```yaml
   cover:
     image: "/images/posts/my-custom-image.jpg"
     alt: "Description"
     caption: "Your caption"
   ```

### 5. Update Google Analytics (Optional)
Edit `hugo.toml` lines 94-96:
```toml
[params.analytics.google]
  SiteVerificationTag = "YOUR_GOOGLE_SITE_VERIFICATION_TAG"
  GoogleAnalyticsID = "G-XXXXXXXXXX"
```

## 🚀 Development

### Build the Site
```bash
hugo --cleanDestinationDir
```

### Run Local Server
```bash
hugo server -D
```

Then open http://localhost:1313 in your browser.

### Deploy
The site is configured to deploy to GitHub Pages:
- Base URL: `https://nitishkmr005.github.io/`
- Build directory: `public/`

## 🎯 Design Philosophy

This design follows modern web design principles:
- **Dark-first**: Dark mode as default for better readability
- **Minimalist**: Clean, distraction-free reading experience
- **Fast**: Optimized images and minimal CSS
- **Accessible**: Good contrast ratios and semantic HTML
- **Responsive**: Mobile-friendly design
- **Professional**: Showcases your work effectively

## 📚 Resources

- [PaperMod Documentation](https://github.com/adityatelange/hugo-PaperMod/wiki)
- [Hugo Documentation](https://gohugo.io/documentation/)
- [Unsplash](https://unsplash.com/) - Free high-quality images
- [TinyPNG](https://tinypng.com/) - Image optimization
- [Favicon Generator](https://realfavicongenerator.net/)

---

**Status**: ✅ Site configured and building successfully

**Last Updated**: December 23, 2025

