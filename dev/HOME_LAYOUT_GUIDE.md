# Home Page Layout Guide

## 🎨 New 3-Column Layout

Your portfolio home page now features a modern 3-column layout matching the reference design:

```
┌─────────────────────────────────────────────────────────────┐
│                      Top Navigation Bar                      │
├──────────┬────────────────────────────────┬─────────────────┤
│          │                                │                 │
│  LEFT    │        MAIN CONTENT           │  RIGHT SIDEBAR  │
│ SIDEBAR  │                                │                 │
│          │   • Post 1 (with cover)       │  • Recently     │
│ Profile  │   • Post 2 (with cover)       │    Updated      │
│ Picture  │   • Post 3 (with cover)       │                 │
│          │   • ...                        │  • Trending     │
│ Name     │                                │    Tags         │
│ Bio      │   Pagination                   │                 │
│          │                                │                 │
│ Social   │                                │                 │
│ Icons    │                                │                 │
│          │                                │                 │
│ Nav Menu │                                │                 │
│  - Home  │                                │                 │
│  - Posts │                                │                 │
│  - Proj  │                                │                 │
│  - About │                                │                 │
│          │                                │                 │
└──────────┴────────────────────────────────┴─────────────────┘
```

## 📐 Layout Specifications

### Left Sidebar (300px)
- **Profile Card** with:
  - Circular profile picture (120px)
  - Your name and title
  - Professional bio/tagline
  - Social media icons (GitHub, LinkedIn, Email)
  - Vertical navigation menu with icons
- **Sticky positioning**: Stays visible while scrolling
- **Responsive**: Hidden on tablets and mobile (< 1200px)

### Main Content Area (Flexible)
- **Post listings** with:
  - Large cover images
  - Post title (clickable)
  - Post summary/description
  - Post metadata (date, reading time)
  - Hover effects for better UX
- **Shows posts and projects** combined
- **Pagination** at bottom (if needed)
- **Fully responsive**

### Right Sidebar (320px)
- **Recently Updated** widget:
  - Shows 5 most recent posts/projects
  - Clean list with hover effects
- **Trending Tags** widget:
  - Top 15 most used tags
  - Pill-style tag buttons
  - Hover animations
- **Sticky positioning**: Stays visible while scrolling
- **Responsive**: Becomes static on mobile (< 768px)

## 🎯 Key Features

### 1. Sticky Sidebars
Both sidebars use `position: sticky` to remain visible while scrolling through posts:
```css
.left-sidebar,
.right-sidebar {
    position: sticky;
    top: 80px;
    height: fit-content;
}
```

### 2. Responsive Design
- **Desktop (> 1200px)**: Full 3-column layout
- **Tablet (768px - 1200px)**: 2 columns (main + right sidebar)
- **Mobile (< 768px)**: Single column, stacked layout

### 3. Profile Navigation
Left sidebar includes a custom navigation menu with:
- Home
- Posts
- Projects
- About

Each item has an icon and highlights when active.

### 4. Widget System
Right sidebar features modular widgets that can be customized:
- Recently Updated (auto-populated with latest 5 posts)
- Trending Tags (auto-populated with top 15 tags)

## 📁 Files Modified/Created

### New Layout Files
```
layouts/
├── index.html                      # Custom home page layout
└── partials/
    ├── profile-sidebar.html       # Left sidebar profile card
    └── sidebar-widgets.html       # Right sidebar widgets
```

### Updated Files
1. **hugo.toml**
   - Disabled profile mode: `enabled = false`
   - Added `mainSections = ["posts", "projects"]`

2. **assets/css/extended/custom.css**
   - Added 3-column grid layout styles
   - Profile card styling
   - Sidebar widget styling
   - Responsive breakpoints
   - Enhanced dark mode colors

## 🎨 Styling Details

### Color Scheme (Dark Mode)
```css
Background:           #141414
Card Background:      #1e1e1e
Border:              #2d2d2d
Text:                #e4e6eb
Primary (Blue):      #4da3ff
Secondary (Gray):    #adb5bd
```

### Profile Card
- **Background**: Card background color
- **Border**: 1px solid with border color
- **Border radius**: 12px for modern look
- **Padding**: 2rem 1.5rem
- **Profile image**: 120px circle with primary color border

### Post Cards
- **Zero padding** on container (for full-width cover images)
- **Content padding**: 1.5rem
- **Hover effect**: Lifts up 4px with shadow
- **Cover images**: Full width, max height 300px
- **Border radius**: 12px
- **Smooth transitions**: 0.2s ease

### Sidebar Widgets
- **Background**: Card background
- **Spacing**: 1.5rem margin between widgets
- **List items**: Separated by subtle borders
- **Tags**: Pill-style with hover animations

## 🚀 How to Customize

### Change Profile Information
Edit `hugo.toml`:
```toml
[params]
  author = "Your Name"
  
[params.profileMode]
  subtitle = "Your professional tagline here"
  imageUrl = "profile.jpg"
```

### Modify Sidebar Widgets
Edit `layouts/partials/sidebar-widgets.html`:
- Change number of recent posts: `range first 5` → `range first 10`
- Change number of tags: `range first 15` → `range first 20`

### Adjust Layout Widths
Edit `assets/css/extended/custom.css`:
```css
.three-column-layout {
    grid-template-columns: 300px 1fr 320px;
    /* Change these values as needed */
}
```

### Add More Widgets
Create new widgets in `sidebar-widgets.html`:
```html
<div class="sidebar-widget your-widget-name">
  <h3>Widget Title</h3>
  <!-- Your widget content -->
</div>
```

## 📱 Mobile Experience

### Mobile Layout (< 768px)
- Single column layout
- Profile sidebar hidden (use top navigation instead)
- Main content takes full width
- Right sidebar appears below content
- All widgets remain functional

### Tablet Layout (768px - 1200px)
- Two columns: Main content + Right sidebar
- Profile sidebar hidden (use top navigation)
- Wider main content area
- Sticky right sidebar

## 🔧 Troubleshooting

### Profile Image Not Showing
1. Place your image in `static/profile.jpg`
2. Or update path in `hugo.toml`: `imageUrl = "your-image.jpg"`
3. Clear browser cache

### Sidebars Not Sticky
- Ensure your browser supports `position: sticky`
- Check that parent containers don't have `overflow: hidden`
- Adjust `top` value if header height changed

### Layout Breaking on Mobile
- Clear your browser cache
- Check responsive breakpoints in CSS
- Ensure viewport meta tag is present in theme

### Widgets Not Showing Content
- Ensure you have posts/projects published
- Check that `draft: false` in post frontmatter
- Verify `mainSections = ["posts", "projects"]` in config

## ✨ Benefits of This Layout

1. **Professional Appearance**: Matches modern portfolio designs
2. **Better Navigation**: Easy access to all sections via sidebar
3. **Content Discovery**: Recently updated and tags help visitors explore
4. **Sticky Elements**: Important info stays visible while scrolling
5. **Mobile Friendly**: Adapts beautifully to all screen sizes
6. **Fast Loading**: Minimal CSS, optimized images
7. **Easy to Maintain**: Modular widget system

## 🎯 Next Steps

### Recommended Actions
1. **Add your profile picture** to `static/profile.jpg`
2. **Customize the bio** in `hugo.toml`
3. **Test on mobile** to see responsive behavior
4. **Add more posts** to populate the homepage
5. **Customize colors** if desired in CSS

### Optional Enhancements
- Add a search widget to right sidebar
- Include categories widget
- Add a newsletter signup form
- Include recent comments (if using comments)
- Add a "Featured Posts" section

## 📊 Performance

- **Build time**: ~130ms
- **Total pages**: 96
- **Layout files**: 3 custom templates
- **CSS additions**: ~400 lines (organized and commented)
- **JavaScript**: None (pure CSS implementation)

---

**Status**: ✅ Home page layout successfully implemented

**Layout Type**: 3-Column Grid with Sticky Sidebars

**Compatibility**: Hugo 0.153.2+, PaperMod Theme

**Last Updated**: December 23, 2025

