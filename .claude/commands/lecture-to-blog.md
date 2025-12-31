# Generate Blog Post from LLM Lecture

Generate a technical blog post from lecture document and transcript.

## Arguments

- `$LECTURE_NUM`: Lecture number (e.g., "2" for lecture2)

## Instructions

You are generating a technical ML/AI blog post from lecture materials. Follow these steps:

### Step 1: Locate Source Materials

Find and read the source files:
- **Document**: `data/llm/documents/lecture$LECTURE_NUM.docx` (or similar naming)
- **Transcript**: `data/llm/transcripts/lecture$LECTURE_NUM.txt` (or similar naming)

If files are named differently, search for files containing "lecture" and "$LECTURE_NUM" in the respective directories.

### Step 2: Extract Key Content

From the slides and transcript, identify:
1. **Core problem being solved** — What pain point does this lecture address?
2. **Key concepts and techniques** — Extract the main ideas with specific numbers/benchmarks
3. **Implementation details** — Any code, architectures, or configs mentioned
4. **Practical applications** — Real-world use cases and constraints

### Step 3: Write the Blog Post

Follow this writing style strictly:

**VOICE & TONE:**
- Direct and conversational — like explaining to a smart colleague, not lecturing
- Confident without hedging ("does" not "might"), but honest about limitations
- Open with concrete problems, not definitions ("Running a 70B model on a single GPU? Not happening.")

**STRUCTURE:**
- Problem → Quick Theory → Working Code → Practical Takeaways
- Cover theory "just enough to understand," then immediately show implementation
- Include specific numbers always: memory sizes, benchmark results, config values

**FORMATTING:**
- Use tables for comparisons across multiple items
- Use Mermaid diagrams for architecture flows and decision trees
- Code blocks with file names as comments and inline explanations
- Bold key terms on first introduction only
- Blockquotes for critical warnings

**TECHNICAL DEPTH:**
- Show real model names, real GPU types, real numbers — no abstractions
- Explicitly list constraints and "what this doesn't solve"
- End with decision frameworks, quick reference commands, and checklists

**AVOID:**
- Fluff intros ("In this article, we will explore...")
- Walls of text — break up with code, diagrams, formatting
- Unsupported performance claims — back with benchmarks
- Passive voice — use active ("Each GPU computes" not "is computed by")

**TARGET:** Reader finishes knowing WHY it matters, HOW it works, has WORKING CODE to run, and knows WHEN to use it vs alternatives.

### Step 4: Generate Blog File

Create the blog post as a markdown file:
- **Location**: `content/posts/`
- **Filename**: Use kebab-case based on the lecture topic (e.g., `attention-mechanisms-deep-dive.md`)
- Include appropriate frontmatter (title, date, tags, description)

### Step 5: Generate Thumbnail

Create a thumbnail image for the blog post:
- Generate a visually appealing thumbnail that represents the lecture topic
- Save to the appropriate assets/images directory used by the portfolio
- Reference the thumbnail in the blog post frontmatter

### Step 6: Update Portfolio Ordering

Ensure the new post appears at the top:
- **Home section**: Update the home page to display this post first in the recent posts/featured section
- **Posts section**: Verify posts are sorted by date (newest first) or manually update ordering if needed
- Check frontmatter date is set to current date to ensure proper sorting

### Step 7: Confirm Output

After creating the file:
1. Confirm the blog post file was created successfully
2. Confirm the thumbnail was generated and linked
3. Verify the post appears at the top in both home and posts sections
4. Summarize the key topics covered
5. Note any areas where the source material was unclear or could use additional research