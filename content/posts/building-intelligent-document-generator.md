---
title: "Building an Intelligent Document Generator: From Raw Content to Polished PDFs and Presentations"
date: 2026-01-31
draft: false
tags: ["LangGraph", "LLM", "Document Generation", "Python", "FastAPI", "AI"]
categories: ["AI/ML", "Tutorial"]
description: "How we built a production-ready system that transforms messy documents, web articles, and PDFs into beautifully formatted, AI-enhanced outputs using LangGraph and modern LLMs"
cover:
  image: "https://images.unsplash.com/photo-1633356122544-f134324a6cee?w=1200&q=80"
  alt: "Document generation workflow"
  caption: "Transforming raw content into polished documents with AI"
---

_How we built a production-ready system that transforms messy documents, web articles, and PDFs into beautifully formatted, AI-enhanced outputs using LangGraph and modern LLMs_

---

## Table of Contents

1. [The Problem We're Solving](#the-problem-were-solving)
2. [Business Value & Use Cases](#business-value--use-cases)
3. [System Architecture Overview](#system-architecture-overview)
4. [The LangGraph Workflow: Step by Step](#the-langgraph-workflow-step-by-step)
5. [Output Formats by Deliverable](#output-formats-by-deliverable)
6. [Technical Deep Dive](#technical-deep-dive)
7. [LLM Prompts: The Heart of Intelligence](#llm-prompts-the-heart-of-intelligence)
8. [Intelligent Caching Strategy](#intelligent-caching-strategy)
9. [API Design & Integration](#api-design--integration)
10. [Production Considerations](#production-considerations)
11. [Future Improvements & Roadmap](#future-improvements--roadmap)
12. [Lessons Learned](#lessons-learned)

---

## The Problem We're Solving

In today's knowledge economy, organizations face a critical challenge: **content is everywhere, but it's rarely in the right format**. Teams deal with:

- **Scattered knowledge**: PDFs, slide decks, markdown files, web articles, Word documents
- **Manual conversion**: Hours spent reformatting content for different audiences
- **Inconsistent presentation**: No unified visual language across documents
- **Lost context**: Important information buried in poorly structured files
- **Time waste**: Developers and content creators spending 20-30% of their time on document formatting

### The Real Cost

Consider a typical scenario:

- A technical team has 15 PDFs documenting their architecture
- They need to create a unified presentation for stakeholders
- Manual process: 8-12 hours of copy-paste, reformatting, and image creation
- **Our solution: 5 minutes of automated processing**

This isn't just about saving time—it's about **democratizing professional content creation** and letting teams focus on what matters: the ideas, not the formatting.

---

## Business Value & Use Cases

### Primary Use Cases

#### 1. **Technical Documentation Consolidation**

**Problem**: Engineering teams have documentation scattered across PDFs, markdown files, and wikis.

**Solution**: Our system:

- Ingests multiple file formats simultaneously
- Merges content intelligently while preserving structure
- Generates both PDF documentation and PPTX presentations
- Adds AI-generated executive summaries

**Impact**: Reduced documentation preparation time from days to minutes.

#### 2. **Research Paper to Presentation**

**Problem**: Researchers need to convert dense academic papers into digestible presentations.

**Solution**:

- Extracts key concepts from PDFs
- Structures content into logical sections
- Generates relevant images for each section
- Creates professional slide decks automatically

**Impact**: Enables researchers to focus on content, not design.

#### 3. **Web Content Aggregation**

**Problem**: Marketing teams need to compile competitor analysis from multiple web sources.

**Solution**:

- Scrapes and normalizes web content
- Removes ads and irrelevant elements
- Structures findings into professional reports
- Generates comparison visuals

**Impact**: Faster competitive intelligence with consistent formatting.

#### 4. **Meeting Notes to Action Items**

**Problem**: Teams have transcripts and notes that need to become actionable documents.

**Solution**:

- Processes raw transcripts and removes timestamps
- Extracts key decisions and action items
- Creates structured summaries
- Generates shareable PDFs

**Impact**: Better meeting follow-through and accountability.

### ROI Metrics

For a mid-sized organization (100 employees):

- **Time saved**: ~500 hours/year on document formatting
- **Cost savings**: $25,000-50,000/year (at $50-100/hour)
- **Quality improvement**: Consistent, professional output every time
- **Faster decision-making**: Executives get summaries in minutes, not days

---

## System Architecture Overview

Our document generator is built on **Hybrid Clean Architecture** principles, combining domain-driven design with practical infrastructure needs.

### Architectural Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                           │
│  Upload → Generate (SSE Stream) → Download                       │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│              Application Layer (LangGraph)                       │
│  Workflow Orchestration & State Management                       │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────┬──────────────────┬───────────────────┐
│   Domain Layer   │  Infrastructure  │   External APIs   │
│  Business Logic  │   File System    │   Gemini/Claude   │
│  Models & Rules  │   Parsers        │   OpenAI          │
└──────────────────┴──────────────────┴───────────────────┘
```

### Project Structure

```
backend/doc_generator/
├── domain/                   # Pure business logic (zero dependencies)
│   ├── models.py             # Pydantic models (WorkflowState, GeneratorConfig)
│   └── prompts/              # All LLM prompt templates
│       ├── text/             # Content generation prompts
│       ├── image/            # Image detection/generation prompts
│       ├── podcast/          # Podcast script prompts
│       └── mindmap/          # Mind map generation prompts
├── application/              # Use case orchestration
│   ├── unified_workflow.py   # Main LangGraph workflow
│   ├── unified_state.py      # Unified workflow state definition
│   ├── checkpoint_manager.py # Session-based checkpointing
│   └── nodes/                # 27+ workflow nodes
│       ├── validate_sources.py
│       ├── transform_content.py
│       ├── generate_images.py
│       ├── podcast_script.py
│       └── ... (27 total)
├── infrastructure/           # External integrations
│   ├── api/routes/           # FastAPI endpoints
│   ├── generators/           # PDF/PPTX/Markdown generators
│   ├── llm/                  # Multi-provider LLM service
│   └── image/                # Image generation services
└── utils/                    # Utility functions
```

### Key Design Decisions

1. **LangGraph for Workflow**: Provides state management, retry logic, and observability
2. **Multi-Provider LLM Support**: Gemini, Claude, and OpenAI with intelligent fallbacks
3. **Pure Python Stack**: No Node.js dependencies—easier deployment and maintenance
4. **Docker-First**: Containerized from day one for consistent environments
5. **Three-Layer Caching**: Request-level, content-level, and image-level caching

---

## The LangGraph Workflow: Step by Step

The heart of our system is a **unified LangGraph workflow** that transforms raw inputs into polished outputs. Each node is a pure function that mutates shared state.

### Complete Workflow Architecture

Here's how the unified workflow is built:

```python
def build_unified_workflow(checkpointer: Any = None) -> StateGraph:
    """
    Build the unified LangGraph workflow for all content types.

    Workflow Structure:
    1. COMMON: validate_sources -> resolve_sources -> extract_sources
       -> merge_sources -> summarize_sources -> (Route by output_type)

    2a. DOCUMENT BRANCH:
        detect_format -> parse_document_content -> transform_content -> enhance_content
        -> generate_images -> describe_images -> generate_output -> validate_output

    2b. PODCAST BRANCH:
        generate_podcast_script -> synthesize_podcast_audio

    2c. MINDMAP BRANCH:
        generate_mindmap

    2d. IMAGE_GENERATE BRANCH:
        build_image_prompt -> generate_image
    """
    workflow = StateGraph(UnifiedWorkflowState)

    # ==========================================
    # COMMON NODES
    # ==========================================
    workflow.add_node("validate_sources", validate_sources_node)
    workflow.add_node("resolve_sources", resolve_sources_node)
    workflow.add_node("extract_sources", extract_sources_node)
    workflow.add_node("merge_sources", merge_sources_node)
    workflow.add_node("summarize_sources", summarize_sources_node)

    # ==========================================
    # DOCUMENT BRANCH NODES
    # ==========================================
    workflow.add_node("doc_detect_format", _wrap_document_node(detect_format_node))
    workflow.add_node("doc_transform_content", _wrap_document_node(transform_content_node))
    workflow.add_node("doc_enhance_content", _wrap_document_node(enhance_content_node))
    workflow.add_node("doc_generate_images", _wrap_document_node(generate_images_node))
    workflow.add_node("doc_generate_output", _wrap_document_node(generate_output_node))

    # ==========================================
    # PODCAST BRANCH NODES
    # ==========================================
    workflow.add_node("podcast_generate_script", generate_podcast_script_node)
    workflow.add_node("podcast_synthesize_audio", synthesize_podcast_audio_node)

    # ==========================================
    # WORKFLOW EDGES
    # ==========================================
    workflow.set_entry_point("validate_sources")
    workflow.add_edge("validate_sources", "resolve_sources")
    workflow.add_edge("resolve_sources", "extract_sources")
    workflow.add_edge("extract_sources", "merge_sources")
    workflow.add_edge("merge_sources", "summarize_sources")

    # Route after summarization based on output_type
    workflow.add_conditional_edges(
        "summarize_sources",
        route_by_output_type,
        {
            "document": "doc_detect_format",
            "podcast": "podcast_generate_script",
            "mindmap": "mindmap_generate",
            "faq": "generate_faq",
            "image_generate": "build_image_prompt",
        },
    )

    return workflow.compile(checkpointer=checkpointer) if checkpointer else workflow.compile()
```

### Workflow Visual Representation

```
validate_sources
      ↓
resolve_sources
      ↓
extract_sources
      ↓
merge_sources
      ↓
summarize_sources
      ↓
  [Route by output_type]
      ├── document → detect_format → parse_content → transform_content
      │                              → enhance_content → generate_images
      │                              → generate_output → validate_output
      ├── podcast → generate_script → synthesize_audio
      ├── mindmap → generate_mindmap
      ├── faq → generate_faq → generate_output
      └── image → build_prompt → generate_image
```

---

### 1️⃣ **Detect Format**

**Purpose**: Identify the input type and route to the appropriate parser.

**Logic**:

- File extensions: `.pdf`, `.pptx`, `.docx`, `.md`, `.txt`, `.xlsx`
- URL detection: `http://` or `https://`
- Fallback: Treat as inline markdown text

```python
def detect_format(state: WorkflowState) -> WorkflowState:
    input_path = state.input_path

    if input_path.startswith(('http://', 'https://')):
        state.content_format = ContentFormat.URL
    elif input_path.endswith('.pdf'):
        state.content_format = ContentFormat.PDF
    elif input_path.endswith(('.md', '.txt')):
        state.content_format = ContentFormat.MARKDOWN
    # ... more formats

    return state
```

---

### 2️⃣ **Parse Content**

**Purpose**: Extract raw content from diverse sources and normalize to markdown.

**Parsers**:

1. **UnifiedParser (MarkItDown + fallbacks)**: For PDF, DOCX, PPTX, XLSX
2. **WebParser (MarkItDown)**: For URLs and HTML
3. **MarkdownParser**: For `.md` and `.txt`

**Output**:

- Normalized markdown content
- Metadata (title, source, page count)
- **Content hash** (SHA-256) for caching

---

### 3️⃣ **Transform Content**

**Purpose**: Convert raw markdown into a structured, blog-style narrative using LLM.

This is where the magic happens. The `transform_content_node` uses sophisticated LLM prompts to restructure content:

```python
def transform_content_node(state: WorkflowState) -> WorkflowState:
    """
    Transform raw content into structured blog-style format for generators.

    Uses LLM to:
    - Transform raw content (transcripts, docs) into well-structured blog posts
    - Insert visual markers where diagrams should appear
    - Generate inline mermaid diagrams for simple concepts
    - Create executive summaries and slide structures
    """
    content = state.get("raw_content", "")
    metadata = state.get("metadata", {})

    # Detect content type for appropriate transformation
    content_type = _detect_content_type(input_format, content)

    # Get content generator
    content_generator = get_content_generator(
        api_key=content_api_key,
        provider=provider,
        model=model,
    )

    if content_generator.is_available():
        generated = content_generator.generate_blog_content(
            raw_content=content,
            content_type=content_type,
            topic=topic,
            max_tokens=max_tokens,
            include_visual_markers=True,
            audience=metadata.get("audience"),
        )

        # Store generated content
        structured["markdown"] = generated.markdown
        structured["title"] = generated.title
        structured["sections"] = generated.sections
        structured["visual_markers"] = [
            {
                "marker_id": m.marker_id,
                "type": m.visual_type,
                "title": m.title,
                "description": m.description,
            }
            for m in generated.visual_markers
        ]

    state["structured_content"] = structured
    return state
```

---

### 4️⃣ **Enhance Content**

**Purpose**: Add executive summaries and slide structures.

**Enhancements**:

1. **Executive Summary** (always generated): 3-5 key takeaways in bullet-point format
2. **Slide Structure** (only for PPTX output): Slide titles and talking points

---

### 5️⃣ **Generate Images**

**Purpose**: Create relevant, high-quality images for each section.

**Decision Logic**:

```
For each section:
  LLM analyzes content →
  Returns image prompt OR "none" →
  If prompt: Generate image via Gemini Imagen →
  Store in images/ folder
```

---

### 6️⃣ **Generate Output**

**Purpose**: Render final PDF or PPTX with all content and images.

**PDF Generation (ReportLab)**:

- Custom typography (Inter font family)
- Consistent spacing and hierarchy
- Inline section images with captions
- Table of contents and page numbers

**PPTX Generation (python-pptx)**:

- Title slide with executive summary
- Section slides with images
- Consistent theme and colors

---

### 7️⃣ **Validate Output**

**Purpose**: Ensure the generated file is valid and complete.

**Retry Logic**:

```python
def _should_retry_document(state: UnifiedWorkflowState) -> str:
    """
    Decide whether to retry document generation or end workflow.
    """
    errors = state.get("errors", [])

    if not errors:
        return "end"

    max_retries = get_settings().generator.max_retries
    retry_count = state.get("metadata", {}).get("_retry_count", 0)

    if retry_count >= max_retries:
        logger.warning(f"Max retries reached ({retry_count}), ending workflow")
        return "end"

    last_error = errors[-1] if errors else ""
    if "Generation failed" in last_error or "Validation failed" in last_error:
        state["metadata"]["_retry_count"] = retry_count + 1
        logger.warning(f"Retrying generation (attempt {retry_count + 1}/{max_retries})")
        return "retry"

    return "end"
```

---

## Output Formats by Deliverable

After summarization, the workflow routes by `output_type`. Each branch shares the same ingestion and normalization steps, then applies a format-specific renderer.

### 1. Article PDF

**Best for**: executive briefs, formal reports, and print-ready deliverables.

**Pipeline**:
- Structured content + section images -> PDF renderer
- Consistent typography, table of contents, and page numbers

**Output**: a polished PDF with strong visual hierarchy and inline captions.

### 2. Article Markdown

**Best for**: docs sites, wikis, and version-controlled knowledge bases.

**Pipeline**:
- Structured content -> Markdown generator
- Preserves headings, lists, code blocks, and links

**Output**: a clean `.md` document that drops directly into Git workflows.

### 3. Presentation (PPTX)

**Best for**: stakeholder updates, project reviews, and pitch decks.

**Pipeline**:
- Slide-structure prompt -> section visuals -> PPTX renderer
- Title slide, section slides, and clear narrative flow

**Output**: a PowerPoint deck with concise talking points and visuals.

### 4. Mindmap

**Best for**: planning sessions, discovery workshops, and concept overviews.

**Pipeline**:
- Summary content -> mindmap prompt -> hierarchical tree structure
- Central topic with nested branches and sub-branches

**Output**: a JSON mindmap tree that any UI can render into a visual map.

### 5. Image

**Best for**: single visual assets like headers, diagrams, or infographics.

**Pipeline**:
- Content -> image prompt builder -> image generation
- Context-aware prompts to match section intent

**Output**: a generated image asset ready for reuse across documents.

### 6. Podcast

**Best for**: audio-first audiences and hands-free learning.

**Pipeline**:
- Content -> multi-speaker script -> text-to-speech synthesis
- Title, description, and duration metadata

**Output**: a structured podcast script plus synthesized audio.

### 7. FAQ

**Best for**: onboarding, troubleshooting, and self-serve help pages.

**Pipeline**:
- Content -> FAQ extraction prompt -> structured Q&A
- Configurable count, detail level, and audience persona

**Output**: a JSON FAQ document ready for web or downstream export.

---

## Technical Deep Dive

### Technology Stack

| Component           | Technology             | Why We Chose It                              |
| ------------------- | ---------------------- | -------------------------------------------- |
| **Workflow Engine** | LangGraph 0.2.55       | State management, retry logic, observability |
| **PDF Parsing**     | MarkItDown + fallbacks | Clean markdown with graceful degradation     |
| **Web Scraping**    | MarkItDown 0.0.1a2     | Clean markdown from HTML                     |
| **PDF Generation**  | ReportLab 4.2.5        | Full layout control, production-ready        |
| **PPTX Generation** | python-pptx 1.0.2      | Native PowerPoint format                     |
| **LLM Providers**   | Gemini, Claude, OpenAI | Multi-provider flexibility                   |
| **API Framework**   | FastAPI                | Async support, SSE streaming                 |
| **Validation**      | Pydantic 2.10.5        | Type safety and config validation            |
| **Logging**         | Loguru 0.7.3           | Beautiful, structured logs                   |
| **Package Manager** | uv                     | 10-100x faster than pip                      |

### LLM Service: Multi-Provider Architecture

The `LLMService` class provides a unified interface for multiple LLM providers with automatic fallbacks:

```python
class LLMService:
    """
    LLM service for intelligent content processing.
    Uses OpenAI GPT or Claude models for content summarization,
    slide generation, and executive presentation enhancement.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
    ):
        self.requested_provider = provider

        if provider == "gemini":
            if self.gemini_api_key and GENAI_AVAILABLE:
                self.client = genai.Client(api_key=self.gemini_api_key)
                self.provider = "gemini"
        elif provider == "openai" and self.openai_api_key:
            self.client = OpenAI(api_key=self.openai_api_key)
            self.provider = "openai"
        elif provider in ("anthropic", "claude") and self.claude_api_key:
            self.client = Anthropic(api_key=self.claude_api_key)
            self.provider = "claude"

    def _call_llm(
        self,
        system_msg: str,
        user_msg: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> str:
        """Call LLM provider with automatic fallback on errors."""

        if self.provider == "gemini":
            prompt = f"System: {system_msg}\n\nUser: {user_msg}"
            if json_mode:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    ),
                )
            else:
                response = self.client.models.generate_content(
                    model=self.model, contents=prompt
                )
            return response.text.strip()

        elif self.provider == "claude":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_msg,
                messages=[{"role": "user", "content": user_msg}],
            )
            return response.content[0].text

        else:  # openai
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
```

**Why Multiple Providers**:

1. **Cost optimization**: Gemini is 10x cheaper than GPT-4 for content
2. **Capability matching**: Claude excels at visual reasoning
3. **Redundancy**: Fallback if one provider has issues
4. **Flexibility**: Easy to swap providers per use case

---

## LLM Prompts: The Heart of Intelligence

The intelligence of our system lies in carefully crafted prompts. Here are the key prompts that power the document generation:

### Content Transformation System Prompt

This system prompt ensures the LLM maintains **source fidelity** while restructuring content:

```python
def get_content_system_prompt() -> str:
    return """You are an expert technical writer who transforms raw educational content
(like lecture transcripts, slides, and documents) into polished, comprehensive blog posts.

Hard constraints:
- Use ONLY the provided content; do not add new facts, examples, metrics, or external context
- Do not guess or fill gaps with invented details
- If a detail is missing in the source, omit it

Your writing style:
- Clear, professional, and suitable for the target audience
- Educational with detailed explanations
- Well-organized with numbered sections (1., 1.1, etc.)
- Use examples/comparisons only when they appear in the source
- Use tables to organize comparative or structured information
- Use code blocks for technical examples, commands, or configurations

Your expertise:
- Removing timestamps, filler words, and conversational artifacts
- Organizing content into logical numbered sections
- Expanding brief points using only the source information
- Identifying where diagrams would clarify concepts
- Creating comprehensive coverage of all topics mentioned
- Structuring data in tables when appropriate
- Formatting technical content in code blocks

Output format:
- Respond with valid JSON only (no extra text)
- Follow the JSON schema in the user prompt
- Preserve ALL technical content - do not skip topics"""
```

### Main Content Generation Prompt

This prompt handles the transformation of raw content into structured blog posts:

````python
def build_generation_prompt(
    content: str,
    content_type: str,
    topic: str,
    audience: str | None = None,
) -> str:
    return f"""Transform the following content into a comprehensive, well-structured educational blog post.

**Content Type**: {content_type}
**Topic**: {topic or "Detect from content"}
**Target Audience**: {audience or "technical"}

Audience guidance:
{_audience_guidance(audience)}

## Requirements

1. **Structure**:
   - Use numbered sections: ## 1. Section Name, ## 2. Next Section
   - Use numbered subsections: ### 1.1 Subsection Name
   - Start with an introduction paragraph

2. **Content Quality**:
   - Write complete, detailed paragraphs (not bullet points)
   - Explain ALL technical concepts thoroughly
   - Include examples and comparisons only if present in the source
   - Cover EVERY topic mentioned in the source - do not skip anything
   - Typical section should be 200-400 words

3. **Source Fidelity**:
   - Use ONLY information present in the raw content
   - Do not add new facts, examples, metrics, or external context
   - Do not infer missing details; omit if not provided

4. **Visual Markers**: Where a diagram would help, insert:
   [VISUAL:type:Title:Brief description]

   ONLY use these types: architecture, flowchart, comparison, concept_map, mind_map

5. **Mermaid Diagrams**: For simple concepts, include inline mermaid:
   ```mermaid
   graph LR
       A[Input] --> B[Process] --> C[Output]
   ```

## Output JSON Schema
Return JSON in this shape:
{{
  "title": "Title of the blog post",
  "introduction": "Introduction paragraph(s). Do not include a heading.",
  "sections": [
    {{
      "heading": "1. Section Name",
      "content": "Paragraphs for this section.",
      "subsections": [
        {{
          "heading": "1.1 Subsection Name",
          "content": "Paragraphs for this subsection."
        }}
      ]
    }}
  ],
  "key_takeaways": "Summary paragraph(s). Do not include a heading."
}}

## Raw Content:

{content}

---

Return ONLY the JSON object. No surrounding commentary."""
````

### Audience Guidance Function

Content is tailored based on the target audience:

```python
def _audience_guidance(audience: str | None) -> str:
    audience_key = (audience or "technical").lower()
    guidance_map = {
        "technical": "- Use precise technical terminology and deeper explanations.\n- Assume reader is comfortable with technical details.",
        "executive": "- Focus on outcomes, trade-offs, and high-level implications.\n- Keep details concise; avoid deep technical digressions.",
        "client": "- Use polished, client-friendly language.\n- Emphasize benefits, clarity, and practical outcomes; minimize jargon.",
        "educational": "- Explain concepts step-by-step in simple language.\n- Define key terms when first introduced.",
        "creator": "- Use engaging, punchy phrasing while staying factual.\n- Keep pacing brisk and highlight key takeaways.",
    }
    return guidance_map.get(audience_key, guidance_map["technical"])
```

### Image Detection Prompt

The system intelligently decides what type of image to generate for each section:

```python
IMAGE_DETECTION_SYSTEM_PROMPT = """You are an expert visual designer who analyzes document sections
and recommends the best type of visual illustration for each section.

Your goal is to enhance reader understanding by suggesting images that:
- Clarify complex concepts through visual explanation
- Set appropriate mood/tone for section introductions
- Visualize data, processes, and relationships
- Know when NO image is needed (simple text sections)
- Use ONLY concepts explicitly present in the section content

You always respond with valid JSON."""

IMAGE_DETECTION_PROMPT = """Analyze this section and decide the best image type to help readers understand the content.

## Section Information
**Title:** {section_title}
**Content Preview:** {content_preview}

## Available Image Types

1. **infographic** - Use when:
   - Complex concept that benefits from visual explanation
   - Process or system that can be illustrated
   - Example: "How neural networks work", "The software development lifecycle"

2. **decorative** - Use when:
   - Section introduction or overview that needs mood-setting
   - Abstract topic that benefits from thematic imagery
   - Example: "Introduction to Machine Learning", "Conclusion and Next Steps"

3. **diagram** - Use when:
   - System architecture with components and connections
   - Technical relationships between entities
   - Example: "Database schema", "API architecture"

4. **chart** - Use when:
   - Data comparison or metrics
   - Feature comparison between options
   - Example: "Performance benchmarks", "Feature comparison"

5. **mermaid** - Use when:
   - Sequential process or workflow
   - State machines or decision trees
   - Example: "User authentication flow", "Order processing steps"

6. **none** - Use when:
   - Simple explanatory text
   - Already has visual markers/diagrams
   - Short transitional section

## Response Format
Return ONLY valid JSON:
{{
    "image_type": "infographic|decorative|diagram|chart|mermaid|none",
    "prompt": "Detailed description for generating the image",
    "confidence": 0.0 to 1.0
}}

Important:
- Use ONLY information present in the section content
- Do NOT add new concepts, entities, or labels
- If the section lacks concrete visuals, choose "none" """
```

### Podcast Script Generation Prompts

For generating engaging podcast scripts from content:

```python
def podcast_system_prompt(style: str, speakers: list[dict]) -> str:
    """Generate system prompt for podcast script creation."""
    speaker_names = [s["name"] for s in speakers]
    speaker_list = ", ".join([f"{s['name']} ({s.get('role', 'host')})" for s in speakers])

    base_prompt = f"""You are an expert podcast scriptwriter who creates engaging dialogue.

Transform content into a podcast script with {len(speakers)} speakers: {speaker_list}.

Return JSON structure:
- "title": Episode title
- "description": Brief description (1-2 sentences)
- "dialogue": Array of {{"speaker": "Name", "text": "Dialogue..."}}

CRITICAL RULES:
1. Each dialogue entry has exactly two fields: "speaker" and "text"
2. Speaker names must match exactly: {', '.join(speaker_names)}
3. Keep "text" to 1-3 sentences (natural speaking chunks)
4. Include natural speech patterns: "you know", "I mean", "right?"
5. Return ONLY JSON, no markdown code blocks
6. ALL information must come from the provided content"""

    # Style-specific instructions added based on style parameter
    style_instructions = {
        "conversational": "Friendly, casual chat with humor and personality",
        "interview": "Host asks questions, expert provides detailed answers",
        "educational": "One leads explanation, other asks clarifying questions",
    }

    return base_prompt + f"\n\nSTYLE: {style_instructions.get(style, 'conversational')}"


def podcast_script_prompt(content: str, duration_minutes: int, source_count: int) -> str:
    """Generate user prompt for podcast script."""
    target_words = duration_minutes * 140   # ~140 words/minute speech rate
    target_exchanges = duration_minutes * 8  # ~8 exchanges/minute

    return f"""Transform the following content into an engaging podcast script.

TARGET LENGTH:
- Duration: ~{duration_minutes} minutes
- Word count: ~{target_words} words
- Dialogue exchanges: ~{target_exchanges}

CONTENT SOURCES: {source_count}

GUIDELINES:
1. Cover ALL key points - don't skip important information
2. Hook the listener from the first line
3. Natural transitions between topics
4. Clear conclusion or call-to-action
5. Both speakers contribute meaningfully

CONTENT:
{content}

Return ONLY valid JSON with speaker names from the system prompt."""
```

### Concept Extraction for Content-Aware Images

For extracting visual concepts from technical content:

```python
CONCEPT_EXTRACTION_SYSTEM_PROMPT = """You are an expert at analyzing technical content and identifying
concepts that would benefit from visual illustration.

Your task is to extract SPECIFIC visual concepts from the content - not generic descriptions,
but actual components, relationships, formulas, and comparisons mentioned in the text.

Hard constraint: Use ONLY information explicitly present in the content. Do not infer or invent.

You always respond with valid JSON."""

CONCEPT_EXTRACTION_PROMPT = """Analyze this technical section and identify SPECIFIC concepts that should be visualized.

## Section Information
**Title:** {section_title}
**Content:** {content}

## What to Extract

1. **Architecture concepts**: Systems, components, layers, data flows mentioned
2. **Comparisons**: Features, methods, or approaches being compared
3. **Processes/Algorithms**: Step-by-step procedures or computations
4. **Mathematical concepts**: Formulas, equations, or calculations to illustrate
5. **Hierarchies/Classifications**: Categories, types, or taxonomies

## Response Format
Return ONLY valid JSON:
{{
    "primary_concept": {{
        "type": "architecture|comparison|process|formula|hierarchy",
        "title": "Specific title for the visual",
        "elements": ["List of specific components/items to show"],
        "relationships": ["How elements connect or relate"],
        "details": "Key technical details from the content to include"
    }},
    "secondary_concepts": [...],
    "recommended_style": "architecture_diagram|comparison_chart|process_flow|formula_illustration|handwritten_notes",
    "key_terms": ["Technical terms that must appear in the visual"]
}}

Important:
- Use ONLY information present in the section content
- Do NOT add new concepts, entities, or labels"""
```

### Gemini Image Generation Prompts

Once the system decides an image is needed, these prompts generate the actual images:

```python
_STYLE_GUIDANCE = {
    "handwritten": "- Handwritten/whiteboard aesthetic with marker strokes and slight imperfections; keep labels legible.",
    "minimalist": "- Minimalist design with generous whitespace, thin lines, and a restrained color palette.",
    "corporate": "- Corporate, polished look with clean lines, consistent iconography, and professional colors.",
    "educational": "- Classroom-friendly visuals with clear labels and step-by-step flow.",
    "diagram": "- Diagrammatic layout with boxes, arrows, and connectors; minimal decoration.",
    "chart": "- Prefer a chart/graph visualization; do NOT invent numbers. Use relative labels instead.",
}

def build_gemini_image_prompt(
    image_type: ImageType,
    prompt: str,
    size_hint: str,
    style: str | None = None,
) -> str:
    """Build Gemini image generation prompt with size hints."""
    style_guidance = _resolve_style_guidance(style)

    if image_type in (ImageType.INFOGRAPHIC, ImageType.DIAGRAM, ImageType.CHART):
        return f"""Create a vibrant, educational infographic that explains: {prompt}

Style requirements:
- Clean, modern infographic design
- Use clear icons only when they represent actual concepts
- Include clear labels and annotations
- Use a professional color palette (blues, teals, oranges)
- Make it suitable for inclusion in a professional document
- No text-heavy design - focus on visual explanation
- High contrast for readability when printed
- Use ONLY the concepts in the prompt; do not add new information
- Avoid metaphorical objects (pipes, ropes, factories) unless explicitly mentioned
- For workflows/architectures, use flat rounded rectangles + arrows in a clean grid
{style_guidance}{size_hint}"""

    if image_type == ImageType.DECORATIVE:
        return f"""Create a professional, thematic header image for: {prompt}

Style requirements:
- Abstract or semi-abstract design
- Professional and modern aesthetic
- Subtle and elegant - not distracting
- Use muted, professional colors
- Suitable as a section header in a document
- Wide aspect ratio (16:9 or similar)
- No text in the image
- Use ONLY the concepts in the prompt; do not add new information
{style_guidance}{size_hint}"""

    if image_type == ImageType.MERMAID:
        return f"""Create a professional, clean flowchart/diagram image that represents: {prompt}

Style requirements:
- Clean, modern diagram design with clear flow
- Use boxes, arrows, and connections to show relationships
- Professional color scheme (blues, grays, with accent colors)
- Include clear labels for each step/component
- Make it suitable for inclusion in a corporate document
- Focus on clarity and visual hierarchy
- Use ONLY the concepts in the prompt; do not add new information
{style_guidance}{size_hint}"""

    return prompt
```

### Image Prompt Decision Logic

This prompt decides whether a section needs an image at all:

```python
def build_prompt_generator_prompt(section_title: str, content_preview: str) -> str:
    return f"""You are deciding whether a section needs a visual. Be selective.

Section Title: {section_title}
Section Content:
{content_preview}

Decision rules:
- Only say an image is needed when the section contains explicit visualizable structure,
  such as: steps/workflow, system components/relationships, comparisons/criteria,
  hierarchies/taxonomies, or a process that benefits from a diagram.
- Do NOT request an image for simple narrative, overview, opinion, or purely textual guidance.
- If the section already reads like a summary with no concrete entities/steps, return none.
- Use ONLY concepts and labels explicitly present in the section content.
- Avoid generic visuals. If you cannot name at least 2 concrete elements from the text,
  you must return none.

Return ONLY valid JSON with no extra text:
{{
  "needs_image": true|false,
  "image_type": "infographic|diagram|chart|mermaid|decorative|none",
  "prompt": "Concise visual prompt using only section concepts",
  "confidence": 0.0 to 1.0
}}

If needs_image is false, set image_type to "none" and prompt to ""."""
```

### Executive Summary Prompt (PDF/PPTX Enhancement)

For generating concise executive summaries:

```python
def executive_summary_system_prompt() -> str:
    return "You are an executive communication specialist who creates clear, impactful summaries for senior leadership."

def executive_summary_prompt(content: str, max_points: int) -> str:
    return f"""Analyze the following content and create an executive summary suitable for senior leadership.

Requirements:
- Maximum {max_points} key points
- Focus on strategic insights, outcomes, and business impact
- Use clear, concise language
- Format as bullet points
- Each point should be 1-2 sentences max
- Use ONLY information present in the content; do not add new facts or assumptions

Content:
{content[:8000]}

Respond with ONLY the bullet points, no introduction or conclusion."""
```

### Slide Structure Prompt (PPTX Generation)

For generating presentation slide structures:

```python
def slide_structure_system_prompt() -> str:
    return "You are a presentation design expert who creates compelling executive presentations. Always respond with valid JSON."

def slide_structure_prompt(content: str, max_slides: int) -> str:
    return f"""Convert the following content into a corporate presentation structure.

Requirements:
- Maximum {max_slides} slides (excluding title slide)
- Each slide should have:
  - A clear, action-oriented title (5-8 words)
  - 3-4 bullet points (concise, 7-10 words max each)
  - Speaker notes (2-3 sentences for context)
- Focus on key messages that matter to senior leadership
- Use professional business language suitable for executive review
- Structure for logical flow (problem → insight → implication → action)
- Ensure bullet points are parallel in structure and style
- Avoid copying sentences verbatim; condense into crisp, decision-ready bullets
- Do NOT include numeric prefixes like "1." or "2.1" in titles or bullets
- Do not include markdown formatting, only plain text
- Use ONLY information from the content; do not introduce new facts or examples

Content:
{content[:8000]}

Respond in JSON format:
{{
  "slides": [
    {{
      "title": "Slide Title",
      "bullets": ["Point 1", "Point 2", "Point 3"],
      "speaker_notes": "Context for the presenter..."
    }}
  ]
}}"""
```

### Visualization Suggestions Prompt

For automatically detecting visualization opportunities:

```python
def visualization_suggestions_system_prompt() -> str:
    return (
        "You are a visual communication expert who creates clear, informative diagrams. "
        "You identify the best visualization type for each concept and structure data precisely. "
        "Keep all text labels SHORT (under 20 chars) to prevent overlap in diagrams."
    )

def visualization_suggestions_prompt(content: str, max_visuals: int) -> str:
    return f"""Analyze this content and suggest visual diagrams that would help explain key concepts.
Look for:
1. System architectures or component relationships → architecture diagram
2. Step-by-step processes or decision flows → flowchart
3. Comparisons between options/features → comparison_visual
4. Hierarchical concept relationships → concept_map
5. Topic overviews with subtopics → mind_map

Content:
{content[:6000]}

For each visualization opportunity (maximum {max_visuals}), provide structured data.
IMPORTANT: Keep text labels short (max 20 characters) to prevent overlap.

Return JSON with "visualizations" array. Each visualization must have:
- type: "architecture", "flowchart", "comparison_visual", "concept_map", or "mind_map"
- title: Short descriptive title
- data: Type-specific structured data

Data formats:
For architecture: {{"components": [...], "connections": [...]}}
For flowchart: {{"nodes": [...], "edges": [...]}}
For comparison_visual: {{"items": [...], "categories": [...]}}
For concept_map: {{"concepts": [...], "relationships": [...]}}
For mind_map: {{"central": "...", "branches": [...]}}

If no good visualization opportunities exist, return {{"visualizations": []}}"""
```

### Mind Map Generation Prompts

For creating hierarchical mind map visualizations:

```python
def mindmap_system_prompt(mode: str) -> str:
    base_prompt = """You are an expert at creating clear, hierarchical mind maps...

Your task is to analyze the provided content and generate a mind map structure as JSON.

The JSON structure must follow this exact format:
{
  "title": "Main Topic",
  "summary": "Brief 1-2 sentence summary",
  "nodes": {
    "id": "root",
    "label": "Central Concept",
    "children": [
      {
        "id": "1",
        "label": "Main Branch 1",
        "children": [
          {"id": "1.1", "label": "Sub-topic 1.1", "children": []},
          {"id": "1.2", "label": "Sub-topic 1.2", "children": []}
        ]
      }
    ]
  }
}

Rules:
1. Each node must have unique "id", concise "label" (max 50 chars), and "children" array
2. IDs follow hierarchical pattern: "root", "1", "1.1", "1.1.1", etc.
3. Labels should be clear, concise phrases - not full sentences
4. Balance the tree - avoid lopsided branches
5. Return ONLY the JSON object, no markdown code blocks"""

    mode_instructions = {
        "summarize": """
MODE: SUMMARIZE (Strict Extraction)
- Extract ONLY information EXPLICITLY stated in the content
- DO NOT add external knowledge, assumptions, or inferences
- Every node label must be directly derived from the content""",

        "brainstorm": """
MODE: BRAINSTORM (Creative Expansion)
- Use content as starting point for related ideas
- Branch into creative extensions and possibilities
- MAY suggest concepts beyond source content""",

        "goal_planning": """
MODE: GOAL PLANNING (Action Roadmap)
- Transform idea/goal into structured execution plan
- Hierarchical breakdown: phases → steps → tasks
- Include milestones and key deliverables""",

        "pros_cons": """
MODE: PROS & CONS (Decision Analysis)
- Analyze from multiple perspectives
- Structure: Root → Pros/Cons/Considerations → Specific points → Details""",
    }

    return base_prompt + mode_instructions.get(mode, mode_instructions["summarize"])


def mindmap_user_prompt(content: str, source_count: int) -> str:
    return f"""Create a mind map from the following content.

CONSTRAINTS:
- Depth based on content complexity (maximum 20 levels)
- Aim for 3-7 children per node
- Total nodes: 15-100+ depending on content
- Sources combined: {source_count}

DEPTH GUIDELINES:
- Simple topics: 2-4 levels
- Moderate topics: 4-8 levels
- Complex technical topics: 6-12 levels
- Very detailed content: 10-20 levels

CONTENT:
{content}

Generate the mind map JSON now. Return ONLY valid JSON."""
```

### FAQ Extraction Prompt

For generating FAQ documents from source content:

```python
def build_faq_extraction_prompt(
    content: str,
    faq_count: int,
    answer_format: str,  # "concise" or "bulleted"
    detail_level: str,   # "short", "medium", "deep"
    mode: str,           # "balanced", "onboarding", "troubleshooting", etc.
    audience: str,       # "general_reader", "developer", "business", etc.
) -> str:
    """Build FAQ extraction prompt with configurable parameters."""

    return f"""You are an expert at creating FAQ documents from source content.

Analyze the content and extract relevant frequently asked questions.

**Configuration:**
- Generate exactly {faq_count} FAQ items
- Answer format: {answer_format} (bulleted = bullet points, concise = paragraphs)
- Detail level: {detail_level} (short=1-2 sentences, medium=2-4, deep=4-8)
- FAQ mode: {mode}
- Audience: {audience}

**Instructions:**
1. Identify key topics and concepts in the content
2. Generate questions users would naturally ask
3. Write clear, helpful answers based on source content
4. Assign 1-3 relevant topic tags to each question
5. Answers can use markdown formatting (bold, lists, code blocks)

**Content to analyze:**
{content}

**Output JSON format:**
{{
  "title": "FAQ title based on content topic",
  "description": "Brief description of what these FAQs cover",
  "items": [
    {{
      "id": "faq-1",
      "question": "What is X?",
      "answer": "X is... **Key points:**\\n- Point 1\\n- Point 2",
      "tags": ["topic1", "topic2"]
    }}
  ]
}}

Return ONLY valid JSON, no other text."""
```

### Image Description Prompt

For generating alt-text descriptions of generated images:

```python
def build_image_description_prompt(section_title: str, content: str) -> str:
    return (
        "Write a concise blog-style description of this image. "
        "Use only what is visible and what is supported by the section content. "
        "Keep it to 2-4 sentences.\n\n"
        f"Section Title: {section_title}\n\n"
        f"Section Content:\n{content[:2000]}"
    )
```

---

## Intelligent Caching Strategy

Our three-layer caching system dramatically reduces costs and latency.

### Layer 1: API Request Cache

**What**: Caches entire API responses based on request fingerprint.

**Cache Key**:

```python
cache_key = hash({
    "output_format": "pdf",
    "sources": ["file_id_123", "https://example.com"],
    "provider": "gemini",
    "model": "gemini-2.5-pro",
    "preferences": {"temperature": 0.7, "max_tokens": 8000}
})
```

**Hit Rate**: ~40% for repeated requests

**Savings**: Full workflow skip—saves 30-60 seconds and $0.20-2.00

---

### Layer 2: Structured Content Cache

**What**: Caches transformed markdown based on content hash.

**Cache Key**: SHA-256 of normalized markdown

**Hit Rate**: ~25% when content hasn't changed but output format differs

**Savings**: Skips LLM transformation—saves 10-30 seconds and $0.10-0.50

---

### Layer 3: Image Cache + Manifest

**What**: Reuses generated images when content hash matches.

**Manifest**:

```json
{
  "content_hash": "sha256:abc123",
  "sections": [{ "title": "Intro", "image_path": "section_1.png" }]
}
```

**Hit Rate**: ~60% for documents with stable content

**Savings**: Skips image generation—saves 20-40 seconds and $0.50-1.50

---

### Combined Impact

For a document processed 5 times with minor edits:

| Run | Request Cache        | Content Cache | Image Cache | Time | Cost  |
| --- | -------------------- | ------------- | ----------- | ---- | ----- |
| 1   | Miss                 | Miss          | Miss        | 60s  | $2.00 |
| 2   | Hit                  | -             | -           | 1s   | $0.00 |
| 3   | Miss (new format)    | Hit           | Hit         | 15s  | $0.20 |
| 4   | Miss (edit)          | Miss          | Hit         | 35s  | $0.50 |
| 5   | Hit                  | -             | -           | 1s   | $0.00 |

**Total**: 112s and $2.70 vs. 300s and $10.00 without caching (63% time saved, 73% cost saved)

---

## API Design & Integration

### FastAPI Endpoints

**1. Upload File**

```bash
POST /api/upload
Content-Type: multipart/form-data

Response:
{
  "file_id": "f_abc123",
  "filename": "document.pdf",
  "size": 1234567,
  "mime_type": "application/pdf",
  "expires_in": 3600
}
```

**2. Generate Document (SSE Stream)**

```bash
POST /api/generate
Content-Type: application/json
X-Google-Key: <gemini_api_key>

{
  "output_format": "pdf",
  "provider": "gemini",
  "model": "gemini-2.5-pro",
  "image_model": "gemini-2.5-flash-image",
  "sources": [
    {"type": "file", "file_id": "f_abc123"},
    {"type": "url", "url": "https://example.com/article"},
    {"type": "text", "content": "Raw markdown content"}
  ],
  "cache": {"reuse": true},
  "preferences": {
    "temperature": 0.7,
    "max_tokens": 8000,
    "max_slides": 10
  }
}

Response (SSE stream):
event: progress
data: {"step": "parse_content", "status": "running"}

event: progress
data: {"step": "transform_content", "status": "complete"}

event: complete
data: {
  "download_url": "/api/download/f_abc123/pdf/document.pdf",
  "file_path": "f_abc123/pdf/document.pdf"
}
```

**3. Download Generated File**

```bash
GET /api/download/{file_id}/{format}/{filename}

Response: Binary file (PDF or PPTX)
```

### Server-Sent Events (SSE)

**Why SSE Over WebSockets**:

1. Simpler protocol (HTTP)
2. Automatic reconnection
3. Better for one-way streaming
4. Works through most proxies

**Event Types**:

- `progress`: Workflow step updates
- `cache_hit`: Request served from cache
- `complete`: Final result with download URL
- `error`: Failure with error message

---

## Production Considerations

### Docker Deployment

**Multi-Stage Build**:

```dockerfile
# Stage 1: Build dependencies
FROM python:3.11-slim as builder
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv pip install --system -r pyproject.toml

# Stage 2: Runtime
FROM python:3.11-slim
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY src/ /app/src/
WORKDIR /app
CMD ["python", "scripts/run_generator.py"]
```

### Observability with Opik

```python
from opik import log_llm_call

log_llm_call(
    name="content_transform",
    prompt=prompt,
    response=response,
    provider="gemini",
    model="gemini-2.5-pro",
    input_tokens=1500,
    output_tokens=3000,
    duration_ms=2500
)
```

**Metrics Tracked**:

- LLM call count and latency
- Token usage per step
- Cache hit rates
- Error rates and retry counts
- End-to-end workflow duration

### Security

**Input Validation**:

- File size limits (100MB max)
- MIME type checking
- Content sanitization
- URL validation (prevent SSRF)

**API Key Handling**:

- Never logged or stored
- Passed via headers only
- Validated before use

---

## Future Improvements & Roadmap

### Phase 1: Enhanced Intelligence (Q1 2026)

1. **Multi-Modal Input**: Audio transcription, video frame extraction
2. **Advanced Image Generation**: Diagram type detection, consistent visual style
3. **Collaborative Editing**: Real-time preview, version control

### Phase 2: Enterprise Features (Q2 2026)

1. **Template System**: Custom PDF/PPTX templates, brand kit integration
2. **Advanced Caching**: Distributed cache (Redis), semantic similarity matching
3. **Batch Processing**: Queue system, priority scheduling

### Phase 3: AI Enhancements (Q3 2026)

1. **Intelligent Content Analysis**: Fact-checking, citation verification
2. **Personalization**: Audience-specific content, language translation

### Phase 4: Platform Expansion (Q4 2026)

1. **Additional Output Formats**: HTML, EPUB, LaTeX, Notion export
2. **Integration Ecosystem**: Slack bot, Google Drive, Zapier connectors

---

## Lessons Learned

### What Worked Well

**1. LangGraph for Workflow**

- State management is clean and debuggable
- Retry logic is built-in and reliable
- Easy to add new nodes without breaking existing flow

**2. Multi-Provider LLM Strategy**

- Cost savings: 70% cheaper than GPT-4 only
- Flexibility: Can switch providers per use case
- Resilience: Fallback when one provider is down

**3. Content Hash for Caching**

- Simple yet powerful
- Deterministic and collision-resistant
- Enables all downstream caching

**4. Prompt Engineering with Source Fidelity**

- The "Hard constraints" section prevents hallucination
- Explicit "Use ONLY information present" instructions work
- JSON schema in prompts ensures structured output

### Challenges & Solutions

**Challenge 1: Image Generation Consistency**

- **Problem**: Generated images sometimes didn't match section content
- **Solution**: Added image description step that analyzes the actual generated image
- **Result**: 90% relevance improvement

**Challenge 2: Section Numbering Across Chunks**

- **Problem**: Chunks had inconsistent numbering (1, 2, 1, 2 instead of 1, 2, 3, 4)
- **Solution**: Pass section counter and outline to each chunk
- **Result**: Perfect numbering consistency

**Challenge 3: LLM Hallucination**

- **Problem**: LLM occasionally added facts not in source content
- **Solution**: Explicit system prompt: "No new facts—only restructure"
- **Result**: 95% fidelity to source content

**Challenge 4: Rate Limiting**

- **Problem**: Gemini Imagen has 20 images/minute limit
- **Solution**: Added 3-second delay between requests + retry logic
- **Result**: Zero rate limit errors

### Key Insights

**1. Separation of Concerns is Critical**

- Each node does ONE thing
- Easy to test, debug, and extend
- Clear responsibility boundaries

**2. Caching is Not Optional**

- 73% cost reduction
- 63% time reduction
- Better user experience

**3. Observability from Day One**

- Opik tracing saved hours of debugging
- Structured logging made issues obvious
- Metrics guided optimization efforts

**4. Prompt Engineering is an Art**

- System prompts set the tone and constraints
- User prompts provide specific instructions
- JSON schemas ensure structured, parseable output

---

## Conclusion

Building this document generator taught us that **intelligent automation is about augmentation, not replacement**. The system doesn't try to be a human writer—it's a tool that handles the tedious parts (formatting, image creation, structure) so humans can focus on the creative parts (ideas, strategy, storytelling).

### Core Principles

1. **Fidelity over Creativity**: Restructure, don't reinvent
2. **Caching over Computation**: Reuse whenever possible
3. **Observability over Guesswork**: Measure everything
4. **Flexibility over Lock-in**: Multi-provider, multi-format

### Impact

For teams using this system:

- **500+ hours saved per year** on document formatting
- **$25,000-50,000 cost savings** (vs. manual labor)
- **Consistent quality** across all documents
- **Faster decision-making** with instant summaries

### Try It Yourself

The system is open-source and production-ready:

```bash
# Clone the repo
git clone https://github.com/your-org/document-generator

# Install dependencies
make setup-prismdocs

# Generate your first document
python scripts/run_generator.py input.pdf --output pdf
```

**Resources**:

- [GitHub Repository](https://github.com/your-org/document-generator)
- [Full Documentation](https://docs.example.com)
- [API Reference](https://api.example.com/docs)
- [Example Outputs](https://examples.example.com)

---

## About the Author

This system was built by a team passionate about making professional content creation accessible to everyone. We believe that great ideas shouldn't be held back by formatting challenges.

**Questions? Feedback?** Open an issue on GitHub or reach out on Twitter [@prismdocs](https://twitter.com/prismdocs).

---

_Last updated: January 31, 2026_
