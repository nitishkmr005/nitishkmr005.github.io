---
title: "Building PrismDocs: An Intelligent Document Generator for Making Complex Content Accessible"
date: 2026-02-01
draft: false
tags: ["LangGraph", "LLM", "Document Generation", "Python", "FastAPI", "AI", "Production ML"]
categories: ["AI Engineering", "Tutorial"]
description: "How we built a production-ready system that transforms research papers, web articles, and documents into multiple accessible formats—PDFs, presentations, mind maps, podcasts, and FAQs—using LangGraph and modern LLMs"
cover:
  image: "https://raw.githubusercontent.com/nitishkmr005/PrismDocs/main/.github/banner.svg"
  alt: "PrismDocs - AI-Powered Document Generation"
  caption: "Transform content into professional PDFs, presentations, mind maps, and podcasts"
---

*How we built a production-ready system that transforms research papers, web articles, and documents into multiple accessible formats—PDFs, presentations, mind maps, podcasts, and FAQs—using LangGraph and modern LLMs*

---

## Table of Contents

1. [The Problem We're Solving](#the-problem-were-solving)
2. [Business Value & Use Cases](#business-value--use-cases)
3. [System Architecture Overview](#system-architecture-overview)
4. [The LangGraph Workflow: Step by Step](#the-langgraph-workflow-step-by-step)
5. [Deliverable Formats: Architecture & Assembly](#deliverable-formats-architecture--assembly)
6. [Technical Deep Dive](#technical-deep-dive)
7. [Intelligent Caching Strategy](#intelligent-caching-strategy)
8. [API Design & Integration](#api-design--integration)
9. [Production Considerations](#production-considerations)
10. [Future Improvements & Roadmap](#future-improvements--roadmap)
11. [Lessons Learned](#lessons-learned)

---

## The Problem We're Solving

In today's knowledge economy, organizations face a dual challenge: **content is everywhere, but it's rarely in the right format—and complex ideas often remain inaccessible**. Teams deal with:

- **Scattered knowledge**: PDFs, slide decks, markdown files, web articles, research papers, Word documents
- **Complex content that's hard to digest**: Dense research papers, technical documentation, and lengthy articles that require significant effort to understand
- **Manual conversion**: Hours spent reformatting content for different audiences
- **Inconsistent presentation**: No unified visual language across documents
- **Lost context**: Important information buried in poorly structured files
- **One-size-fits-all output**: No way to adapt content for different learning styles (visual, auditory, interactive)

### Making Complex Content Accessible

Not everyone learns the same way. A 50-page research paper might be impenetrable to some readers but becomes crystal clear when:

- **Visualized as a mind map** showing concept relationships
- **Explained through a podcast** with conversational dialogue
- **Broken into FAQs** answering the questions readers actually have
- **Enhanced with diagrams** illustrating key processes and architectures

This is what PrismDocs delivers: **transforming complex content into multiple accessible formats** that match how people actually consume information.

### The Real Cost

Consider a typical scenario:

- A research team publishes a dense 30-page technical paper
- They need to share findings with executives, developers, and external stakeholders—each with different needs
- Manual process: Create slides for executives, diagrams for developers, FAQs for support teams
- **Our solution**: One input, multiple outputs—each optimized for its audience

This isn't just about saving time—it's about **making knowledge accessible to everyone** regardless of their learning preferences or time constraints.

---

## Business Value & Use Cases

### Primary Use Cases

#### 1. **Research Paper Comprehension**

**Problem**: Research papers and technical articles are dense, time-consuming, and difficult to share with non-experts.

**Solution**: Our system transforms complex research into multiple accessible formats:

- **Mind maps** showing concept hierarchies and relationships at a glance
- **Podcasts** explaining findings through natural, conversational dialogue
- **FAQs** answering the questions readers actually have
- **Visual summaries** with AI-generated diagrams for key concepts
- **Executive presentations** highlighting actionable insights

**Impact**: A 30-page research paper becomes a 10-minute podcast, a visual mind map, and a 5-slide deck—each serving different audiences.

#### 2. **Technical Documentation Consolidation**

**Problem**: Engineering teams have documentation scattered across PDFs, markdown files, and wikis.

**Solution**: Our system:

- Ingests multiple file formats simultaneously
- Merges content intelligently while preserving structure
- Generates both PDF documentation and PPTX presentations
- Adds AI-generated executive summaries

**Impact**: Reduced documentation preparation time from days to minutes.

#### 3. **Web Article Synthesis**

**Problem**: Long-form articles and blog posts require significant time investment to consume and understand.

**Solution**:

- Extracts and normalizes web content
- Creates structured mind maps for quick comprehension
- Generates podcast versions for hands-free learning
- Builds FAQs for interactive exploration
- Produces visual summaries with relevant diagrams

**Impact**: Users choose how they consume content—read, listen, explore, or visualize.

#### 4. **Educational Content Transformation**

**Problem**: Educational materials often exist in a single format that doesn't suit all learners.

**Solution**:

- Transforms lectures and course materials into visual mind maps
- Creates podcast versions for auditory learners
- Generates FAQ documents for self-paced review
- Builds presentation decks for instructors
- Produces enhanced PDFs with diagrams and visual explanations

**Impact**: One piece of content serves visual, auditory, and reading learners equally well.

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
        -> generate_images -> describe_images -> persist_images
        -> generate_output -> validate_output

    2b. PODCAST BRANCH:
        generate_podcast_script -> synthesize_podcast_audio

    2c. MINDMAP BRANCH:
        generate_mindmap

    2d. IMAGE_GENERATE BRANCH:
        build_image_prompt -> generate_image

    2e. IMAGE_EDIT BRANCH:
        edit_image
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
    workflow.add_node("doc_describe_images", _wrap_document_node(describe_images_node))
    workflow.add_node("doc_persist_images", _wrap_document_node(persist_image_manifest_node))
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
            "image_edit": "edit_image",
        },
    )

    return workflow.compile(checkpointer=checkpointer) if checkpointer else workflow.compile()
```

### Chunked Summarization (No Truncation)

We run a **chunked map-reduce summarizer** before routing so large inputs are never truncated. This produces a stable `summary_content` field that downstream branches can safely consume without token overflow.

**What it does**:
- Splits raw content into chunks
- Summarizes each chunk
- Reduces the summaries into a single cohesive summary

**When it's used**:
- Always runs after `merge_sources`
- **Used directly** by Podcast, Mindmap, FAQ, and Image (standalone) branches
- **Optional context** for Article and Presentation (which still use full `raw_content`)

**When it's not used**:
- If summarization fails or LLM is unavailable, branches fall back to `raw_content`
- For document outputs, the full content is preserved for higher-fidelity rendering

**Why this is required**: It prevents token overflow on long or multi-source inputs while keeping coverage high.

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
      ├── document → detect_format → parse_document_content → transform_content
      │                              → enhance_content → generate_images
      │                              → describe_images → persist_images
      │                              → generate_output → validate_output
      ├── podcast → generate_script → synthesize_audio
      ├── mindmap → generate_mindmap
      ├── faq → generate_faq → generate_output → validate_output
      ├── image_generate → build_prompt → generate_image
      └── image_edit → edit_image
```

### Node-by-Node Breakdown

The table below explains **every workflow node**, why it exists, and what it does.

| Node | Branch | What it does | Why it is needed |
| --- | --- | --- | --- |
| `validate_sources` | Common | Verifies sources and decides skip/reuse behavior. | Avoids wasted work and fails fast on bad input. |
| `resolve_sources` | Common | Resolves file IDs/URLs/text into canonical paths and normalized inputs. | Ensures downstream parsers receive consistent inputs. |
| Parse Sources (`extract_sources`) | Common | Parses each source (file/URL/text) into `content_blocks` (includes OCR for images). | Centralizes parsing across all output types. |
| `merge_sources` | Common | Merges content blocks into `raw_content` and writes a temp markdown file for docs. | Creates a single, ordered source of truth before summarization/transform. |
| `summarize_sources` | Common | Chunked map-reduce summarization to produce `summary_content`. | Prevents token overflow while preserving coverage. |
| `doc_detect_format` | Document | Detects the format of the **merged document input** (temp markdown or source file). | Required because this step operates on the merged document file, not the original sources. |
| Parse Document Input (`doc_parse_document_content`) | Document | Parses the merged doc input into `raw_content` + metadata (title/pages/hash). | Required to generate canonical content and metadata for rendering and caching. |
| `doc_transform_content` | Document | LLM transforms `raw_content` into structured markdown/sections. | Creates a stable schema for PDF/Markdown/PPTX renderers. |
| `doc_enhance_content` | Document | Adds executive summary and (optionally) slide structure. | Enables executive-ready slides and summaries. |
| Generate Section Images (`doc_generate_images`) | Document | Per-section image decision + raster generation. | Adds visuals where they improve comprehension. |
| `doc_describe_images` | Document | Generates short captions/alt text and embeds base64 when needed. | Improves accessibility and PDF embedding. |
| `doc_persist_images` | Document | Writes an image manifest to support cache reuse. | Avoids regenerating images on reruns. |
| Render Output (`doc_generate_output`) | Document/FAQ | Renders final PDF/PPTX/Markdown/FAQ output. | Produces the deliverable artifact. |
| `doc_validate_output` | Document/FAQ | Validates output file and applies retry rules. | Prevents returning incomplete or corrupt files. |
| `podcast_generate_script` | Podcast | Generates structured dialogue JSON. | Provides TTS-ready script format. |
| `podcast_synthesize_audio` | Podcast | Converts dialogue to audio using TTS. | Produces the final podcast deliverable. |
| `mindmap_generate` | Mindmap | Generates hierarchical mindmap JSON. | Produces a renderable tree structure. |
| `generate_faq` | FAQ | Extracts Q&A JSON from content. | Produces structured FAQ output. |
| Build Standalone Image Prompt (`build_image_prompt`) | Image | Builds a single prompt from mindmap summary or user input. | Creates a scoped prompt for standalone image generation. |
| `image_generate` | Image | Renders raster/SVG from the prompt. | Produces the final image asset. |
| `image_edit` | Image Edit | Applies edits (style/region) to an existing image. | Enables iterative refinement on generated images. |

---

## Deliverable Formats: Architecture & Assembly

After summarization, the workflow routes by `output_type`. Each branch shares the same ingestion and normalization steps, then applies a format-specific renderer powered by carefully orchestrated LLM prompts.

This section explains **how each output format is assembled**—from the prompts that drive content generation to the final rendering pipeline. Each format follows a consistent pattern: **prompt → transform → assemble → validate**.

> **See it in action**: [Sample Outputs on GitHub](https://github.com/nitishkmr005/PrismDocs/tree/main/sampleOutputs) | [Generated Documents](https://github.com/nitishkmr005/PrismDocs/tree/main/sampleOutputs/Generated%20Documents)

### 1. Article (PDF + Markdown)

**Best for**: executive briefs, formal reports, print-ready deliverables, and version-controlled knowledge bases.

**What it produces**: Professional documents with structured sections, AI-generated images, captions, and consistent typography.

#### How Article Generation Works: The Complete Flow

The article pipeline transforms raw content into a structured, publication-ready document through a carefully orchestrated sequence of prompts and rendering steps:

```
Raw Content
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. CONTENT TRANSFORMATION (LLM Prompt #1)                   │
│    System: "You are an expert technical writer..."          │
│    Input: Raw markdown, content type, audience              │
│    Output: Structured JSON (title, sections, takeaways)     │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. IMAGE DECISION (LLM Prompt #2 - Per Section)             │
│    System: "Decide whether a section needs a visual..."     │
│    Input: Section title + content preview                   │
│    Output: {needs_image, image_type, prompt, confidence}    │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. IMAGE GENERATION (Gemini Imagen)                         │
│    Input: Image prompt + style guidance                     │
│    Output: Generated PNG/SVG image                          │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. IMAGE DESCRIPTION (LLM Prompt #3)                        │
│    System: "Write a concise blog-style description..."      │
│    Input: Section content + generated image                 │
│    Output: 2-4 sentence caption                             │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. DOCUMENT ASSEMBLY (ReportLab or Markdown Writer)         │
│    Input: Structured content + images + descriptions        │
│    Process: Align sections → embed images → add captions    │
│    Output: Final PDF or Markdown file                       │
└─────────────────────────────────────────────────────────────┘
```

**Sample Output**:

![Article PDF Sample](https://raw.githubusercontent.com/nitishkmr005/PrismDocs/main/sampleOutputs/Screenshots/Article%5BPDF%5D.png)

*Professional PDF with AI-generated images, captions, and structured sections*

**Download**: [Article PDF](https://github.com/nitishkmr005/PrismDocs/blob/main/sampleOutputs/Generated%20Documents/Article%5BPDF%5D.pdf) | [Article Markdown](https://github.com/nitishkmr005/PrismDocs/blob/main/sampleOutputs/Generated%20Documents/Article%5BMarkdown%5D.md)

### 2. Presentation (PPTX)

**Best for**: stakeholder updates, project reviews, and pitch decks.

**What it produces**: Executive-ready PowerPoint decks with title slides, agenda, executive summary, content slides, and embedded visuals.

**Sample Output**:

![Presentation Sample](https://raw.githubusercontent.com/nitishkmr005/PrismDocs/main/sampleOutputs/Screenshots/Slides%5BPDF%5D.png)

*Executive-ready PowerPoint with title slides, agenda, summary, and embedded visuals*

**Download**: [Slides PPTX](https://github.com/nitishkmr005/PrismDocs/blob/main/sampleOutputs/Generated%20Documents/Slides%5BPPTX%5D.pptx) | [Slides PDF](https://github.com/nitishkmr005/PrismDocs/blob/main/sampleOutputs/Generated%20Documents/Slides%5BPDF%5D.pdf)

### 3. Mindmap

**Best for**: planning sessions, discovery workshops, concept overviews, and rapid understanding of complex topics.

**What it produces**: Hierarchical JSON tree structures (up to 20 levels deep) with central concepts and nested branches.

**Sample Output**:

![Mindmap Sample](https://raw.githubusercontent.com/nitishkmr005/PrismDocs/main/sampleOutputs/Screenshots/Mindmap.png)

*Hierarchical tree structure visualizing concept relationships and dependencies*

**Download**: [Mindmap PNG](https://github.com/nitishkmr005/PrismDocs/blob/main/sampleOutputs/Generated%20Documents/Mindmap.png)

### 4. Podcast

**Best for**: audio-first audiences, hands-free learning, commuters, and auditory learners.

**What it produces**: Multi-speaker conversational scripts (JSON) + synthesized audio files (WAV) with natural dialogue flow.

**Sample Output**:

![Podcast Sample](https://raw.githubusercontent.com/nitishkmr005/PrismDocs/main/sampleOutputs/Screenshots/Podcast.png)

*Multi-speaker conversational audio with natural dialogue flow*

**Download**: [Podcast Audio WAV](https://github.com/nitishkmr005/PrismDocs/blob/main/sampleOutputs/Generated%20Documents/Podcast.wav)

### 5. FAQ

**Best for**: onboarding, troubleshooting, self-serve help pages, and knowledge base articles.

**What it produces**: Structured Q&A documents (JSON) with configurable detail levels, answer formats, and audience-specific language.

**Sample Output**:

![FAQ Sample](https://raw.githubusercontent.com/nitishkmr005/PrismDocs/main/sampleOutputs/Screenshots/FAQ.png)

*Structured Q&A with configurable detail levels and audience-specific language*

**Download**: [FAQ Markdown](https://github.com/nitishkmr005/PrismDocs/blob/main/sampleOutputs/Generated%20Documents/FAQ.md)

### 6. Image

**Best for**: single visual assets like headers, diagrams, or infographics.

**Sample Output** (Image Generation + Editing):

| Original Generated Image | Area Selection for Edit | Edited Result |
|:---:|:---:|:---:|
| ![Original](https://raw.githubusercontent.com/nitishkmr005/PrismDocs/main/sampleOutputs/Screenshots/Original%20Generated%20Image.png) | ![Selection](https://raw.githubusercontent.com/nitishkmr005/PrismDocs/main/sampleOutputs/Screenshots/Image%20Editing%20By%20Selecting%20area%20to%20edit.png) | ![Edited](https://raw.githubusercontent.com/nitishkmr005/PrismDocs/main/sampleOutputs/Screenshots/Edited%20Image.png) |

*Context-aware image generation with region-based editing capabilities*

**Download**: [Original Image](https://github.com/nitishkmr005/PrismDocs/blob/main/sampleOutputs/Generated%20Documents/Original%20Image.png) | [Edited Image](https://github.com/nitishkmr005/PrismDocs/blob/main/sampleOutputs/Generated%20Documents/Edited%20Image.png)

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
```

**Why Multiple Providers**:

1. **Cost optimization**: Different providers offer different pricing models
2. **Capability matching**: Each provider has unique strengths (e.g., Claude for visual reasoning)
3. **Redundancy**: Fallback if one provider has issues
4. **Flexibility**: Easy to swap providers per use case

---

## Intelligent Caching Strategy

Our three-layer caching system reduces costs and latency by avoiding redundant LLM calls and image generation.

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

**Benefit**: Full workflow skip for identical requests—eliminates all processing time and API costs

### Layer 2: Structured Content Cache

**What**: Caches transformed markdown based on content hash.

**Cache Key**: SHA-256 of normalized markdown

**Benefit**: Reuses transformed content when only output format changes—skips expensive LLM transformation step

### Layer 3: Image Cache + Manifest

**What**: Reuses generated images when content hash matches.

**Manifest**:

```json
{
  "content_hash": "sha256:abc123",
  "sections": [{ "title": "Intro", "image_path": "section_1.png" }]
}
```

**Benefit**: Avoids regenerating images for unchanged content—saves significant time and image generation costs

### Combined Impact

The three-layer caching architecture means:
- **Request cache**: Instant responses for repeated identical requests
- **Content cache**: Fast regeneration when only format changes (PDF → PPTX)
- **Image cache**: Stable images across runs with same content

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

- Cost optimization: Flexibility to choose most cost-effective provider per use case
- Capability matching: Can leverage each provider's strengths
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
- **Result**: Significantly improved image-section relevance

**Challenge 2: Section Numbering Across Chunks**

- **Problem**: Chunks had inconsistent numbering (1, 2, 1, 2 instead of 1, 2, 3, 4)
- **Solution**: Pass section counter and outline to each chunk
- **Result**: Consistent numbering across all sections

**Challenge 3: LLM Hallucination**

- **Problem**: LLM occasionally added facts not in source content
- **Solution**: Explicit system prompt: "Use ONLY information present in the source. Do not add new facts."
- **Result**: High fidelity to source content with minimal hallucinations

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

- Significant time reduction for repeated requests
- Substantial cost savings on LLM and image generation APIs
- Better user experience with faster response times

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

Building PrismDocs taught me that **making knowledge accessible is as important as creating it**. A brilliant research paper that nobody can understand has limited impact. The same paper transformed into a mind map, a podcast, and an FAQ can reach executives, commuters, and support teams alike.

The system doesn't try to be a human writer—it's a tool that handles the tedious parts (formatting, image creation, structure, format conversion) so humans can focus on the creative parts (ideas, strategy, storytelling).

### Core Principles

1. **Accessibility over Exclusivity**: One input, multiple outputs for different audiences
2. **Fidelity over Creativity**: Restructure, don't reinvent
3. **Caching over Computation**: Reuse whenever possible
4. **Flexibility over Lock-in**: Multi-provider, multi-format

### Try It Yourself

The system is open-source and production-ready:

```bash
# Clone the repo
git clone https://github.com/nitishkmr005/PrismDocs

# Install dependencies
make setup-prismdocs

# Generate your first document
python scripts/run_generator.py input.pdf --output pdf
```

**Resources**:

- [GitHub Repository](https://github.com/nitishkmr005/PrismDocs)
- [Example Outputs](https://github.com/nitishkmr005/PrismDocs/tree/main/sampleOutputs/Generated%20Documents)

---

*Last updated: February 1, 2026*
