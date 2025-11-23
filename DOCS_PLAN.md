# Documentation Site Implementation Plan

## Overview
Create a Material for MkDocs documentation site similar to the BSBR library (https://jacobfv.github.io/bsbr/), providing comprehensive documentation for the eggroll-trainer library.

## Structure

### 1. Documentation Framework Setup
- **Tool**: Material for MkDocs
- **Configuration**: `mkdocs.yml` in project root
- **Documentation Source**: `docs/` directory
- **Build Output**: `site/` directory (gitignored)
- **Deployment**: GitHub Pages (via GitHub Actions)

### 2. Documentation Structure

```
docs/
├── index.md                    # Home page
├── getting-started/
│   ├── index.md               # Getting Started overview
│   ├── installation.md        # Installation instructions
│   └── quick-start.md         # Quick start guide
├── user-guide/
│   ├── index.md               # User Guide overview
│   ├── core-concepts.md       # ES concepts, EGGROLL algorithm
│   ├── trainers.md            # Using ESTrainer and EGGROLLTrainer
│   ├── fitness-functions.md   # Writing fitness functions
│   └── advanced-usage.md      # Advanced features, customization
├── api-reference/
│   ├── index.md               # API Reference overview
│   ├── base.md                # ESTrainer API
│   ├── simple.md              # SimpleESTrainer API
│   └── eggroll.md             # EGGROLLTrainer API
├── examples/
│   ├── index.md               # Examples overview
│   ├── basic-usage.md         # Basic example walkthrough
│   ├── mnist-classification.md # MNIST example
│   └── custom-trainers.md     # Creating custom trainers
└── research/
    ├── index.md               # Research overview
    ├── background.md          # Evolution Strategies background
    ├── eggroll-algorithm.md   # EGGROLL algorithm details
    └── benchmarks.md          # Performance benchmarks
```

### 3. Content Plan

#### Home Page (`docs/index.md`)
- Project title and tagline
- Key features (with icons/emojis)
- Quick start code snippet
- Installation link
- Links to main sections
- Badges (Python version, License, etc.)

#### Getting Started (`docs/getting-started/`)
- **Installation**: pip, uv, conda options
- **Quick Start**: Minimal working example
- Links to examples and user guide

#### User Guide (`docs/user-guide/`)
- **Core Concepts**: 
  - What are Evolution Strategies?
  - What is EGGROLL?
  - Low-rank vs full-rank perturbations
  - Fitness functions
- **Trainers**:
  - ESTrainer base class
  - EGGROLLTrainer usage
  - Parameter configuration
- **Fitness Functions**:
  - Writing effective fitness functions
  - Common patterns
  - Performance considerations
- **Advanced Usage**:
  - Custom ES algorithms
  - Device management
  - Population size tuning
  - Hyperparameter optimization

#### API Reference (`docs/api-reference/`)
- Auto-generated from docstrings (using mkdocstrings)
- Class documentation
- Method signatures
- Parameter descriptions
- Examples for each class

#### Examples (`docs/examples/`)
- **Basic Usage**: Walkthrough of `basic_example.py`
- **MNIST Classification**: Full example with EGGROLL vs SGD
- **Custom Trainers**: How to subclass ESTrainer

#### Research (`docs/research/`)
- **Background**: Evolution Strategies history and theory
- **EGGROLL Algorithm**: Deep dive into the algorithm
- **Benchmarks**: Performance comparisons, speedups

### 4. Configuration Files

#### `mkdocs.yml`
- Material theme configuration
- Navigation structure
- Plugins (mkdocstrings for API docs)
- Site metadata
- Extensions (code highlighting, admonitions, etc.)

#### `.github/workflows/docs.yml`
- GitHub Actions workflow
- Build and deploy documentation to GitHub Pages
- Trigger on pushes to main branch

### 5. Dependencies

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
docs = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.0.0",
    "mkdocstrings[python]>=0.23.0",
]
```

### 6. Implementation Steps

1. **Setup**
   - Create `docs/` directory
   - Create `mkdocs.yml` configuration
   - Add docs dependencies to `pyproject.toml`
   - Update `.gitignore` to exclude `site/`

2. **Content Creation**
   - Write home page (`docs/index.md`)
   - Create getting started pages
   - Write user guide sections
   - Document API reference structure
   - Create example walkthroughs
   - Write research/background content

3. **API Documentation**
   - Configure mkdocstrings plugin
   - Ensure docstrings are comprehensive
   - Test API reference generation

4. **Styling & Polish**
   - Configure Material theme (colors, logo, etc.)
   - Add code examples with syntax highlighting
   - Add diagrams/images where helpful
   - Ensure mobile responsiveness

5. **CI/CD**
   - Create GitHub Actions workflow for docs
   - Configure GitHub Pages deployment
   - Test build process

6. **Testing**
   - Build documentation locally
   - Verify all links work
   - Check code examples are correct
   - Test on different screen sizes

### 7. Features to Include

- **Search**: Material theme includes search
- **Code Highlighting**: Syntax highlighting for Python
- **Admonitions**: Callouts for tips, warnings, notes
- **Tabs**: For showing different installation methods
- **Code Blocks**: Copy-to-clipboard functionality
- **Navigation**: Sidebar navigation with search
- **Responsive**: Mobile-friendly design

### 8. Content Sources

- Extract from existing `README.md`
- Expand docstrings in code
- Use examples from `examples/` directory
- Reference EGGROLL paper/blog
- Include performance benchmarks from examples

### 9. Deployment

- GitHub Pages via GitHub Actions
- Update `pyproject.toml` documentation URL
- Add documentation badge to README
- Link from repository description

### 10. Maintenance

- Keep docs in sync with code changes
- Update examples when API changes
- Add new examples as they're created
- Keep research section updated

## Timeline Estimate

- Setup & Configuration: 30 min
- Content Creation: 2-3 hours
- API Documentation: 1 hour
- Styling & Polish: 1 hour
- CI/CD Setup: 30 min
- Testing & Fixes: 1 hour

**Total**: ~6-7 hours

## Success Criteria

- ✅ Documentation builds without errors
- ✅ All pages are accessible and linked correctly
- ✅ Code examples run successfully
- ✅ API reference is complete and accurate
- ✅ Site is deployed and accessible
- ✅ Mobile-responsive design
- ✅ Search functionality works
- ✅ Matches quality of BSBR documentation site

