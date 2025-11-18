# GitHub Wiki Pages

This directory contains markdown files for the GitHub Wiki. These files need to be pushed to the wiki repository separately from the main code repository.

## Wiki Structure

The wiki includes the following pages:

- **Home.md** - Landing page with project overview and navigation
- **Architecture-Overview.md** - System architecture with Mermaid diagrams
- **Core-Components.md** - Detailed crate structure and responsibilities
- **Training-Algorithms.md** - Forward/backward propagation algorithms with sequences
- **Data-Flow.md** - Complete data flow with sequence diagrams
- **_Sidebar.md** - Navigation sidebar

## Features

✅ **Mermaid Diagrams**: All diagrams use Mermaid (no HTML break elements)
✅ **Proper Links**:
  - Wiki page-to-page: `[[Page-Name]]`
  - Wiki to repo files: `../../blob/main/path/to/file.md`
✅ **Comprehensive Coverage**: Architecture, algorithms, data flow, components
✅ **Visual Documentation**: UML-like diagrams for all key concepts

## How to Populate the GitHub Wiki

### Method 1: Using Git (Recommended)

```bash
# Clone the wiki repository
git clone https://github.com/softwarewrighter/neural-network-examples-rs.wiki.git

# Copy wiki files
cp wiki/*.md neural-network-examples-rs.wiki/

# Commit and push
cd neural-network-examples-rs.wiki
git add .
git commit -m "Add comprehensive architecture documentation with diagrams"
git push origin master
```

### Method 2: Using GitHub Web Interface

1. Go to the Wiki tab in your GitHub repository
2. Click "Create the first page" or "New Page"
3. For each markdown file:
   - Create a new page with the same name (without .md extension)
   - Copy the content from the corresponding .md file
   - Save the page

### Method 3: Using GitHub API

```bash
# Requires gh CLI tool
gh api repos/softwarewrighter/neural-network-examples-rs/wiki \
  --method POST \
  --field title="Home" \
  --field content="$(cat wiki/Home.md)"
```

## Link Formats Used

### Internal Wiki Links (Page to Page)
```markdown
[[Page-Name]]                    # Link to another wiki page
[[Custom Text|Page-Name]]        # Link with custom text
```

Examples in the wiki:
- `[[Architecture-Overview]]` - Links to Architecture-Overview page
- `[[Core-Components]]` - Links to Core-Components page

### Repository File Links (Wiki to Repo)
```markdown
../../blob/main/path/to/file.md           # Link to main branch file
../../tree/main/path/to/directory         # Link to directory
```

Examples in the wiki:
- `[Architecture Document](../../blob/main/documentation/architecture.md)`
- `[Source Code](../../tree/main/crates/neural-net-core/src)`

## Diagram Types Included

1. **System Architecture Diagrams**
   - Component relationships
   - Dependency graphs
   - Layer architecture

2. **Class Diagrams**
   - Data structures
   - Trait implementations
   - Relationships

3. **Sequence Diagrams**
   - Forward propagation flow
   - Backpropagation flow
   - Training loops
   - Initialization sequences

4. **Flow Charts**
   - Algorithm flows
   - Decision trees
   - State transitions

5. **Block Diagrams**
   - Module structure
   - Data flow
   - Memory layout

## Maintenance

When updating the wiki:

1. Update the markdown files in this directory
2. Commit changes to the main repository (for version control)
3. Copy updated files to the wiki repository
4. Push to the wiki repository

This ensures wiki content is version-controlled with the main codebase.

## Testing Diagrams

To test if Mermaid diagrams render correctly:

1. View on GitHub (wiki pages auto-render Mermaid)
2. Use VS Code with Mermaid extension
3. Use online editor: https://mermaid.live/

## Notes

- All Mermaid diagrams avoid HTML break elements (as requested)
- Links are consistent within their context (wiki-to-wiki vs wiki-to-repo)
- Sidebar uses `[[Page-Name]]` format for consistency
- Repository links use relative paths from wiki context
