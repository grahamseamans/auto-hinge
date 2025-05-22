#!/bin/bash

# Create the mega doc
echo "# UV Documentation" > UV_MEGA_DOC.md

# Function to add a file to the compilation with section header
add_file() {
    local file=$1
    local section=$2
    
    echo "" >> UV_MEGA_DOC.md
    echo "# $section" >> UV_MEGA_DOC.md
    echo "" >> UV_MEGA_DOC.md
    cat "$file" >> UV_MEGA_DOC.md
}
echo "" >> compiled_docs/full_docs.md

# Core documentation
add_file "uv/README.md" "Overview and Quick Start"

# Getting Started
add_file "uv/docs/getting-started/index.md" "Getting Started Guide"
add_file "uv/docs/getting-started/first-steps.md" "First Steps with UV"

# Core Concepts
add_file "uv/docs/concepts/index.md" "Core Concepts Overview"
add_file "uv/docs/concepts/python-versions.md" "Python Version Management"
add_file "uv/docs/concepts/tools.md" "Tool Management"
add_file "uv/docs/concepts/resolution.md" "Package Resolution"
add_file "uv/docs/concepts/cache.md" "Caching System"

# Main Interfaces
add_file "uv/docs/pip/index.md" "The Pip Interface"
add_file "uv/docs/guides/scripts.md" "Script Management"
add_file "uv/docs/guides/projects.md" "Project Management"

# Additional Guides
add_file "uv/docs/guides/index.md" "Additional Guides"

echo "Documentation compiled to UV_MEGA_DOC.md"
