#!/bin/bash
set -e

echo "Deploying docs/ to GitHub Wiki..."

# Ensure the docs directory exists
if [ ! -d "docs" ]; then
    echo "Error: docs/ directory not found in the current path."
    exit 1
fi

# Clone the wiki repository
WIKI_URL="https://github.com/KartavyaDikshit/Thermite.wiki.git"
TEMP_WIKI_DIR=$(mktemp -d)

echo "Cloning Wiki repository..."
if ! git clone "$WIKI_URL" "$TEMP_WIKI_DIR"; then
    echo "Error: Failed to clone Wiki repository."
    echo "Please ensure you have initialized the Wiki by creating the first page manually on GitHub."
    rm -rf "$TEMP_WIKI_DIR"
    exit 1
fi

# Copy all markdown files
echo "Copying documentation..."
cp docs/*.md "$TEMP_WIKI_DIR/"

# Commit and push
cd "$TEMP_WIKI_DIR"
git add .
git commit -m "Update Wiki documentation for v2.6.6" || echo "No changes to commit."
git push origin master

echo "Wiki deployed successfully!"

# Clean up
cd - > /dev/null
rm -rf "$TEMP_WIKI_DIR"
