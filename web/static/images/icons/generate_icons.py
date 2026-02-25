#!/usr/bin/env python3
"""
Generate PWA icons from a source image.

Requirements:
    pip install Pillow

Usage:
    python generate_icons.py [source_image.png]

If no source image is provided, it will look for icon.png in the current directory.
"""

import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)


# Icon sizes to generate
ICON_SIZES = [
    (16, 16),
    (32, 32),
    (72, 72),
    (96, 96),
    (128, 128),
    (144, 144),
    (152, 152),
    (167, 167),
    (180, 180),
    (192, 192),
    (384, 384),
    (512, 512),
]

# Maskable icon sizes (need padding for safe zone)
MASKABLE_SIZES = [
    (192, 192),
    (512, 512),
]


def generate_icons(source_path: Path, output_dir: Path):
    """Generate all required icon sizes from a source image."""

    print(f"Loading source image: {source_path}")
    source = Image.open(source_path)

    # Convert to RGBA if necessary
    if source.mode != 'RGBA':
        source = source.convert('RGBA')

    # Generate standard icons
    print("\nGenerating standard icons...")
    for width, height in ICON_SIZES:
        output_path = output_dir / f"icon-{width}x{height}.png"
        resized = source.resize((width, height), Image.Resampling.LANCZOS)
        resized.save(output_path, 'PNG', optimize=True)
        print(f"  Created: {output_path.name}")

    # Generate maskable icons (with padding for safe zone)
    print("\nGenerating maskable icons...")
    for width, height in MASKABLE_SIZES:
        output_path = output_dir / f"icon-maskable-{width}x{height}.png"

        # Create a new image with padding (10% on each side for safe zone)
        padding = int(width * 0.1)
        inner_size = width - (padding * 2)

        # Resize source to fit within safe zone
        resized = source.resize((inner_size, inner_size), Image.Resampling.LANCZOS)

        # Create new image with background color
        maskable = Image.new('RGBA', (width, height), (26, 26, 46, 255))  # #1a1a2e

        # Paste resized image centered
        maskable.paste(resized, (padding, padding), resized)

        maskable.save(output_path, 'PNG', optimize=True)
        print(f"  Created: {output_path.name}")

    print("\nDone! All icons generated successfully.")


def main():
    # Determine source image path
    if len(sys.argv) > 1:
        source_path = Path(sys.argv[1])
    else:
        # Look for icon.png in current directory
        source_path = Path(__file__).parent / "icon.png"

    if not source_path.exists():
        print(f"Error: Source image not found: {source_path}")
        print("\nUsage: python generate_icons.py [source_image.png]")
        print("\nTo create icons:")
        print("1. Create a high-resolution (1024x1024) PNG image")
        print("2. Run: python generate_icons.py your_icon.png")
        sys.exit(1)

    output_dir = Path(__file__).parent
    generate_icons(source_path, output_dir)


if __name__ == "__main__":
    main()
