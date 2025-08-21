#!/usr/bin/env python3
"""Test node imports and registration."""

import sys
import os

# Add parent directory to path for module import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly from modules
from nodes import NODE_CLASS_MAPPINGS as NODES_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as NODES_DISPLAY_MAPPINGS
from grabcut_nodes import NODE_CLASS_MAPPINGS as GRABCUT_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as GRABCUT_DISPLAY_MAPPINGS

# Combine mappings as done in __init__.py
NODE_CLASS_MAPPINGS = {**NODES_MAPPINGS, **GRABCUT_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**NODES_DISPLAY_MAPPINGS, **GRABCUT_DISPLAY_MAPPINGS}

print(f"✓ Successfully imported and combined node mappings")
print(f"Found {len(NODE_CLASS_MAPPINGS)} total nodes:")
for name, cls in NODE_CLASS_MAPPINGS.items():
    display = NODE_DISPLAY_NAME_MAPPINGS.get(name, "Unknown")
    print(f"  - {name}: {display}")

# Check GrabCut nodes specifically
grabcut_nodes = [k for k in NODE_CLASS_MAPPINGS if 'GrabCut' in k]
print(f"\n✓ Found {len(grabcut_nodes)} GrabCut nodes:")
for node in grabcut_nodes:
    print(f"  - {node}: {NODE_DISPLAY_NAME_MAPPINGS[node]}")