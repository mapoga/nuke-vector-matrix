"""
recursively adding all subfolders to plugin path
"""

import os
import scandir
import nuke


CWD = os.path.dirname((os.path.abspath(__file__)))

# add Nuke Directory Recursively
nuke_dir = os.path.join(CWD, 'nuke')

for root, dirs, files in scandir.walk(nuke_dir):
    nuke.pluginAddPath(root)
