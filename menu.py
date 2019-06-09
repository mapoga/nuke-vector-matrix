"""
Creates a menu and dynamically populate it with .gizmo, .so and .nk files
Supports icons by adding them either at the same level as the tool/subdir or in a /icons directory
All subdirectories are added to the nuke.pluginPath() (see init.py)
"""

import os
import re
import nuke
try:
    import scandir as walk_module
except ImportError:
    import os as walk_module

CWD = os.path.dirname((os.path.abspath(__file__)))


# Functions
def find_icon(name):
    path = os.path.join(CWD, 'icons')
    img = None
    for icon_ext in ['.jpg', '.png']:
        icon_path = os.path.join(path, name + icon_ext)
        if os.path.isfile(icon_path):
            img = icon_path

    return str(img) if img else None


def populate_menu_rcsv(tool_path, menu):
    if not tool_path.endswith(os.sep):
        tool_path += os.sep

    for root, dirs, files in walk_module.walk(tool_path):
        category = root.replace(tool_path, '')
        # build the dynamic menus, ignoring empty dirs:
        for dir_name in natural_sort(dirs):
            if os.listdir(os.path.join(root, dir_name)):
                img = find_icon(dir_name)
                menu.addMenu(os.path.join(category, dir_name), icon=img)

        # if we have both dirs and files, add a separator
        if files and dirs:
            submenu = menu.addMenu(category)  # menu() and findItem() do not return a menu object.
            submenu.addSeparator()

        # Populate the menus
        for f in natural_sort(files):
            f_name, ext = os.path.splitext(f)
            if ext.lower() in ['.gizmo', '.so', '.nk']:
                img = find_icon(f_name)
                # Adding the menu command
                if ext.lower() in ['.nk']:
                    menu.addCommand(os.path.join(category, f_name),
                                    'nuke.nodePaste( "{}" )'.format(os.path.join(root, f)),
                                    icon=img)
                if ext.lower() in ['.gizmo', '.so']:
                    menu.addCommand(os.path.join(category, f_name),
                                    'nuke.createNode( "{}" )'.format(f_name),
                                    icon=img)
    return menu


def natural_sort(values, case_sensitive=False):
    """
    Returns a human readable list with integers accounted for in the sort.
    items = ['xyz.1001.exr', 'xyz.1000.exr', 'abc11.txt', 'xyz.0999.exr', 'abc10.txt', 'abc9.txt']
    natural_sort(items) = ['abc9.txt', 'abc10.txt', 'abc11.txt', 'xyz.0999.exr', 'xyz.1000.exr', 'xyz.1001.exr']
    :param values: string list
    :param case_sensitive: Bool. If True capitals precede lowercase, so ['a', 'b', 'C'] sorts to ['C', 'a', 'b']
    :return: list
    """
    def alpha_to_int(a, _case_sensitive=False):
        return int(a) if a.isdigit() else (a if _case_sensitive else a.lower())

    def natural_sort_key(_values):
        try:
            return tuple(alpha_to_int(c, case_sensitive) for c in re.split('(\d+)', _values) if c)
        except (TypeError, ValueError):
            return _values

    return sorted(values, key=natural_sort_key)


# Running code
toolbar = nuke.toolbar("Nodes")
toolbar_math_tools = toolbar.addMenu("Math Tools", icon=find_icon("Math"))

nuke_dir = os.path.join(CWD, 'nuke')

populate_menu_rcsv(nuke_dir, toolbar_math_tools)
