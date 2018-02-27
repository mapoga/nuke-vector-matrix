import nuke
import os

def findParentDirSubMenu(toolbar='Nodes', limitSearchToMenus=True, searchedMenus=['Tools']):
	# Search for a menu matching the name of the directory containing this file
	# The search can be limited to only specified menus like custom ones.
	# This limit can be used to avoid adding menu to nuke's default menu.
	# The searched menu must reside at the first level of the specfified toolbar

	# Parent directory of the directory containning this file
	parentdirName = os.path.basename(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
	menus = nuke.toolbar(toolbar).items()
	depth = 0
	while menus:
		newMenus = []
		for idx, m in enumerate(menus):
			if type(m) == nuke.Menu:
				if parentdirName == m.name():
					return m
				if depth == 0 and limitSearchToMenus:
					if m.name() in searchedMenus:
						newMenus.extend(m.items())
				else:
					newMenus.extend(m.items())
			if idx == len(menus)-1:
				menus = newMenus
		depth += 1
	return None

def autoAddCommand(fileTypes=['.nk', '.gizmo'], sort=True):
	# Adds icon with the same name as the file with a '_icon' suffix
	initDir = os.path.dirname(os.path.realpath(__file__))
	files = []
	for i in os.listdir(initDir):
		filePath = os.path.join(initDir, i)
		if os.path.isfile(filePath):
			root, ext = os.path.splitext(filePath)
			if ext in fileTypes:
				files.append(filePath)
	if files:
		if sort:
			files.sort()
		for f in files:
			fullName = os.path.basename(f)
			name, ext = os.path.splitext(fullName)
			m.addCommand(name, 'nuke.createNode(\'{0}\')'.format(fullName), icon='{0}_icon.png'.format(name))


###########################
# Automated menu creation #
###########################

dirName = os.path.basename(os.path.dirname(os.path.realpath(__file__))) # name of the directory containning this file
#parentMenu = findParentDirSubMenu(toolbar='Nodes', limitSearchToMenus=True, searchedMenus=['User'])
parentMenu = nuke.toolbar('Nodes')
if parentMenu:
	m = parentMenu.addMenu(dirName, icon='{0}_icon.png'.format(dirName)) # Name menu after directory
	m.addSeparator() # Trick to force Menu class completion when no commands are added.
	autoAddCommand(fileTypes=['.nk', '.gizmo'], sort=True)