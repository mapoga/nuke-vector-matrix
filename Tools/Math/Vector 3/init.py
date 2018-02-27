import nuke
import os

# Add sub directories to plugin path
initDir = os.path.dirname(os.path.realpath(__file__))
for i in os.listdir(initDir):
	subDir = os.path.join(initDir, i)
	if os.path.isdir(subDir):
		nuke.pluginAddPath( subDir )