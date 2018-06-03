# nuke-vector-matrix
Suite of mathematical nodes for Nuke to deal with Vectors and Matrices, by @mapoga and @herronelou.
 
### Table of Contents
**[Installation Instructions](#installation-instructions)**<br>
**[Advanced Installation](#advance-installation)**

## Installation Instructions
Installation instructions are similar for any OS. However, the paths we are using in the example are formatted for an 
UNIX system (Mac or Linux).

1. Download the full content of the nuke-vector-matrix repository. If downloaded
as a .ZIP, uncompress the zip in the desired location. For the following steps, we will assume the folder is present 
on disk as: `/my_nuke_gizmos/nuke-vector-matrix/`.
2. Add the path to the folder in your NUKE_PATH, either via an environment variable 
(https://learn.foundry.com/nuke/content/comp_environment/configuring_nuke/defining_nuke_plugin_path.html) or 
via an existing/new `init.py` file, in which you would add the line: 

    ```python
    nuke.pluginAddPath('/my_nuke_gizmos/nuke-vector-matrix/')
    ```
    
This should be enough to Install the suite of tools.


## Advanced Installation
We provide by default a menu.py and init.py to add all the tools in the NUKE_PATH and generate menu entried for them.

While this is ideal for many users, it may not be the best for Studio Environments where tools need to be installed 
in a specific manner.

For manual installation of the tools, only the content of the `nuke` folder is necessary, and can be reorganized as 
required.

All the icons are located in the `icons` folder. 




