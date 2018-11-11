# nuke-vector-matrix
Suite of mathematical nodes for Nuke to deal with Vectors and Matrices, by @mapoga (Mathieu Goulet-Aubin) 
and @herronelou (Erwan Leroy).
 
### Table of Contents
**[Installation Instructions](#installation-instructions)**<br>
**[Manual Installation](#manual-installation)**<br>
**[Resources to learn about Vectors and Matrices](#resources-to-learn-about-vectors-and-matrices)**

## Installation Instructions
For easy installation of the toolset, we provide a script that will make menu entries for each of our tools and ensure 
they are all part of the Nuke plugin path.

Installation instructions are similar for any OS. However, the paths we are using in the example are formatted for an 
UNIX system (Mac or Linux).

1. Download the full content of the nuke-vector-matrix repository. If downloaded
as a .ZIP, uncompress the zip in the desired location. For the following steps, we will assume the folder is present 
on disk as: `/my_nuke_gizmos/nuke-vector-matrix/`.
2. Add the path to the folder in your NUKE_PATH, either via an environment variable ([Defining Nuke plugin path](
https://learn.foundry.com/nuke/content/comp_environment/configuring_nuke/defining_nuke_plugin_path.html)) or 
via an existing / new `init.py` file, in which you would add the line: 

    ```python
    nuke.pluginAddPath('/my_nuke_gizmos/nuke-vector-matrix/')
    ```
    
This should be enough to Install the suite of tools.


## Manual Installation
While the default installation is probably ideal for many users, it may not be the best for Studio Environments 
where tools need to be installed in a specific location or for users who already have their own Gizmo loader.

For manual installation of the tools, only the content of the `nuke` folder is necessary and contains all the .nk and 
.gizmo files. 
It can be reorganized as required.

.gizmo files need to be added to the nuke plugin path. See instructions by the foundry: 
- [Loading Gizmos, Plugins, Scripts](
https://learn.foundry.com/nuke/content/comp_environment/configuring_nuke/loading_gizmos_plugins_scripts.html)
- [Custom Menus](
https://learn.foundry.com/nuke/content/comp_environment/configuring_nuke/custom_menus_toolbars.html)


All the icons are located in the `icons` folder. 

## Resources to learn about Vectors and Matrices

Most tools in this toolset are mathematical tools and require some basic knowledge about Vectors and Matrices for 
optimal use.

- [Math is Fun: Scalar, Vector, Matrix](https://www.mathsisfun.com/algebra/scalar-vector-matrix.html)
- [Wikipedia: Transformation Matrices](https://en.wikipedia.org/wiki/Transformation_matrix)
- [Nukepedia: Python Vector and Matrix Math](http://www.nukepedia.com/written-tutorials/using-the-nukemath-python-module-to-do-vector-and-matrix-operations/)
- [Nukepedia: The Matrix Knob](http://www.nukepedia.com/expressions/enter-the-matrix-knob)


