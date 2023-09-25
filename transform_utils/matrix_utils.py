"""
Set of utility functions to perform matrix operations in Nuke
Author: Erwan Leroy
"""

import threading
import math
import nuke
import nukescripts

from .tracker4_api import Tracker


class NodeMatrixWrapper(object):
    """
    Wrap nuke Node to unify the method used to set matrices on these nodes.
    Supported node classes: Transform (limited), CornerPin2D, Tracker4, Roto, RotoPaint, SplineWarp3
    """

    def __init__(self, node=None):
        self._supported_out_types = ['Transform', 'CornerPin2D', 'Tracker4', 'Roto', 'RotoPaint', 'SplineWarp3']
        self.type = None
        self.node = node
        self._set_type()

    def _set_type(self):
        """ Set `self.type` to a string corresponding to the type of node wrapped """
        if not self.node:
            self.type = None
        else:
            node_type = self.node.Class()
            if node_type in self._supported_out_types:
                self.type = node_type
            else:
                self.type = None
                raise ValueError("NodeMatrixWrapper does not support %s nodes" % node_type)

    def set_node(self, node):
        """ Wrap a new node """
        self.node = node
        self._set_type()

    def set_matrix_at(self, matrix, frame, set_animated=True, rotation_hint=None):
        """ Set the matrix on the wrapped node at a certain frame.

        :param nuke.math.Matrix4 matrix: The matrix to set
        :param int frame: The frame at which to set the matrix
        :param bool set_animated: Set knobs being set to animated if not already
        :param float rotation_hint: If setting the matrix to a node with rotation support, use this
                                    number (in degrees) as a hint, as matrices can only represent -180 to +180
                                    degrees for rotations. The result will use the closest value to that number
                                    that matches the matrix transform.
        """
        node = self.node

        if self.type == 'Transform':
            if set_animated:
                for knob_name in ['translate', 'rotate', 'scale', 'skewX']:
                    knob = node[knob_name]
                    if not knob.isAnimated():
                        knob.setAnimated()

            center_x = node['center'].getValueAt(frame, 0)
            center_y = node['center'].getValueAt(frame, 1)

            # If the pixel aspect ratio is not 1, a Transform would take it into account,
            # resulting in a different result than our Matrix. Compensate for it.
            par = node.format().pixelAspect()
            if par != 1.0:
                # The transform already sandwiches our transform with 2 pixel aspect transforms:
                # First is applies the pixel aspect ratio, then puts the matrix, then un-applies it.
                # In order to compensate, we need to do the opposite.

                # We start by centering our center (putting it a 0, 0)
                ctr_m = nuke.math.Matrix4()
                ctr_m.makeIdentity()
                ctr_m.translate(center_x, center_y, 0)
                # Prepare the matrix to stretch the PAR
                par_m = nuke.math.Matrix4()
                par_m.makeIdentity()
                par_m.scale(par, 1, 1)
                # To un_apply properly, we move center to 0, stretch, move center back in place.
                un_apply_par = ctr_m * par_m.inverse() * ctr_m.inverse()
                # To re-apply the transform, we need to do the opposite, but the center may have moved.
                # Get the new center
                vector = nuke.math.Vector3(center_x, center_y, 0)
                offset = matrix.transform(vector)
                ctr_m.translate(offset[0] - center_x, offset[1] - center_y, 0)
                # Now undo the PAR transformed, based on the new center
                apply_par = ctr_m * par_m * ctr_m.inverse()
                # Prep the final matrix
                matrix = apply_par * matrix * un_apply_par

            translate_x, translate_y, rotation, scale_x, scale_y, skew_x = decompose_matrix(matrix, center_x, center_y)
            node['translate'].setValueAt(translate_x, frame, 0)
            node['translate'].setValueAt(translate_y, frame, 1)
            node['rotate'].setValueAt(rotation, frame)
            node['scale'].setValueAt(scale_x, frame, 0)
            node['scale'].setValueAt(scale_y, frame, 1)
            node['skewX'].setValueAt(skew_x, frame)

        elif self.type in ['Roto', 'RotoPaint', 'SplineWarp3']:
            # We assume a layer Tracked_Layer1 is present, otherwise we make it
            layer = self._get_rp_layer('Tracked_Layer1')
            transform = layer.getTransform()
            matrix.transpose()
            for y_index in range(4):
                for x_index in range(4):
                    index = 4 * x_index + y_index
                    curve = transform.getExtraMatrixAnimCurve(x_index, y_index)
                    if set_animated:
                        curve.addKey(frame, matrix[index])
                    else:
                        curve.constantValue = matrix[index]
                    transform.setExtraMatrixAnimCurve(x_index, y_index, curve)
            node['curves'].changed()

        elif self.type == 'Tracker4':
            points = matrix_to_corners(matrix, self.node.width(), self.node.height())
            self.set_points_at(points, frame, set_animated)

        elif self.type == 'CornerPin2D':
            matrix_knob = node['transform_matrix']
            if set_animated and not matrix_knob.isAnimated():
                matrix_knob.setAnimated()
            matrix.transpose()
            for index in range(16):
                matrix_knob.setValueAt(matrix[index], frame, index)

        else:
            raise NotImplementedError("Method not implemented for class {}".format(self.type))

    def set_points_at(self, points, frame, set_animated=True):
        """ Set points in a certain position at a defined frame. Depending on the node type, the points might get
        converted to a matrix and set instead.

        :param list points: list of nuke.math.Vector2 points
        :param int frame: The frame at which to set the points
        :param bool set_animated: Set knobs being set to animated if not already
        """
        if self.type in ['Transform', 'Roto', 'RotoPaint', 'SplineWarp3']:
            matrix = corners_to_matrix(points, self.node.width(), self.node.height())
            self.set_matrix_at(matrix, frame, set_animated)

        elif self.type == 'CornerPin2D':
            for index, point in enumerate(points[0:4]):
                knob = self.node['to%d' % (index + 1)]
                if set_animated and not knob.isAnimated():
                    knob.setAnimated()
                knob.setValueAt(point.x, frame, 0)
                knob.setValueAt(point.y, frame, 1)

        elif self.type == "Tracker4":
            tracker = Tracker(self.node)
            while len(tracker) < len(points):
                tracker.add_point(translate=True, rotate=True, scale=True)
            for index, point in enumerate(points):
                track_x = tracker[index]['track_x']
                track_y = tracker[index]['track_y']
                if set_animated:
                    if not track_y.isAnimated():
                        track_x.setAnimated()
                    if not track_y.isAnimated():
                        track_y.setAnimated()
                track_x.setValueAt(point.x, frame)
                track_y.setValueAt(point.y, frame)

        else:
            raise NotImplementedError("Method not implemented for class {}".format(self.type))

    def _get_rp_layer(self, name):
        """ Get (or make) the rotopaint layer of a certain name
        :param str name: Name of the layer to obtain
        :return: The Layer
        """
        if self.type not in ['Roto', 'RotoPaint', 'SplineWarp3']:
            raise TypeError("Node doesn't support rotopaint layers")
        curve_knob = self.node['curves']
        layer = curve_knob.toElement(name)
        if layer is None:
            layer = nuke.rotopaint.Layer(curve_knob)
            layer.name = name
            curve_knob.rootLayer.append(layer)
        return layer


class ReformatWrapper(object):
    """ Wrapper class to easily extract Matrices from Reformats, including non-existent reformats. """

    NoneMode = 'none'
    WidthMode = 'width'
    HeightMode = 'height'
    FitMode = 'fit'
    FillMode = 'fill'
    DistortMode = 'distort'
    Modes = [NoneMode, WidthMode, HeightMode, FitMode, FillMode, DistortMode]

    @classmethod
    def from_node(cls, node):
        """ Init from an existing node """
        from_format = node.input(0).format() if node.input(0) else nuke.root().format()
        to_format = node.format()
        wrapper = cls(from_format, to_format)
        wrapper.resize = node['resize'].value()
        wrapper.center = node['center'].value()
        wrapper.flip = node['flip'].value()
        wrapper.flop = node['flop'].value()
        wrapper.turn = node['turn'].value()
        return wrapper

    def __init__(self, from_format=None, to_format=None):
        self.from_format = from_format
        self.to_format = to_format
        self.resize = self.NoneMode
        self.center = True
        self.flip = False
        self.flop = False
        self.turn = False

    def to_node(self):
        """
        Create a node for this reformat.
        Note that "from_format" is ignored as it will depend on what the node gets connected to.
        """
        # TODO
        raise NotImplementedError

    def get_matrix(self):
        """ Convert Reformat to a Matrix. """
        from_format = self.from_format
        to_format = self.to_format
        if not from_format or not to_format:
            raise ValueError("Both from_format and to_format must be provided in order to calculate a matrix.")
        from_par = from_format.pixelAspect()
        from_width = from_format.width() * from_par
        from_height = from_format.height()
        from_aspect = from_width / from_height
        to_par = to_format.pixelAspect()
        to_width = to_format.width() * to_par
        to_height = to_format.height()
        to_aspect = to_width / to_height

        # Make input pixels square
        par_in = nuke.math.Matrix4()
        par_in.makeIdentity()
        par_in.scale(from_par, 1, 1)

        # We need a few matrices:
        # center matrix ( places center of original res at 0, 0)
        center_matrix = nuke.math.Matrix4()
        center_matrix.makeIdentity()
        center_matrix.translate(from_width / -2.0, from_height / -2.0, 0)

        # If "Turn" is enabled, the source width and height must be swapped.
        if self.turn:
            from_width, from_height = from_height, from_width
            from_aspect = 1 / from_aspect

        # invert center matrix (places 0, 0 at center of new res)
        invert_center_matrix = nuke.math.Matrix4()
        invert_center_matrix.makeIdentity()
        if self.center:
            # If center is enabled, we want to go to the center of the new format
            invert_center_matrix.translate(to_width / 2.0, to_height / 2.0, 0)
        else:
            # otherwise just invert of center_matrix
            invert_center_matrix.translate(from_width / 2.0, from_height / 2.0, 0)

        # Resize matrix
        resize_mode = self.resize
        resize_matrix = nuke.math.Matrix4()
        resize_matrix.makeIdentity()
        if resize_mode == self.FillMode:
            # Fill is only really an alias to pick between width and height based on aspect
            resize_mode = self.WidthMode if to_aspect > from_aspect else self.HeightMode
        elif resize_mode == self.FitMode:
            # Fit is only really an alias to pick between width and height based on aspect
            resize_mode = self.WidthMode if to_aspect < from_aspect else self.HeightMode
        if resize_mode == self.WidthMode:
            scale_factor = to_width / float(from_width)
            resize_matrix.scale(scale_factor, scale_factor, 1)
        elif resize_mode == self.HeightMode:
            scale_factor = (to_height / float(from_height))
            resize_matrix.scale(scale_factor, scale_factor, 1)
        elif resize_mode == self.DistortMode:
            scale_factor_x = to_width / float(from_width)
            scale_factor_y = to_height / float(from_height)
            resize_matrix.scale(scale_factor_x, scale_factor_y, 1)

        # Flip/flop/turn
        flipflop_matrix = nuke.math.Matrix4()
        flipflop_matrix.makeIdentity()
        if self.turn:
            flipflop_matrix.rotateZ(math.radians(90))
        if self.flop:
            flipflop_matrix.scale(-1, 1, 1)
        if self.flip:
            flipflop_matrix.scale(1, -1, 1)

        # Apply the output pixel aspect ratio
        par_out = nuke.math.Matrix4()
        par_out.makeIdentity()
        par_out.scale(1 / to_par, 1, 1)

        # The flip/flop/turn is always centered, but not the resize
        if self.center:
            result = par_out * invert_center_matrix * resize_matrix * flipflop_matrix * center_matrix * par_in
        else:
            result = par_out * resize_matrix * invert_center_matrix * flipflop_matrix * center_matrix * par_in
        return result


# Panel Classes
class MergeTransformsPanel(nukescripts.PythonPanel):
    """ Panel presenting options for merging transforms """
    def __init__(self):
        nukescripts.PythonPanel.__init__(self, 'Merge Transforms')

        # CREATE KNOBS
        self.first = nuke.Int_Knob('first', 'First Frame')
        self.first.setValue(int(nuke.root()['first_frame'].value()))
        self.last = nuke.Int_Knob('last', 'Last Frame')
        self.last.setValue(int(nuke.root()['last_frame'].value()))
        self.force_cp = nuke.Boolean_Knob('force_cp', 'Force Merge as CornerPin')
        self.force_cp.setFlag(nuke.STARTLINE)
        self.force_cp.setTooltip('Tool will merge transforms a a new Transform if possible, or Cornerpin if necessary.'
                                 '\nChecking this box will force a corner pin output')
        self.force_matrix = nuke.Boolean_Knob('force_matrix', 'CornerPin as extra_matrix')
        self.force_matrix.setTooltip("Uses the cornerpin's extra_matrix to match the transform rather than the corners")
        self.force_matrix.setEnabled(False)
        self.force_matrix.setFlag(nuke.STARTLINE)

        # ADD KNOBS
        for k in (self.first, self.last, self.force_cp, self.force_matrix):
            self.addKnob(k)

    def knobChanged(self, knob):
        """ knobChanged callback """
        # ONLY SHOW FORCE MATRIX IF CORNERPIN IS ON
        if knob is self.force_cp:
            self.force_matrix.setEnabled(self.force_cp.value())


class MatrixConversionPanel(nukescripts.PythonPanel):
    """ Panel presenting options for converting Matrices from a node class to another"""
    def __init__(self):
        nukescripts.PythonPanel.__init__(self, 'Corner Pin To Matrix')

        # ANALYZE NUKE SCRIPT TO GATHER VALUES
        camera_nodes = []
        nodes_with_matrix = []
        selected_cam = None
        selected_node = None
        selected_reformat = None
        for node in nuke.allNodes():
            if 'Camera' in node.Class():
                camera_nodes.append(node.name())
                if node.isSelected():
                    selected_cam = node
            elif node.Class() in ['Transform', 'CornerPin2D', 'Tracker4', 'Card2', 'Card3D']:
                nodes_with_matrix.append(node.name())
                if node.isSelected():
                    selected_node = node
            elif node.Class() == 'Reformat':
                if node.isSelected() and node['type'].getValue() == 0.0:
                    selected_reformat = node
        camera_nodes.sort()
        nodes_with_matrix.sort()

        # CREATE KNOBS
        self.first = nuke.Int_Knob('first', 'First Frame')
        self.first.setValue(int(nuke.root()['first_frame'].value()))
        self.last = nuke.Int_Knob('last', 'Last Frame')
        self.last.setValue(int(nuke.root()['last_frame'].value()))
        self.last.clearFlag(nuke.STARTLINE)
        self.node = nuke.Enumeration_Knob('original_node', 'Node to Convert', nodes_with_matrix)

        # Set defaults based on selection
        if selected_node:
            self.node.setValue(selected_node.name())
        self.camera = nuke.Enumeration_Knob('camera_node', 'Camera', camera_nodes)
        if selected_cam:
            self.camera.setValue(selected_cam.name())

        # In cases where no node was selected in the first place, the current node is the first entry in the list
        node = nuke.toNode(self.node.value())
        if not node or node.Class() not in ['Card2', 'Card3D']:
            self.camera.setVisible(False)
        options = ['Roto',
                   'RotoPaint',
                   'CornerPin',
                   'CornerPin (Matrix only)',
                   'Transform (No Perspective)',
                   'Tracker',
                   'SplineWarp']
        self.destination = nuke.Enumeration_Knob('target', 'Convert to', options)

        div1 = nuke.Text_Knob('div1', '')
        self.force_ref = nuke.Boolean_Knob('force_reference', '')
        self.force_ref.setTooltip("Forces the resulting node to leave the reference frame untouched")
        self.force_ref.setFlag(nuke.STARTLINE)
        self.reference = nuke.Int_Knob('reference', 'Reference Frame')
        self.reference.clearFlag(nuke.STARTLINE)
        self.reference.setEnabled(False)
        self.reference.setValue(nuke.frame())

        self.invert = nuke.Boolean_Knob('invert', 'Invert Matrix')

        # Reformat knobs
        div2 = nuke.Text_Knob('div2', '')
        self.specify_format = nuke.Boolean_Knob('specify_format', 'Specify Format')
        self.specify_format.setTooltip("If Enabled, will allow you to specify at which resolution the "
                                       "transform should be converted.\nIf not enabled, the resolution "
                                       "will be guessed based on the node being converted.")
        self.specify_format.setFlag(nuke.STARTLINE)
        self.format = nuke.Format_Knob('format')

        if selected_reformat:
            self.specify_format.setValue(True)
            self.format.setValue(selected_reformat['format'].value())
        else:
            self.format.setEnabled(False)

        div3 = nuke.Text_Knob('div3', '')

        # ADD KNOBS
        for k in (self.first, self.last, self.node, self.camera, self.destination, div1, self.force_ref, self.reference,
                  self.invert, div2, self.specify_format, self.format, div3):
            self.addKnob(k)

    def knobChanged(self, knob):
        """ knobChanged callback """
        if knob is self.node:
            node = nuke.toNode(self.node.value())
            if node and node.Class() in ['Card2', 'Card3D']:
                self.camera.setVisible(True)
            else:
                self.camera.setVisible(False)
        elif knob is self.force_ref:
            self.reference.setEnabled(knob.value())
        elif knob is self.specify_format:
            value = knob.value()
            self.format.setEnabled(value)


class TrackerToSplinewarpPanel(nukescripts.PythonPanel):
    """ Panel presenting options for converting a tracker to splinewarp """
    def __init__(self, ref=1001):
        nukescripts.PythonPanel.__init__(self, 'Merge Transforms')

        # CREATE KNOBS
        self.mode = nuke.Enumeration_Knob('mode', 'Mode', ['Bake values', 'Expression Link'])
        self.mode.setTooltip("You can either create the SplineWarp in Baked mode, where the SplineWarp "
                             "becomes self-contained, or expression-linked, where the positions are kept "
                             "live from the tracker.\nNote that new points added to the tracker after expression "
                             "linking will not be added to the SplineWarp, and deleted points on the tracker will "
                             "result in points at 0, 0 in the SplineWarp.\nBaking recommended.")
        self.first = nuke.Int_Knob('first', 'First Frame')
        self.first.setValue(int(nuke.root()['first_frame'].value()))
        self.last = nuke.Int_Knob('last', 'Last Frame')
        self.last.setValue(int(nuke.root()['last_frame'].value()))
        self.last.clearFlag(nuke.STARTLINE)
        self.ref = nuke.Int_Knob('ref', 'Reference Frame')
        self.ref.setValue(ref)
        self.ref.setTooltip("At this frame, the Splinewarp will not deform the image.\n"
                            "Defaults to the Tracker's reference frame knob.")
        self.direction = nuke.Enumeration_Knob('direction', 'Direction', ['Stabilize', 'Matchmove'])
        self.direction.setTooltip("Choose whether to Stabilize or Matchmove.\nNote that these operations"
                                  "are not perfectly opposite and applying both a stabilize and a matchmove after one "
                                  "another will approximate the original plate but not match it 100%.")

        # ADD KNOBS
        for k in (self.mode, self.first, self.last, self.ref, self.direction):
            self.addKnob(k)

    def knobChanged(self, knob):
        """ knobChanged callback """
        if knob is self.mode:
            self.first.setEnabled(knob.getValue() == 0)
            self.last.setEnabled(knob.getValue() == 0)


# Defining Helper Functions
def check_classes(nodes, allowed_classes):
    """ Check that the classes of a given list of nodes are all allowed classes """
    valid = True
    for node in nodes:
        if node.Class() not in allowed_classes:
            nuke.message("Please select only supported Nodes: " + ', '.join(allowed_classes))
            valid = False
            break
    return valid


def corners_to_matrix(corners, frame_width, frame_height):
    """ Generate a matrix from 4 corner points

    :param list corners: List of 4 corners as vector2+ objects
    :param int frame_width:
    :param int frame_height:
    :return:
    """
    if len(corners) != 4:
        raise ValueError("corners_to_matrix() requires 4 points")
    to_matrix = nuke.math.Matrix4()
    to_matrix.mapUnitSquareToQuad(
        corners[0].x, corners[0].y, corners[1].x, corners[1].y, corners[2].x, corners[2].y, corners[3].x, corners[3].y)
    from_matrix = nuke.math.Matrix4()
    from_matrix.mapUnitSquareToQuad(0, 0, frame_width, 0, frame_width, frame_height, 0, frame_height)
    return to_matrix * from_matrix.inverse()


def decompose_matrix(matrix, center_x=0, center_y=0):
    """ Decompose a matrix into translation, rotation, scale, skew """
    # Solve Translation
    vector = nuke.math.Vector3(center_x, center_y, 0)
    vector_trans = matrix.transform(vector)
    translate_x = vector_trans[0] - center_x
    translate_y = vector_trans[1] - center_y
    # Solve Rotation/Scale/Skew.
    # Skew Y is never solved, will be reflected in Rotation instead.
    delta = (matrix[0] * matrix[5]) - (matrix[4] * matrix[1])
    ratio = pow(matrix[0], 2) + pow(matrix[1], 2)
    rotation = math.degrees(math.atan2(matrix[1], matrix[0]))
    scale_x = math.sqrt(ratio)
    scale_y = delta / scale_x
    skew_x = (matrix[0] * matrix[4] + matrix[1] * matrix[5]) / delta
    return translate_x, translate_y, rotation, scale_x, scale_y, skew_x


def get_camera_projection_matrix(camera, frame, image_format):
    """ Calculate a camera's projection matrix (can be used to reconcile points)

    :param nuke.Node camera: Camera node
    :param int frame: frame number
    :param image_format: nuke format for aspect ratio calculations
    :return: projection matrix
    """
    # modified code from nukescripts/Snap3D
    # We support camera == None for the case of ScanlineRender or Card3D without a CAM, they use a default cam

    # Matrix to transform points into camera-relative coordinates.
    matrix_world = nuke.math.Matrix4()
    if camera:
        for index in range(16):
            matrix_world[index] = camera['world_matrix'].getValueAt(frame, index)
        matrix_world.transpose()
        cam_transform = matrix_world.inverse()
    else:
        matrix_world.makeIdentity()
        cam_transform = matrix_world

    # Matrix to take the camera projection knobs into account

    post_matrix = nuke.math.Matrix4()
    post_matrix.makeIdentity()
    if camera:
        roll = float(camera['winroll'].getValueAt(frame, 0))
        scale_x = float(camera['win_scale'].getValueAt(frame, 0))
        scale_y = float(camera['win_scale'].getValueAt(frame, 1))
        translate_x = float(camera['win_translate'].getValueAt(frame, 0))
        translate_y = float(camera['win_translate'].getValueAt(frame, 1))
        post_matrix.rotateZ(math.radians(roll))
        post_matrix.scale(1.0 / scale_x, 1.0 / scale_y, 1.0)
        post_matrix.translate(-translate_x, -translate_y, 0.0)

    # Projection matrix based on the focal length, aperture and clipping planes of the camera
    if camera:
        focal_length = float(camera['focal'].getValueAt(frame))
        h_aperture = float(camera['haperture'].getValueAt(frame))
        near = float(camera['near'].getValueAt(frame))
        far = float(camera['far'].getValueAt(frame))
        projection_mode = int(camera['projection_mode'].getValueAt(frame))
    else:
        focal_length = 50.0
        h_aperture = 50.0
        near = 0.1
        far = 10000.0
        projection_mode = 0
    projection_matrix = nuke.math.Matrix4()
    projection_matrix.projection(focal_length / h_aperture, near, far, projection_mode == 0)

    # Matrix to translate the projected points into normalised pixel coords
    image_aspect = float(image_format.height()) / float(image_format.width())
    aspect_matrix = nuke.math.Matrix4()
    aspect_matrix.makeIdentity()
    aspect_matrix.translate(1.0, 1.0 - (1.0 - image_aspect / float(image_format.pixelAspect())), 0.0)

    # Matrix to scale normalised pixel coords into actual pixel coords.
    x_scale = float(image_format.width()) / 2.0
    y_scale = x_scale * image_format.pixelAspect()
    format_matrix = nuke.math.Matrix4()
    format_matrix.makeIdentity()
    format_matrix.scale(x_scale, y_scale, 1.0)

    # The projection matrix transforms points into camera coords, modifies based
    # on the camera knob values, projects points into clip coords, translates the
    # clip coords so that they lie in the range 0,0 - 2,2 instead of -1,-1 - 1,1,
    # then scales the clip coords to proper pixel coords.
    return format_matrix * aspect_matrix * projection_matrix * post_matrix * cam_transform


def get_card_matrix(card, frame):
    """ Returns the matrix of a card, no matter the card settings (except distortion)"""

    try:
        image_format = card.input(0).format()
    except AttributeError:
        image_format = nuke.root()['format'].value()
    # grab data from our snapped card
    if card.Class() == 'Card3D' or card['image_aspect'].value():
        aspect = float(image_format.height()) / float(image_format.width()) / image_format.pixelAspect()
    else:
        aspect = 1.0

    # make a 0 -> 1px matrix
    card_matrix = nuke.math.Matrix4()
    card_matrix.makeIdentity()
    card_matrix.scale(1 / float(image_format.width()), 1 / float(image_format.height()), 1)
    # shift matrix to -0.5 -> 0.5
    center_matrix = nuke.math.Matrix4()
    center_matrix.makeIdentity()
    center_matrix.translate(-.5, -.5, 0)
    card_matrix = center_matrix * card_matrix
    # Set aspect ratio
    aspect_matrix = nuke.math.Matrix4()
    aspect_matrix.makeIdentity()
    aspect_matrix.scale(1, aspect, 1)
    card_matrix = aspect_matrix * card_matrix

    # Deal with camera built into the card node
    try:
        z_distance = card['z'].getValueAt(frame)
    except NameError:
        # We're in a Card3D
        z_distance = 0
    calculated_z = z_distance or 1.0

    # Project internal camera
    focal_ratio = card['lens_in_focal'].getValueAt(frame) / card['lens_in_haperture'].getValueAt(frame)
    internal_camera_matrix = nuke.math.Matrix4()
    internal_camera_matrix.projection(focal_ratio, -calculated_z, calculated_z, True)
    inversed_cam = internal_camera_matrix.inverse()
    # I can't figure out the proper matrix to transform directly, so I transform a vector and use that to
    # calculate a scale factor. I feel like I'm doing something redundant but can't figure out what.
    # The projection seems to scale everything down by 2
    corner_top_right = nuke.math.Vector4(1, aspect, 1, 1)
    original_length = nuke.math.Vector2(corner_top_right.x/2, corner_top_right.y/2).length()
    corner_top_right = inversed_cam.transform(corner_top_right)
    corner_top_right /= corner_top_right.w
    new_length = nuke.math.Vector2(corner_top_right.x, corner_top_right.y).length()
    scale_factor = new_length / original_length

    projection_matrix = nuke.math.Matrix4()
    projection_matrix.makeIdentity()
    projection_matrix.scale(scale_factor, scale_factor, 1)
    card_matrix = projection_matrix * card_matrix

    # offset internal card to calculated Z
    offset_matrix = nuke.math.Matrix4()
    offset_matrix.makeIdentity()
    offset_matrix.translate(0, 0, -z_distance)
    card_matrix = offset_matrix * card_matrix

    # Deal with card orientation
    try:
        orientation = card['orientation'].value()
    except NameError:
        orientation = 'XY'

    orientation_matrix = nuke.math.Matrix4()
    orientation_matrix.makeIdentity()
    if orientation == 'YZ':
        orientation_matrix[0] = 0
        orientation_matrix[2] = 1
        orientation_matrix[8] = 1
        orientation_matrix[10] = 0
    elif orientation == 'ZX':
        orientation_matrix[5] = 0
        orientation_matrix[6] = 1
        orientation_matrix[9] = 1
        orientation_matrix[10] = 0

    card_matrix = orientation_matrix * card_matrix

    # Handle card transformation
    values = card['matrix'].getValueAt(frame)
    matrix = nuke.math.Matrix4()
    for i in range(len(values)):
        matrix[i] = values[i]
    matrix.transpose()

    return matrix * card_matrix


def get_matrix_at_frame(node, frame):
    """ Calculate a matrix for a Transform, Tracker4 or CornerPin2D node at a given frame

    :param nuke.Node node: Node to extract the matrix from
    :param int frame: Frame number
    :return: a 4x4 matrix
    """
    matrix = None
    if node.Class() == 'Transform' or node.Class() == 'Tracker4':
        k = node.knob('matrix')
        context = nuke.OutputContext()
        context.setFrame(frame)
        matrix = k.value(context)
    elif node.Class() == 'CornerPin2D':
        # Calculate 'to' matrix
        to_matrix = nuke.math.Matrix4()
        to1x = node['to1'].getValueAt(frame)[0]
        to1y = node['to1'].getValueAt(frame)[1]
        to2x = node['to2'].getValueAt(frame)[0]
        to2y = node['to2'].getValueAt(frame)[1]
        to3x = node['to3'].getValueAt(frame)[0]
        to3y = node['to3'].getValueAt(frame)[1]
        to4x = node['to4'].getValueAt(frame)[0]
        to4y = node['to4'].getValueAt(frame)[1]
        to_matrix.mapUnitSquareToQuad(to1x, to1y, to2x, to2y, to3x, to3y, to4x, to4y)
        # Calculate 'from' matrix
        from_matrix = nuke.math.Matrix4()
        from1x = node['from1'].getValueAt(frame)[0]
        from1y = node['from1'].getValueAt(frame)[1]
        from2x = node['from2'].getValueAt(frame)[0]
        from2y = node['from2'].getValueAt(frame)[1]
        from3x = node['from3'].getValueAt(frame)[0]
        from3y = node['from3'].getValueAt(frame)[1]
        from4x = node['from4'].getValueAt(frame)[0]
        from4y = node['from4'].getValueAt(frame)[1]
        from_matrix.mapUnitSquareToQuad(from1x, from1y, from2x, from2y, from3x, from3y, from4x, from4y)
        # Calculate the extra matrix
        k = node.knob('transform_matrix')
        values = k.getValueAt(frame)
        extra_matrix = nuke.math.Matrix4()
        for index in range(len(values)):
            extra_matrix[index] = values[index]
        extra_matrix.transpose()

        matrix = extra_matrix * (to_matrix * from_matrix.inverse())

        if node['invert'].getValueAt(frame):
            matrix = matrix.inverse()

    elif node.Class() == 'Reformat':
        # Dealing with animated reformats is an absolute pain...
        # Nuke does not allow to query formats with a frame number, however,
        # getting it from TCL does seem to force Nuke to refresh its internal state,
        # and python then returns the correct values.
        original_frame = nuke.frame()  # Store current frame
        nuke.frame(frame)  # Set frame of interest
        nuke.tcl('value {}.pixel_aspect'.format(node.fullName()))  # Kick Nuke so it refreshes
        reformat = ReformatWrapper.from_node(node)
        matrix = reformat.get_matrix()
        nuke.frame(original_frame)  # Restore current frame so user UI doesn't change frame

    elif node.Class() == "Card3D":
        # Need to find the Camera, and check if there's a frame-hold
        cam_frame = frame
        cam = None
        top_node = node.input(1)
        while top_node:
            if top_node.Class() == 'FrameHold' and not top_node['disable'].value():
                if nuke.NUKE_VERSION_MAJOR >= 13:
                    cam_frame = top_node['firstFrame'].value()
                else:
                    cam_frame = top_node['first_frame'].value()
            elif top_node.Class() in ['Camera', 'Camera2', 'Camera3', 'Camera4']:
                cam = top_node
                break
            top_node = top_node.input(0)
        matrix = reconcile_card(node, cam, frame, camera_frame=cam_frame)

    return matrix


def matrix_to_corners(matrix, frame_width, frame_height):
    """ Convert a Matrix to 4 corners

    :param nuke.math.Matrix4 matrix: matrix to convert
    :param int frame_width:
    :param int frame_height:
    :return: list of corners
    """
    vec1 = nuke.math.Vector4(0, 0, 0, 1)
    vec1 = matrix.transform(vec1)
    vec1 /= vec1.w
    vec2 = nuke.math.Vector4(frame_width, 0, 0, 1)
    vec2 = matrix.transform(vec2)
    vec2 /= vec2.w
    vec3 = nuke.math.Vector4(frame_width, frame_height, 0, 1)
    vec3 = matrix.transform(vec3)
    vec3 /= vec3.w
    vec4 = nuke.math.Vector4(0, frame_height, 0, 1)
    vec4 = matrix.transform(vec4)
    vec4 /= vec4.w
    return [vec1, vec2, vec3, vec4]


def print_matrix4(matrix):
    """ Print a matrix """
    row = '| ' + 4 * '{: .4f} ' + '|'
    print('\n' + 'Matrix4'.center(35, '-'))
    print(row.format(matrix[0], matrix[4], matrix[8], matrix[12]))
    print(row.format(matrix[1], matrix[5], matrix[9], matrix[13]))
    print(row.format(matrix[2], matrix[6], matrix[10], matrix[14]))
    print(row.format(matrix[3], matrix[7], matrix[11], matrix[15]))
    print('-'*35)


def reconcile_card(card, camera, frame, camera_frame=None):
    """ Reconcile the matrix of a 3D card into a 2D matrix

    :param nuke.Node card: Card Node
    :param nuke.Node camera: Camera Node
    :param int frame: Frame number
    :rtype: nuke.math.Vector4
    :param int camera_frame: Frame number to use for the Camera. Optional. Uses the frame if not provided.
    """
    try:
        image_format = card.input(0).format()
    except AttributeError:
        image_format = nuke.root()['format'].value()
    card_matrix = get_card_matrix(card, frame)
    cam_matrix = get_camera_projection_matrix(camera, camera_frame or frame, image_format)
    if cam_matrix is None:
        raise RuntimeError("matrix_util.get_camera_projection() returned None for camera.")
    matrix_2d = cam_matrix * card_matrix

    # 2D Matrices don't like to have 3D attributes, let's kill them
    matrix_2d[10] = 1
    for index in [2, 6, 8, 9, 11, 14]:
        matrix_2d[index] = 0
    try:
        matrix_2d = matrix_2d / matrix_2d[15]
    except ZeroDivisionError:
        pass
    # TODO: When the card is behind the camera, right now the 2D matrix doesn't know.
    #       Hard to handle as the card center could be behind while some corners are in front.
    return matrix_2d


def set_cornerpin_to_size(node, width, height, set_from=True, set_to=True):
    """
    Set the 'to' and/or 'from' points on a CornerPin to the provided width/height
    """

    for (point, (x, y)) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
        if set_from:
            node['from%d' % (point + 1)].setValue((x * width, y * height))
        if set_to:
            node['to%d' % (point + 1)].setValue((x * width, y * height))


def sort_nodes(node_list):
    """ Sort nodes in tree order """
    # Sorts selected nodes by number of parents of an allowed class
    nodes_in_list = 0
    sorted_list = []
    for node in node_list:
        has_parents = True
        number_of_nodes = 1
        list_of_nodes = [node]
        # we count how many parents the node has
        while has_parents:
            parent = node.input(0)
            if parent:
                if parent['selected'].value():
                    node = parent
                    number_of_nodes += 1
                    list_of_nodes.append(node)
                else:
                    has_parents = False
            else:
                has_parents = False

                # the node with the biggest number of parents is our last node
        if number_of_nodes > nodes_in_list:
            nodes_in_list = number_of_nodes
            sorted_list = list_of_nodes

    # We want our first node first though, so we reverse the list
    sorted_list.reverse()
    return sorted_list


# Defining core functions
def merge_transforms(transform_list, first, last, cornerpin=False, force_matrix=False):
    """ Merge multiple nodes with a 2D matrix together

    :param list transform_list: List of Nodes to merge
    :param int first: First Frame
    :param int last: Last Frame
    :param bool cornerpin: Makes a CornerPin if True, else tries to make a Transform
    :param bool force_matrix: In case of Cornerpin, use the extra matrix instead of corners
    """

    def _calculate_rotation_hint(_node, _matrix, _frame):
        if 'rotate' in _node.knobs():
            return _node['rotate'].getValueAt(_frame)
        else:
            return math.degrees(math.atan2(_matrix[1], _matrix[0]))

    # Set Threading
    task = nuke.ProgressTask("Merging Transforms")
    task.setMessage("Checking Settings")
    # Check if we have Cornerpins in the list
    for node in transform_list:
        if node.Class() in ['CornerPin2D', 'Card3D']:
            cornerpin = True
            break

    # Our nodes resolution might be useful too
    height = transform_list[0].height()
    width = transform_list[0].width()

    # Create the node to receive the baked transformations
    if cornerpin:
        new_node = nuke.nodes.CornerPin2D(inputs=[transform_list[0].input(0)])
        set_cornerpin_to_size(new_node, width, height)

    else:
        new_node = nuke.nodes.Transform(inputs=[transform_list[0].input(0)])
        new_node['center'].setValue(width / 2, 0)
        new_node['center'].setValue(height / 2, 1)

    new_node.setXpos(transform_list[0].xpos() + 100)
    new_node.setYpos(transform_list[0].ypos())
    new_node['label'].setValue("Merged Transforms")

    wrapped_node = NodeMatrixWrapper(new_node)

    animated = first != last

    task.setMessage("Merging transforms")

    # We need the calculation for each frame
    try:
        for frame in range(first, last + 1):
            if task.isCancelled():
                break
            # set thread progress
            task.setProgress(int((frame - first) / ((last - first + 1) * 0.01)))
            # Generate matrix
            current_matrix = get_matrix_at_frame(transform_list[0], frame)
            # Keep track of all the rotations to avoid breaking rotations when doing a full flip.
            # Will handle transforms well, but might fail in situations where a full flip in introduced by a cornerpin.
            # Luckily, it's really annoying to do rotations with cornerpin, so I don't expect to run into this often.
            rotation_hint = _calculate_rotation_hint(transform_list[0], current_matrix, frame)

            # We merge the nodes 2 by two
            # TODO: Keep track of reformats to include them if needed. We can just use the last node's format as a shortcut and compare if different from the first node's
            for transform in transform_list[1:]:
                # Access the matrix knobs of the next transformation
                transform_matrix = get_matrix_at_frame(transform, frame)
                rotation_hint += _calculate_rotation_hint(transform, transform_matrix, frame)
                current_matrix = transform_matrix * current_matrix

            if force_matrix or not cornerpin:
                wrapped_node.set_matrix_at(current_matrix, frame, animated, rotation_hint=rotation_hint)
            else:
                points = matrix_to_corners(current_matrix, width, height)
                wrapped_node.set_points_at(points, frame, animated)

    finally:
        task.setProgress(100)
        del task


def do_matrix_conversion(old_node, new_class, first, last,
                         raw_matrix=False, camera=None, reference_frame=None, invert=False, target_format=None):
    """ Create a new node with a matrix from another node, for multiple frames.

    :param nuke.Node old_node: Original Node to extract the matrix from
    :param str new_class: Name of the new class of node to create
    :param int first: First Frame
    :param int last: Last Frame
    :param bool raw_matrix: For nodes that can have either corners or a matrix set, set True to force matrix
    :param nuke.Node camera: Camera Node, only used when converting a card
    :param int reference_frame: Set frame as reference frame (makes the resulting matrix identity at that frame)
    :param bool invert: Invert the matrix
    :param nuke.Format target_format: If provided, bake the matrix as if in that format.
    """
    # Set Threading
    task = nuke.ProgressTask("Converting Matrix")
    task.setMessage("Checking Settings")

    # Deselect Nodes
    for node in nuke.selectedNodes():
        node.setSelected(False)

    try:
        image_format = old_node.input(0).format()
    except AttributeError:
        image_format = nuke.root()['format'].value()

    if target_format is None:
        reformat_node = None
        reformat_matrix = None
        target_format = image_format
    else:
        reformat_node = nuke.nodes.Reformat()
        reformat_node['format'].setValue(target_format)
        reformat_node['pbb'].setValue(True)
        reformat_node.setInput(0, old_node.input(0))
        reformat_node.setXpos(old_node.xpos() + 100)
        reformat_node.setYpos(old_node.ypos())
        reformat_wrapper = ReformatWrapper.from_node(reformat_node)
        reformat_matrix = reformat_wrapper.get_matrix()

    # Create the node to receive the baked transformations
    new_node = nuke.createNode(new_class, inpanel=False)
    if reformat_node:
        new_node.setInput(0, reformat_node)
        new_node.setYpos(old_node.ypos() + 36)
    else:
        new_node.setInput(0, old_node.input(0))
        new_node.setYpos(old_node.ypos())
    new_node.setXpos(old_node.xpos() + 100)

    label_string = "Baked Matrix from {}".format(old_node.name())
    label_string += "\n{}x{}".format(int(target_format.width()), int(target_format.height()))
    if reference_frame is not None:
        label_string += "\nReference Frame {}".format(reference_frame)
    new_node['label'].setValue(label_string)

    if new_class == 'CornerPin2D':
        set_cornerpin_to_size(new_node, target_format.width(), target_format.height())
    elif new_class == "Transform":
        new_node['center'].setValue(target_format.width() / 2, 0)
        new_node['center'].setValue(target_format.height() / 2, 1)
    elif new_class in ['Roto', 'RotoPaint']:
        # Set format value, so we can disconnect the node without risk
        new_node['format'].setValue(target_format)

    wrapped_node = NodeMatrixWrapper(new_node)

    animated = first != last

    task.setMessage("Baking Matrix")
    ref_matrix = None
    if reference_frame is not None:
        if old_node.Class() in ['Transform', 'CornerPin2D', 'Tracker4']:
            ref_matrix = get_matrix_at_frame(old_node, reference_frame)
        elif old_node.Class() in ['Card2', 'Card3D']:
            ref_matrix = reconcile_card(old_node, camera, reference_frame)

    # We need the calculation for each frame
    try:
        for frame in range(first, last + 1):
            if task.isCancelled():
                break

            matrix = None

            if old_node.Class() in ['Transform', 'CornerPin2D', 'Tracker4']:
                matrix = get_matrix_at_frame(old_node, frame)
            elif old_node.Class() in ['Card2', 'Card3D']:
                matrix = reconcile_card(old_node, camera, frame)

            if not matrix:
                raise RuntimeError("Something went wrong, could not calculate matrix")
            if reformat_matrix:
                matrix = matrix * reformat_matrix.inverse()

            if reference_frame is not None:
                matrix = matrix * ref_matrix.inverse()

            if invert:
                matrix = matrix.inverse()

            if reformat_matrix:
                matrix = reformat_matrix * matrix

            if raw_matrix:
                wrapped_node.set_matrix_at(matrix, frame, animated)
            else:
                current_points = matrix_to_corners(matrix, target_format.width(), target_format.height())
                wrapped_node.set_points_at(current_points, frame, animated)

            # set thread progress
            task.setProgress(int(((frame - first) + 1) / (((last - first) + 1) * 0.01)))
    finally:
        task.setProgress(100)
        del task


def convert_tracker_to_splinewarp(tracker_node, first=0, last=0, ref=0, stabilize=False, link=False):
    """
    Convert a Tracker4 Node's point to a SpineWarp3 node
    Some inspiration taken from:
        https://www.nukepedia.com/python/misc/track-to-pins
        https://www.nukepedia.com/python/nodegraph/kj_tracker_to_pins
    """
    import nuke.splinewarp

    def _make_anim_curve(source_knob_name, track_point, is_ref=False):
        curve = nuke.splinewarp.AnimCurve()
        if link:
            if is_ref:
                curve.expressionString = '{}.tracks.{}.{}({})'.format(tracker_node.fullName(), track_point.index+1,
                                                                      source_knob_name, ref)
            else:
                curve.expressionString = '{}.tracks.{}.{}'.format(tracker_node.fullName(), track_point.index+1,
                                                                  source_knob_name)
            curve.useExpression = True

        else:
            if is_ref:
                curve.addKey(ref, track_point[source_knob_name].getValueAt(ref))
            else:
                for f in range(first, last+1):
                    curve.addKey(f, track_point[source_knob_name].getValueAt(f))
        return curve

    label = "{} {} - {}".format('Linked to' if link else "Baked from",
                                tracker_node.name(),
                                'Stabilize' if stabilize else 'MatchMove')

    tracker = Tracker(tracker_node)

    # Create the splinewarp node
    splinewarp_node = nuke.Node("SplineWarp3")
    splinewarp_node.setInput(0, tracker_node.input(0))
    splinewarp_node.setXpos(tracker_node.xpos() + 100)
    splinewarp_node.setYpos(tracker_node.ypos())
    splinewarp_node['label'].setValue(label)
    splinewarp_node['boundary_bbox'].setValue(False)
    splinewarp_node['crop_to_format'].setValue(False)

    curve_knob = splinewarp_node['curves']
    for point in tracker:
        point_name = point['name'].value()
        # Add a layer to keep things clean
        layer = nuke.splinewarp.Layer(curve_knob)
        layer.name = point_name
        curve_knob.rootLayer.append(layer)
        # We create the points at 0, 0, and animate them from the transform instead of keyframes on the point itself
        src = nuke.splinewarp.Shape(curve_knob, (0.0, 0.0))
        src.name = '{} source'.format(point_name)
        dst = nuke.splinewarp.Shape(curve_knob, (0.0, 0.0))
        dst.name = '{} destination'.format(point_name)
        src_transform = src.getTransform()
        dst_transform = dst.getTransform()
        for curve_index, knob_name in enumerate(['track_x', 'track_y']):
            src_transform.setTranslationAnimCurve(curve_index, _make_anim_curve(knob_name, point, is_ref=not stabilize))
            dst_transform.setTranslationAnimCurve(curve_index, _make_anim_curve(knob_name, point, is_ref=stabilize))
        layer.append(src)
        layer.append(dst)
        curve_knob.defaultJoin(src, dst)

    # Special hack:
    # The SpineWarp does not like it when we make single point Bezier Shapes, it prefers that we use "Pins"
    # It's unclear if Pins can be made directly by API, however, the only thing that changes if the cubic curve's
    # flags. It changes from 8192 to 8224. These flags are not documented so we do it via serialization.
    # 10000000000000 -> 10000000100000
    curve_knob.fromScript(curve_knob.toScript().replace("{f 8192}", "{f 8224}"))


# Defining runner functions
def run_merge_transforms():
    """ Show the merge transforms panel and starts the merge process"""
    nodes = nuke.selectedNodes()
    valid_nodes = check_classes(nodes, ['Transform', 'CornerPin2D', 'Tracker4', 'Reformat', 'Card3D'])
    if valid_nodes:
        transform_list = sort_nodes(nodes)
    else:
        return

    # We check that we have at least 2 transforms, otherwise no point in merging
    if len(transform_list) < 2:
        nuke.message("You need at least 2 transforms selected")
        return
    elif len(transform_list) != len(nodes):
        nuke.message("Please make sure all nodes form a single Branch")
        return
    panel = MergeTransformsPanel()
    if panel.showModalDialog():
        first = panel.first.value()
        last = panel.last.value()
        cornerpin = panel.force_cp.value()
        force_matrix = panel.force_matrix.value()
        exec_thread = threading.Thread(None, merge_transforms(transform_list, first, last, cornerpin, force_matrix))
        exec_thread.start()


def run_convert_matrix():
    """ Show the convert matrix panel and starts the conversion process"""
    # Display panel
    panel = MatrixConversionPanel()
    if panel.showModalDialog():
        first = panel.first.value()
        last = panel.last.value()
        node = nuke.toNode(panel.node.value())
        if not node:
            raise ValueError("No node to convert")
        camera = nuke.toNode(panel.camera.value())
        target = panel.destination.value()
        matrix = True
        if target == 'Roto':
            new_class = 'Roto'
        elif target == 'RotoPaint':
            new_class = 'RotoPaint'
        elif target == 'CornerPin':
            new_class = 'CornerPin2D'
            matrix = False
        elif target == 'CornerPin (Matrix only)':
            new_class = 'CornerPin2D'
        elif target == 'Transform (No Perspective)':
            new_class = 'Transform'
        elif target == 'Tracker':
            new_class = 'Tracker4'
        elif target == 'SplineWarp':
            new_class = 'SplineWarp3'
        else:
            raise ValueError('Unknown Conversion Target')
        invert = panel.invert.value()
        # Force a reference number
        ref = None
        if panel.force_ref.value():
            ref = panel.reference.value()

        # Set the format
        if panel.specify_format.value():
            target_format = panel.format.value()
        else:
            target_format = None

        exec_thread = threading.Thread(None, do_matrix_conversion(node, new_class, first, last,
                                                                  matrix, camera, ref, invert, target_format))
        exec_thread.start()


def run_convert_tracker_to_splinewarp():
    try:
        node = nuke.selectedNode()
    except ValueError:
        node = None

    if not node or node.Class() != 'Tracker4':
        nuke.critical("Please select exactly 1 Tracker4 node.")

    # We set the reference frame to be by default the node's reference frame, if it falls within the shot range
    ref_frame = node['reference_frame'].value()
    ref_frame = int(min(max(ref_frame, nuke.root()['first_frame'].value()), nuke.root()['last_frame'].value()))
    panel = TrackerToSplinewarpPanel(ref_frame)
    if panel.showModalDialog():
        first = panel.first.value()
        last = panel.last.value()
        ref = panel.ref.value()
        stabilize = panel.direction.getValue() == 0
        link = panel.mode.getValue() == 1
        convert_tracker_to_splinewarp(node, first, last, ref, stabilize=stabilize, link=link)
