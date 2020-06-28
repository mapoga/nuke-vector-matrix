import nuke
import nukescripts
from math import pi, degrees, radians


# Math Functions


def split_axis_order(order='XYZ'):
    """ Converts a string 'XYZ' into a list of the corresponding indices: [0, 1, 2]

    :param str order: A string containing the letters X, Y or Z, in any order.
    :return: List of axis indices.
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    return [axis_map[axis] for axis in order.lower()]


def flip_euler(euler, rotation_order):
    axis0, axis1, axis2 = split_axis_order(rotation_order)

    flipped = nuke.math.Vector3(euler)
    flipped[axis0] += pi
    flipped[axis1] *= -1
    flipped[axis1] += pi
    flipped[axis2] += pi
    return flipped


def euler_filter_1d(previous, current):
    """ Naively rotates the current angle until it's as close as possible to the previous angle

    :param float previous: Previous angle in radians
    :param float current: Current angle in radians
    :return:  Modified current angle towards previous angle.
    """
    while abs(previous - current) > pi:
        if previous < current:
            current -= 2 * pi
        else:
            current += 2 * pi

    return current

def euler_filter_3d(previous, current, rotation_order="XYZ"):
    """ Attempts to minimize the amount of rotation between the current orientation and the previous one.

    Orientations are preserved, but amount of rotation is minimized.

    :param nuke.math.Vector3 previous: Previous XYZ rotation values as a vector
    :param nuke.math.Vector3 current: Current XYZ rotation values as a vector
    :param str rotation_order: String representing the rotation order (ex: "XYZ" or "ZXY")
    :return: Modified angles to minimize rotation
    :rtype: nuke.math.Vector3
    """
    # Start with a pass of Naive 1D filtering
    filtered = nuke.math.Vector3(current)
    for axis in range(3):
        filtered[axis] = euler_filter_1d(previous[axis], filtered[axis])

    # Then flip the whole thing and do another pass of Naive filtering
    flipped = flip_euler(filtered, rotation_order)
    for axis in range(3):
        flipped[axis] = euler_filter_1d(previous[axis], flipped[axis])

    # Return the vector with the shortest distance from the target value.
    if filtered.distanceSquared(previous) > flipped.distanceSquared(previous):
        return flipped
    return filtered


# Nuke functions
def get_keyframes_for_knob(knob):
    return sorted(list(set([key.x for curve in knob.animations() for key in curve.keys()])))


def filter_rotations(knob, strategy, use_3d=False, rotation_order=None, use_degrees=True):
    # TODO: Implement different strategies, cleanup implementation
    keyframes = get_keyframes_for_knob(knob)
    new_keys = []
    previous = None

    for frame in keyframes:
        # We probably want to sample all the values before we start modifying the curves, otherwise we may sample wrong stuff
        current = knob.valueAt(frame)
        if previous is not None:  # This filters out the first keyframe
            if use_3d:
                # TODO: Need to convert the lists to vectors, or not use vectors at all, this currently fails
                if use_degrees:
                    current = [radians(value) for value in current]
                    previous = [radians(value) for value in previous]
                current = euler_filter_3d(previous, current, rotation_order)
                if use_degrees:
                    current = [degrees(value) for value in current]
            else:
                if use_degrees:
                    current = radians(current)
                    previous = radians(previous)
                current = euler_filter_1d(previous, current)
                if use_degrees:
                    current = degrees(current)

        new_keys.append(current)
        previous = current

    for frame, value in zip(keyframes, new_keys):
        knob.setValueAt(value, frame)


def setup_filter_rotations(knob=None):

    valid_rotation_orders = ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX']
    strategies_1d = ["Minimum Rotation (Euler Filter)", "Preserve Angular Velocity"]
    strategies_3d = ["Minimum Rotation (Euler Filter)"]  # 3d Angular Velocity needs to be figured out (but could try Naive Angular, on a per curve basis)

    if knob is None:
        knob = nuke.thisKnob()

    rotation_order = None

    if knob.Class() == 'XYZ_Knob':
        use_3d_filter = True
    elif knob.Class() == 'Double_Knob':
        use_3d_filter = False
    else:
        raise ValueError("Don't know how to apply a rotation filter on knobs of type %s" % knob.Class())

    node = knob.node()
    # Figure out if the node has a built in rotation order
    if rotation_order is None and node.knob('rot_order'):
        value = node.knob('rot_order').value()
        if value in valid_rotation_orders:
            rotation_order = value

    # Build panel
    panel = RotationFilterPanel(
        strategies_3d if use_3d_filter else strategies_1d,
        rotation_orders= valid_rotation_orders if use_3d_filter and rotation_order is None else None)

    if panel.showModalDialog():
        filter_rotations(knob, None, use_3d_filter, rotation_order)
        # TODO: This is super WIP, will need to hookup the functions and arguments



# Panel Classes
class RotationFilterPanel(nukescripts.PythonPanel):
    """ Panel presenting options for merging transforms """
    def __init__(self, strategies, rotation_orders=None):
        nukescripts.PythonPanel.__init__(self, 'Rotation Filter')

        # CREATE KNOBS
        self.strategy = nuke.Enumeration_Knob('strategy', 'Filter Strategy', strategies)
        self.strategy.setTooltip('Pick a strategy for rotation filtering. Results might differ based on picked strategy')
        self.addKnob(self.strategy)

        if rotation_orders:
            self.rotation_order = nuke.Enumeration_Knob('rot_order', 'Rotation Order', rotation_orders)
            self.addKnob(self.rotation_order)

        self.unit = nuke.Enumeration_Knob('unit', 'Angle Units', ['Degrees', 'Radians'])
        self.addKnob(self.unit)

        # TODO: Option to specify initial Angular Velocity


nuke.menu('Animation').addCommand('Edit/Test', 'setup_filter_rotations()')