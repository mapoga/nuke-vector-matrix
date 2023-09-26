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

    flipped = list(euler)
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


def distance_squared(vec1, vec2):
    """ Calculate distance between two vector3 represented as lists of len 3

    :param list vec1: List of 3 floats
    :param list vec2: List of 3 floats
    :return: Squared distance between the 2 vectors
    """
    return (vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2 + (vec1[2] - vec2[2])**2


def euler_filter_3d(previous, current, rotation_order="XYZ"):
    """ Attempts to minimize the amount of rotation between the current orientation and the previous one.

    Orientations are preserved, but amount of rotation is minimized.

    :param list previous: Previous XYZ rotation values as a vector
    :param list current: Current XYZ rotation values as a vector
    :param str rotation_order: String representing the rotation order (ex: "XYZ" or "ZXY")
    :return: Modified angles to minimize rotation
    :rtype: list
    """
    # Start with a pass of Naive 1D filtering
    filtered = list(current)
    for axis in range(3):
        filtered[axis] = euler_filter_1d(previous[axis], filtered[axis])

    # Then flip the whole thing and do another pass of Naive filtering
    flipped = flip_euler(filtered, rotation_order)
    for axis in range(3):
        flipped[axis] = euler_filter_1d(previous[axis], flipped[axis])

    # Return the vector with the shortest distance from the target value.
    if distance_squared(filtered, previous) > distance_squared(flipped, previous):
        return flipped
    return filtered


def match_target_rotation_1d(current, target_rotation):
    """ Applies 360 degree (2pi radians) rotations to a value to approach a target value,
    without modifying apparent orientation"""
    while abs(target_rotation - current) > pi:
        if current < target_rotation:
            current += 2 * pi
        else:
            current -= 2 * pi

    return current


# Nuke functions
class CurveList:
    def __init__(self, knob):
        values = []
        # TODO: Test behavior on multiview scripts
        for channel in range(knob.arraySize()):
            curve = knob.animation(channel)
            if curve is None:
                knob.setAnimated(channel)
                curve = knob.animation(channel)
            values.append(curve)
        self.list = values

    def __getitem__(self, idx):
        return self.list[idx]

    def __len__(self):
        return len(self.list)

    def value_at(self, frame, convert_to_rad=False):
        if len(self) == 1:
            value = self[0].evaluate(frame)
            if convert_to_rad:
                value = radians(value)
            return [value]
        else:
            values = [curve.evaluate(frame) for curve in self.list]
            if convert_to_rad:
                values = [radians(val) for val in values]
            return values

    def set_values_at(self, values, frame, convert_to_deg=False):
        if len(values) != len(self):
            raise ValueError("Number of values to set doesn't match number of curves")
        if convert_to_deg:
            values = [degrees(value) for value in values]
        for idx, value in enumerate(values):
            self.list[idx].setKey(frame, value)

    def get_all_keyframes(self):
        return sorted(list(set([key.x for curve in self.list for key in curve.keys()])))


def euler_filter(knob, use_3d=False, rotation_order=None, use_degrees=True):
    curves = CurveList(knob)
    keyframes = curves.get_all_keyframes()
    new_keys = []
    previous = None

    for frame in keyframes:
        current = curves.value_at(frame, convert_to_rad=use_degrees)
        if previous is not None:  # This filters out the first keyframe
            if use_3d:
                current = euler_filter_3d(previous, current, rotation_order)
            else:
                current = [euler_filter_1d(previous[0], current[0])]  # Value is in a list for consistency with 3d

        new_keys.append(current)
        previous = current

    for frame, value in zip(keyframes, new_keys):
        curves.set_values_at(value, frame, convert_to_deg=use_degrees)


def angular_velocity_filter(knob, use_degrees=True):
    """ Naive angular velocity filter, calculates velocity on a per axis basis """
    curves = CurveList(knob)
    keyframes = curves.get_all_keyframes()
    new_keys = []
    velocities = []
    previous = None
    previous_frame = None

    for frame in keyframes:
        current = curves.value_at(frame, convert_to_rad=use_degrees)
        if previous is not None:  # This filters out the first keyframe
            if velocities:
                for channel in range(len(current)):
                    current_value = match_target_rotation_1d(
                        current=current[channel],
                        target_rotation=previous[channel] + velocities[channel] * (frame - previous_frame))
                    current[channel] = current_value
            # Calculate new velocities
            velocities = [(c-p)/(frame-previous_frame) for p, c in zip(previous, current)]

        new_keys.append(current)
        previous = current
        previous_frame = frame

    for frame, value in zip(keyframes, new_keys):
        curves.set_values_at(value, frame, convert_to_deg=use_degrees)


def setup_filter_rotations(knob=None):
    print("AAAAAAA")

    valid_rotation_orders = ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX']
    strategies = ["Minimum Rotation (Euler Filter)", "Preserve Angular Velocity"]
    #strategies_3d = ["Minimum Rotation (Euler Filter)"]  # 3d Angular Velocity needs to be figured out (but could try Naive Angular, on a per curve basis)

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
    # Figure out if the node has a built-in rotation order
    if rotation_order is None and node.knob('rot_order'):
        value = node.knob('rot_order').value()
        if value in valid_rotation_orders:
            rotation_order = value

    # Build panel
    panel = RotationFilterPanel(
        strategies,
        rotation_orders=valid_rotation_orders if use_3d_filter and rotation_order is None else None)

    if panel.showModalDialog():
        strategy = strategies.index(panel.strategy.value())
        use_degrees = panel.unit.value() == 'Degrees'
        if strategy == 0:
            if use_3d_filter and rotation_order is None:
                rotation_order = panel.rotation_order.value()
            euler_filter(knob, use_3d_filter, rotation_order, use_degrees)
        elif strategy == 1:
            angular_velocity_filter(knob, use_degrees)


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
