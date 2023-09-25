"""
Shim to give the Tracker4 node a slightly more convenient API, closer to interacting with regular Nuke knobs.
While it can do a lot of things conveniently, it's not as fast as some direct manipulation as it may parse
values multiple times while doing some operations.

Examples:

    tracker = Tracker(nuke.toNode('Tracker1'))
    point = tracker.add_point('My Tracker')
    point['T'].setValue(True)
    for point in tracker:
        point['R'].setValue(False)

    # Point can be obtained by name or index
    first_point = tracker[0]
    mine = tracker['My Tracker']

"""
import nuke


class Tracker(object):
    _columns = None

    def __init__(self, node):
        if not node.Class() == 'Tracker4':
            raise ValueError("Tracker4 node required, got {}".format(node.Class()))
        self.node = node
        self.knob = node['tracks']
        self._parser = TCLListParser()

    def __getitem__(self, item):
        if not isinstance(item, int):
            item = self.point_names.index(item)
        n_points = len(self)
        if item >= n_points:
            return None
        if item < 0:  # Support negative indices
            item = n_points+item
            if item < 0:
                return None
        return TrackerPoint(self, item)

    def __contains__(self, item):
        if not isinstance(item, int):
            try:
                item = self.point_names.index(item)
            except ValueError:
                return False
        if item >= len(self):
            return False
        if item < 0:
            return False
        return True

    def __len__(self):
        return len(self.get_internals()[2])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @staticmethod
    def _get_point_names(internals):
        return [p[1] for p in internals[2]]

    def get_internals(self):
        tcl_list = self.knob.toScript()
        py = self._parser.parse(tcl_list)
        return py

    @property
    def point_names(self):
        return self._get_point_names(self.get_internals())

    @property
    def columns(self):
        if self._columns:
            return self._columns
        columns = [k[3] for k in self.get_internals()[1]]
        self.__class__._columns = columns
        TrackerPoint.col_count = len(columns)
        return columns

    def add_point(self, name='track', ref_frame=None, translate=False, rotate=False, scale=False):
        parsed = self.get_internals()

        n = 1
        name_candidate = name
        existing_names = self._get_point_names(parsed)
        while name_candidate in existing_names:
            name_candidate = '{} {}'.format(name, n)
            n += 1

        # TODO: Nuke actually calculates the default track size and search size based on the image res. Do the same.
        track = 22
        search = 32

        # Gross, but we have to inject the point in TCL otherwise it may not work if the control panel is closed
        f = 'x{}'.format(int(nuke.frame()) if ref_frame is None else int(ref_frame))
        blank_row = [['curve', 'K', f, '1']]
        blank_row += [name_candidate]
        blank_row += [['curve', f, '0']] * 2
        blank_row += [['curve', 'K', f, '0']] * 2
        blank_row += [int(translate), int(rotate), int(scale), ['curve', f, '0']]
        blank_row += ['1', '0', -search, -search, search, search, -track, -track, track, track]
        blank_row += [[]] * 11

        parsed[0][2] = int(parsed[0][2])+1
        parsed[2].append(blank_row)
        if self.knob.fromScript(self._parser.encode(parsed)):
            return self[-1]
        else:
            return None

    def delete_point(self, item):
        if item not in self:
            return False
        idx = self[item].index
        parsed = self.get_internals()
        parsed[2].pop(idx)
        parsed[0][2] = int(parsed[0][2]) - 1

        return self.knob.fromScript(self._parser.encode(parsed))

    def rename_point(self, item, name):
        if item not in self:
            return False
        idx = self[item].index
        parsed = self.get_internals()
        parsed[2][idx][1] = name
        return self.knob.fromScript(self._parser.encode(parsed))


class TrackerPoint(object):
    col_count = 31

    def __init__(self, parent, index):
        self.parent = parent
        self.index = index

    def __getitem__(self, item):
        if not isinstance(item, int):
            item = self.parent.columns.index(item)
        return TrackerPointKnob(self, item)


class TrackerPointKnob(object):
    def __init__(self, parent, index):
        self.parent = parent
        self.index = index
        self.real_index = self.parent.index * self.parent.col_count + self.index

    def clearAnimated(self):
        return self.parent.parent.knob.clearAnimated(self.real_index)

    def getValue(self):
        return self.value()

    def getValueAt(self, t):
        # Special case for name
        if self.index == 1:
            return self.parent.parent.get_internals()[2][self.parent.index][self.index]
        else:
            return self.parent.parent.knob.getValueAt(t, self.real_index)

    def hasExpression(self):
        # Seems to always return False? Even if explicitly using setExpression
        return self.parent.parent.knob.hasExpression(self.real_index)

    def isAnimated(self):
        return self.parent.parent.knob.isAnimated(self.real_index)

    def removeKey(self):
        return self.parent.parent.knob.removeKey(self.real_index)

    def removeKeyAt(self, t):
        return self.parent.parent.knob.removeKeyAt(t, self.real_index)

    def setAnimated(self):
        return self.parent.parent.knob.setAnimated(self.real_index)

    def setExpression(self, expression):
        return self.parent.parent.knob.setExpression(expression, self.real_index)

    def setValue(self, value):
        # Special case for name
        if self.index == 1:
            return self.parent.parent.rename_point(self.parent.index, value)
        else:
            return self.parent.parent.knob.setValue(value, self.real_index)

    def setValueAt(self, value, t):
        return self.parent.parent.knob.setValueAt(value, t, self.real_index)

    def value(self):
        # Special case for name
        if self.index == 1:
            return self.parent.parent.get_internals()[2][self.parent.index][self.index]
        else:
            return self.parent.parent.knob.value(self.real_index)


class TCLListParser(object):

    NO_ESCAPE = 0
    SINGLE_ESCAPE = 1
    STRING_ESCAPE = 2
    WHITESPACE = [" ", "\t", "\r", "\n"]

    def __init__(self):
        self._out = None
        self._buffer = None
        self._stack = None

    def _flush(self):
        if self._buffer is not None:
            self._stack[-1].append(self._buffer)
        self._buffer = None

    def _add_char(self, char):
        if self._buffer is None:
            self._buffer = char
        else:
            self._buffer += char

    def parse(self, tcl_list):
        self._out = []
        self._stack = [self._out]
        self._buffer = None

        escape = self.NO_ESCAPE

        for char in tcl_list:
            # Single escapes
            if escape & self.SINGLE_ESCAPE:
                self._add_char(char)
                escape &= ~self.SINGLE_ESCAPE
            elif char == '\\':
                escape |= self.SINGLE_ESCAPE
            # Strings with spaces, like "hello world"
            elif char == '"':
                escape ^= self.STRING_ESCAPE
            else:
                if escape & self.STRING_ESCAPE:
                    self._add_char(char)
                elif char in self.WHITESPACE:
                    self._flush()
                elif char == "{":
                    _ = []
                    self._stack[-1].append(_)
                    self._stack.append(_)
                elif char == "}":
                    self._flush()
                    self._stack.pop()
                else:
                    self._add_char(char)
        return self._out

    def encode(self, python_list):
        """ Brute force dumb re-encoding from the output of parse """
        def _encode(item):
            str_buf = ''
            if isinstance(item, list):
                str_buf += '{'
                str_buf += ' '.join([_encode(sub_item) for sub_item in item])
                str_buf += '}'
            else:
                sub_item = str(item)
                if any((c in sub_item for c in self.WHITESPACE)):
                    str_buf += '"' + sub_item.replace('"', '\\"') + '"'
                else:
                    str_buf += sub_item.replace('{', '\\{').replace('"', '\\"').replace('\\', '\\\\')

                pass
            return str_buf[:]

        encoded = _encode(python_list)[1:-1]
        return encoded
