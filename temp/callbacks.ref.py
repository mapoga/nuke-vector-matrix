""" knobChanged """

# Set knobChanged:
nuke.selectedNode()['knobChanged'].setValue("""# Knobchanged code:
node = nuke.thisNode()
knob = nuke.thisKnob()
print("Node: {}, Knob: {}".format(node.name(), knob.name()))
""")


# Other:
def get_input_by_number(number):
    """ Returns the input node or None if not found """
    inputs = nuke.allNodes('Input')
    for i in inputs:
        if int(i.knob('number').value()) == int(number):
            return i
    return None


def connect_input(input, node, input_number):
    """ Connect input on top of node and reposition input """
    # connect
    node.connectInput(input_number, input)
    # position
    con_xpos = node.xpos()
    con_ypos = node.ypos()
    con_width = node.screenWidth()
    con_height = node.screenHeight()
    width = input.screenWidth()
    height = input.screenHeight()
    input.setXYpos(int(con_xpos+(con_width/2.0)-(width/2.0)),
                   int(con_ypos-height-10))


def pulldown_idx(knob):
    """ returns the index of current value """
    values = knob.values()
    val = knob.value()
    return values.index(val)


def add_input(name, connect_name, connect_pipe_nbr):
    """ Creates an input node and a connection """
    input_node = nuke.nodes.Input()
    input_node.knob('name').setValue(name)
    input_connect_node = nuke.toNode(connect_name)
    connect_input(input_node, input_connect_node, connect_pipe_nbr)
    return input_node


n = nuke.thisNode()
k = nuke.thisKnob()
with n:
    checkbox_name = 'custom_format'
    mode_pulldown_name = 'mode'
    input_name = 'Input'
    input_number = 0
    input_connect_node_name = 'InputAConnect'
    is_custom_format = n.knob(checkbox_name).value()
    input_node = get_input_by_number(input_number)

    if k.name() == checkbox_name:
        # Custom format change
        if is_custom_format:
            n.knob('format').setEnabled(True)
            if input_node:
                nuke.delete(input_node)
        else:
            n.knob('format').setEnabled(False)
            if not input_node:
                input_node = add_input(input_name, input_connect_node_name, 0)
    elif k.name() == mode_pulldown_name:
        # Mode change
        modes = ['image', 'knob']
        visiblity = [['group_image', ],
                     ['group_knob', ]]
        for idx, value in enumerate(modes):
            if value == k.value():
                for v in visiblity[idx]:
                    n.knob(v).setVisible(True)
            else:
                for v in visiblity[idx]:
                    n.knob(v).setVisible(False)
        # Update input node
        if k.value() == 'knob':
            if input_node:
                nuke.delete(input_node)
        else:
            if not input_node and not is_custom_format:
                input_node = add_input(input_name, input_connect_node_name, 0)


""" onCreate """
n = nuke.thisNode()
with n:  # enables group scope
    READ_ONLY = 0x10000000
    n.knob('output_vector').setFlag(READ_ONLY)
