import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import viznet
from viznet import EdgeBrush, NodeBrush


def _show(fig_size=None, show=True):
    plt.axis('off')
    plt.axis('equal')
    if fig_size:
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(*fig_size)
    if show:
        plt.show()

def draw_feed_forward(ax, num_node_list, size=2):
    '''
    draw a feed forward neural network.

    Args:
        num_node_list (list<int>): number of nodes in each layer.
    '''
    num_hidden_layer = len(num_node_list) - 2
    token_list = ['\sigma^z'] + \
        ['y^{(%s)}' % (i + 1) for i in range(num_hidden_layer)] + ['\psi']
    kind_list = ['nn.input'] + ['nn.hidden'] * num_hidden_layer + ['nn.output']
    radius_list = [0.3] + [0.2] * num_hidden_layer + [0.3]
    y_list = 1.5 * np.arange(len(num_node_list))

    text_list = ['$\phi$'] * 3

    seq_list = []
    for n, kind, radius, y, text in zip(num_node_list, kind_list, radius_list, y_list, text_list):
        b = NodeBrush(kind)
        _nodes = viznet.node_sequence(b, n, center=(0, y))
        for _n in _nodes:
            _n.text(text, 'center', fontsize=18)
        # print(_nodes)
        seq_list.append(_nodes)


    eb = EdgeBrush('<--')
    for st, et in zip(seq_list[:-1], seq_list[1:]):
        viznet.connecta2a(st, et, eb)

    # viznet.node_sequence(NodeBrush('box', ax, size=size), 1, (0,4))
    #
    node = NodeBrush('box', size=size)
    node.size = (0.5, 0.3)
    viznet.node_sequence(node, 1, (0,))


def real_bp():
#     with viznet.DynamicShow((6, 6), '_feed_forward.png') as d:
    draw_feed_forward(plt.gca(), num_node_list=[7, 7, 1])
    _show()

real_bp()



#%%
size=.5
step = 1.5

font_size = 18
font_size_small = 12

node_spacing = (2.5*size, 0)
hidden_layer_node_num = 7

def box_with_connectable(text, box_size, box_loc, box_colour='lime'):
    # the box
    nb = NodeBrush('box', size=size)
    nb.size = box_size
    nb.color = box_colour
    nodes3 = viznet.node_sequence(nb, 1, box_loc, space=node_spacing)
    nodes3[0].text(text, 'center', fontsize=font_size_small)
    # the connectable-invisible nodes
    nb = NodeBrush('invisible', size=box_size[1])  # box_size[1] is the height of box
    # nb = NodeBrush([
    #         None,
    #         "rectangle",
    #         "none"
    #     ], size=box_size[1])  # box_size[1] is the height of box
    invis = viznet.node_sequence(nb, hidden_layer_node_num, box_loc, space=node_spacing)

    return nodes3, invis



eb = EdgeBrush('-->>', lw=size)


# input
nb = NodeBrush('nn.input', size=size)
nodes1 = viznet.node_sequence(nb, 1, (0,-0.2), space=node_spacing)
nodes1[0].text("$\phi$", 'center', fontsize=font_size)

def pin_top(iterable):
    return [_.pin("top") for _ in iterable]

def pin_bot(iterable):
    return [_.pin("bottom") for _ in iterable]

def block_1(inputs, text1, text2, use_121=False, offset=0, box_colour='grey'):
    # hidden layers 1
    nb = NodeBrush('nn.hidden', size=size*.5)
    _inner_block_1 = viznet.node_sequence(nb, hidden_layer_node_num, (0, offset + -1*step), space=node_spacing)
    for i, n in enumerate(_inner_block_1):
        if i > len(_inner_block_1) // 2:
            i = 500 - (len(_inner_block_1) - i)
        _t = f"$h_{{{i+1}}}$"
        if i == len(_inner_block_1) // 2 :
            _t = "$h_{...}$"
        n.text(_t, 'center', fontsize=font_size_small)
    if use_121:
        viznet.connect121(pin_bot(inputs), _inner_block_1, eb)
    else:
        viznet.connecta2a(pin_bot(inputs), pin_top(_inner_block_1), eb)

    # hidden layers 1 Act
    _inner_block_2, _inner_block_2_connectable = box_with_connectable(text=text1,
                                                 box_loc=(0, offset + -1.4*step),
                                                 box_size=(size * 9, size/2.8)
                                                 )
    viznet.connect121(_inner_block_1, _inner_block_2_connectable, eb)

    # hidden layers 1 BN
    _inner_block_3, _inner_block_3_connectable = box_with_connectable(text=text2,
                                                 box_loc=(0, offset + -1.7*step),
                                                 box_size=(size * 9, size/2.8),
                                                 box_colour=box_colour
                                                 )
    viznet.connect121(_inner_block_2_connectable, _inner_block_3_connectable, eb)

    return _inner_block_1, _inner_block_2, _inner_block_2_connectable, _inner_block_3, _inner_block_3_connectable


block_step = 1.15

block1s = block_1(inputs=nodes1,
                  text1="ReLu",
                  text2="Batch Normalisation",
                  box_colour='blue',
                  )
block2s = block_1(inputs=block1s[-1],
                  text1="ReLu",
                  text2="Dropout (0.25 rate)",
                  offset=-block_step*1*step, use_121=False)
block3s = block_1(inputs=block2s[-1],
                  text1="ReLu",
                  text2="Dropout (0.25 rate)",
                  offset=-block_step*2*step, use_121=False)
block4s = block_1(inputs=block3s[-1],
                  text1="ReLu",
                  text2="Dropout (0.25 rate)",
                  offset=-block_step*3*step, use_121=False)
block5s = block_1(inputs=block4s[-1],
                  text1="ReLu",
                  text2="Dropout (0.25 rate)",
                  offset=-block_step*4*step, use_121=False)




# # box
# nb = NodeBrush('box', size=size)
# nb.size = (size * 9, size/2.5)
# nb.color = "lime"
# nodes = viznet.node_sequence(nb, 1, (0,-5*step), space=(2.5*size, 0))
# nodes[0].text("ReLu", fontsize=font_size_small)




# # nodes
# nb = NodeBrush('box', size=size)
# nb.size = (size * 5, size/3)
# nb.color = "lime"
# nodes = viznet.node_sequence(nb, 1, (0,-10*step), space=(2.5*size, 0))
# nodes[0].text("ReLu", fontsize=font_size_small)




_show(fig_size=(18.5, 10.5))


#%%
size=.5
step = 1.5

font_size = 18
font_size_small = 12

node_spacing = (2.*size, 0)
hidden_layer_node_num = 7



def box_with_connectable(text, box_size, box_loc, node_spacing, hidden_layer_node_num, box_colour='#55CC77'):
    # the box
    nb = NodeBrush('box', size=size)
    nb.size = box_size
    nb.color = box_colour
    nodes3 = viznet.node_sequence(nb, 1, box_loc, space=node_spacing)
    nodes3[0].text(text, 'right', fontsize=font_size_small)
    # the connectable-invisible nodes
    nb = NodeBrush('invisible', size=box_size[1])  # box_size[1] is the height of box
    # nb = NodeBrush([
    #         None,
    #         "rectangle",
    #         "none"
    #     ], size=box_size[1])  # box_size[1] is the height of box
    invis = viznet.node_sequence(nb, hidden_layer_node_num, box_loc, space=node_spacing)

    return nodes3, invis



eb = EdgeBrush('-->>', lw=size)

#
# dir(nodes1[0].brush)
# input
nb = NodeBrush('nn.input', size=size*.6)
nodes1 = viznet.node_sequence(nb, 1, (0,-0.7), space=node_spacing)
nodes1[0].text("$\phi$", 'center', fontsize=font_size)



def custom_node_seq(num_node, center, space=(1,0)):
    import numpy as np
    x_list = np.arange(-num_node / 2. + .5, num_node / 2., 1)
    xylist = center + np.asarray(space) * x_list[:, None]

    return list(zip(xylist[:, 0], xylist[:, 1]))




def pin_top(iterable):
    return [_.pin("top") for _ in iterable]

def pin_bot(iterable):
    return [_.pin("bottom") for _ in iterable]

def block_1(inputs, text1, text2, use_121=False, offset=0, box_colour='grey'):
    # hidden layers 1
    nb = NodeBrush(['lime','circle','none'], size=size*.5)
    # _inner_block_1 = viznet.node_sequence(nb, hidden_layer_node_num, (0, offset + -1*step), space=node_spacing)
    xys = custom_node_seq(hidden_layer_node_num, (0, offset + -1*step), space=node_spacing)
    _inner_block_1 = []
    for i, xy in enumerate(xys):
        if i == len(xys) // 2 :
            __nb = NodeBrush('invisible', size=size*.5)
            _inner_block_1.append(__nb >> xy)
            _inner_block_1[-1].text("$\cdots$", 'center', fontsize=font_size_small)
        else:
            if i > len(xys) // 2:
                i = 500 - (len(xys) - i)
            _inner_block_1.append(nb >> xy)
            _inner_block_1[-1].text(f"$h_{{{i+1}}}$", 'center', fontsize=font_size_small)
    if use_121:
        viznet.connect121(pin_bot(inputs), _inner_block_1, eb)
    else:
        viznet.connecta2a(pin_bot(inputs), pin_top(_inner_block_1), eb)

    # hidden layers 1 Act
    _inner_block_2, _inner_block_2_connectable = box_with_connectable(text=text1,
                                              node_spacing=node_spacing,
                                              hidden_layer_node_num=hidden_layer_node_num,
                                                 box_loc=(0, offset + -1.25*step),
                                                 box_size=(size * 6.6, size/7.3),
                                                 )
    viznet.connect121(_inner_block_1, _inner_block_2_connectable, eb)

    # hidden layers 1 BN
    _inner_block_3, _inner_block_3_connectable = box_with_connectable(text=text2,
                                              node_spacing=node_spacing,
                                              hidden_layer_node_num=hidden_layer_node_num,
                                                 box_loc=(0, offset + -1.38*step),
                                                 box_size=(size * 6.6, size/7.3),
                                                 box_colour=box_colour
                                                 )
    viznet.connect121(_inner_block_2_connectable, _inner_block_3_connectable, eb)

    return _inner_block_1, _inner_block_2, _inner_block_2_connectable, _inner_block_3, _inner_block_3_connectable

def block_2(inputs, text1, output_text, hidden_layer_text="$h_{2M \\times Q}$", no_box=False, use_121=False, offset=0, x_offset=0, box_colour='grey'):
    hidden_layer_node_num = 3
    node_spacing = (1.2*size, 0)


    # hidden layers 1
    nb = NodeBrush(['lime','circle','none'], size=size*.5)
    xys = custom_node_seq(hidden_layer_node_num, (x_offset, offset + -1*step), space=node_spacing)
    __nb = NodeBrush('invisible', size=size*.5)
    _inner_block_1 = [
        nb >> xys[0],
        __nb >> xys[1],
        nb >> xys[2],
    ]
    _inner_block_1[0].text("$h_{1}$", 'center', fontsize=font_size_small)
    _inner_block_1[1].text("$\cdots$", 'center', fontsize=font_size_small)
    _inner_block_1[1].brush.style = "invisible"

    _inner_block_1[2].text(hidden_layer_text, 'center', fontsize=font_size_small)
    if use_121:
        viznet.connect121(pin_bot(inputs), _inner_block_1, eb)
    else:
        viznet.connecta2a(pin_bot(inputs), pin_top(_inner_block_1), eb)

    if not no_box:
        # hidden layers 1 Act
        _inner_block_2, _inner_block_2_connectable = box_with_connectable(text=text1,
                                                  node_spacing=node_spacing,
                                                  hidden_layer_node_num=hidden_layer_node_num,
                                                     box_loc=(x_offset, offset + -1.25*step),
                                                     box_size=(size * 1.8, size/7.3),
                                                     box_colour=box_colour
                                                     )
        viznet.connect121(_inner_block_1, _inner_block_2_connectable, eb)

    # output
    nb = NodeBrush('nn.output', size=size*.5)
    output = viznet.node_sequence(nb, 1, (x_offset, offset + -1.6*step), space=node_spacing)
    output[0].text(output_text, 'center', fontsize=font_size)

    if not no_box:
        viznet.connecta2a(pin_bot(_inner_block_2_connectable), pin_top(output), eb)
        return _inner_block_1, _inner_block_2, _inner_block_2_connectable, output
    else:
        viznet.connecta2a(pin_bot(_inner_block_1), pin_top(output), eb)
        return _inner_block_1, output



block_step = 1.15
block_step = 1.0
block_step = .8

block1s = block_1(inputs=nodes1,
                  text1="ReLu",
                  text2="Batch Normalisation",
                  box_colour='blue',
                  )
block2s = block_1(inputs=block1s[-1],
                  text1="ReLu",
                  text2="Dropout (0.25 rate)",
                  offset=-block_step*1*step, use_121=False)
block3s = block_1(inputs=block2s[-1],
                  text1="ReLu",
                  text2="Dropout (0.25 rate)",
                  offset=-block_step*2*step, use_121=False)
block4s = block_1(inputs=block3s[-1],
                  text1="ReLu",
                  text2="Dropout (0.25 rate)",
                  offset=-block_step*3*step, use_121=False)
block5s = block_1(inputs=block4s[-1],
                  text1="ReLu",
                  text2="Dropout (0.25 rate)",
                  offset=-block_step*4*step, use_121=False)



block_sigma = block_2(inputs=block5s[-1],
                  text1="Exponential",
                  output_text="$\\sigma_{q,m}$",
                  box_colour='red',
                  offset=-block_step*5*step, use_121=False)

block_alpha = block_2(inputs=block5s[-1],
                  text1="Softmax",
                  output_text="$\\alpha_{q,m}$",
                  box_colour='orange',
                  x_offset=6*size,
                  hidden_layer_text="$h_{ Q}$",
                  offset=-block_step*5*step, use_121=False)

block_mu = block_2(inputs=block5s[-1],
                  text1="Softmax",
                  output_text="$\\alpha_{q,m}$",
                  no_box=True,
                  box_colour='orange',
                  x_offset=-6*size,
                  offset=-block_step*5*step, use_121=False)




# # box
# nb = NodeBrush('box', size=size)
# nb.size = (size * 9, size/2.5)
# nb.color = "lime"
# nodes = viznet.node_sequence(nb, 1, (0,-5*step), space=(2.5*size, 0))
# nodes[0].text("ReLu", fontsize=font_size_small)




# # nodes
# nb = NodeBrush('box', size=size)
# nb.size = (size * 5, size/3)
# nb.color = "lime"
# nodes = viznet.node_sequence(nb, 1, (0,-10*step), space=(2.5*size, 0))
# nodes[0].text("ReLu", fontsize=font_size_small)


# import matplotlib
# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


_show(fig_size=(18.5, 16.5), show=False)
plt.savefig("out.pdf")
#%%
size=.2
step = 1.5

font_size = 18
font_size_small = 12

font_size = 20
font_size_small = 22

node_spacing = (.6*size, 0)
hidden_layer_node_num = 15



def box_with_connectable(text, box_size, box_loc, node_spacing, hidden_layer_node_num, box_colour='lime', text_pos='right', text_kwargs={}):
# def box_with_connectable(text, box_size, box_loc, node_spacing, hidden_layer_node_num, box_colour='#55CC77'):
    # the box
    nb = NodeBrush('box', size=size)
    nb.size = box_size
    nb.color = box_colour
    nodes3 = viznet.node_sequence(nb, 1, box_loc, space=node_spacing)
    text_colour = box_colour
    if box_colour == 'lime':
        text_colour = 'green'
    if box_colour == 'grey':
        text_colour = 'black'
    nodes3[0].text(text, text_pos, fontsize=font_size_small, color=text_colour)
    # the connectable-invisible nodes
    nb = NodeBrush('invisible', size=box_size[1])  # box_size[1] is the height of box
    # nb = NodeBrush([
    #         None,
    #         "rectangle",
    #         "none"
    #     ], size=box_size[1])  # box_size[1] is the height of box
    invis = viznet.node_sequence(nb, hidden_layer_node_num, box_loc, space=node_spacing)

    return nodes3, invis



eb = EdgeBrush('-->>', lw=size*.4, color='grey')

#
# dir(nodes1[0].brush)
# input
nb = NodeBrush('nn.input', size=size*.4)
nodes1 = viznet.node_sequence(nb, 1, (0,-1.3), space=node_spacing)
nodes1[0].text("$\phi$", 'center', fontsize=font_size*2)



def custom_node_seq(num_node, center, space=(1,0)):
    import numpy as np
    x_list = np.arange(-num_node / 2. + .5, num_node / 2., 1)
    xylist = center + np.asarray(space) * x_list[:, None]

    return list(zip(xylist[:, 0], xylist[:, 1]))




def pin_top(iterable):
    return [_.pin("top") for _ in iterable]

def pin_bot(iterable):
    return [_.pin("bottom") for _ in iterable]

def block_1(inputs, text1, text2, use_121=False, offset=0, box_colour='grey'):
    # hidden layers 1
    nb = NodeBrush(['#55CC77','circle','none'], size=size*.1)
    # _inner_block_1 = viznet.node_sequence(nb, hidden_layer_node_num, (0, offset + -1*step), space=node_spacing)
    xys = custom_node_seq(hidden_layer_node_num, (0, offset + -1*step), space=node_spacing)
    _inner_block_1 = []
    for i, xy in enumerate(xys):
        if i == len(xys) // 2:
            __nb = NodeBrush('invisible', size=size*.1)
            _inner_block_1.append(__nb >> xy)
            _inner_block_1[-1].text("$\cdots\cdots$", 'center', fontsize=font_size_small)
        else:
            _inner_block_1.append(nb >> xy)
            if i == len(xys) - 1:
                _inner_block_1[-1].text(f"$h_{1} \sim h_{{500}}$", 'right', fontsize=font_size_small)

    if use_121:
        viznet.connect121(pin_bot(inputs), _inner_block_1, eb)
    else:
        viznet.connecta2a(pin_bot(inputs), pin_top(_inner_block_1), eb)

    # hidden layers 1 Act
    _inner_block_2, _inner_block_2_connectable = box_with_connectable(text=text1,
                                              node_spacing=node_spacing,
                                              hidden_layer_node_num=hidden_layer_node_num,
                                              text_pos='left',
                                                 box_loc=(0, offset + -1.04*step),
                                                 box_size=(size * 4.5, size*.05),
                                                 # text_kwargs={'color' : 'red'}
                                                 )
    viznet.connect121(_inner_block_1, _inner_block_2_connectable, eb)

    # hidden layers 1 BN
    _inner_block_3, _inner_block_3_connectable = box_with_connectable(text=text2,
                                              node_spacing=node_spacing,
                                              hidden_layer_node_num=hidden_layer_node_num,
                                                 box_loc=(0, offset + -1.06*step),
                                                 box_size=(size * 4.5, size*.05),
                                                 box_colour=box_colour
                                                 )
    viznet.connect121(_inner_block_2_connectable, _inner_block_3_connectable, eb)

    return _inner_block_1, _inner_block_2, _inner_block_2_connectable, _inner_block_3, _inner_block_3_connectable

def block_2(inputs, text1, output_text, hidden_layer_text="$h_1 \sim h_{2M \\times Q}$", no_box=False, use_121=False, offset=0, x_offset=0, box_colour='grey'):
    hidden_layer_node_num = 3
    node_spacing = (.65*size, 0)


    # hidden layers 1
    nb = NodeBrush(['#55CC77','circle','none'], size=size*.1)
    xys = custom_node_seq(hidden_layer_node_num, (x_offset, offset + -1*step), space=node_spacing)
    __nb = NodeBrush('invisible', size=size*.1)
    _inner_block_1 = [
        nb >> xys[0],
        __nb >> xys[1],
        nb >> xys[2],
    ]
    _inner_block_1[1].text("$\cdots\cdots$", 'center', fontsize=font_size_small)
    _inner_block_1[1].brush.style = "invisible"

    _inner_block_1[2].text(hidden_layer_text, 'right', fontsize=font_size_small)
    if use_121:
        viznet.connect121(pin_bot(inputs), _inner_block_1, eb)
    else:
        viznet.connecta2a(pin_bot(inputs), pin_top(_inner_block_1), eb)

    if not no_box:
        # hidden layers 1 Act
        _inner_block_2, _inner_block_2_connectable = box_with_connectable(text=text1,
                                                  node_spacing=node_spacing,
                                                  hidden_layer_node_num=hidden_layer_node_num,
                                                     box_loc=(x_offset, offset + -1.04*step),
                                                     box_size=(size * .9, size*.05),
                                                     box_colour=box_colour
                                                     )
        viznet.connect121(_inner_block_1, _inner_block_2_connectable, eb)

    # output
    nb = NodeBrush('nn.output', size=size*.3)
    output = viznet.node_sequence(nb, 1, (x_offset, offset + -1.1*step), space=node_spacing)
    output[0].text(output_text, 'right', fontsize=font_size*2)

    if not no_box:
        viznet.connecta2a(pin_bot(_inner_block_2_connectable), pin_top(output), eb)
        return _inner_block_1, _inner_block_2, _inner_block_2_connectable, output
    else:
        viznet.connecta2a(pin_bot(_inner_block_1), pin_top(output), eb)
        return _inner_block_1, output



block_step = 1.15
block_step = 1.0
block_step = .8
block_step = .3
block_step = .2

block1s = block_1(inputs=nodes1,
                  text1="ReLu",
                  text2="Batch Norm",
                  box_colour='blue',
                  )
block2s = block_1(inputs=block1s[-1],
                  text1="ReLu",
                  text2="Dropout (0.25)",
                  offset=-block_step*1*step, use_121=False)
block3s = block_1(inputs=block2s[-1],
                  text1="ReLu",
                  text2="Dropout (0.25)",
                  offset=-block_step*2*step, use_121=False)
block4s = block_1(inputs=block3s[-1],
                  text1="ReLu",
                  text2="Dropout (0.25)",
                  offset=-block_step*3*step, use_121=False)
block5s = block_1(inputs=block4s[-1],
                  text1="ReLu",
                  text2="Dropout (0.25)",
                  offset=-block_step*4*step, use_121=False)



block_sigma = block_2(inputs=block5s[-1],
                  text1="Exponential",
                  output_text="$\\sigma_{q,m}$",
                  box_colour='red',
                  offset=-block_step*5*step, use_121=False)

block_alpha = block_2(inputs=block5s[-1],
                  text1="Softmax",
                  output_text="$\\alpha_{q,m}$",
                  box_colour='orange',
                  x_offset=4.5*size,
                  hidden_layer_text="$h_1 \sim h_{ Q}$",
                  offset=-block_step*5*step, use_121=False)

block_mu = block_2(inputs=block5s[-1],
                  text1="Softmax",
                  output_text="$\\mu_{q,m}$",
                  no_box=True,
                  box_colour='orange',
                  x_offset=-4.5*size,
                  offset=-block_step*5*step, use_121=False)




# # box
# nb = NodeBrush('box', size=size)
# nb.size = (size * 9, size/2.5)
# nb.color = "lime"
# nodes = viznet.node_sequence(nb, 1, (0,-5*step), space=(2.5*size, 0))
# nodes[0].text("ReLu", fontsize=font_size_small)




# # nodes
# nb = NodeBrush('box', size=size)
# nb.size = (size * 5, size/3)
# nb.color = "lime"
# nodes = viznet.node_sequence(nb, 1, (0,-10*step), space=(2.5*size, 0))
# nodes[0].text("ReLu", fontsize=font_size_small)


# import matplotlib
# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
#
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# rc('text.latex', preamble=r'\usepackage{cmbright}')

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.family'] = 'cmu serif'

_show(fig_size=(18.5, 16.5), show=False)
plt.show()
plt.savefig("out.pdf")
#%%
size=.2
step = 1.5

font_size = 18
font_size_small = 12

font_size = 20
font_size_small = 22

node_spacing = (.6*size, 0)
hidden_layer_node_num = 15



def box_with_connectable(text, box_size, box_loc, node_spacing, hidden_layer_node_num, box_colour='lime', text_pos='right', text_kwargs={}):
# def box_with_connectable(text, box_size, box_loc, node_spacing, hidden_layer_node_num, box_colour='#55CC77'):
    # the box
    box_colour = 'grey'
    nb = NodeBrush('box', size=size)
    nb.size = box_size
    nb.color = box_colour
    nodes3 = viznet.node_sequence(nb, 1, box_loc, space=node_spacing)
    text_colour = box_colour
    if box_colour == 'lime':
        text_colour = 'green'
    if box_colour == 'grey':
        text_colour = 'black'
    text_colour = 'black'
    nodes3[0].text(text, text_pos, fontsize=font_size_small, color=text_colour)
    # the connectable-invisible nodes
    nb = NodeBrush('invisible', size=box_size[1])  # box_size[1] is the height of box
    # nb = NodeBrush([
    #         None,
    #         "rectangle",
    #         "none"
    #     ], size=box_size[1])  # box_size[1] is the height of box
    invis = viznet.node_sequence(nb, hidden_layer_node_num, box_loc, space=node_spacing)

    return nodes3, invis



eb = EdgeBrush('-->>', lw=size*.4, color='grey')

#
# dir(nodes1[0].brush)
# input
nb = NodeBrush(['grey', 'circle', 'none'], size=size*.4)
nodes1 = viznet.node_sequence(nb, 1, (0,-1.3), space=node_spacing)
nodes1[0].text("$\phi$", 'right', fontsize=font_size*2)



def custom_node_seq(num_node, center, space=(1,0)):
    import numpy as np
    x_list = np.arange(-num_node / 2. + .5, num_node / 2., 1)
    xylist = center + np.asarray(space) * x_list[:, None]

    return list(zip(xylist[:, 0], xylist[:, 1]))




def pin_top(iterable):
    return [_.pin("top") for _ in iterable]

def pin_bot(iterable):
    return [_.pin("bottom") for _ in iterable]

def block_1(inputs, text1, text2, use_121=False, offset=0, box_colour='grey'):
    # hidden layers 1
    nb = NodeBrush(['grey','circle','none'], size=size*.1)
    # _inner_block_1 = viznet.node_sequence(nb, hidden_layer_node_num, (0, offset + -1*step), space=node_spacing)
    xys = custom_node_seq(hidden_layer_node_num, (0, offset + -1*step), space=node_spacing)
    _inner_block_1 = []
    for i, xy in enumerate(xys):
        if i == len(xys) // 2:
            __nb = NodeBrush('invisible', size=size*.1)
            _inner_block_1.append(__nb >> xy)
            _inner_block_1[-1].text("$\cdots\cdots$", 'center', fontsize=font_size_small)
        else:
            _inner_block_1.append(nb >> xy)
            if i == len(xys) - 1:
                _inner_block_1[-1].text(f"$h_{1} \sim h_{{500}}$", 'right', fontsize=font_size_small)

    if use_121:
        viznet.connect121(pin_bot(inputs), _inner_block_1, eb)
    else:
        viznet.connecta2a(pin_bot(inputs), pin_top(_inner_block_1), eb)

    # hidden layers 1 Act
    _inner_block_2, _inner_block_2_connectable = box_with_connectable(text=text1,
                                              node_spacing=node_spacing,
                                              hidden_layer_node_num=hidden_layer_node_num,
                                              text_pos='left',
                                                 box_loc=(0, offset + -1.04*step),
                                                 box_size=(size * 4.5, size*.05),
                                                 # text_kwargs={'color' : 'red'}
                                                 )
    viznet.connect121(_inner_block_1, _inner_block_2_connectable, eb)

    # hidden layers 1 BN
    _inner_block_3, _inner_block_3_connectable = box_with_connectable(text=text2,
                                              node_spacing=node_spacing,
                                              hidden_layer_node_num=hidden_layer_node_num,
                                                 box_loc=(0, offset + -1.06*step),
                                                 box_size=(size * 4.5, size*.05),
                                                 box_colour=box_colour
                                                 )
    viznet.connect121(_inner_block_2_connectable, _inner_block_3_connectable, eb)

    return _inner_block_1, _inner_block_2, _inner_block_2_connectable, _inner_block_3, _inner_block_3_connectable

def block_2(inputs, text1, output_text, hidden_layer_text="$h_1 \sim h_{2M \\times Q}$", no_box=False, use_121=False, offset=0, x_offset=0, box_colour='grey'):
    hidden_layer_node_num = 3
    node_spacing = (.65*size, 0)


    # hidden layers 1
    nb = NodeBrush(['grey','circle','none'], size=size*.1)
    xys = custom_node_seq(hidden_layer_node_num, (x_offset, offset + -1*step), space=node_spacing)
    __nb = NodeBrush('invisible', size=size*.1)
    _inner_block_1 = [
        nb >> xys[0],
        __nb >> xys[1],
        nb >> xys[2],
    ]
    _inner_block_1[1].text("$\cdots\cdots$", 'center', fontsize=font_size_small)
    _inner_block_1[1].brush.style = "invisible"

    _inner_block_1[2].text(hidden_layer_text, 'right', fontsize=font_size_small)
    if use_121:
        viznet.connect121(pin_bot(inputs), _inner_block_1, eb)
    else:
        viznet.connecta2a(pin_bot(inputs), pin_top(_inner_block_1), eb)

    if not no_box:
        # hidden layers 1 Act
        _inner_block_2, _inner_block_2_connectable = box_with_connectable(text=text1,
                                                  node_spacing=node_spacing,
                                                  hidden_layer_node_num=hidden_layer_node_num,
                                                     box_loc=(x_offset, offset + -1.04*step),
                                                     box_size=(size * .9, size*.05),
                                                     box_colour=box_colour
                                                     )
        viznet.connect121(_inner_block_1, _inner_block_2_connectable, eb)

    # output
    nb = NodeBrush(['grey', 'circle', 'none'], size=size*.3)
    output = viznet.node_sequence(nb, 1, (x_offset, offset + -1.1*step), space=node_spacing)
    output[0].text(output_text, 'right', fontsize=font_size*2)

    if not no_box:
        viznet.connecta2a(pin_bot(_inner_block_2_connectable), pin_top(output), eb)
        return _inner_block_1, _inner_block_2, _inner_block_2_connectable, output
    else:
        viznet.connecta2a(pin_bot(_inner_block_1), pin_top(output), eb)
        return _inner_block_1, output



block_step = 1.15
block_step = 1.0
block_step = .8
block_step = .3
block_step = .2

block1s = block_1(inputs=nodes1,
                  text1="ReLu",
                  text2="Batch Norm",
                  box_colour='blue',
                  )
block2s = block_1(inputs=block1s[-1],
                  text1="ReLu",
                  text2="Dropout (0.25)",
                  offset=-block_step*1*step, use_121=False)
block3s = block_1(inputs=block2s[-1],
                  text1="ReLu",
                  text2="Dropout (0.25)",
                  offset=-block_step*2*step, use_121=False)
block4s = block_1(inputs=block3s[-1],
                  text1="ReLu",
                  text2="Dropout (0.25)",
                  offset=-block_step*3*step, use_121=False)
block5s = block_1(inputs=block4s[-1],
                  text1="ReLu",
                  text2="Dropout (0.25)",
                  offset=-block_step*4*step, use_121=False)



block_sigma = block_2(inputs=block5s[-1],
                  text1="Exponential",
                  output_text="$\\sigma_{q,m}$",
                  box_colour='red',
                  offset=-block_step*5*step, use_121=False)

block_alpha = block_2(inputs=block5s[-1],
                  text1="Softmax",
                  output_text="$\\alpha_{q,m}$",
                  box_colour='orange',
                  x_offset=4.5*size,
                  hidden_layer_text="$h_1 \sim h_{ Q}$",
                  offset=-block_step*5*step, use_121=False)

block_mu = block_2(inputs=block5s[-1],
                  text1="Softmax",
                  output_text="$\\mu_{q,m}$",
                  no_box=True,
                  box_colour='orange',
                  x_offset=-4.5*size,
                  offset=-block_step*5*step, use_121=False)





# import matplotlib
# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
#
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# rc('text.latex', preamble=r'\usepackage{cmbright}')

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.rcParams['font.family'] = 'cmu serif'

_show(fig_size=(18.5, 16.5), show=False)
plt.savefig("out.pdf")
plt.show()
