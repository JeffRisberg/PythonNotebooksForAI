from inspect import getsource

from IPython.display import HTML
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


#______________________________________________________________________________
# Magic Words


def pseudocode(algorithm):
    """Print the pseudocode for the given algorithm."""
    from urllib.request import urlopen
    from IPython.display import Markdown

    algorithm = algorithm.replace(' ', '-')
    url = "https://raw.githubusercontent.com/aimacode/aima-pseudocode/master/md/{}.md".format(algorithm)
    f = urlopen(url)
    md = f.read().decode('utf-8')
    md = md.split('\n', 1)[-1].strip()
    md = '#' + md
    return Markdown(md)


def psource(*functions):
    """Print the source code for the given function(s)."""
    source_code = '\n\n'.join(getsource(fn) for fn in functions)
    try:
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import PythonLexer
        from pygments import highlight

        display(HTML(highlight(source_code, PythonLexer(), HtmlFormatter(full=True))))

    except ImportError:
        print(source_code)

# ______________________________________________________________________________
# MDP


def make_plot_grid_step_function(columns, rows, U_over_time):
    """ipywidgets interactive function supports single parameter as input.
    This function creates and return such a function by taking as input
    other parameters."""

    def plot_grid_step(iteration):
        data = U_over_time[iteration]
        data = defaultdict(lambda: 0, data)
        grid = []
        for row in range(rows):
            current_row = []
            for column in range(columns):
                current_row.append(data[(column, row)])
            grid.append(current_row)
        grid.reverse() # output like book
        fig = plt.imshow(grid, cmap=plt.cm.bwr, interpolation='nearest')

        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        for col in range(len(grid)):
            for row in range(len(grid[0])):
                magic = grid[col][row]
                fig.axes.text(row, col, "{0:.2f}".format(magic), va='center', ha='center')

        plt.show()

    return plot_grid_step

def make_visualize(slider):
    """Takes an input a sliderand returns callback function
    for timer and animation."""

    def visualize_callback(Visualize, time_step):
        if Visualize is True:
            for i in range(slider.min, slider.max + 1):
                slider.value = i
                time.sleep(float(time_step))

    return visualize_callback

# ______________________________________________________________________________


_canvas = """
<script type="text/javascript" src="./js/canvas.js"></script>
<div>
<canvas id="{0}" width="{1}" height="{2}" style="background:rgba(158, 167, 184, 0.2);" onclick='click_callback(this, event, "{3}")'></canvas>
</div>

<script> var {0}_canvas_object = new Canvas("{0}");</script>
"""  # noqa


class Canvas:
    """Inherit from this class to manage the HTML canvas element in jupyter notebooks.
    To create an object of this class any_name_xyz = Canvas("any_name_xyz")
    The first argument given must be the name of the object being created.
    IPython must be able to reference the variable name that is being passed."""

    def __init__(self, varname, width=800, height=600, cid=None):
        self.name = varname
        self.cid = cid or varname
        self.width = width
        self.height = height
        self.html = _canvas.format(self.cid, self.width, self.height, self.name)
        self.exec_list = []
        display_html(self.html)

    def mouse_click(self, x, y):
        """Override this method to handle mouse click at position (x, y)"""
        raise NotImplementedError

    def mouse_move(self, x, y):
        raise NotImplementedError

    def execute(self, exec_str):
        """Stores the command to be executed to a list which is used later during update()"""
        if not isinstance(exec_str, str):
            print("Invalid execution argument:", exec_str)
            self.alert("Received invalid execution command format")
        prefix = "{0}_canvas_object.".format(self.cid)
        self.exec_list.append(prefix + exec_str + ';')

    def fill(self, r, g, b):
        """Changes the fill color to a color in rgb format"""
        self.execute("fill({0}, {1}, {2})".format(r, g, b))

    def stroke(self, r, g, b):
        """Changes the colors of line/strokes to rgb"""
        self.execute("stroke({0}, {1}, {2})".format(r, g, b))

    def strokeWidth(self, w):
        """Changes the width of lines/strokes to 'w' pixels"""
        self.execute("strokeWidth({0})".format(w))

    def rect(self, x, y, w, h):
        """Draw a rectangle with 'w' width, 'h' height and (x, y) as the top-left corner"""
        self.execute("rect({0}, {1}, {2}, {3})".format(x, y, w, h))

    def rect_n(self, xn, yn, wn, hn):
        """Similar to rect(), but the dimensions are normalized to fall between 0 and 1"""
        x = round(xn * self.width)
        y = round(yn * self.height)
        w = round(wn * self.width)
        h = round(hn * self.height)
        self.rect(x, y, w, h)

    def line(self, x1, y1, x2, y2):
        """Draw a line from (x1, y1) to (x2, y2)"""
        self.execute("line({0}, {1}, {2}, {3})".format(x1, y1, x2, y2))

    def line_n(self, x1n, y1n, x2n, y2n):
        """Similar to line(), but the dimensions are normalized to fall between 0 and 1"""
        x1 = round(x1n * self.width)
        y1 = round(y1n * self.height)
        x2 = round(x2n * self.width)
        y2 = round(y2n * self.height)
        self.line(x1, y1, x2, y2)

    def arc(self, x, y, r, start, stop):
        """Draw an arc with (x, y) as centre, 'r' as radius from angles 'start' to 'stop'"""
        self.execute("arc({0}, {1}, {2}, {3}, {4})".format(x, y, r, start, stop))

    def arc_n(self, xn, yn, rn, start, stop):
        """Similar to arc(), but the dimensions are normalized to fall between 0 and 1
        The normalizing factor for radius is selected between width and height by
        seeing which is smaller."""
        x = round(xn * self.width)
        y = round(yn * self.height)
        r = round(rn * min(self.width, self.height))
        self.arc(x, y, r, start, stop)

    def clear(self):
        """Clear the HTML canvas"""
        self.execute("clear()")

    def font(self, font):
        """Changes the font of text"""
        self.execute('font("{0}")'.format(font))

    def text(self, txt, x, y, fill=True):
        """Display a text at (x, y)"""
        if fill:
            self.execute('fill_text("{0}", {1}, {2})'.format(txt, x, y))
        else:
            self.execute('stroke_text("{0}", {1}, {2})'.format(txt, x, y))

    def text_n(self, txt, xn, yn, fill=True):
        """Similar to text(), but with normalized coordinates"""
        x = round(xn * self.width)
        y = round(yn * self.height)
        self.text(txt, x, y, fill)

    def alert(self, message):
        """Immediately display an alert"""
        display_html('<script>alert("{0}")</script>'.format(message))

    def update(self):
        """Execute the JS code to execute the commands queued by execute()"""
        exec_code = "<script>\n" + '\n'.join(self.exec_list) + "\n</script>"
        self.exec_list = []
        display_html(exec_code)


def display_html(html_string):
    display(HTML(html_string))


################################################################################


############################################################################################################

#####################           Functions to assist plotting in search.ipynb            ####################

############################################################################################################
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import lines

import ipywidgets as widgets
from IPython.display import display
import time
from search import GraphProblem, romania_map

def show_map(graph_data, node_colors = None):
    G = nx.Graph(graph_data['graph_dict'])
    node_colors = node_colors or graph_data['node_colors']
    node_positions = graph_data['node_positions']
    node_label_pos = graph_data['node_label_positions']
    edge_weights= graph_data['edge_weights']
    
    # set the size of the plot
    plt.figure(figsize=(18,13))
    # draw the graph (both nodes and edges) with locations from romania_locations
    nx.draw(G, pos={k: node_positions[k] for k in G.nodes()},
            node_color=[node_colors[node] for node in G.nodes()], linewidths=0.3, edgecolors='k')

    # draw labels for nodes
    node_label_handles = nx.draw_networkx_labels(G, pos=node_label_pos, font_size=14)
    
    # add a white bounding box behind the node labels
    [label.set_bbox(dict(facecolor='white', edgecolor='none')) for label in node_label_handles.values()]

    # add edge lables to the graph
    nx.draw_networkx_edge_labels(G, pos=node_positions, edge_labels=edge_weights, font_size=14)
    
    # add a legend
    white_circle = lines.Line2D([], [], color="white", marker='o', markersize=15, markerfacecolor="white")
    orange_circle = lines.Line2D([], [], color="orange", marker='o', markersize=15, markerfacecolor="orange")
    red_circle = lines.Line2D([], [], color="red", marker='o', markersize=15, markerfacecolor="red")
    gray_circle = lines.Line2D([], [], color="gray", marker='o', markersize=15, markerfacecolor="gray")
    green_circle = lines.Line2D([], [], color="green", marker='o', markersize=15, markerfacecolor="green")
    plt.legend((white_circle, orange_circle, red_circle, gray_circle, green_circle),
               ('Un-explored', 'Frontier', 'Currently Exploring', 'Explored', 'Final Solution'),
               numpoints=1, prop={'size':16}, loc=(.8,.75))
    
    # show the plot. No need to use in notebooks. nx.draw will show the graph itself.
    plt.show()
    
## helper functions for visualisations
   
def final_path_colors(initial_node_colors, problem, solution):
    "Return a node_colors dict of the final path provided the problem and solution."
    
    # get initial node colors
    final_colors = dict(initial_node_colors)
    # color all the nodes in solution and starting node to green
    final_colors[problem.initial] = "green"
    for node in solution:
        final_colors[node] = "green"  
    return final_colors

def display_visual(graph_data, user_input, algorithm=None, problem=None):
    initial_node_colors = graph_data['node_colors']
    if user_input == False:
        def slider_callback(iteration):
            # don't show graph for the first time running the cell calling this function
            try:
                show_map(graph_data, node_colors=all_node_colors[iteration])
            except:
                pass
        def visualize_callback(Visualize):
            if Visualize is True:
                button.value = False
                
                global all_node_colors
                
                iterations, all_node_colors, node = algorithm(problem)
                solution = node.solution()
                all_node_colors.append(final_path_colors(all_node_colors[0], problem, solution))
                
                slider.max = len(all_node_colors) - 1
                
                for i in range(slider.max + 1):
                    slider.value = i
                     #time.sleep(.5)
        
        slider = widgets.IntSlider(min=0, max=1, step=1, value=0)
        slider_visual = widgets.interactive(slider_callback, iteration=slider)
        display(slider_visual)

        button = widgets.ToggleButton(value=False)
        button_visual = widgets.interactive(visualize_callback, Visualize=button)
        display(button_visual)
    
    if user_input == True:
        node_colors = dict(initial_node_colors)
        if isinstance(algorithm, dict):
            assert set(algorithm.keys()).issubset({"Breadth First Tree Search",
                                                       "Depth First Tree Search", 
                                                       "Breadth First Search", 
                                                       "Depth First Graph Search", 
                                                       "Uniform Cost Search", 
                                                       "A-star Search"})

            algo_dropdown = widgets.Dropdown(description="Search algorithm: ",
                                             options=sorted(list(algorithm.keys())),
                                             value="Breadth First Tree Search")
            display(algo_dropdown)
        elif algorithm is None:
            print("No algorithm to run.")
            return 0
        
        def slider_callback(iteration):
            # don't show graph for the first time running the cell calling this function
            try:
                show_map(graph_data, node_colors=all_node_colors[iteration])
            except:
                pass
            
        def visualize_callback(Visualize):
            if Visualize is True:
                button.value = False
                
                problem = GraphProblem(start_dropdown.value, end_dropdown.value, romania_map)
                global all_node_colors
                
                user_algorithm = algorithm[algo_dropdown.value]
                
                iterations, all_node_colors, node = user_algorithm(problem)
                solution = node.solution()
                all_node_colors.append(final_path_colors(all_node_colors[0], problem, solution))

                slider.max = len(all_node_colors) - 1
                
                for i in range(slider.max + 1):
                    slider.value = i
                    #time.sleep(.5)
                         
        start_dropdown = widgets.Dropdown(description="Start city: ",
                                          options=sorted(list(node_colors.keys())), value="Arad")
        display(start_dropdown)

        end_dropdown = widgets.Dropdown(description="Goal city: ",
                                        options=sorted(list(node_colors.keys())), value="Fagaras")
        display(end_dropdown)
        
        button = widgets.ToggleButton(value=False)
        button_visual = widgets.interactive(visualize_callback, Visualize=button)
        display(button_visual)
        
        slider = widgets.IntSlider(min=0, max=1, step=1, value=0)
        slider_visual = widgets.interactive(slider_callback, iteration=slider)
        display(slider_visual)


# Function to plot a heatmap, given a grid
def heatmap(grid, cmap='binary', interpolation='nearest'):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.set_title('Heatmap')
    plt.imshow(grid, cmap=cmap, interpolation=interpolation)
    fig.tight_layout()
    plt.show()

# Generates a gaussian kernel
def gaussian_kernel(l=5, sig=1.0):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))
    return kernel
