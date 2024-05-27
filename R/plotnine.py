# Imports =====================================================================

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

import itertools
from datetime import * 

from plotnine import * 
from plotnine.data import *
from plotnine.facets import *
from plotnine.themes import *

import subprocess
print(subprocess.run(['pip', 'list'], capture_output=True, text=True).stdout)

# import warnings
# warnings.filterwarnings("ignore")

# Globals/data ================================================================

qual = pd.read_csv('./inst/data/qual.csv',
                   keep_default_na=False,
                   na_values=[' '],
                   parse_dates=['date'],
                   dtype={'region': 'string', 
                          'Historic': 'string', 
                          'Forecast': 'string'})
                          
fr = pd.read_csv('./inst/data/fill_region.csv',
                 parse_dates=['start', 'end'],
                 dtype={'region': 'category',
                        'color': 'category'})

# Forecast Historic Start April 1, 2009
fcst_hs = datetime(2009, 4, 1, tzinfo=timezone.utc)

# Historic Historic End    June 1, 2019
hist_he = datetime(2019, 6, 1, tzinfo=timezone.utc)

# Functions ===================================================================

def log_breaks(x):
    """
    Breaks y data from (min, max) return log scale decades. Will always capture
    one scale above and one scale below. Used to set scale_y_log10 breaks 
    option.
  
    Parameters:
      x (array-like): Data points used to calculate log breaks.
    
    Returns:
      numpy.ndarray: Logarithmic breaks as a NumPy array of numeric values.
    """
    lower = np.floor(np.log10(np.min(x)))
    upper = np.ceil(np.log10(np.max(x)))
    cycles = np.arange(lower, upper + 1, 1)
    return np.power(10, cycles)

# def log_minor_breaks(x):
    # """
    # Calculate logarithmic minor breaks based on major breaks.
    # 
    # Parameters:
    #     x (array-like): Numeric vector for which logarithmic breaks are
    #                     calculated.
    # 
    # Returns:
    #     numpy.ndarray: Logarithmic minor breaks as a NumPy array of numeric
    #                    values.
    # """
    # major_values = log_breaks(x)
    # minor_values = np.arange(2, 10)
    # y = pd.DataFrame({'minor': np.tile(minor_values, len(major_values)),
    #                   'cycle': np.repeat(major_values, len(minor_values))})
    # return y['cycle'] * y['minor']

def log_limits(x):
    """
    Set scale_y_log10 limits option based on major log breaks. 
    
    Parameters:
        x (array-like): Numeric vector for which logarithmic breaks are 
                        calculated.
    
    Returns:
        tuple: A tuple containing the minimum and maximum limits of the 
               logarithmic breaks.
    """
    breaks = log_breaks(x)
    return [np.min(breaks), np.max(breaks)]

def label_comma(x, digits=0):
    """
    Format numeric values with commas as thousands separators.
    
    Parameters:
      x (array-like): Numeric values to be formatted.
      digits (int): Number of decimal places to display.
    
    Returns:
      list: Formatted strings.
    """
    return [f"{value:,.{digits}f}" for value in np.asarray(x)]

# Forecast/Historic
def p9_main(qual, metric1, metric2, quality=True, log=False):
    """
    Create plotnine plot.
  
    Parameters:
      qual (pd.DataFrame): Approximately 10 years of historical natural gas well
        production data including: 
        rate time decline curves
        cumulative possible estimated ultimate recovery
        cumulative probable estimated ultimate recovery
        assessed quality of forecasts made on the data set
      metric1 (string): Observed historical production rate ('rate', 'cum')
      metric2 (string): Predicted or Probable production rate ('rate_hat', 
                        'cum_hat_fit', 'cum_hat_pred')
      quality (boolean): Whether to plot a graph representing the quality of the
                         forecast that accompanies historical production over
                         time. Default True.
      log (boolean): Whether to plot production in actual or log scale. Default
                     False.
    Returns:
      Side effect is to draw a plotnine object to the `Plots` pane in R Studio.
    """
    plt.close('all')
    
    if metric1 == "rate":
      ylab = "Rate MCF/D"
    elif metric1 == "cum":
      ylab = "Cumulative Volume CF"
        
    t1="NCC-1701-D-1138 Gas Forecast        "     
    t2="April 1, 2009 - Jan 1, 2011         "

    p = (
      ggplot(qual)
  
      # Historic --------------------------------------------------------------

      + geom_point(mapping=aes('date', metric1, shape='Historic'),
                   color='firebrick',
                   alpha=0.57,
                   #fill='None',
                   size=2.4,
                   stroke=0.42)
                   
      + scale_shape_manual(labels=['Gas'], values=['o'])

      # Forecast --------------------------------------------------------------

      + geom_line(data=qual,
                  mapping=aes('date', metric2, color='Forecast'), alpha=0.77)
                  
      + scale_color_manual(labels=['Model'], values=['firebrick'])
               
      # Segments --------------------------------------------------------------
      
      # Forecast Start
      + geom_vline(xintercept=[fcst_hs, hist_he], linetype='dashed', alpha=0.5)
      # Historic Start
      # + geom_vline(xintercept=hist_he, alpha=0.5, color='red')
      
      # Shaded fit/predict
      + geom_rect(data=fr, 
                  mapping=aes(xmin='start', xmax='end', fill='color'), 
                  ymin=float('-inf'), 
                  ymax=float('inf'))
                  
      + scale_fill_manual(
          name="Regions",
          labels=['Gap', 'Fit', 'Predict', 'Forecast'],
          values=['#4300642a', '#43ff642a', '#ffff642a', '#4300642a'])

      # Scales, Themes, Labels ------------------------------------------------

      + scale_x_datetime(date_breaks='1 year', 
                         date_labels="%Y",
                              limits=[fcst_hs - pd.DateOffset(months=6), 
                                      hist_he + pd.DateOffset(months=6)],
                              expand=(0, 0))

      + labs(x='',
             y=ylab, 
             title=t1+t2+"Report Created:"+datetime.today().strftime("%B %d, %Y"))
        
      + theme(panel_grid_major_y=element_line(size=0.77, color='white'),
              panel_grid_minor_y=element_line(size=0.42, color='white'),
              # axis_ticks_length_minor=50
              panel_background=element_rect(fill='#01010111'), 
              
              plot_margin_top=0.042,
              plot_margin_bottom=0.042,
              plot_title=element_text(hjust=0.9, margin={'b': 30}, size=12),
              
              legend_key = element_rect(fill='#00000000')
              )
    )
    
    # Log Scale ---------------------------------------------------------------
  
    if log == True:
      p = p + scale_y_log10(breaks=log_breaks,
                            limits=log_limits,
                            labels=label_comma)
    else:
      p = p + scale_y_continuous(labels =label_comma)
      
    # Quality Plot ------------------------------------------------------------
    
    if quality == True:
      
      p = p + theme(axis_text_x=element_blank(),
                    axis_ticks_x=element_blank(),
                    axis_ticks_x=element_blank(),
                    plot_margin_bottom=0.01
                    )
    return p

# Quality
def p9_qual():
  
  return None

# Main + secondary x + quality
def plot_nine(qual, metric1, metric2, quality, log):
  """
  Draw a secondary axis on top of existing plotnine figure. The function will
  first call p9 which generates the actual plotnine object (or draws it), then
  matplotlib.pyplot will be used to create a second x axis on top. The axis
  will display the days elapsed since production started.
  
  Because this feature is not yet implemented into plotnine functionality,
  (i.e. scale_x_datetime(sec.axis)), this function emulates the desired output.
  
  Parameters are inherited from p9()
  
  Returns:
    Side effect is to draw a plotnine graph to the `Plots` pane, then a 
    secondary x axis containing the 'days' scale.
  """
  # Main plot -----------------------------------------------------------------
  p9_main(qual, metric1, metric2, quality, log) 
  
  # Quality plot --------------------------------------------------------------
  # p9_qual

  # Secondary Axis hack -------------------------------------------------------
  #
  # matplotlib.pyplot
  plt.plot([], [])
  
  #look and feel of 2nd axis 
  # plt.rcParams.update({'font.size': 10, 'font.weight': 'normal', 'font.family': 'Fira Code'})

  # new axis 
  ax = plt.gca()

  # overlay secondary axis on plotnine plot, match limits of ggplot
  sec = ax.twiny()
  sec.set_xlim(ax.get_xlim());
  sec.set_ylim(ax.get_ylim());
  
  # Historic/Forecast start
  tick_locations = [fcst_hs]
  tick_labels = [0]  # Days since reference for start_date

  # Add tick locations for January 1st of subsequent years
  for year in range(2010, 2020):
    january_1st = datetime(year, 1, 1, tzinfo=timezone.utc)
    tick_locations.append(january_1st)
    tick_labels.append((january_1st - fcst_hs).days)
    
  # Set ticks and labels
  sec.set_xticks(ticks=tick_locations, labels=tick_labels)

  # Hide secondary spines
  sec.spines['top'].set_visible(False)
  sec.spines['right'].set_visible(False)
  sec.spines['bottom'].set_visible(False)
  sec.spines['left'].set_visible(False)

  # Hide ax minor ticks
  ax.tick_params(axis='x', which='both', color='white')
  
  plt.show()

  return figure


#==============================================================================
#=========================== plotnine plots ===================================
#==============================================================================

# default plotnine
theme_set(theme_gray())

theme_set(theme_classic(base_size=12, base_family='Fira Code'))
theme_set(theme_seaborn   (style='darkgrid', context='paper',    font='Fira Code', font_scale=1.42))
# theme_set(theme_matplotlib()) # Default matplotlib - has minor log ticks but no 
theme_set(theme_minimal   (base_size=12, base_family='Fira Code')) # Minimalistic no background annotations

theme_set(theme_seaborn   (style='darkgrid', context='notebook', font='Fira Code', font_scale=1))

# Historical rate time decline curve with forecasted production of a gas well.

theme_set(theme_seaborn   (style='darkgrid', context='notebook', font='MS Gothic', font_scale=1))

plot_nine(qual, 'rate', 'rate_hat', log=False)
plot_nine(qual, 'rate', 'rate_hat', log=True)

# Cumulative estimate production
plot_nine(qual, 'cum', 'cum_hat_fit', log=False)
plot_nine(qual, 'cum', 'cum_hat_fit', log=True)

# Cumulative predicted production
plot_nine(qual, 'cum', 'cum_hat_pred', log=False)
plot_nine(qual, 'cum', 'cum_hat_pred', log=True)





















#==============================================================================
#==============================================================================
#==============================================================================

# dir(plotnine.themes)
# ['__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', 
# '__name__', '__package__', '__path__', '__spec__', 'elements', 'targets', 
# 'theme', 'theme_538', 'theme_bw', 'theme_classic', 'theme_dark', 'theme_get', 
# 'theme_gray', 'theme_grey', 'theme_light', 'theme_linedraw', 
# 'theme_matplotlib', 'theme_minimal', 'theme_seaborn', 'theme_set', 
# 'theme_tufte', 'theme_update', 'theme_void', 'theme_xkcd', 'themeable']
# 
# theme_set(theme_classic(base_size=12, base_family='Times New Roman'))


# scales are named according to the type of variable they align with: 
#
# continuous   number   (integer, numeric)
# discrete     category (character, logical)
# datetime     dttm
# date         date
#
# Override the default:
#
#  change breaks on the axis or key labels on the legend
#
#  replace scale all together, because we know more about the data


# In Python, you can achieve similar functionalities as the cowplot package in
# R with a combination of libraries such as:
#   matplotlib
#   seaborn
#   gridspec
#   mpl_toolkits



# plt.gca()
#======================================================================================
# if there is no axes on this figure (but we havent created a figure yet, only a plot?)
# a new one is created using .Figure.add_subplot
#
# To test whether there is currently an axes on a figure, check whether 
#     ``figure.axes`` is empty
#
# to test whether there is currently an Figure on the pyplot figure stack, check whether 
#     .pyplot.get_fignums() is empty
# 
# plt.cla()             # clear the current axes
# plt.clf()             # clear the current figure
# plt.close()           # close a figure window
# 
# plt.get_fignums()
# fig.axes
# 


#==============================================================================
#========================== subplots ==========================================
#==============================================================================

plot1 = (ggplot(mtcars) + geom_point(aes('wt', 'mpg'), color='red'))
plot2 = (ggplot(mtcars) + geom_point(aes('wt', 'mpg'), color='blue'))

plot1.save("plot1.png")
plot2.save("plot2.png")


# matplotlib.pyplot (aliased as plt) module, 
# state-based interface to matplotlib plotting functions like MATLAB
#
#   create figures and plots and modify them
#   
#   what is the difference between a figure and a plot.
#
# plt.figure() Function:                                              figure is container
#
#   plt.figure() is a function within the matplotlib.pyplot module. 
#   When you call plt.figure(), it creates a new figure object. 
#   A figure in matplotlib is essentially a container for all plot elements, 
#   such as axes, titles, labels, and the plot itself.
#
# fig (an instance of matplotlib.figure.Figure)
# 
#   Object: fig is an instance of the matplotlib.figure.Figure class. 
#           This object represents the entire figure window or page <----------*
#           which can contain multiple plots (axes).
#
# Interaction Between plt and fig
# 
#   State-based Interface: The plt module maintains a stateful interface where 
#                          it keeps track of the current figure and axes. 
# 
#   Functions like plt.plot(), plt.xlabel(), etc., apply to the current axes in the current figure. 
# 
#   When you call plt.figure(), it `sets the current figure to the newly created figure`.
#
# Key Points:
# 
#   Module (plt): Contains functions to create and manipulate figures and plots.
#
#   Function (plt.figure()): Creates a new figure object and sets it as the current figure.
#
#   Object (fig): An instance of matplotlib.figure.Figure representing the entire figure. 
#                 It can be modified directly through methods specific to the Figure class.
#
#   Stateful Interface: plt functions modify the current figure and axes. 
#                       When you create a new figure with plt.figure(), subsequent plotting 
#                       commands apply to this new figure.
# 
#   By understanding these components, you can better grasp how to create and manage figures 
#   and plots using matplotlib. The state-based interface provided by plt is convenient for 
#   quick and simple plotting, while the object-oriented interface (direct manipulation of 
#   Figure and Axes objects) offers more control and flexibility.
# 
#  fig.add_artist() custom graphic elements directly to the figure canvas.
#
# plot1.save('plot1.png', height=height, width=width, dpi=500, verbose=True)
# plot2.save('plot2.png', height=height, width=width, dpi=500, verbose=True)
# 
# fig=plt.figure(figsize=figsize)                    # create new figure <class 'matplotlib.figure.Figure'>
#                                                    # and assigns it to fig everytime. 
# # plt.figure(fig.number)                           # current figure
# plt.autoscale(tight=True)                          # set autoscale for current axes in current figure
# 
# # plot1----------------------------------
# fig.add_subplot(211)                               # create new axes to figure as part of subplot
#                                                    # 2 rows 1 column 1st position
#                                                    
# plt.imshow(img.imread('plot1.png'), aspect='auto') # display data as an image on a 2D regular raster
# fig.tight_layout()                                 # adjust padding
# fig.get_axes()[0].axis('off')                      # hide axes lines, ticks, labels
# os.unlink('plot1.png')                             # 
# fig.patch.set_visible(False)                       # hide figure background patch
# 
# # plot2----------------------------------
# fig.add_subplot(212)
# plt.imshow(img.imread('plot2.png'), aspect='auto')
# fig.tight_layout()
# fig.get_axes()[1].axis('off')
# os.unlink('plot2.png')
# fig.patch.set_visible(False) # a rectangle defined via an anchorpoint
# 
# plt.show()

# from matplotlib import gridspec
# 
# row=2
# col=1
# height=None
# width=None 
# dpi=500
# ratio=None
# pixels=10000
# figsize=(12, 8)
# 
# if ratio is None: 
#   ratio = 1.5 * col / row
# 
# if height is None and width is not None: 
#   height = ratio * width
# 
# if height is not None and width is None: 
#   width = height / ratio
# 
# if height is None and width is None:
#   area = pixels / dpi
#   width = np.sqrt(area/ratio)
#   height = ratio * width

# plot1 = (ggplot(mtcars) + geom_point(aes('wt', 'mpg'), color='red') + labs(x='', y=''))
# plot2 = (ggplot(mtcars) + geom_point(aes('wt', 'mpg'), color='red') + labs(x='', y=''))
# 
# p1path = 'plot1.png'
# p2path = 'plot2.png'
# 
plot1.save(p1path, height=height, width=5, dpi=500)
plot2.save(p2path, height=height, width=5, dpi=500)

# fig, axs = plt.subplots(2, 1, figsize=figsize)

fig = plt.figure()

spec = gridspec.GridSpec(ncols=1, nrows = 2, height_ratios=[3, 1], width_ratios=[1])

img1 = img.imread(p1path)
ax0 = fig.add_subplot(spec[0, 0])
ax0.imshow(img1, aspect='auto')


#axs[0].imshow(img1, aspect=".56") # aspect='auto'

img2 = img.imread(p2path)
#axs[1].imshow(img2, aspect=".56")
ax1 = fig.add_subplot(spec[1, 0])
ax1.imshow(img2, aspect='auto')


ax0.axis('off')
ax1.axis('off')

plt.tight_layout()

plt.show()


# 

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

x = np.arange(0, 10, 0.1)
y = np.cos(x)
 
plt.close('all')
fig = plt.figure(figsize=(8, 8))

spec = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1])
 
ax0 = fig.add_subplot(spec[0])
ax0.plot(x, y)

ax2 = fig.add_subplot(spec[1])
ax2.plot(x, y)

plt.show()


import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.image as img
import numpy as np
from plotnine import ggplot, aes, geom_line

# Generate some data
x = np.arange(0, 10, 0.1)
y = np.cos(x)

# Create and save plotnine plots

(ggplot(pd.DataFrame({'x': x, 'y': y})) 
 + geom_line(aes('x', 'y'))
 + labs(y = 'yyyyyyyyyyyyyyyyyyyyyyyyyyy' , x = '')
 + theme(axis_text_x=element_blank(),
         axis_ticks_major_x=element_blank(),
         plot_margin=0.01)
 #+ theme_void()
 ).save('p1.png')
 
(ggplot(pd.DataFrame({'x': x, 'y': y})) 
 + geom_line(aes('x', 'y'))
 #+ theme_void()
 ).save('p2.png')

plt.close('all')

# Create a figure and a gridspec layout
fig = plt.figure(figsize=(12, 8))                               # Set the figure size
gs = gridspec.GridSpec(nrows=2, 
                       ncols=1, 
                       height_ratios=[4, 1]
                       )

plot1 = img.imread('p1.png')
plot2 = img.imread('p2.png')

# Add subplots to the gridspec layout
ax1 = fig.add_subplot(gs[0])  # Top subplot
ax2 = fig.add_subplot(gs[1])  # Bottom subplot

# Display images
ax1.imshow(plot1, aspect='auto') # , aspect='auto')
ax1.set_title('Top Plot')
ax1.axis('off')  # Hide the axis

ax2.imshow(plot2, aspect='auto')
#ax2.set_title('Bottom Plot')
ax2.axis('off')  # Hide the axis

# Adjust spacing
plt.subplots_adjust(hspace=0.0)

# Show the plot
plt.show()



#==============================================================================
#================ patchworklib ================================================
#==============================================================================
#
# ONLY PLOTNINE 0.12.4
# 
# we lose: 
#   
# plot_margin_left
# plot_margin_right
# plot_margin_top
# plot_margin_bottom
#
# axis_ticks_length_major_x
# axis_ticks_length_major_y
# axis_ticks_length_minor_x
# axis_ticks_length_minor_y
#
# axis_ticks_pad_major_x
# axis_ticks_pad_minor_y
#
=====================================================================================
=====================================================================================
========================== patchworklib =============================================
=====================================================================================
=====================================================================================

import os
from PIL import Image
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import matplotlib.image as img
import patchworklib as pw
import plotnine as plotnine

p1 = (ggplot(mtcars) 
      + geom_point(aes('wt', 'mpg', color='mpg'))
      + scale_color_gradient(low='red', high='green')
      + theme(
          axis_text_x        =element_blank(),
          axis_ticks_major_x =element_blank(),
          axis_title_x       =element_blank(),
          subplots_adjust    ={'left': 0.1, 'right': 0.9, 'top': 0.9, 'bottom': 0.0}
      )
      + labs(x='')
      )
      
p2 = (ggplot(mtcars) + geom_point(aes('wt', 'mpg'), color='blue') 
      # + theme(
      #     axis_title       =element_blank(),
      #     axis_ticks_major =element_blank(),
      #     axis_ticks_minor =element_blank(),
      #     plot_margin=(0)
      # )
      # + labs(x='', y='', title='')
      )
              
pw1 = pw.load_ggplot(p1, figsize=(10, 5))
pw2 = pw.load_ggplot(p2, figsize=(10, 1.42))

pw.param["margin"] = 0.1 # 'none'
fin = (pw1/pw2)
fin.savefig("finalplot.png", dpi=500)

# scale: handle the changes of dimension while keeping the proportions
# 
# plot = img.imread('finalplot.png')
# fig = plt.figure(figsize=(6.66, 5), dpi=200) 
# ax = fig.add_subplot(111)
# ax.imshow(plot1, aspect='.75')
# ax.margins(0.0, tight=True)
# ax.axis('off')
# plt.show()
# 











def set_ylab(metric1):
    """
    Sets the label for the y-axis based on the given metric.

    Parameters:
    metric1 (str): The metric to determine the y-axis label. Can be 'rate' or 'cum'.

    Returns:
    str: The y-axis label corresponding to the given metric.
    """
    if metric1 == "rate":
        ylab = "Rate MCF/D"
    elif metric1 == "cum":
        ylab = "Cumulative Volume CF"
    else:
        ylab = "Unknown Metric"

    return ylab




import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_point

def plot_with_secondary_axis():

    ggplot_plot = (ggplot(mtcars, aes(x='wt', y='mpg')) + geom_point() + theme(figure_size=(12,8)))

    fig, ax = plt.subplots()
    gg = ggplot_plot.draw(return_ggplot=True)

    # Your additional matplotlib plotting code here
    ax2 = ax.twinx()  
    ax2.set_yticks(ticks=[1, 2, 3, 4], labels=['1', '2', '3', '4'])

    return fig

# Call the function
plot_with_secondary_axis()



import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_point

def plot_with_secondary_axis():

    ggplot_plot = (ggplot(mtcars, aes(x='wt', y='mpg')) + geom_point() + theme_matplotlib())

    fig, ax = plt.subplots()
    combined_plot = ggplot_plot.draw(return_ggplot=True)

    ax2 = ax.twinx()  
    ax2.set_yticks(ticks=[1, 2, 3, 4], labels=['1', '2', '3', '4'])

    # Return the combined plotnine.ggplot object
    return combined_plot



import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_point, theme
from PIL import Image
import io

def pp():

    ggplot_plot = (ggplot(mtcars, aes(x='wt', y='mpg')) + geom_point())

   # Save plotnine plot to a bytes buffer
    buf = io.BytesIO()
    ggplot_plot.save(buf, format='png')
    buf.seek(0)

    # Load image with PIL
    img = Image.open(buf)

    # Create a matplotlib figure
    fig, ax = plt.subplots()

    # Display the plotnine plot as an image
    ax.imshow(img)

    # Add secondary axis using matplotlib
    ax2 = fig.add_subplot(111, frameon=False)
    ax2.plot(x_values, y_values, color='red')

    # Hide axes
    ax.axis('off')
    ax2.axis('off')

    # Return the ggplot object for further usage
    return ggplot_plot








(
  ggplot(qual)
  + geom_point(aes(x='date', y='rate'), color='#00000000')
  + scale_x_datetime(labels=tick_labels, breaks=tick_locations)
  + theme(
      axis_text_y=element_blank(),
      axis_ticks_major_y=element_blank(),
      
      axis_text_x=element_text(vjust=0.5), # .30, margin={'t': 100, 'b': -50}),
      
      axis_ticks_length_major_x=4.2,
      axis_ticks_pad_major_x=-10.42,
  
      axis_ticks_x=element_text(color='red'),
      
      panel_grid=element_blank(),
      panel_background=element_rect(fill="#00000000"),
      
      aspect_ratio=0.02,
    )
  + labs(x='', y='')
)







































