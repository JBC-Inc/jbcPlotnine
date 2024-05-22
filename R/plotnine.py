#import subprocess
#print(subprocess.run(['pip', 'list'], capture_output=True, text=True).stdout)

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import itertools
import numpy as np

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

from plotnine import *
from plotnine.facets import facet_grid
from plotnine.themes import *
from plotnine.data import *

from datetime import datetime, timezone

# import warnings
# warnings.filterwarnings("ignore")

# dir(plotnine.themes)
# ['__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', 
# '__name__', '__package__', '__path__', '__spec__', 'elements', 'targets', 
# 'theme', 'theme_538', 'theme_bw', 'theme_classic', 'theme_dark', 'theme_get', 
# 'theme_gray', 'theme_grey', 'theme_light', 'theme_linedraw', 
# 'theme_matplotlib', 'theme_minimal', 'theme_seaborn', 'theme_set', 
# 'theme_tufte', 'theme_update', 'theme_void', 'theme_xkcd', 'themeable']
# 
# theme_set(theme_classic(base_size=12, base_family='Times New Roman'))

#==============================================================================
# Rules:
# 
#  1.Technically impressive
#
#  2.Well documented example
#
#  3.Demonstrate novel, useful elements of plot design
#
#  4.Aesthetically pleasing
#
#==============================================================================

qual = pd.read_csv('./inst/data/qual.csv', 
                   keep_default_na=False, 
                   na_values=[' '],
                   parse_dates=['date'],
                   dtype={'region': 'string', 
                          'Historic': 'string', 
                          'Forecast': 'string'})
                          
reference_date = qual['date'].min()
qual['days_since_ref'] = (qual['date'] - reference_date).dt.days

clr = 'firebrick'

fcst_hs = datetime(2009, 4, 1, tzinfo=timezone.utc)   # Forecast Historic Start April 1, 2009
hist_he = datetime(2019, 6, 1, tzinfo=timezone.utc)   # Historic Historic End    June 1, 2019

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

def log_minor_breaks(x):
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

def date_to_days(date):
    return (date - start_d).days

def p9(qual, metric1, metric2, quality, log):

  plt.close('all')
  
  p = (
    ggplot()
  
    # Historic ----------------------------------------------------------------
    + geom_point(data=qual,
                 mapping=aes('date', metric1, shape='Historic'),
                 color='firebrick',
                 fill='None',
                 size=2.4,
                 stroke=0.42)
                   
    # Forecast ----------------------------------------------------------------
    + geom_line(data=qual,
                mapping=aes('date', metric2, group='Forecast', color='Forecast'))

    + scale_x_datetime(date_breaks='1 year', 
                       date_labels="%Y",
                            limits=[fcst_hs - pd.DateOffset(months=6), 
                                    hist_he + pd.DateOffset(months=6)],
                            expand=(0, 0))
    
    + theme(plot_margin_top=0.05)
    
    + labs(x='', y='')
  )
  
  if log == True:
    p = p + scale_y_log10(breaks=log_breaks,
                            limits=log_limits,
                            labels=label_comma)
  else:
    p = p + scale_y_continuous(labels = label_comma)

  return p

def plot_nine(qual, metric1, metric2, quality=False, log=False):

  # plotnine.ggplot
  p9(qual, metric1, metric2, quality, log).show()
  
  # matplotlib.pyplot
  plt.plot([], [])

  # new axis 
  ax = plt.gca()

  # overlay secondary axis on plotnine plot
  sec = ax.twiny()
  sec.set_xlim(ax.get_xlim());
  sec.set_ylim(ax.get_ylim());
  
  # Historic/Forecast start
  start_date = qual['date'].iloc[0]
  tick_locations = [start_date]
  tick_labels = [0]  # Days since reference for start_date

  # Add tick locations for January 1st of subsequent years
  for year in range(2010, 2020):
    january_1st = datetime(year, 1, 1, tzinfo=timezone.utc)
    tick_locations.append(january_1st)
    tick_labels.append((january_1st - start_date).days)
  
  # Set ticks and labels
  sec.set_xticks(ticks=tick_locations, labels=tick_labels)

  # Hide spines
  sec.spines['top'].set_visible(False)
  sec.spines['right'].set_visible(False)
  sec.spines['bottom'].set_visible(False)
  sec.spines['left'].set_visible(False)

  # Hide x minor ticks
  ax.tick_params(axis='x', which='minor', color='white')
  
  plt.show()
  
#==============================================================================
#=========================== plotnine plots ===================================
#==============================================================================

# Historical rate time decline curve with forecasted production of a gas well.

plot_nine(qual, 'rate', 'rate_hat', log=False)




plot_nine(qual, 'rate', 'rate_hat', log=True)

plot_nine(qual, 'cum', 'cum_hat_fit', log=False).show()
plot_nine(qual, 'cum', 'cum_hat_fit', log=True).show()

plot_nine(qual, 'cum', 'cum_hat_predict', log=False).show()
plot_nine(qual, 'cum', 'cum_hat_predict', log=True).show()


#==============================================================================
#==============================================================================
#==============================================================================


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





































# plotnine secondary x axis with matplotlib pyplot method ==================

# [1] - 1 figure exists on the pyplot stack. 
#       everytime "p" is defined, another figure is created
#
# [1, 2, 3, 4, etc...]

plt.close('all')
plt.get_fignums()

# this creates a figure
p = (ggplot() 
     + geom_point(aes(x=[1, 2, 3], y=[4, 5, 6]), color = 'purple', size = 5)
     
     # make room for second x axis
     + theme(plot_margin_top=0.05))

# this also creates a figure
# p.show()

plt.get_fignums()

# this creates a plot on the current figure, or creates a figure
plt.plot([], [])

plt.get_fignums()

# create new MATPLOTLIB axis for existing figure plot
ax1 = plt.gca()

# set a seconday axis
sec = ax1.twiny()

# match the limits to the original
sec.set_xlim(ax1.get_xlim())
sec.set_ylim(ax1.get_ylim())

# remove the borders
sec.spines['top'].set_visible(False)
sec.spines['right'].set_visible(False)
sec.spines['bottom'].set_visible(False)
sec.spines['left'].set_visible(False)

# this does NOT create a figure
plt.show()


# the problem now becomes: how to shrink ggplot for top x axis? 









































































# create new axes for plt, which is part of fig 1 plt
ax1 = plt.gca()

plt.get_fignums() # [1]

# create a secondary X axis on top
secondary_x = ax1.twiny()

# lookat the x, y limits
ax1.get_ylim()
ax1.get_xlim()

# set the same limits for the secondary axis
secondary_x.set_xlim(ax1.get_xlim())
secondary_x.set_ylim(ax1.get_ylim())

# make optional label with position 
# secondary_x.set_xlabel("AAAAAA", x=0.4, y=0)

plt.show()




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

plt.cla()             # clear the current axes
plt.clf()             # clear the current figure
plt.close()           # close a figure window

plt.get_fignums()
fig.axes
ax.clear
ax.cla
ax

ax = plt.gca()
ax.clear()
fig.axes





ax = plt.gca()

ax.twiny()

ax.set_xlabel("secondary x")

#plt.subplots_adjust(hspace=10)

plt.show()




plt.plot([1, 2, 3, 4], [10, 20, 25, 30])
plt.xlabel('Primary X-axis')
plt.ylabel('Y-axis')

# Create a secondary x-axis on top
ax1 = plt.gca()
ax2 = ax1.twiny()

# Set the same limits for the secondary x-axis
ax2.set_xlim(ax1.get_xlim())

# Set the labels for the secondary x-axis
ax2.set_xlabel('Secondary X-axis')

# Optionally customize the secondary x-axis ticks and labels
ax2.set_xticks([1, 2, 3, 4])
ax2.set_xticklabels(['A', 'B', 'C', 'D'])

# Adjust the layout to shorten the height of the actual plot area
plt.subplots_adjust(bottom=0.15)  # Decrease the bottom margin

# Show the plot
plt.show()






# plt.twinx()
#==============================================================================

import matplotlib.pyplot as plt
import numpy as np

# Create some mock data
t = np.arange(0.01, 10.0, 0.01)
data1 = np.exp(t)
data2 = np.sin(2 * np.pi * t)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('exp', color=color)
ax1.plot(t, data1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
ax2.plot(t, data2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()



# SUBPLOT ====================================================

plt.close('all')
 
plt.subplot(221)
plt.show()

# equivalent but more general
ax1 = plt.subplot(2, 2, 1)
plt.show()

# add a subplot with no frame
ax2 = plt.subplot(222, frameon=False)
plt.show()

# add a polar subplot
plt.subplot(223, projection='polar')
plt.show()

# add a red subplot that shares the x-axis with ax1
plt.subplot(224, sharex=ax1, facecolor='red')
plt.show()

# delete ax2 from the figure
plt.delaxes(ax2)
plt.show()

# add ax2 to the figure again
# plt.subplot(ax2)
# plt.show()

# make the first Axes "current" again
plt.subplot(221)
plt.show()
















































