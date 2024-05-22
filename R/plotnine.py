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

#=================== 2024 PLOTNINE CONTEST - POSIT ============================
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

def p9(qual, metric1, metric2, quality, log):
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
  
    p = (
      ggplot()
  
      # Historic --------------------------------------------------------------
    
      + geom_point(data=qual,
                   mapping=aes('date', metric1, shape='Historic'), 
                   color='firebrick',
                   fill='None',
                   size=2.4,
                   stroke=0.42)
                   
      # Forecast --------------------------------------------------------------
    
      + geom_line(data=qual,
                  mapping=aes('date', metric2, group='Forecast', color='Forecast'))
                
      # Scales, Themes, Labels ------------------------------------------------

      + scale_x_datetime(date_breaks='1 year', 
                         date_labels="%Y",
                              limits=[fcst_hs - pd.DateOffset(months=6), 
                                      hist_he + pd.DateOffset(months=6)],
                              expand=(0, 0))
    
      + theme(plot_margin_top=0.05)
    
      + labs(x='', y='')
    )
  
    # Log Scale ---------------------------------------------------------------
  
    if log == True:
      p = p + scale_y_log10(breaks=log_breaks,
                            limits=log_limits,
                            labels=label_comma)
    else:
      p = p + scale_y_continuous(labels = label_comma)
      
    return p

def plot_nine(qual, metric1, metric2, quality=False, log=False):
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
  # plotnine.ggplot
  p9(qual, metric1, metric2, quality, log).show()
  
  # Secondary Axis hack -------------------------------------------------------
  # matplotlib.pyplot
  plt.plot([], [])

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
  ax.tick_params(axis='x', which='minor', color='white')
  
  plt.show()
  
#==============================================================================
#=========================== plotnine plots ===================================
#==============================================================================

# Historical rate time decline curve with forecasted production of a gas well.

# Rate Time decline curve
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

plt.cla()             # clear the current axes
plt.clf()             # clear the current figure
plt.close()           # close a figure window

plt.get_fignums()
fig.axes



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
















































