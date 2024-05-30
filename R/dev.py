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
import plotnine as pn

from shiny import ui

# import subprocess
# print(subprocess.run(['pip', 'list'], capture_output=True, text=True).stdout)

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
                        
ddf = pd.read_csv('./inst/data/ddf.csv')

with open("./inst/data/subtitle.txt", "r") as file:
  subtitle = file.read().rstrip()

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
    Create plotnine main plot.
  
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
      plotnine.ggplot.ggplot object.
    """
    plt.close('all')
    
    if metric1 == "rate":
      ylab = "Rate MCF/D"
    elif metric1 == "cum":
      ylab = "Cumulative Volume CF"
        
    t1="NCC-1701-D-1138 Gas Forecast\u2003\u2003\u2003"     
    t2="April 1, 2009 - Jan 1, 2011\u2003\u2003\u2003"
    t3="Report Created: "+datetime.today().strftime("%B %d, %Y")

    p = (
      ggplot(qual)
  
      # Historic --------------------------------------------------------------
      + geom_point(
          mapping=aes('date', metric1, shape='Historic'),
          color='firebrick',
          alpha=0.57,
          #fill='None',
          size=2.4,
          stroke=0.42)
                   
      + scale_shape_manual(labels=['Gas'], values=['o'])

      # Forecast --------------------------------------------------------------
      + geom_line(
          data=qual,
          mapping=aes('date', metric2, color='Forecast'), 
          alpha=0.77)
                  
      + scale_color_manual(labels=['Model'], values=['firebrick'])

      # Segments --------------------------------------------------------------
      
      # Forecast Start
      + geom_vline(xintercept=[fcst_hs, hist_he], linetype='dashed', alpha=0.5)
      # Historic Start
      # + geom_vline(xintercept=hist_he, alpha=0.5, color='red')
      
      # Shaded fit/predict
      + geom_rect(
          data=fr,
          mapping=aes(xmin='start', xmax='end', fill='color'), 
          ymin=float('-inf'), 
          ymax=float('inf'))
                  
      + scale_fill_manual(
          name="Regions",
          labels=['Gap', 'Fit', 'Predict', 'Forecast'],
          values=['#4300642a', '#43ff642a', '#ffff642a', '#4300642a'])

      # Scales, Labels, Themes ------------------------------------------------
      + scale_x_datetime(date_breaks='1 year', 
                         date_labels="%Y",
                              limits=[fcst_hs - pd.DateOffset(months=6), 
                                      hist_he + pd.DateOffset(months=6)],
                              expand=(0, 0))

      + labs(x='', y=ylab, title=t1+t2+t3+"\n", subtitle=subtitle)
        
      + pn.theme(
          panel_grid_major_y=element_line(size=0.77, color='white'),
          panel_grid_minor_y=element_line(size=0.42, color='white'),
          axis_ticks_minor_x=element_blank(), 
          panel_background=element_rect(fill='#01010111'),
          plot_title=element_text(size=9), # hjust=0.9, 
          plot_subtitle=element_text(size=8), 
          legend_key = element_rect(fill='#ffffff00'))
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
      p = p + pn.theme(axis_text_x=element_blank(),
                    axis_ticks_major_x=element_blank(),
                    plot_margin=0.01)
    return p

# secondary axis
def secondary():
   plt.plot([], [])
   
   # look and feel of 2nd axis 
   # plt.rcParams.update({'font.size': 10, 'font.weight': 'normal', 'font.family': 'Fira Code'})
   
   # new axis 
   ax = plt.gca()
   
   # overlay secondary axis on plotnine plot, match limits of ggplot
   sec = ax.twiny()
   sec.set_xlim(ax.get_xlim())
   sec.set_ylim(ax.get_ylim())
   
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
  
# ax2
def ax2(ddf):
  """
  Dummy dataframe for secondary axis. Was going to use matplotlib.pyplot but
  using patchworklib requires all objects to be plotnine.ggplot
  
  Parameters:
    ddf (pd.DataFrame): a dummy data frame containing the translated datetime
                        to days and datetimes used in the main plot.
  Returns:
    A plotnine.ggplot horizontal axis with inverted tickmarks.
  """
  p = (
    ggplot(ddf)
    + geom_point(aes(x='tick_locations', y='tick_labels'), color='#00000000')
    + scale_x_datetime(labels=ddf['tick_labels'].tolist(), 
                       breaks=ddf['tick_locations'].tolist())
    + theme(
        axis_title_x=element_blank(),
        axis_title_y=element_blank(),
        axis_text_y=element_blank(),                    # y
        axis_ticks_major_y=element_blank(),
        axis_text_x=element_text(vjust=0.5),            # x
        axis_ticks_length_major=4.2,                    # ticks
        axis_ticks_pad_major=-10.42,
        axis_ticks_major_x=element_text(color='black'),
        panel_grid=element_blank(),                     # panel
        panel_background=element_rect(fill="#00000000"),
        aspect_ratio=0.02
    )
  )
  return p

# Quality
def p9_qual(qual):
  p = (
    ggplot(qual)
    
    # Quality report 
    + geom_line(aes(x='date', y='quality', color='..y..'), 
                size=1, 
                show_legend=False)
    + scale_color_gradient(high='green', 
                           mid='yellow', 
                           low='red', 
                           midpoint=0.95)
    + scale_x_datetime(date_labels="%Y", 
                       date_breaks='1 year', 
                       expand=[0, 0])
    + scale_y_continuous(labels = lambda l: ["%d%%" % (v * 100) for v in l],
                         breaks = np.arange(0, 1.05, 0.05).tolist(),
                         minor_breaks = np.arange(0.9, 1, 0.01).tolist())
                         
    # Shaded 
    + geom_rect(data=fr,
                mapping=aes(xmin='start', xmax='end', fill='color'), 
                ymin=float('-inf'), 
                ymax=float('inf'), 
                alpha=0.15, 
                show_legend=False)
    + scale_fill_manual(values=['#4300642a', '#43ff642a', '#ffff642a', '#4300642a'])
    
    # Forecast start/end
    + geom_vline(xintercept=[fcst_hs, hist_he], linetype='dashed', alpha=0.5)
    
    # Labels, themes
    + labs(x='', y='Quality')
                        
    + theme(panel_grid_major_y=element_line(size=0.77, color='white'),
            panel_grid_minor_y=element_line(size=0.42, color='white'),
            axis_ticks_minor_x=element_blank(),
            panel_background=element_rect(fill='#01010111'))
  )
  
  return p

# Final plot
# def plot_nine(qual, metric1, metric2, quality, log):
#   """
#   Use patchworklib library to either save 1 or 2 plots. The plot that will 
#   always be active is the main plot. The quality plot is optional. 
  
#   There is currently no way to implement a secondary x axis while rendering 
#   multiple plots. This would be possible in ggplot2 with cowplot and 
#   scale_x_datetime(sec.axis)

#   Parameters are inherited from p9_main().
  
#   Returns:
#     Side effect is to output the final plot: "plotnine.png" into the working
#     directory.
#   """
#   main = p9_main(qual, metric1, metric2, quality, log) 
#   pw1 = pw.load_ggplot(main, figsize=(10, 5))
  
#   qual = p9_qual(qual)
#   pw2 = pw.load_ggplot(qual, figsize=(10, 1))
  
#   if quality == True:
#     pw.param["margin"]=0.1
#     fin = (pw1/pw2)
#     fin.savefig("plotnine.png", dpi=500)
#   else: 
#     fin = (pw1)
#     fin.savefig("plotnine.png", dpi=500)

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

# theme_set(theme_seaborn   (style='darkgrid', context='notebook', font='MS Gothic', font_scale=1))

# plot_nine(qual, 'rate', 'rate_hat', quality=True, log=False)
# plot_nine(qual, 'rate', 'rate_hat', quality=True, log=True)
# plot_nine(qual, 'rate', 'rate_hat', quality=False, log=False)
# plot_nine(qual, 'rate', 'rate_hat', quality=False, log=True)

# # Cumulative estimate production
# plot_nine(qual, 'cum', 'cum_hat_fit', quality=True, log=False)
# plot_nine(qual, 'cum', 'cum_hat_fit', quality=True, log=True)

# # Cumulative predicted production
# plot_nine(qual, 'cum', 'cum_hat_pred', quality=True, log=False)
# plot_nine(qual, 'cum', 'cum_hat_pred', quality=True, log=True)





with open("./inst/data/subtitle.txt", "r") as file:
  subtitle = file.read().rstrip()

with open("./inst/data/subtitle.txt", "w") as file:
  file.write(subtitle)

subtitle = 'Days:0       275        640       1005        1371       1736        2101       2466       2832       3197        3562'















