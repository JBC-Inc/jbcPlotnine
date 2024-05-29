import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import *

from plotnine import *
from plotnine.data import *
from plotnine.themes import *
import plotnine as pn

from shiny import *
import shinyswatch as ss

# python -m pip install

# data ========================================================================
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

# functions ===================================================================

def log_breaks(x):
  """
  Breaks y data from (min, max) return log scale decades. Will always capture 
  one scale above and one scale below. Used to set scale_y_log10 breaks option.
  
  Parameters
  ----------
  x : array-like
      Data points used to calculate log breaks.
  
  Returns
  -------
  numpy.ndarray
    Logarithmic breaks as a NumPy array of numeric values.
  """
  lower = np.floor(np.log10(np.min(x)))
  upper = np.ceil(np.log10(np.max(x)))
  cycles = np.arange(lower, upper + 1, 1)
  return np.power(10, cycles)

def log_limits(x):
  """
  Set scale_y_log10 limits option based on major log breaks. 
    
  Parameters
  ----------
  x : array-like
      Numeric vector for which logarithmic breaks are calculated.
    
  Returns
  -------
    Tuple with minimum and maximum limits of the logarithmic breaks.
  """
  breaks = log_breaks(x)
  return [np.min(breaks), np.max(breaks)]

def label_comma(x, digits=0):
  """
  Format numeric values with commas as thousands separators.
  
  Parameters
  ----------
  x : array-like 
      Numeric values to be formatted. 
  
  digits : int 
           Number of decimal places to display.
    
  Returns
  -------
    List of formatted strings.
  """
  return [f"{value:,.{digits}f}" for value in np.asarray(x)]

def p9_main(qual, metric1, metric2, quality=True, log=False, sec=subtitle):
  """
  Create plotnine main plot.
  
  Parameters
  ----------
  qual : pd.DataFrame
         Approximately 10 years of monthly historical natural gas well 
         production data including: 
          Rate time decline curves.
          Cumulative possible predictions.
          Cumulative probable estimated ultimate recovery.
          Assessed quality of forecasts made on rate/cum models.
  metric1 : string 
            Observed historical production rate ('rate', 'cum')
  metric2 : string 
            Predicted or Probable production rate ('rate_hat', 'cum_hat_fit', 
            'cum_hat_pred')
  quality : boolean
            Whether to plot a graph representing the quality of the forecast 
            that accompanies historical production over time. Default True.
  log : boolean
        Whether to plot production in actual or log scale. Default False.
    
  Returns
  -------
    plotnine.ggplot.ggplot object
  """
  plt.close('all')
    
  if metric1 == "rate":
    ylab = "Rate MCF/D"
  elif metric1 == "cum":
    ylab = "Cumulative Volume CF"
        
  t1="NCC-1701-D-1138 Gas Forecast\u2003\u2003\u2003\u2003\u2003\u2003"     
  t2="April 1, 2009 - Jan 1, 2011\u2003\u2003\u2003\u2003\u2003\u2003"
  t3="Report Created: "+datetime.today().strftime("%B %d, %Y")

  p = (
    ggplot(qual)
  
    # Historic ----------------------------------------------------------------
    + geom_point(mapping=aes('date', metric1, shape='Historic'),
                 color='firebrick',
                 alpha=0.5,
                 size=2.4,
                 stroke=0.42)
    + scale_shape_manual(labels=['Gas'], values=['o'])

    # Forecast ----------------------------------------------------------------
    + geom_line(mapping=aes('date', metric2, color='Forecast'), 
                alpha=0.77)
    + scale_color_manual(labels=['Model'], values=['firebrick'])

    # Segments ----------------------------------------------------------------
      
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

    # Scales, Labels, Themes --------------------------------------------------
    + scale_x_datetime(date_breaks='1 year', 
                       date_labels="%Y",
                       limits=[fcst_hs - pd.DateOffset(months=6), 
                               hist_he + pd.DateOffset(months=6)],
                       expand=(0, 0))
                       
    + labs(x='', y=ylab, title=t1+t2+t3+"\n", subtitle=sec)
        
    + theme_seaborn(style='darkgrid', 
                    context='notebook', 
                    font='MS Gothic', 
                    font_scale=1)  
        
    + theme(
        panel_grid_major_y=element_line(size=0.77, color='white'),
        panel_grid_minor_y=element_line(size=0.42, color='white'),
        axis_ticks_minor_x=element_blank(),
        panel_background=element_rect(fill='#01010111'),
        legend_key=element_rect(fill='#ffffff00'),
        axis_text_y=element_text(margin={'r': 0.125, 'units': 'in'}),
        figure_size=(19.2, 10.8))
  )
  # Log Scale -----------------------------------------------------------------
  if log == True:
    p = p + scale_y_log10(breaks=log_breaks,
                          limits=log_limits,
                          labels=label_comma)           
    p = p + theme(axis_text_y=element_text(margin={'r': 0.0, 'units': 'in'}))
  else:
    p = p + scale_y_continuous(labels =label_comma)
      
  # Quality Plot --------------------------------------------------------------
  if quality == True:
    p = p + theme(axis_text_x=element_blank(),
                  axis_ticks_major_x=element_blank(),
                  axis_ticks_minor_x=element_blank(),
                  axis_title_x=element_blank(),
                  plot_margin_bottom=0,
                  figure_size=(19.2, 10.8))
  return p

def p9_qual(qual):
  """
  Plot forecast quality metrics
  
  This plot will be optional and allows the user to view the quality of the 
  forecasts/predictions made over time.
  
  Parameters
  ----------
  qual : pd.DataFrame
         The main dataframe with quality data.

  Returns
  -------
    plotnine.ggplot.ggplot
  """
  q = (
    ggplot(qual)
    
    # Quality report ---------------------------------------------------------- 
    + geom_line(aes(x='date', y='quality', color='..y..'), 
                size=1, 
                show_legend=False)   
    + scale_color_gradient(high='green', low='red')
    + scale_x_datetime(date_labels="%Y", 
                       date_breaks='1 year',
                       limits=[fcst_hs - pd.DateOffset(months=6), 
                               hist_he + pd.DateOffset(months=6)],
                       expand=(0, 0))
    + scale_y_continuous(labels = lambda l: ["%d%%" % (v * 100) for v in l],
                         breaks = np.arange(0, 1.05, 0.05).tolist(),
                         minor_breaks = np.arange(0.9, 1, 0.01).tolist())
                         
    # Shaded ------------------------------------------------------------------
    + geom_rect(data=fr,
                mapping=aes(xmin='start', xmax='end', fill='color'), 
                ymin=float('-inf'), 
                ymax=float('inf'), 
                alpha=0.15, 
                show_legend=False)
    + scale_fill_manual(values=['#4300642a', '#43ff642a', '#ffff642a', '#4300642a'])
    
    # Forecast start/end ------------------------------------------------------
    + geom_vline(xintercept=[fcst_hs, hist_he], linetype='dashed', alpha=0.5)
    
    # Labels, themes ----------------------------------------------------------
    + labs(x='', y='Quality', title='')
        
    + theme_seaborn(style='darkgrid', 
                    context='notebook', 
                    font='MS Gothic', 
                    font_scale=1) 
                        
    + theme(
        axis_ticks_minor_x=element_blank(),
        panel_grid_major_y=element_line(size=0.77, color='white'),
        panel_grid_minor_y=element_line(size=0.42, color='white'),
        panel_background=element_rect(fill='#01010111'),
        plot_margin_top=0,
        axis_text_y=element_text(margin={'r': 0.1, 'units': 'in'}),
        figure_size=(19.2, 10.8))
  )
  return q  

# app =========================================================================

app_ui = ui.page_navbar(
  ui.nav_panel(
    "Gas Forecast",
    ui.tags.style(".shiny-input-container .checkbox input { border: 2px solid black; }"),
    ui.tags.style(".shiny-input-select { border: 2px solid black; }"),
    ss.theme.lux, 
    ui.layout_sidebar(
      ui.sidebar(
        ui.input_select("metrics", "Metrics:", 
                        {"Production Rate": {"rate_hat": "Decline Projection"}, 
                         "Total Yield": {"cum_hat_fit": "Historical Projection",
                                         "cum_hat_pred": "Estimated Ultimate Recovery"},},),
        ui.input_checkbox("quality", "Quality Plot", True),
        ui.input_checkbox("log", "Log Scale", False),
        ui.input_checkbox("sec", "Secondary Axis", True),
      ),
      ui.output_plot("plotnine", width="80%", height='500px'),
      ui.output_plot("quality_plot", width="71.5%", height='142px'),
    ),
  ),
  title="2024 Plotnine Contest\u2003\u2003\u2003\u2003"
)

def server(input, output, session):

  @output
  @render.plot()
  def plotnine():
    if input.metrics() == 'rate_hat':
      metric1='rate'
      metric2='rate_hat'
    quality = True if input.quality() else False
    log = True if input.log() else False
    sec = subtitle if input.sec() else ""
    return p9_main(qual, metric1, metric2, quality, log, sec)
  
  @output
  @render.plot()
  def quality_plot():
    if input.quality() == True:
      return p9_qual(qual)

app = App(app_ui, server)
 
 

# shiny run --reload --launch-browser app.py