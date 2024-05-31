import numpy as np
import pandas as pd

from datetime import *

from plotnine import *
from plotnine.themes import *

from shiny import App, render, ui
import shinyswatch as ss

# data ========================================================================

# Data set with historic production, forecast models and quality metrics
qual = pd.read_csv(
  filepath_or_buffer='./inst/data/qual.csv',
  keep_default_na=False,
  na_values=[' '],
  parse_dates=['date'],
  dtype={'region': 'string',
         'Historic': 'string',
         'Forecast': 'string'}
)

# DataFrame containing the regions which are to be colored.                  
fr = pd.read_csv(
  filepath_or_buffer='./inst/data/fill_region.csv',
  parse_dates=['start', 'end'],
  dtype={'region': 'category',
  'color': 'category'}
)

# Experimental secondary axis text.
with open(file="./inst/data/subtitle.txt", mode="r") as file:
  subtitle = file.read().rstrip()

# Production Historic Start Date April 1, 2009
hs = datetime(2009, 4, 1, tzinfo=timezone.utc)

# Forecast Model Start Date Janurary 1, 2011
fs = datetime(2011, 1, 1, tzinfo=timezone.utc)

# Historic Production/Forecast End Date June 1, 2019
he = datetime(2019, 6, 1, tzinfo=timezone.utc)

# functions ===================================================================

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

def p9_main(qual, metric1, metric2, quality=True, log=False, sec=subtitle):
  """
  Create plotnine main plot.
  
  Parameters
  ----------
  qual : pd.DataFrame
         Approximately 10 years of monthly historical natural gas well 
         production data including: 
          Rate time & predicted decline curves.
          Cumulative & predicted estimated ultimate recovery.
          Assessed quality of forecasts made on rate/cum models.
  metric1 : string 
            Observed historical production rate ('rate', 'cum')
  metric2 : string 
            Predicted or Probable production rate ('rate_hat', 'cum_hat_pred')
  quality : boolean
            Whether to plot a graph representing the quality of the forecast 
            that accompanies historical production over time. Default True.
  log : boolean
        Whether to plot production in actual or log scale. Default False.
    
  Returns
  -------
    plotnine.ggplot.ggplot
  """
  if metric1 == "rate":
    ylab = "Flow Rate CCF"
  elif metric1 == "cum":
    ylab = "Estimated Ultimate Recovery: CF"
        
  t1="NCC-1701-D-1138 Gas Forecast\u2003\u2003\u2003\u2003\u2003\u2003"     
  t2="April 1, 2009 - Jan 1, 2011\u2003\u2003\u2003\u2003\u2003\u2003"
  t3="Report Created: "+datetime.today().strftime("%B %d, %Y")

  p = (
    ggplot(qual)
    # Historic Production -----------------------------------------------------
    + geom_point(
        mapping=aes(
                    x='date', 
                    y=metric1, 
                    shape='Historic'
                ),
        color='firebrick',
        alpha=0.5,
        size=2.4,
        stroke=0.42
    )
    + scale_shape_manual(
        labels=['Gas'], 
        values=['o']
    )

    # Forecast Model ----------------------------------------------------------
    + geom_line(
        mapping=aes(
                    x='date', 
                    y=metric2, 
                    color='Forecast'
                ),
        alpha=0.77
    )
    + scale_color_manual(
        labels=['Model'], 
        values=['firebrick']
    )

    # Segments ----------------------------------------------------------------
      
    # Historic Start
    + geom_vline(
        xintercept=[hs, he], 
        linetype='dashed', 
        alpha=0.5
    )

    # Forecast Start
    + geom_vline(
        xintercept=fs,
        alpha=0.5, 
        color='red'
    )
      
    # Shaded fit/predict
    + geom_rect(
        data=fr,
        mapping=aes(
                    xmin='start', 
                    xmax='end', 
                    fill='color'
                ),
        ymin=float('-inf'), 
        ymax=float('inf')
    )
    + scale_fill_manual(
        name="Regions",
        labels=['Gap', 'Fit', 'Forecast', 'Gap'],
        values=['#4300642a', '#43ff642a', '#ffff642a', '#4300642a']
    )

    # Scales, Labels, Themes --------------------------------------------------
    + scale_x_datetime( 
        date_breaks='1 year', 
        date_labels="%Y",
        expand=(0, 0),
        limits=[hs - pd.DateOffset(months=6), he + pd.DateOffset(months=6)]
    )
    + labs(
        x='', 
        y=ylab, 
        title=t1+t2+t3+"\n", 
        subtitle=sec
    )   
    + theme_seaborn(
        style='darkgrid', 
        context='notebook', 
        font_scale=1
    )    
    + theme(
        axis_text_y=element_text(margin={'r': 0.15, 'units': 'in'}), # default
        axis_ticks_minor_x=element_blank(),
        figure_size=(19.2, 10.8),
        legend_justification_right="top",        
        legend_key=element_rect(fill='#ffffff00'),
        legend_text=element_text(margin={'l': 7, 'units': 'pt'}),
        panel_background=element_rect(fill='#01010111'),
        panel_grid_major_y=element_line(size=0.77, color='white'),
        panel_grid_minor_y=element_line(size=0.42, color='white'),
        plot_subtitle=element_text(size=9)
    )
  )
  # Log Scale -----------------------------------------------------------------
  if log == True:
    p = p + scale_y_log10(
              breaks=log_breaks,
              limits=log_limits,
              labels=label_comma
            ) 
    p = p + theme(
              axis_text_y=element_text(margin={'r': 0.06, 'units': 'in'})
            )
    
  else:
    p = p + scale_y_continuous(labels=label_comma)

  # Quality Plot --------------------------------------------------------------
  if quality == True:
    p = p + theme(
              axis_text_x=element_blank(),
              axis_ticks_major_x=element_blank(),
              axis_ticks_minor_x=element_blank(),
              axis_title_x=element_blank(),
              figure_size=(19.2, 10.8),   
              plot_margin_bottom=0
            )
  return p

def p9_qual(qual):
  """
  Plot forecast quality metrics.
  
  This plot will be optional and allows the user to view the quality of the 
  forecasts made over time.
  
  Parameters
  ----------
  qual : pd.DataFrame
         Dataframe containing quality data.

  Returns
  -------
    plotnine.ggplot.ggplot
  """
  q = (
    ggplot(qual)
    
    # Quality report ---------------------------------------------------------- 
    + geom_line(
        aes(
            x='date', 
            y='quality', 
            color='..y..'), 
        size=1, 
        show_legend=True
    )   
    + scale_color_gradient(
        high='green', 
        low='red'
    )
    + scale_x_datetime(
        date_breaks='1 year',
        date_labels="%Y",
        expand=(0, 0),
        limits=[hs - pd.DateOffset(months=6), he + pd.DateOffset(months=6)]
    )
    + scale_y_continuous(
        breaks = np.arange(0, 1.05, 0.05).tolist(),
        labels = lambda l: ["%d%%" % (v * 100) for v in l],
        minor_breaks = np.arange(0.9, 1, 0.01).tolist()
    )

    # Segments ----------------------------------------------------------------              

    # Historic Start
    + geom_vline(
        xintercept=[hs, he], 
        linetype='dashed', 
        alpha=0.5
    )

    # Forecast Start
    + geom_vline(
        xintercept=fs,
        alpha=0.5, 
        color='red'
    )

    # Shaded fit/predict
    + geom_rect(
        data=fr,
        mapping=aes(
                    xmin='start', 
                    xmax='end', 
                    fill='color'
                ),
        ymin=float('-inf'), 
        ymax=float('inf'), 
        alpha=0.15,
        show_legend=False
    )
    + scale_fill_manual(
        values=['#4300642a', '#43ff642a', '#ffff642a', '#4300642a']
    )

    # Labels, themes ----------------------------------------------------------
    + labs(
        x='', 
        y='Quality', 
        title=''
    )
    + theme_seaborn(
        style='darkgrid', 
        context='notebook', 
        font_scale=1
    )
    + theme(
        axis_text_y=element_text(margin={'r': 0.08, 'units': 'in'}), # default
        axis_ticks_minor_x=element_blank(),
        figure_size=(19.2, 10.8),
        legend_frame=element_rect(fill='#ffffff', size=77),
        legend_box_spacing=0.045,
        panel_background=element_rect(fill='#01010111'),
        panel_grid_major_y=element_line(size=0.77, color='white'),
        panel_grid_minor_y=element_line(size=0.42, color='white'),
        plot_margin_top=0
    )
  )
  return q  

# Shiny for Python ============================================================

app_ui = ui.page_navbar(
  ui.nav_panel(
    "Gas Forecast",
    ui.tags.style(".collapse-toggle { display: none !important; }"),
    ui.tags.style(".shiny-input-container .checkbox input { border: 2px solid black; }"),
    ui.tags.style(".shiny-input-select { border: 2px solid black; }"),
    ss.theme.lux,
    ui.layout_sidebar(
      ui.sidebar(
        ui.input_select("metrics", "Metrics:",
                        {"Production Rate": {"rate_hat": "Curve Projection"},
                         "Total Yield": {"cum_hat_pred": "EUR"},},),
        ui.input_checkbox("quality", "Quality Plot", True),
        ui.input_checkbox("log", "Log Scale", False),
        ui.input_checkbox("sec", "Sec.Axis(experimental)", False),
      ),
      ui.output_plot("plotnine", width="80%"), #, height='500px'),
      ui.output_plot("quality_plot", width="80%", height='142px'),
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
      
    elif input.metrics() == 'cum_hat_pred':
      metric1='cum'
      metric2='cum_hat_pred'
      
    sec = subtitle if input.sec() else ""
    
    return p9_main(qual, metric1, metric2, input.quality(), input.log(), sec)
  
  @output
  @render.plot()
  def quality_plot():
    if input.quality() == True:
      q = p9_qual(qual)
      
      if input.metrics() == 'cum_hat_pred':
        q = q + theme(
                  axis_text_y=element_text(margin={'r': 0.32, 'units': 'in'})
                )
        
    else:
      q = (ggplot() + theme_void())
    return q

app = App(app_ui, server) 
