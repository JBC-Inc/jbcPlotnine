
# jbcPlotnine

### Overview

Official entry for the 2024 Plotnine Contest

### Installation

You can install jbcPlotnine from [GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("JBC-Inc/jbcPlotnine")
```

<table style="border-collapse: collapse; border: none;">
<tr style="vertical-align: top; border: none;">
<td style="vertical-align: top; border: none;">

#### Rules:

<ol>
<li>
    Technically impressive
</li>
<li>
    Well documented example
</li>
<li>
    Demonstrate novel, useful elements of plot design
</li>
<li>
    Aesthetically pleasing
</li>
</ol>
</td>
<td style="vertical-align: top; border: none;">
<img src="https://posit.co/wp-content/uploads/2024/05/Screenshot-2024-05-15-at-4.48.47%E2%80%AFPM.jpg" alt="POSIT" width="420">
</td>
</tr>
</table>

------------------------------------------------------------------------

#### DATA:

This data set encapsulates a detailed analysis of historical and
forecasted data for a natural gas well, organized for strategic decision
making in the energy sector.

Each entry in the pandas.DataFrame, with 123 observations, demonstrates
predictive modeling used in hydrocarbon exploration and production. The
original forecasts for decline and recovery were generated about 2 years
after production had began. The results shown are the actual production
8 years later, and how well the original forecasts held up over time,
observable within the quality plot.

    Region: The designation of data points within the plot, categorized as
            either FIT or PREDICT, offering clarity regarding the origin of
            each observation and its respective utilization in historical
            analysis or future predictions.

    Date: A chronological record, timestamped in datetime64 format, which
          provide insight into historical production trends over time.

##### Production Metrics:

     Rate: instantaneous rate of natural gas production, measured in standard
           units CCF (hundred cubic feet) indicative of the well's productivity 
           at one month time intervals.

     Rate_Hat: A prognostic indicator denoting the projected production rate
               derived from forecasting methodologies.

     Cum: The cumulative volume of natural gas extracted from the well since
          inception, a pivotal metric in assessing reservoir depletion and
          long-term yield potential.

     Cum Fit: Forecasts encompassing both fitted and predictive models
              for cumulative production, enabling stakeholders to
              anticipate estimated ultimate recovery reservoir dynamics and 
              optimize extraction strategies accordingly.

##### Statistical Insights:

      Error: The deviation between actual and predicted production metrics,
             offering insights into the efficacy of forecasting methodologies
             and the inherent uncertainty associated with hydrocarbon
             extraction.

      Mean Squared Error: A quantitative measure of predictive accuracy,
                          facilitating rigorous performance evaluation and
                          model refinement.

      Quality Metric: An aggregate assesment of data quality and modeling
                      efficacy, paramount in ensuring the reliability and
                      robustness of analytical insights.

      Drift Analysis: Examination of production drift, elucidating underlying
                      reservoir behavior and guiding reservoir management
                      strategies for sustained productivity.
                      

##### Historic and Forecasted Trends:

    Historical: Archival records capturing past production trends and
                operational benchmarks, serving as a foundational reference
                for trend analysis and predictive modeling.

    Forecasted: Projections and prognostications extrapolated from historical
                trends, empowering stakeholders with anticipatory insights
                into future production dynamics and resource depletion
                trajectories.

This DataFrame combines past data with future predictions. It gives
useful information to help make smart decisions, empowering stakeholders
with actionable insights for strategic decision-making and operations in
natural gas exploration and production.

------------------------------------------------------------------------

#### Usage

The plot was originally intended to be a stand alone python file but
many technical hurdles (mostly because plotnine is missing functionality
or supporting libraries present in ggplot2) prevented this so I decided
to integrate the plot(s) into an interactive Shiny for Python web app
hosted on [shinyapps.io](https://shinyapps.io). My first Python for
Shiny app!

To view the completed plot visit this link: [2024 PLOTNINE
CONTEST](https://kraggle.shinyapps.io/jcplotnine/)

Enjoy!
