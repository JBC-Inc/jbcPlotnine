
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

This comprehensive data set encapsulates a detailed analysis of
historical and forecasted data for a natural gas well, meticulously
curated and organized for strategic decision making in the energy
sector.

Each entry in this pandas.DataFrame, comprising 123 observations, is a
testament to the meticulous monitoring and predictive modeling employed
in the realm of hydrocarbon exploration and production.

    Region: The designation of data points within the plot, categorized as
            either FIT or PREDICT, offering clarity regarding the origin of
            each observation and its respective utilization in historical
            analysis or future predictions.

    Date: A chronological record, timestamped in datetime64 format, which
          provide insight into historical production trends over time.

##### Production Metrics:

     Rate: instantaneous rate of natural gas production, measured in standard
           units, indicative of the well's productivity at one month time
           intervals.

     Rate_Hat: A prognostic indicator denoting the projected production rate
               derived from forecasting methodologies.

     Cum: The cumulative volume of natural gas extracted from the well since
          inception, a pivotal metric in assessing reservoir depletion and
          long-term yield potential.

     Cum Hat/Fit: Forecasts encompassing both fitted and predictive models
                  for cumulative production, enabling stakeholders to
                  anticipate future reservoir dynamics and optimize extraction
                  strategies accordingly.

##### Statistical Insights:

      Error: The deviation between actual and predicted production metrics,
             offering insights into the efficacy of forecasting methodologies
             and the inherent uncertainty associated with hydrocarbon
             extraction.

      Mean Squared Error: A quantitative measure of predictive accuracy,
                          facilitating rigorous performance evaluation and
                          model refinement.

      Quality Metrics: An aggregate assesment of data quality and modeling
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

This meticulously curated DataFrame amalgamates historical precedence
with forward-looking forecasts, empowering stakeholders with actionable
insights for strategic decision-making and operational optimization in
the dynamic landscape of natural gas exploration and production.

------------------------------------------------------------------------

#### Usage

Plotnine code was written within a virtual python environment utilizing
the `reticulate` package. The main plotnine code resides within:
`/R/plotnine.py`

There are helper functions that assist in plotnine generation, so this
script should be run beforehand. to load the environment. The actual
plotnine plots are located in `/R/plots.py`.

Enjoy!
