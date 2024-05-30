# install.packages("reticulate")
library(reticulate)

# Run once ======================================================================
# reticulate::install_python(version = "3.12.3", force = TRUE)
#
# Virtual Environments are created from another "starter" or "seed" Python already installed on the system.
# Suitable Pythons installed on the system are found by
# reticulate::virtualenv_starter()
#
# Create interface to Python Virtual Environment
# reticulate::virtualenv_create(
#   force = TRUE,
#   envname = "python-env",
#   version = "3.12.3"
#   )
#
# Install packages (once)
# reticulate::virtualenv_install(envname = "python-env", packages = c("datetime"))
# reticulate::virtualenv_install(envname = "python-env", packages = c("patchworklib"))
# reticulate::virtualenv_install(envname = "python-env", packages = c("shiny"))
# reticulate::virtualenv_install(envname = "python-env", packages = c("shinyswatch"))
# reticulate::virtualenv_install(envname = "python-env", packages = c("shinywidgets"))
# reticulate::virtualenv_install(envname = "python-env", packages = c("plotnine"))

# lookat information about the version of Python currently being used by reticulate.
# reticulate::py_config()

# Select the version of Python to be used by reticulate
reticulate::use_virtualenv(
  virtualenv = "python-env",
  required = TRUE
)

# Create interactive Python console within R.
# Objects created within Python are available for R session (and vice-versa).
reticulate::repl_python()




# qual ------------------------------------------------------------------------
# qual <- readr::read_csv("./inst/data/qual.csv")
# library(tidyverse)
# glimpse(qual)
# qual$Historic <- ""
# qual$Forecast <- ""
# glimpse(qual)
# qual <- qual |> select(-c(Historic, Forecast))
# readr::write_csv(qual, "./inst/data/qual.csv")

# format region ---------------------------------------------------------------
# fr = pd.read_csv('./inst/data/rf.csv',
#                  parse_dates=['start', 'end'],
#                  dtype={'region': 'category'})
# fr['color'] = pd.Categorical([1, 2, 3, 4])
# fr <- readr::read_csv('./inst/data/rf.csv')
# fr <- fr |> dplyr::mutate(color = as.factor(c(1, 2, 3, 4)))
# fr <- fr |> dplyr::mutate(region = as.factor(region))
# fr <- fr |> dplyr::select(-color)
# fr
# readr::write_csv(fr, "./inst/data/fill_region.csv")
#
# dummy data frame ------------------------------------------------------------
# ddf <- data.frame(tick_labels = c(0, 275, 640, 1005, 1371, 1736, 2101, 2466, 2832, 3197, 3562),
#                   tick_locations = c(as.POSIXct("2009-04-01 00:00:00"),
#                                      as.POSIXct("2010-01-01 00:00:00"),
#                                      as.POSIXct("2011-01-01 00:00:00"),
#                                      as.POSIXct("2012-01-01 00:00:00"),
#                                      as.POSIXct("2013-01-01 00:00:00"),
#                                      as.POSIXct("2014-01-01 00:00:00"),
#                                      as.POSIXct("2015-01-01 00:00:00"),
#                                      as.POSIXct("2016-01-01 00:00:00"),
#                                      as.POSIXct("2017-01-01 00:00:00"),
#                                      as.POSIXct("2018-01-01 00:00:00"),
#                                      as.POSIXct("2019-01-01 00:00:00")))
#
# readr::write_csv(ddf, "./inst/data/ddf.csv")



