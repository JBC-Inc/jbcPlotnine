# run once
# install.packages("reticulate")
library(reticulate)

# run once
# reticulate::install_python(version = "3.12.3")

# Virtual Environments are created from another "starter" or "seed" Python already installed on the system.
# Suitable Pythons installed on the system are found by
# reticulate::virtualenv_starter()

# Create interface to Python Virtual Environment
reticulate::virtualenv_create(
  envname = "python-env",
  version = "3.12.3"
  )

# Select the version of Python to be used by reticulate
reticulate::use_virtualenv(
  virtualenv = "python-env",
  required = TRUE
  )

# lookat information about the version of Python currently being used by reticulate.
reticulate::py_config()

# Create interactive Python console within R.
# Objects created within Python are available for R session (and vice-versa).
reticulate::repl_python()




# qual <- readr::read_csv("./inst/data/qual.csv")
# library(tidyverse)
# glimpse(qual)
# qual$Historic <- ""
# qual$Forecast <- ""
# glimpse(qual)
# qual <- qual |> select(-c(Historic, Forecast))
# readr::write_csv(qual, "./inst/data/qual.csv")


# fr = pd.read_csv('./inst/data/rf.csv',
#                  parse_dates=['start', 'end'],
#                  dtype={'region': 'category'})
# fr['color'] = pd.Categorical([1, 2, 3, 4])


# fr <- readr::read_csv('./inst/data/rf.csv')
# fr
#
# fr <- fr |> dplyr::mutate(color = as.factor(c(1, 2, 3, 4)))
# fr <- fr |> dplyr::mutate(region = as.factor(region))
# fr <- fr |> dplyr::select(-color)
# fr
# readr::write_csv(fr, "./inst/data/fill_region.csv")
#
# fr





























