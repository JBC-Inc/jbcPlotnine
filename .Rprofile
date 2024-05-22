source("renv/activate.R")
setHook("rstudio.sessionInit", function(newSession) {

  console_width <- options()$width
  leading_spaces <- floor((80 - 29) / 2)

  cat(paste0("\n"))

  cat(paste0(
    ransid::col2fg('darkgreen'),
    ransid::col2bg('chartreuse'),
    strrep(" ", leading_spaces),
    "2024 PLOTNINE CONTEST - POSIT"),
    strrep(" ", leading_spaces),
    "\n")

  cat(paste0(ransid::reset_code))

  image <- magick::image_read("./inst/www/p9.jpg")
  cat(ransid::im2ansi(image))

  cat(paste0(ransid::reset_code))

}, action = "append")
