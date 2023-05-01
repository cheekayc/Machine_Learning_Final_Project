Final Project
================
Chee Kay Cheong

``` r
knitr::opts_chunk$set(message = FALSE, warning = FALSE)

library(tidyverse)
library(caret)
library(glmnet)
library(randomForest)
```

# Introduction

Every year, the seasonal flu claims numerous lives across the globe.
This highly contagious virus can spread easily and quickly, causing
severe health complications, especially in people with weakened immune
systems. Vaccinations are essential to prevent illness, and if enough
people receive them, they can create herd immunity, which can help
protect the wider community.

In the spring of 2009, an outbreak of H1N1 influenza, commonly known as
“swine flu,” spread worldwide, causing a death toll estimated to range
from 151,000 to 575,000 in its first year. A vaccine for the H1N1 flu
virus became publicly available in October 2009. Later, in late 2009 and
early 2010, the United States conducted the National 2009 H1N1 Flu
Survey. This phone survey collected personal information from
participants and asked about their social, economic, and demographic
background, views on disease and vaccine efficacy, and steps taken to
reduce the spread of infection. Gaining insights into the relationship
between these characteristics and personal vaccination behavior can help
inform future public health initiatives.

### Research Question

The final project will employ a dataset acquired from
[Kaggle](https://www.kaggle.com/datasets/arashnic/flu-data) that
contains all the survey data gathered. We will use this dataset **to
identify the factors that may influence individuals’ likelihood of
receiving the H1N1 or seasonal flu vaccinations**.
