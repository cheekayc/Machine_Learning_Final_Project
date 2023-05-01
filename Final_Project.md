Final Project
================
Chee Kay Cheong

## Load needed packages

``` r
knitr::opts_chunk$set(message = FALSE, warning = FALSE)

library(tidyverse)
library(finalfit)
library(mice)
library(caret)
library(glmnet)
library(randomForest)
library(pROC)
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

### Rationale for research question

It is essential for public health to recognize the elements that affect
people’s probability to receive the H1N1 or seasonal flu shots for a
variety of reasons:

1.  **Improve disease prevention**
    - Understand what factors influence vaccine uptake can help public
      health officials target their efforts to increase vaccination
      rates and prevent outbreaks.
2.  **Promote health equity**
    - Identifying factors that influence vaccine uptake can help address
      health disparities and ensure that everyone has access to
      vaccines.
    - If certain groups are less likely to get vaccinated due to access
      barriers or mistrust of vaccines, public health officials can work
      to address these issues and increase vaccination rates in those
      populations.
3.  **Reduce economic impact**
    - The flu can have a significant economic impact due to lost of
      productivity, healthcare costs, and other related expenses.
      Increasing vaccination rates and preventing outbreaks can help
      reduce these economic costs and benefit society as a whole.

# Data Preparation

##### Read in and clean dataset.

- Convert all character variables to factor variables.
- Convert the binary outcomes `h1n1_vaccine` and `seasonal_vaccine` from
  numeric to factor variables.

``` r
flu_df = read_csv("./H1N1_Flu_Vaccines.csv") %>% 
  janitor::clean_names() %>% 
  select(-respondent_id) %>% 
  mutate(
    age_group = as_factor(age_group),
    education = as_factor(education),
    race = as_factor(race),
    sex = as_factor(sex),
    income_poverty = as_factor(income_poverty),
    marital_status = as_factor(marital_status),
    rent_or_own = as_factor(rent_or_own),
    employment_status = as_factor(employment_status),
    hhs_geo_region = as_factor(hhs_geo_region),
    census_msa = as_factor(census_msa),
    employment_industry = as_factor(employment_industry),
    employment_occupation = as_factor(employment_occupation),
    h1n1_vaccine = as_factor(h1n1_vaccine),
    seasonal_vaccine = as_factor(seasonal_vaccine))
```

##### Examine missing data.

``` r
missing_plot(flu_df)
```

![](Final_Project_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

Based on the plot above, `health_insurance`, `employment_industry`, and
`employment_occupation` have the most number of missing values. We could
simply remove these three variables, but this may violate the purpose of
our research question. Simply removing these variables with missing
values may introduce bias to our results.

# Impute missing data

Impute missing data using *random forest imputation*.

``` r
impute_missing_data = mice(flu_df, meth = "rf", ntree = 5, m = 2, seed = 100)
flu_nomiss = complete(impute_missing_data)
```

Examine the “new” dataset.

``` r
missing_plot(flu_nomiss)
```

![](Final_Project_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

Based on the plot above, there is no more missing values in the new
dataset `flu_nomiss`. Next, we will perform *Elastic Net* to reduce the
dimension of the dataset by removing features that are not as important
and significant for seasonal and H1N1 vaccines.

# Data Partitioning

We will partition the data into a 70/30 split separately for the two
different dependent variables: `h1n1_vaccine` and `seasonal_vaccine`.

##### H1N1 Vaccines

To understand what variables are responsible for influencing people’s
probability of receiving the H1N1 vaccines, we will remove the variable
`seasonal_vaccine` from this dataset before partitioning.

``` r
set.seed(123)

h1n1 = flu_nomiss %>% 
  select(-seasonal_vaccine)

h1n1.train.index = createDataPartition(y = h1n1$h1n1_vaccine, p = 0.7, list = FALSE)
h1n1.train.data = h1n1[h1n1.train.index, ]
h1n1.test.data = h1n1[-h1n1.train.index, ]
```

##### Seasonal Flu vaccines

To understand what variables are responsible for influencing people’s
probability of receiving the seasonal flu vaccines, we will remove the
variable `h1n1_vaccine` from this dataset before partitioning.

``` r
set.seed(123)

seasonal = flu_nomiss %>% 
  select(-h1n1_vaccine)

seasonal.train.index = createDataPartition(y = seasonal$seasonal_vaccine, p = 0.7, list = FALSE)
seasonal.train.data = seasonal[seasonal.train.index, ]
seasonal.test.data = seasonal[-seasonal.train.index, ]
```

# Feature Selection using Elastic Net

The dataset includes 35 independent variables. To address our research
question, it is necessary to conduct dimension reduction and select the
most relevant features that have the greatest impact on our outcome
variable.

#### H1N1 vaccines

Fit model into H1N1 training set.

``` r
set.seed(123)

# Create vectors of lambda and alpha
lambda = 10^seq(-3, 3, length = 100)
alpha = 0.1*seq(1, 10, length = 10)
tune_grid = expand.grid(alpha = alpha, lambda = lambda)

# Set validation method and options
control.settings = trainControl(method = "cv", number = 10)

# Fit model 
h1n1_model = train(h1n1_vaccine ~ ., data = h1n1.train.data, method = "glmnet", trControl = control.settings, preProcess = c("center", "scale"), tuneGrid = tune_grid)

# Output best values of alpha & lambda
h1n1_model$bestTune
```

    ##   alpha      lambda
    ## 6   0.1 0.002009233

Examine model coefficients.

``` r
coef(h1n1_model$finalModel, h1n1_model$bestTune$lambda)
```

    ## 94 x 1 sparse Matrix of class "dgCMatrix"
    ##                                                    s1
    ## (Intercept)                             -1.8050360006
    ## h1n1_concern                            -0.0968645994
    ## h1n1_knowledge                           0.0818986251
    ## behavioral_antiviral_meds                0.0483465397
    ## behavioral_avoidance                     0.0057763494
    ## behavioral_face_mask                     0.0392215698
    ## behavioral_wash_hands                    0.0058868775
    ## behavioral_large_gatherings             -0.0777444905
    ## behavioral_outside_home                 -0.0250806702
    ## behavioral_touch_face                    .           
    ## doctor_recc_h1n1                         0.7940265437
    ## doctor_recc_seasonal                    -0.2412061766
    ## chronic_med_condition                    0.0424905175
    ## child_under_6_months                     0.0763769905
    ## health_worker                            0.1688611118
    ## health_insurance                         0.1370582697
    ## opinion_h1n1_vacc_effective              0.6078847763
    ## opinion_h1n1_risk                        0.4461569790
    ## opinion_h1n1_sick_from_vacc             -0.0043158062
    ## opinion_seas_vacc_effective              0.1179316878
    ## opinion_seas_risk                        0.2043110899
    ## opinion_seas_sick_from_vacc             -0.0887267717
    ## age_group35 - 44 Years                  -0.1268154844
    ## age_group18 - 34 Years                  -0.1348799819
    ## age_group65+ Years                       0.0541323995
    ## age_group45 - 54 Years                  -0.1086533131
    ## education12 Years                        0.0486995069
    ## educationCollege Graduate                0.0850986884
    ## educationSome College                    0.0738359563
    ## raceBlack                               -0.0790924954
    ## raceOther or Multiple                    0.0290288619
    ## raceHispanic                             0.0002917278
    ## sexMale                                  0.0797940561
    ## income_poverty<= $75,000, Above Poverty -0.0581787277
    ## income_poverty> $75,000                 -0.0295790963
    ## marital_statusMarried                    0.0571771702
    ## rent_or_ownRent                         -0.0170728476
    ## employment_statusEmployed               -0.0239494550
    ## employment_statusUnemployed              0.0042966907
    ## hhs_geo_regionbhuqouqj                  -0.0211149222
    ## hhs_geo_regionqufhixun                  -0.0030065856
    ## hhs_geo_regionlrircsnp                  -0.0223678442
    ## hhs_geo_regionatmpeygn                  -0.0293364481
    ## hhs_geo_regionlzgpxyit                  -0.0757597557
    ## hhs_geo_regionfpwskwrf                  -0.0601346364
    ## hhs_geo_regionmlyzmhmf                   0.0129346187
    ## hhs_geo_regiondqpwygqj                  -0.0295138240
    ## hhs_geo_regionkbazzjca                  -0.0323797153
    ## census_msaMSA, Not Principle  City      -0.0446394573
    ## census_msaMSA, Principle City           -0.0143775880
    ## household_adults                         0.0150685164
    ## household_children                       .           
    ## employment_industryrucpziij             -0.0113583344
    ## employment_industrywxleyezf              0.0778836632
    ## employment_industrysaaquncn              0.0604674735
    ## employment_industryxicduogh              0.0888047863
    ## employment_industryldnlellj              0.0398073653
    ## employment_industrywlfvacwt              0.0142395541
    ## employment_industrynduyfdeo              0.0840191271
    ## employment_industryfcxhlnwr              0.1035092798
    ## employment_industryvjjrobsf             -0.0200432653
    ## employment_industryarjwrbjb              0.0935957229
    ## employment_industryatmlpfrs             -0.0262371664
    ## employment_industrymsuufmds             -0.0058069907
    ## employment_industryxqicxuve             -0.0057213808
    ## employment_industryphxvnwax              0.0335283607
    ## employment_industrydotnnunm             -0.0206263758
    ## employment_industrymfikgejo              0.0532426690
    ## employment_industrycfqqtusy              0.0056619347
    ## employment_industrymcubkhph              0.0196402884
    ## employment_industryhaxffmxo              0.0123556618
    ## employment_industryqnlwzans              0.0279771649
    ## employment_occupationxtkaffoo            0.0106310538
    ## employment_occupationemcorrxb            0.0509969778
    ## employment_occupationvlluhbov            0.0146508920
    ## employment_occupationxqwwgdyp            0.0498468007
    ## employment_occupationccgxvspp           -0.0280149100
    ## employment_occupationqxajmpny           -0.0771817579
    ## employment_occupationkldqjyjy            0.0443321512
    ## employment_occupationmxkfnird            0.0261067973
    ## employment_occupationhfxkjkmi           -0.0523018405
    ## employment_occupationbxpfxfdn            0.0306160726
    ## employment_occupationukymxvdu            .           
    ## employment_occupationcmhcxjea            0.0892379868
    ## employment_occupationhaliazsg            0.0084909012
    ## employment_occupationdlvbwzss            0.0365974146
    ## employment_occupationxzmlyyjv            0.0047460544
    ## employment_occupationoijqvulv            0.0079547712
    ## employment_occupationrcertsgn           -0.0243362019
    ## employment_occupationtfqavkke           -0.0044761022
    ## employment_occupationhodpvpew            0.0183612686
    ## employment_occupationuqqtjvyb           -0.0085685592
    ## employment_occupationpvmttkik            0.0041387946
    ## employment_occupationdcjcmpih            0.1697796919

We will remove variables that have an average coefficient values that
are smaller than \|0.05\| as they are not as important for predicting
the probability of receiving H1N1 vaccines.

Based on the model coefficients, we would remove `behavioral_avoidance`,
`behavioral_face_mask`, `behavioral_wash_hands`,
`behavioral_outside_home`, `behavioral_touch_face`,
`chronic_med_condition`, `opinion_h1n1_sick_from_vacc`,
`income_poverty`, `rent_or_own`, `employment_status`, `hhs_geo_region`,
`census_msa`, `household_adults`, `household_children`,
`employment_industry`, and `employment_occupation`.

Apply final model in the test set.

``` r
h1n1_test_outcome = predict(h1n1_model, h1n1.test.data)

confusionMatrix(h1n1_test_outcome, h1n1.test.data$h1n1_vaccine, positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 5979 1006
    ##          1  330  696
    ##                                           
    ##                Accuracy : 0.8332          
    ##                  95% CI : (0.8249, 0.8413)
    ##     No Information Rate : 0.7875          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.4171          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ##                                           
    ##             Sensitivity : 0.40893         
    ##             Specificity : 0.94769         
    ##          Pos Pred Value : 0.67836         
    ##          Neg Pred Value : 0.85598         
    ##              Prevalence : 0.21246         
    ##          Detection Rate : 0.08688         
    ##    Detection Prevalence : 0.12807         
    ##       Balanced Accuracy : 0.67831         
    ##                                           
    ##        'Positive' Class : 1               
    ## 
