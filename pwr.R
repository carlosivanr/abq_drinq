ibrary(tidyverse)
library(rstatix)
library(car)



### Load Data ------------------------------------------------------------------
# Codebook 
codes <- read.csv("./data/codebook.csv")

# Drinq
drinq <- read.csv("./data/occ_alpha_data_v2.csv")


### Remove outliers IQR3 method ------------------------------------------------
pwr_outliers <- drinq %>%
  identify_outliers(peak_pwr) %>%
  select(ursi, peak_pwr)

# Not in function
`%notin%` <- Negate(`%in%`)

# Remove outliers
drinq <- drinq %>%
  filter(ursi %notin% pwr_outliers$ursi)


### Independent variables  -----------------------------------------------------
set_7d <- codes$ï..variable[2:12]
set_30d <- codes$ï..variable[14:24]
set_60d <- codes$ï..variable[26:36]
set_90d <- codes$ï..variable[38:48]

sets <- list(set_7d, set_30d, set_60d, set_90d)
#sets <- c("set_7d", "set_30d", "set_60d", "set_90d")


### High-through put analysis
for (i in 1:length(sets)){
  vars = sets[[i]]
  
  co_variates = c(vars[10:11], "age", "adjusted_age", "male", 
                  "depressive_disorder", "anxiety_disorder")
  outcomes = vars[1:9]
  
  for (j in 1:9){
    file_out <- paste("./results/pwr_", outcomes[j], ".txt", sep = "")
    sink(file = file_out, split = FALSE, append = FALSE)
    f <- as.formula(paste("peak_pwr", paste(c(outcomes[j], co_variates), collapse = " + "), sep = " ~ "))
    pwr <- lm(f, data = drinq)
    print("Call")
    print(f)
    print(summary(pwr))
    print("Variance Inflation Factor:")
    print(vif(pwr))
    sink()
  }
}
