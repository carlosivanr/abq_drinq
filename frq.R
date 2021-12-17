library(tidyverse)
library(rstatix)
library(MASS)
library(nnet)
library(brant)

### Load Data ------------------------------------------------------------------
# Codebook 
codes <- read.csv("./data/codebook.csv")

# Drinq
drinq <- read.csv("./data/occ_alpha_data_v2.csv")


# Remove individuals with zero peak frequency ----------------------------------
frq <- drinq %>%
  filter(peak_freq > 0)

# Round peak frequency values
frq$peak_freq <- round(frq$peak_freq)

# 8 Hz is a problem because there's only one participant
frq <- frq[frq$peak_freq != 8,]

# 13 Hz is a problem because there's only one participant
frq <- frq[frq$peak_freq != 13,]

# Convert to factor
frq$peak_freq <- as.factor(frq$peak_freq)

# set reference level
frq$peak_freq <- relevel(frq$peak_freq, ref = "4")



### Independent variables  -----------------------------------------------------
set_7d <- codes$ï..variable[2:13]
set_30d <- codes$ï..variable[14:25]
set_60d <- codes$ï..variable[26:37]
set_90d <- codes$ï..variable[38:49]

sets <- list(set_7d, set_30d, set_60d, set_90d)

### High throughput analysis --------------------------------------------------
# Loop through the different sets of variables at either 7day, 30day, 60day,
# or 90day TLFB variables
for (i in 1:length(sets)){
  vars = sets[[i]] # loop through sets of 7,30,60,and 90 day TLFB
  
  
  outcomes = vars[1:9]
  
  for (j in 1:9){
    file_out <- paste("./results/frq_", outcomes[j], ".txt", sep = "")
    sink(file = file_out, split = FALSE, append = FALSE)
    
    # multinomial regression -------------------------------
    # https://stats.idre.ucla.edu/r/dae/multinomial-logistic-regression/
    
    # propthc, depression, and anxiety produce NaNs
    co_variates = c(vars[10], "age", "male")
    f <- as.formula(paste("peak_freq", paste(c(outcomes[j], co_variates), collapse = " + "), sep = " ~ "))
    cat("Multinomial regression")
    cat("\n")
    cat("Formula: ")
    print(f)
    cat("\n")
    
    # Create Model
    freq <- multinom(f, data = frq) # multinomial regression
    cat("\n")
    
    # Print Summary
    print(summary(freq))
    
    # Z values
    z <- summary(freq)$coefficients/summary(freq)$standard.errors
    cat("\n")
    cat("Z-values:")
    cat("\n")
    print(round(z, 3))
    
    # P values
    p <- (1 - pnorm(abs(z), 0, 1)) * 2 # mean zero and std of 1, * 2 tailed
    cat("\n")
    cat("P-values:")
    cat("\n")
    print(round(p, 3))
    
    # Relative Risk ratios
    cat("\n")
    cat("Exponentiated Coefficients:")
    cat("\n")
    print(exp(coef(freq)))
    
    
    
    sink()
  }
}
