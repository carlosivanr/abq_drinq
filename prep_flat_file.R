######### "!/usr/bin/env Rscript --vanilla" set up to run on server, but required 
######### will not run on server, packages not installed

# Carlos Rodriguez, MRN
# 12/15/2021
# Prep Occipital Alpha Data
# This script will compiles the MNE peak power, peak frequency, sex, age, 
# onset age of AUD, adjusted age, presence of depression/anxiety, and the TLFB
# data from abqdrinq datasets to produce a .csv file that can be used in statis-
# tical analyses

# Built with R Version 4.0.0 "Arbor Day"
# Requires tidyverse 1.3.0


# Check for system OS ----------------------------------------------------------
sysinf <- Sys.info()
if (sysinf["sysname"] == "Linux"){
  setwd("/export/research/analysis/human/jhouck/cobre06_65007/carlos_work/abqdrinq/occ_alpha")
}

# Load Libraries ---------------------------------------------------------------
library(tidyverse)

# Set files to load ------------------------------------------------------------
tlfb_file <- "./data/summary_tlfb_data_123120.txt"  #contains drinking measures
tlfb_long_file <- "./data/TLFB_data_long.csv"       #contains age and sex
scid_file <- "./data/abq_drinq_scid.csv"            #contains comorbidity disorders
mod_e_enrolled <- "./data/crossCollapse_2021-12-01_06-28-49_enrolled.csv"   # SCID Age of onset enrolled
mod_e_withdrawn <- "./data/crossCollapse_2021-12-01_06-32-12_withdrawn.csv" # SCID Age of onset withdrawn
alpha_file <- "./data/drinq_theta_alpha_jh.txt"     #contains mne output peak power and frequency
codes_file <- "./data/codebook.csv"                 #contains all of the tlfb codes for reference

# Set output fname ------------------------------------------------------------
output_name <- "./data/occ_alpha_data_v3.csv"

# Load files -------------------------------------------------------------------
tlfb <- read.delim(tlfb_file)

long_tlfb <- read.csv(tlfb_long_file)

scid <- read.csv(scid_file, 
                 header = TRUE)

alpha <- read.delim(alpha_file, 
                    header = FALSE, 
                    sep = "\t")

codes <- read.csv(codes_file)

onset_enrolled <- read.csv(mod_e_enrolled)

onset_withdrawn <- read.csv(mod_e_withdrawn)


# Process Alpha ----------------------------------------------------------------
# Subset columns
alpha <- alpha[c(2:7)]

# Rename columns and sort
alpha <- alpha %>% rename(ursi = V2, 
                          visit = V3,
                          proc_time = V4,
                          coreg_error = V5,
                          peak_freq = V6,
                          peak_pwr = V7) %>%
  arrange() 

# separate the M out of the ursi
alpha <- separate(alpha, ursi, c("M", "ursi"), sep = 1)

# Remove the M column
alpha <- alpha %>% select(-M)

# Set ursi as numeric
alpha$ursi <- as.numeric(alpha$ursi)


# Process TLFB long for age and sex --------------------------------------------
# Filter out the subjects
long_tlfb <- long_tlfb %>% filter(ursi %in% alpha$ursi) %>% arrange()

# Keep only the non duplicated rows by ursi
long_tlfb <- long_tlfb[!duplicated(long_tlfb$ursi),]

long_tlfb <- long_tlfb %>% select(ursi, age, male)

age_sex <- long_tlfb

rm(long_tlfb)


# Process SCID module E for age of onset ---------------------------------------
# Separate the M out of the URSI
onset_enrolled <- separate(onset_enrolled, queried_ursi, c("M", "ursi"), sep = 1)

onset_withdrawn <- separate(onset_withdrawn, queried_ursi, c("M", "ursi"), sep = 1)

onset_enrolled$ursi <- as.numeric(onset_enrolled$ursi)

onset_withdrawn$ursi <- as.numeric(onset_withdrawn$ursi)


# Check which if any of the SCID age of onset are in the drinq data set
onset_enrolled <- filter(onset_enrolled, ursi %in% alpha$ursi) %>%
  arrange()

onset_withdrawn <- filter(onset_withdrawn, ursi %in% alpha$ursi) %>%
  arrange()

# Bind rows
onset_age <- rbind(onset_enrolled, onset_withdrawn)

# Drop columns
onset_age <- subset(onset_age, select = c(ursi, Baseline_SCID_ModE_E19))

# Take care of missing data
# Columns that need to be cleaned
columns_to_clean <- c("Baseline_SCID_ModE_E19")

# Function to clean up condition skipped missing values
clean_up_Vals <- function(columns){
  idx <- which(colnames(onset_age)== columns)
  onset_age[,idx][onset_age[,idx] == "~<condSkipped>~"] <<- NA
  onset_age[,idx][onset_age[,idx] == "Missing data"] <<- NA
  onset_age[,idx][onset_age[,idx] == 99] <<- NA
  onset_age[,idx] <- as.numeric(onset_age[,idx])
}  

# Apply function to onset_age  
clean_up_Vals(columns_to_clean[1])

# Rename scid module e
onset_age <- onset_age %>% rename(onset_age = Baseline_SCID_ModE_E19) %>%
  arrange() 

# convert to numeric
onset_age$onset_age <- as.numeric(onset_age$onset_age)

# Remove unused data frames
rm(onset_enrolled, onset_withdrawn)


# Process SCID for depression/anxiety ------------------------------------------
# Separate the M out of the ursi
scid <- separate(scid, ï..queried_ursi, c("M", "ursi"), sep = 1)

scid$ursi <- as.numeric(scid$ursi)

# Filter SCID
scid <- filter(scid, ursi %in% alpha$ursi) %>%
  arrange()

# remove duplicate rows
scid <- scid[!duplicated(scid$ursi),]

# Columns that need to be cleaned for missing values condition skippeds
columns_to_clean <- c("Baseline_SCIDS_007A", 
                      "Baseline_SCIDS_032A",
                      "Baseline_SCIDS_034A", 
                      "Baseline_SCIDS_036A")

# Function to clean up values
clean_up_Vals <- function(arg_1){
  #print(arg_1)
  idx <- which(colnames(scid)==arg_1)
  scid[,idx][scid[,idx] == "~<condSkipped>~"] <<- 0
  scid[,idx][scid[,idx] == 1] <<- 0
  scid[,idx][scid[,idx] == -1001] <<- 0
  scid[,idx][scid[,idx] == 3] <<- 1
  scid[,idx] <- as.numeric(scid[,idx])
  #print(data[,idx])
}

# For loop to modify each colum  
for (i in 1:length(columns_to_clean)){
  print(columns_to_clean[i])
  clean_up_Vals(columns_to_clean[i])
}

# Select the columns for major depression and anxiety disorder
scid <- scid %>% select(ursi, all_of(columns_to_clean))

# Conver to numeric to sum and see which participants report a disorder
scid[columns_to_clean[1:4]] <- sapply(scid[columns_to_clean[1:4]], as.numeric)

# Sum anxiety columns by rows
scid <- scid %>% rowwise() %>% 
  mutate(anxiety = sum(c(Baseline_SCIDS_032A, 
                         Baseline_SCIDS_034A, 
                         Baseline_SCIDS_036A)))

# Rename column
scid <- scid %>% rename(depression = Baseline_SCIDS_007A)

# Drop columns
scid <- scid %>% select(ursi, depression, anxiety)


# Stitch frames together -------------------------------------------------------
drinq <- left_join(alpha, age_sex, by = "ursi")
drinq <- left_join(drinq, onset_age, by = "ursi")

# create the adjusted age variable
drinq <- drinq %>% rowwise() %>% mutate(adjusted_age = age - onset_age)

drinq <- left_join(drinq, scid, by = "ursi")
drinq <- left_join(drinq, tlfb, by = "ursi")



# Write data -------------------------------------------------------------------
write.csv(drinq, output_name)
