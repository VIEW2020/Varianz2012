
# VARIANZ PHO Data Componets

# last updated May 2020
# Created by Billy Wu

# Two versions: Strict vs relaxed cohort definitions
# Strict = used for baseline making
# Relaxed = used for follow up evaluation

library(data.table)
library(fst)

# DATA
files.path <- "V:/source_data/R/PHO/SIMPLE/"
files.list <- list.files(files.path, pattern = "2012|2013|2014|2015|2016|2017|2018|2019")

ALL_PHO <- rbindlist(lapply(files.list,
                            function(x)
                              read.fst(paste0(files.path, x), 
                                       as.data.table = TRUE)
                            ))

ALL_PHO[is.na(LAST_CONSULTATION_DATE),
        LAST_CONSULTATION_DATE := ENROLMENT_DATE]

ALL_PHO[is.na(ENROLMENT_DATE),
        ENROLMENT_DATE := LAST_CONSULTATION_DATE]

# > nrow(ALL_PHO)
# [1] 121686353
# > uniqueN(ALL_PHO$VSIMPLE_INDEX_MASTER)
# [1] 5340377
# sum(is.na(ALL_PHO$LAST_CONSULTATION_DATE))
# [1] 1705311
# sum(is.na(ALL_PHO$ENROLMENT_DATE))
# [1] 11

# ---- 1. Strict PHO Component ----
# Last consultation date is the ideal hard date (as it indicate the date of an actual visit)
# However, it has 1.4% missing. Therefore, fill missing with enrolment date (optional as enrolment data is combined eventually)

CONSULT <- ALL_PHO[year(LAST_CONSULTATION_DATE) %in% 2012:2018,
                   .(VSIMPLE_INDEX_MASTER, EN_DEPRIVATION_QUINTILE, EN_DHB, EN_MESHBLOCK, LAST_CONSULTATION_DATE)]

ENROL <- ALL_PHO[year(ENROLMENT_DATE) %in% 2012:2018,
                 .(VSIMPLE_INDEX_MASTER, EN_DEPRIVATION_QUINTILE, EN_DHB, EN_MESHBLOCK, ENROLMENT_DATE)]

# rm(ALL_PHO); gc()

setnames(CONSULT, "LAST_CONSULTATION_DATE", "DATE")
setnames(ENROL, "ENROLMENT_DATE", "DATE")

# Create Yearly PHO Contacts
for(year in 2012:2018){
  
  CONSULT.Y <- CONSULT[year(DATE) == year]
  ENROL.Y   <- ENROL[year(DATE) == year]
  
  PHO <- rbind(CONSULT.Y, ENROL.Y); rm(CONSULT.Y, ENROL.Y)
  
  PHO <- PHO[PHO[, .I[which.max(DATE)],
                 by = VSIMPLE_INDEX_MASTER]$V1]
  
  setnames(PHO, "DATE", "MAX_CONTACT_DATE")
  
  write.fst(PHO, paste0("V:/source_data/R/PHO/SIMPLE/YEARLY/STRICT_PHO_", year, ".fst"), 75); rm(PHO)
  
}

# rm(CONSULT, ENROL); gc()

# ---- 2. Extended (relaxed) Extended PHO Component ---
# Build a "relaxed" VARIANZ for 2012:2018
# This is the typical method for a non-strict VARIANZ Cohort
# NB: Data in subsequent year is evaluated - added to prior year of PHO,
#     IF consult date / enrolment year falls in the prior year
`%fin%` <- function(x, table) {
  stopifnot(require(fastmatch))
  fmatch(x, table, nomatch = 0L) > 0L
}

for(year in 2012:2018){
  
  year.qrts <- as.numeric(paste(year, 1:4, sep = "."))
  
  next.year <- as.numeric(
    if(year == 2018){
    paste(year + 1, 1:2, sep = ".")
  } else {
    paste(year + 1, 1:4, sep = ".")
  })
  
  PHO.Y <- ALL_PHO[CALENDAR_YEAR_AND_QUARTER %fin% year.qrts]
  PHO.Z <- ALL_PHO[CALENDAR_YEAR_AND_QUARTER %fin% next.year]
  
  PHO.Z <- PHO.Z[!VSIMPLE_INDEX_MASTER %fin% PHO.Y$VSIMPLE_INDEX_MASTER]
  PHO.Z <- PHO.Z[year(LAST_CONSULTATION_DATE) == year | year(ENROLMENT_DATE) == year]
  
  PHO <- rbind(PHO.Y, PHO.Z); rm(PHO.Y, PHO.Z)
  
  # Make sure the max contact date is within the year of interest!
  # i.e. latest contact during the PHO year. Evalute both dates and find date nearest to year-12-31 
  PHO[year(LAST_CONSULTATION_DATE) != year,
      LAST_CONSULTATION_DATE := ENROLMENT_DATE]
  
  PHO[year(ENROLMENT_DATE) != year,
      ENROLMENT_DATE := LAST_CONSULTATION_DATE]
  
  PHO[, MAX_CONTACT_DATE := as.Date(ifelse(LAST_CONSULTATION_DATE >= ENROLMENT_DATE, 
                                           LAST_CONSULTATION_DATE, 
                                           ENROLMENT_DATE), origin = "1970-01-01")]
  
  PHO <- PHO[PHO[, .I[which.max(MAX_CONTACT_DATE)],
                 by = VSIMPLE_INDEX_MASTER]$V1]
  
  PHO <- PHO[, .(VSIMPLE_INDEX_MASTER, EN_DEPRIVATION_QUINTILE, EN_DHB, EN_MESHBLOCK, MAX_CONTACT_DATE, CALENDAR_YEAR_AND_QUARTER)]
  
  write.fst(PHO, paste0("V:/source_data/R/PHO/SIMPLE/YEARLY/EXTENDED_PHO_", year, ".fst"), 75); rm(PHO)
  
  # write.fst(PHO, paste0("V:/source_data/R/VARIANZ/Components/EXTENDED_PHO_", year, ".fst"), 75); rm(PHO)
  print(paste(year, "completed"))
}

