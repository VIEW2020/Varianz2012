

library(fst)
library(data.table)
library(feather)

Baseline <- read.fst("sbar288/working/R_VARIANZ_2012_v1.fst", as.data.table = T)

DIM_FORM_PACK <- read.fst("common_lookups/CURRENT_PHARMS_LOOKUP_VIEW.fst")

drug.index <- c("chem_id", "form_id")

for(year in 2008:2012){
  
  files.list <- list.files("V:/source_data/R/PHARMS/SIMPLE/", pattern = paste0(year))
  
  PHH_YEAR <- rbindlist(lapply(files.list, function(x){
    
    dat <- read.fst(paste0("V:/source_data/R/PHARMS/SIMPLE/", x), as.data.table = T)
    dat <- dat[VSIMPLE_INDEX_MASTER %in% Baseline$VSIMPLE_INDEX_MASTER][DIM_FORM_PACK_SUBSIDY_KEY!=0]
    dat <- dat[DAYS_SUPPLY > 730, 
               DAYS_SUPPLY := 730][, SUPPLY_ENDATE := DATE_DISPENSED + DAYS_SUPPLY]
    dat <- dat[, (drug.index) := list(DIM_FORM_PACK$CHEMICAL_ID[match(DIM_FORM_PACK_SUBSIDY_KEY, DIM_FORM_PACK$DIM_FORM_PACK_SUBSIDY_KEY)],
                                      DIM_FORM_PACK$FORMULATION_ID[match(DIM_FORM_PACK_SUBSIDY_KEY, DIM_FORM_PACK$DIM_FORM_PACK_SUBSIDY_KEY)])]
    dat <- dat[,.(VSIMPLE_INDEX_MASTER, DATE_DISPENSED, SUPPLY_ENDATE, DAYS_SUPPLY, DIM_FORM_PACK_SUBSIDY_KEY, chem_id, form_id)]
    return(dat)
    
  }))
  
  write.fst(PHH_YEAR, paste0("sbar288/data/PHH_", year, ".fst"), compress = 75)
  write_feather(PHH_YEAR, paste0("sbar288/data/PHH_", year, ".feather"))
  
  print(paste(year, " completed")); rm(PHH_YEAR)
}

# Manipulate each year
files.list <- list.files("V:/sbar288/data/PHH/")

for(file in files.list){
  
  dat <- read.fst(paste0("V:/sbar288/data/PHH/", file), as.data.table = T)
  dat[, disp_month59 := 12*(year(DATE_DISPENSED) - 2008) + (month(DATE_DISPENSED) - 1)]
  dat[, end_month59 := 12*(year(SUPPLY_ENDATE) - 2008) + (month(SUPPLY_ENDATE) - 1)]
  
  dat <- dat[,.(VSIMPLE_INDEX_MASTER, chem_id, disp_month59, end_month59)]
  dat <- dat[, unique(.SD)]
  dat <- dat[, ntimes := (end_month59 - disp_month59) + 1]
  dat <- copy(as.data.table(lapply(dat, rep, dat$ntimes)))

  dat <- dat[, index := seq_len(.N), by = list(VSIMPLE_INDEX_MASTER, chem_id, disp_month59, end_month59)]
  dat[, index := index - 1]
  dat$dispmonth_index <- dat$disp_month59 + dat$index
  
  dat <- dat[,.(VSIMPLE_INDEX_MASTER, chem_id, dispmonth_index)][, unique(.SD)]
  dat <- dat[order(VSIMPLE_INDEX_MASTER, chem_id, dispmonth_index)]
  
  write.fst(dat, paste0("sbar288/data/", gsub(".fst", "", file), ".fst"), compress = 75)
  write_feather(dat, paste0("sbar288/data/", gsub(".fst", "", file), ".feather"))
}


# Admissions


# Hospitalisation

library(fst)
library(data.table)
library(feather)

Baseline <- read.fst("sbar288/working/R_VARIANZ_2012_v1.fst", as.data.table = T)

files.list <- list.files("V:/source_data/R/HOSPITALISATION/SIMPLE/", pattern = "EVENTS_2008|EVENTS_2009|EVENTS_2010|EVENTS_2011|EVENTS_2012")

for(file in files.list){
  
  year <- as.numeric(substr(file, 14, 17))
  
  event <- read.fst(paste0("V:/source_data/R/HOSPITALISATION/SIMPLE/", file), as.data.table = T)
  event <- event[VSIMPLE_INDEX_MASTER %in% Baseline$VSIMPLE_INDEX_MASTER][FAC_TYPE == 1, .(VSIMPLE_INDEX_MASTER, EVSTDATE, EVENDATE, EVENT_ID)]
  event <- event[year(EVSTDATE) >= 2008]
  
  diags <- read.fst(paste0("V:/source_data/R/HOSPITALISATION/SIMPLE/", gsub("EVENTS", "DIAGS", file)), as.data.table = T)
  
  diags <- diags[EVENT_ID %in% event$EVENT_ID][, .(EVENT_ID, CLIN_CD_10, DIAG_TYP)][, unique(.SD)]
  events <- merge(event, diags,
                  by = "EVENT_ID", 
                  all.y = T)
  
  events <- events[, EVENT_LENGTH := as.numeric(EVENDATE - EVSTDATE)][EVENT_LENGTH > 730,
                                                                      EVENT_LENGTH := 730]
  events <- events[EVENT_LENGTH == 730, 
                   EVENDATE := EVSTDATE + EVENT_LENGTH]
  
  events[, evst_month59 := 12*(year(EVSTDATE) - 2008) + (month(EVSTDATE) - 1)]
  events[, evend_month59 := 12*(year(EVENDATE) - 2008) + (month(EVENDATE) - 1)]
  
  events <- events[,.(VSIMPLE_INDEX_MASTER, CLIN_CD_10, DIAG_TYP, evst_month59, evend_month59)]
  events <- events[, unique(.SD)]
  events <- events[, ntimes := (evend_month59 - evst_month59) + 1]
  events <- copy(as.data.table(lapply(events, rep, events$ntimes)))
  
  events <- events[, index := seq_len(.N), by = list(VSIMPLE_INDEX_MASTER, CLIN_CD_10, DIAG_TYP, evst_month59, evend_month59)]
  events[, index := index - 1]
  events$eventmonth_index <- events$evst_month59 + events$index
  
  events <- events[,.(VSIMPLE_INDEX_MASTER, eventmonth_index, CLIN_CD_10, DIAG_TYP)][, unique(.SD)]
  events <- events[order(VSIMPLE_INDEX_MASTER, eventmonth_index)]
  
  write.fst(events, paste0("sbar288/data/HX_ADM_v2", year, ".fst"), compress = 75)
  write_feather(events, paste0("sbar288/data/HX_ADM_v2", year, ".feather"))
  
  print(paste(year, " completed")); rm(events)
  
}











