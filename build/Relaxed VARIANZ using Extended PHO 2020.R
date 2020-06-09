
# Relaxed VARIANZ using Extended PHO collection

# Last updated May 2020
# Created by Billy Wu

library(data.table)
library(fst)

`%fin%` <- function(x, table) {
        stopifnot(require(fastmatch))
        fmatch(x, table, nomatch = 0L) > 0L
}

NHI <- read.fst("V:/source_data/R/NHI/VSIMPLE_NHI_LOOKUP_AUG2019.fts",
                as.data.table = T)

pho.date     <- "MAX_CONTACT_DATE"
gms.date     <- "VISIT_DATE"
lab.date     <- "VISIT_DATE"
nmds.date    <- "EVENDATE"
pharms.date  <- "DATE_DISPENSED"
nap.date     <- "service_date" 

source.vars   <- c("pho", "gms", "lab", "nmds", "pharms", "nap")
all.date.vars <- paste0(source.vars, "_date")

for(year in 2012:2018){
 
 # i.  Bring in source data for each year
 PHO     <- read.fst(paste0("V:/source_data/R/PHO/SIMPLE/YEARLY/EXTENDED_PHO_", year, ".fst"), as.data.table = T)
 GMS     <- read.fst(paste0("V:/source_data/R/GMS/SIMPLE/VSIMP_GSM_", year, ".fst"), as.data.table = T)
 LAB     <- read.fst(paste0("V:/source_data/R/LAB/SIMPLE/YEARLY/VSIMP_LAB_", year, ".fst"), as.data.table = T)
 NMDS    <- read.fst(paste0("V:/source_data/R/HOSPITALISATION/SIMPLE/VSIMP_EVENTS_", year, ".fst"), as.data.table = T)
 PHARMS  <- read.fst(paste0("V:/source_data/R/PHARMS/SIMPLE/YEARLY/VSIMP_PHARMS_", year, ".fst"), as.data.table = T)
 NAP     <- read.fst(paste0("V:/source_data/R/NAP/VSIMP_NAP_", year, ".fst"), as.data.table = T) 
 
 VARIANZ <- NHI[VSIMPLE_INDEX_MASTER %fin% unique(c(PHO$VSIMPLE_INDEX_MASTER,
                                                    GMS$VSIMPLE_INDEX_MASTER,
                                                    LAB$VSIMPLE_INDEX_MASTER,
                                                    NMDS$VSIMPLE_INDEX_MASTER,
                                                    PHARMS$VSIMPLE_INDEX_MASTER,
                                                    NAP$VSIMPLE_INDEX_MASTER))]
 
 # Remove duplicates
 VARIANZ <- VARIANZ[VARIANZ[, .I[which.max(LAST_UPDATED_DATE)], 
                            by = VSIMPLE_INDEX_MASTER]$V1]
 
 # Capture last contact
 for(var in source.vars){
  
  DAT       <- get(toupper(var))
  date.var  <- get(paste0(var, ".date"))
  new.dtvar <- paste0(var, "_date")
  
  setnames(DAT, date.var, "Date")
  
  DAT <- DAT[year(Date) == year]
  DAT <- DAT[DAT[, .I[which.max(Date)], 
                 by = VSIMPLE_INDEX_MASTER]$V1]
  
  VARIANZ[, (new.dtvar) := DAT$Date[match(VSIMPLE_INDEX_MASTER, DAT$VSIMPLE_INDEX_MASTER)]]
  rm(DAT)
  
 }
 
 VARIANZ[, LAST_CONTACT_DATE := as.Date(apply(.SD, 1, function(i)
  
  if(all(is.na(i))){
   NA
  } else {
   max(i, na.rm = T)
  }),
  origin = "1970-01-01"), 
  .SDcols = all.date.vars]

 # Plausible age
 # 0 - 110
 study.index <- as.Date(paste(year, "12", "31", sep="-"))
 
 VARIANZ <- VARIANZ[, EN_AGE :=  as.numeric(floor((study.index - EN_DOB) / 365.25))][EN_AGE >= 0 & EN_AGE <= 110]
 
 write.fst(VARIANZ, paste0("V:/source_data/R/VARIANZ/EXTENDED_VARIANZ_", year, ".fst"), 75)
 
 print(paste0(year, " completed"))
 
}



