
# Sabastiano / Suneela VARIANZ 2012 Population v2

# Updated April,2020
# Created by Billy Wu

# v2 update (begins at section F):
# Flags people who were PHO enroled (using extended version) during index year
# Flags people who are in VARIANZ (for the year) during the next 5 years of follow-up

# Criteria:
# Age as at 2012-12-31

# Libraries
library(data.table)
library(fst)
library(dplyr)

# ---- A. Capture Cohort Base ----

# Cohort Year
year <- 2012

# Import
VARIANZ <- readRDS(paste0("source_data/R/VARIANZ/VARIANZ_EN_MB_", year, ".rds")); setDT(VARIANZ)

# 1.  Unify secondary to master ENHI
#     nb: this processes uses the latest NHI master / secondary mapping: June 2019
BOTHTABLE <- read.fst("source_data/R/ENHI BOTH TABLES/VIEW_VSIMP_MOH__PS_AUG2019.fst", as.data.table = T)
BOTHTABLE <- BOTHTABLE[,.(VSIMPLE_INDEX_MASTER, VIEW_ENHI_2NDARY)]

setkey(VARIANZ, VIEW_ENHI_MASTER)
setkey(BOTHTABLE, VIEW_ENHI_2NDARY)

VARIANZ <- BOTHTABLE[VARIANZ, nomatch = 0] # Starting 3972777

#     Remove duplicates
VARIANZ <- VARIANZ[order(last_updated_date, decreasing = T)
                   , index := seq_len(.N)
                   , by= "VSIMPLE_INDEX_MASTER"][index==1, .(VSIMPLE_INDEX_MASTER, en_meshblock)] #3972833


# 2.  Demography - Use latest NHI
NHI <- read.fst("source_data/R/NHI/VSIMPLE_NHI_LOOKUP_AUG2019.fts", as.data.table = TRUE)

#     Remove people with missing enriched MBs
VARIANZ <- VARIANZ[!is.na(VARIANZ$en_meshblock)] # -379 / 3972398

#     Add demo data
VARIANZ <- merge(VARIANZ, NHI,
                 by.x="VSIMPLE_INDEX_MASTER", by.y="VSIMPLE_INDEX_2NDARY",
                 all.x=T)

names(VARIANZ)[-1] <- tolower(names(VARIANZ)[-1])


# 3.  Age at 2012-12-31
# Filter age 30-74
study.index <- as.Date(paste(year, "12", "31", sep="-"))

COHORT <- VARIANZ[, nhi_age :=  as.numeric(floor((study.index - en_dob) / 365.25))][nhi_age >= 30 & nhi_age <=74]  # -1850129 / 2122269

COHORT <- COHORT[,.(VSIMPLE_INDEX_MASTER, nhi_age, en_meshblock, gender, en_dob, en_dod, en_prtsd_eth)]


# Make sure each individual is still alive on 2012-12-31
# i.e remove those who died before day 1 of follow-up
COHORT <- COHORT[is.na(en_dod) | en_dod > study.index] # -9875 / 2112394

# Meshblock data
MBLOOKUP <- readRDS("V:/common_lookups/MB NZDep/MB0613_NZDep0613_Concordance.rds")

# NZDep
COHORT$en_nzdep_q <- apply(COHORT, 1,
                            function(x){
                              px.mesh <- as.numeric(x["en_meshblock"])
                              dep.dec <- MBLOOKUP$NZDep2006[match(px.mesh, MBLOOKUP$MB_2006)]
                              dep.qui <- ifelse(dep.dec %in% c(1,3,5,7,9), 
                                                (dep.dec+1)/2, 
                                                dep.dec/2)
                              return(dep.qui)
                            })

Baseline <- setDT(COHORT)[!is.na(en_nzdep_q)] # Removes 3689

write.fst(Baseline, "sbar288/working/Baseline_EndPart_A.fst", compress = 75)


# ---- B. Hospitalisaton  ----

library(fst)

# Baseline <- read.fst("sbar288/working/Baseline_EndPart_A.fst", as.data.table = T)

files.list <- list.files("source_data/R/HOSPITALISATION/SIMPLE/", pattern="EVENTS")

for(file in files.list){

   EVENTS <- read.fst(paste0("source_data/R/HOSPITALISATION/SIMPLE/", file), as.data.table = T)
   EVENTS <- EVENTS[VSIMPLE_INDEX_MASTER %in% Baseline$VSIMPLE_INDEX_MASTER]

   file.diag <- gsub("EVENTS", "DIAGS", file)

   DIAGS <- read.fst(paste0("source_data/R/HOSPITALISATION/SIMPLE/", file.diag), as.data.table = T)
   DIAGS <- DIAGS[EVENT_ID %in% EVENTS$EVENT_ID, .(EVENT_ID, CC_SYS, DIAG_TYP, DIAG_SEQ, CLIN_CD, CLIN_CD_10)]

   if(file==files.list[1]){
      ALL_COHORT_EVENTS <- EVENTS
      ALL_COHORT_DIAGS <- DIAGS
   } else {
      ALL_COHORT_EVENTS <- rbind(ALL_COHORT_EVENTS, EVENTS)
      ALL_COHORT_DIAGS <- rbind(ALL_COHORT_DIAGS, DIAGS)
   }

   print(paste0(file, " completed")); rm(EVENTS, DIAGS)
}

# Ensure dates within range (1988 - 2018)
ALL_COHORT_EVENTS <- ALL_COHORT_EVENTS[EVSTDATE >= "1988-01-01" & EVSTDATE <= "2018-12-31"]

# Save
write.fst(ALL_COHORT_EVENTS, "sbar288/working/ALL_COHORT_EVENTS_v2.fst", 75)
write.fst(ALL_COHORT_DIAGS, "sbar288/working/ALL_COHORT_DIAGS_v2.fst", 75)

ALL_EVENTS <- read.fst("sbar288/working/ALL_COHORT_EVENTS_v2.fst", as.data.table=T)
ALL_DIAGS <- read.fst("sbar288/working/ALL_COHORT_DIAGS_v2.fst", as.data.table=T)

# Index Date
index.year <- 2012
index.date <- as.Date(paste(index.year, "12", "31", sep="-"))
fu.range <- 2013:2017

ALL_EVENTS <- ALL_EVENTS[, study_index_date := index.date][FAC_TYPE==1] # Last day of data collection = 2012-12-31 / set FAC_TYPE to 1


# 1.   Definitions

# CVD Clin Codes
VIEW_CVD_ICD <- read.fst("common_lookups/CURRENT_VIEW_CVD_LOOKUP.fst")
VIEW_DM_ICD   <- readxl::read_xlsx("V:/common_lookups/Definitions/VIEW_CVD_ICD10_II_28JUN18_Clean042.xlsx", sheet = 2)

# Retrieve ICD10 Codes for Events
cvd.var.names <- c("hx_broad_cvd", "hx_mi", "hx_unst_angina", "hx_other_chd", "hx_ischaemic_stroke", "hx_tia", "hx_other_cevd",
                   "hx_pci", "hx_cabg", "hx_other_chd_procs", "hx_pvd_diags", "hx_pvd_procs", "hx_haemorrhagic_stroke", "hx_heart_failure",
                   "out_broad_cvd", "out_mi", "out_unst_angina", "out_other_chd", "out_ischaemic_stroke", "out_tia", "out_other_cevd", 
                   "out_pvd_diags", "out_haemorrhagic_stroke", "out_heart_failure", 
                   "mortality_broad_cvd_with_other",
                   "out_pci_cabg", "out_pvd_procs")

for(var in cvd.var.names){
  
  var.codes <- VIEW_CVD_ICD$CLINICALCODE[which(eval(VIEW_CVD_ICD[,var])=="Y")]
  
  assign(gsub("_", ".", var), var.codes)
  
}

hx.diabetes    <- VIEW_DM_ICD$CLINICALCODE[which(VIEW_DM_ICD$hx_diabetes=="Y")]
out.diabetes   <- VIEW_DM_ICD$CLINICALCODE[which(VIEW_DM_ICD$out_diabetes=="Y")]

hx.af <- "^I48"


# -- 2.  All-time History --
HX_EVENTS <- copy(ALL_EVENTS)[EVSTDATE <= study_index_date]

history.vars <- ls(pattern = "^hx.")

source("V:/bwu009/My Projects/Global Procedures/Any History Procedure.R") # Run engine


# -- 3.  Five year Charlson CMI --
HX_EVENTS <- copy(ALL_EVENTS)[(EVSTDATE < study_index_date) & (EVSTDATE - study_index_date >= -1825)]

source("V:/bwu009/My Projects/Global Procedures/CMI Charlson 5year.R") # Run engine


# -- 4.  Outcomes (admissions) --
OUT_EVENTS <- copy(ALL_EVENTS)[year(EVSTDATE) %in% fu.range]

outcome.vars <- c("out.broad.cvd", "out.diabetes")

source("V:/bwu009/My Projects/Global Procedures/Any Outcome Procedure.R") # Run engine


# -- 5.  28 Day rule -- 
out.28d.vars <- c("out.mi", "out.unst.angina", "out.ischaemic.stroke", "out.haemorrhagic.stroke", "out.heart.failure", "out.other.chd",
                  "out.tia", "out.other.cevd", "out.pvd.diags", "out.pci.cabg", "out.pvd.procs")

# Add info - date of death
ALL_EVENTS[, "en_dod" := Baseline$en_dod[match(VSIMPLE_INDEX_MASTER, Baseline$VSIMPLE_INDEX_MASTER)]]

# Find events
# The following conditions are extremely important to include - ALL must be included!
#   - DOD less than "2017-12-31 (ensure all deaths occur within study period and not after)
#   - Event admission greater than index date
#   - Event admission is within 28 days of death
#   - Event diagnosis includes outcome of broad CVD and procedures
EVENTS_28D_PRIOR <- copy(ALL_EVENTS)[en_dod <="2017-12-31" & EVSTDATE > study_index_date][(EVSTDATE-en_dod >=-28) & (EVSTDATE <= en_dod)]

source("V:/bwu009/My Projects/Global Procedures/28day Rule & CVD-type Outcome Procedure.R")


# --- 6.  Outcome Episodes of Care Count --

# nb: some EOC counts are very high - these are likely administrative anomalies 
# 1.  Bundle Events
OUT_EVENTS <- copy(OUT_EVENTS)[order(VSIMPLE_INDEX_MASTER)
                               , ID := .GRP
                               , by=VSIMPLE_INDEX_MASTER][,.(ID, VSIMPLE_INDEX_MASTER, EVENT_ID, EVSTDATE, EVENDATE)]

setkey(OUT_EVENTS, EVSTDATE)

source("bwu009/My Projects/Generic Functions/Hospitalisation Bundling.R", keep.source = T)

# Apply function
OUT_EVENTS[, BUNDLE:=Bundle_Hosp_Dates(.SD)
           , by=ID]

# Capture bundle admissions / discharge
vars <- c("EOC_ADM_DATE", "EOC_DIS_DATE")

OUT_EVENTS[, by=list(VSIMPLE_INDEX_MASTER, BUNDLE)
           , (vars) := .(min(EVSTDATE)[1],
                         max(EVENDATE)[1])]

write.fst(OUT_EVENTS, "sbar288/working/OUT_EVENTS_BUNDLED_v2.fst", 75)

OUT_DIAGS <- copy(ALL_DIAGS)[EVENT_ID %in% OUT_EVENTS$EVENT_ID]

# 2.  Count total EOCs and EOCs containg CVD

# Total
TOT <- OUT_EVENTS[, out_total_eoc_count := uniqueN(BUNDLE)
                  , by = VSIMPLE_INDEX_MASTER][,.(VSIMPLE_INDEX_MASTER, out_total_eoc_count)]

# CVD
VIEW_CVD_ICD  <- readxl::read_xlsx("V:/common_lookups/Definitions/VIEW_CVD_ICD10_II_28JUN18_Clean042.xlsx")
out.broad.cvd <- VIEW_CVD_ICD$CLINICALCODE[which(VIEW_CVD_ICD$out_broad_cvd=="Y")]

CVD_EVENTS <- OUT_EVENTS[, out_event:= +(EVENT_ID %in% OUT_DIAGS$EVENT_ID[which(OUT_DIAGS$CLIN_CD %in% out.broad.cvd)])][out_event==1]

CVD <- CVD_EVENTS[, out_broad_cvd_eoc_count := uniqueN(BUNDLE)
                  , by = VSIMPLE_INDEX_MASTER][,.(VSIMPLE_INDEX_MASTER, out_broad_cvd_eoc_count)]

# Non-CVD (ie. Total EOCs - CVD EOCs)
ALL_EOC_VARS <- merge(unique(TOT), unique(CVD), 
                      by = "VSIMPLE_INDEX_MASTER", 
                      all.x=T)

ALL_EOC_VARS[, c("out_broad_cvd_eoc_count",
                 "out_non_cvd_eoc_count") := list(ifelse(is.na(out_broad_cvd_eoc_count), 0, out_broad_cvd_eoc_count),
                                                  out_total_eoc_count - out_broad_cvd_eoc_count)]

# nb: NA is not the same as 0! ie. NA = no EOC, Zero = EOC but 0 are CVD or 0 are non-CVD


# -- 7. VDR Diabetes History -- 

files.list <- list.files("source_data/R/VDR/SIMPLE/", pattern="VDR")[1:8]

for(file in files.list){
  
  VDR <- read.fst(paste0("source_data/R/VDR/SIMPLE/", file), as.data.table = T)
  VDR <- VDR[VSIMPLE_INDEX_MASTER %in% Baseline$VSIMPLE_INDEX_MASTER]
  
  VDR[, (file) := 1]
  
  ALL_VDR_VARS <- if(file == files.list[1]){
    
    merge(Baseline[,.(VSIMPLE_INDEX_MASTER)], VDR[,c("VSIMPLE_INDEX_MASTER", file), with = F],
          by = "VSIMPLE_INDEX_MASTER",
          all.x = T) 
    
  } else {
    
    merge(ALL_VDR_VARS, VDR[,c("VSIMPLE_INDEX_MASTER", file), with = F],
          by = "VSIMPLE_INDEX_MASTER",
          all.x = T)
    
  }
  
  # setnames(ALL_VDR_VARS, "hx_vdr_diabetes", )
  
  
  print(paste(file, " completed")); rm(VDR)
  
}

ALL_VDR_VARS[, hx_vdr_diabetes := apply(.SD, 1, function(x)
  +(any(x==1))), 
  .SDcols = files.list]

ALL_VDR_VARS[, hx_vdr_diabetes := +(!is.na(hx_vdr_diabetes))]

ALL_VDR_VARS <- ALL_VDR_VARS[, unique(.SD), .SDcols = c("VSIMPLE_INDEX_MASTER", "hx_vdr_diabetes")]


# -- 8.  Merge all to Baseline --
Baseline <- Reduce(function(...)
  merge(..., 
        by="VSIMPLE_INDEX_MASTER",
        all.x = T),
  list(Baseline, ALL_HX_VARS, ALL_CMI_VARS[,.(VSIMPLE_INDEX_MASTER, cmi_charlson_score_5yr, cmi_charlson_score_5yr_nodiab)], 
       ALL_OUT_VARS, ALL_28D_OUT_VARS[,.(VSIMPLE_INDEX_MASTER, out_fatal_28d_cvd_adm, out_fatal_28d_cvd_adm_typedate, out_fatal_28d_cvd_adm_type)],
       ALL_VDR_VARS, ALL_EOC_VARS))

# save
write.fst(Baseline, "sbar288/working/Baseline_EndPart_B.fst", 75)

rm(list=setdiff(ls(), c("Baseline", "mortality.broad.cvd.with.other")))


# ---- C.   Mortality ----

fu_range <- 2013:2017

ALL_DEATHS <- read_fst("V:/source_data/R/MORTALITY/VSIMPLE_MORTALITY_DEC2017_v2.fst", as.data.table=TRUE)

# 1.  CVD / non-CVD causes of death
# Define death codes and categories: Capture all ICD codes for CVD Deaths
COHORT_DEATHS <- ALL_DEATHS[VSIMPLE_INDEX_MASTER %in% Baseline$VSIMPLE_INDEX_MASTER & REG_YEAR %in% fu_range & ICDD!=""]

death.vars <- "mortality.broad.cvd.with.other"

source("V:/bwu009/My Projects/Global Procedures/Mortality Collection Procedure.R")

Baseline <- merge(Baseline, CVD_DEATHS,
                  by="VSIMPLE_INDEX_MASTER",
                  all.x=T)

# Make sure DODs do not exceed start or end of study!
Baseline[en_dod >= "2018-01-01", en_dod := NA]

sum(Baseline$en_dod > "2017-12-31", na.rm = T)
sum(Baseline$en_dod < "2013-01-01", na.rm = T)

# Complementary Information
vars <- c("mortality_uncoded", "mortality_all_cause")

Baseline[, (vars) := .(+(!is.na(en_dod) & mortality_broad_cvd_with_other==0 & mortality_non_cvd==0),
                       +(!is.na(en_dod)))]

# Make sure all death categories add up to total all-cause mortality
sum(Baseline$mortality_all_cause) == sum(Baseline$mortality_broad_cvd_with_other + Baseline$mortality_non_cvd + Baseline$mortality_uncoded)

# 2.  All cause of death subcategories - Underlying cause of death only
ICD_CATS <- readRDS("common_lookups/ICD 10 Mapping/ICD10_Categories.rds"); setDT(ICD_CATS)

dementia <- c("F00","F01","F02","F03","G30","G31")
copd <- c("J40","J41","J42","J43","J44")
lungcancer <- c("C33","C34")
colcancer <- c("C18","C19","C20","C21")
diabetes <- c("E10","E11","E12","E13","E14")
unintinj <- c(sprintf("V%02d", 0:99),
              sprintf("W%02d", 0:99),
              sprintf("X%02d", 0:59))

vars <- c("dementia", "copd", "lungcancer", "colcancer", "diabetes", "unintinj")

ICD_CATS[, (vars) := lapply(vars, function(x){
  +(subcategory %in% get(x))
})]


COHORT_DEATHS$subcategory <- substr(COHORT_DEATHS$ICDD, 1, 3)

COHORT_DEATHS <- merge(COHORT_DEATHS, ICD_CATS[,-"cat_desc", with=F],
                       by="subcategory",
                       all.x=T)
library(dplyr)

COHORT_DEATHS[, mortality_ud_label := dplyr::case_when(
  dementia == 1 ~ 1,
  copd == 1 ~ 2,
  lungcancer == 1 ~ 3,
  colcancer == 1 ~ 4,
  diabetes == 1 ~ 5,
  unintinj == 1 ~ 6,
  TRUE ~ 7)]

Baseline <- merge(Baseline, COHORT_DEATHS[,.(VSIMPLE_INDEX_MASTER, mortality_ud_label)],
                  by = "VSIMPLE_INDEX_MASTER",
                  all.x = T)

Baseline[is.na(mortality_ud_label),
         mortality_ud_label := ifelse(mortality_uncoded==1, 8, mortality_ud_label)]

# 3.  Improved Death

# Improve CVD deaths / non-CVD-death
Baseline[, imp_fatal_cvd := +(out_fatal_28d_cvd_adm==1 | mortality_broad_cvd_with_other==1)]
Baseline[, imp_fatal_non_cvd := +(imp_fatal_cvd==0 & mortality_non_cvd==1)]
Baseline[, imp_fatal_uncoded_death := +(imp_fatal_cvd==0 & imp_fatal_non_cvd==0 & mortality_uncoded==1)]  

# Check: Improved fatal variables should all add up to total all cause mortality
sum(Baseline$mortality_all_cause) == sum(Baseline$imp_fatal_cvd + Baseline$imp_fatal_non_cvd + Baseline$imp_fatal_uncoded_death)

# Save
write.fst(Baseline, "sbar288/working/Baseline_EndPart_C.fst", compress = 75)



# ---- D.  Pharmaceutical Dispensing ----- 

library(dplyr)

files.list <- list.files("V:/source_data/R/PHARMS/SIMPLE/", pattern = "08|09|10|11|12")

PHARMS <- rbindlist(lapply(files.list, function(x){
  
  dat <- read.fst(paste0("V:/source_data/R/PHARMS/SIMPLE/", x), as.data.table = T)
  return(dat[VSIMPLE_INDEX_MASTER %in% Baseline$VSIMPLE_INDEX_MASTER]); 
}))

# NB: Uses study_index_date as reference point for prior dispensing
index.date <- as.Date("2012-12-31")

PHARMS[, study_index_date := index.date]

## Definitions
DIM_FORM_PACK <- read.fst("V:/common_lookups/CURRENT_PHARMS_LOOKUP_VIEW.fst")

All.Mx.Groups <- c("antiplatelets", "anticoagulants", "lipid_lowering", "bp_lowering", "antidiabetes", "antianginals", "loopdiuretics")

# Capture Form Pack IDs (Lookup method)
for(class in All.Mx.Groups){
  
  class.codes <- eval(substitute(
    DIM_FORM_PACK$DIM_FORM_PACK_SUBSIDY_KEY[which(DIM_FORM_PACK[, class]==1)]
  ))
  
  assign(class, class.codes)  
  
}

# Flag drug categories
for(drug.name in All.Mx.Groups){
  
  drug.codes <- get(drug.name)
  
  # restrict to drug group of interest
  DISPENSES <- copy(PHARMS)[DIM_FORM_PACK_SUBSIDY_KEY %in% drug.codes]

  # Identify dispensing periods
  # i.  Prior
  if(drug.name %in% c("antianginals", "loopdiuretics")){
    
    DISPENSES[, prior_5yrs_3evts := +((DATE_DISPENSED - study_index_date >=-1825) & (DATE_DISPENSED <= study_index_date))]
    
    DISPENSES <- DISPENSES[prior_5yrs_3evts==1][, event_n := uniqueN(DATE_DISPENSED) #count dispensing occassions
                                                , by=VSIMPLE_INDEX_MASTER][event_n>=3, .(VSIMPLE_INDEX_MASTER, prior_5yrs_3evts)] #keep >=3 events
    
    PRIOR_DISP <- unique(DISPENSES)
    
  } else {
    
    DISPENSES[, prior_6mths := +((DATE_DISPENSED - study_index_date >=-182) & (DATE_DISPENSED <= study_index_date))]
    DISPENSES[, prior_1yr := +((DATE_DISPENSED - study_index_date >=-365) & (DATE_DISPENSED <= study_index_date))]
    
    # Find history of dispensing 1yr, 6mths in past [reduce to one person per row]
    PRIOR_DISP <- DISPENSES[prior_1yr==1][order(DATE_DISPENSED, decreasing = T)
                                          , index:=seq_len(.N)
                                          , by=VSIMPLE_INDEX_MASTER]
    
    PRIOR_DISP <- PRIOR_DISP[index==1, .(VSIMPLE_INDEX_MASTER, prior_6mths, prior_1yr)]
    
  }
  
  # Merge to baseline / turn to binary
  Baseline <- merge(setDF(Baseline), setDF(PRIOR_DISP),
                    by="VSIMPLE_INDEX_MASTER",
                    all.x=T)%>%
    setNames(gsub("^prior", paste0("ph_", drug.name, "_prior"), names(.))) %>% 
    as.data.table()
  
  # Progress
  print(paste0(drug.name, " completed"))
  rm(DISPENSES)
  
}

# Mutate each pharms variable to pure binary
phh.vars <- names(Baseline)[startsWith(names(Baseline), "ph")] 

Baseline[, (phh.vars) := lapply(.SD, function(x){
  replace(x, which(is.na(x)), 0)
}), .SDcols = phh.vars]


# Pharms quality check
# Ensure no record of dispensing of any drug > 28 days after date of death 
files.list <- list.files("V:/source_data/R/PHARMS/SIMPLE/")[33:56]

ALL_PHARMS <- rbindlist(lapply(files.list, 
                               function(x){
                                 DAT <- read.fst(paste0("V:/source_data/R/PHARMS/SIMPLE/", x), as.data.table = TRUE)
                                 return(DAT[VSIMPLE_INDEX_MASTER %in% Baseline$VSIMPLE_INDEX_MASTER])
                               }))

ALL_PHARMS[, dod := Baseline$en_dod[match(VSIMPLE_INDEX_MASTER, Baseline$VSIMPLE_INDEX_MASTER)]]

DOD_PHARMS <- ALL_PHARMS[!is.na(dod)]

# Capture last dispensing
DOD_PHARMS <- DOD_PHARMS[order(DATE_DISPENSED, decreasing = T), 
                         index := seq_len(.N), by="VSIMPLE_INDEX_MASTER"][index==1]

DOD_PHARMS <- DOD_PHARMS[, exclude := +(DATE_DISPENSED > dod+28)][exclude==1]

# Merge flag to Baseline & exclude
# Removes 692 people
Baseline[, phh_post_dod := DOD_PHARMS$exclude[match(VSIMPLE_INDEX_MASTER, DOD_PHARMS$VSIMPLE_INDEX_MASTER)]]

Baseline <- Baseline[phh_post_dod==0 | is.na(phh_post_dod)][, -"phh_post_dod", with=F] # removes 692, 2108013 remaining


write.fst(Baseline, "sbar288/working/Baseline_EndPart_D.fst", 75)


# ---- E.   End of followup ----

# Exclude history of CVD: -128816
Baseline <- Baseline[hx_broad_cvd==0]

# Exclude history of loopdiuretics (5 year 3x): -19090
Baseline <- Baseline[ph_loopdiuretics_prior_5yrs_3evts==0]

# End of FU
Baseline$end_fu_date <- as.Date(apply(Baseline[, c("en_dod", "out_broad_cvd_adm_date"), with = F], 1, function(x){
  
  xx <- as.Date(x)
  fu <- if(all(is.na(xx))){
    as.Date("2017-12-31")
  } else {
    as.Date(min(xx, na.rm = T))
  }
  
  return(as.Date(fu))
  
}), origin = "1970-01-01")

summary(Baseline$end_fu_date)

write.fst(Baseline, "sbar288/working/Baseline_EndPart_E.fst", 75)


# ---- F. VARIANZ & Index year PHO ----

library("fst", "data.table")

# Version 2 begins here
Baseline <- read.fst("sbar288/working/Baseline_EndPart_E.fst",
                     as.data.table = T)

PHO  <- read.fst("V:/source_data/R/PHO/SIMPLE/YEARLY/EXTENDED_PHO_2012.fst", 
                  as.data.table = T)

Baseline[, pho_enrolled_2012 := +(VSIMPLE_INDEX_MASTER %in% PHO$VSIMPLE_INDEX_MASTER)]

# Flags people who are in VARIANZ (for the year) during the next 5 years of follow-up
# Uses the "relaxed" VARAINZ with extended PHO

for(year in 2013:2017){
  
  VARIANZ <- read.fst(paste0("V:/source_data/R/VARIANZ/EXTENDED_VARIANZ_", year, ".fst"),
                      as.data.table = F)
  
  Baseline[, varianz := +(VSIMPLE_INDEX_MASTER %in% VARIANZ$VSIMPLE_INDEX_MASTER)]
  
  setnames(Baseline, "varianz", paste0("extended_varianz_", year))
  
  print(paste0(year, "completed"))
  
}


# ---- G.  Final Tidyup ----- 
library(fst)

# Gender - remove U; code M=1, F=0
Baseline <- Baseline[gender!="U"] #remove 36 people
Baseline[, gender_code := +(gender=="M")]

library(dplyr)

Baseline_FINAL <- Baseline %>%
  select(VSIMPLE_INDEX_MASTER, nhi_age, gender_code, en_prtsd_eth, en_nzdep_q, en_dod, end_fu_date,
         hx_diabetes, hx_vdr_diabetes, hx_af,
         starts_with("cmi"),
         starts_with("out"),
         starts_with("mortality"),
         starts_with("imp"),
         starts_with("ph"),
         pho_enrolled_2012,
         starts_with("extended"))

# Max nchar 32 for Stata
setnames(Baseline_FINAL, "ph_loopdiuretics_prior_5yrs_3evts", "ph_loopdiur_prior_5yrs_3evts")

# Name changes:
# Loopdiur, antidiabetes, en_prtsd_eth
fst::write.fst(Baseline_FINAL, "sbar288/working/R_VARIANZ_2012_v2.fst", compress = 75)
haven::write_dta(Baseline_FINAL, "sbar288/working/Stata14_VARIANZ_2012_v2.dta", version = 14)
haven::write_dta(Baseline_FINAL, "V:/smeh005/2012 VARIANZ/Stata14_VARIANZ_2012_v2.dta", version = 14)
feather::write_feather(Baseline_FINAL, "sbar288/working/Py_VARIANZ_2012_v2.feather")
