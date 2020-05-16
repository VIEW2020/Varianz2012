
library(fst)
library(data.table)

dat <- read.fst("sbar288/data/PHH/PHH_2008.fst", as.data.table = T)


dat[, nrow := max(seq_len(.N)), by = VSIMPLE_INDEX_MASTER]
dat[, nchem := max(uniqueN(chem_id)), by = VSIMPLE_INDEX_MASTER]
# 
# summary(dat$nrow)
# 
# library(dplyr)
# 
# dat %>% summarise()
# 
# quantile(dat$nrow,  probs = c(95,99, 100)/100)
# 
# uniqueN(dat$VSIMPLE_INDEX_MASTER[which(dat$nrow>186)])
# 
# 
# dat2 <- dat[, disp_month := month(DATE_DISPENSED)]
# dat2 <- dat2[, end_month := month(SUPPLY_ENDATE)]
# 
# dat3a <- dat2[, unique(.SD), .SDcols = c("VSIMPLE_INDEX_MASTER", "form_id", "disp_month", "end_month")]
# dat3b <- dat2[, unique(.SD), .SDcols = c("VSIMPLE_INDEX_MASTER", "chem_id", "disp_month", "end_month")]

dat[, disp_month59 := 12*(year(DATE_DISPENSED) - 2008) + (month(DATE_DISPENSED) - 1)]
dat[, end_month59 := 12*(year(SUPPLY_ENDATE) - 2008) + (month(SUPPLY_ENDATE) - 1)]


dat <- dat[,.(VSIMPLE_INDEX_MASTER, chem_id, disp_month59, end_month59)]
dat <- dat[, unique(.SD)]
dat <- dat[, ntimes := (end_month59 - disp_month59) + 1]
dat <- copy(as.data.frame(lapply(dat, rep, dat$ntimes)))

setDT(dat)
dat <- dat[, index := seq_len(.N), by = list(VSIMPLE_INDEX_MASTER, chem_id, disp_month59, end_month59)]
dat[, index := index-1]
dat$dispmonth_index <- dat$disp_month59 + dat$index
