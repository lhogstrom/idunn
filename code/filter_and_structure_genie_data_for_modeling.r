
library(tidylog)
#load positoin lookup table
#genomicPositionLookup

#Steps: 
#- filter variants with coverage <20
#- filter variants where panel coverage is less than 90%
#- filter PANCAN and Hartwig Data - GENIE only
#- rename race fields
#- Therapy selection events only (Level 1) only

#- add criteria of >250 actionable patients per cancer type

# Identify those genomic positions with genomic panel coverage in most of the samples tested
genomicPositionSubset <- genomicPositionLookup %>%
  dplyr::filter(perc_of_patients_with_coverage > 90.0)

dataForModeling <- perPatientOncoKBLevelSorted %>%
  dplyr::filter(SourceStudy=="AACR-GENIE-data") %>%
  dplyr::filter(t_alt_plus_ref_count >= 20) %>%
  dplyr::mutate(positionCtypeID=paste0(ONCOTREE_CODE,"-",Hugo_Symbol,"-",Start_Position)) %>%
  dplyr::filter(positionCtypeID %in% genomicPositionSubset$positionCtypeID)



#### Alterinative approach retain filtered data for counting: 

# 1) Precompute the genomic position subset (and the key you'll join on)
genomicPositionSubset <- genomicPositionLookup %>%
  filter(perc_of_patients_with_coverage > 90) #%>%
  #mutate(positionCtypeID = paste0(ONCOTREE_CODE, "-", Hugo_Symbol, "-", Start_Position)) %>%
  #select(positionCtypeID)

# 2) Start and step through filters, recording counts
df0a <- perPatientOncoKBLevelSorted %>%
  mutate(
    RACE = recode(
      RACE,
      "White" = "European",
      "Black" = "African"
    )
  )
df0b <- df0a %>% filter(isOncoKbLevel1 == T)                
df0b <- df0a %>% filter(ONCOKB_HAS_SENSITIVE_ENTRY == T)                
df1 <- df0b %>% filter(SourceStudy == "AACR-GENIE-data")                # PANCAN/Hartwig removed (GENIE only)
df2 <- df1 %>% filter(t_alt_plus_ref_count >= 20)                      # coverage >= 20
df3 <- df2 %>% mutate(positionCtypeID = paste0(ONCOTREE_CODE, "-", Hugo_Symbol, "-", Start_Position))
df4 <- df3 %>% filter(positionCtypeID %in% genomicPositionSubset$positionCtypeID)  # panel coverage >= 90% positions
#df5 <- df4 %>% rename(                                                 # renaming doesn't change row count
  # race_new = race_old
#)

cTypeCntTble <- df4 %>%
  dplyr::group_by(ONCOTREE_CODE) %>%
  dplyr::summarise(n=n()) %>%
  dplyr::mutate(cTypeCntThresh = n >= 250) %>%
  dplyr::arrange(desc(n))
cTypeSelected <- cTypeCntTble[cTypeCntTble$cTypeCntThresh==T,]

df5 <- df4 %>%
  dplyr::filter(ONCOTREE_CODE %in% cTypeSelected$ONCOTREE_CODE)

# 3) Build the audit table
audit_table_actionable <- tibble(
  step = c(
    "Start",
    "Filter: GENIE only",
    "Filter: coverage ≥ 20",
    "Mutate: add positionCtypeID",
    "Filter: panel coverage ≥ 90% positions"#,
    #"Rename: race fields"
  ),
  rows = c(nrow(df0b), nrow(df1), nrow(df2), nrow(df3), nrow(df4))#, nrow(df5))
) %>%
  mutate(
    delta = rows - dplyr::lag(rows),
    pct_change = if_else(is.na(dplyr::lag(rows)), NA_real_, 100 * (delta / dplyr::lag(rows)))
  )

audit_table_actionable

### repeat for non-actionable
dfNotLvl10a <- perPatientOncoKBLevelSorted
dfNotLvl10b <- dfNotLvl10a %>% filter(isOncoKbLevel1 == F)               
dfNotLvl11 <- dfNotLvl10b %>% filter(SourceStudy == "AACR-GENIE-data")                # PANCAN/Hartwig removed (GENIE only)
dfNotLvl12 <- dfNotLvl11 %>% filter(t_alt_plus_ref_count >= 20)                      # coverage >= 20

dfNotLvl13 <- dfNotLvl12 %>%
  dplyr::filter(ONCOTREE_CODE %in% cTypeSelected$ONCOTREE_CODE)


dataForModeling <- rbind(df4,dfNotLvl12)