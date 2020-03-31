# VARIANZ 2012

As part of VIEW's drive toward machine learning in CVD risk prediction, the team began a research collaboration with Sebastiano Barbieri 
and Louisa Jorm from the University of New South Wales in December 2019. The initial project aim to establish baseline survival models 
using a 2012 VARIANZ population. Suneela Mehta from VIEW will update the 2006 VARIANZ policy equation using the prior statistical learning method, whereas Sebastiano will aim to create a survival model using a deep learning method. Further, Sebastiano aims to explore additional risk factors and predictors as part of a deep learning pipeline, and determine the added effects of prior unknown variables.

## Core Data
The core dataset is the VARIANZ 2012 health contact population. Information from the National Health Collection are linked to provide demographic, hospitalised history, hospitalised outcomes, death-specific outcomes, and baseline treatment. To ensure consistency, the exclusion criteria have been applied in data management. <a href="https://github.com/VIEW2020/Varianz2012/wiki" target="_blank">See the repository Wiki</a> for more information regarding the core dataset.

## Auxiliary Dataset
A monthly index from 1-60 was created to mark each month prior to index time-point; as there are 60 months in the 5 years prior to 2012-12-31. The index value of 1 marks January 2008 and the index value of 60 marks December 2012. The aim is to obtain a sequence of between 1 and 60 that represents the monthly duration in which each feature is activated. Note: a feature is a distinct chemical or ICD code. Two auxillary datasets were provided to Sebastinano:

- An index of all drug dispensing by chemical ID in the last 5 years
- An index of all admissions by ICD-10 code in the last 5 years
  
For drug dispensing: the medicated duration was calculated for each unique chemical using date dispensed and days supply. The theoretical drug coverage for each month was converted to an indexed sequence containing values between 1 and 60. This was repeated for each chemical and each individual. The resulting dataset appears as below.

![picture](/images/adm_index.png)
![picture](/images/disp_index.png)

For hospital admissions, similar process was followed. The time spent in hospital for each month was converted to an indexed sequence. This was repeated for each ICD-10 code and each individual. The resulting dataset appears below.
