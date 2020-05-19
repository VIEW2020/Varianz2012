# VARIANZ 2012

As part of VIEW's drive toward machine learning in CVD risk prediction, the team began a research collaboration with Sebastiano Barbieri and Louisa Jorm from the University of New South Wales in December 2019. The initial project aim to establish baseline survival models using a 2012 VARIANZ population. Suneela Mehta from VIEW will update the 2006 VARIANZ policy equation using the prior statistical learning method, whereas Sebastiano will aim to create a survival model using a deep learning method. Further, Sebastiano aims to explore additional risk factors and predictors as part of a deep learning pipeline, and determine the added effects of prior unknown variables.

## Core Data
<a href="https://github.com/VIEW2020/Varianz2012/wiki" target="_blank">See Wiki</a> for details regarding the core dataset including variable descriptions.<br></br>
The core dataset is the VARIANZ 2012 health contact population. Information from the National Health Collection are linked to provide demographic, hospitalised history, hospitalised outcomes, death-specific outcomes, and baseline treatment. To ensure consistency, the exclusion criteria have been applied in data management. 

<h3>Data Management</h3>
<p>The initial raw VARIANZ dataset contained 4,582,807 individuals. Using the latest primary / secondary mapping provided in the NHI by MoH (August 2019), duplicate ENHIs were removed (where multiple secondary keys were mapped to a single primary key). The following steps were taken to remove cohort members deemed necessary to ensure quality. In total, 2,381,112 people were removed. An additional 96,914 were flagged for removal as they did not have a valid last contact date and could not be censored. The study consists of 5 years of follow-up with the end-of-study at 2017-12-31.</p>

<table>
  <colgroup>
    <col/>
    <col/>
    <col/>
  </colgroup>
  <tbody>
    <tr>
      <th class="numberingColumn">
        <br/>
      </th>
      <th>Process</th>
      <th>n</th>
    </tr>
    <tr>
      <td class="numberingColumn">1</td>
      <td>Remove people outside of 30 - 74 years of age</td>
      <td>-2170506</td>
    </tr>
    <tr>
      <td class="numberingColumn" colspan="1">2</td>
      <td colspan="1">Remove people with missing enriched meshblocks</td>
      <td colspan="1">-39375</td>
    </tr>
    <tr>
      <td class="numberingColumn" colspan="1">3</td>
      <td colspan="1">Remove people with dispensing 28 days after death</td>
      <td colspan="1">-1006</td>
    </tr>
    <tr>
      <td class="numberingColumn" colspan="1">4</td>
      <td colspan="1">Remove people who died before time zero</td>
      <td colspan="1">-13779</td>
    </tr>
    <tr>
      <td class="numberingColumn" colspan="1">5</td>
      <td colspan="1">Remove people with a history of CVD admission</td>
      <td colspan="1">
        <p>-130143 </p>
      </td>
    </tr>
    <tr>
      <td class="numberingColumn" colspan="1">6</td>
      <td colspan="1">Remove people with 3x loop-diuretics in last 5 years</td>
      <td colspan="1">
        <p>-19168</p>
      </td>
    </tr>
    <tr>
      <td class="numberingColumn" colspan="1">7</td>
      <td colspan="1">Remove people with unknown gender "U"</td>
      <td colspan="1">-91</td>
    </tr>
    <tr>
      <td class="numberingColumn" colspan="1">8</td>
      <td colspan="1">Remove people without a valid NZDep score"</td>
      <td colspan="1">-6984</td>
    </tr>
  </tbody>
</table>

### Censoring
End of followup was calculated by first capturing time to first CVD event or death during follow-up (n = 90945). The remaining population without an event (n= 2110750) were eligible for an adjusted end of followup date. This was done using all VARIANZ component: PHO, GSM, PHARMS, NMDS, NAP, and LAB - with each year's records evaluated for the latest contact during the followup period. In effect, a "last contact date" was established during followup. For 96,914 individuals, a last contact date could not be established and therefore, an end of followup could not be determined. These observations we left in the dataset for analysis or removal.

## Auxiliary Dataset
A monthly index from 1-60 was created to mark each month prior to index time-point; as there are 60 months in the 5 years prior to 2012-12-31. The index value of 1 marks January 2008 and the index value of 60 marks December 2012. The aim is to obtain a sequence of between 1 and 60 that represents the monthly duration in which each feature is activated. Note: a feature is a distinct chemical or ICD code. Two auxillary datasets were provided to Sebastinano:

- An index of all drug dispensing by chemical ID in the last 5 years
- An index of all admissions by ICD-10 code in the last 5 years
  
For drug dispensing: the medicated duration was calculated for each unique chemical using date dispensed and days supply. The theoretical drug coverage for each month was converted to an indexed sequence containing values between 1 and 60. This was repeated for each chemical and each individual. The resulting dataset appears as below.

![picture](/images/adm_index.png)
![picture](/images/disp_index.png)

For hospital admissions, similar process was followed. The time spent in hospital for each month was converted to an indexed sequence. This was repeated for each ICD-10 code and each individual. The resulting dataset appears below.

## FAQ
<strong>Was PHO enrolment based on new/re-enrolment in 2012 only (ie tight criterion for PHO enrolment) or new/reenrolment over the previous 3 years?</strong>

<p>The core dataset used the original tight criteria of requiring hard-coded dates. However, this shouldn't be considered "tight". In reality, I initiated the whole process by first combining ALL YEARS and ALL QUARTERS of the PHO collection which created a large 2004-2017 PHO dataset (~260M row). Individuals were then grouped by their YEAR of contact using two types of dates – the ‘last consult date’ and the ‘enrolment date’. From there, I was able to establish their YEAR of contact based on actual hardcoded dates (i.e. either the last consult date or the enrolment date). So indeed this is still the “tight” criterion (as hardcoded dates are required for each year) but this approach allows for the inclusion and usage of all other yearly/quarterly datasets in the entire PHO collection. </p>

