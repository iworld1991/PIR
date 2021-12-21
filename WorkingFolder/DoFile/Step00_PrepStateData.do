****************************************************
***   This do files cleans SCE individual density **
***   forecasts and moments. It exclude the top and *
***   bottom 5 percentiles of mean and uncertainty. *
***   It also plots histograms of mean forecast and *
***   uncertainty. **********************************
*****************************************************


clear
global mainfolder "/Users/Myworld/Dropbox/IncExpProject/WorkingFolder"
global folder "${mainfolder}/OtherData/"
global sum_graph_folder "${mainfolder}/Graphs/pop"
global sum_table_folder "${mainfolder}/Tables"

cd ${folder}
pwd
set more off 


import excel "${folder}laus.xlsx", sheet("Sheet1") firstrow clear

gen month_new = .
replace month_new =1 if month=="Jan"
replace month_new = 2 if month =="Feb"
replace month_new = 3 if month=="Mar"
replace month_new = 4 if month=="Apr"
replace month_new = 5 if month == "May"
replace month_new = 6 if month =="Jun"
replace month_new = 7 if month=="Jul"
replace month_new = 8 if month =="Aug"
replace month_new = 9 if month=="Sep"
replace month_new =10 if month=="Oct"
replace month_new = 11 if month =="Nov"
replace month_new = 12 if month =="Dec"
drop month
rename month_new month
order state year month

rename state statename 
rename stateabbr state
save stateM.dta,replace

import excel "${folder}data_quarter.xlsx", sheet("Sheet1") firstrow clear

expand 3
sort state year quarter
bysort state year quarter: gen month =(quarter-1)*3+_n
order state statecode year quarter month

merge 1:1 state year month using "${folder}stateM.dta", keep(match master using) 
rename _merge state_merge

duplicates drop statecode year month,force
save "${folder}stateM.dta",replace 

**********************
** state code file
***********************

import excel "${folder}statecode.xlsx", sheet("Sheet1") firstrow clear 
save "${folder}statecode.dta",replace 
