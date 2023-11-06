****************************************************
***   This do files cleans CPS_worker_flows_SA **
*****************************************************


clear
*global mainfolder "/Users/Myworld/Dropbox/PIR/WorkingFolder"
*global datafolder "${mainfolder}/OtherData/"

global mainfolder XXXX\PIR\WorkingFolder\
global datafolder XXXX\PIR\WorkingFolder\OtherData\

*cd ${datafolder}
*pwd
set more off 

import delimited "${datafolder}CPS_worker_flows_SA.csv", clear

rename month month_name

gen month = .

replace month =1 if month_name=="january"
replace month =2 if month_name=="february"
replace month =3 if month_name=="march"
replace month =4 if month_name=="april"
replace month =5 if month_name=="may"
replace month =6 if month_name=="june"
replace month =7 if month_name=="july"
replace month =8 if month_name=="august"
replace month =9 if month_name=="september"
replace month =10 if month_name=="october"
replace month =11 if month_name=="november"
replace month =12 if month_name=="december"

drop timeline month_name 
keep if year!=. & month!=.
save "${datafolder}CPS_worker_flows_SA.dta",replace 
