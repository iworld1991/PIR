clear
global mainfolder "/Users/Myworld/Dropbox/IncExpProject/WorkingFolder"
global folder "${mainfolder}/SurveyData/"
global sum_graph_folder "${mainfolder}/Graphs/ind"
global sum_table_folder "${mainfolder}/Tables"

cd ${folder}
pwd
set more off 
capture log close


log using "${mainfolder}/indSCE_Maro_log",replace

use "${folder}/SCE/IncExpSCEIndMacroM.dta"

************************
***** date *************
***********************
generate new_date = dofc(date)
gen year = year(new_date)
gen month = month(new_date)
gen date_str = string(year)+ "m"+string(month)
drop date index new_date 
gen date = monthly(date_str,"YM")
format date %tm
drop date_str
order userid date 
xtset userid date 

**********************
***** Other measures **
************************


local Moments incvar rincvar inciqr incskew 

foreach mom in `Moments'{
gen `mom'_ch = `mom' - l1.`mom'
label var `mom'_ch "change in `mom'"
}

************************
***** regression *******
***********************


xtset userid date 
eststo clear
foreach mom in `Moments'{
eststo `mom': reg `mom' Stkprob
*eststo `mom'_macro: reg `mom' f10.sp500
*eststo `mom'_both: reg `mom' Stkprob f10.sp500
*eststo `mom'_educ: reg `mom' c.Stkprob##i.educ_gr
*eststo `mom'_HHinc: reg `mom' c.Stkprob##i.HHinc_gr
*eststo `mom'_age: reg `mom' c.Stkprob##i.age_gr
*eststo `mom'_byear: reg `mom' c.Stkprob##i.byear_gr
*newey  `mom'  c.f1.sp500, lag(10)
*reg `mom'  c.l1.sp500##i.byear_gr c.l1.sp500##i.HHinc_gr c.l1.sp500##i.educ_gr c.l1.sp500##i.age_gr
*reg `mom'  c.Stkprob##i.byear_gr c.Stkprob##i.HHinc_gr c.Stkprob##i.educ_gr c.Stkprob##i.age_gr
}

esttab using "${sum_table_folder}/micro_reg_stk.csv", ///
       mtitles b(%8.3f) se(%8.3f) scalars(N r2) sfmt(%8.3f %8.3f %8.3f) replace

log close 
