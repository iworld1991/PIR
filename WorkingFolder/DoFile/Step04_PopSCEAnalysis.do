clear
global mainfolder "/Users/Myworld/Dropbox/PIR/WorkingFolder"
global folder "${mainfolder}/SurveyData/"

global sum_graph_folder "${mainfolder}/Graphs/pop"
global sum_table_folder "${mainfolder}/Tables"
global otherdata_folder "${mainfolder}/OtherData"
global graph_folder "/Users/Myworld/Dropbox/PIR/WorkingFolder/Graphs/sipp/"

cd ${folder}
pwd
set more off 

***************************
**  Clean and Merge Data **
***************************

use "${folder}/SCE/IncExpSCEDstIndM",clear 

duplicates report year month userid


******************************
*** Merge with demographics **
*****************************

merge 1:1 year month userid using "${folder}/SCE/IncExpSCEProbIndM",keep(master match) 
rename _merge hh_info_merge

************************************
**  Collapse to Population Series **
************************************

collapse (median)  Q24_mean Q24_rmean Q24_var Q24_rvar Q24_iqr ///
         (p25) Q24_rvar_p25 = Q24_rvar  ///
		 (p75) Q24_rvar_p75 = Q24_rvar  ///
		 , by(year month date) 
order date year month
duplicates report date 

drop date 
gen date_str=string(year)+"m"+string(month)
gen date= monthly(date_str,"YM")
format date %tm
tsset date 

* income risk estimation data
gen YM = year*100+month

** merge population estimation 
merge m:1 YM using "${otherdata_folder}/sipp/sipp_history_vol_decomposed_annual.dta", keep(master match)
drop _merge 

*****************************************
****  Renaming so that more consistent **
*****************************************

rename Q24_mean incmean
rename Q24_var incvar
rename Q24_iqr inciqr
rename Q24_rmean rincmean
rename Q24_rvar rincvar

rename Q24_rvar_p25 rincvar_p25
rename Q24_rvar_p75 rincvar_p75

***********************
**  Moving  Average  **
***********************

tsset date 

foreach mom in incmean incvar inciqr rincmean rincvar rincvar_p25 rincvar_p75 prisk2_all_rl trisk2_all_rl rincvar_all_rl{
gen `mom'mv3 = (F1.`mom' + `mom' + L1.`mom') / 3
label var `mom'mv3 "`mom' (3-month average)"
}

************************************
**  Times Series Plots  of PR **
************************************

foreach mom in incmean incvar inciqr rincmean rincvar{
twoway (tsline `mom',lwidth(med) lpattern(dash)) ///
       (tsline `mom'mv3,lwidth(thick) lpattern(solid)), ///
	   legend(label(1 "`mom'") label(2 "3-month moving average `mom'")) ///
	   title("`mom' of expected income growth") ///
       ytitle("`mom'") 
	   
graph export "${sum_graph_folder}/median_`mom'.png",as(png) replace  
}


*********************************************************
**  Times Series Plots  of PR agains estimate risks **
*******************************************************

** real PR 

twoway (tsline rincvarmv3,lwidth(thick)  lcolor(navy) lpattern(dash)) ///
       (tsline rincvar_all_rlmv3,lwidth(thick) lpattern(solid) lcolor(red) yaxis(2)), ///
	   legend(label(1 "Perceived") label(2 "Estimated (RHS)")) ///
	   title("Perceived and estimated risk") ///
       ytitle("Perceived Risk") ///
	   ytitle("Estimated Risk", axis(2)) 
	   
graph export "${graph_folder}/real_volatility_compare.png",as(png) replace 


twoway (tsline rincvarmv3,lwidth(thick)  lcolor(navy) lpattern(dash)) ///
       (tsline prisk2_all_rlmv3,lwidth(thick) lpattern(solid)  lcolor(red) yaxis(2)), ///
	   legend(label(1 "Perceived") label(2 "Estimated (RHS)")) ///
	   title("Perceived and estimated permanent risk") ///
       ytitle("Perceived Risk") ///
	   ytitle("Estimated Risk", axis(2)) 
	   
graph export "${graph_folder}/real_permanent_compare.png",as(png) replace 


twoway (tsline rincvarmv3,lwidth(thick)  lcolor(navy) lpattern(dash)) ///
       (tsline trisk2_all_rlmv3,lwidth(thick) lpattern(solid)  lcolor(red) yaxis(2)), ///
	   legend(label(1 "Perceived") label(2 "Estimated (RHS)")) ///
	   title("Perceived and estimated transitory risk") ///
       ytitle("Permanent Risk") ///
	   ytitle("Estimated Risk", axis(2)) 
	   
graph export "${graph_folder}/real_transitory_compare.png",as(png) replace   
