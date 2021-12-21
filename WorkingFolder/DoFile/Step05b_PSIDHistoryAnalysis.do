clear
global mainfolder "/Users/Myworld/Dropbox/IncExpProject/WorkingFolder"
global scefolder "/Users/Myworld/Dropbox/IncExpProject/WorkingFolder/SurveyData/SCE/"
global folder "${mainfolder}/SurveyData/"
global other "${mainfolder}/OtherData/"
global sum_graph_folder "${mainfolder}/Graphs/ind"
global sum_table_folder "${mainfolder}/Tables/"
global graph_folder "/Users/Myworld/Dropbox/IncExpProject/WorkingFolder/Graphs/psid/"

cd ${folder}
pwd
set more off 
capture log close

***************************************************
********** DECOMPOSED SHOCK and PERCEPTION of the COHORT
*********************************************************

/*
*** by byear_cohort 

use "${other}PSID/psid_history_vol_decomposed_edu_gender_byear5.dta", clear 
drop index 
drop if year >1997

collapse (mean) p_shk_sd_av = permanent ///
		 (mean) t_shk_sd_av = transitory, ///
		 by(byear_5yr edu_i_g sex_h)

gen edu_g = edu_i_g 
gen gender = sex_h 

merge 1:1 byear_5yr edu_g gender using "${scefolder}incvar_by_byear_5yr_edu_gender.dta",keep(match) 

gen lincvar = sqrt(incvar)
gen lrincvar = sqrt(rincvar)

drop if edu_g==1 
*keep if gender==2

* standard deviation of permanent shk and risk perception 

twoway (scatter lincvar p_shk_sd_av, color(blue)) ///
	   (lfit lincvar p_shk_sd_av,lcolor(red)) ///
	   (scatter lincvar t_shk_sd_av, color(gray)) ///
	   (lfit lincvar t_shk_sd_av,lcolor(black)), ///
	   xtitle("Permanent/transitory risks")  ///
	   ytitle("Perceived risk") ///
       title("Decomposed Risks and PR within Cohort/Education/Gender")  ///
	   legend(col(2) lab(1 "Permanent") lab(2 "Permanent (fitted)")  ///
	                  lab(3 "Transitory") lab(4 "Transitory (fitted)"))			  
graph export "${graph_folder}/log_wage_ptshk_by_byear_5yr_edu_gender_compare.png", as(png) replace 


*** by age_5yr 

use "${other}PSID/psid_history_vol_decomposed_edu_gender_age5.dta", clear 
drop index 
drop if year >1997

collapse (mean) p_shk_sd_av = permanent ///
		 (mean) t_shk_sd_av = transitory, ///
		 by(age_5yr edu_i_g sex_h)

gen edu_g = edu_i_g 
gen gender = sex_h 

gen pt_ratio = p_shk_sd_av/t_shk_sd_av 

merge 1:1 age_5yr edu_g gender using "${scefolder}incvar_by_age5y_edu_gender.dta",keep(match) 

gen lincvar = sqrt(incvar)
gen lrincvar = sqrt(rincvar)

drop if edu_g==1 
*keep if gender==2

* pt ratio 

twoway (scatter lincvar pt_ratio, color(blue)) ///
	   (lfit lincvar pt_ratio,lcolor(red)), ///
	   xtitle("Permanent/transitory risk ratio")  ///
	   ytitle("Perceived risk") ///
       title("Permanent/transitory ratio and PR within Age/Education/Gender")  ///
	   legend(col(2) lab(1 "Permanent") lab(2 "Permanent (fitted)")  ///
	                  lab(3 "Transitory") lab(4 "Transitory (fitted)"))			  
graph export "${graph_folder}/log_wage_ptratio_by_age_5yr_edu_gender_compare.png", as(png) replace 


* standard deviation of permanent shk and risk perception 

twoway (scatter lincvar p_shk_sd_av, color(blue)) ///
	   (lfit lincvar p_shk_sd_av,lcolor(red)) ///
	   (scatter lincvar t_shk_sd_av, color(gray)) ///
	   (lfit lincvar t_shk_sd_av,lcolor(black)), ///
	   xtitle("Permanent/transitory risks")  ///
	   ytitle("Perceived risk") ///
	   ytitle("risk (std)") ///
	   ysc(titlegap(3) outergap(0)) ///
       title("Decomposed Risks and Perceived Risks by Age/Education/Gender")  ///
	   legend(col(2) lab(1 "Permanent") lab(2 "Permanent (fitted)")  ///
	                  lab(3 "Transitory") lab(4 "Transitory (fitted)"))			  
graph export "${graph_folder}/log_wage_ptshk_by_age_5yr_edu_gender_compare.png", as(png) replace 

* permanent shock 
twoway (scatter p_shk_sd_av age_5yr, color(orange)) ///
       (lfit p_shk_sd_av age_5yr, lcolor(red)) ///
       (scatter lincvar age_5yr, color(gray) yaxis(2)) ///
	   (lfit lincvar age_5yr,lcolor(black) yaxis(2)), ///
       title("Permanent Risk and Perceived Risks by Age/Education/Gender")  ///
	   xtitle("year of birth")  ///
	   ytitle("permanent risk (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
					  
graph export "${graph_folder}/log_wage_pshk_by_age_5yr_edu_gender_compare.png", as(png) replace 

* transitory shock 
twoway (scatter t_shk_sd_av age_5yr, color(purple)) ///
       (lfit t_shk_sd_av age_5yr, lcolor(red)) ///
       (scatter lincvar age_5yr, color(gray) yaxis(2)) ///
	   (lfit lincvar age_5yr,lcolor(black) yaxis(2)), ///
       title("Transitory Risk and Perceived Risks by Age/Education/Gender")  ///
	   xtitle("year of birth")  ///
	   ytitle("transitory risk (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
					  
graph export "${graph_folder}/log_wage_tshk_by_age_5yr_edu_gender_compare.png", as(png) replace 
*/

******************************************************
********** Experience *******************************
****************************************************


import excel "${other}PSID/psid_history_vol_decomposed_whole.xlsx", sheet("Sheet1") firstrow

destring year cohort av_shk_gr var_shk_gr var_shk av_id_shk_gr ///
         var_id_shk_gr var_id_shk av_ag_shk_gr var_ag_shk_gr ///
		 var_ag_shk permanent transitory N, force replace

***********************
** generate variables 
***********************

gen age = year-cohort + 20
label var age "age"

***********************
** relabel ************
**********************

label var N "history sample size"
label var av_shk_gr "experienced log unexplained income growth"
label var var_shk_gr "experienced growth volatility"
label var var_shk "experienced level volatility"

label var av_id_shk_gr "experienced log unexplained income growth (idiosyncratic)"
label var var_id_shk_gr "experienced growth volatility (idiosyncratic)"
label var var_id_shk "experienced level volatility (idiosyncratic)"

label var av_ag_shk_gr "experienced log unexplained income growth (aggregate)"
label var var_ag_shk_gr "experienced growth volatility (aggregate)"
label var var_ag_shk "experienced level volatility (aggregate)"

label var permanent "experienced permanent volatility std"
label var transitory "experienced transitory volatility std"


***********************************
** extend psid history data to 2019 
************************************

/*
expand 3 if year==2017
sort cohort year 
replace year = 2018 if year==2017 & year[_n-1]==2017 & cohort ==cohort[_n-1]
replace year = 2019 if year==2017 & year[_n-1]==2018 & cohort ==cohort[_n-1]
*/

***********************************
** merge with perceived risk data 
************************************

gen Q32 = age
*gen educ_gr = edu

merge 1:m Q32 year using "${folder}/SCE/IncExpSCEProbIndM", keep(using match) 
rename _merge sce_ind_merge 

merge 1:1 year month userid using "${folder}/SCE/IncExpSCEDstIndM", keep(using match) 
rename _merge sce_ind_merge2

***********************
** format the date 
**********************

drop date 
gen date_str=string(year)+"m"+string(month) 
gen date= monthly(date_str,"YM")
format date %tm
order userid date year month 


*****************************************
****  Renaming so that more consistent **
*****************************************

rename Q24_mean incmean
rename Q24_var incvar
rename Q24_iqr inciqr
rename IncSkew incskew 
rename Q24_rmean rincmean
rename Q24_rvar rincvar

rename D6 HHinc 
rename Q33 gender 
rename Q10_1 fulltime
rename Q10_2 parttime
rename Q12new selfemp
rename Q6new Stkprob
rename Q4new UEprobAgg
rename Q13new UEprobInd
rename Q26v2 spending_dum
rename Q26v2part2 spending 


***********************
** filters
**********************

keep if age > 20 & age <= 55
 
*********************************************
** generate new group variables 
*******************************************

** income group 
egen inc_gp = cut(Q47), group(3) 

** finanial condition improvement 
gen Q1_gp = .
replace Q1_gp =1 if Q1<=2
replace Q1_gp =2 if Q1==3
replace Q1_gp =3 if Q1>3 & Q1!=.

** cohort group

egen cohort_gp = cut(cohort), at(1970,1980,1990,2000,2010,2020)
label var cohort_gp "cohort"

** age group
egen age_gp = cut(age), at(20 35 55,70)
label var age_gp "age group"

*********************************************
** generate variables 
*******************************************

foreach var in incvar rincvar var_shk var_shk_gr var_id_shk var_id_shk_gr var_ag_shk var_ag_shk_gr permanent transitory{
gen l`var' = log(`var')
}

gen lprobUE= log(UEprobAgg)
label var lprobUE "log probability of UE higher next year"

*****************
** chart 
*****************

/*
graph box rmse if year == 2017, ///
           over(cohort_gp,relabel(1 "1970" 2 "1980" 3 "1990" 4 "2000" 5 "2010")) ///
		   medline(lcolor(black) lw(thick)) ///
		   box(1,bfcolor(red) blcolor(black)) ///
		   title("Experienced volatility of different cohorts up to 2017") ///
		   b1title("year of entering job market")

graph export "${sum_graph_folder}/experience_var_bycohort.png", as(png) replace 
*/


** different experience of different cohort 

/*
*** by cohort and time

preserve
bysort year age: gen ct = _N

drop if ct<=30

collapse lQ24_var av_gr av_id_gr av_ag_gr lvar_shk lvar_id_shk lvar_ag_shk lpermanent ltransitory ue_av ue_var, by(year age) 

gen pt_ratio = lpermanent-ltransitory
label var pt_ratio "permanent/transitory risk ratio"

label var lQ24_var "Perceived risk"
label var av_gr "Experienced log income change"
label var lvar_shk "Experienced volatility"
label var av_id_gr "Experienced log idiosyncratic change"
label var lvar_id_shk "Experienced idiosyncratic volatility"
label var av_ag_gr "Experienced log aggregate change"
label var lvar_ag_shk "Experienced aggregate volatility"

label var lpermanent "Experienced permanent volatility"
label var ltransitory "Experienced transitory volatility"

label var ue_av "average UE rate"
label var ue_var "volatility of UE rate"

* ag ue
twoway (scatter lQ24_var ue_av, color(ltblue)) ///
       (lfitci lQ24_var ue_av, lcolor(red) lw(thick) lpattern(dash)) if ue_av!=., ///
	   title("Experienced UE and perceived income risks") ///
	   xtitle("experienced UE rate") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_ue_var_data.png", as(png) replace 

* ag ue var
twoway (scatter lQ24_var ue_var, color(ltblue)) ///
       (lfit lQ24_var ue_var, lcolor(red) lw(thick) lpattern(dash)) if ue_var!=., ///
	   title("Experienced UE volatility and perceived income risks") ///
	   xtitle("experienced UE rate") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_ue_var_var_data.png", as(png) replace 


* growth 
twoway (scatter lQ24_var av_gr, color(ltblue)) ///
       (lfit lQ24_var av_gr, lcolor(red) lw(thick) lpattern(dash)) if av_gr!=., ///
	   title("Experienced income growth and perceived income risks") ///
	   xtitle("experienced income growth") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_gr_var_data.png", as(png) replace 

* risk 
twoway (scatter lQ24_var lvar_shk, color(ltblue)) ///
       (lfit lQ24_var lvar_shk, lcolor(red) lw(thick) lpattern(dash)) if lvar_shk!=., ///
	   title("Experienced volatility and perceived income risks") ///
	   xtitle("log experienced volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_var_data.png", as(png) replace 

* id growth
twoway (scatter lQ24_var av_id_gr, color(ltblue)) ///
       (lfit lQ24_var av_id_gr, lcolor(red) lw(thick) lpattern(dash)) if av_gr!=., ///
	   title("Experienced income growth and perceived income risks") ///
	   xtitle("experienced idiosyncratic growth") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_id_gr_var_data.png", as(png) replace 

* id risk 
twoway (scatter lQ24_var lvar_id_shk, color(ltblue)) ///
       (lfit lQ24_var lvar_id_shk, lcolor(red) lw(thick) lpattern(dash)) if lvar_id_shk!=., ///
	   title("Experienced volatility and perceived income risks") ///
	   xtitle("log experienced idiosyncratic volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_id_var_data.png", as(png) replace 


* ag growth
twoway (scatter lQ24_var av_ag_gr, color(ltblue)) ///
       (lfit lQ24_var av_ag_gr, lcolor(red) lw(thick) lpattern(dash)) if av_gr!=., ///
	   title("Experienced income growth and perceived income risks") ///
	   xtitle("experienced aggregate growth") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_ag_gr_var_data.png", as(png) replace 

* ag risk 
twoway (scatter lQ24_var lvar_ag_shk, color(ltblue)) ///
       (lfit lQ24_var lvar_ag_shk, lcolor(red) lw(thick) lpattern(dash)) if lvar_ag_shk!=., ///
	   title("Experienced volatility and perceived income risks") ///
	   xtitle("log experienced aggregate volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_ag_var_data.png", as(png) replace 

* permanent risk
twoway (scatter lQ24_var lpermanent, color(ltblue)) ///
       (lfit lQ24_var lpermanent, lcolor(red) lw(thick) lpattern(dash)) if lpermanent!=., ///
	   title("Experienced permanent volatility and perceived income risks") ///
	   xtitle("log experienced permanent volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_permanent_var_data.png", as(png) replace 

* transitory risk
twoway (scatter lQ24_var ltransitory, color(ltblue)) ///
       (lfit lQ24_var ltransitory, lcolor(red) lw(thick) lpattern(dash)) if ltransitory!=., ///
	   title("Experienced transitory volatility and perceived income risks") ///
	   xtitle("log experienced transitory volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_transitory_var_data.png", as(png) replace 

* permanent/transitory ratio

twoway (scatter lQ24_var pt_ratio, color(ltblue)) ///
       (lfit lQ24_var pt_ratio, lcolor(red) lw(thick) lpattern(dash)) if pt_ratio!=., ///
	   title("Experienced volatility ratio and perceived income risks") ///
	   xtitle("log experienced permanent/transitory volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_ratio_var_data.png", as(png) replace 

restore
*/

*** by cohort/time/educ

preserve
bysort year age educ_gr: gen ct = _N

drop if ct<=30

collapse lincvar lrincvar av_shk_gr av_id_shk_gr av_ag_shk_gr lvar_shk lvar_shk_gr lvar_id_shk lvar_id_shk_gr lvar_ag_shk lvar_ag_shk_gr lpermanent ltransitory ue_av ue_var, by(year age educ_gr) 

gen pt_ratio = lpermanent-ltransitory
label var pt_ratio "permanent/transitory risk ratio"

label var lincvar "Perceived risk"
label var lrincvar "Perceived risk"
label var av_shk_gr "Experienced log income change"
label var av_id_shk_gr "Experienced log idiosyncratic change"
label var av_ag_shk_gr "Experienced log aggregate change"

label var lvar_shk "Experienced volatility"
label var lvar_id_shk "Experienced idiosyncratic volatility"
label var lvar_ag_shk "Experienced aggregate volatility"

label var lvar_shk_gr "Experienced growth volatility"
label var lvar_id_shk_gr "Experienced growth idiosyncratic volatility"
label var lvar_ag_shk_gr "Experienced growth aggregate volatility"

label var lpermanent "Experienced permanent volatility"
label var ltransitory "Experienced transitory volatility"

label var ue_av "average UE rate"
label var ue_var "volatility of UE rate"


* ag ue
twoway (scatter lrincvar ue_av, color(ltblue)) ///
       (lfitci lrincvar ue_av, lw(med) ciplot(rline) blpattern(dash)) if ue_av!=., ///
	   title("Experienced UE and perceived income risks") ///
	   xtitle("experienced UE rate") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_ue_var_data.png", as(png) replace 

* ag ue var
twoway (scatter lrincvar ue_var, color(ltblue)) ///
       (lfitci lrincvar ue_var, lw(med) ciplot(rline) blpattern(dash)) if ue_var!=., ///
	   title("Experienced UE volatility and perceived income risks") ///
	   xtitle("experienced UE rate") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_ue_var_var_data.png", as(png) replace 

* growth 
twoway (scatter lrincvar av_shk_gr, color(ltblue)) ///
       (lfitci lrincvar av_shk_gr, lw(med) ciplot(rline) blpattern(dash)) if av_shk_gr!=., ///
	   title("Experienced income growth and perceived income risks") ///
	   xtitle("experienced income growth") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_gr_var_data.png", as(png) replace 

* risk 
twoway (scatter lrincvar lvar_shk, color(ltblue)) ///
       (lfitci lrincvar lvar_shk, lw(med) ciplot(rline) blpattern(dash)) if lvar_shk!=., ///
	   title("Experienced volatility and perceived income risks") ///
	   xtitle("log experienced volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_var_data.png", as(png) replace 


* id growth
twoway (scatter lrincvar av_id_shk_gr, color(ltblue)) ///
       (lfitci lrincvar av_id_shk_gr, lw(med) ciplot(rline) blpattern(dash)) if av_id_shk_gr!=., ///
	   title("Experienced income growth and perceived income risks") ///
	   xtitle("experienced idiosyncratic growth") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_id_gr_var_data.png", as(png) replace 

* id risk 
twoway (scatter lrincvar lvar_id_shk, color(ltblue)) ///
       (lfitci lrincvar lvar_id_shk, lw(med) ciplot(rline) blpattern(dash)) if lvar_id_shk!=., ///
	   title("Experienced volatility and perceived income risks") ///
	   xtitle("log experienced idiosyncratic volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_id_var_data.png", as(png) replace 


* id growth risk 
twoway (scatter lrincvar lvar_id_shk_gr, color(ltblue)) ///
       (lfitci lrincvar lvar_id_shk_gr, lw(med) ciplot(rline) blpattern(dash)) if lvar_id_shk!=., ///
	   title("Experienced volatility and perceived income risks") ///
	   xtitle("log experienced idiosyncratic volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_id_gr_var_data.png", as(png) replace 


* ag growth
twoway (scatter lrincvar av_ag_shk_gr, color(ltblue)) ///
       (lfitci lrincvar av_ag_shk_gr, lw(med) ciplot(rline) blpattern(dash)) if av_ag_shk_gr!=., ///
	   title("Experienced income growth and perceived income risks") ///
	   xtitle("experienced aggregate growth") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_ag_gr_var_data.png", as(png) replace 

* ag risk 
twoway (scatter lrincvar lvar_ag_shk, color(ltblue)) ///
       (lfitci lrincvar lvar_ag_shk, lw(med) ciplot(rline) blpattern(dash)) if lvar_ag_shk!=., ///
	   title("Experienced volatility and perceived income risks") ///
	   xtitle("log experienced aggregate volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_ag_var_data.png", as(png) replace 

* permanent risk
twoway (scatter lrincvar lpermanent, color(ltblue)) ///
       (lfitci lrincvar lpermanent, lw(med) ciplot(rline) blpattern(dash)) if lpermanent!=., ///
	   title("Experienced permanent volatility and perceived income risks") ///
	   xtitle("log experienced permanent volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_permanent_var_data.png", as(png) replace 

* transitory risk
twoway (scatter lrincvar ltransitory, color(ltblue)) ///
       (lfitci lrincvar ltransitory, lw(med) ciplot(rline) blpattern(dash)) if ltransitory!=., ///
	   title("Experienced transitory volatility and perceived income risks") ///
	   xtitle("log experienced transitory volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_transitory_var_data.png", as(png) replace 

* permanent/transitory ratio

twoway (scatter lrincvar pt_ratio, color(ltblue)) ///
       (lfitci lrincvar pt_ratio, lw(med) ciplot(rline) blpattern(dash)) if pt_ratio!=., ///
	   title("Experienced volatility ratio and perceived income risks") ///
	   xtitle("log experienced permanent/transitory volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_ratio_var_data.png", as(png) replace 

restore

/*
*** by age only 
preserve
bysort age: gen ct = _N


collapse lincvar lrincvar av_shk_gr av_id_shk_gr av_ag_shk_gr lvar_shk lvar_id_shk lvar_ag_shk lpermanent ltransitory ue_av ue_var, by(year age) 

gen pt_ratio = lpermanent-ltransitory
label var pt_ratio "permanent/transitory risk ratio"

label var lincvar "Perceived risk"
label var lrincvar "Perceived risk"
label var av_shk_gr "Experienced log income change"
label var lvar_shk "Experienced volatility"
label var av_id_shk_gr "Experienced log idiosyncratic change"
label var lvar_id_shk "Experienced idiosyncratic volatility"
label var av_ag_shk_gr "Experienced log aggregate change"
label var lvar_ag_shk "Experienced aggregate volatility"

label var lpermanent "Experienced permanent volatility"
label var ltransitory "Experienced transitory volatility"

twoway (scatter lQ24_var lpermanent, color(ltblue)) ///
       (lfit lQ24_var lpermanent, lcolor(red) lw(thick) lpattern(dash)) if lpermanent!=., ///
	   title("Experienced permanent volatility and perceived income risks") ///
	   xtitle("log experienced permanent volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_permanent_var_data_by_age.png", as(png) replace 

twoway (scatter lQ24_var ltransitory, color(ltblue)) ///
       (lfit lQ24_var ltransitory, lcolor(red) lw(thick) lpattern(dash)) if ltransitory!=., ///
	   title("Experienced transitory volatility and perceived income risks") ///
	   xtitle("log experienced transitory volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_transitory_var_data_by_age.png", as(png) replace 

twoway (scatter lQ24_var pt_ratio, color(ltblue)) ///
       (lfit lQ24_var pt_ratio, lcolor(red) lw(thick) lpattern(dash)) if pt_ratio!=., ///
	   title("Experienced volatility ratio and perceived income risks") ///
	   xtitle("log experienced permanent/transitory volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_ratio_var_data_by_age.png", as(png) replace 

restore 
*/
/*
preserve

bysort year age: gen ct = _N
collapse lQ24_var lrmse, by(cohort inc_gp) 

label var lQ24_var "log perceived risk"
label var lrmse "log experienced volatility"
twoway (scatter lQ24_var lrmse , color(ltblue)) ///
       (lfit lQ24_var lrmse, lcolor(red) lw(thick)) if lrmse!=., ///
	   by(inc_gp,title("Experienced volatility and perceived income risks") note("Graph by income group") rows(1)) ///
	   xtitle("log experienced volatility") ///
	   ytitle("log perceived income riks") ///
	   legend(off)
	   
graph export "${sum_graph_folder}/experience_var_var_by_income_data.png", as(png) replace 
restore


*********************************************
** experienced volatility and perceived risk regression
*******************************************

label var lQ24_var "log perceived risk"
label var lQ24_iqr "log perceived iqr"
label var lprobUE "log prob of higher UE"

eststo clear
foreach var in lQ24_var lQ24_iqr lprobUE{
eststo: reg `var' lrmse i.age_gp
estadd local hasage "Yes",replace
estadd local haseduc "No",replace
estadd local hasinc "No",replace


eststo: reg `var' lrmse i.age_gp i.Q36
estadd local hasage "Yes",replace
estadd local haseduc "Yes",replace
estadd local hasinc "No",replace

eststo: reg `var' lrmse i.age_gp i.Q36 i.inc_gp
estadd local hasage "Yes",replace
estadd local haseduc "Yes",replace
estadd local hasinc "Yes",replace

}

label var lrmse "log experienced volatility"
esttab using "${sum_table_folder}/micro_reg_history_vol.csv", ///
         keep(lrmse) st(r2 N hasage haseduc hasinc,label("R-squre" "N" "Control age" "Control educ" "Control income")) ///
		 label ///
		 replace 
		 
		 
************************************************
**  experienced volatility and state wage growth
***********************************************

label var lQ24_var "log perceived risk"
label var lQ24_iqr "log perceived iqr"

eststo clear
foreach var in lQ24_var lQ24_iqr{

eststo: reg `var' c.lrmse##c.wagegrowth i.age_gp i.Q36 i.inc_gp
estadd local hasage "Yes",replace
estadd local haseduc "Yes",replace
estadd local hasinc "Yes",replace

}

label var lrmse "log experienced volatility"
esttab using "${sum_table_folder}/micro_reg_history_vol_state.csv", ///
         keep(lrmse *lrmse) st(r2 N hasage haseduc hasinc,label("R-squre" "N" "Control age" "Control educ" "Control income")) ///
		 label ///
		 replace 

	*/
	
************************************************
**  experienced volatility and numeracy
***********************************************


label var lQ24_var "log perceived risk"
label var lQ24_iqr "log perceived iqr"

eststo clear
foreach var in lQ24_var lQ24_iqr{

eststo: reg `var' c.lrmse##c.nlit i.age_gp i.Q36 i.inc_gp
estadd local hasage "Yes",replace
estadd local haseduc "Yes",replace
estadd local hasinc "Yes",replace

}

label var lrmse "log experienced volatility"
esttab using "${sum_table_folder}/micro_reg_history_vol_nlit.csv", ///
         keep(lrmse *lrmse) st(r2 N hasage haseduc hasinc,label("R-squre" "N" "Control age" "Control educ" "Control income")) ///
		 label ///
		 replace 
