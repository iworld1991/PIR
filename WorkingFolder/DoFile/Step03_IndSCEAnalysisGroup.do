clear
*global mainfolder "/Users/Myworld/Dropbox/PIR/WorkingFolder"
*global folder "${mainfolder}/SurveyData/"
*global graph_folder "${mainfolder}/Graphs/"
*global sum_table_folder "${mainfolder}/Tables"
*global otherdata_folder "${mainfolder}/OtherData"


global mainfolder XXXX\PIR\WorkingFolder\
global folder XXXX\WorkingFolder\SurveyData\
global graph_folder XXXX\PIR\WorkingFolder\Graphs\
global sum_table_folder XXXX\PIR\WorkingFolder\Tables
global otherdata_folder XXXX\PIR\WorkingFolder\OtherData\

set more off 
capture log close


log using "${mainfolder}/indSCE_Est_log",replace

***************************
**  Clean and Merge Data **
***************************

use "${folder}/SCE/IncExpSCEDstIndM",clear 

duplicates report year month userid

************************************************
*** Merge with demographics and other moments **
************************************************

merge 1:1 year month userid using "${folder}/SCE/IncExpSCEProbIndM",keep(master match using) 
rename _merge hh_info_merge


** format the date 
drop date 
gen date_str=string(year)+"m"+string(month) 
gen date= monthly(date_str,"YM")
format date %tm
order userid date year month   


*************************************
*** Merge with macro data   **
*************************************

merge m:1 date using "${mainfolder}/OtherData/macroM.dta", keep(master match) 
rename _merge sp_merge


*******************************
**  Set Panel Data Structure **
*******************************
rename userid ID 
xtset ID date   /* this is not correct. ID is unique here.*/
sort ID year month 

*******************************
** Exclude extreme outliers 
******************************

keep if Q32 <= 65 & Q32 >= 20

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
rename Q32 age 
rename Q33 gender 
rename Q10_1 fulltime
rename Q10_2 parttime
rename Q12new selfemp
rename Q6new Stkprob
rename Q4new UEprobAgg
rename Q13new UEprobInd
rename Q26v2 spending_dum
rename Q26v2part2 spending 


************************
** focus on non-zero skewness
****************************

replace incskew = . if incskew==0


*************************
*** Exclude outliers *****
*************************

local Moments incmean rincmean incvar rincvar inciqr incskew 

foreach var in `Moments'{
      egen `var'pl=pctile(`var'),p(1)
	  egen `var'pu=pctile(`var'),p(99)
	  replace `var' = . if `var' <`var'pl | (`var' >`var'pu & `var'!=.)
}

/*
* other thresholds 

foreach var in `Moments'{
      egen `var'l_truc=pctile(`var'),p(8)
	  egen `var'u_truc=pctile(`var'),p(92)
	  replace `var' = . if `var' <`var'l_truc | (`var' >`var'u_truc & `var'!=.)
}
*/

*****************************
*** generate other vars *****
*****************************

gen age2 = age^2
label var age2 "Age-squared"

encode state, gen(state_id)
label var state_id "state id"

egen byear_5yr = cut(byear), ///
     at(1945 1950 1955 1960 1965 1970 ///
	    1975 1980 1985 1990 1995 2000 ///
		2005 2010)
label var byear_5yr "5-year cohort"

egen age_5yr = cut(age), ///
     at(20 25 30 35 40 45 ///
	    50 55 60)
label var age_5yr "5-year age"

gen byear_g = cond(byear>=1980,1,0)
label define byearglb 0 "before 1980s" 1 "after 1980s"
*label define byearglb 0 "1950s" 1 "1960s" 2 "1970s" 3 "1980s"
label value byear_g byearlb

egen age_g = cut(age), group(3)  
label var age_g "age group"
label define agelb 0 "young" 1 "middle-age" 2 "old"
label value age_g agelb

gen edu_g = . 
replace edu_g = 1 if educ==1
replace edu_g = 2 if educ==2 | educ ==3 | educ == 4
replace edu_g = 3 if educ <=9 & educ>4

label var edu_g "education group"
label define edu_glb 1 "HS dropout" 2 "HS graduate" 3 "College/above"
label value edu_g edu_glb

label define gdlb 1 "Male" 2 "Female" 
label value gender gdlb

egen HHinc_g = cut(HHinc), group(2)
label var HHinc_g "Household income group"
label define HHinc_glb 0 "low inc" 1 "high inc"
label value HHinc_g HHinc_glb

label define gender_glb 1 "male" 2 "female"
label value gender gender_glb

gen fbetter =cond(Q1>2,1,0)
replace fbetter = . if Q1 ==3  
label var fbetter "finance better"

label define better_glb 0 "worse" 1 "better"
label value fbetter better_glb

gen nlit_g = cond(nlit>=3,1,0) 
replace nlit_g = . if nlit ==.
label var nlit_g "numeracy literacy score group"
label define nlilb 0 "low" 1 "high" 
label value nlit_g nlitlb

local group_vars byear_g age_g edu_g HHinc_g fbetter nlit_g

/*
*********************************
*** bar charts *****
**********************************

graph bar incvar, ///
           over(HHinc,relabel(1 "<10k" 2 "<20k" 3 "<30k" 4 "<40k" 5 "<50k" 6 "<60k" 7 "<75k" 8 "<100k" 9 "<150k" 10 "<200k" 11 ">200k")) ///
		   bar(1, color(navy)) ///
		   title("Perceived Risk by Household Income") ///
		   b1title("Household income") ///
		   ytitle("Average perceived risk") 
graph export "${sum_graph_folder}/boxplot_var_HHinc_stata.png", as(png) replace 


graph bar rincvar, ///
           over(HHinc,relabel(1 "<10k" 2 "<20k" 3 "<30k" 4 "<40k" 5 "<50k" 6 "<60k" 7 "<75k" 8 "<100k" 9 "<150k" 10 "<200k" 11 ">200k")) ///
		   bar(1, color(navy)) ///
		   title("Perceived Real Income Risk by Household Income") ///
		   b1title("Household income") ///
		   ytitle("Average perceived risk of real income") 
graph export "${sum_graph_folder}/boxplot_rvar_HHinc_stata.png", as(png) replace 
*/

*********************************
*** generate group summary data file *****
**********************************

local SCEgroup incmean rincmean incvar rincvar

* by age 

preserve 
collapse `SCEgroup', by(age) 
save "${folder}/SCE/incvar_by_age.dta",replace
restore 

* by age x gender 
preserve
collapse `SCEgroup', by(age gender) 
save "${folder}/SCE/incvar_by_age_gender.dta",replace 
restore 

* by age x education 
preserve
collapse `SCEgroup', by(age edu_g) 
save "${folder}/SCE/incvar_by_age_edu.dta",replace 
restore 

* by age x education x gender
preserve
collapse `SCEgroup', by(age edu_g gender) 
save "${folder}/SCE/incvar_by_age_edu_gender.dta",replace 
restore 

* by age5 x education x gender
preserve
collapse `SCEgroup', by(age_5yr edu_g gender) 
save "${folder}/SCE/incvar_by_age5y_edu_gender.dta",replace 
restore 

* by year of birth

preserve 
collapse `SCEgroup', by(byear) 
save "${folder}/SCE/incvar_by_byear.dta",replace
restore 

* by year of birth(5year) and age

preserve 
collapse `SCEgroup', by(byear_5yr age) 
save "${folder}/SCE/incvar_by_byear_5_yr_age.dta",replace
restore 

* by year of birth and gender

preserve 
collapse `SCEgroup', by(byear gender) 
save "${folder}/SCE/incvar_by_byear_gender.dta",replace
restore 

* by year of birth and education

preserve 
collapse `SCEgroup', by(byear edu_g) 
save "${folder}/SCE/incvar_by_byear_edu.dta",replace
restore 

* by year of birth(5 year cohort) and education

preserve 
collapse `SCEgroup', by(byear_5yr edu_g) 
save "${folder}/SCE/incvar_by_byear_5yr_edu.dta",replace
restore 


* by year of birth(5 year cohort) and education and gender 

preserve 
collapse `SCEgroup', by(byear_5yr edu_g gender) 
save "${folder}/SCE/incvar_by_byear_5yr_edu_gender.dta",replace
restore 

/*
**********************************
*** tables and hists of Vars *****
**********************************


local Moments incmean incvar inciqr rincmean rincvar incskew

foreach gp in `group_vars' {
tabstat `Moments', st(p10 p50 p90) by(`gp')
}


foreach gp in `group_vars' {
table `gp', c(median incvar mean incvar median rincvar mean rincvar) by(year)
}


** histograms 

foreach mom in `Moments'{

twoway (hist `mom',fcolor(ltblue) lcolor(none)), ///
	   ytitle("") ///
	   title("`mom'")
graph export "${sum_graph_folder}/hist/hist_`mom'.png",as(png) replace  

}


* 4 groups 
foreach gp in byear_g{
foreach mom in `Moments'{
twoway (hist `mom' if `gp'==0,fcolor(gs15) lcolor("")) /// 
       (hist `mom' if `gp'==1,fcolor(ltblue) lcolor("")) ///
	   (hist `mom' if `gp'==2,fcolor(red) lcolor("")) ///
	   (hist `mom' if `gp'==3,fcolor(green) lcolor("")), ///
	   xtitle("") ///
	   ytitle("") ///
	   title("`mom'") ///
	   legend(label(1 `gp'=0) label(2 `gp'=1) label(3 `gp'=2) label(4 `gp'=3) col(1))

graph export "${sum_graph_folder}/hist/hist_`mom'_`gp'.png",as(png) replace  
}
}

* 3 groups 
foreach gp in HHinc_g age_g{
foreach mom in `Moments'{

twoway (hist `mom' if `gp'==0,fcolor(gs15) lcolor("")) /// 
       (hist `mom' if `gp'==1,fcolor(ltblue) lcolor("")) ///
	   (hist `mom' if `gp'==2,fcolor(red) lcolor("")), ///
	   xtitle("") ///
	   ytitle("") ///
	   title("`mom'") ///
	   legend(label(1 `gp'=0) label(2 `gp'=1) label(3 `gp'=2) col(1))

graph export "${sum_graph_folder}/hist/hist_`mom'_`gp'.png",as(png) replace  

}
}

* 2 groups 


foreach gp in edu_g fbetter{
foreach mom in `Moments'{

twoway (hist `mom' if `gp'==0,fcolor(gs15) lcolor("")) /// 
       (hist `mom' if `gp'==1,fcolor(ltblue) lcolor("")), ///
	   xtitle("") ///
	   ytitle("") ///
	   title("`mom'") ///
	   legend(label(1 `gp'=0) label(2 `gp'=1) col(1))

graph export "${sum_graph_folder}/hist/hist_`mom'_`gp'.png",as(png) replace  

}
}
*/


*******************************************
*** comparison with SIPP realizations *****
********************************************

gen YM = year*100+month

/*

** sub sample

preserve
drop educ
rename edu_g educ
merge m:1 gender educ age_5yr YM ///
         using "${otherdata_folder}/sipp/sipp_history_vol_decomposed_edu_gender_age5_annual.dta", keep(master match)
drop _merge 
xtset ID date
collapse (mean) incvar rincvar prisk2_sub trisk2_sub rincvar_sub_now rincvar_sub_rl, by(date year month educ) 

table date if prisk2_sub!=.

foreach var in prisk2_sub trisk2_sub rincvar_sub_rl rincvar_sub_now{
egen `var'_p5 = pctile(`var'),p(1) by(date)
egen `var'_p95 = pctile(`var'),p(99) by(date)
replace `var'=. if `var'<`var'_p5 | `var'>=`var'_p95
}

corr rincvar rincvar_sub_rl rincvar_sub_now prisk2_sub trisk2_sub

xtset educ date 
egen rincvar_sub_rlmv3 = filter(rincvar_sub_rl), coef(1 1 1) lags(-1/1) normalise 
egen pvarmv3 = filter(prisk2_sub), coef(1 1 1) lags(-1/1) normalise 
egen tvarmv3 = filter(trisk2_sub), coef(1 1 1) lags(-1/1) normalise
egen rincvarmv3 = filter(rincvar), coef(1 1 1) lags(-1/1) normalise


twoway (tsline rincvarmv3 if educ==3,lwidth(thick)) ///
       (tsline tvarmv3 if educ==3,lcolor(red) lwidth(thick)), ///
	    xtitle("date") ///
	   ytitle("") ///
	   ytitle("") ///
	   legend(pos(6) label(1 "perceived") label(2 "realized transitory risk(RHS)") col(2)) ///
	   title("Perceived and realized transitory risk") 
graph export "${graph_folder}/sipp/real_transitory_by_edu_compare.png",as(png) replace  

 
twoway (tsline rincvarmv3 if educ==3,lwidth(thick)) ///
       (tsline pvarmv3 if educ==3,lcolor(red) lwidth(thick)), ///
	   xtitle("date") ///
	   ytitle("") ///
	   ytitle("") ///
	   	legend(pos(6) label(1 "perceived") label(2 "realized permanent risk(RHS)") col(2)) ///
	   title("Perceived and realized permanent risk") 
graph export "${graph_folder}/sipp/real_permanent_by_edu_compare.png",as(png) replace 


twoway  (tsline rincvarmv3 if educ==3,lwidth(thick)) ///
       (tsline rincvar_sub_rlmv3 if educ==3,lcolor(red) lwidth(thick)), ///
	    xtitle("date") ///
	   ytitle("") ///
	   legend(pos(6) label(1 "perceived") label(2 "realized volatility (RHS)") col(2)) ///
	   title("Perceived and realized volatility") 
graph export "${graph_folder}/sipp/real_volatility_by_edu_compare.png",as(png) replace
 
restore
*/

********************
** Regression ******
********************


global other_control i.gender i.Q34 Q35_1 Q35_2 Q35_3 Q35_4 Q35_5 Q35_6 
global macro_ex_var UEprobAgg Stkprob Q9_mean UEprobInd

eststo clear

foreach mom in var iqr mean{
eststo: reg inc`mom' i.age_g i.edu_g i.HHinc_g i.byear_g i.year i.state_id, robust 
eststo: reg inc`mom' i.age_g i.edu_g i.HHinc_g i.byear_g i.year i.state_id ${other_control}, robust 
eststo: reg inc`mom' i.age_g i.edu_g i.HHinc_g i.byear_g i.year i.state_id ${other_control} ${macro_ex_var}, robust 
}

esttab using "${sum_table_folder}/mom_group.csv", ///
             se r2 drop(0.age_g 1.edu_g 0.HHinc_g 0.byear_g  *.year *state_id 1.gender 1.Q34 _cons) ///
			 label replace

*/	
log close 
