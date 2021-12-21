**************************************************************
*! PSID data cleaning
*! Last modified: May1 2020 by Tao 
**************************************************************

global datafolder "/Users/Myworld/Dropbox/PSID/J276289/"
global scefolder "/Users/Myworld/Dropbox/IncExpProject/WorkingFolder/SurveyData/SCE/"
global otherdatafolder "/Users/Myworld/Dropbox/IncExpProject/WorkingFolder/OtherData/"
global table_folder "/Users/Myworld/Dropbox/IncExpProject/WorkingFolder/OtherData/PSID/"
global graph_folder "/Users/Myworld/Dropbox/IncExpProject/WorkingFolder/Graphs/psid/"

cd ${datafolder}

capture log close
clear all
set more off

***************
** CPI data
***************

import delimited "${otherdatafolder}CPIAUCSL.csv"
gen year_str = substr(date,1,4)
destring year_str,force replace
rename year_str year
drop date 
save "${otherdatafolder}cpiY.dta",replace 


*************************
** UE and recession data
************************
clear
import delimited "${otherdatafolder}UNRATE.csv"

gen year_str = substr(date,1,4)
destring year_str,force replace
rename year_str year
drop date 
save "${otherdatafolder}UNRATE.dta",replace 


****************
** PSID data
***************

use "psidY.dta",clear 


merge m:1 year using "${otherdatafolder}cpiY.dta",keep(master match)
drop _merge 
rename cpiaucsl CPI

merge m:1 year using "${otherdatafolder}UNRATE.dta", keep(master match) 
drop _merge 
rename unrate ue

** Set the panel 
xtset uniqueid year 

*************************
** drop variables
***********************
drop if wage_h == 9999999

drop if laborinc_h == 9999999

drop if sex_h ==0 

drop if age_h ==999
drop if occupation_h ==9999
drop if rtoh ==98 | rtoh ==0
drop if race_h == 9 | race_h ==0


** fill education with past values 

replace edu_i = l1.edu_i if year==1969 & sex_h ==l1.sex_h & ///
                            age_h ==l1.age_h+1 & race_h ==l1.race_h
replace edu_i = l1.edu_i if year==1970 & sex_h ==l1.sex_h & ///
                            age_h ==l1.age_h+1 & race_h ==l1.race_h	
replace edu_i = l1.edu_i if year==1971 & sex_h ==l1.sex_h & ///
                            age_h ==l1.age_h+1 & race_h ==l1.race_h	
replace edu_i = l1.edu_i if year==1972 & sex_h ==l1.sex_h & ///
                            age_h ==l1.age_h+1 & race_h ==l1.race_h	
replace edu_i = l1.edu_i if year==1973 & sex_h ==l1.sex_h & ///
                            age_h ==l1.age_h+1 & race_h ==l1.race_h	
replace edu_i = l1.edu_i if year==1974 & sex_h ==l1.sex_h & ///
                            age_h ==l1.age_h+1 & race_h ==l1.race_h	
							
drop if edu_i ==99 | edu_i==98 | edu_i ==0

drop if year<=1970  

** wage is a category variable before 1970

***********************
** drop observations 
***********************

* farm workers 
drop if occupation_h>=600 & occupation<=613
drop if occupation_h >=800 & occupation_h <=802

************************
** education group
**********************

gen edu_i_g =.
replace edu_i_g =1 if edu_i<12
replace edu_i_g =2 if edu_i>=12 & edu_i<16
replace edu_i_g = 3 if edu_i>=16 

*************************
** some label values
*************************

label define race_h_lb 1 "white" 2 "black" 3 "american indian" ///
                    4 "asian/pacific" 5 "latino" 6 "color no black/white" ///
					7 "other"
label values race_h race_h_lb

label define sex_h_lb 1 "male" 2 "female"
label values sex_h sex_h_lb

label define edu_i_g_lb 1 "HS dropout" 2 "HS graduate" 3 "college graduates/above"
label values edu_i_g edu_i_g_lb


***********************
** other filters ******
***********************

by uniqueid: gen tenure = _N if wage_h!=.

*drop latino family after 1990
drop if race_h ==5 | race_h ==6

* only household head 
keep if (rtoh ==1 & year<=1982) | (rtoh ==10 & year>1982)
* head is 1 before 1982 and 10 after 1982

* age 
drop if age_h <20 | age_h > 58 

* stay in sample for at least 9 years
keep if tenure >=15

*******************
** new variables
*******************

** year of birth 

gen byear = year-age
label var byear "year of birth"

egen byear_5yr = cut(byear), ///
     at(1915 1920 1925 1930 1935 1940 ///
	    1945 1950 1955 1960 1965 1970 ///
	    1975 1980 1985 1990 1995 2000 ///
		2005 2010)
label var byear_5yr "5-year cohort"

egen age_5yr = cut(age), ///
     at(20 25 30 35 40 45 ///
	    50 55 60)
label var age_5yr "5-year age"

** nominal to real terms 
replace wage_h = wage_h*CPI/100
label var wage_h "real wage"

replace laborinc_h = laborinc_h*CPI/100
label var laborinc_h "real labor income"

** take the log
gen lwage_h =log(wage_h)
label var lwage_h "log wage"
gen llbinc_h = log(laborinc_h)
label var llbinc_h "log labor income"

** age square
gen age_h2 = age_h^2
label var age_h2 "age squared"

** demean the data
egen lwage_h_av = mean(lwage_h), by(year) 
egen lwage_h_sd = sd(lwage_h), by(year)

*egen laborinc_h_av = mean(llbinc_h), by(year) 
*egen laborinc_h_sd = sd(llbinc_h), by(year)

** mincer regressions 
reghdfe lwage_h age_h age_h2, a(i.sex_h i.edu_i_g i.occupation_h) resid
predict lwage_shk, residuals 
* including aggregate shock

reghdfe lwage_h age_h age_h2, a(i.year i.edu_i_g i.sex_h i.occupation_h) resid
predict lwage_id_shk, residuals

gen lwage_ag_shk = lwage_shk- lwage_id_shk

label var lwage_shk "log wage shock"
label var lwage_id_shk "log wage idiosyncratic shock"
label var lwage_ag_shk "log wage aggregate shock"

** first difference

foreach var in lwage_shk lwage_id_shk lwage_ag_shk{
gen `var'_gr = `var'- l1.`var' if uniqueid==l1.uniqueid ///
                                   & sex_h ==l1.sex_h & ///
								   age_h ==l1.age_h+1 & year==l.year+1
replace `var'_gr = (`var'-l2.`var')/2 if year>=1999 ///
                                   & `var'_gr ==. ///
                                   & uniqueid==l2.uniqueid ///
                                   & sex_h ==l2.sex_h & ///
								   age_h ==l2.age_h+2 & year==l2.year+2
}
label var lwage_shk_gr "log growth of unexplained wage"
label var lwage_id_shk_gr "log growth of idiosyncratic unexplained wage"
label var lwage_ag_shk_gr "log growth of aggregate unexplained wage"


foreach var in ue{
gen ue_gr = ue-l2.ue
}
label var ue_gr "change in ue in 2 year"

** gross volatility 

egen lwage_shk_gr_sd = sd(lwage_shk_gr), by(year)
label var lwage_shk_gr_sd "standard deviation of log shocks"

*egen laborinc_shk_gr_sd = sd(laborinc_shk_gr), by(year)
*label var laborinc_shk_gr_sd "standard deviation of log labor income shocks"


***********************************************
** summary chart of unconditional wages ********
************************************************
/*
preserve

collapse (mean) lwage_h lwage_h_sd laborinc_h_av laborinc_h_sd, by(year) 

** average log household income whole sample
twoway  (connected laborinc_h_av year) if year<=1990, title("The mean of log real labor income")
twoway  (connected laborinc_h_sd year)  if year<=1990, title("The std of log real labor income") 

** average log wage whole sample
twoway  (connected lwage_h year) if lwage_h!=., title("The mean of log real wages") 
graph export "${graph_folder}/log_wage_av.png", as(png) replace 

** std log wage whole sample
twoway  (connected lwage_h_sd year) if lwage_h_sd!=., title("The standard deviation of log real wages") 
graph export "${graph_folder}/log_wage_sd.png", as(png) replace 
restore 
ddd

preserve 
collapse (mean) lwage_h lwage_h_sd lwage_h_av_educ=lwage_h (sd) lwage_h_sd_educ = lwage_h, by(year edu_i_g) 

* average log wage
twoway  (connected lwage_h_av_educ year if lwage_h_av!=. & edu_i_g==1) ///
        (connected lwage_h_av_educ year if lwage_h_av!=. & edu_i_g==2) ///
		(connected lwage_h_av_educ year if lwage_h_av!=. & edu_i_g==3), ///
        title("The mean of log real wages") ///
		legend(label(1 "HS dropout") label(2 "HS") label(3 "college") col(1)) 
graph export "${graph_folder}/log_wage_av_by_edu.png", as(png) replace 

* standard deviation log wage
twoway  (connected lwage_h_sd_educ year if lwage_h_sd_educ!=. & edu_i_g==1) ///
        (connected lwage_h_sd_educ year if lwage_h_sd_educ!=. & edu_i_g==2) ///
		(connected lwage_h_sd_educ year if lwage_h_sd_educ!=. & edu_i_g==3), ///
        title("The standard deviation of log real wages") ///
		legend(label(1 "HS dropout") label(2 "HS") label(3 "college") col(1)) 
graph export "${graph_folder}/log_wage_sd_by_edu.png", as(png) replace 

restore 

************************************************
** summary chart of conditional wages ********
************************************************


preserve

collapse (mean) lwage_shk_gr lwage_shk_gr_sd, by(year) 

*twoway  (connected laborinc_shk_gr year) if year<=1990, title("The mean of log real labor incomes shocks") 
*twoway  (connected laborinc_shk_gr_sd year) if year<=1990, title("The standard deviation of log real labor incomes shocks") 

** average log wage shock whole sample
twoway  (connected lwage_shk_gr year) if lwage_shk_gr!=., title("The mean of log real wage shocks") 
graph export "${graph_folder}/log_wage_shk_gr.png", as(png) replace 

** std log wage whole sample
twoway  (connected lwage_shk_gr_sd year) if lwage_shk_gr_sd!=., title("The standard deviation of log real wage shocks") 
graph export "${graph_folder}/log_wage_shk_gr_sd.png", as(png) replace
restore 

* education profile bar
preserve 
collapse (mean) lwage_shk_gr_av_edu=lwage_shk_gr (sd) lwage_shk_gr_sd_edu = lwage_shk_gr, by(edu_i_g) 
* average log wage

* standard deviation log wage
graph bar lwage_shk_gr_sd_edu, over(edu_i_g) ///
                               ytitle("standard deviation of log wage shocks") ///
                               title("Gross volatility and education") 							   
graph export "${graph_folder}/log_wage_shk_gr_sd_bar_by_edu.png", as(png) replace 
restore 


* education profile: time series 

preserve 
collapse (mean) lwage_shk_gr_av_edu=lwage_shk_gr (sd) lwage_shk_gr_sd_edu = lwage_shk_gr, by(year edu_i_g) 
* average log wage

twoway  (connected lwage_shk_gr_av_edu year if lwage_shk_gr_av_edu!=. & edu_i_g==1) ///
        (connected lwage_shk_gr_av_edu year if lwage_shk_gr_av_edu!=. & edu_i_g==2) ///
		(connected lwage_shk_gr_av_edu year if lwage_shk_gr_av_edu!=. & edu_i_g==3), ///
        title("The mean of log real wage shocks") ///
		legend(label(1 "HS dropout") label(2 "HS") label(3 "college") col(1)) 
graph export "${graph_folder}/log_wage_shk_gr_by_edu.png", as(png) replace 

* standard deviation log wage

twoway  (connected lwage_shk_gr_sd_edu year if lwage_shk_gr_sd_edu!=. & edu_i_g==1) ///
        (connected lwage_shk_gr_sd_edu year if lwage_shk_gr_sd_edu!=. & edu_i_g==2) ///
		(connected lwage_shk_gr_sd_edu year if lwage_shk_gr_sd_edu!=. & edu_i_g==3), ///
        title("The standard deviation of log real wage shocks") ///
		legend(label(1 "HS dropout") label(2 "HS") label(3 "college") col(1)) 
graph export "${graph_folder}/log_wage_shk_gr_sd_by_edu.png", as(png) replace 
 
restore 
*/


*****************************************************************************
**** comparison and perceptions and realizations for idiosyncratic shocks ***
****************************************************************************

** notice here we use lwage_id_shk_gr !!

** byear_5yr and age

preserve
* average log wage
collapse (count) ct = lwage_shk_gr ///
         (mean) lwage_shk_gr_av_byear= lwage_id_shk_gr ///
         (sd) lwage_shk_gr_sd_byear= lwage_id_shk_gr ///
		 (mean) lwage_shk_av_byear= lwage_id_shk ///
         (sd) lwage_shk_sd_byear = lwage_id_shk, by(byear_5yr age_h)
gen age = age_h
summarize ct
keep if ct >10
merge 1:1 byear_5yr age using "${scefolder}incvar_by_byear_5_yr_age.dta",keep(match) 

gen lincvar = sqrt(incvar)
gen lrincvar = sqrt(rincvar)

reg lrincvar lwage_shk_gr_sd_byear

* standard deviation real log wage growth and risk perception

twoway (scatter lwage_shk_gr_sd_byear byear_5yr, color(ltblue)) ///
       (lfit lwage_shk_gr_sd_byear byear_5yr, lcolor(red)) ///
       (scatter lrincvar byear_5yr, color(gray) yaxis(2)) ///
	   (lfit lrincvar byear_5yr,lcolor(black) yaxis(2)), ///
       title("Realized Volatility and Perceived Risks by Cohort/Age")  ///
	   xtitle("year of birth")  ///
	   ytitle("income volatility (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0))  ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
graph export "${graph_folder}/real_log_wage_shk_gr_by_byear_age_compare.png", as(png) replace 


* standard deviation log wage level and risk perception 

twoway (scatter lwage_shk_sd_byear byear_5yr, color(ltblue)) ///
       (lfit lwage_shk_sd_byear byear_5yr, lcolor(red)) ///
       (scatter lrincvar byear_5yr, color(gray) yaxis(2)) ///
	   (lfit lrincvar byear_5yr,lcolor(black) yaxis(2)), ///
       title("Realized Inequality and Perceived Risks by Cohort/Age")  ///
	   xtitle("year of birth")  ///
	   ytitle("income inequality (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
graph export "${graph_folder}/real_log_wage_shk_by_byear_age_compare.png", as(png) replace 

restore

** cohort and gender 

preserve
* average log wage
collapse (mean) lwage_shk_gr_av_byear= lwage_id_shk_gr ///
         (sd) lwage_shk_gr_sd_byear= lwage_id_shk_gr ///
		 (mean) lwage_shk_av_byear= lwage_id_shk ///
         (sd) lwage_shk_sd_byear = lwage_id_shk, by(byear sex_h)

gen gender = sex_h 

merge 1:1 byear gender using "${scefolder}incvar_by_byear_gender.dta",keep(match) 

gen lincvar = sqrt(incvar)
gen lrincvar = sqrt(rincvar)

keep if gender==1

* standard deviation real log wage growth and risk perception

twoway (scatter lwage_shk_gr_sd_byear byear, color(ltblue)) ///
       (lfit lwage_shk_gr_sd_byear byear, lcolor(red)) ///
       (scatter lrincvar byear, color(gray) yaxis(2)) ///
	   (lfit lrincvar byear,lcolor(black) yaxis(2)), ///
       title("Realized Volatility and Perceived Risks by Cohort/Gender")  ///
	   xtitle("year of birth")  ///
	   ytitle("income volatility (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
graph export "${graph_folder}/real_log_wage_shk_gr_by_byear_gender_compare.png", as(png) replace 


* standard deviation log wage level and risk perception 

twoway (scatter lwage_shk_sd_byear byear, color(ltblue)) ///
       (lfit lwage_shk_sd_byear byear, lcolor(red)) ///
       (scatter lrincvar byear, color(gray) yaxis(2)) ///
	   (lfit lrincvar byear,lcolor(black) yaxis(2)), ///
       title("Realized Inequality and Perceived Risks by Cohort/Gender")  ///
	   xtitle("year of birth")  ///
	    ytitle("income inequality (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
graph export "${graph_folder}/real_log_wage_shk_by_byear_gender_compare.png", as(png) replace 

restore


** cohort 

preserve
* average log wage
collapse (mean) lwage_shk_gr_av_byear= lwage_id_shk_gr ///
         (sd) lwage_shk_gr_sd_byear= lwage_id_shk_gr ///
		 (mean) lwage_shk_av_byear= lwage_id_shk ///
         (sd) lwage_shk_sd_byear = lwage_id_shk, by(byear) 

merge 1:1 byear using "${scefolder}incvar_by_byear.dta",keep(match) 

gen lincvar = sqrt(incvar)
gen lrincvar = sqrt(rincvar)


* standard deviation log wage growth and risk perception

twoway (scatter lwage_shk_gr_sd_byear byear, color(ltblue)) ///
       (lfit lwage_shk_gr_sd_byear byear, lcolor(red)) ///
       (scatter lincvar byear, color(gray) yaxis(2)) ///
	   (lfit lincvar byear,lcolor(black) yaxis(2)), ///
       title("Realized and Perceived Income Risks by Cohort")  ///
	   xtitle("year of birth")  ///
	   ytitle("income inequality (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
graph export "${graph_folder}/log_wage_shk_gr_by_byear_compare.png", as(png) replace 


* standard deviation real log wage growth and risk perception

twoway (scatter lwage_shk_gr_sd_byear byear, color(ltblue)) ///
       (lfit lwage_shk_gr_sd_byear byear, lcolor(red)) ///
       (scatter lrincvar byear, color(gray) yaxis(2)) ///
	   (lfit lrincvar byear,lcolor(black) yaxis(2)), ///
       title("Realized Volatility and Perceived Risks by Cohort")  ///
	   xtitle("year of birth")  ///
	   ytitle("income volatility (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
graph export "${graph_folder}/real_log_wage_shk_gr_by_byear_compare.png", as(png) replace 


* standard deviation log wage level and risk perception 

twoway (scatter lwage_shk_sd_byear byear, color(ltblue)) ///
       (lfit lwage_shk_sd_byear byear, lcolor(red)) ///
       (scatter lrincvar byear, color(gray) yaxis(2)) ///
	   (lfit lrincvar byear,lcolor(black) yaxis(2)), ///
       title("Realized Inequality and Perceived Risks by Cohort")  ///
	   xtitle("year of birth")  ///
	   ytitle("income inequality (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
graph export "${graph_folder}/real_log_wage_shk_by_byear_compare.png", as(png) replace 

restore


** cohort/education profile

preserve
* average log wage
collapse (mean) lwage_shk_gr_av_byear_edu = lwage_id_shk_gr ///
         (sd) lwage_shk_gr_sd_byear_edu = lwage_id_shk_gr ///
		 (mean) lwage_shk_av_byear_edu = lwage_id_shk ///
         (sd) lwage_shk_sd_byear_edu = lwage_id_shk, by(byear_5yr edu_i_g) 

gen edu_g = edu_i_g 

merge 1:1 byear_5yr edu_g using "${scefolder}incvar_by_byear_5yr_edu.dta",keep(match) 

gen lincvar = sqrt(incvar)
gen lrincvar = sqrt(rincvar)

drop if edu_g==1

* standard deviation log wage growth and risk perception 

twoway (scatter lwage_shk_gr_sd_byear_edu byear_5yr, color(ltblue)) ///
       (lfit lwage_shk_gr_sd_byear_edu byear_5yr, lcolor(red)) ///
       (scatter lrincvar byear_5yr, color(gray) yaxis(2)) ///
	   (lfit lrincvar byear_5yr,lcolor(black) yaxis(2)), ///
       title("Realized Volatility and Perceived Risks by Cohort/Education")  ///
	   xtitle("year of birth")  ///
	   ytitle("income volatility (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
					  
graph export "${graph_folder}/real_log_wage_shk_gr_by_byear_5yr_edu_compare.png", as(png) replace 

* standard deviation log wage level and risk perception 

twoway (scatter lwage_shk_sd_byear_edu byear_5yr, color(ltblue)) ///
       (lfit lwage_shk_sd_byear_edu byear_5yr, lcolor(red)) ///
       (scatter lrincvar byear_5yr, color(gray) yaxis(2)) ///
	   (lfit lrincvar byear_5yr,lcolor(black) yaxis(2)), ///
       title("Realized Inequality and Perceived Risks by Cohort/Education")  ///
	   xtitle("year of birth")  ///
	   ytitle("income inequality (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
	   
graph export "${graph_folder}/real_log_wage_shk_by_byear_5yr_edu_compare.png", as(png) replace 

restore


** 5-year cohort/education/gender profile

preserve
* average log wage
collapse (mean) lwage_shk_gr_av_byear_5yr_edu = lwage_id_shk_gr ///
         (sd) lwage_shk_gr_sd_byear_5yr_edu = lwage_id_shk_gr ///
		 (mean) lwage_shk_av_byear_5yr_edu = lwage_id_shk ///
         (sd) lwage_shk_sd_byear_5yr_edu = lwage_id_shk, by(byear_5yr edu_i_g sex_h)
		 
gen edu_g = edu_i_g 
gen gender = sex_h 

merge 1:1 byear_5yr edu_g gender using "${scefolder}incvar_by_byear_5yr_edu_gender.dta",keep(match) 

gen lincvar = sqrt(incvar)
gen lrincvar = sqrt(rincvar)

drop if edu_g==1 
*keep if gender==2

* standard deviation log wage growth and risk perception 
twoway (scatter lwage_shk_gr_sd_byear_5yr_edu byear_5yr, color(ltblue)) ///
       (lfit lwage_shk_gr_sd_byear_5yr_edu byear_5yr, lcolor(red)) ///
       (scatter lrincvar byear_5yr, color(gray) yaxis(2)) ///
	   (lfit lrincvar byear_5yr,lcolor(black) yaxis(2)), ///
       title("Realized Volatility and Perceived Risks by Cohort/Education")  ///
	   xtitle("year of birth")  ///
	   ytitle("income volatility (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
					  
graph export "${graph_folder}/real_log_wage_shk_gr_by_byear_5yr_edu_gender_compare.png", as(png) replace 


* standard deviation log wage level and risk perception 

twoway (scatter lwage_shk_sd_byear_5yr_edu byear_5yr, color(ltblue)) ///
       (lfit lwage_shk_sd_byear_5yr_edu byear_5yr, lcolor(red)) ///
       (scatter lrincvar byear_5yr, color(gray) yaxis(2)) ///
	   (lfit lrincvar byear_5yr,lcolor(black) yaxis(2)), ///
       title("Realized Inequality and Perceived Risks by Cohort/Education")  ///
	   xtitle("year of birth")  ///
	   ytitle("income inequality (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
	   
graph export "${graph_folder}/real_log_wage_shk_by_byear_5yr_edu_gender_compare.png", as(png) replace 

restore


** age profile 
preserve
* average log wage
collapse (mean) lwage_shk_gr_av_age = lwage_id_shk_gr ///
         (sd)   lwage_shk_gr_sd_age = lwage_id_shk_gr ///
		 (mean)	lwage_shk_av_age = lwage_id_shk ///
		 (sd)    lwage_shk_sd_age = lwage_id_shk, by(age_h) 

gen age = age_h 
merge 1:1 age using "${scefolder}incvar_by_age.dta",keep(master match) 
gen lincvar = sqrt(incvar)
gen lrincvar = sqrt(rincvar)

/*
* growth 
twoway (scatter lwage_shk_gr_av_age age_h) (lfit lwage_shk_gr_av_age age_h), ///
       title("Growth rates of log real wage and age") ///
                xtitle("age") 
graph export "${graph_folder}/log_wage_shk_gr_by_age.png", as(png) replace 

* standard deviation log wage
twoway (scatter lwage_shk_gr_sd_age age_h) (lfit lwage_shk_gr_sd_age age_h), ///
       title("Gross volatility of log real wage and age")  ///
	   xtitle("age") 
graph export "${graph_folder}/log_wage_shk_gr_sd_by_age.png", as(png) replace 


* level 
twoway (scatter lwage_shk_av_age age_h) (lfit lwage_shk_av_age age_h), ///
       title("Growth rates of log real wage and age") ///
                xtitle("age") 
graph export "${graph_folder}/log_wage_shk_by_age.png", as(png) replace 


* standard deviation level 
twoway (scatter lwage_shk_sd_age age_h) (lfit lwage_shk_sd_age age_h), ///
       title("Growth rates of log real wage and age") ///
                xtitle("age") 
graph export "${graph_folder}/log_wage_shk_sd_by_age.png", as(png) replace 
*/

* standard deviation log wage and real risk perception 

twoway (scatter lwage_shk_gr_sd_age age_h, color(ltblue)) ///
       (lfit lwage_shk_gr_sd_age age_h, lcolor(red)) ///
       (scatter lrincvar age_h, color(gray) yaxis(2)) ///
	   (lfit lrincvar age_h,lcolor(black) yaxis(2)), ///
       title("Realized Volatility and Perceived Risks by Age")  ///
	   xtitle("age")  ///
	    ytitle("income volatility (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
graph export "${graph_folder}/real_log_wage_shk_gr_by_age_compare.png", as(png) replace 

** age-specific level risks and perceptions

twoway (scatter lwage_shk_sd_age age_h, color(ltblue)) ///
       (lfit lwage_shk_sd_age age_h, lcolor(red)) ///
       (scatter lrincvar age_h, color(gray) yaxis(2)) ///
	   (lfit lrincvar age_h,lcolor(black) yaxis(2)), ///
       title("Realized Inequality Perceived Risks by Age")  ///
	   xtitle("age")  ///
	   ytitle("income inequality (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
graph export "${graph_folder}/real_log_wage_shk_by_age_compare.png", as(png) replace 



** age-specific level risks and perceptions 
 
twoway (scatter lrincvar lwage_shk_gr_sd_age, color(ltblue)) ///
       (lfit lrincvar lwage_shk_gr_sd_age, lcolor(red)), ///
       title("Realized Volatility and Perceived Risks")  ///
	   xtitle("Age-specific volatility") ///
	   ytitle("Perceived risk") 
graph export "${graph_folder}/real_realized_perceived_risks_by_age.png", as(png) replace 
*/

restore


** age x education profile

preserve
* average log wage
collapse (mean) lwage_shk_gr_av_age_edu = lwage_id_shk_gr ///
         (sd) lwage_shk_gr_sd_age_edu = lwage_id_shk_gr ///
		 (mean) lwage_shk_av_age_edu = lwage_id_shk ///
         (sd) lwage_shk_sd_age_edu = lwage_id_shk, by(age_h edu_i_g) 

gen age = age_h 
gen edu_g = edu_i_g 
keep if edu_g !=1

merge 1:1 age edu_g using "${scefolder}incvar_by_age_edu.dta",keep(master match) 

gen lincvar = sqrt(incvar)
gen lrincvar = sqrt(rincvar)

* standard deviation log wage growth and risk perception 
twoway (scatter lwage_shk_gr_sd_age_edu age_h, color(ltblue)) ///
       (lfit lwage_shk_gr_sd_age_edu age_h, lcolor(red)) ///
       (scatter lrincvar age_h, color(gray) yaxis(2)) ///
	   (lfit lrincvar age_h,lcolor(black) yaxis(2)), ///
       title("Realized Volatility and Perceived Risks by Age/Education")  ///
	   xtitle("age")  ///
	   ytitle("income volatility (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
graph export "${graph_folder}/real_log_wage_shk_gr_by_age_edu_compare.png", as(png) replace 


* standard deviation log wage level and risk perception 
twoway (scatter lwage_shk_sd_age_edu age_h, color(ltblue)) ///
       (lfit lwage_shk_sd_age_edu age_h, lcolor(red)) ///
       (scatter lrincvar age_h, color(gray) yaxis(2)) ///
	   (lfit lrincvar age_h,lcolor(black) yaxis(2)), ///
       title("Realized Inequality and Perceived Risks by Age/Education")  ///
	   xtitle("age")  ///
	   ytitle("income inequality (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
graph export "${graph_folder}/real_log_wage_shk_by_age_edu_compare.png", as(png) replace 


* perceived riks by age/education  
twoway (scatter lrincvar lwage_shk_gr_sd_age, color(ltblue)) ///
       (lfit lrincvar lwage_shk_gr_sd_age, lcolor(red)) if lwage_shk_gr_sd!=., ///
       title("Realized Volatility and Perceived Risks by Age/Education")  ///
	   xtitle("Volatility") ///
	   ytitle("Perceived risk") 
graph export "${graph_folder}/real_realized_perceived_risks_by_age_edu.png", as(png) replace  


restore

/*
** age/gender profile 

** scatter 
preserve 
collapse (mean) lwage_shk_gr_av_age_sex = lwage_id_shk_gr ///
         (sd) lwage_shk_gr_sd_age_sex = lwage_id_shk_gr ///
		 (mean) lwage_shk_av_age_sex = lwage_id_shk ///
         (sd) lwage_shk_sd_age_sex = lwage_id_shk, by(age_h sex_h) 
gen age = age_h

gen gender = sex_h

merge 1:1 age gender using "${scefolder}incvar_by_age_gender.dta",keep(master match) 
gen lincvar = sqrt(incvar)
gen lrincvar = sqrt(rincvar)

* standard deviation log wage growth and risk perception 
twoway (scatter lwage_shk_gr_sd_age_sex age_h, color(ltblue)) ///
       (lfit lwage_shk_gr_sd_age_sex age_h, lcolor(red)) ///
       (scatter lrincvar age_h, color(gray) yaxis(2)) ///
	   (lfit lrincvar age_h,lcolor(black) yaxis(2)), ///
       title("Realized Volatility and Perceived Risks by Age/Gender")  ///
	   xtitle("age")  ///
	   ytitle("income volatility (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
graph export "${graph_folder}/real_log_wage_shk_gr_by_age_gender_compare.png", as(png) replace 

* standard deviation log wage level and risk perception 
twoway (scatter lwage_shk_sd_age_sex age_h, color(ltblue)) ///
       (lfit lwage_shk_sd_age_sex age_h, lcolor(red)) ///
       (scatter lrincvar age_h, color(gray) yaxis(2)) ///
	   (lfit lrincvar age_h,lcolor(black) yaxis(2)), ///
       title("Realized Inequality and Perceived Risks by Age/Gender")  ///
	   xtitle("age")  ///
	   ytitle("income inequality (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
graph export "${graph_folder}/real_log_wage_shk_by_age_gender_compare.png", as(png) replace 

restore
*/

** age/gender/educ profile 

** scatter 
preserve 
collapse (mean) lwage_shk_gr_av_age_sex = lwage_id_shk_gr ///
         (sd) lwage_shk_gr_sd_age_sex = lwage_id_shk_gr ///
		 (mean) lwage_shk_av_age_sex = lwage_id_shk ///
         (sd) lwage_shk_sd_age_sex = lwage_id_shk, by(age_h sex_h edu_i_g) 
gen age = age_h
gen gender = sex_h
gen edu_g = edu_i_g 
drop if edu_g ==1

merge 1:1 age gender edu_g using "${scefolder}incvar_by_age_edu_gender.dta",keep(master match) 
gen lincvar = sqrt(incvar)
gen lrincvar = sqrt(rincvar)

* standard deviation log wage growth and risk perception 
twoway (scatter lwage_shk_gr_sd_age_sex age_h, color(ltblue)) ///
       (lfit lwage_shk_gr_sd_age_sex age_h, lcolor(red)) ///
       (scatter lrincvar age_h, color(gray) yaxis(2)) ///
	   (lfit lrincvar age_h,lcolor(black) yaxis(2)), ///
       title("Realized Volatility and Perceived Risks by Age/Gender/Educ")  ///
	   xtitle("age")  ///
	   ytitle("income volatility (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
graph export "${graph_folder}/real_log_wage_shk_gr_by_age_edu_gender_compare.png", as(png) replace 

* standard deviation log wage level and risk perception 
twoway (scatter lwage_shk_sd_age_sex age_h, color(ltblue)) ///
       (lfit lwage_shk_sd_age_sex age_h, lcolor(red)) ///
       (scatter lrincvar age_h, color(gray) yaxis(2)) ///
	   (lfit lrincvar age_h,lcolor(black) yaxis(2)), ///
       title("Realized Inequality and Perceived Risks by Age/Gender/Educ")  ///
	   xtitle("age")  ///
	   ytitle("income inequality (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
graph export "${graph_folder}/real_log_wage_shk_by_age_edu_gender_compare.png", as(png) replace 

restore

/*
** time series 

preserve 
collapse (mean) lwage_shk_gr_av_sex=lwage_id_shk_gr ///
         (sd) lwage_shk_gr_sd_sex = lwage_id_shk_gr, by(year sex_h) 
* average log wage

twoway  (connected lwage_shk_gr_av_sex year if lwage_shk_gr_av_sex!=. & sex_h==1) ///
        (connected lwage_shk_gr_av_sex year if lwage_shk_gr_av_sex!=. & sex_h==2), ///
        title("The mean of log real wage shocks") ///
		legend(label(1 "male") label(2 "female") col(1)) 
graph export "${graph_folder}/log_wage_shk_gr_by_sex.png", as(png) replace 

* standard deviation log wage

twoway  (connected lwage_shk_gr_sd_sex year if lwage_shk_gr_sd_sex!=. & sex_h==1) ///
        (connected lwage_shk_gr_sd_sex year if lwage_shk_gr_sd_sex!=. & sex_h==2), ///
        title("The standard deviation of log real wage shocks") ///
		legend(label(1 "male") label(2 "female") col(1)) 
graph export "${graph_folder}/log_wage_shk_gr_sd_by_sex.png", as(png) replace

restore 
*/

********************************************
** Unconditional summary statistics *****
*****************************************

tabstat lwage_shk_gr, st(sd) by(edu_i_g)
tabstat lwage_shk_gr, st(sd) by(age_h)
tabstat lwage_shk_gr, st(sd) by(sex_h)

********************************************
** Prepare the matrix for GMM estimation
*****************************************

preserve 
tsset uniqueid year 
tsfill,full
*replace lwage_shk_gr = f1.lwage_shk_gr if year>=1998 ///
*                                       & f1.lwage_shk_gr !=. ///
*									   & lwage_shk_gr ==. 									    
keep uniqueid year lwage_id_shk_gr edu_i_g sex_h age_5yr
*keep if year<=1997
reshape wide lwage_id_shk_gr, i(uniqueid edu_i_g sex_h age_5yr) j(year)
save "psid_matrix.dta",replace 
restore 
ddd
******************************
** cohort-time-specific experience
*********************************

preserve 

putexcel set "${table_folder}/psid_history_vol.xls", sheet("") replace
putexcel A1=("year") B1=("cohort") C1=("av_shk_gr") D1=("var_shk") E1=("av_id_shk_gr") ///
         F1=("var_id_shk") G1=("av_ag_shk_gr") H1=("var_ag_shk") I1 =("N") J1=("ue_av") K1=("ue_var") ///
		 L1=("var_shk_gr") M1=("var_id_shk_gr") N1=("var_ag_shk_gr")
local row = 2
forvalues t =1973(1)2017{
local l = `t'-1971
forvalues i = 2(1)`l'{
*quietly: reghdfe lwage_h edu_i age_h age_h2 if year <=`t'& year>=`t'-`i', a(year sex_h occupation_h) resid

** shk
summarize lwage_shk if year <=`t'& year>=`t'-`i'
return list 
local N = r(N)
local var_shk = r(sd)^2
disp `var_shk'
summarize lwage_shk_gr if year <=`t'& year>=`t'-`i'
return list 
local av_shk_gr = r(mean)
disp `av_shk_gr'
local var_shk_gr = r(sd)^2
disp `var_shk_gr'


** id shk
summarize lwage_id_shk if year <=`t'& year>=`t'-`i'
return list 
local var_id_shk = r(sd)^2
disp `var_id_shk'
summarize lwage_id_shk_gr if year <=`t'& year>=`t'-`i'
return list 
local av_id_shk_gr = r(mean)
disp `av_id_shk_gr'
local var_id_shk_gr = r(sd)^2
disp `var_id_shk_gr'


** ag shk
summarize lwage_ag_shk if year <=`t'& year>=`t'-`i'
return list 
local var_ag_shk = r(sd)^2
disp `var_ag_shk'
summarize lwage_ag_shk_gr if year <=`t'& year>=`t'-`i'
return list 
local av_ag_shk_gr = r(mean)
disp `av_ag_shk_gr'
local var_ag_shk_gr = r(sd)^2
disp `var_ag_shk_gr'

** ag ue
summarize ue if year <=`t'& year>=`t'-`i'
return list 
local ue_av = r(mean)
disp `ue_av'
local ue_var = r(sd)^2
disp `ue_var'

putexcel A`row'=("`t'")
local c = `t'-`i'
putexcel B`row'=("`c'")
putexcel C`row'=("`av_shk_gr'")
putexcel D`row'=("`var_shk'")
putexcel E`row'=("`av_id_shk_gr'")
putexcel F`row'=("`var_id_shk'")
putexcel G`row'=("`av_ag_shk_gr'")
putexcel H`row'=("`var_ag_shk'")
putexcel I`row'=("`N'")
putexcel J`row'=("`ue_av'")
putexcel K`row'=("`ue_var'")
putexcel L`row'=("`var_shk_gr'")
putexcel M`row'=("`var_id_shk_gr'")
putexcel N`row'=("`var_ag_shk_gr'")

local ++row
}
}
restore


**** age-time-education 

preserve 

putexcel set "${table_folder}/psid_history_vol_edu.xls", sheet("") replace
putexcel A1=("year") B1=("cohort") C1 =("edu") D1=("av_shk_gr") E1=("var_shk") F1=("av_id_shk_gr") ///
         G1=("var_id_shk") H1=("av_ag_shk_gr") I1=("var_ag_shk") J1 =("N") K1=("ue_av") L1=("ue_var") ///
		 M1=("var_shk_gr") N1=("var_id_shk_gr") O1=("var_ag_shk_gr")
local row = 2
forvalues ed = 1(1)3{
forvalues t =1973(1)2017{
local l = `t'-1971
forvalues i = 2(1)`l'{
*quietly: reghdfe lwage_h edu_i age_h age_h2 if year <=`t'& year>=`t'-`i', a(year sex_h occupation_h) resid

** shk
summarize lwage_shk if year <=`t'& year>=`t'-`i' & edu_i_g ==`ed'
return list 
local N = r(N)
local var_shk = r(sd)^2
disp `var_shk'
summarize lwage_shk_gr if year <=`t'& year>=`t'-`i' & edu_i_g ==`ed'
return list 
local av_shk_gr = r(mean)
disp `av_shk_gr'
local var_shk_gr = r(sd)^2
disp `var_shk_gr'


** id shk
summarize lwage_id_shk if year <=`t'& year>=`t'-`i' & edu_i_g ==`ed'
return list 
local var_id_shk = r(sd)^2
disp `var_id_shk'
summarize lwage_id_shk_gr if year <=`t'& year>=`t'-`i' & edu_i_g ==`ed'
return list 
local av_id_shk_gr = r(mean)
disp `av_id_shk_gr'
local var_id_shk_gr = r(sd)^2
disp `var_id_shk_gr'

** ag shk
summarize lwage_ag_shk if year <=`t'& year>=`t'-`i'& edu_i_g ==`ed'
return list 
local var_ag_shk = r(sd)^2
disp `var_ag_shk'
summarize lwage_ag_shk_gr if year <=`t'& year>=`t'-`i'& edu_i_g ==`ed'
return list 
local av_ag_shk_gr = r(mean)
disp `av_ag_shk_gr'
local var_ag_shk_gr = r(sd)^2
disp `var_ag_shk_gr'

** ag ue
summarize ue if year <=`t'& year>=`t'-`i' & edu_i_g ==`ed'
return list 
local ue_av = r(mean)
disp `ue_av'
local ue_var = r(sd)^2
disp `ue_var'

putexcel A`row'=("`t'")
local c = `t'-`i'
putexcel B`row'=("`c'")
putexcel C`row'=("`ed'")
putexcel D`row'=("`av_shk_gr'")
putexcel E`row'=("`var_shk'")
putexcel F`row'=("`av_id_shk_gr'")
putexcel G`row'=("`var_id_shk'")
putexcel H`row'=("`av_ag_shk_gr'")
putexcel I`row'=("`var_ag_shk'")
putexcel J`row'=("`N'")
putexcel K`row'=("`ue_av'")
putexcel L`row'=("`ue_var'")
putexcel M`row'=("`var_shk_gr'")
putexcel N`row'=("`var_id_shk_gr'")
putexcel O`row'=("`var_ag_shk_gr'")

local ++row
}
}
}
restore



*/


