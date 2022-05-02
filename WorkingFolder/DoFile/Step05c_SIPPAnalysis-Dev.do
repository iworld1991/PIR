*************************************************************
*! SIPP data cleaning
*! Last modified: Feb 2022 by Tao 
**************************************************************
global datafolder "/Users/Myworld/Dropbox/SIPP/"
global scefolder "/Users/Myworld/Dropbox/PIR/WorkingFolder/SurveyData/SCE/"
global otherdatafolder "/Users/Myworld/Dropbox/PIR/WorkingFolder/OtherData/"
global table_folder "/Users/Myworld/Dropbox/PIRder/OtherData/sipp/"
global graph_folder "/Users/Myworld/Dropbox/PIR/WorkingFolder/Graphs/sipp/"

clear
use "${datafolder}sipp.dta"
describe
count 


**************************
** date and id ***********
***************************


egen uniqueid=group(SSUID PNUM)

gen month = MONTHCODE

gen quarter = .
replace quarter=1 if month==1 | month==2 | month ==3 
replace quarter=2 if month==4 | month==5 | month ==6 
replace quarter=3 if month==7 | month==8 | month ==9 
replace quarter=4 if month==10 | month==11 | month ==12 

gen date_str=string(year)+"M"+string(month)
gen date = monthly(date_str,"YM")
format date %tm
drop date_str

table date

xtset uniqueid date
unique uniqueid

by uniqueid: gen tenure = _N if TJB1_MSUM!=.

************************
*** merge macro data **
************************

merge m:1 year month using "${otherdatafolder}InfM.dta",keep(master match)
drop _merge 


*******************************
***** Creating variables ******
*******************************
sort uniqueid date 

/*
** yearly 
 
gen ntot_earning_jb1 = TJB1_MSUM
label var ntot_earning_jb1 "nominal total monthly earning"

gen ntot_earning_jb1_y = l11.ntot_earning_jb1+ l10.ntot_earning_jb1+ ///
                        l9.ntot_earning_jb1+l8.ntot_earning_jb1+l7.ntot_earning_jb1 + ///
						l6.ntot_earning_jb1+ l5.ntot_earning_jb1 +l4.ntot_earning_jb1 + ///
						l3.ntot_earning_jb1+ l2.ntot_earning_jb1+l1.ntot_earning_jb1+ntot_earning_jb1
						
label var ntot_earning_jb1 "nominal total monthly earning"

gen rtot_earning_jb1 = TJB1_MSUM*100/CPIAU
label var rtot_earning_jb1 "real total monthly earning"

gen rtot_earning_jb1_y = l11.rtot_earning_jb1+ l10.rtot_earning_jb1+ ///
                        l9.rtot_earning_jb1+l8.rtot_earning_jb1+l7.rtot_earning_jb1 + ///
						l6.rtot_earning_jb1+ l5.rtot_earning_jb +l4.rtot_earning_jb1 + ///
						l3.rtot_earning_jb1+ l2.rtot_earning_jb1+l1.rtot_earning_jb1+rtot_earning_jb1
						
label var rtot_earning_jb1_y "yearly real total earning from primary job"

gen tot_hours_jb1 = TJB1_MWKHRS

gen tot_hours_jb1_y = l11.tot_hours_jb1+ l10.tot_hours_jb1+ ///
                        l9.tot_hours_jb1+l8.tot_hours_jb1+l7.tot_hours_jb1 + ///
						l6.tot_hours_jb1+ l5.tot_hours_jb1 +l4.tot_hours_jb1 + ///
						l3.tot_hours_jb1+ l2.tot_hours_jb1+l1.tot_hours_jb1+tot_hours_jb1
label var tot_hours_jb1_y "yearly total hours of working for primary job"

gen wage_Y =  rtot_earning_jb1_y/tot_hours_jb1_y				
label var wage_Y "real yearly wage rate of primary job"

gen wage_n_Y = ntot_earning_jb1_y/tot_hours_jb1_y				
label var wage_n_Y "nominal yearly wage rate of primary job"
*/
				
** monthly 
gen wage = TJB1_MSUM/TJB1_MWKHRS
** or divided by average nb of weeks of work TJB1_MWKHRS
** use the primary job monthly earning for now  
** nominal to real terms 
gen wage_n = wage
label var wage_n "nominal monthly wage rate of the primary job"

replace wage = wage*100/CPIAU
label var wage "real monthly wage rate of the primary job"


**************************
***** Winsorization ******
**************************

egen wage_mean = mean(wage),by(uniqueid)
label var wage_mean "average monthly real earning of the individual"
drop if wage<0.1* wage_mean | wage>1.9*wage_mean


foreach var in wage{
egen `var'_p1 = pctile(`var'),p(1) by(date)
egen `var'_p99 = pctile(`var'),p(99) by(date)
replace `var'=. if `var'<`var'_p1 | `var'>=`var'_p99
}


**************************
***** Validation ******
**************************
xtset uniqueid date

drop if (ESEX!=l1.ESEX & ESEX!=.)| (ERACE!=l1.ERACE & ERACE!=.)
gen AGE_df = TAGE-l1.TAGE
drop if AGE_df >=2

keep if tenure>=4

** conditional on having no days off from the job 
keep if EJB1_AWOP1 ==2

** keep job that continues to interview year 

gen same_job = .
replace same_job=RJB1_CFLG if RJB1_CFLG !=.
replace same_job=RJB1_CONTFLG if same_job==. & RJB1_CONTFLG!=.
keep if same_job==1

** type of employment arrangement 

gen work_type = EJB1_JBORSE
drop if work_type!=1
** only focus on employed by someone else
** already dropp self-employed and others from raw data

** drop imputed values
keep if EINTTYPE==1 | EINTTYPE==2 

** first job industry code 
destring TJB1_IND, force replace

** drop some industries, i.e. government jobs

drop if TJB1_IND>=9400

table AJB1_MSUM
**9 indicates allocation flags for the components

/*
******************************************************
***** Quarterly                                     **
***** Set on if we want to collapse data into quarterly 
*******************************************************


gen wageQ = wage+l1.wage+l2.wage
replace wage =wageQ
drop wageQ
label var wage "real quarterly earning"
** sum m-2 m-1 and m for quarterly wage 

keep if month==3 | month==6 |month==9|month==12

gen date_str=string(year)+"Q"+string(quarter)
gen dateq = quarterly(date_str,"YQ")
format dateq %tq
drop date_str

table dateq
xtset uniqueid dateq
*/




**************************
***** Summary stats ******
**************************

tabstat wage wage_n TPEARN TJB1_MSUM TJB1_MWKHRS, st(p5 p10 p25 p50 p75 p90 p95) 

*******************************
*** Other group variables ***
*******************************

gen age = TAGE
gen gender = ESEX
gen race = ERACE
gen educ = .
replace educ=1 if EEDUC<=38
replace educ=2 if EEDUC>=39 & EEDUC<=42
replace educ=3 if EEDUC>42

**************************
***** label variables ***
*************************

label define race_lb 1 "white" 2 "black" 3 "asian" ///
                    4 "residual"
label values race race_lb

label define gender_lb 1 "male" 2 "female"
label values gender gender_lb

label define educ_lb 1 "HS dropout" 2 "HS graduate" 3 "college graduates/above"
label values educ edu_lb


***********************
** Create new variables
***********************


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

** age square

gen age2 = age^2
label var age2 "age squared"

** age^3
gen age3 = age^3
label var age3 "age 3"
** age^4
gen age4 = age^4
label var age4 "age 4"

** end/begining of year

gen end_yr = cond(month==12,1,0)
label var end_yr "december dummy"

gen beg_yr = cond(month==1,1,0)
label var beg_yr "january dummy"

** take the log
gen lwage_n = log(wage_n) 
label var lwage_n "log nominal monthly wage rate"

gen lwage =log(wage)
label var lwage "log real monthly wage rate"

/*
gen lwage_n_Y = log(wage_n_Y)
label var lwage_n_Y "log nomimal yearly wage rate"

gen lwage_Y = log(wage_Y)
label var lwage_Y "log real yearly wage rate"
*/

** demean the data
egen lwage_av = mean(lwage), by(date) 
egen lwage_sd = sd(lwage), by(date)

************************************
*** Deterministic income component *
***********************************

reghdfe lwage end_yr beg_yr, a(i.race i.gender i.educ i.TJB1_IND) resid
predict lwage_age_1st,residuals
reg lwage_age_1st age age2 age3 age4
predict lwage_age

preserve 
collapse (mean) lwage_age, by(age)
gen wage_age = exp(lwage_age) 
drop lwage_age 
label var wage_age "average wage rate at age t (polynomial regression)"

twoway (connected wage_age age) if age<=65, ///
        xtitle("Age") ///
        title("The deterministic earning profile over life cycle") 
graph export "${graph_folder}/age_profile.png", as(png) replace 

save "${otherdatafolder}age_profile.dta",replace 

restore 


*********************************
*** Stochastic income component *
*********************************

***********************************************************************************
** mincer regressions 
** monthly 
reghdfe lwage age age2 end_yr beg_yr, a(i.race i.gender i.educ i.TJB1_IND) resid
predict lwage_shk, residuals
 
* including aggregate shock
reghdfe lwage age age2 end_yr beg_yr, a(i.date i.race i.gender i.educ i.TJB1_IND) resid
predict lwage_id_shk, residuals

gen lwage_ag_shk = lwage_shk- lwage_id_shk

label var lwage_shk "log wage shock"
label var lwage_id_shk "log wage idiosyncratic shock"
label var lwage_ag_shk "log wage aggregate shock"

/*
** yearly 
reghdfe lwage_Y age age2 end_yr beg_yr, a(i.year i.race i.gender i.educ i.TJB1_IND) resid
predict lwage_Y_id_shk, residuals
label var lwage_Y_id_shk "log wage idiosyncratic shock"
*/
*************************************************************************************

** first difference for monthly 
foreach var in lwage lwage_n lwage_shk lwage_id_shk lwage_ag_shk{
gen `var'_gr = `var'- l1.`var'
}
label var lwage_n_gr "log growth of nominal wage"
label var lwage_gr "log growth of real wage"
label var lwage_shk_gr "log growth of unexplained wage"
label var lwage_id_shk_gr "log growth of idiosyncratic unexplained wage"
label var lwage_ag_shk_gr "log growth of aggregate unexplained wage"


** first difference for monthly 
foreach var in lwage lwage_n lwage_id_shk{
gen `var'_Y_gr = `var'- l12.`var'
}
label var lwage_Y_gr "log yearly growth of nominal wage"
label var lwage_n_Y_gr "log yearly growth of real wage"
label var lwage_id_shk_Y_gr "log yearly growth of idiosyncratic unexplained wage"


*foreach var in ue{
*gen ue_gr = ue-l2.ue
*}
*label var ue_gr "change in ue in 2 year"

**  volatility 
egen lwage_shk_gr_sd = sd(lwage_shk_gr), by(date)
label var lwage_shk_gr_sd "standard deviation of log shocks"

***********************************************
** summary chart of unconditional wages ********
************************************************

** histograms of wage distribution 
hist  wage, title("distribution of real wage rate") 
graph export "${graph_folder}/hist_wage.png", as(png) replace 

/*
hist  wage_Y, title("distribution of real wage rate (yearly)") 
graph export "${graph_folder}/hist_wage_Y.png", as(png) replace 
*/

** time series plots 
preserve
collapse (mean) lwage lwage_sd, by(date year month) 
** average log wage whole sample
twoway  (connected lwage date) if lwage!=., title("The mean of log real wages") 
graph export "${graph_folder}/log_wage_av.png", as(png) replace 

** std log wage whole sample
twoway  (connected lwage_sd date) if lwage_sd!=., title("The standard deviation of log real wages") 
graph export "${graph_folder}/log_wage_sd.png", as(png) replace 
restore 


preserve 
collapse (mean) lwage lwage_sd lwage_av_educ=lwage (sd) lwage_sd_educ = lwage, by(date year month educ) 

* average log wage
twoway  (connected lwage_av_educ date if lwage_av!=. & educ==1) ///
        (connected lwage_av_educ date if lwage_av!=. & educ==2) ///
		(connected lwage_av_educ date if lwage_av!=. & educ==3), ///
        title("The mean of log real wages") ///
		legend(label(1 "HS dropout") label(2 "HS") label(3 "college") col(1)) 
graph export "${graph_folder}/log_wage_av_by_edu.png", as(png) replace 

* standard deviation log wage
twoway  (connected lwage_sd_educ date if lwage_sd_educ!=. & educ==1) ///
        (connected lwage_sd_educ date if lwage_sd_educ!=. & educ==2) ///
		(connected lwage_sd_educ date if lwage_sd_educ!=. & educ==3), ///
        title("The standard deviation of log real wages") ///
		legend(label(1 "HS dropout") label(2 "HS") label(3 "college") col(1)) 
graph export "${graph_folder}/log_wage_sd_by_edu.png", as(png) replace 

restore 

************************************************
** summary chart of conditional wages ********
************************************************

preserve

collapse (mean) lwage_shk_gr lwage_shk_gr_sd, by(year month date) 

replace lwage_shk_gr=. if month==1
replace lwage_shk_gr_sd=. if month==1

** average log wage shock whole sample
twoway  (connected lwage_shk_gr date) if lwage_shk_gr!=., title("The mean of log real wage shocks") 
graph export "${graph_folder}/log_wage_shk_gr.png", as(png) replace 

** std log wage whole sample
twoway  (connected lwage_shk_gr_sd date) if lwage_shk_gr_sd!=., title("The standard deviation of log real wage shocks") 
graph export "${graph_folder}/log_wage_shk_gr_sd.png", as(png) replace
restore 

* education profile bar
preserve 
collapse (mean) lwage_shk_gr_av_edu=lwage_shk_gr (sd) lwage_shk_gr_sd_edu = lwage_shk_gr, by(educ) 
* average log wage

* standard deviation log wage
graph bar lwage_shk_gr_sd_edu, over(educ) ///
                               ytitle("standard deviation of log wage shocks") ///
                               title("Gross volatility and education") 							   
graph export "${graph_folder}/log_wage_shk_gr_sd_bar_by_edu.png", as(png) replace 
restore 

* education profile: time series 

preserve 
collapse (mean) lwage_shk_gr_av_edu=lwage_shk_gr (sd) lwage_shk_gr_sd_edu = lwage_shk_gr, by(year month date educ) 

replace lwage_shk_gr_av_edu=. if month==1
replace lwage_shk_gr_sd_edu=. if month==1

xtset educ date 

egen avmv3 = filter(lwage_shk_gr_av_edu), coef(1 1 1) lags(-1/1) normalise 
egen sdmv3 = filter(lwage_shk_gr_sd_edu), coef(1 1 1) lags(-1/1) normalise 

* average log wage
twoway  (connected avmv3 date if avmv3!=. & educ==1) ///
        (connected avmv3 date if avmv3!=. & educ==2) ///
		(connected avmv3 date if avmv3!=. & educ==3), ///
        title("The mean of log real wage shocks") ///
		legend(label(1 "HS dropout") label(2 "HS") label(3 "college") col(1)) 
graph export "${graph_folder}/log_wage_shk_gr_by_edu.png", as(png) replace 

* standard deviation log wage
twoway  (connected sdmv3 date if sdmv3!=. & educ==1) ///
        (connected sdmv3 date if sdmv3!=. & educ==2) ///
		(connected sdmv3 date if sdmv3!=. & educ==3), ///
        title("The standard deviation of log real wage shocks") ///
		legend(label(1 "HS dropout") label(2 "HS") label(3 "college") col(1)) 
graph export "${graph_folder}/log_wage_shk_gr_sd_by_edu.png", as(png) replace 
 
restore 

********************************************
** Unconditional summary statistics *****
*****************************************

tabstat lwage_shk_gr, st(sd) by(educ)
tabstat lwage_shk_gr, st(sd) by(age)
tabstat lwage_shk_gr, st(sd) by(gender)

********************************************
** Prepare the matrix for GMM estimation
*****************************************

preserve 
tsset uniqueid date 
tsfill,full						
replace year=year(dofm(date)) if year==.
replace month=month(dofm(date)) if month==.
replace lwage_id_shk_gr=. if month==1		    
keep uniqueid year month lwage_id_shk_gr educ gender age_5yr
gen date_temp = year*100+month
drop if date_temp==.
drop year month 
reshape wide lwage_id_shk_gr, i(uniqueid educ gender age_5yr) j(date_temp)
save "${datafolder}sipp_matrix.dta",replace 
restore 



*****************************************************************************
**** comparison and perceptions and realizations for idiosyncratic shocks ***
****************************************************************************


******
* some renaming to be consistent with the codes for PSID

gen age_h = age 
gen sex_h = gender
gen edu_i_g = educ
gen lwage_h_gr = lwage_Y_gr 
gen lwage_h_n_gr = lwage_n_Y_gr 

** notice here we use lwage_id_shk_gr !!

/*
** byear_5yr and age

preserve
* average log wage
collapse (count) ct = lwage_shk_gr ///
         (mean) lwage_shk_gr_av_byear= lwage_id_shk_Y_gr ///
         (sd) lwage_shk_gr_sd_byear= lwage_id_shk_Y_gr ///
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
collapse (mean) lwage_shk_gr_av_byear= lwage_id_shk_Y_gr ///
         (sd) lwage_shk_gr_sd_byear= lwage_id_shk_Y_gr ///
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
collapse (mean) lwage_shk_gr_av_byear= lwage_id_shk_Y_gr ///
         (sd) lwage_shk_gr_sd_byear= lwage_id_shk_Y_gr ///
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
collapse (mean) lwage_shk_gr_av_byear_edu = lwage_id_shk_Y_gr ///
         (sd) lwage_shk_gr_sd_byear_edu = lwage_id_shk_Y_gr ///
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
collapse (mean) lwage_shk_gr_av_byear_5yr_edu = lwage_id_shk_Y_gr ///
         (sd) lwage_shk_gr_sd_byear_5yr_edu = lwage_id_shk_Y_gr ///
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
collapse (mean) lwage_shk_gr_av_age = lwage_id_shk_Y_gr ///
         (sd)   lwage_shk_gr_sd_age = lwage_id_shk_Y_gr ///
		 (mean)	lwage_shk_av_age = lwage_id_shk ///
		 (sd)    lwage_shk_sd_age = lwage_id_shk, by(age_h) 

gen age = age_h 
merge 1:1 age using "${scefolder}incvar_by_age.dta",keep(master match) 
gen lincvar = sqrt(incvar)
gen lrincvar = sqrt(rincvar)


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


restore


** age x education profile

preserve
* average log wage
collapse (mean) lwage_shk_gr_av_age_edu = lwage_id_shk_Y_gr ///
         (sd) lwage_shk_gr_sd_age_edu = lwage_id_shk_Y_gr ///
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


** age/gender profile 

** scatter 
preserve 
collapse (mean) lwage_shk_gr_av_age_sex = lwage_id_shk_Y_gr ///
         (sd) lwage_shk_gr_sd_age_sex = lwage_id_shk_Y_gr ///
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
collapse  (mean) lwage_h_gr_av_age_sex = lwage_h_gr ///
          (mean) lwage_h_n_gr_av_age_sex = lwage_h_n_gr ///
          (mean) lwage_shk_gr_av_age_sex = lwage_id_shk_Y_gr ///
         (sd) lwage_shk_gr_sd_age_sex = lwage_id_shk_Y_gr ///
		 (mean) lwage_shk_av_age_sex = lwage_id_shk ///
         (sd) lwage_shk_sd_age_sex = lwage_id_shk, by(age_h sex_h edu_i_g) 
gen age = age_h
gen gender = sex_h
gen edu_g = edu_i_g 
drop if edu_g ==1

merge 1:1 age gender edu_g using "${scefolder}incvar_by_age_edu_gender.dta",keep(master match) 
gen lincvar = sqrt(incvar)
gen lrincvar = sqrt(rincvar)


* average nominal growth rate and expected growth rate. 

twoway (scatter lwage_h_n_gr_av_age_sex age_h, color(ltblue)) ///
       (lfit lwage_h_n_gr_av_age_sex age_h, lcolor(red)) ///
       (scatter incmean age_h, color(gray) yaxis(2)) ///
	   (lfit incmean age_h,lcolor(black) yaxis(2)), ///
       title("Realized and Expected Nominal Earning Growth by Age/Gender/Educ")  ///
	   xtitle("age")  ///
	   ytitle("realized growth") ///
	   ytitle("expected growth ", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
graph export "${graph_folder}/real_log_wage_shk_gr_nlevel_by_age_edu_gender_compare.png", as(png) replace 


* average growth rate and expected growth rate. 

twoway (scatter lwage_h_gr_av_age_sex age_h, color(ltblue)) ///
       (lfit lwage_h_gr_av_age_sex age_h, lcolor(red)) ///
       (scatter rincmean age_h, color(gray) yaxis(2)) ///
	   (lfit rincmean age_h,lcolor(black) yaxis(2)), ///
       title("Realized and Expected Earning Growth by Age/Gender/Educ")  ///
	   xtitle("age")  ///
	   ytitle("realized growth") ///
	   ytitle("expected growth ", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
graph export "${graph_folder}/real_log_wage_shk_gr_level_by_age_edu_gender_compare.png", as(png) replace 


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
