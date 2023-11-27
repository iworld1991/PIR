*************************************************************
*! SIPP data cleaning
*! Last modified: Feb 2022 by Tao 
**************************************************************
*global datafolder "/Users/Myworld/Dropbox/SIPP/"
*global scefolder "/Users/Myworld/Dropbox/PIR/WorkingFolder/SurveyData/SCE/"
*global otherdatafolder "/Users/Myworld/Dropbox/PIR/WorkingFolder/OtherData/"
*global table_folder "/Users/Myworld/Dropbox/PIR/WorkingFolder/Tables/sipp/"
*global graph_folder "/Users/Myworld/Dropbox/PIR/WorkingFolder/Graphs/sipp/"


global datafolder XXX\SIPP\
global scefolder XXX\PIR\WorkingFolder\SurveyData\SCE\
global otherdatafolder XXX\PIR\WorkingFolder\OtherData\
global table_folder XXX\PIR\WorkingFolder\Tables\sipp\
global graph_folder XXX\PIR\WorkingFolder\Graphs\sipp\



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
codebook uniqueid

by uniqueid: gen tenure = _N if TJB1_MSUM!=.

by uniqueid: gen tenure_m = _N if TJB1_MSUM!=. & month==1

************************
*** merge macro data **
************************

merge m:1 year month using "${otherdatafolder}InfM.dta",keep(master match)
drop _merge 


*******************************
***** Creating variables ******
*******************************
sort uniqueid date 

gen ntot_earning_jb1 = TJB1_MSUM
label var ntot_earning_jb1 "nominal total monthly earning"
gen rtot_earning_jb1 = TJB1_MSUM*100/CPIAU
label var rtot_earning_jb1 "real total monthly earning"
gen tot_hours_jb1 = TJB1_MWKHRS
label var tot_hours_jb1 "average hours worked per week in the month"

****************************
** deal with SEAM problem
****************************

** find the entry date of each uniqueid

by uniqueid: gen show_counts = _n if TJB1_MSUM!=.
gen start_survey = cond(show_counts==1,1,0)
label var start_survey "the indicator if the person just enters the survey"
gen end_survey = cond(show_counts==tenure,1,0)
label var end_survey "the indicator if the person just finishes the survey"

by uniqueid: egen start_month = sum(start_survey*month)
label var start_month "the month of year entering the survey"
by uniqueid: egen end_month = sum(end_survey*month)
label var end_month "the month of year exiting the survey"


****************************
** some prefiltering 
**************************

** exclude obs with irregular nb of hours worked. 

foreach var in tot_hours_jb1{
	egen `var'_mean = mean(`var'),by(uniqueid)
	label var `var' "average hours worked by the individual"
	replace `var'=. if `var'<0.5* `var'_mean | `var'>1.5*`var'_mean
}


foreach var in tot_hours_jb1{
egen `var'_p1 = pctile(`var'),p(1) by(date)
egen `var'_p99 = pctile(`var'),p(99) by(date)
replace `var'=. if `var'<`var'_p1 | `var'>=`var'_p99
}


** monthly 
gen wage = ntot_earning_jb1/tot_hours_jb1
** TJB1_MSUM: total earning from the primary job in the month
** TJB1_MWKHRS: Average number of hours worked per week at job 1 during the reference month. 
** Note: the average includes all weeks in the reference month. Weeks before the job began, 
*** after the job ended, and during an away without pay spell are counted as 0 hours worked. 
** Therefore, the total hours of worked for this job is essentially proportional to TJB1_MWKHRS given 
** most of the months have approximately same number of weeks. 
** use the primary job monthly earning for now  
gen wage_n = wage
label var wage_n "nominal monthly wage rate of the primary job"

** nominal to real terms 
replace wage = wage*100/CPIAU
label var wage "real monthly wage rate of the primary job"



*********************
** other frequency **
**********************

*Create some adjusted date for calculating quarterly and yearly wages 
** Q1 is month =2 to month 4 Q4 is month = 11 to month 1 next year 
** year is month =2 to month =2 next year 

gen year_adj = year 
replace year_adj = year-1 if month==1 
label var year_adj "year nb starting from Feb every year"

gen quarter_adj = . 
replace quarter_adj = 1 if month==2 | month==3 | month==4 
replace quarter_adj = 2 if month==5 | month==6 | month==7 
replace quarter_adj = 3 if month==8 | month==9 | month==10 
replace quarter_adj = 4 if month==11 | month==12 | month==1

label var quarter_adj "quarter nb starting from Feb every year"



***********
** quarterly 
***********

egen ntot_earning_jb1_q = sum(ntot_earning_jb1), by(uniqueid year_adj quarter_adj)
label var ntot_earning_jb1_q "quarterly nominal total earning"

egen rtot_earning_jb1_q = sum(rtot_earning_jb1), by(uniqueid year_adj quarter_adj)
label var rtot_earning_jb1_q "quarterly real total earning from primary job"

egen tot_hours_jb1_q = sum(tot_hours_jb1), by(uniqueid year_adj quarter_adj) 
label var tot_hours_jb1_q "quarterly total hours of working for primary job"

gen wage_Q =  rtot_earning_jb1_q/tot_hours_jb1_q				
label var wage_Q "real yearly wage rate of primary job"

gen wage_n_Q = ntot_earning_jb1_q/tot_hours_jb1_q				
label var wage_n_Q "nominal yearly wage rate of primary job"

***********
** yearly 
***********

egen ntot_earning_jb1_y = sum(ntot_earning_jb1), by(uniqueid year_adj)
label var ntot_earning_jb1 "nominal total monthly earning"

egen rtot_earning_jb1_y = sum(rtot_earning_jb1), by(uniqueid year_adj)
label var rtot_earning_jb1_y "yearly real total earning from primary job"

egen tot_hours_jb1_y = sum(tot_hours_jb1), by(uniqueid year_adj) 
label var tot_hours_jb1_y "yearly total hours of working for primary job"

gen wage_Y =  rtot_earning_jb1_y/tot_hours_jb1_y				
label var wage_Y "real yearly wage rate of primary job"

gen wage_n_Y = ntot_earning_jb1_y/tot_hours_jb1_y				
label var wage_n_Y "nominal yearly wage rate of primary job"


*************************************
***** Focus on job-stayers *********
***** following Low et al 2010 ******
**************************************

*****CREATES A NO-GAP SEQUENCE OF FIRM IDS: 123... OR 213...; 
** THIS VARIABLE (NF) REPLACES FIRM ID JOBID1****************
egen nf=group(uniqueid EJB1_JOBID)
egen min=min(nf),by(uniqueid)
replace nf=nf-min+1
drop min

*****CREATES A VARIABLE # OF JOBS***************************************************************************************************
egen minnf=min(nf),by(uniqueid)
egen maxnf=max(nf),by(uniqueid)
gen num_jobs=maxnf-minnf+1

*****IDENTIFIES Those who NEVER change employer*****
egen sdf=sd(nf),by(uniqueid)
keep if sdf==0 & num_jobs==1

**************************
***** Winsorization ******
**************************

foreach var in wage wage_n wage_Q wage_n_Q wage_Y wage_n_Y{
	egen `var'_mean = mean(`var'),by(uniqueid)
	label var `var' "average monthly real earning of the individual"
	replace `var'=. if `var'<0.1* `var'_mean | `var'>10*`var'_mean
}

foreach var in wage wage_n wage_Q wage_n_Q wage_Y wage_n_Y{
egen `var'_p1 = pctile(`var'),p(1) by(date)
egen `var'_p99 = pctile(`var'),p(99) by(date)
replace `var'=. if `var'<`var'_p1 | `var'>=`var'_p99
}

************************************************
***** Validation and Additional Filtering *****
*************************************************
xtset uniqueid date

drop if (ESEX!=l1.ESEX & ESEX!=.)| (ERACE!=l1.ERACE & ERACE!=.)
gen AGE_df = TAGE-l1.TAGE
drop if AGE_df >=2

keep if tenure>=24

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

** drop extreme values of hours worked. 


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

gen lwage_n_Q = log(wage_n_Q)
label var lwage_n_Q "log nomimal quarterly wage rate"
gen lwage_Q = log(wage_Q)
label var lwage_Q "log real quarterly wage rate"

gen lwage_n_Y = log(wage_n_Y)
label var lwage_n_Y "log nomimal yearly wage rate"

gen lwage_Y = log(wage_Y)
label var lwage_Y "log real yearly wage rate"

** demean the data
egen lwage_av = mean(lwage), by(date) 
egen lwage_sd = sd(lwage), by(date)


egen lwage_Q_av = mean(lwage_Q), by(year quarter) 
egen lwage_Q_sd = sd(lwage_Q), by(year quarter)

egen lwage_Y_av = mean(lwage_Y), by(year) 
egen lwage_Y_sd = sd(lwage_Y), by(year)

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


*****************************************
*** Stochastic wage component/ shocks *
*****************************************

***********************************************************************************
** mincer regressions 

** monthly 
reghdfe lwage age age2 age3 end_yr beg_yr, a(i.month i.race i.gender i.educ i.TJB1_IND) resid
predict lwage_shk, residuals
 
* including aggregate shock
reghdfe lwage age age2 age3 end_yr beg_yr, a(i.month i.date i.race i.gender i.educ i.TJB1_IND) resid
predict lwage_id_shk, residuals

gen lwage_ag_shk = lwage_shk- lwage_id_shk

label var lwage_shk "log wage shock"
label var lwage_id_shk "log wage idiosyncratic shock"
label var lwage_ag_shk "log wage aggregate shock"


** monthly nominal

reghdfe lwage_n age age2 age3 end_yr beg_yr, a(i.month i.race i.gender i.educ i.TJB1_IND) resid
predict lwage_n_shk, residuals
 
* including aggregate shock
reghdfe lwage_n age age2 age3 end_yr beg_yr, a(i.month i.date i.race i.gender i.educ i.TJB1_IND) resid
predict lwage_n_id_shk, residuals

gen lwage_n_ag_shk = lwage_n_shk- lwage_n_id_shk

label var lwage_n_shk "log nominal wage shock"
label var lwage_n_id_shk "log nominal wage idiosyncratic shock"
label var lwage_n_ag_shk "log nominal wage aggregate shock"


*************************************************************************************
** quarterly  
*************************************************************************************

reghdfe lwage_Q age age2 age3, a(i.quarter_adj i.year_adj#i.quarter_adj i.race i.gender i.educ i.TJB1_IND) resid
predict lwage_Q_id_shk, residuals
label var lwage_Q_id_shk "log yearly wage idiosyncratic shock"

** first difference for quarterly  
foreach var in lwage_Q lwage_n_Q lwage_Q_id_shk{
gen `var'_gr = `var'- l3.`var'
}

label var lwage_Q_gr "log quarterly growth of nominal wage"
label var lwage_n_Q_gr "log quarterly growth of real wage"
label var lwage_Q_id_shk_gr "log quarterly growth of idiosyncratic unexplained wage"


*************************************************************************************
** yearly 
*************************************************************************************

reghdfe lwage_Y age age2 age3, a(i.year_adj i.race i.gender i.educ i.TJB1_IND) resid
predict lwage_Y_id_shk, residuals
label var lwage_Y_id_shk "log yearly wage idiosyncratic shock"

** first difference for monthly 
foreach var in lwage lwage_n lwage_shk lwage_id_shk lwage_ag_shk lwage_n_id_shk lwage_n_ag_shk{
gen `var'_gr = `var'- l1.`var'
}
label var lwage_n_gr "log growth of nominal wage"
label var lwage_gr "log growth of real wage"
label var lwage_shk_gr "log growth of unexplained wage"
label var lwage_id_shk_gr "log growth of idiosyncratic unexplained wage"
label var lwage_ag_shk_gr "log growth of aggregate unexplained wage"
label var lwage_n_id_shk_gr "log nominal growth of idiosyncratic unexplained wage"
label var lwage_n_ag_shk_gr "log nominal growth of idiosyncratic unexplained wage"


foreach var in lwage lwage_id_shk{
gen `var'_y2y_gr = `var'- l12.`var'
}
label var lwage_y2y_gr "log YoY growth of wage"
label var lwage_id_shk_y2y_gr "log YoY growth of unexplained wage"

** 1-year 2-year 3-year difference for yearly 
foreach var in lwage_Y lwage_n_Y lwage_Y_id_shk{
gen `var'_gr = `var'- l12.`var'
gen `var'_gr2 = `var'- l24.`var'
gen `var'_gr3 = `var'- l36.`var'
}
label var lwage_Y_gr "log yearly growth of nominal wage"
label var lwage_n_Y_gr "log yearly growth of real wage"
label var lwage_Y_id_shk_gr "log yearly growth of idiosyncratic unexplained wage"

label var lwage_Y_gr2 "log 2-year growth of nominal wage"
label var lwage_n_Y_gr2 "log 2-year  growth of real wage"
label var lwage_Y_id_shk_gr2 "log yearly 2-year  of idiosyncratic unexplained wage"

label var lwage_Y_gr3 "log 3-year  growth of nominal wage"
label var lwage_n_Y_gr3 "log 3-year  growth of real wage"
label var lwage_Y_id_shk_gr3 "log 3-year growth of idiosyncratic unexplained wage"


** computing risk at the individual level 

egen lwage_id_shk_gr_sq = mean(sqrt(lwage_id_shk_gr^2)), by(uniqueid)
label var lwage_id_shk_gr_sq "individual level monthly risk"

egen lwage_Y_id_shk_gr_sq = mean(sqrt(lwage_Y_id_shk_gr^2)), by(uniqueid)
label var lwage_Y_id_shk_gr_sq "individual level annual risk"

egen lwage_id_shk_y2y_gr_sq = mean(sqrt(lwage_id_shk_y2y_gr^2)), by(uniqueid)
label var lwage_id_shk_y2y_gr_sq "individual y2y risk"

** generate predicted income risks 

foreach var in lwage_id_shk_gr_sq lwage_Y_id_shk_gr_sq lwage_id_shk_y2y_gr_sq{
reghdfe `var' age age2 age3, a(i.year_adj i.race i.gender i.educ i.TJB1_IND) resid
predict `var'_pr, 
}

** exporting all distribution 

preserve 
duplicates drop uniqueid,force
keep lwage_id_shk_gr_sq* lwage_Y_id_shk_gr_sq* lwage_id_shk_y2y_gr_sq*
save "${otherdatafolder}/sipp/sipp_individual_risk.dta",replace 
restore 


**  volatility 
foreach wvar in lwage lwage_Q lwage_Y{
egen `wvar'_id_shk_gr_sd = sd(`wvar'_id_shk_gr), by(date)
label var `wvar'_id_shk_gr_sd "standard deviation of log idiosyncratic shocks"
}


foreach wvar in lwage_Y{
egen `wvar'_id_shk_gr_sd_all = sd(`wvar'_id_shk_gr) if `wvar'_id_shk_gr!=. & `wvar'_id_shk_gr2!=. & `wvar'_id_shk_gr3!=.
gen `wvar'_id_shk_gr_var_all = `wvar'_id_shk_gr_sd_all^2
egen `wvar'_id_shk_gr_sd2_all = sd(`wvar'_id_shk_gr2) if `wvar'_id_shk_gr!=. & `wvar'_id_shk_gr2!=. & `wvar'_id_shk_gr3!=.
gen `wvar'_id_shk_gr_var2_all = `wvar'_id_shk_gr_sd2_all^2
egen `wvar'_id_shk_gr_sd3_all = sd(`wvar'_id_shk_gr3) if `wvar'_id_shk_gr!=. & `wvar'_id_shk_gr2!=. & `wvar'_id_shk_gr3!=.
gen `wvar'_id_shk_gr_var3_all = `wvar'_id_shk_gr_sd3_all^2
}

** estimate permanent risks for the whole sample 
gen pvar_est_all1 = lwage_Y_id_shk_gr_var2_all- lwage_Y_id_shk_gr_var_all
label var pvar_est_all1 "Yearly permanent risks in variance"

gen pvar_est_all2 = lwage_Y_id_shk_gr_var3_all- lwage_Y_id_shk_gr_var2_all
label var pvar_est_all2 "Yearly permanent risks in variance"

gen pvar_est_all3 = (lwage_Y_id_shk_gr_var3_all- lwage_Y_id_shk_gr_var_all)/2
label var pvar_est_all3 "Yearly permanent risks in variance"

***********************************************
** summary chart of unconditional wages ********
************************************************

** histograms of wage distribution 

hist tot_hours_jb1, title("distribution of hours of worked") 
graph export "${graph_folder}/hist_hours.png", as(png) replace 

hist  wage, title("distribution of real wage rate") 
graph export "${graph_folder}/hist_wage.png", as(png) replace 

hist  wage_Q, title("distribution of real wage rate (yearly)") 
graph export "${graph_folder}/hist_wage_Q.png", as(png) replace
 
hist  wage_Y, title("distribution of real wage rate (yearly)") 
graph export "${graph_folder}/hist_wage_Y.png", as(png) replace 

** time series plots 
preserve
collapse (mean) lwage_Q_av lwage_Q_sd, by(year quarter) 

gen date_str=string(year)+"Q"+string(quarter)
gen dateq = quarterly(date_str,"YQ")
drop date_str
*xtset uniqueid dateq
format dateq %tq

** average log wage whole sample
twoway  (connected lwage_Q_av dateq) if lwage_Q_av!=., title("The mean of log real wages") 
graph export "${graph_folder}/log_wage_av.png", as(png) replace 

** std log wage whole sample
twoway  (connected lwage_Q_sd dateq) if lwage_Q_sd!=., title("The standard deviation of log real wages") 
graph export "${graph_folder}/log_wage_sd.png", as(png) replace 
restore 


preserve 
replace lwage = lwage_Q

collapse (mean) lwage_av_educ=lwage (sd) lwage_sd_educ = lwage, by(year quarter educ) 

gen date_str=string(year)+"Q"+string(quarter)
gen dateq = quarterly(date_str,"YQ")
drop date_str
gen date = dateq 
format date %tq

* average log wage
twoway  (connected lwage_av_educ date if lwage_av!=. & educ==1) ///
        (connected lwage_av_educ date if lwage_av!=. & educ==2) ///
		(connected lwage_av_educ date if lwage_av!=. & educ==3), ///
        title("The mean of log real wages") ///
		legend(pos(6) label(1 "HS dropout") label(2 "HS") label(3 "college") col(1)) 
graph export "${graph_folder}/log_wage_av_by_edu.png", as(png) replace 

* standard deviation log wage
twoway  (connected lwage_sd_educ date if lwage_sd_educ!=. & educ==1) ///
        (connected lwage_sd_educ date if lwage_sd_educ!=. & educ==2) ///
		(connected lwage_sd_educ date if lwage_sd_educ!=. & educ==3), ///
        title("The standard deviation of log real wages") ///
		legend(pos(6) label(1 "HS dropout") label(2 "HS") label(3 "college") col(1)) 
graph export "${graph_folder}/log_wage_sd_by_edu.png", as(png) replace 

restore 

************************************************
** summary table of conditional wages ********
************************************************


label define edu_glb 1 "HS dropout" 2 "HS graduate" 3 "College/above"
label value educ edu_glb

tabout year educ gender using "${table_folder}sum_table_sipp.csv", ///
            c(N lwage_id_shk_gr sd lwage_id_shk_gr sd lwage_id_shk_y2y_gr) ///
			 f(0c 2c) clab(Obs Volatility VolatilityY2Y) sum ///
			  npos(tufte) style(csv) rep bt cltr2(.75em) 

************************************************
** summary chart of conditional wages ********
************************************************


* monthly 
preserve

replace lwage_id_shk_gr=. if start_month==1 & end_month ==12 & month==1 

collapse (mean) lwage_id_shk_gr ///
          (sd) lwage_id_shk_gr_sd = lwage_id_shk_gr ///
		  lwage_id_shk_y2y_gr_sd = lwage_id_shk_y2y_gr, by(year month date) 

*replace lwage_id_shk_gr=. if month==1 | month==2 
*replace lwage_id_shk_gr_sd=. if month==1| month==2

** average log wage shock whole sample
twoway  (connected lwage_id_shk_gr date) if lwage_id_shk_gr!=., title("The mean of log real wage shocks") 
graph export "${graph_folder}/log_wage_shk_gr.png", as(png) replace 

** std log wage whole sample
twoway  (connected lwage_id_shk_gr_sd date) if lwage_id_shk_gr_sd!=., title("The standard deviation of log real wage shocks") 
graph export "${graph_folder}/log_wage_shk_gr_sd.png", as(png) replace


** std log wage whole sample
twoway  (connected lwage_id_shk_y2y_gr_sd date, lpattern(dot)) if lwage_id_shk_y2y_gr_sd!=., title("The standard deviation of log real wage shocks (y2y)") 
graph export "${graph_folder}/log_y2y_wage_shk_gr_sd.png", as(png) replace

restore 


* education profile bar
preserve 
replace lwage_shk_gr = lwage_id_shk_gr

collapse (mean) lwage_shk_gr_av_edu=lwage_shk_gr (sd) lwage_shk_gr_sd_edu = lwage_shk_gr, by(educ) 
* average log wage

* standard deviation log wage
graph bar lwage_shk_gr_sd_edu, over(educ) ///
                               ytitle("standard deviation of log wage shocks") ///
                               title("Gross volatility and education") 							   
graph export "${graph_folder}/log_wage_shk_gr_sd_bar_by_edu.png", as(png) replace 
restore 


* quarterly 
preserve
keep if quarter_adj!=.

collapse (sd) lwage_id_shk_gr_sd = lwage_Q_id_shk_gr, by(year_adj quarter_adj) 

gen date_str=string(year_adj)+"Q"+string(quarter_adj)
gen dateq = quarterly(date_str,"YQ")
drop date_str
format date %tq

** std log wage whole sample
twoway  (connected lwage_id_shk_gr_sd date) if lwage_id_shk_gr_sd!=., title("The standard deviation of log real wage shocks") 
graph export "${graph_folder}/log_wage_shk_gr_sd_quarterly.png", as(png) replace
restore 


* education profile: time series 

preserve 
replace lwage_shk_gr = lwage_id_shk_gr

collapse (mean) lwage_shk_gr_av_edu=lwage_shk_gr (sd) lwage_shk_gr_sd_edu = lwage_shk_gr, by(year month date educ) 

replace lwage_shk_gr_av_edu=. if month==1 | month==2
replace lwage_shk_gr_sd_edu=. if month==1| month==2
xtset educ date 

gen avmv3 = (l1.lwage_shk_gr_av_edu+lwage_shk_gr_av_edu+f1.lwage_shk_gr_av_edu)/3
gen sdmv3 = (l1.lwage_shk_gr_sd_edu+lwage_shk_gr_sd_edu+f1.lwage_shk_gr_sd_edu)/3

* average log wage
twoway  (connected avmv3 date if avmv3!=. & educ==1) ///
        (connected avmv3 date if avmv3!=. & educ==2) ///
		(connected avmv3 date if avmv3!=. & educ==3), ///
        title("The mean of log real wage shocks") ///
		legend(pos(6) label(1 "HS dropout") label(2 "HS") label(3 "college") col(1)) 
graph export "${graph_folder}/log_wage_shk_gr_by_edu.png", as(png) replace 

* standard deviation log wage
twoway  (connected sdmv3 date if sdmv3!=. & educ==1) ///
        (connected sdmv3 date if sdmv3!=. & educ==2) ///
		(connected sdmv3 date if sdmv3!=. & educ==3), ///
        title("The standard deviation of log real wage shocks") ///
		legend(pos(6) label(1 "HS dropout") label(2 "HS") label(3 "college") col(1)) 
graph export "${graph_folder}/log_wage_shk_gr_sd_by_edu.png", as(png) replace 
 
restore 

********************************************
** Unconditional and conditional summary statistics *****
*****************************************

tabstat lwage_gr lwage_id_shk_gr, st(sd) by(educ)
tabstat lwage_gr lwage_id_shk_gr, st(sd) by(age)
tabstat lwage_gr lwage_id_shk_gr, st(sd) by(gender)

********************************************
** Generate income volatility data by group *****
*****************************************
** can be done by other grouping method  
preserve 

collapse  (count) ct = lwage_id_shk_y2y_gr ///
          (sd) lwage_shk_gr_sd_age_sex = lwage_id_shk_y2y_gr , by(age_5yr gender educ) 


save "${otherdatafolder}/sipp/sipp_vol_edu_gender_age5.dta",replace

restore 

********************************************
** Prepare the matrix for GMM estimation
*****************************************


preserve 
tsset uniqueid date 
tsfill,full						
replace year=year(dofm(date)) if year==.
replace month=month(dofm(date)) if month==.
*replace lwage_n_gr=. if month==1
keep uniqueid year month lwage_n_gr educ gender age_5yr
gen date_temp = year*100+month
drop if date_temp==.
drop year month 
reshape wide lwage_n_gr, i(uniqueid educ gender age_5yr) j(date_temp)
save "${datafolder}sipp_nwage_growth_matrix.dta",replace 
restore 

preserve 
tsset uniqueid date 
tsfill,full						
replace year=year(dofm(date)) if year==.
replace month=month(dofm(date)) if month==.
*replace lwage_n_gr=. if month==1
keep uniqueid year month lwage_n_gr educ gender age_5yr
gen date_temp = year*100+month
drop if date_temp==.
drop year month 
reshape wide lwage_n_gr, i(uniqueid educ gender age_5yr) j(date_temp)
save "${datafolder}sipp_nwage_growth_matrix.dta",replace 
restore 



preserve 
tsset uniqueid date 
tsfill,full						
replace year=year(dofm(date)) if year==.
replace month=month(dofm(date)) if month==.
*replace lwage_gr=. if month==1
keep uniqueid year month lwage_gr educ gender age_5yr
gen date_temp = year*100+month
drop if date_temp==.
drop year month 
reshape wide lwage_gr, i(uniqueid educ gender age_5yr) j(date_temp)
save "${datafolder}sipp_wage_growth_matrix.dta",replace 
restore 


preserve 
tsset uniqueid date 
tsfill,full						
replace year=year(dofm(date)) if year==.
replace month=month(dofm(date)) if month==.
*replace lwage_gr=. if month==1
keep uniqueid year month lwage_y2y_gr educ gender age_5yr
gen date_temp = year*100+month
drop if date_temp==.
drop year month 
reshape wide lwage_y2y_gr, i(uniqueid educ gender age_5yr) j(date_temp)
save "${datafolder}sipp_wage_y2y_growth_matrix.dta",replace 
restore 


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


preserve 
tsset uniqueid date 
tsfill,full						
replace year=year(dofm(date)) if year==.
replace month=month(dofm(date)) if month==.
replace lwage_n_id_shk_gr=. if month==1
keep uniqueid year month lwage_n_id_shk_gr educ gender age_5yr
gen date_temp = year*100+month
drop if date_temp==.
drop year month 
reshape wide lwage_n_id_shk_gr, i(uniqueid educ gender age_5yr) j(date_temp)
save "${datafolder}sipp_matrix_nomimal.dta",replace 
restore 

preserve
keep if quarter_adj!=.
duplicates drop uniqueid year_adj quarter_adj,force
gen date_str=string(year_adj)+"Q"+string(quarter_adj)
gen dateq = quarterly(date_str,"YQ")
drop date_str
format date %tq
tsset uniqueid dateq 
tsfill,full		    
keep uniqueid year_adj quarter_adj lwage_Q_id_shk_gr educ gender age_5yr
gen date_temp = year_adj*10+quarter_adj
drop if date_temp==.
drop year_adj quarter_adj
reshape wide lwage_Q_id_shk_gr, i(uniqueid educ gender age_5yr) j(date_temp)
save "${datafolder}sipp_matrix_Q.dta",replace 
restore 


preserve
duplicates drop uniqueid year_adj,force
tsset uniqueid year_adj 
tsfill,full						
*replace year=year(dofm(date)) if year==.
*replace month=month(dofm(date)) if month==.
*replace lwage_id_shk_gr=. if month==1		    
keep uniqueid year_adj lwage_Y_id_shk_gr educ gender age_5yr
*gen date_temp = year
drop if year_adj==.
reshape wide lwage_Y_id_shk_gr, i(uniqueid educ gender age_5yr) j(year_adj)
save "${datafolder}sipp_matrix_Y.dta",replace 
restore 


preserve
keep if month==1 
tsset uniqueid year
tsfill,full						
keep uniqueid year lwage_id_shk_y2y_gr educ gender age_5yr
*gen date_temp = year
drop if year==.
reshape wide lwage_id_shk_y2y_gr, i(uniqueid educ gender age_5yr) j(year)
save "${datafolder}sipp_matrix_Y2Y.dta",replace 
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
         (mean) lwage_shk_gr_av_byear= lwage_Y_id_shk_gr ///
         (sd) lwage_shk_gr_sd_byear= lwage_id_shk_gr ///
		 (mean) lwage_shk_av_byear= lwage_Y_id_shk ///
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


restore

** cohort and gender 

preserve
* average log wage
collapse (mean) lwage_shk_gr_av_byear= lwage_Y_id_shk_gr ///
         (sd) lwage_shk_gr_sd_byear= lwage_id_shk_gr ///
		 (mean) lwage_shk_av_byear= lwage_Y_id_shk ///
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


restore


** cohort 

preserve
* average log wage
collapse (mean) lwage_shk_gr_av_byear= lwage_Y_id_shk_gr ///
         (sd) lwage_shk_gr_sd_byear= lwage_id_shk_gr ///
		 (mean) lwage_shk_av_byear= lwage_Y_id_shk ///
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

restore


** cohort/education profile

preserve
* average log wage
collapse (mean) lwage_shk_gr_av_byear_edu = lwage_Y_id_shk_gr ///
         (sd) lwage_shk_gr_sd_byear_edu = lwage_id_shk_gr ///
		 (mean) lwage_shk_av_byear_edu = lwage_Y_id_shk ///
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

restore


** 5-year cohort/education/gender profile

preserve


* average log wage
collapse (mean) lwage_shk_gr_av_byear_5yr_edu = lwage_Y_id_shk_gr ///
         (sd) lwage_shk_gr_sd_byear_5yr_edu = lwage_id_shk_gr ///
		 (mean) lwage_shk_av_byear_5yr_edu = lwage_Y_id_shk ///
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


restore


** age profile 
preserve
* average log wage
collapse (mean) lwage_shk_gr_av_age = lwage_Y_id_shk_gr ///
         (sd)   lwage_shk_gr_sd_age = lwage_id_shk_gr ///
		 (mean)	lwage_shk_av_age = lwage_Y_id_shk ///
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
	    ytitle("wage volatility (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
graph export "${graph_folder}/real_log_wage_shk_gr_by_age_compare.png", as(png) replace 


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
collapse (mean) lwage_shk_gr_av_age_edu = lwage_Y_id_shk_gr ///
         (sd) lwage_shk_gr_sd_age_edu = lwage_id_shk_gr ///
		 (mean) lwage_shk_av_age_edu = lwage_Y_id_shk ///
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
collapse (mean) lwage_shk_gr_av_age_sex = lwage_Y_id_shk_gr ///
         (sd) lwage_shk_gr_sd_age_sex = lwage_id_shk_gr ///
		 (mean) lwage_shk_av_age_sex = lwage_Y_id_shk ///
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

restore

** age/gender/educ profile 

** scatter 
preserve 

collapse  (count) ct = lwage_id_shk_gr ///
          (mean) lwage_h_gr_av_age_sex = lwage_h_gr ///
          (mean) lwage_h_n_gr_av_age_sex = lwage_h_n_gr ///
          (mean) lwage_shk_gr_av_age_sex = lwage_Y_id_shk_gr ///
         (sd) lwage_shk_gr_sd_age_sex = lwage_id_shk_gr ///
		 (mean) lwage_shk_av_age_sex = lwage_Y_id_shk ///
         (sd) lwage_shk_sd_age_sex = lwage_id_shk, by(age_5yr sex_h edu_i_g) 

gen age_h = age_5yr
gen gender = sex_h
gen edu_g = edu_i_g 


merge 1:1 age_5yr gender edu_g using "${scefolder}incvar_by_age5y_edu_gender.dta",keep(master match) 
gen lincvar = sqrt(incvar)
gen lrincvar = sqrt(rincvar)


* average nominal growth rate and expected growth rate. 

twoway (scatter lwage_h_n_gr_av_age_sex age_h, color(ltblue)) ///
       (lfit lwage_h_n_gr_av_age_sex age_h, lcolor(red)) ///
       (scatter incmean age_h, color(gray) yaxis(2)) ///
	   (lfit incmean age_h,lcolor(black) yaxis(2)), ///
       title("Realized and Expected Nominal Wage Growth by Age/Gender/Educ",size(med))  ///
	   xtitle("age")  ///
	   ytitle("realized growth") ///
	   ytitle("expected growth ", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
graph export "${graph_folder}/real_log_wage_gr_nlevel_by_age_edu_gender_compare.png", as(png) replace 


* average growth rate and expected growth rate. 

twoway (scatter lwage_h_gr_av_age_sex age_h, color(ltblue)) ///
       (lfit lwage_h_gr_av_age_sex age_h, lcolor(red)) ///
       (scatter rincmean age_h, color(gray) yaxis(2)) ///
	   (lfit rincmean age_h,lcolor(black) yaxis(2)), ///
       title("Realized and Expected Wage Growth by Age/Gender/Educ",size(med))  ///
	   xtitle("age")  ///
	   ytitle("realized growth") ///
	   ytitle("expected growth ", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
graph export "${graph_folder}/real_log_wage_gr_level_by_age_edu_gender_compare.png", as(png) replace 


* standard deviation log wage growth and risk perception 
twoway (scatter lwage_shk_gr_sd_age_sex age_h, color(ltblue)) ///
       (lfit lwage_shk_gr_sd_age_sex age_h, lcolor(red)) ///
       (scatter lrincvar age_h, color(gray) yaxis(2)) ///
	   (lfit lrincvar age_h,lcolor(black) yaxis(2)), ///
       title("Realized Volatility and Perceived Risks by Age/Gender/Educ",size(med))  ///
	   xtitle("age")  ///
	   ytitle("wage volatility (std)") ///
	   ytitle("risk perception (std)", axis(2)) ///
	   ysc(titlegap(3) outergap(0)) ///
	   legend(col(2) lab(1 "Realized") lab(2 "Realized (fitted)")  ///
	                  lab(3 "Perceived (RHS)") lab(4 "Perceived (fitted)(RHS)"))
graph export "${graph_folder}/real_log_wage_shk_gr_by_age_edu_gender_compare.png", as(png) replace 


* perceived riks by age/education  
twoway (scatter lrincvar lwage_shk_gr_sd_age_sex, color(ltblue)) ///
       (lfit lrincvar lwage_shk_gr_sd_age_sex, lcolor(red)) if lwage_shk_gr_sd_age_sex!=., ///
       title("Realized Volatility and Perceived Risks by Age/Gendeer/Educ")  ///
	   xtitle("Volatility") ///
	   ytitle("Perceived risk") 
graph export "${graph_folder}/real_realized_perceived_risks_by_age_edu_gender.png", as(png) replace  


restore

/*
** time series 

preserve 
collapse (mean) lwage_shk_gr_av=lwage_id_shk_gr ///
         (sd) lwage_shk_gr_sd = lwage_id_shk_gr, by(year month date) 

replace lwage_shk_gr_sd=. if month==1 | month==2
* average log wage
tset date
twoway  (connected lwage_shk_gr_av date), ///
        title("The mean of log real wage shocks") 
graph export "${graph_folder}/log_wage_shk_gr.png", as(png) replace 

* standard deviation log wage

twoway  (connected lwage_shk_gr_sd date if lwage_shk_gr_sd!=.), ///
        title("The standard deviation of log real wage shocks") 
graph export "${graph_folder}/log_wage_shk_gr_sd.png", as(png) replace
restore 
*/
