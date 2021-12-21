****************************************************
***   This do files cleans SCE individual density **
***   forecasts and moments. It exclude the top and *
***   bottom 5 percentiles of mean and uncertainty. *
***   It also plots histograms of mean forecast and *
***   uncertainty. **********************************
*****************************************************


clear
global mainfolder "/Users/Myworld/Dropbox/IncExpProject/WorkingFolder"
global folder "${mainfolder}/SurveyData/"
global data_folder "/Users/Myworld/Dropbox/ExpProject/workingfolder/SurveyData/SCE/"
global otherfolder "${mainfolder}/OtherData/"
global sum_graph_folder "${mainfolder}/Graphs/pop"
global sum_table_folder "${mainfolder}/Tables"

cd ${folder}
pwd
set more off 


use "${data_folder}NYFED_SCE_13_16.dta",clear
append using "${data_folder}NYFED_SCE_17_19.dta",force
append using  "${data_folder}NYFED_SCE_20.dta",force

sort date
unique userid

********************
*** Date format ****
********************
gen date_str = string(date)
gen year = substr(date_str,1,4)
gen month = substr(date_str,5,2)
gen date2=year+"m"+month
gen date3= monthly(date2,"YM")
format date3 %tm
drop date_str date2 date
rename date3 date
destring year, replace
destring month, replace 
order date year month
xtset userid date

**********************
** Label variables ***
**********************

label var Q1 "finance better or worse(5 vals) from y-1 to y"
label var Q2 "finance better or worse(5 vals) from y to y+1"
label var Q3 "chance of moving (%)"
label var Q4new "chance of UE higher from y to y+1(0-1)"
label var Q5new "chance of saving interest rate higher from y to y+1(%)"
label var Q6new "chance of stock market up from y to y+1(%)"
label var Q8v2 "inflation or deflation from y to y+1 (1/0)"
label var Q8v2part2 "inflation(or deflation) from y to y+1 (%) "
label var Q9_cent25 "25 percentile of inflation from y to y+1(%)"
label var Q9_cent50 "50 percentile of inflation from y to y+1(%)"
label var Q9_cent75 "75 percentile of inflation from y to y+1(%)"
label var Q9_var "var of inflation from y to y+1"
label var Q9_iqr "25/75 inter-quantile range of inflation from y to y+1(%)"
label var Q9_mean "mean of inflation from y to y+1(%)"
label var Q9_probdeflation "prob of deflation from y to y+1 (0-1)"
label var Q9_bin1 "density: >12% inflation from y to y+1(%)"
label var Q9_bin2 "density: 8%-12% inflation from y to y+1(%)"
label var Q9_bin3 "density: 4%-8% inflation from y to y+1(%)"
label var Q9_bin4 "density: 2%-4% inflation from y to y+1(%)"
label var Q9_bin5 "density: 0%-2% inflation from y to y+1(%)"
label var Q9_bin6 "density: -2%-0% inflation from y to y+1(%)"
label var Q9_bin7 "density: -4%- -2% inflation from y to y+1(%)"
label var Q9_bin8 "density: -8%--4% inflation from y to y+1(%)"
label var Q9_bin9 "density: -12%- -8% inflation from y to y+1(%)"
label var Q9_bin10 "density: <-12% inflation from y to y+1(%)"
label var Q9bv2 "inflation or deflation from y+1 to y+2 (1/0)"
label var Q9bv2part2 "inflation(or deflation) from y+1 to y+2 (%)"
label var Q9c_cent25 "25 percentile of inflation from y+1 to y+2(%)"
label var Q9c_cent50 "50 percentile of inflation from y+1 to y+2(%)"
label var Q9c_cent75 "75 percentile of inflation from y+1 to y+2(%)"
label var Q9c_var "var of inflation from y+1 to y+2"
label var Q9c_mean "mean of inflation from y+1 to y+2(%)"
label var Q9c_iqr "25/75 inter-quantile range of inflation from y+1 to y+2(%)"
label var Q9c_probdeflation "prob of deflation from y+1 to y+2 (0-1)"
label var Q9c_bin1 "density: >12% inflation from y+1 to y+2(%)"
label var Q9c_bin2 "density: 8%-12% inflation from y+1 to y+2(%)"
label var Q9c_bin3 "density: 4%-8% inflation from y+1 to y+2(%)"
label var Q9c_bin4 "density: 2%-4% inflation from y+1 to y+2(%)"
label var Q9c_bin5 "density: 0%-2% inflation from y+1 to y+2(%)"
label var Q9c_bin6 "density: -2%-0% inflation from y+1 to y+2(%)"
label var Q9c_bin7 "density: -4%- -2% inflation from y+1 to y+2(%)"
label var Q9c_bin8 "density: -8%--4% inflation from y+1 to y+2(%)"
label var Q9c_bin9 "density: -12%- -8% inflation from y+1 to y+2(%)"
label var Q9c_bin10 "density: <-12% inflation from y+1 to y+2(%)"
label var Q10_1 "current emp situations:full-time"
label var Q10_2 "current emp situations:part-time"
label var Q10_3 "current emp situations: not working but wants to work"
label var Q10_4 "current emp situations: temporary laid-off"
label var Q10_5 "current emp situations: on sick or other leave"
label var Q10_6 "current emp situations: permanently disabled/unable to work"
label var Q10_7 "current emp situations: retiree or early retiree"
label var Q10_8 "current emp situations: student or in training"
label var Q10_9 "current emp situations: homemaker"
label var Q10_10 "current emp situations: others"
label var Q11 "number of jobs"
label var Q12new "work for someone or self-employed(1/0)"
label var ES1_1 ""
label var ES1_2 ""
label var ES1_3 ""
label var ES1_4 ""
label var ES2 ""
label var ES3new ""
label var ES4 ""
label var ES5 ""
label var Q13new "chance of losing job from y to y+1(%)"
label var Q14new "chance of voluntarily leaving the job from y to y+1(%)"
label var Q15 "currently looking for a job (1/0)"
label var Q16 "duration of unemployment (months)"
label var Q17new "chance of finding and accepting a job from y to y+1(%)"
label var Q18new "chance of finding and accepting a job from m to m+3"
label var Q19 "duration of out of work(%)"
label var Q20new "chance of starting looking for a job from y to y+1(%)"
label var Q21new "chance of starting looking for a job from m to m+3(%)"
label var Q22new "chance of finding a new job if losing the current one within 3 months"
label var Q23v2 "earning increase/decrease from the same job/time/place from y to y+1(%)"
label var Q23v2part2 "change of earning from the same job/time/place from y to y+1(%)"
label var Q24_cent25 "25 percentile of earning growth of same job/time/place from y to y+1(%)"
label var Q24_cent50 "50 percentile of earning growth of same job/time/place from y to y+1(%)"
label var Q24_cent75 "75 percentile of earning growth of same job/time/place from y to y+1(%)"
label var Q24_var "var of earning growth of same job/time/place from y to y+1(%)"
label var Q24_mean "mean of earning growth of same job/time/place from y to y+1(%)"
label var Q24_iqr "25/75 inter-quantile range of earning growth of same job/time/place from y to y+1(%)"
label var Q24_probdeflation "???"
label var Q24_bin1 ">12% earning growth of same job/time/place from y to y+1(%)"
label var Q24_bin2 "8%-12% earning growth of same job/time/place from y to y+1(%)"
label var Q24_bin3 "4%-8% earning growth of same job/time/place from y to y+1(%)"
label var Q24_bin4 "2%-4% earning growth of same job/time/place from y to y+1(%)"
label var Q24_bin5 "0-2% earning growth of same job/time/place from y to y+1(%)"
label var Q24_bin6 "-2%-0 earning growth of same job/time/place from y to y+1(%)"
label var Q24_bin7 "-4%- -2% earning growth of same job/time/place from y to y+1(%)"
label var Q24_bin8 "-8% - -4% earning growth of same job/time/place from y to y+1(%)"
label var Q24_bin9 "-12%- -8% earning growth of same job/time/place from y to y+1(%)"
label var Q24_bin10 "<-12% earning growth of same job/time/place from y to y+1(%)"
label var Q25v2 "increase/decrease of total household income from y to y+1(1/0)"
label var Q25v2part2 "increase of total household income from y to y+1(%)"
label var Q26v2 "increase/decrease of total household spending from y to y+1(1/0)"
label var Q26v2part2 "increase of total household spending from y to y+1(%)"
label var Q27v2 "increase/decrease of total household tax payment given the same income (1/0)"
label var Q27v2part2 "increase of total household tax payment given the same income (%)"
label var Q28 "easier/harder to get credit/loan from y-1 to y(1-5)"
label var Q29 "easier/harder to get credit/loan from y to y+1(1-5)"
label var Q30new "chance of non-payment of debt from m to m+3(%)"
label var Q31v2 "increase/decrease of nationwide house price(1/0)"
label var Q31v2part2 "increase of nationwide house price(%)"
label var C1_cent25 "25 percentile of increase of nationwide house price(%)"
label var C1_cent50 "50 percentile of increase of nationwide house price(%)"
label var C1_cent75 "75 percentile of increase of nationwide house price(%)"
label var C1_var "var of increase of nationwide house price"
label var C1_mean "mean of increase of nationwide house price(%)"
label var C1_iqr "25/75 inter-quantile range  of increase of nationwide house price(%)"
label var C1_probdeflation "??"
label var C1_bin1 ">12% increase in nationwide house price(%)"
label var C1_bin2 "8%-12% increase in nationwide house price(%)"
label var C1_bin3 "4%-8% increase in nationwide house price(%)"
label var C1_bin4 "2%-4% increase in nationwide house price(%)"
label var C1_bin5 "0-2% increase in nationwide house price(%)"
label var C1_bin6 "-2%-0% increase in nationwide house price(%)"
label var C1_bin7 "-4%- -2% increase in nationwide house price(%)"
label var C1_bin8 "-8%- -4% increase in nationwide house price(%)"
label var C1_bin9 "-12%- -8% increase in nationwide house price(%)"
label var C1_bin10 "<-12% increase in nationwide house price(%)"
label var C2 "increase/decrease in nationwide house price from y+1 to y+2(1/0)"
label var C2part2 "increase in nationwide house price from y+1 to y+2(%)"
label var C3 "increase/decrease of U.S. gov debt from y to y+1(1/0)"
label var C3part2"increase of U.S. gov debt from y to y+1(%)"
label var C4_1 "increase of a gallon of gas from y to y+1(%)"
label var C4_2 "increase of food price from y to y+1(%)"
label var C4_3 "increase of a medical care from y to y+1(%)"
label var C4_4 "increase of college education from y to y+1(%)"
label var C4_5 "increase of price of renting a typical house/apt from y to y+1(%)"
label var C4_6  "increase of gold price from y to y+1(%)"
label var QNUM1 "num q 1: (correct if ==150)"
label var QNUM2 "num q 2: (correct if ==242)"
label var QNUM3 "num q 3: (correct if ==10)"
label var QNUM5 "num q 5: (correct if ==100)"
label var QNUM6 "num q 6: (correct if ==5)"
label var QNUM8 "num q 8: (correct if ==3)"
label var QNUM9 "num q 9: (correct if ==2)"
label var Q32 "age(in years)"
label var Q33 "gender"
label var Q34 "hispanic/latino/spanish(1/0)"
label var Q35_1 "race: white"
label var Q35_2 "race: black/african american"
label var Q35_3 "race: american indian/alaska native"
label var Q35_4 "race: asian"
label var Q35_5 "race: native hawaiian or other pacific islander"
label var Q35_6 "race: other"
label var Q36 "education (1-8 low to high, 9 other)"
label var Q37 "months of current work"
label var Q38 "living with partner/not (1/0)"
label var HH2_1 "emp of partner: full-time for someone"
label var HH2_2 "emp of partner: part-time for someone"
label var HH2_3 "emp of partner: self-employed"
label var HH2_4 "emp of partner: not working but wants to work"
label var HH2_5 "emp of partner: temporary laid-off"
label var HH2_6 "emp of partner: on sick or other leave"
label var HH2_7 "emp of partner: permanently disabled/unable to work"
label var HH2_8 "emp of partner: retiree or early retiree"
label var HH2_9 "emp of partner: student or in training"
label var HH2_10 "emp of partner: homemaker"
label var HH2_10 "emp of partner: other"
label var _STATE "state (2-digit code)"
label var Q41 "years of living in the current residence"
label var Q42 "years of living in the current states"
label var Q43 "own/rent/other a house(1/2/3)"
label var Q43a "my/spounser/both's name under which current residence is owned/rent"
label var Q44 "own other homes(1/0)"
label var Q45b "health condition(1-5 from good to poor)"
label var Q45new_1 ""
label var Q45new_2 ""
label var Q45new_3 ""
label var Q45new_4 ""
label var Q45new_5 ""
label var Q45new_6 ""
label var Q45new_7 ""
label var Q45new_8 ""
label var Q45new_9 ""
label var Q46 "financial decision making of household(1-5, together to individual)"
label var Q47 "total pre-tax household income from y-1 to y(1-11, low to high)"
label var D1 "same household as last year(1/0)"
label var D3 "date of moving to the current residence(month/year)"
label var D6 "total pre-tax household income from y-1 to y(1-11,low to high)"
label var D2new_1 "memeber in the current resident:spounse/partner"
label var D2new_2 "memeber in the current resident: child >25"
label var D2new_3 "memeber in the current resident: child 18-24"
label var D2new_4 "memeber in the current resident: child 6-17"
label var D2new_5 "memeber in the current resident: child <=5 "
label var D2new_6 "memeber in the current resident: own/sponse's parents"
label var D2new_7 "memeber in the current resident:other relatives"
label var D2new_8 "memeber in the current resident:non-relatives"
label var DSAME "same job as last year in survey"
label var DQ38 "living as a partner or married with some one(1/0)"
label var Q48 "interesting/uninteresting of the questions in the survey(1/0)"


** Only keep useful variables so that the data file is smaller. 

global keeplist date year month userid tenure weight ///        
       Q24_cent25 Q24_cent50 Q24_cent75 Q24_var Q24_mean Q24_iqr ///
	   Q24_bin1 Q24_bin2 Q24_bin3 Q24_bin4 Q24_bin5 Q24_bin6 Q24_bin7 Q24_bin8 Q24_bin9 Q24_bin10 ///
	   Q32 Q33 Q34 Q35_1 Q35_2 Q35_3 Q35_4 Q35_5 Q35_6 Q36 Q37 Q38 ///
	   HH2_1 HH2_2 HH2_3 HH2_4 HH2_5 HH2_6 HH2_7 HH2_8 HH2_9 HH2_10 HH2_11 ///
	   _STATE Q41 Q42 Q43 Q43a Q44 Q45b ///
	   Q45new_1 Q45new_2 Q45new_3 Q45new_4 Q45new_5 Q45new_6 Q45new_7 Q45new_8 Q45new_9 ///
	   Q46 Q47 D1 D3 D6 D2new_1 D2new_2 D2new_3 D2new_4 D2new_5 D2new_6 D2new_7 D2new_8 D2new_9 DSAME DQ38 ///
	   DHH2_1 DHH2_2 DHH2_3 DHH2_4 DHH2_5 DHH2_6 DHH2_7 DHH2_8 DHH2_9 DHH2_10 DHH2_11 DHH2_11_other D5b Q48 QRA1 QRA2 ///
	   _AGE_CAT _NUM_CAT _REGION_CAT _COMMUTING_ZONE _EDU_CAT _HH_INC_CAT ///
	   Q1 Q2 Q3 Q4new Q5new Q6new Q9_mean Q9_var ///
	   Q10_1 Q10_2 Q10_3 Q10_4 Q10_5 Q10_6 Q10_7 Q10_8 Q10_9 Q10_10 Q11 Q12new Q13new Q14new ///
	   Q25v2 Q25v2part2 Q31v2 Q31v2part2 C1_mean C1_var ///
	   Q26v2 Q26v2part2 QNUM1 QNUM2 QNUM3 QNUM5 QNUM6 QNUM8 QNUM9  ///
	   Q17new Q18new Q19 Q20new Q21new Q22new 
 
keep ${keeplist}

*************************
*** Merge with state level 
*************************

rename _STATE state 


merge m:1 state using "${otherfolder}statecode.dta"
rename _merge state_code_merge 

merge m:1 statecode year month using "${otherfolder}stateM.dta",keep(master match) 
rename _merge stateM_merge 

*************************
*** Exclude outliers ****
*************************

local Moments Q24_mean Q24_var Q9_mean Q9_var 

foreach var in `Moments'{
      egen `var'pl=pctile(`var'),p(5)
	  egen `var'pu=pctile(`var'),p(95)
	  replace `var' = . if `var' <`var'pl | (`var' >`var'pu & `var'!=.)
}


foreach var in Q24{
foreach mom in mean var{
    * nominal 
     egen `var'_`mom'p75 =pctile(`var'_`mom'),p(75) by(year month)
	 egen `var'_`mom'p25 =pctile(`var'_`mom'),p(25) by(year month)
	 egen `var'_`mom'p50 =pctile(`var'_`mom'),p(50) by(year month)
	 local lb: variable label `var'_`mom'
	 label var `var'_`mom'p75 "`lb': 75 pctile"
	 label var `var'_`mom'p25 "`lb': 25 pctile"
	 label var `var'_`mom'p50 "`lb': 50 pctile"
}
}


*************************
*** Nominal to real *****
*************************
gen Q24_rmean = Q24_mean - Q9_mean 
label var Q24_rmean "mean of real earning growth of same job/time/place from y to y+1(%) "

gen Q24_rvar = Q24_var + Q9_var 
label var Q24_rvar "variance of real earning growth of same job/time/place from y to y+1(%) "

****************************************
*** Persontage point to log change *****
* for example: 10% is 0.1 in log change 
****************************************

foreach pc_var in Q24_mean Q24_rmean Q9_mean Q24_iqr{
replace `pc_var' = `pc_var'/100
}

foreach pc_var in Q24_var Q24_rvar Q9_var{
replace `pc_var' = `pc_var'/10000
}


*************************
*** Other Measures *****
*************************


egen Q24_sd = sd(Q24_mean), by(date)
gen Q24_disg = Q24_sd^2
label var Q24_disg "Disagreements of 1-yr-ahead expted income growth"


egen Q24_rsd = sd(Q24_rmean), by(date)
gen Q24_rdisg = Q24_rsd^2
label var Q24_rdisg "Disagreements of 1-yr-ahead real expted income growth"


** calculate financial/numeric literacy score

gen QNUM1_ = 0 if QNUM1!=.
gen QNUM2_ = 0 if QNUM2!=.
gen QNUM3_ = 0 if QNUM3!=.
gen QNUM5_ = 0 if QNUM5!=.
gen QNUM6_ = 0 if QNUM6!=.
gen QNUM8_ = 0 if QNUM8!=.
gen QNUM9_ = 0 if QNUM9!=.

replace QNUM1_ = 1 if QNUM1==150
replace QNUM2_ = 1 if QNUM2<=244 & QNUM2>=240
replace QNUM3_ = 1 if QNUM3==10
replace QNUM5_ = 1 if QNUM5==100
replace QNUM6_ = 1 if QNUM6==5
replace QNUM8_ = 1 if QNUM8==3
replace QNUM9_ = 1 if QNUM9==2

gen nlit = QNUM1_+QNUM2_+QNUM3_+QNUM5_+QNUM6_+QNUM8_+QNUM9_
label var nlit "numeracy literacy score: (0-7)"

***************************************************
**** Fill the age/education for the same household
*************************************************


sort userid date 

replace nlit = nlit[_n-1] if nlit==. & userid == userid[_n-1] 
replace Q32 = Q32[_n-1] if Q32==. & userid == userid[_n-1] 
replace Q36 = Q36[_n-1] if Q36==. & userid == userid[_n-1]
replace Q33 = Q33[_n-1] if Q33==. & userid == userid[_n-1] & D1==1
replace Q47 = Q47[_n-1] if Q47==. & userid ==userid[_n-1] & D1 ==1 

***************************************************
**** Winsorization
*************************************************

drop if Q32 > 70 | Q32 < 20 /*age greater than 100 */ 

***************************************************
**** create cohort 
*************************************************

gen byear = year - Q32
label var byear "year of birth"

egen byear_gr = cut(byear), group(5)
label var byear_gr "cohort by year of birth"

label define byear_grlb 0 "1952" 1 "1961" 2 "1970" 3 "1980" 4 "1998"
label value byear_gr byear_grlb


***********************
** generate variables 
************************

** replace educ with highest degree for each individual
egen educ = max(Q36), by(userid) 
*although actually no change ine ducation within the panel 

** education group 
gen educ_gr = .
replace educ_gr = 1 if educ==1
replace educ_gr = 2 if educ==2 | educ ==3 | educ == 4
replace educ_gr = 3 if educ <=9 & educ>4
drop if educ_gr==.

label var educ_gr "education group"
label define educ_grlb 1 "HS dropout" 2 "HS graduate" 3 "College/above"
label value educ_gr educ_grlb


save "${folder}/SCE/IncExpSCEProbIndM",replace 

***************************************
**   Histograms of Moments  ***********
** Maybe replaced by kernel desntiy **
***************************************

* for forecasting

gen SCE_mean = .
gen SCE_var = .

/*
* Kernal density plot only 

label var Q24_mean "1-yr-ahead forecast of income growth "

 foreach var in SCE{
 foreach mom in mean{
    replace `var'_`mom' = Q24_`mom'
	local lb: variable label Q24_`mom'
    twoway (kdensity `var'_`mom',lcolor(red) lwidth(thick) ), ///
	       by(year,title("Distribution of `lb'",size(med)) note("")) xtitle("Mean forecast") ///
		   ytitle("Fraction of population")
	graph export "${sum_graph_folder}/hist/`var'`mom'_hist", as(png) replace 
}
}

* Kernal density plot only 


label var Q24_var "1-yr-ahead uncertainty of income growth"

foreach mom in var{
foreach var in SCE{
    replace `var'_`mom' = Q24_`mom'
	local lb: variable label Q24_`mom'
    twoway (kdensity `var'_`mom',lcolor(blue) lwidth(thick)), ///
	       by(year,title("Distribution of `lb'",size(med)) note("")) xtitle("Uncertainty") ///
		   ytitle("Fraction of population")
	graph export "${sum_graph_folder}/hist/`var'`mom'_hist", as(png) replace 
}
}

*/

*************************
*** Population SCE ******
*************************

local Moments Q24_mean Q24_var Q24_iqr Q24_cent50 Q24_disg 
local rMoments Q24_rmean Q24_rvar Q24_rdisg 
local MomentsMom Q24_meanp25 Q24_meanp50 Q24_meanp75 Q24_varp25 Q24_varp50 Q24_varp75


collapse (mean) `Moments' `rMoments' `MomentsMom', by(date year month)

label var Q24_mean "Average 1-yr-ahead expected earning growth(%)"
label var Q24_var "Average Uncertainty of 1-yr-ahead expected earning growth"
label var Q24_iqr "Average 25/75 IQR of 1-yr-ahead expected earning growth(%)"
label var Q24_cent50 "Average Median of 1-yr-ahead expected earning growth(%)"
label var Q24_disg "Disagreements of 1-yr-ahead expected earning growth"


label var Q24_rmean "Average 1-yr-ahead expected real earning growth(%)"
label var Q24_rvar "Average Uncertainty of 1-yr-ahead real expected earning growth"
label var Q24_rdisg "Disagreements of 1-yr-ahead expected real earning growth"

save "${folder}/SCE/IncExpSCEProbPopM",replace 



