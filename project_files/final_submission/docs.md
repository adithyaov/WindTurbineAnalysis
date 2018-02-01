
## Documentation

___

**Specifying the Data**    
A data file should containg the data for a specific Machine, and the coulmn names should be one of the following:  
Active Power Column: **ActivePower**    
Wind Speed Column: **WindSpeed**    
Rotor Speed Column: **Rotor Speed**    
Pitch Angle Column: **PitchAngle**    
Turbine State Column: **TurbineState**

___
___
___

### Pitch Angle vs Active Power

**How it works**    
The **main** function give a specific score to the data.    
Depending on the score, the data can be classified.

**Dependencies**    
1. [numpy][numpy link]: Fundamental package for scientific computing with Python.  
2. [pandas][pandas link]: Library providing high-performance, easy-to-use data structures and data analysis tools for the Python.  
3. [sklearn.cluster.DBSCAN][dbscan link]: Clustring algorithm, (**D**ensity-**b**ased **s**patial **c**lustering of **a**pplications with **n**oise).  
4. [matplotlib.pyplot][pyplot link]: Python 2D plotting library.  

___

**Function:** def main(...):    
_Main Function_

**Parameters:**    
_Arguments default to Chakla_  
1. filePath  
    * Path of the file for running the function on.  
    * File Format should be **csv**.  
    * Should contain data on: PitchAngle, ActivePower, TurbineState.  
2. xMinFilter  
    * Essentially cropping the data.  
    * The x-offset min value, ie. The valid least PitchAngle value to crop from.  
    * Default value: **-2.82**  
3. xMaxFilter  
    * Essentially cropping the data.  
    * The x-offset max value, ie. The valid highest PitchAngle value to crop to.  
    * Default value: **1.18**  
4. yMinFilter  
    * Essentially cropping the data.  
    * The y-offset min value, ie. The valid least ActivePower value to crop from.  
    * Default value: **188**  
5. yMaxFilter  
    * Essentially cropping the data.  
    * The y-offset max value, ie. The valid highest ActivePower value to crop to.  
    * Default value: **1047**  
6. turbineStateFilter  
    * Valid Turbine State to filter proper data.  
    * Default value: **11**  
7. dbscanEps  
    * The variable **eps** for the DBSCAN Clustring Algorithm, used to **filter the noise**.  
    * ref: [sklearn.cluster.DBSCAN][dbscan link]  
    * Default value: **3**  
8. dbscanMinSamples  
    * The variable **min_samples** for the DBSCAN Clustring Algorithm, used to **filter the noise**.  
    * ref: [sklearn.cluster.DBSCAN][dbscan link]  
    * Default value: **4**  
9. roundingMult  
    * Variable used in the function **def rounding(mult, div):**  
    * The value to multiply to the scaled value _(for discretizing)_.  
    * Default value: **2**  
10. roundingDiv  
    * Variable used in the function **def rounding(mult, div):**  
    * The value to divide the scaled value after multiplyting _(for discretizing)_, then converted to **int** format.  
    * Default value: **1**  
11. maxSpacePositionLimits  
    * A tuple defining the positional limits of the max space. (max gap in between the points).  
    * Default value: **(0.2, 0.8)**  
12. limitMaxScore  
    * A limiting value to the score in each discritized line.  
    * Used to avoid abnotmal scores.  
    * Default value: **8**  
13. limitGrpSize  
    * A limiting value for the size of the group that the DBSCAN acts on.  
    * Default value: **2**  
14. plotGraph  
    * Expects a BOOLEAN.  
        * True: Prints Graph  
        * False: Does not print Graph  
    * Default value: **False**  

**Return Value:**    
_A Python [dictionary][python dict link]_  
* filePath: The complete path of the input File.  
* instances: The number of valid discrete points used in computation.  
* maxInstances: The number of valid discrete points possible.  
* maxScorePossible: The max score if maxInstances were used.  
* maxScore: The max score if valid instances are used.  
* score: The computed score.  

___

**Function:** def rounding(...)::    
_Used in discritizing the scaled values_  

**Parameters**
1. mult  
    * Default value: 1  
2. div  
    * Default value: 2  

**Defination:**    
int(**number** * mult / div)  

___  
___
___


### Rotor Speed vs Active Power

**How it works**    
The **main** function gives specific numbers correcponding to the Data.    
Depending on the values of the numbers, the data can be classified.

**Dependencies**    
1. [numpy][numpy link]: Fundamental package for scientific computing with Python.  
2. [pandas][pandas link]: Library providing high-performance, easy-to-use data structures and data analysis tools for the Python.  
3. [sklearn.cluster.DBSCAN][dbscan link]: Clustring algorithm, (**D**ensity-**b**ased **s**patial **c**lustering of **a**pplications with **n**oise).  
4. [matplotlib.pyplot][pyplot link]: Python 2D plotting library.  
5. [scipy.stats][scipy stats link]: Module containing large number of _probability distributions_ as well as _statistical functions_.  

___

**Function:** def main(...):    
_Main Function_

**Parameters:**    
_Arguments default to Burgula_
1. filePath  
    * Path of the file for running the function on.  
    * File Format should be **csv**.  
    * Should contain data on: RotorSpeed, ActivePower, TurbineState.  
2. xMinFilter  
    * Essentially cropping the data.  
    * The x-offset min value, ie. The valid least RotorSpeed value to crop from.  
    * Default value: **21**  
3. xMaxFilter  
    * Essentially cropping the data.  
    * The x-offset max value, ie. The valid highest RotorSpeed value to crop to.  
    * Default value: **25**  
4. yMinFilter  
    * Essentially cropping the data.  
    * The y-offset min value, ie. The valid least ActivePower value to crop from.  
    * Default value: **0**  
6. turbineStateFilter  
    * Valid Turbine State to filter proper data.  
    * Default value: **100**  
7. dbscanEps  
    * The variable **eps** for the DBSCAN Clustring Algorithm, used to **filter the noise**.  
    * ref: [sklearn.cluster.DBSCAN][dbscan link]  
    * Default value: **2**  
8. dbscanMinSamples  
    * The variable **min_samples** for the DBSCAN Clustring Algorithm, used to **filter the noise**.  
    * ref: [sklearn.cluster.DBSCAN][dbscan link]  
    * Default value: **4**  
9. rotationAngle  
    * The angle(in degrees) by which the axis should be rotated clockwise, so that the data transforms into a straight line.  
    * Default value: **24**  
10. blockSize  
    * The size of the block used in the computation of the stats([kurtosis][kurtosis link]).  
    * Default value: **10**  
11. plotGraph  
    * Expects a BOOLEAN.  
        * True: Prints Graph  
        * False: Does not print Graph  
    * Default value: **False**  
    
**Return Value:**    
_A Python [dictionary][python dict link]_  
* filePath: The complete path of the input File.  
* positiveInstances: The number of instances that gave a positive [kurtosis][kurtosis link] value.  
* positiveScore: The positive kurosis summation.  
* negativeInstances: The number of instances that gave a negative [kurtosis][kurtosis link] value.  
* negativeScore: The negative kurosis summation.  

_The complete score would be, **positiveScore + negativeScore**_, but sometimes it is nessary to take the number of instances into account too.

___
___
___

### Wind Speed vs Active Power

**How it works**    
The **main** function gives a performance drop score correcponding to the Data.    
Depending on that score, the data can be classified.

**Dependencies**    
1. [numpy][numpy link]: Fundamental package for scientific computing with Python.  
2. [pandas][pandas link]: Library providing high-performance, easy-to-use data structures and data analysis tools for the Python.  
3. [sklearn.cluster.DBSCAN][dbscan link]: Clustring algorithm, (**D**ensity-**b**ased **s**patial **c**lustering of **a**pplications with **n**oise).  
4. [matplotlib.pyplot][pyplot link]: Python 2D plotting library.  

___

**Function:** def main(...):    
_Main Function_

**Parameters:**    
_Arguments default to Mokal_
1. filePath  
    * Path of the file for running the function on.  
    * File Format should be **csv**.  
    * Should contain data on: WindSpeed, ActivePower, TurbineState.  
2. turbineStateFilter  
    * Valid Turbine State to filter proper data.  
    * Default value: **11**  
3. roundingMult  
    * Variable used in the function **def rounding(mult, div):**  
    * The value to multiply to the scaled value _(for discretizing)_.  
    * Default value: **2**  
4. roundingDiv  
    * Variable used in the function **def rounding(mult, div):**  
    * The value to divide the scaled value after multiplyting _(for discretizing)_, then converted to **int** format.  
    * Default value: **1**  
5. dbscanErrorEps  
    * The variable **eps** for the DBSCAN Clustring Algorithm, used to **filter the noise**.  
    * ref: [sklearn.cluster.DBSCAN][dbscan link]  
    * Default value: **2**  
6. dbscanErrorMinSamples  
    * The variable **min_samples** for the DBSCAN Clustring Algorithm, used to **filter the noise**.  
    * ref: [sklearn.cluster.DBSCAN][dbscan link]  
    * Default value: **4**  
7. yMinFilter  
    * Essentially cropping the data.  
    * The y-offset min value, ie. The valid least ActivePower value to crop from.  
    * Default value: **107**  
8. dbscanDegradeEps  
    * The variable **eps** for the DBSCAN Clustring Algorithm, used to **find the degraded points**.  
    * Only for clustring, does not give a **-1** label.  
    * ref: [sklearn.cluster.DBSCAN][dbscan link]  
    * Default value: **4**  
9. dbscanDegradeMinSamples  
    * The variable **min_samples** for the DBSCAN Clustring Algorithm, used to **find the degraded points**.  
    * Only for clustring, does not give a **-1** label.  
    * ref: [sklearn.cluster.DBSCAN][dbscan link]  
    * Default value: **0**  
10. dummyInputXMin  
    * The expected WindSpeed from with the ActivePower is supposed to be maximum.  
    * Default value: **21**  
11. scaledActivePowerStdLimit  
    * The maximum deviation in performance which is allowed.  
    * Default value: **4**  
12. grpSizeLimit  
    * The minimum number of power points at a specific WindSpeed for computation.  
    * Default value: **5**  
13. plotGraph  
    * Expects a BOOLEAN.  
        * True: Prints Graph  
        * False: Does not print Graph  
    * Default value: **False**  
    
**Return Value:**    
_A Python [dictionary][python dict link]_  
* filePath: The complete path of the input File.  
* performanceDropScore: The complete path of the input File.  
* dropInstances: The number of valid discrete points used in computation of Performance Drop.  
* activePowerStdScore: The summation of [standard deviation][std link] of ActivePower at ever discrete WindSpeed  
* activePowerStdInstances: The number of valid discrete points used in computation of [standard deviation][std link].  
* abnormalPerformancePoints: [DataFrame][pandas dataframe link] of possible abnormal points.  
* filteredErrorPointsDbscan: [DataFrame][pandas dataframe link] of points filtered by DBSCAN as errors.  
* duplicatedPoints: [DataFrame][pandas dataframe link] of points which are duplicated.  
___

**Function:** def rounding(...)::    
_Used in discritizing the scaled values_  

**Parameters**
1. mult  
    * Default value: 1  
2. div  
    * Default value: 2  

**Defination:**    
int(**number** * mult / div)  

___
___
___

[pandas dataframe link]: pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
[python dict link]: https://docs.python.org/2/tutorial/datastructures.html#dictionaries
[numpy link]: http://www.numpy.org/
[pandas link]: http://pandas.pydata.org
[dbscan link]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
[pyplot link]: http://matplotlib.org/api/pyplot_summary.html
[scipy stats link]: https://docs.scipy.org/doc/scipy/reference/stats.html
[kurtosis link]: https://en.wikipedia.org/wiki/Kurtosis
[std link]: https://en.wikipedia.org/wiki/Standard_deviation
