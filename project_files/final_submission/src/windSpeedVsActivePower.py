
# coding: utf-8

# In[18]:


## import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os

def rounding(mult=1, div=2):
    return lambda x: int((x * mult) / div)


def main(
    filePath,
    turbineStateFilter=11,
    roundingMult=2,
    roundingDiv=1,
    dbscanErrorEps=2,
    dbscanErrorMinSamples=4,
    yMinFilter=107,
    dbscanDegradeEps=4,
    dbscanDegradeMinSamples=0,
    dummyInputXMin=13,
    scaledActivePowerStdLimit=4,
    grpSizeLimit=5,
    plotGraph=False
):

    try:

        abnormalPerfGrpList = []
        dataSet = pd.read_csv(filePath)
        duplicatedPoints = dataSet[dataSet.duplicated(subset=['WindSpeed', 'ActivePower', 'RotorSpeed', 'WindDirection'], keep='first')].reset_index()
        dataSet = dataSet.drop_duplicates(subset=['WindSpeed', 'ActivePower', 'RotorSpeed', 'WindDirection'], keep='first')

        # Filter values
        dataSet = dataSet[
            (dataSet['WindSpeed'] > 0)
            & (dataSet['TurbineState'] == turbineStateFilter)
            & (dataSet['ActivePower'] > yMinFilter)
        ]

        # Normalize Values
        minActivePower = dataSet['ActivePower'].min()
        maxActivePower = dataSet['ActivePower'].max()
        minWindSpeed = dataSet['WindSpeed'].min()
        maxWindSpeed = dataSet['WindSpeed'].max()

        dataSet['scaledActivePower'] = (dataSet['ActivePower'] - minActivePower) * 100 / (maxActivePower - minActivePower)
        dataSet['scaledWindSpeed'] = (dataSet['WindSpeed'] - minWindSpeed) * 100 / (maxWindSpeed - minWindSpeed)
        
        # Cluster for errors
        dataSet['dbscanLabel'] = DBSCAN(eps=dbscanErrorEps, min_samples=dbscanErrorMinSamples).fit_predict(dataSet[['scaledWindSpeed', 'scaledActivePower']])

        # Discritize
        dataSet['discreteWindSpeed'] = dataSet['scaledWindSpeed'].apply(rounding(roundingMult, roundingDiv))

        # Additional Filtering
        filteredErrorsDbscan = dataSet[(dataSet['dbscanLabel'] == -1)].reset_index()
        dataSet = dataSet[
            (dataSet['dbscanLabel'] != -1)
        ]

        # Check Performance
        performanceDropScore = 0
        dropInstances = 0
        performanceDropList = []
        activePowerStdScore = 0
        activePowerStdInstances = 0

        dummyRow = dataSet.loc[dataSet['scaledActivePower'].idxmax()].copy()
        dummyRow['Timestamp'] = '00-00-00T00:00:00'

        dbscanDegradeAlg = DBSCAN(eps=dbscanDegradeEps, min_samples=dbscanDegradeMinSamples)

        for grp in dataSet.groupby('discreteWindSpeed'):
            workGrp = grp[1].reset_index().copy()
            
            # Append Dummy Data
            dummyInputScaledXMin = rounding(roundingMult, roundingDiv)((dummyInputXMin - minWindSpeed) * 100 / (maxWindSpeed - minWindSpeed))
            
            if grp[0] > dummyInputScaledXMin:
                dummyRow['discreteWindSpeed'] = grp[0]
                dummyRow['scaledWindSpeed'] = grp[0]
                workGrp = workGrp.append(dummyRow).append(dummyRow)

            workGrp = workGrp.sort_values('scaledActivePower')
            workGrp['shiftedDown'] = workGrp.scaledActivePower.shift(1)
            grpSize = workGrp.shape[0]
            scaledActivePowerStd = np.std(workGrp['scaledActivePower'])
            if scaledActivePowerStd > scaledActivePowerStdLimit:
                workGrp['degradingPointsLabel'] = dbscanDegradeAlg.fit_predict(workGrp[['discreteWindSpeed' ,'scaledActivePower']])
                for _grp in workGrp.groupby('degradingPointsLabel'):
                    if ((_grp[0] == workGrp.iloc[-2]['degradingPointsLabel']) or
                           (_grp[0] == workGrp.iloc[-1]['degradingPointsLabel'])):
                        continue

                    abnormalPerfGrpList.append(_grp[1])

            activePowerStdScore = activePowerStdScore + scaledActivePowerStd
            activePowerStdInstances = activePowerStdInstances + 1

            if grpSize < grpSizeLimit:
                continue
            spaceSeries = list(workGrp['scaledActivePower'] - workGrp['shiftedDown'])[1:]
            performanceDropScore = performanceDropScore + np.max(spaceSeries)
            performanceDropList.append(np.max(spaceSeries))
            dropInstances = dropInstances + 1

        if len(abnormalPerfGrpList) != 0:
            abnormalPerfDataSet = pd.concat(abnormalPerfGrpList).reset_index()
        else:
            abnormalPerfDataSet = dataSet[[False] * dataSet.shape[0]]

        if plotGraph:
            plt.figure(figsize=(10, 10))
            plt.title('Scaled Scatter Plot')
            plt.xlabel('Scaled Wind Speed')
            plt.ylabel('Scaled Active Power')
            plt.scatter(filteredErrorsDbscan['discreteWindSpeed'], filteredErrorsDbscan['scaledActivePower'], s=np.pi*4*4, c='#89da59', label='Errors filtered after DBSCAN')
            plt.scatter(dataSet['discreteWindSpeed'], dataSet['scaledActivePower'], s=np.pi*4*4, c='#80bd9e', label='Proper Performance behaviour')
            plt.scatter(abnormalPerfDataSet['discreteWindSpeed'], abnormalPerfDataSet['scaledActivePower'], s=np.pi*4*4, c='#f98866', label='Abnormal Performance behaviour')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.show()            
    
    except Exception as e:
        return {
            'error': e
        }

    
    print 'Abnormal Performance Points in WindSpeed vs ActivePower graph.'
    print 'Arguments default to Mokal'
    
    return {
        'performanceDropScore': performanceDropScore,
        'dropInstances': dropInstances,
        'activePowerStdScore': activePowerStdScore,
        'activePowerStdInstances': activePowerStdInstances,
        'filePath': filePath,
        'abnormalPerformancePoints': abnormalPerfDataSet[['Timestamp', 'WindSpeed', 'ActivePower', 'TurbineState', 'WindDirection', 'RotorSpeed']],
        'filteredErrorPointsDbscan': filteredErrorsDbscan[['Timestamp', 'WindSpeed', 'ActivePower', 'TurbineState', 'WindDirection', 'RotorSpeed']],
        'duplicatedPoints': duplicatedPoints
    }
