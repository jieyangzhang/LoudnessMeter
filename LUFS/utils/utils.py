import math
import numpy as np
from utils.lufs_base import *

import matplotlib.pyplot as plt

def TruePeakCal48k():
    return

# RMS值计算
def RMS(sig):
    rms = 20 * math.log10((np.sum(sig ** 2) / len(sig)) ** 0.5)

    return rms

def computeLoudnessRange(ShortTermLoudness):
    ABS_THRES = -70 # LUFS   (= absolute measure)
    REL_THRES = -20 # LU     (= relative measure)

    PRC_LOW = 10 # lower percentile
    PRC_HIGH = 95 # upper percentile
    ShortTermLoudnessArray = np.array(ShortTermLoudness) # list -> array

    # Apply the absolute-threshold gating
    abs_gate_vec = np.where(ShortTermLoudnessArray > ABS_THRES)
    # abs_gate_vec is indices of loudness levels above absolute threshold
    stl_absgated_vec = ShortTermLoudnessArray[abs_gate_vec]
    # only include loudness levels that are above gate threshold

    # Apply the relative-threshold gating (non-recursive definition)
    n = len(stl_absgated_vec); # 大于-70LUFS的帧数

    if n == 0:
        LRA = 0
        return LRA
    ten_array = np.zeros(n)
    ten_array += 10
    stl_power = np.sum(np.power(ten_array, stl_absgated_vec / 10)) # undo 10log10, and calculate mean
    stl_integrated = 10 * math.log10(stl_power) # LUFS
    # Relative Threshold Gate2的阈值计算：stl_integrated + REL_THRES
    rel_gate_vec = np.where(stl_absgated_vec >= stl_integrated + REL_THRES)
    # rel_gate_vec is indices of loudness levels above relative threshold
    stl_relgated_vec = stl_absgated_vec[rel_gate_vec]
    # only include loudness levels that are above gate threshold

    # Compute the high and low percentiles of the distribution of
    # values in stl_relgated_vec
    n = len(stl_relgated_vec)

    if n == 0:
        LRA = 0
        return LRA

    stl_sorted_vec = np.sort(stl_relgated_vec)
    # sort elements in ascending order
    stl_perc_low = stl_sorted_vec[round((n - 1) * PRC_LOW / 100)]
    stl_perc_high = stl_sorted_vec[round((n - 1) * PRC_HIGH / 100)]

    # Compute the Loudness Range measure
    LRA = stl_perc_high - stl_perc_low # in LU

    return LRA

def computeSampleLRA(lVec, prevLRA):

    LRA_sample = prevLRA
    if len(lVec) != 0:
        LRA_sample = computeLoudnessRange(lVec)
        prevLRA = LRA_sample

    return LRA_sample, prevLRA

# 计算综合响度
def computeSampleIntegratedLoudness(lVec, sumBlockPowerVec, sumGreaterThanAbsThres, \
    numGreaterThanAbsThres, relativeThreshold, prevLoudness):
    sn1 = 0.691
    loudness = prevLoudness
    if len(lVec) != 0:
        # 门限1: -70LUFS， 大于门限1的记入总缓存
        if lVec[-1] > -70:
            # sumGreaterThanAbsThres : 所有大于-70lufs的帧响度和
            # numGreaterThanAbsThres : 所有大于-70lufs的帧计数
            sumGreaterThanAbsThres = sumGreaterThanAbsThres + sumBlockPowerVec[-1]
            numGreaterThanAbsThres += 1
            # 计算相对响度的门限
            relativeThreshold = -sn1 + 10 * math.log10(sumGreaterThanAbsThres / numGreaterThanAbsThres) - 10
        # 计算响度帧缓存中大于门限的响度集合均值，取对数即为综合响度
        lVecArray = np.array(lVec)
        sumBlockPowerVecArray = np.array(sumBlockPowerVec)
        indexs = np.where(lVecArray > relativeThreshold)
        sum_term = np.sum(sumBlockPowerVecArray[indexs])
        N_Jg = len(lVecArray[indexs])
        if N_Jg > 0:
            # 计算综合响度
            loudness = -sn1 + 10 * math.log10(sum_term / N_Jg)
            prevLoudness = loudness

    return loudness, sumGreaterThanAbsThres, numGreaterThanAbsThres, relativeThreshold, prevLoudness

def computeSampleLoudness(meanSquaredSum, segmentCount, numSegments, segmentIndex, prevLoudness):
    loudness = prevLoudness
    sumBlockPower = 0
    # compute a new value for loudness
    # block等效于一帧，一帧有三段对应100ms× (3 / 29)
    # 前3 for momentary / 29 for shortTerm segment内不计算响度，当达到一帧长度后，每当有0.1s新数据，就计算响度
    segmentCount += 1
    if segmentCount > numSegments - 1:
        sumBlockPower = meanSquaredSum[segmentIndex].copy()
        loudness = -0.691 + 10 * math.log10(sumBlockPower) # 依据标准计算响度
        prevLoudness = loudness
    meanSquaredSum[segmentIndex] = 0 # 清空a block长度数据的均方值
    # 索引循环
    if segmentIndex == numSegments - 1:
        segmentIndex = 0
    else:
        segmentIndex += 1
    return loudness, meanSquaredSum, segmentCount, segmentIndex, sumBlockPower, prevLoudness

def computeLoudnessAndLRA(u_filtered,
                            pParamtersMomentary:ParametersMomentary,
                            pParamtersShortTerm:ParametersShortTerm,
                            pstateMomentary:StatesMomentary,
                            pstatesShortTerm:StatesShortTerm,
                            pstatesIntergrated:StatesIntergrated,
                            pstatesLRA:StatesLRA):
    # Create local copies
    newSamplesMomentary = pParamtersMomentary.newSamples # 构造一帧需要的新采样点个数
    windowLengthMomentary = pParamtersMomentary.windowLength # 即时帧长
    numSegmentsMomentary = pParamtersMomentary.numSegments # 即时帧对应的段数，a segment = new Samples

    newSamplesShortTerm = pParamtersShortTerm.newSamples
    windowLengthShortTerm = pParamtersShortTerm.windowLength
    numSegmentsShortTerm = pParamtersShortTerm.numSegments
    '''
    shape=[4, 1], 4*1队列, 每次循环队列的每个单位加上a segment数据的均方和（平均除的是a block的长度），循环3次后，
    每次取出一个单元的数据; 取出来的数据刚好就是a block数据的均方和。
    '''
    meanSquaredSumMomentary = pstateMomentary.meanSquaredSum
    # 输入数据长度和a segment长度的余数，他们的均方和会计算到meanSquaredSum中，下一次的输入需要补齐数据刚好到a segment的长度
    samplesProcessedMomentary = pstateMomentary.samplesProcessed
    segmentIndexMomentary = pstateMomentary.segmentIndex # meanSquaredSumMomentary队列的循环索引
    prevLoudnessMomentary = pstateMomentary.prevLoudness # 当输入数据不够一帧时，使用上一帧的响度
    segmentCountMomentary = pstateMomentary.segmentCount # 即时指标分析一帧的分段数

    '''
    shape=[30, 1], 30*1队列, 每次循环队列的每个单位加上a segment数据的均方和（平均除的是a block的长度），循环29次构成一帧后，
    每次循环取出一个单元的数据; 取出来的数据刚好就是a block数据的均方和。
    '''
    meanSquaredSumShortTerm = pstatesShortTerm.meanSquaredSum
    samplesProcessedShortTerm = pstatesShortTerm.samplesProcessed
    segmentIndexShortTerm = pstatesShortTerm.segmentIndex # meanSquaredSumShortTerm队列的循环索引
    prevLoudnessShortTerm = pstatesShortTerm.prevLoudness # 当输入数据不够一帧时，使用上一帧的响度
    segmentCountShortTerm = pstatesShortTerm.segmentCount # 短时指标分析一帧的分段数

    prevLoudnessIntegrated = pstatesIntergrated.prevLoudness
    sumBlockPowerVec = pstatesIntergrated.sumBlockPowerVec
    lMomentaryVec = pstatesIntergrated.lMomentaryVec
    sumGreaterThanAbsThres = pstatesIntergrated.sumGreaterThanAbsThres
    numGreaterThanAbsThres = pstatesIntergrated.numGreaterThanAbsThres
    relativeThreshold = pstatesIntergrated.relativeThreshold

    lShortTermVec = pstatesLRA.lShortTermVec
    prevLRA = pstatesLRA.prevLRA

    numRows = len(u_filtered)
    loudnessMomentary = np.zeros(numRows)
    loudnessShortTerm = np.zeros(numRows)
    loudnessIntegrated = np.zeros(numRows)
    LRA = np.zeros(numRows)

    # Calculate mean squared value
    u_filtered_squared = u_filtered ** 2; # 输入信号平方
    meanSquaredMomentary = u_filtered_squared / windowLengthMomentary # 即时均方值，按照一帧长度平均
    meanSquaredShortTerm = u_filtered_squared / windowLengthShortTerm # 短时均方值，按照一帧长度平均
    rowIndex = 0
    samplesLeftForSegment = newSamplesMomentary - samplesProcessedMomentary # number of samples missing in a segment
    samplesLeftInInput = numRows # number of samples available from input, 输入数据的采样点数

    # Momentary and integrated loudness
    while (samplesLeftForSegment <= samplesLeftInInput):
        loudnessMomentary[int(rowIndex) : int(rowIndex + samplesLeftForSegment - 1)] = prevLoudnessMomentary
        loudnessIntegrated[int(rowIndex) : int(rowIndex + samplesLeftForSegment - 1)] = prevLoudnessIntegrated
        # Update mean squared sum of one segment
        meanSquaredSumMomentary += np.sum(meanSquaredMomentary[int(rowIndex) : int(rowIndex + samplesLeftForSegment)])
        # Compute new value of momentary loudness
        lMomentary, meanSquaredSumMomentary, segmentCountMomentary, segmentIndexMomentary, \
            sumBlockPower, prevLoudnessMomentary = computeSampleLoudness( \
            meanSquaredSumMomentary,  segmentCountMomentary, numSegmentsMomentary, \
            segmentIndexMomentary, prevLoudnessMomentary)
        # if segmentIndexMomentary > 0 and rowIndex < 480:
        loudnessMomentary[int(rowIndex+samplesLeftForSegment-1)] = lMomentary

        # 当处理数据超过1帧长度后，将响度和每一帧的能量保存
        if (segmentCountMomentary > (numSegmentsMomentary-1)):
            sumBlockPowerVec.append(sumBlockPower)
            lMomentaryVec.append(lMomentary)

        # Compute new value of integrated loudness
        lIntegrated, sumGreaterThanAbsThres, numGreaterThanAbsThres, \
            relativeThreshold, prevLoudnessIntegrated = computeSampleIntegratedLoudness(\
            lMomentaryVec, sumBlockPowerVec, sumGreaterThanAbsThres, \
            numGreaterThanAbsThres, relativeThreshold, prevLoudnessIntegrated)
        loudnessIntegrated[int(rowIndex+samplesLeftForSegment-1)] = lIntegrated

        rowIndex = rowIndex + samplesLeftForSegment # 每次处理a segment数据
        samplesLeftInInput = samplesLeftInInput - samplesLeftForSegment
        samplesLeftForSegment = newSamplesMomentary
        samplesProcessedMomentary = 0
    # 如果输入数据残留长度小于a segment部分记录长度，并计算均方值并存入meanSquaredSumMomentary
    samplesProcessedMomentary += samplesLeftInInput
    if samplesLeftInInput > 0:
        loudnessMomentary[int(rowIndex):] = prevLoudnessMomentary
        loudnessIntegrated[int(rowIndex):] = prevLoudnessIntegrated
        meanSquaredSumMomentary += np.sum(meanSquaredMomentary[int(rowIndex):])

    # 计算短时响度以及响度范围指标
    rowIndex = 0
    samplesLeftForSegment = newSamplesShortTerm - samplesProcessedShortTerm
    samplesLeftInInput = numRows
    while (samplesLeftForSegment <= samplesLeftInInput):
        loudnessShortTerm[int(rowIndex):int(rowIndex+samplesLeftForSegment-1)] = prevLoudnessShortTerm
        LRA[int(rowIndex):int(rowIndex+samplesLeftForSegment-1)] = prevLRA
        # Updata mean squared sum of one segment
        meanSquaredSumShortTerm += np.sum(meanSquaredShortTerm[int(rowIndex) : int(rowIndex + samplesLeftForSegment)])
        # compute new value of ShortTerm loudness
        lShortTerm, meanSquaredSumShortTerm, segmentCountShortTerm, segmentIndexShortTerm, \
            _, prevLoudnessShortTerm = computeSampleLoudness(\
            meanSquaredSumShortTerm,  segmentCountShortTerm, numSegmentsShortTerm, \
            segmentIndexShortTerm, prevLoudnessShortTerm)
        loudnessShortTerm[int(rowIndex+samplesLeftForSegment-1)] = lShortTerm
        if (segmentCountShortTerm > (numSegmentsShortTerm-1)):
            lShortTermVec.append(lShortTerm)
        LRA_sample, prevLRA = computeSampleLRA(lShortTermVec, prevLRA)
        LRA[int(rowIndex+samplesLeftForSegment-1)] = LRA_sample

        rowIndex = rowIndex + samplesLeftForSegment
        samplesLeftInInput = samplesLeftInInput - samplesLeftForSegment
        samplesLeftForSegment = newSamplesShortTerm
        samplesProcessedShortTerm = 0
    samplesProcessedShortTerm = samplesProcessedShortTerm + samplesLeftInInput
    # There might be some input samples remaining
    if samplesLeftInInput>0:
        loudnessShortTerm[int(rowIndex):] = prevLoudnessShortTerm
        LRA[int(rowIndex):] = prevLRA
        meanSquaredSumShortTerm += np.sum(meanSquaredShortTerm[int(rowIndex):])

    pstateMomentary.meanSquaredSum = meanSquaredSumMomentary
    pstateMomentary.samplesProcessed = samplesProcessedMomentary
    pstateMomentary.segmentIndex = segmentIndexMomentary
    pstateMomentary.prevLoudness = prevLoudnessMomentary
    pstateMomentary.segmentCount = segmentCountMomentary

    pstatesShortTerm.meanSquaredSum = meanSquaredSumShortTerm
    pstatesShortTerm.samplesProcessed = samplesProcessedShortTerm
    pstatesShortTerm.segmentIndex = segmentIndexShortTerm
    pstatesShortTerm.prevLoudness = prevLoudnessShortTerm
    pstatesShortTerm.segmentCount = segmentCountShortTerm

    pstatesIntergrated.prevLoudness = prevLoudnessIntegrated
    pstatesIntergrated.sumBlockPowerVec = sumBlockPowerVec
    pstatesIntergrated.lMomentaryVec = lMomentaryVec
    pstatesIntergrated.sumGreaterThanAbsThres = sumGreaterThanAbsThres
    pstatesIntergrated.numGreaterThanAbsThres = numGreaterThanAbsThres
    pstatesIntergrated.relativeThreshold = relativeThreshold

    pstatesLRA.lShortTermVec = lShortTermVec
    pstatesLRA.prevLRA = prevLRA

    return loudnessMomentary, loudnessShortTerm, loudnessIntegrated, LRA


