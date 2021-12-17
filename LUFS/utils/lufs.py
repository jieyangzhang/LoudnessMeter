from utils.k_weighting import *
from utils.lufs_base import *
from utils.utils import *

class LUFS(object):
    def __init__(self):
        self.fs = 48000 # FIXME， 目前仅支持48k采样率的音频响度计算
        self.preFilter = kWeightingFilter() # 预处理滤波，K计权滤波
        self.paramMomentary = ParametersMomentary() # 短时计算相关参数
        self.paramShortTerm = ParametersShortTerm() # 长时计算相关参数
        self.stateMomentary = StatesMomentary() # 短时相关状态
        self.stateShortTerm = StatesShortTerm() # 长时相关状态
        self.stateIntergrated = StatesIntergrated() # 综合指标相关状态
        self.stateLRA = StatesLRA() # 响度范围相关状态

        windowDurationMomentary  = 0.4 # 短时帧长400ms
        overlapDurationMomentary = 0.3 # 短时帧重叠时长300ms, 75%重叠率
        updateDurationMomentary = windowDurationMomentary - overlapDurationMomentary # 构造新的帧需要的时长， 100ms
        newSamplesMomentary = updateDurationMomentary * self.fs # 构造一帧需要更新的采样点数, 48k * 0.1 = 4800(samples)
        # 实时分析帧对应的采样点数
        windowLengthMomentary = windowDurationMomentary / updateDurationMomentary * newSamplesMomentary
        # a segment对应更新一帧需要的信号，一帧实时帧一共有4segment，重叠3 segment
        numSegmentsMomentary = round(windowLengthMomentary / newSamplesMomentary)

        windowDurationShortTerm = 3
        updateDurationShortTerm = 0.1
        newSamplesShortTerm = updateDurationShortTerm * self.fs
        windowLengthShortTerm = windowDurationShortTerm / updateDurationShortTerm * newSamplesShortTerm
        numSegmentsShortTerm = round(windowLengthShortTerm/newSamplesShortTerm)

        self.paramMomentary.newSamples = newSamplesMomentary
        self.paramMomentary.windowLength = windowLengthMomentary
        self.paramMomentary.numSegments = numSegmentsMomentary

        self.paramShortTerm.newSamples = newSamplesShortTerm
        self.paramShortTerm.windowLength = windowLengthShortTerm
        self.paramShortTerm.numSegments = numSegmentsShortTerm

        return

    def step(self, u):
        # step1, K计权预滤波
        data_len = len(u)
        u_filtered = np.zeros(data_len)
        for i in range(data_len):
            u_filtered[i] = self.preFilter.step(u[i])

        # step2, 计算即时响度/综合响度/短时响度/响度范围
        loudnessMomentary, loudnessShortTerm, loudnessIntegrated, LRA = \
        computeLoudnessAndLRA(u_filtered, self.paramMomentary,\
                                        self.paramShortTerm, \
                                        self.stateMomentary, \
                                        self.stateShortTerm, \
                                        self.stateIntergrated,\
                                        self.stateLRA)

        sec = data_len / self.fs
        t = np.linspace(0, sec, data_len)
        plt.subplot(211)
        plt.plot(t, loudnessMomentary, label='loudnessMomentary')
        plt.plot(t, loudnessIntegrated, label='loudnessIntegrated')
        plt.plot(t, loudnessShortTerm, label='loudnessShortTerm')
        plt.xlabel('Time(Sec)')
        plt.ylabel('Loudness')
        plt.title('LUFS (Integrated Loudness:%.2flufs)' % loudnessIntegrated[-1])
        plt.legend()
        plt.subplot(212)
        plt.plot(t, u)
        plt.xlabel('Time(Sec)')
        plt.ylabel('amplitude')
        plt.title('Time domain')
        plt.show()

        # step3, 计算真峰值
        # TODO补充真峰值的计算

        return