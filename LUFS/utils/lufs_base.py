import numpy as np

# 即时指标相关参数
class ParametersMomentary(object):
    def __init__(self):
        self.newSamples = 0 # 构造新帧需要的采样点数，对应一段数据长度（Segment）,对应时长0.1s
        self.windowLength = 0 # 分析帧采样点数， 对应一块数据（Block）,即时的帧长对应0.4s
        self.numSegments = 0 # 一个分析帧需要的新数据个数（A Block = 4 * Segment）

        return

# 短时指标相关参数
class ParametersShortTerm(object):
    def __init__(self):
        self.newSamples = 0 # 构造新帧需要的采样点数，对应一段数据长度（Segment），对应时长0.1s
        self.windowLength = 0 # 分析帧采样点数， 对应一块数据（Block），对应时长3s
        self.numSegments = 0 # 一个分析帧需要的新数据个数（A Block = 30 * Segment）

        return

# 即时相关状态
class StatesMomentary(object):
    def __init__(self):
        self.meanSquaredSum = np.zeros((4, 1)) # 4段采样点对应的均方和
        self.samplesProcessed = 0 # 已经处理的数据长度
        self.segmentIndex = 0 # 段索引
        self.prevLoudness = -np.infty # 上一帧的响度值
        self.segmentCount = 0 # 段计数器

# 短时相关状态
class StatesShortTerm(object):
    def __init__(self):
        self.meanSquaredSum = np.zeros((30, 1)) # 30段采样点对应的均方和
        self.samplesProcessed = 0 # 已经处理的数据长度
        self.segmentIndex = 0 # 段索引
        self.prevLoudness = -np.infty # 上一帧的响度值
        self.segmentCount = 0 # 段计数器

# 综合指标相关状态
class StatesIntergrated(object):
    def __init__(self):
        self.prevLoudness = -np.infty # 上一次计算的综合响度
        self.sumBlockPowerVec = [] # 列表，储存每一帧的能量
        self.lMomentaryVec = [] # 列表，储存每一帧的即时响度
        self.sumGreaterThanAbsThres = 0 # 所有大于绝对响度阈值（-70lufs）的帧能量和
        self.numGreaterThanAbsThres = 0 # 所有大于绝对响度阈值（-70lufs）的帧数量
        self.relativeThreshold = np.infty # 相对阈值门限（对于流程图中Relative Threshold Gate1）

# 响度动态范围指标相关状态
class StatesLRA(object):
    def __init__(self):
        self.prevLRA = 0 # 上一帧计算的响度范围
        self.lShortTermVec = [] # 列表，储存每一帧的短时响度