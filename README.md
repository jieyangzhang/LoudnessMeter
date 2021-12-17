# 响度计算脚本

## A计权

运行A计权脚本，并输入测试音频文件：

```bash
python A_weight.py
please input audio_file to test: audio.wav
```

计算得到A计权RMS值：

```bash
src sig rms : -20.490578dB
filterd sig rms : -26.187433dB
```

信号的RMS值为-20.490578dB； A计权滤波后的RMS值为-26.187433dB

## LUFS

LUFS参考MATLAB LoudnessMeter实现方式，计算瞬时/短时/综合/动态范围/真峰值指标，目前真峰值未实现。

运行LUFS/main.py，输入测试文件:

```bash
python LUFS/main.py
please input audio_file to test: audio.wav
```

得到响度指标画图。