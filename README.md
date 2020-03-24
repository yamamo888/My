# Particle Filter for Nankai Trough simulation


### Directory

- savetxt/eq
	- 全発生年数
- savetxt/lh
	- 尤度
- images/lhs
	- 尤度
- images/numlines
	- 発生年数
- images/deltaU
	- 最小誤差年数1400年を取得するため
- images/PF
	- parameterの変化を見るため

***

## 使い方

### Particle Filter

`PF.py`

- 設定
mode 0:学習と評価, 1:真値がない時だけ+penalty, 2:全てに+penalty

<br>

`makingDataPF.py`

deltaU, theta, V, B を読み取る


- 類似度
	- ガウスではかる



### Result of Experiment

`resultPF.py`

`PlotPF.py`