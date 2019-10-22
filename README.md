# 南海トラフ巨大地震シミュレーション用に改良したEnKF

## 項目 [Contents]

1. [真の南海トラフ巨大地震履歴地震時データ　&　累積変位データ作成 : `NankaiTrough.py & EnKF.py`](#ID_1)
2. [最小年数取得 : `makingData.py`](#ID_2)
3. [Ensamble Kalman Filter : `EnKF.py`](#ID_3)
  1. [使い方](#ID_3-1)
  2. [アンサンブル変数読み取り](#ID_3-2)




## CycleBranch2.cpp 変更点
- CycleBranch2.vcxproj をいじる
- 77 ~ 83行目
  - 初期アンサンブル作成時は、`parfileHM031def`指定
  - それ以外は81行目`fopen_s(&fp, argv[1], "r");` のコメントを外し、`fopen_s(&fp,mlid,"r")`をコメントアウト
  
-102 ~ 105 行目
  - `fscanf_s(fp, "...", &ai[i],..., &pu[i], &pth[i], &pv[i])`のコメントを外す
  - `fscanf_s(fp, "...", &ai[i],..., &vp[i])`をコメントアウト
 - b1[i] = atof...をコメントアウトにする

- 304-319 行目
  - initial v,th,u
  - 初期アンサンブル作成時はdefault以下3行分を使用
  - それ以外は、EnkF以降3行分を使用、更新したu,v,thを初期値次のu,v,thとして使用
  
- 470 ~ 472 行目
- U, theta, V をlogファイルに書き込み

``` python  
printf("%d,%lf,%lE,%lE,%lE,%lE,%lE,%lE,%lE,%lE\n", nr, Xyr, u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7]);
printf("%d,%lf,%lE,%lE,%lE,%lE,%lE,%lE,%lE,%lE\n", nr, Xyr, th[0], th[1], th[2], th[3], th[4], th[5], th[6], th[7]);
printf("%d,%lf,%lE,%lE,%lE,%lE,%lE,%lE,%lE,%lE\n", nr, Xyr, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
```

***
<a id="ID_1"></a>

## 1. 真の南海トラフ巨大地震履歴地震時データ　&　累積変位データ作成 : `NankaiTrough.py & EnKF.py`
- 実行すると地震時データを作成 -> pickle保存(nankairireki.pkl) <br>
※すべり速度指定できる
- convV2YearlyData & main 内のコメントアウトを外すと作成->pickle保存(NankaiCD.pkl) <br>
※累積変位データを作成したいファイルを指定する必要がある(現在は一番多く地震が発生する履歴のみ) 
- 【注意】試すときもファイル数2つ以上でないと、アンサンブルの計算ができない


***
<a id="ID_2"></a>

## 2. 最小年数取得 : `makingData.py`
``` python
sInd = sumAllError.argmin()
# last index
eInd = sInd + Year
```
- `sumAllError` : 真の地震発生回数 < 予測した地震発生回数　場合は、大きい値を代入
- `sInd` : 8000年から1400年間に切り取るときの開始年数. すべて真の地震発生回数 < 予測した地震発生回数の場合は`sInd = 0`、2000年が開始年数

***
<br>

## 1.初期アンサンブル作成　& データ作成 : MakeFirstEn.bat
- サンプリング範囲、サンプリング間隔を決定する
- ディレクトリlogsに、first*.txtが出力される
- `logs`ディレクトリに first Ensemble 格納
- `firstEnsemble`ディレクトリにコピーある
- ** 初期アンサンブル作成年数をtとしていいのか。データごとに異なる上に、南海トラフ地震より発生年数が大きかったらどうする？**

### シミュレーションデータ読み込み
  - ΔUが8000年以降だと大きすぎるので、2000年以前の累積変位量Uから引き算して求める

### 南海トラフ巨大地震履歴
  - `gtV`はすべり速度の大きさは一定
  - `gtU`はTPモデルに従い作成、累積変位量もここから、


***
<a id="ID_3"></a>

## 3.EnKF: `EnKF.py`
- 1年単位で出力してるから、すべり速度が0になることはない
- ~~ΔUを観測行列を使って計算したかったが、行列演算できなかったので、手動で計算~~
- アンサンブルメンバーの作成方法は、1.1 スピンアップ、1.2 地震がどこかで起きた場合 
- `logs`ディレクトリに first Ensemble 格納されているか確認

<a id="ID_3-1"></a>

### 3-1. 使い方

以下、コマンド引数の説明

|コマンド引数番号|変数名|役割|
|:---:|:---|:---|
|1|cell|セル番号|
|2|sigma|観測行列Rの分散|


```python
```

<a id="ID_3-2"></a>

### アンサンブルメンバー作成
- パラメータ
  - `iS` : first Ensembleを読み込むときとの区別、logファイルの通し番号指定(読み取り回数) 
   - ex) files = \[s for s in files if "logs_{}_".format(**iS**) in s]\ iS=100やとすれば、100.txt のファイルがすべて読み込まれる。
  - `jisins (list)` :  地震が発生した or していないの 1,0 が格納される、発生したかどうかは、すべり速度V > silp(=1) で判断 (南海、東南海、東海の5セル分)
   - ex) V = np.array([0,1,0,0,0,5,2,4,4]) -> jisin = [1,0,0,0,1]
　 - `gtY` : 南海地震履歴がアンサンブルの同化年数に選ばれた時にインクリメント。 1.2で使用, `gtY=1`から始める `gtY-1` < ... < `gtY` って使うから
   - `gtJ` : 真値の地震発生年数格納（すべてのセルの分、被ってるのは除外）
   - `isWindows` : windows user
   - `nCell` : シミュレーションのすべてのcell数


``` python: Empty
```
- 何もシミュレーション結果が出力しなかった場合は、logファイルを削除

``` python: Negative
```
- 累積変位量、すべり速度、潜在変数のいずれかがマイナスの値の場合は、logファイルを削除　（あってる？）


``` python
# データ読み取り
U,th,V,B = myData.loadABLV(logsPath,file,nCell)
# U,th,V,paramb取得(タイムステップt-1) [地震発生年数(=?),8]
yU,yth,yV = myData.convV2YearlyData(U,th,V,nCell,nYear)
```
- `U,th,V,B`は発生時データ、`yU,yth,yV`は8セル分の年数時データ(単位year)、8セル分 (必要か？)
- 2回目以降は同化開始年数が `index=0` に対応、最初は8000年分

```python:makingData.py
```
- *1セルの`A,B,L`を読み取るバージョンしか作ってない*


### 3.3 スピンアップ
- 始まりの年はバラバラ、アンサンブルと真値で初めて地震発生時を合わせる
- 0年目に地震が起きたことを想定していない

``` python
 # 最小の1400年間を取得
# U,th,V = [1400,8]
predU,predth,predV = myData.GenerateMinSimlatedNankai(gtV,yU,yth,yV)

# Assimilation start year (index) of all cell with simulation
sjInd = sorted(set(np.where(predV>slip)[0]))[0]
# gt
tjInd = np.where(gtV>slip)[0][0]

# Vt one kalman (※ sjInd==0 is ban), sjInd == year eq.
Vt = np.hstack([predU[sjInd,:],predU[sjInd-1,:],predth[sjInd,:],predV[sjInd,:],B])
# gt (time-step t), tjInd == year eq.
TrueU = gtU[tjInd,:]
```
- `predU,predth,predV`: 8000年あるけど、最小誤差の1400年を取得する (最初の時だけ)
- `jInd` : 1400年に切り出されたシミュレーションデータで初めに起きた地震のindex (slip velocity V を使用。※累積変位量Uはすでに値が入っている状態だから)
- `Vt `: 状態ベクトル
- `TrueU`: 観測量
- `aInd` : 同化開始年数(アンサンブル分格納)

<br>

``` python
# jisin or non jisin -> True or False
tmpJ = [j > slip for j in predV[sjInd,:]] 
# True, False -> 1,0
jisins = [1 if jisin else 0 for jisin in tmpJ]

# assimilation index
aInd = sjInd
```
- `jisins`: 地震が起きた or 起きてないか -> 0,1

***
<br>


### 1.2 2回目以降の地震発生時の同化方法
 
```python
```
  
- `pJ` : シミュレーション(アンサンブルメンバー)の地震発生時を格納 & 真の南海トラフ巨大地震履歴の地震発生時 `gtJ` も格納
- `uthvs` : すべての同化年数分の特徴量格納、のちに同化開始年数指定 
  - `yUex` : 一期前の同化年数は使わないかもしれない

<br>

```python
```

- `pJ`をsortして一期前以降の最小同化開始年数を取得
- `aInds` : 1期前の同化年数格納 (すべてのアンサンブルメンバーで同化年数をそろえる)、のちの`Uex`の同化開始年数に使用
- `aInd` : 各アンサンブルの一期前の同化年数
- `jInd` : 同化開始年数、1期前のものよりも進んだ年数　(あるアンサンブルで同化開始前に起きていても無視でええん？)
<br>

```python
```

- 同化開始年数 `jInd` を指定して特徴量を作るために、もう一回アンサンブル分回す (ダサい)
- `Uex` : 一期前の累積変位時系列、`U`が0年目を仮定していない




***

## EnKFのアルゴリズムの計算

### 2.1 カルマンフィルタ
  - 観測行列`H`は1行目が南海、2行目が東南海、3行目が倒壊に対応する。また、8セルから3セルに合わせるために、南海の2セル目(index=3)、東南海の2セル目(index=4)、東海(index=5)に1を代入し、それに対応する`U_t-1`が格納されているところに-1を格納する

### カルマンゲイン
  - 観測誤差共分散行列`R`>> `HP^f_tH^T`であれば、`K~0`となり予報値の修正はほぼ0


### 予測アンサンブルの更新
- 誤差`r_t`と予報値はアンサンブル分
- 誤差はアンサンブル毎に観測量の実現値`y^o_t + r_t`を用意する必要あり



``` python
Xf_t, Pf_t, yU, yexU, yth, yV, noizeb = Odemetry(Xt) # IN:Ensambles,[number of Ensambles,parameters(8(cell)*5(Ut,Ut-1,th,V,B))]
# 観測値
Yo_t,r_t = Observation(TrueU)
# 4.カルマンゲイン
K_t, H = KalmanFilter(Pf_t)
# 5.予報アンサンブル更新
Xal_t = UpData(noizeb,Yo_t,r_t,H,Xf_t,K_t)
```

<br>



***
<br>


## 3.数値シミュレーション(M) : NumericalSimulation (C#)
- 与えられたparamb,u,v,thを用いて数値計算
- アンサンブル番号の付いたlogファイルをディレクトリlogsに出力

### コード
  - パラメータ
    - `sYear`, `eYear`: シミュレーターで計算する期間 `pJ` は更新時には、t-1 になるので `pJ + 1`、`sYear` はアンサンブルごとで異なる。`eYear` は任意
  
``` python
for lNum in np.arange(Xal_t.shape[0]):
```
- `lNum`がアンサンブルに対応. アンサンブルメンバー分 for 回す

2.に戻る


最終予測したparamb,U,th,Vは、logファイルに残る(発散したパラメータのlogファイルは残らない)


***
<br>


## 4.最終結果
最終的に予測したファイルをLastEnsambleディレクトリに移動



## 平均、予測した摩擦パラメータb の散布図作成

- 南海2、東南海2、東海1セルの合計5セル分 `cellInd` 予測した摩擦パラメータの散布図を作成



### コード



``` python : main
```
