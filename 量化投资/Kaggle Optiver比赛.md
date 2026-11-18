# Kaggle Optiver-Trading at the Close

In this competition, you are challenged to develop a model capable of predicting the closing price movements for hundreds of Nasdaq **listed stocks** using data from the **order book** and the **closing auction** of the stock.

## 背景知识

- listed stocks：上市股票
- delisted stocks：退市股票
- go public：上市
- go private：退市
- closing auction：交叉拍卖。纳斯达克的收盘集合竞价环节，对一段时间内接收的买卖申报一次性集中撮合的竞价方式。
  - 时间：下午 3 点 50 分至 4 点
  - 价格确定：在有效价格范围内选取成交量最大的价位
  - 作用：收盘价决定了指数基金和其他投资策略的基准定价
- Order book：订单簿
- Order：委托
- Bid Order：买入委托
- Ask Order/Offer Order：卖出委托
- Spread：价差
- Cross：经纪人以同一价格收到同一股票的买入和卖出订单，随后在两个不同的客户之间以该价格同时进行交易
- Opening Cross：【类似 A 股开盘集合竞价，确定开盘价】纳斯达克在开盘前两分钟收集并公布所有买卖的意向报价，交易者可以以开盘价购买或者在 imbalance 时买入，可以保证流动性。
- Closing Cross：【类似 A 股收盘集合竞价，确定收盘价】纳斯达克的 Closing Cross 将某只股票的买入价和卖出价相匹配，从而形成当天的最终价格。纳斯达克在下午 3 点 50 分到下午 4 点之间收集 Closing Cross 的数据。交叉指令在下午 4 点到下午 4 点后 5 秒之间执行。因为收盘价是很多指数基金和策略重要的基准，所以引入收盘竞价保证价格不被操纵。
- IO：**Imbalance Only Orders** 美国股票交易的正常时间为美国东部时间每个工作日的上午 9:30 至下午 4:00（只要不是假日或前总统的葬礼）。然而，一些交易发生在开盘前和收盘后。在非工作时段发生的波动会被计入第二天的开盘价。因此，周一下午 4 点收盘时的股票价格可能与周二上午 9 点 30 分开盘时的价格不同。这种变动有时使计算开盘和收盘时的价格变得复杂。交易者通常喜欢在这些时点下单，要么在开盘前（pre-market trading）下单，要么收盘前下单。短时间内大量交易，有可能导致开盘价和收盘价失衡。
  - “不平衡”意味着账簿的一边比另一边有更多的订单，要么是买入订单多于卖出订单，要么是卖出订单多于买入订单。NASDAQ 的 **Imbalance Only Orders** 被用在这些关键时点。它们是**限价单**，**只有在出现不平衡时才会激活**（因此得名，“仅限不平衡”）。其目标是创造流动性，使开盘和收盘时的价格走势更加平稳。

## 评价指标

$$
MAE=\frac{1}{n} \sum_{i=1}^{n}\left|y_{i}-x_{i}\right|
$$

## 数据特征

- stock_id：股票代码
- date_id：日期
- seconds_in_bucket：收盘集合竞价开始后过了多少秒，从 0 开始，到 540s，正好 9 分钟
- imbalance_size：以当前的 reference price 成交的话，不能成交的规模
- imbalance_buy_sell_flag：不平衡指标
  - buy-side imbalance：1
  - sell-side imbalance：-1
  - no imbalance：0
- reference_price：“参考价”，在达到该价格时，可配对股票总数被最大化，股票买卖不平衡被最小化，也可以认为是买一和卖一价之间最接近的价格
- matched_size：当前 reference price 下可成交的数量
- far_price：集合竞价中，能使得成交量最大的意向报价价格（不考虑连续市场订单）
- near_price：集合竞价和连续交易中，成交量最大的价格
- [bid/ask]_price：买一/卖一价（非集合竞价）
- [bid/ask]_size：买一/卖一规模
- wap  
  - weighted average price：加权均价
  - $\frac{ {BidPrice * AskSize + AskPrice * BidSize}}{BidSize + AskSize}$
- target
  - 股票未来 60 秒的 wap 走势 - 合成指数未来 60 秒 wap 走势
  - 位为基点单（0.01%）BP 
  - $Target = (\frac{StockWAP_{t+60}}{StockWAP_{t}} - \frac{IndexWAP_{t+60}}{IndexWAP_{t}}) * 10000$

## 特征处理

### Filtering 过滤法

过滤方法通常用作**预处理**步骤，特征选择完全独立于任何机器学习算法。根据各种统计检验中的分数以及相关性的各项指标来选择特征。

- **方差过滤**：通过特征本身的方差来筛选特征的类。 比如一个特征本身的方差很小，就表示样本在这个特征上基本没有差异，可能特征中的大多数值都一样，甚至整个特征的取值都相同，那这个特征对于样本区分没有什么作用。 所以无论接下来的特征工程要做什么，优先消除方差为 0 的特征。 VarianceThreshold 有重要参数 threshold，表示方差的阈值，表示舍弃所有方差小于 threshold 的特征，不填默认为 0，即删除所有的记录都相同的特征。

```python
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=3)  # 实例化，不填参数默认方差为0
X_var0 = selector.fit_transform(X)  # 获取删除不合格特征之后的新特征矩阵

# 过滤方差中位数
X_fsvar = VarianceThreshold(np.median(X.var().values)).fit_transform(X)
```

- **相关性过滤**：希望选出与标签相关且有意义的特征，如果特征与标签无关则不能提供信息，可能还会给模型带来噪音。sklearn 当中，有卡方，F 检验，互信息三种常用的方法来评判特征与标签之间的相关性。

```python
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
# 选择 K 个最好的特征，返回选择特征后的数据
# 第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数

# 参数 k 为选择的特征个数
SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)
```

- **互信息**：互信息法是用来捕捉每个特征与标签之间的任意关系（包括线性和非线性关系）的过滤方法，feature 和 target 关系越紧密，互信息越大。
  - 说“互信息”的时候，两个随机变量的地位是相同的；说“信息增益”的时候，是把一个变量看成减小另一个变量不确定度的手段。但其实二者的数值是相等的。 “互信息”或者“信息增益”经常作为决策树中选择特征的标准，两种称呼都很常见。 `Gain=H(X)-H(X/Y)`

$$
I(X ; Y)=\sum_{x \in X} \sum_{y \in Y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)}
$$

```python
from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores[::3]  # show a few features with their MI scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)
```

标准互信息：

```python
# 标准互信息
from sklearn import metrics
A = [1, 2, 3, 2, 3, 3]
B = [1, 2, 3, 1, 2, 3]
metrics.normalized_mutual_info_score(A, B)
# 计算结果 [0,1] 越大，关系越紧密
```

### Embedded 嵌入法

让算法自己决定使用哪些特征。

- 树模型：`feature_importances_`
- 回归模型：`coef_`

```python
from sklearn.ensemble import RandomForestRegressor as RF
import numpy as np
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.datasets import load_boston

# 导入波士顿房价数据集
boston = load_boston()
x, y = boston.data, boston.target

RF_ = RF(random_state=0)

selector = SelectFromModel(estimator=RF_).fit(x, y)

# selector 的属性
selector.estimator_.feature_importances_  # 特征重要性
selector.threshold_  # 特征重要性阈值（默认均值）
selector.get_support()  # 哪些特征入选（高于阈值）
selector.transform(x) # 筛选出大于阈值的特征
selector.threshold = 0.1
selector.transform(x) # 筛选出大于阈值的特征

# 如何选择合适的阈值
selector = SelectFromModel(estimator=RF_).fit(x, y)
threshold = np.linspace(0, (selector.estimator_.feature_importances_).max(), 10)
score = []
for i in threshold:
    selector.threshold = i
    X_embedded = selector.transform(x)
    once = cross_val_score(RF_, X_embedded, y, cv=3).mean()
    score.append(once)
plt.plot(threshold, score)
plt.show()
```

- Optiver 数据集

```python
from sklearn.ensemble import RandomForestRegressor as RF
import numpy as np
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

sample_ = df_.copy()
sample_['target'] = df_train['target']
sample = sample_.sample(frac=0.01) # 取 1% 的样本数据
sample = sample.replace([np.inf, -np.inf], np.nan)
sample = sample.dropna()
y = sample['target']
x = sample.drop('target', axis=1)

RF_ = RF(random_state=0)
RF_.fit(x, y)
RF_.feature_importances_

threshold = np.linspace(0, (RF_.feature_importances_).max(), 10)
score = []
for i in threshold:
    X_embedded = SelectFromModel(RF_, threshold=i).fit_transform(x, y)
    once = cross_val_score(RF_, X_embedded, y, cv=3).mean()
    score.append(once)
plt.plot(threshold, score)
plt.show()
```

### Wrapper 封装法

- 与嵌入法相似，封装法也依赖于算法自身的选择，比如 `coef_` 属性或 `feature_importances_` 属性来完成特征选择。但不同的是，封装法不需要人为设置某个评估指标或统计量的阈值，而是使用一个目标函数作为黑盒来选取特征。
- 区别于过滤法和嵌入法一次训练解决所有问题，封装法要使用特征子集进行多次训练，因此计算成本最高。  

RFE（Recursive Feature Elimination）是一种递归特征消除方法，主要目标是通过递归考虑越来越少的特征集来选择特征。在 sklearn 库中，RFE 类用于特征选择，接收外部估计器分配给特征的权重作为参数，例如线性模型的系数。然后根据学习器返回的 `coef_` 属性或 `feature_importances_` 属性来获得每个特征的重要程度，并从当前的特征集合中移除最不重要的特征。这个过程不断重复，直到达到所需的特征数量为止。此外，为了找到最优的特征数量，通常会结合交叉验证来进行优化。但需要注意的是，RFE 方法可能存在计算量大和随着学习器改变最佳特征组合也跟着改变的问题。

```python
from sklearn.ensemble import RandomForestRegressor as RF
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.datasets import load_boston

# 导入波士顿房价数据集
boston = load_boston()
x, y = boston.data, boston.target

RF_ = RF(random_state=0)
selector = RFE(RF_, n_features_to_select=5, step=2).fit(x, y)
# n_features_to_select 要选择的特征数量。如果为 None，则选择一半特征
# step: int or float, optional (default=1)  如果大于或等于 1，则 step 对应于每次迭代要删除的特征个数（整数）。
# 如果在（0.0，1.0）之内，则 step 对应于每次迭代要删除的特征的百分比（向下舍入）。
selector.ranking_
X_wrapper = selector.transform(x)

score = []
for i in range(1, 12, 2):
    print(i)
    X_wrapper = RFE(RF_, n_features_to_select=i, step=2).fit_transform(x, y)
    once = cross_val_score(RF_, X_wrapper, y, cv=3).mean()
    score.append(once)
plt.figure()
plt.plot(range(1, 12, 2), score)
plt.xticks(range(1, 12, 2))
plt.show()
```

## 模型融合

### Bagging

Bootstrap aggregating，一种有放回的抽样方法（可能抽到重复的样本）。

#### 平均法 Averaging

- 主要应用于回归问题
- 简单平均：将多个模型结果进行简单平均，得出最后结果
- 加权平均：根据模型表现（排名），分别赋予不同的权重，基于权重加权平均

#### 投票法 Voting

- 主要用于分类模型
- 硬投票：对多个模型直接进行投票，不区分模型结果的相对重要度
  - 绝对多数：最终结果必须在投票中占一半以上
  - 相对多数：最终结果在投票中票数最多
- 软投票：加权
  - 输出结果加权
  - 模型表现加权

举例：

模型1：A-99%，B-1%

模型2：A-49%，B-51%

模型3：A-40%，B-60%

模型4：A-90%，B-10%

模型5：A-30%，B-70%

- 硬投票：A 两票，B 三票，最终结果为 B
- 软投票：A (0.99 + 0.49 + 0.4 + 0.9 + 0.3) / 5 = 0.616，B (0.01 + 0.51 + 0.6 + 0.1 + 0.7) / 5 = 0.384，最终结果为A

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

model_1 = LogisticRegression(random_state=2020)
model_2 = DecisionTreeClassifier(random_state=2020)
model = VotingClassifier(estimators=[('lr', model_1), ('dt', model_2)], voting='hard')
model.fit(x_train, y_train)
model.score(x_test, y_test)
```

### Boosting 提升法

Boosting 的思路是迭代，每一次训练的时候都更加关心分类错误的样例（或未拟合残差）

### Stacking 堆叠法

当用初始训练数据学习出若干个基学习器后，将这几个学习器的预测结果作为新的训练集，来学习一个新的学习器。

## JS Kaggle

### 任务

金融市场的建模非常具有挑战性，因为涉及到一些在其他类型的建模问题中不常见的特性，如金融市场的分布可能是长尾的（即极端值出现的概率高于正态分布），或者时间序列可能是非平稳的（即统计特性随时间变化），而且可能不满足许多统计方法所依赖的基本假设。此外，金融市场本质上是人类行为的体现，涉及大量个人和机构，他们随着技术进步、社会变化、经济和地缘政治问题的出现而不断变化，这进一步增加了建模的难度。

使用 JS 生产系统的实时数据来构建模型。JS 提供了与运行自动交易策略的市场相关的一系列特征和响应变量，对数据中呈现的一些特征和响应变量进行了匿名化和轻度混淆处理。

### 评价指标

样本加权零均值 ${\rm R^2}$：

$$
{\rm R^2} = 1 - \frac{\sum w_{i}(y_{i} - \hat{y_{i}})^{2}}{\sum w_{i}y_{i}^{2}},
$$

- $y$ 是响应变量（responder_6）的真实值向量
- $\hat{y}$ 是预测值向量
- $w$ 是样本权重向量

样本加权零均值 ${\rm R^2}$ 衡量的是模型预测值与实际值之间的相关程度，${\rm R^2}$ 分数越接近 1，表示模型的预测越准确；负的 ${\rm R^2}$ 表示模型的预测效果比随机猜测还差。

### 数据

79 个特征和 9 个响应变量，目标是预测响应变量中的一个 responder_6，预测未来最多六个月的数据。

#### 训练集：

92 columns

- `date_id` 和 `time_id`：整数值，按顺序排序，为数据提供时间结构，每个 `time_id` 之间的实际时间间隔可能不同。
- `symbol_id`：用于标识的一个独特的金融工具。
- `weight`：计算评分函数的权重。
- `feature_{00-78}`：匿名化的市场数据。
- `responder_{0-8}`：匿名化的响应变量，范围 [-5, 5]，`responder_6` 是需要预测的。

#### 测试集

模拟测试集，来自单个 `date_id` 和 `time_id` 对的数据。不直接使用测试集，而是用 api 评估。

- `is_scored`：表示这一行是否包含在评估指标计算中。

#### Lags

- `responder_{0...8}` 的值延迟一个 `date_id`。评估 API 在`date_id` 的第一个 `time_id` 提供该日期 ID 的所有延迟响应变量。换句话说，前一天的所有响应变量将在后续日期的第一个时间步提供。
- `sample_submission.csv`：模型应预测的格式。
- `features.csv`与：匿名化特征相关的元数据
- `responders.csv`：与匿名化响应响应相关的元数据