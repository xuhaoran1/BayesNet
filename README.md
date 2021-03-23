# BayesNet
该项目基于csci3202，讨论了两种使用python手写静态贝叶斯网络的办法，一种是迭代法，一种是递归法，迭代法基于多条件多结果的条件概率的枚举求法，迭代法基于边缘概率进行分类求取，其中，迭代法存在bug(bug的本质是浮点数计算的精度问题)，递归法由于分情况讨论，精度其实更高
### 构建静态贝叶斯网络https://github.com/xuhaoran1/BayesNet

#### 基于规则的减枝的策略

**边缘概率**

如果是无父节点的节点，直接得边缘概率，如果是有父节点的节点，递归得到父亲节点

**条件概率**

1. 看条件有无自己，如果有自己，根据正反的情况返回结果1或者0
2. 除去无关的相互独立节点，因为如果A，B无关，P(A|B) = P(A)
3. 此时如果无条件(条件被剔除完全)，进入求取边缘概率
4. 如果有条件，则此时为P(A|子节点，父节点)的情况

此时，以子节点向上迭代，求P(子节点,A,父节点)和P(子节点,A',父亲节点)的概率，其中A和A‘的取值不同，如果是二值，是相反，如果是多值，则为不同的取值，最后的值为1

#### 基于迭代的枚举方法

使用基于枚举的迭代方法，就非常简单，首先把两种结果加入条件的已知集合K中，而后从父节点到子节点，不断枚举，如果在K中，直接相乘，不在K中，并加入K中并将概率相加。
