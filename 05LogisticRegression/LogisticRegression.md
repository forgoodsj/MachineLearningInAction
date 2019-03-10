* ### Logistic Regression 中文译名 逻辑回归


* ### 优点：
> 计算代价不高，易于理解和实现

* ### 缺点：
> 容易欠拟合，分类精度可能不高

* ### 适用数据范围：
> 标称型和数值型

* ### 工作原理：
> 我们用一条线对这些点进行拟合，这个拟合过程就称为回归
>
> 逻辑回归的主要思想：根据现有数据对分类边界线简历回归公式，并以此进行分类。


* ### 操作步骤:
>
>（1）收集数据：用任意方法收集数据；
>
>（2）准备数据：由于需要进行距离计算，因此要求数据类型为数值型。另外结构化数据格式最佳；
>
>（3）分析数据：用任意方法对数据进行分析；
>
>（4）训练算法：目的是找到最佳的分类回归系数；
>
>（5）测试算法：测试分类。
>
>（6）使用算法：首先，输入一些数据，将其转换成对应的结构化数值；接着基于训练好的回归系数就可以对这些数值进行回归计算，判定它属于哪个类别，在这之后，我们可以在输出的类别上做一些分析工作。

* ### 逻辑回归和sigmoid函数分类:
> 能接受所有的输入然后预测出类别。
>
> Sigmoid函数  σ(z) = 1/(1+e^(-z))
>
> 当x=0时 函数值为0.5，随着x增大，函数值逼近1，反之逼近0
>
> 为了实现逻辑回归分类器，我们在每个特征上都乘以一个回归系数，然后所有的结果值相加，将这个总和带入Sigmoid函数，进而得到一个范围在0~1的数值，任何大于0.5的数值被分入1类，反之分入0类。实现分类。


* ### 给予最优化方法的最佳回归系数确定:
> z = w0x0+w1x1+w2x2......+wnxn   这个z曲线就是之后用来分割战场，将数据分为两个部分的
>
> 采用向量的写法 z = (W^T) X
>
> 向量w也就是我们要找的最佳系数

* ### 梯度上升法:
> 思想是：要找到某函数的最大值，最好的方法是沿着该函数的梯度方向探寻
>
> 到达每个点后都会重新估计移动的方向，如此迭代循环，直到满足停止条件。移动量为α
>
> 公式  w := w + α（f(w)对w求导）

* ### 训练算法:使用梯度上升找到最佳参数
> 伪代码如下：
>
> 每个回归系数初始化为1
>
> 重复R次
>
>     计算整个数据集的梯度
>
>     使用alpha * gradient更新回归系数的向量
>
> 返回回归系数

* ### 随机梯度上升法:
> 梯度上升算法在每次更新回归系数时都需要遍历整个数据集
>
> 随机梯度上升 一次仅用一个样本点来更新回归系数，这样可以在新的样本到来时对分类器进行增量式更新。 是一个在线学习算法
>
> 伪代码如下：
>
> 每个回归系数初始化为1
>
> 对数据集中每个样本
>
>     计算该样本系数
>
>     使用alpha * gradient更新回归系数的向量
>
> 返回回归系数


