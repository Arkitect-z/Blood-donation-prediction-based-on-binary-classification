# Blood-donation-prediction-based-on-two-classifications
基于二分类的献血预测; 二分类实例; 献血预测; 序贯模型

To use this program, you can run "classification.py" to train a model and predict by running "predict.py".
使用本程序时，通过"classification.py"来训练模型，通过"predict.py"来预测献血可能性。

本实验的数据是2008年采集自台湾新竹市输血服务中心的捐助者数据库。
本数据集为一个二分类问题，共有748条数据，每条数据有4个决策属性以及1个分类标签。
决策属性：
  Recency：自从上次献血以来经过的时间
  Frequency：捐赠总数
  Monetary：捐赠血液总量（单位：cc）
  Time：自从首次捐赠以来的时间
分类标签
  是否捐赠血液：0（是）否（1）
意义：
  通过之前的获取到的数据信息来建立一个预测模型，能够预测这些捐赠者在下个月是否来捐赠血液，从而提前进行血液储备调度，防止出现血库出现长期滞留或者短缺的现象。
