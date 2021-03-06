# Blood-donation-prediction-based-on-Binary-classifications
标签：

    基于二分类的献血预测;  不平衡数据集处理;  献血预测;  序贯模型


To use this program, you can run "classification.py" to train a new model and predict immediately by running "predict.py".

使用本程序时，通过"classification.py"来训练新模型，或通过"predict.py"立即预测献血可能性。


    "model"文件夹是训练好的示例模型，拥有85%左右的准确率，epoch recall曲线在Train_Result.png中。


    本实验的数据是2008年采集自中国台湾省新竹市输血服务中心的捐助者数据库，数据仅供参考。


本数据集为一个二分类问题，共有748条数据，每条数据有4个决策属性以及1个分类标签。

决策属性：

    Recency：自从上次献血以来经过的时间
  
    Frequency：捐赠总数
  
    Monetary：捐赠血液总量（单位：cc）
  
    Time：自从首次捐赠以来的时间
  
分类标签：

    是否捐赠血液：0（是）否（1）

训练结果：
    ![train_result.png](https://github.com/Arkitect-z/Blood-donation-prediction-based-on-two-class-classification/blob/main/Train_Result.png)

意义：

    通过之前的获取到的数据信息来建立一个预测模型，能够预测这些捐赠者在下个月是否来捐赠血液，从而提前进行血液储备调度，防止出现血库出现长期滞留或者短缺的现象。


通过进行数据集平衡、数据标准化、序列化模型及预测，平衡了捐赠者是否会来捐赠的血液数据，消除了血液数据值过大产生的权重过高问题，最后对神经网络的sigmoid函数输出概率值进行加权决策融合实现捐赠者捐赠预测的最终分类。并最终取得了85%的准确率。实验结果表明，与KNN分类方法以及其他较新方法相比，提出的方法具有更优的分类性能。

相对于被动接收捐赠者捐赠血液，被动增加血液储备调度，提出基于序贯模型的捐赠者数据分类方法，通过之前的获取到的数据信息来建立一个预测模型，能够预测这些捐赠者在下个月是否来捐赠血液，从而提前进行血液储备调度，防止出现血库出现长期滞留或者短缺的现象。
