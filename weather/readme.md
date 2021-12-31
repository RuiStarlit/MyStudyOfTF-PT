##### Requirements：
* Python 3.8
* numpy
* pytorch
* matplotlib
* tqdm
* sklearn

##### 操作
* Train 实例化模型类

* Train._build_model 初始化模型  
    opt='xavier'时，使用xavier方法初始化所有Linear层参数
    可选 transformer, LSTM, attenLSTM

* Train._selct_criterion  选择损失函数  
    使用MSE或者Huber损失函数,RMSPE

* Train._selct_optim  选择优化器  
    adam或者sgd

* Train._selct_scheduler 选择学习率更新策略  
    'plateau'为固定学习率，经过patienc次训练loss不下降时调整学习率  
    'noam'为使用Attention is All You Need中提出的Noam策略，先从0线性提升学习率到最大值，再进行指数衰减

* Train.train  训练模型  
    epochs,train_idx, valid_idx, patience（earlystop的耐心参数）

* Train.load()  
    从checkpoint加载模型
* Train.save()  
    保存模型
* Train.reset_dataset  
    重新设置数据集

* Train.test   
    测试在指定index数据上的结果

* Train.PlotLoss()  
    对训练过程中的loss进行画图

* Train.show_plot()  
    对单点进行预测并画图

* class MyDateset()   
    内存足够的情况下将数据读入内存比用DataLoader要快  

* clss NoamOpt()  
    实现Noam策略