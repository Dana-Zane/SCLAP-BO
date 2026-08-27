优化算法库是基于optuna框架下建立的
目前optuna内部部署了9种通用优化算法：
RandomSampler	随机搜索	baseline、初始对比
*TPESampler	TPE，Tree-structured Parzen Estimator	Optuna 默认常用算法，适合大多数黑盒优化
GPSampler	Gaussian Process Bayesian Optimization	高成本、低维连续优化
*CmaEsSampler	CMA-ES 进化策略	连续参数优化、非凸黑盒函数
*NSGAIISampler	NSGA-II 多目标遗传算法	多目标优化
NSGAIIISampler	NSGA-III 多目标遗传算法	更多目标数的多目标优化
GridSampler	网格搜索	小规模离散搜索空间
QMCSampler	Quasi Monte Carlo 低差异序列	空间覆盖更均匀的采样
BoTorchSampler	基于 BoTorch 的贝叶斯优化	高级贝叶斯优化，通常需额外依赖
BruteForceSampler	穷举搜索	有限离散空间穷举

optuna库也提供一个算法选择器，它会根据当前问题的特征，自动选择合适的 Optuna 内置 sampler 来跑，而不是自己提出一套独立的优化理论。
AutoSampler	自动选择采样器	不确定用哪个算法时


补充的算法：BO,morbo,turbo,TSS-BO，SP-BO(复现)，强化学习框架
安装库pip install optuna

但有些 sampler / 功能需要额外依赖：
功能	额外依赖
CmaEsSampler	cmaes(pip install cmaes)
GPSampler	通常需要 scipy，部分版本还依赖其他数值库
BoTorchSampler	torch、botorch、gpytorch
Optuna Dashboard	optuna-dashboard
OptunaHub 扩展算法，如 PSO	optunahub

morbo算法需要创建一个新的环境，
python=3.8
torch=2.4.1,
botorch=0.6.6,
pip install -e 你当前的目录/morbo,
配置好库和问题的objective（函数返回张量或者向量都可以）,照着test_morbo.py的形式运行即可（方便调参）

TSS-BO算法需要创建一个新的环境，
botorch>=0.8.1和对应的gpytorch库，建议：
python=3.10
pytorch=1.13.1
botorch=0.8.5
gpytorch=1.10

turbo算法
python=3.7
numpy==1.17.3
torch==1.3.0
gpytorch==0.3.6

更新中...(HEBO,DNN-OPT,MARS-NEI)

多目标算法：NSGAII,morbo,MoTPE
单目标算法：CMA-ES,Turbo,TSS-BO,遗传算法（GA,DE,PSO），TPE

标准 BO 示例入口：test_bo.py，使用 Optuna GPSampler，形式与 test_tpe.py、test_cma.py 保持一致。


优化库也兼容强化学习算法用做实验，配置环境建议如下：
gymnasium==1.3.0
torch==2.12.0
numpy==2.4.6
scipy==1.17.1
scikit-learn==1.8.0
