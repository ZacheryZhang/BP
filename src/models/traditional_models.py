"""
传统机器学习模型
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')


class TraditionalModelManager:
    """传统机器学习模型管理器"""
    
    def __init__(self, config):
        """
        初始化模型管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.models = {}
        self.best_models = {}
        self.results = {}
        
        # 初始化模型
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化所有模型"""
        
        # 随机森林
        self.models['random_forest'] = {
            'model': MultiOutputRegressor(RandomForestRegressor(random_state=42)),
            'params': {
                'estimator__n_estimators': [100, 200, 300],
                'estimator__max_depth': [10, 20, None],
                'estimator__min_samples_split': [2, 5, 10],
                'estimator__min_samples_leaf': [1, 2, 4]
            }
        }
        
        # 梯度提升
        self.models['gradient_boosting'] = {
            'model': MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),
            'params': {
                'estimator__n_estimators': [100, 200],
                'estimator__learning_rate': [0.01, 0.1, 0.2],
                'estimator__max_depth': [3, 5, 7],
                'estimator__subsample': [0.8, 1.0]
            }
        }
        
        # 支持向量回归
        self.models['svr'] = {
            'model': MultiOutputRegressor(SVR()),
            'params': {
                'estimator__C': [0.1, 1, 10, 100],
                'estimator__gamma': ['scale', 'auto', 0.001, 0.01],
                'estimator__kernel': ['rbf', 'linear', 'poly']
            }
        }
        
        # 岭回归
        self.models['ridge'] = {
            'model': MultiOutputRegressor(Ridge()),
            'params': {
                'estimator__alpha': [0.1, 1, 10, 100, 1000]
            }
        }
        
        # Lasso回归
        self.models['lasso'] = {
            'model': MultiOutputRegressor(Lasso()),
            'params': {
                'estimator__alpha': [0.01, 0.1, 1, 10, 100]
            }
        }
        
        # 弹性网络
        self.models['elastic_net'] = {
            'model': MultiOutputRegressor(ElasticNet()),
            'params': {
                'estimator__alpha': [0.01, 0.1, 1, 10],
                'estimator__l1_ratio': [0.1, 0.5, 0.7, 0.9]
            }
        }
        
        # K近邻
        self.models['knn'] = {
            'model': MultiOutputRegressor(KNeighborsRegressor()),
            'params': {
                'estimator__n_neighbors': [3, 5, 7, 9, 11],
                'estimator__weights': ['uniform', 'distance'],
                'estimator__metric': ['euclidean', 'manhattan']
            }
        }
        
        # 决策树
        self.models['decision_tree'] = {
            'model': MultiOutputRegressor(DecisionTreeRegressor(random_state=42)),
            'params': {
                'estimator__max_depth': [5, 10, 20, None],
                'estimator__min_samples_split': [2, 5, 10],
                'estimator__min_samples_leaf': [1, 2, 4]
            }
        }
    
    def train_all_models(self, X_train, y_train, cv=5):
        """
        训练所有模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            cv: 交叉验证折数
        """
        print("开始训练传统机器学习模型...")
        
        for model_name, model_info in self.models.items():
            print(f"\n训练 {model_name}...")
            
            try:
                # 网格搜索
                grid_search = GridSearchCV(
                    model_info['model'],
                    model_info['params'],
                    cv=cv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )
                
                # 训练
                grid_search.fit(X_train, y_train)
                
                # 保存最佳模型
                self.best_models[model_name] = grid_search.best_estimator_
                
                # 交叉验证评分
                cv_scores = cross_val_score(
                    grid_search.best_estimator_,
                    X_train, y_train,
                    cv=cv,
                    scoring='neg_mean_squared_error'
                )
                
                # 保存结果
                self.results[model_name] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'cv_scores': cv_scores,
                    'cv_mean': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores)
                }
                
                print(f"{model_name} 训练完成")
                print(f"最佳参数: {grid_search.best_params_}")
                print(f"交叉验证得分: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
                
            except Exception as e:
                print(f"{model_name} 训练失败: {e}")
                self.results[model_name] = {'error': str(e)}
    
    def evaluate_all_models(self, X_test, y_test):
        """
        评估所有模型
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            dict: 评估结果
        """
        evaluation_results = {}
        
        for model_name, model in self.best_models.items():
            try:
                # 预测
                y_pred = model.predict(X_test)
                
                # 计算指标
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                # 分别计算收缩压和舒张压的指标
                sbp_mse = mean_squared_error(y_test[:, 0], y_pred[:, 0])
                sbp_mae = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
                sbp_r2 = r2_score(y_test[:, 0], y_pred[:, 0])
                
                dbp_mse = mean_squared_error(y_test[:, 1], y_pred[:, 1])
                dbp_mae = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
                dbp_r2 = r2_score(y_test[:, 1], y_pred[:, 1])
                
                evaluation_results[model_name] = {
                    'overall': {
                        'mse': mse,
                        'mae': mae,
                        'rmse': rmse,
                        'r2': r2
                    },
                    'sbp': {
                        'mse': sbp_mse,
                        'mae': sbp_mae,
                        'r2': sbp_r2
                    },
                    'dbp': {
                        'mse': dbp_mse,
                        'mae': dbp_mae,
                        'r2': dbp_r2
                    },
                    'predictions': y_pred
                }
                
            except Exception as e:
                evaluation_results[model_name] = {'error': str(e)}
        
        return evaluation_results
    
    def get_feature_importance(self, model_name, feature_names=None):
        """
        获取特征重要性
        
        Args:
            model_name: 模型名称
            feature_names: 特征名称列表
            
        Returns:
            dict: 特征重要性
        """
        if model_name not in self.best_models:
            return None
        
        model = self.best_models[model_name]
        
        # 获取基础估计器
        if hasattr(model, 'estimators_'):
            base_estimator = model.estimators_[0]  # 取第一个输出的估计器
        else:
            return None
        
        # 检查是否有feature_importances_属性
        if hasattr(base_estimator, 'feature_importances_'):
            importances = base_estimator.feature_importances_
            
            if feature_names is not None:
                importance_dict = dict(zip(feature_names, importances))
                # 按重要性排序
                sorted_importance = sorted(importance_dict.items(), 
                                         key=lambda x: x[1], reverse=True)
                return dict(sorted_importance)
            else:
                return importances
        else:
            return None
    
    def save_models(self, save_dir):
        """
        保存所有训练好的模型
        
        Args:
            save_dir: 保存目录
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.best_models.items():
            model_path = os.path.join(save_dir, f"{model_name}_model.pkl")
            joblib.dump(model, model_path)
            print(f"{model_name} 模型已保存到 {model_path}")
        
        # 保存结果
        results_path = os.path.join(save_dir, "training_results.pkl")
        joblib.dump(self.results, results_path)
        print(f"训练结果已保存到 {results_path}")
    
    def load_models(self, save_dir):
        """
        加载保存的模型
        
        Args:
            save_dir: 保存目录
        """
        import os
        
        for model_name in self.models.keys():
            model_path = os.path.join(save_dir, f"{model_name}_model.pkl")
            if os.path.exists(model_path):
                self.best_models[model_name] = joblib.load(model_path)
                print(f"{model_name} 模型已从 {model_path} 加载")
        
        # 加载结果
        results_path = os.path.join(save_dir, "training_results.pkl")
        if os.path.exists(results_path):
            self.results = joblib.load(results_path)
            print(f"训练结果已从 {results_path} 加载")
    
    def get_best_model(self, metric='r2'):
        """
        获取最佳模型
        
        Args:
            metric: 评估指标
            
        Returns:
            tuple: (模型名称, 模型对象)
        """
        if not self.results:
            return None, None
        
        best_score = float('-inf') if metric in ['r2'] else float('inf')
        best_model_name = None
        
        for model_name, result in self.results.items():
            if 'error' in result:
                continue
            
            if metric == 'r2':
                score = -result['best_score']  # GridSearchCV使用负MSE，转换为正值
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
            elif metric == 'mse':
                score = -result['best_score']  # 负MSE转正
                if score < best_score:
                    best_score = score
                    best_model_name = model_name
        
        if best_model_name:
            return best_model_name, self.best_models[best_model_name]
        else:
            return None, None
    
    def compare_models(self, evaluation_results):
        """
        比较模型性能
        
        Args:
            evaluation_results: 评估结果
            
        Returns:
            pd.DataFrame: 比较结果表格
        """
        comparison_data = []
        
        for model_name, results in evaluation_results.items():
            if 'error' in results:
                continue
            
            row = {
                'Model': model_name,
                'Overall_MAE': results['overall']['mae'],
                'Overall_RMSE': results['overall']['rmse'],
                'Overall_R2': results['overall']['r2'],
                'SBP_MAE': results['sbp']['mae'],
                'SBP_R2': results['sbp']['r2'],
                'DBP_MAE': results['dbp']['mae'],
                'DBP_R2': results['dbp']['r2']
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # 按Overall R2排序
        if not df.empty:
            df = df.sort_values('Overall_R2', ascending=False)
        
        return df


class EnsembleModel:
    """集成模型"""
    
    def __init__(self, models_dict, method='voting'):
        """
        初始化集成模型
        
        Args:
            models_dict: 模型字典
            method: 集成方法 ('voting', 'stacking')
        """
        self.models = models_dict
        self.method = method
        self.weights = None
        self.meta_model = None
        
        if method == 'stacking':
            from sklearn.linear_model import LinearRegression
            self.meta_model = MultiOutputRegressor(LinearRegression())
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        训练集成模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
        """
        if self.method == 'voting':
            # 基于验证集性能计算权重
            if X_val is not None and y_val is not None:
                weights = []
                for model_name, model in self.models.items():
                    y_pred = model.predict(X_val)
                    r2 = r2_score(y_val, y_pred)
                    weights.append(max(0, r2))  # 确保权重非负
                
                # 归一化权重
                total_weight = sum(weights)
                if total_weight > 0:
                    self.weights = [w / total_weight for w in weights]
                else:
                    self.weights = [1.0 / len(self.models)] * len(self.models)
            else:
                # 等权重
                self.weights = [1.0 / len(self.models)] * len(self.models)
        
        elif self.method == 'stacking':
            # 生成元特征
            meta_features = []
            for model_name, model in self.models.items():
                y_pred = model.predict(X_train)
                meta_features.append(y_pred)
            
            meta_X = np.hstack(meta_features)
            self.meta_model.fit(meta_X, y_train)
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 输入特征
            
        Returns:
            np.ndarray: 预测结果
        """
        if self.method == 'voting':
            predictions = []
            for i, (model_name, model) in enumerate(self.models.items()):
                y_pred = model.predict(X)
                predictions.append(y_pred * self.weights[i])
            
            return np.sum(predictions, axis=0)
        
        elif self.method == 'stacking':
            # 生成元特征
            meta_features = []
            for model_name, model in self.models.items():
                y_pred = model.predict(X)
                meta_features.append(y_pred)
            
            meta_X = np.hstack(meta_features)
            return self.meta_model.predict(meta_X)


def create_ensemble_from_manager(model_manager, top_k=3, method='voting'):
    """
    从模型管理器创建集成模型
    
    Args:
        model_manager: 模型管理器
        top_k: 选择前k个最佳模型
        method: 集成方法
        
    Returns:
        EnsembleModel: 集成模型
    """
    # 根据R2分数选择top-k模型
    model_scores = []
    for model_name, result in model_manager.results.items():
        if 'error' not in result and model_name in model_manager.best_models:
            score = -result['best_score']  # 转换为正的R2分数
            model_scores.append((model_name, score))
    
    # 排序并选择top-k
    model_scores.sort(key=lambda x: x[1], reverse=True)
    top_models = model_scores[:top_k]
    
    # 创建模型字典
    selected_models = {}
    for model_name, _ in top_models:
        selected_models[model_name] = model_manager.best_models[model_name]
    
    return EnsembleModel(selected_models, method=method)
