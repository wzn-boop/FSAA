"""
基于FSAA的动态因果发现框架
集成特征选择和因果发现
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

from .fsaa_feature_selection import FSAASelector
from .granger_causality import GrangerCausality
from .sem_model import SEMModel


class FSAACausalDiscovery:
    """
    FSAA因果发现框架
    集成特征增强、特征选择、因果发现和异常检测
    """

    def __init__(self,
                 fsaa_params: Optional[Dict] = None,
                 granger_params: Optional[Dict] = None,
                 sem_params: Optional[Dict] = None):
        """
        初始化FSAA因果发现框架

        参数:
            fsaa_params: FSAA选择器参数
            granger_params: 格兰杰因果检验参数
            sem_params: 结构方程模型参数
        """
        # 默认参数
        self.fsaa_params = fsaa_params or {
            'threshold': 0.1,
            'alpha': 0.05,
            'max_lag': 3
        }

        self.granger_params = granger_params or {
            'maxlag': 5,
            'verbose': False
        }

        self.sem_params = sem_params or {
            'method': 'lasso',
            'alpha': 0.1
        }

        # 初始化组件
        self.fsaa_selector = FSAASelector(**self.fsaa_params)
        self.granger_test = GrangerCausality(**self.granger_params)
        self.sem_model = SEMModel(**self.sem_params)

        # 存储结果
        self.selected_features_ = None
        self.normal_causal_graph_ = None
        self.anomaly_causal_graph_ = None
        self.causal_changes_ = None

    def fit_normal(self, normal_data: pd.DataFrame,
                   target_col: Optional[str] = None):
        """
        在正常数据上训练模型

        参数:
            normal_data: 正常时段数据
            target_col: 目标变量列名（可选）
        """
        print("=" * 60)
        print("阶段1: 正常时段因果发现")
        print("=" * 60)

        # 1. FSAA特征选择
        print("\n1. 执行FSAA特征选择...")
        if target_col:
            y = normal_data[target_col]
            X = normal_data.drop(columns=[target_col])
            selected_X = self.fsaa_selector.fit_transform(X, y)
        else:
            selected_X = self.fsaa_selector.fit_transform(normal_data)

        self.selected_features_ = selected_X.columns.tolist()
        print(f"选择的关键变量: {self.selected_features_}")

        # 2. 格兰杰因果检验
        print("\n2. 执行格兰杰因果检验...")
        granger_matrix = self.granger_test.fit(selected_X)

        # 3. 构建因果图
        print("\n3. 构建因果图...")
        self.normal_causal_graph_ = self._build_causal_graph(
            selected_X, granger_matrix
        )

        # 4. 结构方程模型优化
        print("\n4. 结构方程模型优化...")
        self.sem_model.fit(selected_X)
        sem_edges = self.sem_model.get_significant_edges(alpha=0.05)

        # 合并结果
        self._refine_causal_graph(self.normal_causal_graph_, sem_edges)

        print(f"\n正常时段因果图构建完成:")
        print(f"- 节点数: {self.normal_causal_graph_.number_of_nodes()}")
        print(f"- 边数: {self.normal_causal_graph_.number_of_edges()}")

        return self

    def detect_anomaly(self, anomaly_data: pd.DataFrame,
                       threshold: float = 2.0) -> Dict:
        """
        检测异常并分析因果变化

        参数:
            anomaly_data: 异常时段数据
            threshold: 变化阈值

        返回:
            异常分析结果
        """
        print("\n" + "=" * 60)
        print("阶段2: 异常检测与因果分析")
        print("=" * 60)

        if self.selected_features_ is None:
            raise ValueError("必须先调用fit_normal方法")

        # 1. 使用选择的特征
        selected_anomaly = anomaly_data[self.selected_features_]

        # 2. 构建异常因果图
        print("\n1. 构建异常时段因果图...")
        granger_matrix_anomaly = self.granger_test.fit(selected_anomaly)
        self.anomaly_causal_graph_ = self._build_causal_graph(
            selected_anomaly, granger_matrix_anomaly
        )

        # 3. 对比因果图变化
        print("\n2. 对比因果图变化...")
        self.causal_changes_ = self._compare_causal_graphs(
            self.normal_causal_graph_,
            self.anomaly_causal_graph_,
            threshold
        )

        # 4. 定位异常源
        print("\n3. 定位异常源...")
        anomaly_sources = self._locate_anomaly_sources(
            self.causal_changes_,
            selected_anomaly
        )

        return {
            'causal_changes': self.causal_changes_,
            'anomaly_sources': anomaly_sources,
            'anomaly_graph': self.anomaly_causal_graph_
        }

    def _build_causal_graph(self, data: pd.DataFrame,
                            granger_matrix: pd.DataFrame) -> nx.DiGraph:
        """构建因果有向图"""
        G = nx.DiGraph()

        # 添加节点
        for feature in data.columns:
            G.add_node(feature)

        # 添加边（基于格兰杰因果）
        n_features = len(data.columns)
        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    p_value = granger_matrix.iloc[i, j]
                    if p_value < 0.05:  # 显著性水平
                        G.add_edge(data.columns[i], data.columns[j],
                                   weight=1 - p_value, p_value=p_value)

        return G

    def _refine_causal_graph(self, graph: nx.DiGraph,
                             sem_edges: List[Tuple[str, str]]):
        """使用SEM结果优化因果图"""
        # 移除SEM中不显著的边
        edges_to_remove = []
        for u, v in graph.edges():
            if (u, v) not in sem_edges:
                edges_to_remove.append((u, v))

        graph.remove_edges_from(edges_to_remove)

        # 添加SEM中显著的边
        for u, v in sem_edges:
            if not graph.has_edge(u, v):
                graph.add_edge(u, v, weight=1.0, source='sem')

    def _compare_causal_graphs(self, G_normal: nx.DiGraph,
                               G_anomaly: nx.DiGraph,
                               threshold: float = 2.0) -> Dict:
        """对比两个因果图的结构差异"""
        changes = {
            'added_edges': [],
            'removed_edges': [],
            'strength_changes': [],
            'node_centrality_changes': {}
        }

        # 边变化
        edges_normal = set(G_normal.edges())
        edges_anomaly = set(G_anomaly.edges())

        changes['added_edges'] = list(edges_anomaly - edges_normal)
        changes['removed_edges'] = list(edges_normal - edges_anomaly)

        # 边强度变化
        common_edges = edges_normal & edges_anomaly
        for u, v in common_edges:
            w_normal = G_normal[u][v].get('weight', 0)
            w_anomaly = G_anomaly[u][v].get('weight', 0)
            if abs(w_anomaly - w_normal) > threshold * 0.1:
                changes['strength_changes'].append({
                    'edge': (u, v),
                    'normal_weight': w_normal,
                    'anomaly_weight': w_anomaly,
                    'change': w_anomaly - w_normal
                })

        # 节点中心性变化
        nodes = set(G_normal.nodes()) | set(G_anomaly.nodes())
        for node in nodes:
            if node in G_normal and node in G_anomaly:
                centrality_normal = self._calculate_node_centrality(G_normal, node)
                centrality_anomaly = self._calculate_node_centrality(G_anomaly, node)

                if abs(centrality_anomaly - centrality_normal) > threshold * 0.1:
                    changes['node_centrality_changes'][node] = {
                        'normal': centrality_normal,
                        'anomaly': centrality_anomaly,
                        'change': centrality_anomaly - centrality_normal
                    }

        return changes

    def _calculate_node_centrality(self, graph: nx.DiGraph, node: str) -> float:
        """计算节点中心性"""
        if graph.number_of_edges() == 0:
            return 0

        # 使用出入度作为中心性度量
        in_degree = graph.in_degree(node, weight='weight')
        out_degree = graph.out_degree(node, weight='weight')

        return (in_degree + out_degree) / (2 * graph.number_of_edges())

    def _locate_anomaly_sources(self, causal_changes: Dict,
                                anomaly_data: pd.DataFrame) -> List[Dict]:
        """定位异常源"""
        anomaly_sources = []

        # 分析节点中心性变化
        for node, changes in causal_changes['node_centrality_changes'].items():
            if changes['change'] > 0:  # 中心性增加
                # 检查该节点的统计特性
                node_data = anomaly_data[node]

                # 计算异常指标
                mean_val = np.mean(node_data)
                std_val = np.std(node_data)
                z_scores = np.abs((node_data - mean_val) / std_val)
                anomaly_count = np.sum(z_scores > 3)  # 3sigma异常

                if anomaly_count > len(node_data) * 0.1:  # 超过10%的点异常
                    anomaly_sources.append({
                        'node': node,
                        'type': 'centrality_increase',
                        'centrality_change': changes['change'],
                        'anomaly_rate': anomaly_count / len(node_data),
                        'description': f"节点'{node}'的中心性显著增加，且包含{anomaly_count}个异常点"
                    })

        # 分析新增的边
        for u, v in causal_changes['added_edges']:
            anomaly_sources.append({
                'node': f"{u}->{v}",
                'type': 'new_causal_link',
                'description': f"新增因果关系: {u} -> {v}"
            })

        # 分析移除的边
        for u, v in causal_changes['removed_edges']:
            anomaly_sources.append({
                'node': f"{u}->{v}",
                'type': 'broken_causal_link',
                'description': f"因果关系消失: {u} -> {v}"
            })

        # 按重要性排序
        anomaly_sources.sort(key=lambda x: abs(x.get('centrality_change', 0)),
                             reverse=True)

        return anomaly_sources

    def visualize_results(self, save_path: Optional[str] = None):
        """可视化结果"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(15, 10))

        # 1. 特征重要性
        plt.subplot(2, 2, 1)
        if self.fsaa_selector.feature_scores_:
            top_features = self.fsaa_selector.get_feature_importance().head(10)
            sns.barplot(x='importance', y='feature', data=top_features)
            plt.title('Top 10 Feature Importance (FSAA)')
            plt.xlabel('Importance Score')
            plt.ylabel('Feature')

        # 2. 正常因果图
        plt.subplot(2, 2, 2)
        if self.normal_causal_graph_:
            pos = nx.spring_layout(self.normal_causal_graph_)
            nx.draw(self.normal_causal_graph_, pos, with_labels=True,
                    node_color='lightblue', node_size=500,
                    edge_color='gray', arrowsize=20)
            plt.title('Normal Causal Graph')

        # 3. 异常因果图
        plt.subplot(2, 2, 3)
        if self.anomaly_causal_graph_:
            pos = nx.spring_layout(self.anomaly_causal_graph_)
            nx.draw(self.anomaly_causal_graph_, pos, with_labels=True,
                    node_color='lightcoral', node_size=500,
                    edge_color='gray', arrowsize=20)
            plt.title('Anomaly Causal Graph')

        # 4. 因果变化总结
        plt.subplot(2, 2, 4)
        if self.causal_changes_:
            changes_summary = {
                'Added Edges': len(self.causal_changes_['added_edges']),
                'Removed Edges': len(self.causal_changes_['removed_edges']),
                'Strength Changes': len(self.causal_changes_['strength_changes']),
                'Node Changes': len(self.causal_changes_['node_centrality_changes'])
            }
            plt.bar(changes_summary.keys(), changes_summary.values())
            plt.title('Causal Graph Changes')
            plt.xticks(rotation=45)
            plt.ylabel('Count')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"结果已保存至: {save_path}")

        plt.show()

    def generate_report(self) -> str:
        """生成分析报告"""
        report = []
        report.append("=" * 60)
        report.append("FSAA因果发现分析报告")
        report.append("=" * 60)

        report.append(f"\n一、特征选择结果")
        report.append(f"- 原始特征数: 待补充")
        report.append(f"- 选择特征数: {len(self.selected_features_)}")
        report.append(f"- 关键变量: {', '.join(self.selected_features_)}")

        report.append(f"\n二、因果图分析")
        if self.normal_causal_graph_:
            report.append(f"- 正常图: {self.normal_causal_graph_.number_of_nodes()}节点, "
                          f"{self.normal_causal_graph_.number_of_edges()}边")
        if self.anomaly_causal_graph_:
            report.append(f"- 异常图: {self.anomaly_causal_graph_.number_of_nodes()}节点, "
                          f"{self.anomaly_causal_graph_.number_of_edges()}边")

        report.append(f"\n三、因果变化检测")
        if self.causal_changes_:
            report.append(f"- 新增边: {len(self.causal_changes_['added_edges'])}")
            report.append(f"- 消失边: {len(self.causal_changes_['removed_edges'])}")
            report.append(f"- 强度变化: {len(self.causal_changes_['strength_changes'])}")
            report.append(f"- 节点中心性变化: {len(self.causal_changes_['node_centrality_changes'])}")

        report.append(f"\n四、异常源定位")
        if self.causal_changes_ and 'anomaly_sources' in self.causal_changes_:
            for i, source in enumerate(self.causal_changes_['anomaly_sources'][:5], 1):
                report.append(f"{i}. {source['description']}")

        report.append(f"\n" + "=" * 60)
        report.append("分析完成")
        report.append("=" * 60)

        return "\n".join(report)