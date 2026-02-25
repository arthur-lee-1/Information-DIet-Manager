
        return float(similarity)

    def calculate_edit_similarity(self,
                                 text1: str,
                                 text2: str,
                                 normalize: bool = True) -> float:
        """
        计算编辑距离相似度

        参数:
            text1: 文本1
            text2: 文本2
            normalize: 是否归一化到 [0, 1]

        返回:
            float: 相似度分数
        """
        if not text1 or not text2:
            return 0.0

        distance = self._calculate_edit_distance(str(text1), str(text2))

        if normalize:
            max_len = max(len(text1), len(text2))
            if max_len == 0:
                return 1.0
            return 1.0 - (distance / max_len)

        return float(distance)

    def find_similar_texts(self,
                          query: str,
                          top_k: int = 5,
                          threshold: float = 0.3) -> List[Tuple[int, float]]:
        """
        查找与查询文本最相似的文档

        参数:
            query: 查询文本
            top_k: 返回前 k 个结果
            threshold: 相似度阈值

        返回:
            List[Tuple[int, float]]: [(文档索引, 相似度分数), ...]
        """
        if self.tfidf_matrix is None:
            logger.error("请先调用 fit() 方法训练模型")
            raise ValueError("Model not fitted. Call fit() first.")

        if not query or pd.isna(query):
            logger.warning("查询文本为空")
            return []

        try:
            # 转换查询为 TF-IDF 向量
            query_vec = self.vectorizer.transform([str(query)])

            # 计算与所有文档的相似度
            similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

            # 过滤低于阈值的结果
            valid_indices = np.where(similarities >= threshold)[0]

            if len(valid_indices) == 0:
                logger.info("没有找到相似度超过阈值的文档")
                return []

            # 排序并返回 top_k
            sorted_indices = valid_indices[np.argsort(-similarities[valid_indices])]
            top_indices = sorted_indices[:top_k]

            results = [(int(idx), float(similarities[idx])) for idx in top_indices]

            logger.info(f"找到 {len(results)} 个相似文档")
            return results

        except Exception as e:
            logger.error(f"查找相似文本失败: {e}")
            return []

    def detect_duplicates(self,
                         texts: List[str],
                         threshold: float = 0.9) -> List[Tuple[int, int, float]]:
        """
        检测重复或高度相似的文本

        参数:
            texts: 文本列表
            threshold: 相似度阈值（默认 0.9）

        返回:
            List[Tuple[int, int, float]]: [(索引1, 索引2, 相似度), ...]
        """
        if not texts:
            logger.warning("输入文本列表为空")
            return []

        logger.info(f"开始检测重复文本，阈值: {threshold}")

        # 训练模型
        self.fit(texts)

        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(self.tfidf_matrix)

        duplicates = []
        n = len(texts)

        # 遍历上三角矩阵（避免重复比较）
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i][j] >= threshold:
                    duplicates.append((i, j, float(similarity_matrix[i][j])))

        logger.info(f"检测到 {len(duplicates)} 对重复文本")
        return duplicates

    def cluster_texts(self,
                     texts: List[str],
                     n_clusters: Optional[int] = None,
                     method: str = 'kmeans',
                     min_samples: int = 2) -> np.ndarray:
        """
        对文本进行聚类

        参数:
            texts: 文本列表
            n_clusters: 聚类数量（KMeans 必需）
            method: 聚类方法 ('kmeans' 或 'dbscan')
            min_samples: DBSCAN 最小样本数

        返回:
            np.ndarray: 聚类标签数组
        """
        if not texts:
            logger.error("输入文本列表为空")
            raise ValueError("texts cannot be empty")

        logger.info(f"开始文本聚类，方法: {method}")

        # 训练 TF-IDF
        self.fit(texts)

        if method == 'kmeans':
            if n_clusters is None:
                # 自动确定聚类数（启发式：sqrt(n/2)）
                n_clusters = max(2, int(np.sqrt(len(texts) / 2)))
                logger.info(f"自动确定聚类数: {n_clusters}")

            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(self.tfidf_matrix)

        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=min_samples, metric='cosine')
            labels = clusterer.fit_predict(self.tfidf_matrix.toarray())

        else:
            logger.error(f"不支持的聚类方法: {method}")
            raise ValueError(f"Unsupported clustering method: {method}")

        unique_labels = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"聚类完成，共 {unique_labels} 个簇")

        return labels

    def batch_calculate_similarity(self,
                                  df: pd.DataFrame,
                                  text_column: str = 'title',
                                  reference_column: Optional[str] = None) -> pd.DataFrame:
        """
        批量计算相似度

        参数:
            df: 输入 DataFrame
            text_column: 文本列名
            reference_column: 参考列名（如果为 None，则计算两两相似度）

        返回:
            pd.DataFrame: 添加了相似度列的 DataFrame
        """
        if text_column not in df.columns:
            logger.error(f"列 '{text_column}' 不存在")
            raise ValueError(f"Column '{text_column}' not found")

        if df.empty:
            logger.warning("输入 DataFrame 为空")
            return df

        logger.info(f"开始批量计算相似度，数据量: {len(df)}")

        result_df = df.copy()

        if reference_column:
            # 与参考列计算相似度
            if reference_column not in df.columns:
                logger.error(f"参考列 '{reference_column}' 不存在")
                raise ValueError(f"Reference column '{reference_column}' not found")

            similarities = []
            for idx, row in result_df.iterrows():
                text = row[text_column]
                ref_text = row[reference_column]
                sim = self.calculate_cosine_similarity(text, ref_text)
                similarities.append(sim)

            result_df['similarity'] = similarities

        else:
            # 计算与前一条记录的相似度
            texts = result_df[text_column].tolist()
            self.fit(texts)

            similarities = [0.0]  # 第一条记录相似度为 0

            for i in range(1, len(texts)):
                sim = self.calculate_cosine_similarity(texts[i-1], texts[i])
                similarities.append(sim)

            result_df['similarity_to_previous'] = similarities

        logger.info("批量相似度计算完成")
        return result_df

    # ==================== 模型持久化方法 ====================

    def save_model(self, path: str) -> None:
        """
        保存模型

        参数:
            path: 保存路径
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            model_data = {
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'documents': self.documents,
                'stopwords': self.stopwords
            }

            with open(path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"✅ 模型已保存到: {path}")

        except Exception as e:
            logger.error(f"保存模型失败: {e}")

    def load_model(self, path: str) -> bool:
        """
        加载模型

        参数:
            path: 模型路径

        返回:
            bool: 是否加载成功
        """
        if not Path(path).exists():
            logger.error(f"模型文件不存在: {path}")
            return False

        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)

            self.vectorizer = model_data['vectorizer']
            self.tfidf_matrix = model_data['tfidf_matrix']
            self.documents = model_data['documents']
            self.stopwords = model_data.get('stopwords', set())

            logger.info(f"✅ 模型已加载: {path}")
            return True

        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False

    # ==================== 统计分析方法 ====================

    def get_similarity_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        统计相似度分布

        参数:
            df: 包含相似度列的 DataFrame

        返回:
            Dict[str, Any]: 统计信息
        """
        similarity_cols = [col for col in df.columns if 'similarity' in col.lower()]

        if not similarity_cols:
            logger.error("DataFrame 中没有相似度列")
            return {}

        stats = {}

        for col in similarity_cols:
            col_stats = df[col].describe().to_dict()
            stats[col] = {
                'mean': round(col_stats.get('mean', 0), 4),
                'std': round(col_stats.get('std', 0), 4),
                'min': round(col_stats.get('min', 0), 4),
                'max': round(col_stats.get('max', 0), 4),
                'median': round(df[col].median(), 4)
            }

        logger.info("相似度统计完成")
        return stats


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 示例用法
    analyzer = SimilarityAnalyzer()

    # 测试数据
    texts = [
        "今天天气很好，适合出去玩",
        "今天天气不错，可以出门散步",
        "明天要下雨了，记得带伞",
        "Python 是一门很好的编程语言",
        "Python 编程语言非常强大"
    ]

    # 训练模型
    analyzer.fit(texts)

    # 测试相似度计算
    sim = analyzer.calculate_cosine_similarity(texts[0], texts[1])
    print(f"相似度: {sim:.4f}")

    # 查找相似文本
    results = analyzer.find_similar_texts("天气很好", top_k=3)
    print(f"相似文本: {results}")

    # 检测重复
    duplicates = analyzer.detect_duplicates(texts, threshold=0.7)
    print(f"重复文本对: {duplicates}")

    # 聚类
    labels = analyzer.cluster_texts(texts, n_clusters=2)
    print(f"聚类标签: {labels}")

    logger.info("测试完成")