import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
from src.models.matrix_factorization import MatrixFactorization


class CombinedRecommender:
    '''
    Комбинированная модель с матричной факторизацией и K-средними.
    '''
    def __init__(self, n_users, n_movies, n_factors=50, n_clusters=5):
        '''
        Инициализация модели.
        :param n_users: Количество пользователей.
        :param n_movies: Количество фильмов.
        :param n_factors: Размерность скрытых факторов (эмбеддингов).
        :param n_clusters: Количество кластеров для K-средних.
        '''
        self.mf_model = MatrixFactorization(n_users, n_movies, n_factors)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.n_factors = n_factors
        self.n_clusters = n_clusters

    def train_mf(self, train_loader, test_loader, n_epochs=10):
        '''
        Обучение модели матричной факторизации
        :param train_loader: DataLoader для обучающих данных.
        :param test_loader: DataLoader для тестовых данных.
        :param n_epochs: Количество эпох обучения.
        :return: Списки потерь на обучении и тесте.
        '''
        criterion = nn.MSELoss()  # Функция потерь помогает определить,
        # насколько хорошо модель обучается на данных. Чем меньше значение
        # функции потерь, тем лучше модель соответствует данным.
        optimizer = optim.Adam(self.mf_model.parameters(), lr=0.01)

        train_losses, test_losses = [], []

        for epoch in range(n_epochs):
            self.mf_model.train()
            train_loss = self._run_epoch(train_loader, criterion, optimizer)
            train_losses.append(train_loss)

            self.mf_model.eval()
            test_loss = self._run_epoch(test_loader, criterion)
            test_losses.append(test_loss)

            print(f'Epoch {epoch+1}/{n_epochs}')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Test Loss: {test_loss:.4f}')

        return train_losses, test_losses

    def _run_epoch(self, loader, criterion, optimizer=None):
        '''
        Выполняет одну эпоху обучения или оценки.
        :param loader: DataLoader для данных.
        :param criterion: Функция потерь.
        :param optimizer: Оптимизатор (если None, то только оценка).
        :return: Средняя потеря за эпоху.
        '''
        total_loss = 0
        for users, movies, ratings in loader:
            if optimizer:
                optimizer.zero_grad()
            predictions = self.mf_model(users, movies)
            loss = criterion(predictions, ratings)
            if optimizer:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def get_combined_embeddings(self, users, movies):
        '''Объединяет эмбеддинги пользователей и фильмов.
        :param users: Тензор с ID пользователей.
        :param movies: Тензор с ID фильмов.
        :return: Объединённые эмбеддинги в виде массива numpy.'''
        self.mf_model.eval()
        with torch.no_grad():
            user_emb, movie_emb = self.mf_model.get_embeddings(users, movies)
            combined_emb = torch.cat([user_emb, movie_emb], dim=1)
            return combined_emb.numpy() 

    def train_kmeans(self, train_loader):
        '''
        Обучает K-средние на объединённых эмбеддингах.
        :param train_loader: DataLoader для обучающих данных.
        '''
        all_embeddings = []
        all_ratings = []

        self.mf_model.eval()
        with torch.no_grad():
            for users, movies, ratings in train_loader:
                combined_emb = self.get_combined_embeddings(users, movies)
                all_embeddings.append(combined_emb)
                all_ratings.extend(ratings.numpy())

        all_embeddings = np.vstack(all_embeddings)
        all_ratings = np.array(all_ratings)

        self.kmeans.fit(all_embeddings)  # Результатом  является обученная
        # модель K-means, которая может быть использована для предсказания
        # кластеров для новых данных.
        self.cluster_ratings = {}

        clusters = self.kmeans.predict(all_embeddings)
        for i in range(self.n_clusters):
            self.cluster_ratings[i] = np.mean(all_ratings[clusters == i])
            # Вычисляет средний рейтинг для этого кластера

    def predict(self, users, movies):
        '''
        Прогнозирование с использованием комбинированной модели.
        :param users: Тензор с ID пользователей.
        :param movies: Тензор с ID фильмов.
        :return: Прогнозируемые рейтинги.
        '''
        combined_emb = self.get_combined_embeddings(users, movies)
        clusters = self.kmeans.predict(combined_emb)  # Обученная k-means для
        # предсказания кластеров этих вложений.

        mf_predictions = self.mf_model(users, movies).numpy()

        cluster_predictions = np.array([self.cluster_ratings[c] for c in clusters])

        final_predictions = 0.7 * mf_predictions + 0.3 * cluster_predictions
        return final_predictions
