import pandas as pd
import torch
from torch.utils.data import Dataset


def load_data():
    """
    Загружает данные о фильмах и рейтингах из CSV-файлов.

    Returns:
        movies_df (pd.DataFrame): DataFrame с информацией о фильмах.
        ratings_df (pd.DataFrame): DataFrame с рейтингами пользователей.
    """
    movies_df = pd.read_csv('data/ml-latest-small/movies.csv')
    ratings_df = pd.read_csv('data/ml-latest-small/ratings.csv')
    return movies_df, ratings_df


def create_mappings(ratings_df):
    """
    Создает словари для сопоставления userId и movieId с индексами.

    Args:
        ratings_df (pd.DataFrame): DataFrame с рейтингами пользователей.

    Returns:
        user_to_idx (dict): Словарь для сопоставления userId с индексами.
        movie_to_idx (dict): Словарь для сопоставления movieId с индексами.
    """
    user_ids = ratings_df['userId'].unique()
    movie_ids = ratings_df['movieId'].unique()
    user_to_idx = {user: idx for idx, user in enumerate(user_ids)}
    movie_to_idx = {movie: idx for idx, movie in enumerate(movie_ids)}
    return user_to_idx, movie_to_idx


class MovieDataset(Dataset):
    """
    Кастомный Dataset для работы с рейтингами фильмов.

    Этот класс преобразует данные о рейтингах в тензоры PyTorch, что позволяет
    использовать их для обучения моделей. Матричная факторизация используется
    для уменьшения объема памяти, занимаемого данными.

    Args:
        ratings_df (pd.DataFrame): DataFrame с рейтингами пользователей.
        user_to_idx (dict): Словарь для сопоставления userId с индексами.
        movie_to_idx (dict): Словарь для сопоставления movieId с индексами.
    """

    def __init__(self, ratings_df, user_to_idx, movie_to_idx):
        """
        Инициализирует Dataset.

        Args:
            ratings_df (pd.DataFrame): DataFrame с рейтингами пользователей.
            user_to_idx (dict): Словарь для сопоставления userId с индексами.
            movie_to_idx (dict): Словарь для сопоставления movieId с индексами.
        """
        self.users = torch.tensor([user_to_idx[user] for user in ratings_df['userId']], dtype=torch.long)
        self.movies = torch.tensor([movie_to_idx[movie] for movie in ratings_df['movieId']], dtype=torch.long)
        self.ratings = torch.tensor(ratings_df['rating'].values, dtype=torch.float)

    def __len__(self):
        """
        Возвращает количество элементов в Dataset.

        Returns:
            int: Количество рейтингов в Dataset.
        """
        return len(self.ratings)

    def __getitem__(self, idx):
        """
        Возвращает элемент Dataset по индексу.

        Args:
            idx (int): Индекс элемента.

        Returns:
            tuple: Кортеж из трех элементов: userId, movieId и рейтинг.
        """
        return self.users[idx], self.movies[idx], self.ratings[idx]