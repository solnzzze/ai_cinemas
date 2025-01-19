import torch.nn as nn


class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=50):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)
        # Создание векторов смещения.
        self.user_biases = nn.Embedding(n_users, 1)
        self.movie_biases = nn.Embedding(n_movies, 1)

    def forward(self, user, movie):
        """
        Прямой проход модели.

        Вычисляет предсказанный рейтинг для заданных пользователя и фильма на основе
        скалярного произведения их эмбеддингов и учёта смещений.

        Args:
            user (torch.Tensor): Тензор с идентификаторами пользователей.
            movie (torch.Tensor): Тензор с идентификаторами фильмов.

        Returns:
            torch.Tensor: Тензор с предсказанными рейтингами.
    """
        user_embedding = self.user_factors(user)
        movie_embedding = self.movie_factors(movie)
        user_bias = self.user_biases(user)
        movie_bias = self.movie_biases(movie)

        prediction = (user_embedding *
                      movie_embedding).sum(dim=1, keepdim=True)
        # Cкалярное произведение, которое позволяет определить, насколько два
        # вектора хорошо соответствуют друг другу. Чем больше значение
        # скалярного произведения, тем более вероятно, что пользователь
        # оценит фильм высоко
        prediction = prediction + user_bias + movie_bias
        return prediction.squeeze()

    def get_embeddings(self, user, movie):
        """
        Извлекает эмбеддинги пользователей и фильмов.

        Возвращает векторные представления (эмбеддинги) для заданных пользователя и фильма.
        Эти эмбеддинги могут быть использованы для дальнейшего анализа или визуализации.

        Args:
            user (torch.Tensor): Тензор с идентификаторами пользователей.
            movie (torch.Tensor): Тензор с идентификаторами фильмов.

        Returns:
            tuple: Кортеж из двух тензоров:
                - user_embedding (torch.Tensor): Эмбеддинг пользователя.
                - movie_embedding (torch.Tensor): Эмбеддинг фильма.
        """
        user_embedding = self.user_factors(user)
        movie_embedding = self.movie_factors(movie)
        return user_embedding, movie_embedding
        # позволяет получить векторные представления пользователей и фильмов
        # для дальнейшего использования в модели машинного обучения.
