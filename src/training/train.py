import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

from src.data_loading import MovieDataset
from src.models.combined_recommender import CombinedRecommender

def train_and_evaluate(ratings_df, n_factors=50, n_clusters=5):
    '''
    Обучает и оценивает комбинированную модель.
    :param ratings_df: DataFrame с рейтингами.
    :param n_factors: Размерность скрытых факторов.
    :param n_clusters: Количество кластеров для K-средних.
    :return: Обученная модель.
    '''
    user_ids = ratings_df['userId'].unique()
    movie_ids = ratings_df['movieId'].unique()

    user_to_idx = {user: idx for idx, user in enumerate(user_ids)}
    movie_to_idx = {movie: idx for idx, movie in enumerate(movie_ids)}

    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

    # Создание загрузчиков данных
    train_dataset = MovieDataset(train_df, user_to_idx, movie_to_idx)
    test_dataset = MovieDataset(test_df, user_to_idx, movie_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    model = CombinedRecommender(len(user_ids), len(movie_ids), n_factors, n_clusters)

    print("Обучение матричной факторизации...")
    train_losses, test_losses = model.train_mf(train_loader, test_loader)

    print("\nОбучение K-means...")
    model.train_kmeans(train_loader)

    print("\nОценка модели...")
    model.mf_model.eval()
    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for users, movies, ratings in test_loader:
            predictions = model.predict(users, movies)
            all_predictions.extend(predictions)
            all_actuals.extend(ratings.numpy())

    mse = mean_squared_error(all_actuals, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_actuals, all_predictions)

    print(f"\nФинальные результаты:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Визуализация результатов
    plt.figure(figsize=(15, 5))

    # Кривая обучения
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Потери от обучения')
    plt.plot(test_losses, label='Тестовые потери')
    plt.title('Кривая обучения')
    plt.xlabel('Epoch')
    plt.ylabel('Потери')
    plt.legend()

    # Предсказания и фактические результаты
    plt.subplot(1, 3, 2)
    plt.scatter(all_actuals, all_predictions, alpha=0.1)
    plt.plot([1, 5], [1, 5], 'r--')
    plt.title('Прогнозы и факты')
    plt.xlabel('Фактические рейтинги')
    plt.ylabel('Прогнозируемые рейтинги')

    # Строим график распределения кластеров
    plt.subplot(1, 3, 3)
    combined_emb = model.get_combined_embeddings(
        train_dataset.users[:1000],
        train_dataset.movies[:1000]
    )
    clusters = model.kmeans.predict(combined_emb)

    # Использование t-SNE для уменьшения размерности
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_emb = tsne.fit_transform(combined_emb)

    plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c=clusters, cmap='viridis')
    plt.title('Распределение кластеров')

    plt.tight_layout()
    plt.show()

    return model
