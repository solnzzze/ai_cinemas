import sys
import os
import torch 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loading import load_data
from src.training.train import train_and_evaluate

def main():
    # Загрузка данных
    movies_df, ratings_df = load_data()

    # Обучение и оценка модели
    model = train_and_evaluate(ratings_df)

    # Сохранение модели
    print("Сохранение модели...")
    torch.save({
        'mf_model_state_dict': model.mf_model.state_dict(),
        'kmeans_model': model.kmeans,
        'cluster_ratings': model.cluster_ratings
    }, 'combined_recommender_model.pth')

if __name__ == "__main__":
    main()

    print("Модель сохранена в файл 'combined_recommender_model.pth'")
