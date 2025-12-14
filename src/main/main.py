import pandas as pd
import numpy as np
import joblib
from recipes import RecipePredictor, RecipeRecommender, NutritionAnalyzer


def input_ingredients():
    print("Введите названия ингредиентов через запятую:")
    ingredients = input().split(',')
    ingredients = [i.strip().lower() for i in ingredients]
    return ingredients


def output_first_paragraph(predict: float) -> None:
    print('1. Наш прогноз для данных ингредиентов:')
    print(
        f'Предсказание оценки блюда с комбинацией таких ингредиентов: {predict:.3f}')


def output_second_paragraph(nutrition_analyzer: NutritionAnalyzer, ingredients: list) -> None:
    print('2. Процент нутриентов от суточной нормы в каждом из ингредиентов')
    nutrition_analyzer.format_nutrition_output(ingredients)


def output_third_paragraph(similar_recipes: list) -> None:
    if similar_recipes:
        print('3. Топ-3 похожих рецептов')
        for recipe in similar_recipes:
            print(f"- {recipe['title'].strip()}, rating: {recipe['rating']}, URL: {recipe['url']}")
    else: 
        print("Похожих ингредиентов в нашей системе не найдено")


def main():
    try:
        ingredients = input_ingredients()
        df = pd.read_csv('../data/clean_epi_r_links.csv', dtype={'link': str})
        nutrients_df = pd.read_csv('../data/nutrients.csv')
        model_data = joblib.load('../final_models/random_forest_regressor.pkl')

        nutrition_analyzer = NutritionAnalyzer(nutrients_df)
        predictor = RecipePredictor(
            df, model_data['model'], model_data['features_column'], nutrients_df)
        recommender = RecipeRecommender(df)

        vector = predictor.create_features_vector(ingredients)
        output_first_paragraph(predictor.predict_class_rating(vector))
        print()
        output_second_paragraph(nutrition_analyzer, ingredients)
        print()
        output_third_paragraph(recommender.find_similar_recipes(ingredients))

    except Exception as e:
        print(f'Произошла ошибка {e}')


if __name__ == "__main__":
    main()
