import pandas as pd
import numpy as np


class RecipePredictor:
    def __init__(self, df, model, model_features, nutrients_df):
        self.model = model
        self.features = model_features
        self.df = df
        self.nutrients_df = nutrients_df
        self.nutrients_df.rename(columns={'Unnamed: 0': 'name'}, inplace=True)

    def create_features_vector(self, ingredients):
        vector = [0] * len(self.features)
        vector = self._add_nutrients_to_vector(vector, ingredients)

        for ingredient in ingredients:
            for i in range(len(self.features)):
                if ingredient == self.features[i]:
                    vector[i] = 1

        return vector

    def _add_nutrients_to_vector(self, vector, ingredients):
        nutrition_totals = {}
        nutient_columns = ['Protein', 'Sodium']

        for nutrient in nutient_columns:
            nutrition_totals[nutrient.lower()] = 0

        for ingredient in ingredients:
            nutrient_row = self.nutrients_df[self.nutrients_df['name'] == ingredient]
            if not nutrient_row.empty:
                for nutrient in nutient_columns:
                    if nutrient in nutrient_row.columns:
                        nutrition_totals[nutrient.lower(
                        )] += nutrient_row[nutrient].values[0]

        for idx, feature_name in enumerate(self.features):
            if feature_name in nutrition_totals:
                vector[idx] = nutrition_totals[feature_name]

        return vector

    def predict_class_rating(self, vector):
        X = pd.DataFrame([vector], columns=self.features)
        prediction = self.model.predict(X)[0]

        return prediction

    def _map_class(self, class_index):
        class_mapping = {0: "bad", 1: "so-so", 2: "great"}
        return class_mapping.get(class_index, 'unknown predict!')


class RecipeRecommender:
    def __init__(self, df):
        self.df = df
        self.ingredient_columns = [i for i in df.columns if i not in [
            'title', 'rating', 'link', 'calories', 'protein_%', 'fat_%', 'sodium_%']]

    def _ingredients_to_vector(self, ingredients):
        vector = np.zeros(len(self.ingredient_columns))
        for ingredient in ingredients:
            if ingredient in self.ingredient_columns:
                idx = self.ingredient_columns.index(ingredient)
                vector[idx] = 1

        return vector

    def find_similar_recipes(self, ingredients, top_n=3):
        query_vector = self._ingredients_to_vector(ingredients)
        similarities_ing = []

        for idx, row in self.df.iterrows():
            recipe_vector = row[self.ingredient_columns].to_numpy()
            norm_q = np.linalg.norm(query_vector)
            norm_r = np.linalg.norm(recipe_vector)

            if norm_q != 0 and norm_r != 0:
                similarity = np.dot(
                    query_vector, recipe_vector) / (norm_q * norm_r)
                similarities_ing.append((idx, similarity))
            else:
                similarity = 0

        similarities_ing.sort(key=lambda x: x[1], reverse=True)
        top_recipes = []

        for idx, similarity_value in similarities_ing[:top_n]:
            if similarity_value > 0:
                recipe = self.df.iloc[idx]
                top_recipes.append(
                    {
                        'title': recipe['title'],
                        'rating': recipe['rating'],
                        'url': recipe['link'],
                        'similarity': similarity_value
                    }
                )

        return top_recipes


class NutritionAnalyzer:
    def __init__(self, nutrients_df):
        self.nutrients_df = nutrients_df

    def _get_nutrients_precentage(self, ingredients):
        ingredients_dict = {}
        for ingredient in ingredients:
            if ingredient in self.nutrients_df['name'].values:
                ingredients_dict[ingredient] = list(
                    list(self.nutrients_df[self.nutrients_df['name'] == ingredient].values)[0])[1:]
            else:
                ingredients_dict[ingredient] = 'Нет данных'

        return ingredients_dict

    def format_nutrition_output(self, ingredients, top_n=3):
        ingredients_dict = self._get_nutrients_precentage(ingredients)
        for key, value in ingredients_dict.items():
            print(f'Название ингредиента: {key.capitalize()}')
            if value != 'Нет данных':
                
                nutrient_pairs = list(zip(self.nutrients_df.columns[1:], value))
                sorted_pairs = sorted(nutrient_pairs, key=lambda x: x[1], reverse=True)
                
                print(f'Топ-{top_n} нутриентов по проценту от суточной нормы')
                
                for name, percentage in sorted_pairs[:top_n]:
                    print(f'{name} – {percentage:.2f}%')
                    
            else:
                print("Информации по нутриентам на такой ингредиент нет!")

            print('\n')
