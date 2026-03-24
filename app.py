import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Настройка страницы
st.set_page_config(
    page_title="Anime Search Engine",
    page_icon="🎌",
    layout="wide"
)

# Заголовок приложения
st.title("🎌 Anime Search Engine")
st.markdown("Найдите аниме по описанию, жанру или названию!")

# Класс модели для поиска аниме
class AnimeRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.anime_data = None

    def fit(self, df):
        """Обучение модели на данных"""
        self.anime_data = df
        # Объединяем ключевые характеристики в один текст для векторного представления
        df['combined_features'] = (
            df['title'] + ' ' +
            df['genre'] + ' ' +
            df['synopsis'].fillna('')
        )
        # Векторизуем текстовые данные
        self.tfidf_matrix = self.vectorizer.fit_transform(df['combined_features'])
        return self

    def predict(self, query, top_k=5):
        """Поиск похожих аниме по запросу"""
        # Векторизуем запрос
        query_vec = self.vectorizer.transform([query])

        # Вычисляем косинусное сходство
        cosine_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Получаем индексы топ‑K похожих аниме
        similar_indices = cosine_sim.argsort()[-top_k:][::-1]

        # Создаём результат
        results = []
        for idx in similar_indices:
            if cosine_sim[idx] > 0:  # Только если есть сходство
                anime = self.anime_data.iloc[idx]
                results.append({
                    'title': anime['title'],
                    'genre': anime['genre'],
                    'synopsis': anime['synopsis'][:200] + '...',
                    'similarity': round(cosine_sim[idx], 3)
                })

        return results

# Функция загрузки и предобработки данных
@st.cache_data
def load_and_preprocess():
    try:
        # Загружаем данные (предполагается, что файл уже скачан с Kaggle)
        df = pd.read_csv('anime_offline_database.csv')

        # Удаляем строки с пропущенными значениями в ключевых колонках
        df = df.dropna(subset=['title', 'genre', 'synopsis'])

        return df
    except FileNotFoundError:
        return None

# Загружаем и обрабатываем данные
df = load_and_preprocess()

if df is None:
    st.error("Не удалось загрузить данные. Проверьте наличие файла 'anime_offline_database.csv'.")
else:
    # Инициализация модели
    @st.cache_resource
    def create_model(data):
        model = AnimeRecommender()
        model.fit(data)
        return model

    model = create_model(df)
    st.success(f"Загружено {len(df)} аниме для поиска!")

    # Интерфейс пользователя
    col1, col2 = st.columns([3, 1])

    with col1:
        user_query = st.text_area(
            "Введите запрос для поиска аниме:",
            placeholder="Например: 'приключения в фэнтези мире' или 'романтика школа'",
            height=100
        )

        if user_query:
            with st.spinner("Ищем подходящие аниме..."):
                results = model.predict(user_query, top_k=5)

            if results:
                st.subheader("Результаты поиска:")
                for i, anime in enumerate(results, 1):
                    with st.container():
                        st.markdown(f"**{i}. {anime['title']}**")
                        st.write(f"**Жанр:** {anime['genre']}")
                        st.write(f"**Описание:** {anime['synopsis']}")
                        st.write(f"**Сходство:** {anime['similarity']}")
                        st.divider()
            else:
                st.warning("По вашему запросу ничего не найдено. Попробуйте изменить запрос.")

    with col2:
        st.subheader("Опции")

        # Загрузка пользовательских данных
        uploaded_file = st.file_uploader(
            "Загрузите свой CSV‑файл с аниме",
            type=['csv']
        )

        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file)
                st.success("Файл успешно загружен!")

                # Показываем первые строки
                st.dataframe(user_df.head())
            except Exception as e:
                st.error(f"Ошибка при загрузке файла: {e}")

        # Информация о данных
        st.markdown("---")
        st.markdown("**Информация о данных:**")
        st.write(f"Всего аниме: {len(df)}")
        st.write(f"Колонки: {', '.join(df.columns)}")

    # Пример запросов
    st.markdown("---")
    st.markdown("**Примеры запросов:**")
    examples = [
        "приключения фэнтези магия",
        "романтика школа повседневность",
        "научная фантастика космос",
        "детектив мистика психология"
    ]
    for example in examples:
        if st.button(example):
            results = model.predict(example, top_k=3)
            for anime in results:
                st.write(f"- {anime['title']} (сходство: {anime['similarity']})")
