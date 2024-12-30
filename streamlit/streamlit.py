import streamlit as st
import numpy as np
from PIL import Image
import random
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import logging
from logging.handlers import RotatingFileHandler
import os
import requests

api_url = "http://backend:8000/generate/"

# ОЧЕНЬ СКРОМНОЕ ЛОГГИРОВАНИЕ
# Создание папки
os.makedirs("/logs", exist_ok=True)

# Настройки
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

log_file = "logs/stremlit.log"

# Ротация - 5 Мб - максимум 5 штук
log_handler = RotatingFileHandler(
    log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
log_handler.setFormatter(log_formatter)

logger = logging.getLogger("MyAppLogger")
logger.setLevel(logging.DEBUG)
logger.addHandler(log_handler)

logger.info("Logger initialized and ready to use.")

if "eda_clicked" not in st.session_state:
    st.session_state.eda_clicked = False
    logger.debug("Initialized session state: eda_clicked = False")

if "dataset_for_EDA" not in st.session_state:
    st.session_state.dataset_for_EDA = None
    logger.debug("Initialized session state: dataset_for_EDA = None")

if "info_clicked" not in st.session_state:
    st.session_state.info_clicked = False
    logger.debug("Initialized session state: info_clicked = False")

themes = [
    "plotly",
    "ggplot2",
    "seaborn",
    "simple_white",
    "xgridoff",
    "presentation",
    "streamlit",
]

def dataset_to_eda(df):
    logger.info("Starting dataset transformation for EDA.")
    
    dataset_transform = pd.DataFrame(columns=[])
    
    try:
        dataset_transform["epitets_num"] = df["text"].apply(lambda x: len(x.split(",")[1:]))
        logger.debug("Calculated epitets_num.")

        dataset_transform["description"] = df["text"].apply(
            lambda x: ",".join(x.split(",")[1:])
        )
        logger.debug("Extracted descriptions.")

        dataset_transform["len"] = df["text"].apply(lambda x: len(x))
        logger.debug("Calculated lengths of text.")

        dataset_transform["shape"] = df["image"].apply(lambda x: np.array(x).shape)
        logger.debug("Extracted shapes of images.")

        dataset_transform["h"] = df["image"].apply(lambda x: np.array(x).shape[0])
        logger.debug("Extracted heights of images.")

        dataset_transform["w"] = df["image"].apply(lambda x: np.array(x).shape[1])
        logger.debug("Extracted widths of images.")

        dataset_transform["rgb"] = dataset_transform["image"].apply(
            lambda x: "RGB" if len(np.array(x).shape) == 3 else "BW"
        )
        logger.debug("Determined color mode (RGB/BW) for images.")

        dataset_transform["ratio"] = dataset_transform["shape"].apply(lambda x: x[0] / x[1])
        logger.debug("Calculated aspect ratios of images.")

        dataset_transform["pixel"] = dataset_transform["shape"].apply(lambda x: np.prod(x))
        logger.debug("Calculated pixel counts for images.")

    except Exception as e:
        logger.error(f"Error during dataset transformation: {str(e)}")
    
    logger.info("Dataset transformation for EDA completed.")
    return dataset_transform
    
st.title("Генерация логотипа по текстовому описанию")

description = st.text_input("Введите описание логотипа:")

if st.button("Сгенерировать логотип"):
    if description:
        generated_images = []
        logger.info(f"Запрос на генерацию логотипа с описанием: {description}")

        for i in range(4):
            payload = {"description": description}
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                # Конвертация изображения из байтов
                response_list = response.json()
                pixel_data = np.array(response_list["image"], dtype=np.uint8)
                generate_image = Image.fromarray(pixel_data)
                generated_images.append(generate_image)
                logger.debug(f"Логотип #{i + 1} успешно сгенерирован.")
            else:
                logger.error(f"Ошибка при подключении к API: {response.status_code}")
                st.write("Ошибка при подключении к API")
        
        st.session_state.generate_image = generated_images
        st.session_state.description = description
        logger.info("Генерация логотипов завершена.")
    else:
        st.warning("Пожалуйста, введите описание.")

if "generate_image" in st.session_state:
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        st.image(
            st.session_state.generate_image[0],
            caption="Сгенерированный логотип #1",
            width=124,
        )
    with col2:
        st.image(
            st.session_state.generate_image[1],
            caption="Сгенерированный логотип #2",
            width=124,
        )
    with col3:
        st.image(
            st.session_state.generate_image[2],
            caption="Сгенерированный логотип #3",
            width=124,
        )
    with col4:
        st.image(
            st.session_state.generate_image[3],
            caption="Сгенерированный логотип #4",
            width=124,
        )

with st.expander("Краткая информация о модели: свернуть/развернуть", expanded=True):
    if st.button("Краткая информация о модели"):
        st.session_state.info_clicked = True

    if st.session_state.info_clicked:
        st.header("Краткая справка")
        st.markdown(
            f"Модель GAN состоит из двух нейросетей: генератора и дискриминатора. "
            f"И генератор, и дискриминатор представляют собой свёрточные нейросети."
            f" Генератор обучается создавать изображения из шума и текстового описания, "
            f"а дискриминатор учится распознавать сгенерированные изображения."
            f" Обучение ввиду ограниченных ресурсов длилось всего 15 эпох."
            f" Для оценки качества генерации использовалась метрика FID."
            f" Ссылка на [репозиторий GitHub](https://github.com/HerrVonBeloff/AI-YP_24-team-42).",
            unsafe_allow_html=True,
        )
        
        
       st.header("Процесс обучения")
theme_model = st.selectbox(
    "Выберите тему для графиков", sorted(themes), key="theme_selector1"
)

try:
    df_loss = pd.read_csv("loss_long.csv")
    logger.info("Данные потерь успешно загружены из loss_long.csv")
except Exception as e:
    logger.error(f"Ошибка при загрузке данных потерь: {e}")
    st.error("Не удалось загрузить данные потерь.")

try:
    df_metric = pd.read_csv("metric.csv")
    logger.info("Данные метрик успешно загружены из metric.csv")
except Exception as e:
    logger.error(f"Ошибка при загрузке данных метрик: {e}")
    st.error("Не удалось загрузить данные метрик.")

fig = px.line(
    df_loss,
    x="Epoch",
    y="Loss Value",
    color="Loss Type",
    title="Потери",
    template=theme_model,
)
fig.update_layout(
    xaxis=dict(
        showgrid=True,
    ),
    yaxis=dict(
        showgrid=True,
    ),
    title=dict(
        x=0.4,
    ),
)
fig.update_layout(
    xaxis_title="Эпохи",
    yaxis_title="Значения потерь",
    legend=dict(title="Нейросеть"),
)
fig.update_traces(name="Дискриминатор", selector=dict(name="D_loss"))
fig.update_traces(name="Генератор", selector=dict(name="G_loss"))
st.plotly_chart(fig)

logger.info("График потерь успешно отображён.")

st.header("Метрика FID")
fig = px.line(
    df_metric,
    x="Epoch",
    y="FID score",
    title="Метрика FID",
    template=theme_model,
)
fig.update_layout(
    xaxis=dict(
        showgrid=True,
    ),
    yaxis=dict(
        showgrid=True,
    ),
    title=dict(
        x=0.4,
    ),
)
fig.update_layout(
    xaxis_title="Эпохи",
    yaxis_title="Значения метрики",
)
st.plotly_chart(fig)

logger.info("График метрики FID успешно отображён.")

with st.expander("Загрузка данных: свернуть/развернуть", expanded=True):
    st.header("Загрузка данных")
    example = {
        "image": ["{'bytes': b'\x89PNG\r\n\x1a\n\x00\...'}"],
        "text": ["Simple elegant logo for Concept, love orange ..."],
    }
    example = pd.DataFrame(example)
    example.index = range(456, 457)
    
    st.markdown("Требования к датасету")
    st.write(example)

    example_dataset = "https://drive.google.com/file/d/1BiUi9TOVgIjEggFQHb9d49Dp-z0pgIvI/view?usp=sharing"
    
    st.markdown(
        "Формат parquet. "
        "Изображения представлены в байтовом виде внутри словаря, "
        "текст представлен в виде обычной строки с перечислением эпитетов. "
        f"Ссылка на [пример датасета]({example_dataset})."
    )



    uploaded_file = st.file_uploader("Загрузите датасет", type=["parquet"])

if uploaded_file is not None:
    try:
        dataset = pd.read_parquet(uploaded_file)
        logger.info("Датасет успешно загружен.")
        
        dataset["image"] = dataset["image"].apply(
            lambda x: Image.open(BytesIO(x.get("bytes")))
        )
        st.markdown(
            f'<span style="color:gray">Количество объектов в датасете: </span>{len(dataset)}',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Ошибка загрузки датасета: {e}")
        logger.error(f"Ошибка загрузки датасета: {e}")

if uploaded_file is not None:
    try:
        dataset = pd.read_parquet(uploaded_file)
        
        dataset["image"] = dataset["image"].apply(
            lambda x: Image.open(BytesIO(x.get("bytes")))
        )
    except Exception as e:
        st.error(f"Ошибка извлечения изображения: {e}")
        logger.error(f"Ошибка извлечения изображения: {e}")

    with st.expander(
        "Получить случайный элемент датасета: свернуть/развернуть", expanded=True
    ):
        if st.button("Получить случайный элемент датасета"):
            ind = random.randint(0, len(dataset) - 1)  # Исправлено на len(dataset) - 1
            st.session_state.index = ind
            st.session_state.dataset_image = dataset["image"][ind]
            st.session_state.dataset_text = dataset["text"][ind]
            logger.info(f"Выбран случайный элемент с индексом: {ind}")

        if "dataset_image" in st.session_state:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(
                    st.session_state.dataset_image,
                    caption=f"Логотип из датасета: индекс {st.session_state.index}",
                )
            st.markdown(
                f'<span style="color:gray">Текстовое описание логотипа: </span>{st.session_state.dataset_text}',
                unsafe_allow_html=True,
            )

    with st.expander("EDA: свернуть/развернуть", expanded=True):
        if st.button("EDA"):
            st.session_state.eda_clicked = True
            try:
                dataset_for_eda = dataset_to_eda(dataset)
                st.session_state.dataset_for_EDA = dataset_for_eda
                logger.info("EDA успешно выполнен.")
            except Exception as e:
                st.error(f"Ошибка данных датасета: {e}")
                logger.error(f"Ошибка форматирования датасета: {e}")

        if st.session_state.eda_clicked:
            if st.session_state.dataset_for_EDA is not None:
                data = st.session_state.dataset_for_EDA
                st.title("Анализ текстовых данных")
                st.subheader("Облако слов")
                
                if "word_cloud" not in st.session_state:
                    wc = WordCloud(background_color="black", width=1000, height=500)
                    words = data["description"].explode().values
                    wc.generate(" ".join(words))

                    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
                    plt.axis("off")
                    plt.tight_layout(pad=0)
                    ax.set_title(f"Облако слов для описаний логотипов", fontsize=30)
                    ax.imshow(wc, alpha=0.98)

                    st.session_state.word_cloud = fig
                
                st.pyplot(st.session_state.word_cloud)

                theme = st.selectbox(
                    "Выберите тему для графиков", sorted(themes), key="theme_selector"
                )


                
                
                st.subheader("Гистограмма распределения длин описаний")
bins = st.slider(
    "Количество интервалов (bins)", 5, 50, 10, key="bins_description"
)
logger.info(f"Выбрано количество интервалов для длин описаний: {bins}")

bin_edges = np.linspace(data["len"].min(), data["len"].max(), bins + 1)
fig = px.histogram(
    data,
    x="len",
    nbins=bins,
    title="Распределение длин описаний",
    template=theme,
)

fig.update_traces(
    xbins=dict(
        start=bin_edges[0],
        end=bin_edges[-1],
        size=(bin_edges[1] - bin_edges[0]),
    ),
)
fig.update_layout(
    bargap=0.1,
    xaxis=dict(
        showgrid=True,
    ),
    yaxis=dict(
        showgrid=True,
    ),
    title=dict(
        x=0.4,
    ),
)
fig.update_layout(
    xaxis_title="Длина описания в символах",
    yaxis_title="Частота",
)

st.plotly_chart(fig)
logger.info("Гистограмма распределения длин описаний успешно отображена.")

st.subheader("Boxplot для длин описаний")
fig = px.box(data, x="len", template=theme)
fig.update_layout(
    title=dict(
        text="Boxplot длин описаний",
        x=0.4,
    ),
    xaxis=dict(
        showgrid=True,
        title="Длина описания в символах",
    ),
)

st.plotly_chart(fig)
logger.info("Boxplot для длин описаний успешно отображен.")

st.subheader("Гистограмма количества эпитетов в описании*")
st.write(
    "*количество слов и словосочетаний, записанных через запятую, в описании изображения"
)
bins6 = st.slider(
    "Количество интервалов (bins)", 5, 50, 10, key="bins_epitets_num"
)
logger.info(f"Выбрано количество интервалов для эпитетов: {bins6}")

bin_edges6 = np.linspace(
    data["epitets_num"].min(), data["epitets_num"].max(), bins6 + 1
)
fig = px.histogram(
    data,
    x="epitets_num",
    nbins=bins6,
    title="Количество эпитетов в описании",
    template=theme,
)
fig.update_traces(
    xbins=dict(
        start=bin_edges6[0],
        end=bin_edges6[-1],
        size=(bin_edges6[1] - bin_edges6[0]),
    ),
)

st.plotly_chart(fig)
logger.info("Гистограмма количества эпитетов в описании успешно отображена.")






                
              # Гистограмма для количества эпитетов
logger.info("Создание гистограммы для количества эпитетов.")
fig.update_layout(
    bargap=0.1,
    xaxis=dict(
        showgrid=True,
    ),
    yaxis=dict(
        showgrid=True,
    ),
    title=dict(
        x=0.4,
    ),
)
fig.update_layout(
    xaxis_title="Количество эпитетов",
    yaxis_title="Частота",
)
st.plotly_chart(fig)
logger.info("Гистограмма для количества эпитетов успешно отображена.")

# Boxplot для количества эпитетов
st.subheader("Boxplot для количества эпитетов")
logger.info("Создание boxplot для количества эпитетов.")
fig = px.box(data, x="epitets_num", template=theme)
fig.update_layout(
    title=dict(
        text="Количество эпитетов в описании",
        x=0.4,
    ),
    xaxis=dict(
        showgrid=True,
        title="Количество эпитетов",
    ),
)
st.plotly_chart(fig)
logger.info("Boxplot для количества эпитетов успешно отображен.")

# Анализ изображений
st.title("Анализ изображений")
value_counts = data["rgb"].value_counts()
pie_data = pd.DataFrame(
    {
        "Value": ["RGB", "BW"],
        "Count": [
            value_counts.get("RGB", 0),
            value_counts.get("BW", 0),
        ],
    }
)
st.subheader("Соотношение RGB и чёрно-белых логотипов")
logger.info("Создание пироговой диаграммы соотношения RGB и BW.")
fig = px.pie(pie_data, names="Value", values="Count", template=theme)

fig.update_layout(
    title=dict(
        text="Тип изображения",
        x=0.4,
    )
)
st.plotly_chart(fig)
logger.info("Пироговая диаграмма соотношения RGB и BW успешно отображена.")

# Распределение высоты изображений
st.subheader("Высота изображений")
bins1 = st.slider(
    "Количество интервалов (bins)", 5, 50, 10, key="bins_height"
)
bin_edges1 = np.linspace(data["h"].min(), data["h"].max(), bins1 + 1)
logger.info("Создание гистограммы распределения высоты изображений.")
fig = px.histogram(
    data,
    x="h",
    nbins=bins1,
    title="Распределение высоты изображений",
    template=theme,
)
fig.update_layout(
    xaxis_title="Высота",
    yaxis_title="Частота",
)
fig.update_traces(
    xbins=dict(
        start=bin_edges1[0],
        end=bin_edges1[-1],
        size=(bin_edges1[1] - bin_edges1[0]),
    ),
)
fig.update_layout(
    title=dict(
        text="Высота изображений",
        x=0.4,
    )
)
st.plotly_chart(fig)
logger.info("Гистограмма высоты изображений успешно отображена.")

# Распределение ширины изображений
st.subheader("Ширина изображений")
bins2 = st.slider(
    "Количество интервалов (bins)", 5, 50, 10, key="bins_width"
)
bin_edges2 = np.linspace(data["w"].min(), data["w"].max(), bins2 + 1)
logger.info("Создание гистограммы распределения ширины изображений.")
fig = px.histogram(data, x="w", nbins=bins2, template=theme)  # Исправлено на 'x="w"'
fig.update_layout(
    xaxis_title="Ширина",
    yaxis_title="Частота",
)
fig.update_traces(
    xbins=dict(
        start=bin_edges2[0],
        end=bin_edges2[-1],
        size=(bin_edges2[1] - bin_edges2[0]),
    ),
)
st.plotly_chart(fig)
logger.info("Гистограмма ширины изображений успешно отображена.")






fig.update_layout(
    title=dict(
        text="Ширина изображений",
        x=0.4,
    )
)
st.plotly_chart(fig)
logger.info("Гистограмма ширины изображений успешно отображена.")

# Соотношение сторон
st.subheader("Соотношение сторон")
bins3 = st.slider(
    "Количество интервалов (bins)", 5, 50, 10, key="bins_ratio"
)
bin_edges3 = np.linspace(
    data["ratio"].min(), data["ratio"].max(), bins3 + 1
)
logger.info("Создание гистограммы распределения соотношения сторон.")
fig = px.histogram(data, x="ratio", nbins=bins3, template=theme)
fig.update_layout(
    xaxis_title="Соотношение сторон",
    yaxis_title="Частота",
)
fig.update_traces(
    xbins=dict(
        start=bin_edges3[0],
        end=bin_edges3[-1],
        size=(bin_edges3[1] - bin_edges3[0]),
    ),
)
fig.update_layout(
    title=dict(
        text="Распределение соотношений сторон h/w",
        x=0.4,
    )
)
st.plotly_chart(fig)
logger.info("Гистограмма распределения соотношения сторон успешно отображена.")

# Соотношение сторон (без квадратных изображений)
st.subheader("Соотношение сторон (без квадратных изображений)")
bins4 = st.slider(
    "Количество интервалов (bins)", 5, 50, 10, key="bins_ratio1"
)
bin_edges4 = np.linspace(
    data[data["ratio"] != 1]["ratio"].min(),
    data[data["ratio"] != 1]["ratio"].max(),
    bins4 + 1,
)
logger.info("Создание гистограммы распределения соотношения сторон без квадратных изображений.")
fig = px.histogram(
    data[data["ratio"] != 1],
    x="ratio",
    nbins=bins4,
    template=theme,
)
fig.update_layout(
    xaxis_title="Соотношение сторон",
    yaxis_title="Частота",
)
fig.update_traces(
    xbins=dict(
        start=bin_edges4[0],
        end=bin_edges4[-1],
        size=(bin_edges4[1] - bin_edges4[0]),
    ),
)
fig.update_layout(
    title=dict(
        text="Распределение соотношений сторон h/w",
        x=0.4,
    )
)
st.plotly_chart(fig)
logger.info("Гистограмма распределения соотношения сторон без квадратных изображений успешно отображена.")

# Количество пикселей
st.subheader("Количество пикселей")
bins5 = st.slider(
    "Количество интервалов (bins)", 5, 50, 10, key="bins_pixel"
)
bin_edges5 = np.linspace(
    data["pixel"].min(), data["pixel"].max(), bins5 + 1
)
logger.info("Создание гистограммы распределения количества пикселей.")
fig = px.histogram(data, x="pixel", nbins=bins5, template=theme)
fig.update_layout(
    xaxis_title="Количество пикселей",
    yaxis_title="Частота",
)
fig.update_traces(
    xbins=dict(
        start=bin_edges5[0],
        end=bin_edges5[-1],
        size=(bin_edges5[1] - bin_edges5[0]),
    ),
)
fig.update_layout(
    title=dict(
        text="Распределение количества пикселей",
        x=0.4,
    )
)
st.plotly_chart(fig)
logger.info("Гистограмма распределения количества пикселей успешно отображена.")
