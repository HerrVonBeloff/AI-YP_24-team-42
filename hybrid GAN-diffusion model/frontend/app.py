import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
import random
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
from typing import Optional

# --- Инициализация session_state ---
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'selected_idx' not in st.session_state:
    st.session_state.selected_idx = None
if 'final_image' not in st.session_state:
    st.session_state.final_image = None
if 'eda_clicked' not in st.session_state:
    st.session_state.eda_clicked = False
if 'dataset_for_EDA' not in st.session_state:
    st.session_state.dataset_for_EDA = None
if 'info_clicked' not in st.session_state:
    st.session_state.info_clicked = False

st.set_page_config(page_title="Генератор логотипов", layout="wide")
st.title("🎨 Генератор логотипов")  

label = st.text_input("Название логотипа:", placeholder="Например: EcoTech")

if st.button("Сгенерировать варианты") and label:
    with st.spinner("Генерация вариантов..."):
        base_images = generate_cgan_images(label, count=5)
        upscaled_images = [upscale_image(img, label) for img in base_images]
        st.session_state.generated_images = [img for img in upscaled_images if img is not None]
        st.session_state.label = label
        st.session_state.selected_idx = None
        st.session_state.final_image = None

if st.session_state.generated_images:
    st.subheader("Выберите понравившийся вариант:")
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        if idx < len(st.session_state.generated_images):
            image_path = st.session_state.generated_images[idx]
            with col:
                if st.button(f"Выбрать №{idx + 1}", key=f"select_{idx}"):
                    st.session_state.selected_idx = idx
                st.image(
                    image_path,
                    caption=f"Шаблон №{idx + 1}",
                    width=150  # <-- Здесь задаём размер
                )

    if st.session_state.selected_idx is not None:
        selected_img = st.session_state.generated_images[st.session_state.selected_idx]
        st.success(f"Выбран вариант #{st.session_state.selected_idx+1}")

        with st.expander("Настройки стилизации", expanded=True):
            strength = st.slider("Интенсивность стилизации", 0.3, 0.9, 0.7, 0.05)
            steps = st.slider("Количество шагов", 10, 100, 30, 5)
            guidance = st.slider("Контроль стиля", 7.0, 20.0, 13.0, 0.5)

            # Выбор LoRA стиля
            available_loras = get_available_loras()
            lora_options = ["Нет"] + available_loras
            selected_lora = st.selectbox("Выберите LoRA стиль", lora_options)

        if st.button("Применить стиль"):
            with st.spinner("Стилизация..."):
                final_path = style_with_diffusion(
                    selected_img,
                    st.session_state.label,
                    strength,
                    steps,
                    guidance,
                    selected_lora if selected_lora != "Нет" else None
                )
                st.session_state.final_image = final_path

        if st.session_state.final_image:
            st.markdown("###  Финальный логотип")
            st.image(
                st.session_state.final_image,
                caption=" Ваш финальный логотип",
                width=512  # <-- здесь задаём нужный размер
            )
            with open(st.session_state.final_image, "rb") as f:
                st.download_button(
                    "📥 Скачать логотип",
                    data=f,
                    file_name=f"{st.session_state.label}_logo.png",
                    mime="image/png"
                )

st.title("ℹ️ Информация о модели")

themes = [
    "plotly",
    "ggplot2",
    "seaborn",
    "simple_white",
    "presentation",
    "streamlit",
]

def dataset_to_eda(df):
    dataset_transform = pd.DataFrame()
    dataset_transform["epitets_num"] = df["text"].apply(lambda x: len(x.split(",")[1:]))
    dataset_transform["description"] = df["text"].apply(lambda x: ",".join(x.split(",")[1:]))
    dataset_transform["len"] = df["text"].apply(lambda x: len(x))
    dataset_transform["shape"] = df["image"].apply(lambda x: np.array(x).shape)
    dataset_transform["h"] = df["image"].apply(lambda x: np.array(x).shape[0])
    dataset_transform["w"] = df["image"].apply(lambda x: np.array(x).shape[1])
    dataset_transform["rgb"] = df["image"].apply(
        lambda x: "RGB" if len(np.array(x).shape) == 3 else "BW"
    )
    dataset_transform["ratio"] = dataset_transform["shape"].apply(lambda x: x[0] / x[1])
    dataset_transform["pixel"] = dataset_transform["shape"].apply(lambda x: np.prod(x))
    return dataset_transform

with st.expander("Краткая информация о модели: свернуть/развернуть", expanded=True):
    if st.button("Краткая информация о модели"):
        st.session_state.info_clicked = True
    if st.session_state.info_clicked:
        st.header("Краткая справка")
        st.markdown(
            f"1. cGAN генерирует базовые варианты логотипов из текстового описания<br>"
            f"   • Генератор: 4-слойная транспонированная CNN (100D шум + 128D текстовый эмбеддинг)<br>"
            f"   • Обучение: 15 эпох, метрика FID<br>"
            f"2. Real-ESRGAN повышает качество изображений (4x апскейл)<br>"
            f"3. Stable Diffusion 1.5 с LoRA-адаптацией применяет стилизацию:<br>"
            f"   • Минимальный размер 512x512<br>"
            f"   • Поддержка пользовательских стилей через LoRA<br>"
            f"   • Автоматическое уменьшение разрешения при нехватке памяти<br>"
            f"<a href='https://github.com/HerrVonBeloff/AI-YP_24-team-42'>🔗  Репозиторий GitHub</a>", 
            unsafe_allow_html=True
        )

        st.header("Процесс обучения cGAN")
        theme_model = st.selectbox(
            "Выберите тему для графиков", sorted(themes), key="theme_selector1"
        )
        try:
            df_loss = pd.read_csv("loss_long.csv")
            df_metric = pd.read_csv("metric.csv")
        except Exception as e:
            st.error("❌ Не удалось загрузить CSV файлы. Проверьте их наличие.")
            df_loss = pd.DataFrame()
            df_metric = pd.DataFrame()

        if not df_loss.empty:
            fig = px.line(
                df_loss,
                x="Epoch",
                y="Loss Value",
                color="Loss Type",
                title="Потери",
                template=theme_model
            )
            fig.update_layout(
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True),
                title=dict(x=0.4)
            )
            fig.update_layout(
                xaxis_title="Эпохи",
                yaxis_title="Значения потерь",
                legend=dict(title="Нейросеть")
            )
            fig.update_traces(name="Дискриминатор", selector=dict(name="D_loss"))
            fig.update_traces(name="Генератор", selector=dict(name="G_loss"))
            st.plotly_chart(fig)

        st.header("Метрика FID")
        if not df_metric.empty:
            fig = px.line(
                df_metric,
                x="Epoch",
                y="FID score",
                title="Метрика FID",
                template=theme_model
            )
            fig.update_layout(
                bargap=0.1,
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True),
                title=dict(x=0.4)
            )
            fig.update_layout(
                xaxis_title="Эпохи",
                yaxis_title="Значения метрики",
            )
            st.plotly_chart(fig)

with st.expander("Загрузка данных: свернуть/развернуть", expanded=True):
    st.header("Загрузка данных")
    example = {
        "image": ["{'bytes': b'\\x89PNG\\x1a'}"],
        "text": ["Simple elegant logo for Concept, love orange ..."]
    }
    example_df = pd.DataFrame(example)
    example_df.index = range(456, 457)
    st.markdown("Требования к датасету")
    st.write(example_df)
    example_dataset_url = "https://drive.google.com/file/d/1BiUi9TOVgIjEggFQHb9d49Dp-z0pgIvI/view?usp=sharing"
    st.markdown(
        "Формат `parquet`. "
        "Изображения представлены в байтовом виде внутри словаря, "
        "текст представлен в виде обычной строки с перечислением эпитетов. "
        f"[🔗 Пример датасета]({example_dataset_url})"
    )
    uploaded_file = st.file_uploader("Загрузите датасет (.parquet)", type=["parquet"])
    if uploaded_file is not None:
        try:
            dataset = pd.read_parquet(uploaded_file)
            dataset["image"] = dataset["image"].apply(
                lambda x: Image.open(BytesIO(x.get("bytes")))
            )
            st.markdown(
                f'<span style="color:gray">Количество объектов в датасете: </span>{len(dataset)}',
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"❌ Ошибка чтения датасета: {e}")

    if uploaded_file is not None:
        try:
            dataset = pd.read_parquet(uploaded_file)
            dataset["image"] = dataset["image"].apply(
                lambda x: Image.open(BytesIO(x.get("bytes")))
            )
            st.session_state.dataset = dataset
        except Exception as e:
            st.error(f"❌ Ошибка извлечения изображения: {e}")

with st.expander("Получить случайный элемент датасета: свернуть/развернуть", expanded=True):
    if st.button("Получить случайный элемент датасета"):
        ind = random.randint(0, len(dataset) - 1) if len(dataset) > 0 else 0
        st.session_state.index = ind
        st.session_state.dataset_image = dataset.iloc[ind]["image"]
        st.session_state.dataset_text = dataset.iloc[ind]["text"]

    if "dataset_image" in st.session_state:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(
                st.session_state.dataset_image,
                caption=f"Логотип из датасета: индекс {st.session_state.index}",
            )
        st.markdown(
            f'<span style="color:gray">Текстовое описание логотипа: </span>{st.session_state.dataset_text}',
            unsafe_allow_html=True
        )

with st.expander("EDA: свернуть/развернуть", expanded=True):
    if st.button("EDA"):
        st.session_state.eda_clicked = True
        try:
            dataset_for_eda = dataset_to_eda(dataset)
            st.session_state.dataset_for_EDA = dataset_for_eda
        except Exception as e:
            st.error(f"❌ Ошибка форматирования датасета: {e}")

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
                ax.axis("off")
                ax.set_title("Облако слов для описаний логотипов", fontsize=30)
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
            bin_edges = np.linspace(data["len"].min(), data["len"].max(), bins + 1)
            fig = px.histogram(
                data,
                x="len",
                nbins=bins,
                title="Распределение длин описаний",
                template=theme
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
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True),
                title=dict(x=0.4)
            )
            fig.update_layout(
                xaxis_title="Длина описания в символах",
                yaxis_title="Частота",
            )
            st.plotly_chart(fig)

            st.subheader("Boxplot для длин описаний")
            fig = px.box(data, x="len", template=theme)
            fig.update_layout(
                title=dict(text="Boxplot длин описаний", x=0.4),
                xaxis=dict(showgrid=True, title="Длина описания в символах"),
            )
            st.plotly_chart(fig)

            st.subheader("Гистограмма количества эпитетов в описании*")
            st.write("*количество слов и словосочетаний, записанных через запятую, в описании изображения")
            bins6 = st.slider(
                "Количество интервалов (bins)", 5, 50, 10, key="bins_epitets_num"
            )
            bin_edges6 = np.linspace(
                data["epitets_num"].min(), data["epitets_num"].max(), bins6 + 1
            )
            fig = px.histogram(
                data,
                x="epitets_num",
                nbins=bins6,
                title="Количество эпитетов в описании",
                template=theme
            )
            fig.update_traces(
                xbins=dict(
                    start=bin_edges6[0],
                    end=bin_edges6[-1],
                    size=(bin_edges6[1] - bin_edges6[0]),
                ),
            )
            fig.update_layout(
                bargap=0.1,
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True),
                title=dict(x=0.4)
            )
            fig.update_layout(
                xaxis_title="Количество эпитетов",
                yaxis_title="Частота",
            )
            st.plotly_chart(fig)

            st.subheader("Boxplot для количества эпитетов")
            fig = px.box(data, x="epitets_num", template=theme)
            fig.update_layout(
                title=dict(text="Количество эпитетов в описании", x=0.4),
                xaxis=dict(showgrid=True, title="Количество эпитетов"),
            )
            st.plotly_chart(fig)

            st.title("Анализ изображений")
            value_counts = data["rgb"].value_counts()
            pie_data = pd.DataFrame({
                "Value": ["RGB", "BW"],
                "Count": [
                    value_counts.get("RGB", 0),
                    value_counts.get("BW", 0)
                ]
            })
            st.subheader("Соотношение RGB и чёрно-белых логотипов")
            fig = px.pie(pie_data, names="Value", values="Count", template=theme)
            fig.update_layout(title=dict(text="Тип изображения", x=0.4))
            st.plotly_chart(fig)

            st.subheader("Высота изображений")
            bins1 = st.slider(
                "Количество интервалов (bins)", 5, 50, 10, key="bins_height"
            )
            bin_edges1 = np.linspace(data["h"].min(), data["h"].max(), bins1 + 1)
            fig = px.histogram(
                data,
                x="h",
                nbins=bins1,
                title="Распределение высоты изображений",
                template=theme
            )
            fig.update_layout(
                xaxis_title="Высота",
                yaxis_title="Частота",
                title=dict(x=0.4)
            )
            fig.update_traces(
                xbins=dict(
                    start=bin_edges1[0],
                    end=bin_edges1[-1],
                    size=(bin_edges1[1] - bin_edges1[0]),
                ),
            )
            st.plotly_chart(fig)

            st.subheader("Ширина изображений")
            bins2 = st.slider(
                "Количество интервалов (bins)", 5, 50, 10, key="bins_wight"
            )
            bin_edges2 = np.linspace(data["w"].min(), data["w"].max(), bins2 + 1)
            fig = px.histogram(
                data,
                x="w",
                nbins=bins2,
                title="Распределение ширины изображений",
                template=theme
            )
            fig.update_layout(
                xaxis_title="Ширина",
                yaxis_title="Частота",
                title=dict(x=0.4)
            )
            fig.update_traces(
                xbins=dict(
                    start=bin_edges2[0],
                    end=bin_edges2[-1],
                    size=(bin_edges2[1] - bin_edges2[0]),
                ),
            )
            st.plotly_chart(fig)

            st.subheader("Соотношение сторон")
            bins3 = st.slider(
                "Количество интервалов (bins)", 5, 50, 10, key="bins_ratio"
            )
            bin_edges3 = np.linspace(data["ratio"].min(), data["ratio"].max(), bins3 + 1)
            fig = px.histogram(
                data,
                x="ratio",
                nbins=bins3,
                title="Распределение соотношений сторон h/w",
                template=theme
            )
            fig.update_layout(
                xaxis_title="Соотношение сторон",
                yaxis_title="Частота",
                title=dict(x=0.4)
            )
            fig.update_traces(
                xbins=dict(
                    start=bin_edges3[0],
                    end=bin_edges3[-1],
                    size=(bin_edges3[1] - bin_edges3[0]),
                ),
            )
            st.plotly_chart(fig)

            st.subheader("Соотношение сторон (без квадратных)")
            filtered_data = data[data["ratio"] != 1]["ratio"]
            if not filtered_data.empty:
                bins4 = st.slider(
                    "Количество интервалов (bins)", 5, 50, 10, key="bins_ratio1"
                )
                bin_edges4 = np.linspace(filtered_data.min(), filtered_data.max(), bins4 + 1)
                fig = px.histogram(
                    filtered_data,
                    x="ratio",
                    nbins=bins4,
                    title="Распределение соотношений сторон h/w",
                    template=theme
                )
                fig.update_layout(
                    xaxis_title="Соотношение сторон",
                    yaxis_title="Частота",
                    title=dict(x=0.4)
                )
                fig.update_traces(
                    xbins=dict(
                        start=bin_edges4[0],
                        end=bin_edges4[-1],
                        size=(bin_edges4[1] - bin_edges4[0]),
                    ),
                )
                st.plotly_chart(fig)

            st.subheader("Количество пикселей")
            bins5 = st.slider(
                "Количество интервалов (bins)", 5, 50, 10, key="bins_pixel"
            )
            bin_edges5 = np.linspace(data["pixel"].min(), data["pixel"].max(), bins5 + 1)
            fig = px.histogram(
                data,
                x="pixel",
                nbins=bins5,
                title="Распределение количества пикселей",
                template=theme
            )
            fig.update_layout(
                xaxis_title="Количество пикселей",
                yaxis_title="Частота",
                title=dict(x=0.4)
            )
            fig.update_traces(
                xbins=dict(
                    start=bin_edges5[0],
                    end=bin_edges5[-1],
                    size=(bin_edges5[1] - bin_edges5[0]),
                ),
            )
            st.plotly_chart(fig)
