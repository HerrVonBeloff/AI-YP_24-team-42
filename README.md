<a name="readme-top"></a>

[![MIT][license-shield]][license-url]
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[!['Black'](https://img.shields.io/badge/code_style-black-black?style=for-the-badge)](https://github.com/psf/black)

<h1 align="center">Генерация логотипов бренда по текстовому описанию</h1>

<details>
  <summary>Содержание</summary>
  <ol>
    <li>
      <a href="#описание-проекта">Описание проекта</a>
    </li>
    <li>
      <a href="#технологии">Технологии</a>
    </li>
    <li>
      <a href="#участники-проекта">Участники проекта</a>
    </li>
    <li>
      <a href="#данные">Данные</a>
    </li>
    <li>
      <a href="#eda">EDA</a>
    </li>
    <li>
      <a href="#baseline">Baseline</a>
    </li>
        <ul>
        <li><a href="#обучающие-данные">Обучающие данные</a></li>
        <li><a href="#характеристики-модели">Характеристики модели</a></li>
        <li><a href="#результаты">Результаты</a></li>
        </ul>
    <li>
      <a href="#сервис">Сервис</a>
    </li>
        <ul>
        <li><a href="#api">API</a></li>
        <li><a href="#streamlit">Streamlit</a></li>
            <ul><li><a href="#генерация-изображений">Генерация изображений</a></li></ul>
            <ul><li><a href="#краткая-информация-о-модели">Краткая информация о модели</a></li></ul>
            <ul><li><a href="#загрузка-датасета">Загрузка датасета</a></li></ul>
            <ul><li><a href="#eda-в-web-приложении">EDA в web-приложении</a></li></ul>
        </ul>
  </ol>
</details>

## Описание проекта

Данный проект предназначен для автоматической генерации уникальных логотипов на основе текстовых описаний, предоставленных пользователем. Используя классичиеские ML-подходы и нейросетевые модели, проект будет создавать графические изображения, соответствующие заданным ключевым словам, указанным в описании.

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

## Технологии

Для реализации проекта используются следующие технологии:

* [![Colab][Colab]][Colab-url]
* [![Kaggle][Kaggle]][Kaggle-url]
* [![Python][Python.org]][Python-url]
  * [![AIOgram][AIOgram]][AIOgram-url]
  * [![Matplotlib][Matplotlib.org]][Matplotlib-url]
  * [![Numpy][Numpy.org]][Numpy-url]
  * [![Pandas][Рandas.pydata.org]][Pandas-url]
  * [![Pytorch][Pytorch]][Pytorch-url]
  * [![scikit-learn][scikit-learn]][scikit-learn-url]
  * [![Spacy][SpaCy]][Spacy-url]
  * [![Seaborn][Seaborn-badge]][Seaborn-url]
  * [![Streamlit][StreamlitBadge]][Streamlit-url]
  * 
  * ...
* [![telegram][telegram]][telegram-url]
* ...

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

## Участники проекта

<div align="center">

|          Имя          |      tg       |GitHub            |
|-----------------------|---------------|------------------|
|**Ева Неудачина (куратор)**|[**@cocosinca**](http://t.me/cocosinca)  |[**neudachina**](https://github.com/neudachina)|
|        Богдан         |[@ghfsbdns](http://t.me/ghfsbdns)            |[HerrVonBeloff](https://github.com/HerrVonBeloff)|
|        Антон          |[@beda3113](http://t.me/beda3113)            |[Beda3113](https://github.com/Beda3113)|
|      Александр        |[@AlexAetM](http://t.me/AlexAetM)            |[GandlinAlexandr](https://github.com/GandlinAlexandr)|
|        Роман          |[@OleynikovRoman](http://t.me/OleynikovRoman)|[roleynikov](https://github.com/roleynikov)|

</div>

## Данные
Первичные данные были взяты из свободного доступа на сайте [Hugging Face](https://huggingface.co/datasets/iamkaikai/amazing_logos_v4/viewer/default/train?p=3972). Датасет представлен в формате `parquet` и содержит изображения логотипов и их текстовое описание на английском языке. В датасете содержится 397 000 объектов.

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

## EDA
Анализ первичных данных проводили по двум направлениям: анализ текста описания логотипов и анализ информации об изображениях. Для текстовых данных строили графики распределения по символам, словосочетаниям, а также использовали график облако слов. В случае графических данных изучали высоту, ширину изображений, их распределения в выборке, подсчитывали количество разных форм изображений и др.

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

## Baseline

В качестве Baseline была обучена генеративно-состязательная модель GAN.

### Обучающие данные

В качестве обучающих данных слыужили последние 100 000 изображений и их описаний из датасета [Hugging Face](https://huggingface.co/datasets/iamkaikai/amazing_logos_v4/viewer/default/train?p=3972). Такое решение было мотивировано двумя причинами. Во-первых, нехватка вычислительных мощностей для обучения модели на полном датасете. Во-вторых, последние изображения в датасете - наиболее разнообразны (в начале датасета более представлены чёрно-белые изображения). Перел обучением изображения конертироались в RGB, сжимались до размеров $28\times28$, а также подвергались нормированию, в результате чего на выходе получался нормированный тензор формы `(3,28,28)`. Текстовое описание подвергалось токенизации и векторизации посредством инмтрументов библитоеки [Spacy](https://spacy.io/) и загруженной из неё англоязычной предобученной NLP-модели `en_core_web_md`. В дальнейшем векторы описания усреднялись. На ыходе описание представляло собой вектор длинной `(300)`.

### Характеристики модели

Модель GAN, как и ожидается, состоит из двух нейросетей: генератор и дискриминатор. И генератор, и дискриминатор представляют собой свёрточные нейросети. Использовали функции активации ReLU, LeakyReLU, Сигмоиду и Гиперболический тангенс, а также модификацию градиентного спуска ADAM. В генератор поступал вектор сгенерированного шума размером `100`, объединённый с вектором описания. В дискриминатор же тензоры изображений (как сгенерированных, так и реальных) также поступали объединённые со своим текстовым описанием. В качестве метрик качества модели были использованы Fréchet inception distance (FID) и Inception Score (IS). Обе метрики рассчитывались в процессе обучения, чтобы можно было отслеживать динамику. Метрики рассчитывались каждые 5 эпох. При этом для IS рассчитывалось среднее значение по батчу.

### Результаты

Потери для генератора и дискриминатора представлены на графике.

<div align="center">
  <img src="https://github.com/user-attachments/assets/5f9f4878-81cb-4eec-97eb-b663bf4ab77e" alt="Описание изображения">
  <p><i>Значения потерь</i></p>
</div>

Как и ожидается, с ходом эпох потери для обеих нейросетей снижаются. Заметен большой, но кратковременный скачок для генератора в первой эпохе, что связано с началом обучения. В дальнейшем никаких крупных скачков потерь не наблюдается.

Ниже представлен график для метрик.

<div align="center">
  <img src="https://github.com/user-attachments/assets/49b05c68-4fcd-45f4-b028-036606148dcf" alt="Описание изображения">
  <p><i>Метрика FID</i></p>
</div>

Значение метрики FID было большим, но при этом заметен тренд на уменьшнение, что свидетельствует о медленном улучшении модели. Однако, значение IS на всём протяжении обучения оставалось равным единице. Такое поведение метрики пока остаётся для нас непонятным: является ли это обычным поведением при услолвиях обучения, подобных нашим (малое количество эпох, относительно простая структура нейросетей, небольшой размер изображений), или же в наших расчётах содержится ошибка. Нам следует прояснить этот вопрос в дальнейшем.

Ниже представлен пример генераций изображений с одними и теми же векторами описаний на разных этапах обучения.

<div align="center">
  <img src="https://github.com/user-attachments/assets/11c423c8-6d2a-4c2e-b24b-5b8417a2c3cb" alt="Описание изображения">
  <p><i>Прогресс в ходе обучения</i></p>
</div>

Так как обучение длилось недолго - всего 15 эпох, а генератор и дискриминатор бьыли устроены достаточно просто, что также было ограничено имеющимися вычислительными мощностями, качество модели оставляет желать лучшего. Тем не менее, на этом примере мы ознакомились с GAN как простой генеративной моделью на практике.

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

## Сервис


### API

### Streamlit

Streamlit-приложение выполняет следующие функции: проверяет работу нашей модели, генерируя четыре изображения по текстовому запросу пользователя; выдаёт краткую информацию о модели и её обучении; позволяет загружать датасет, аналгогичный тому, который использовался при обуучении для его анализа. Любую из секций в приложении можно свернуть и развернуть, чтобы пользователю не приходилось прокручивать длинную ленту web-приложения. При нажатии любой из кнопок в приложении осталльные секции не обновляются, не сворачиваются и не откатываются, сохраняя предыдущее состояение, чтобы пользователю не приходилось перезапускать каждую из них заново.

#### Генерация изображений

Осуществляется после ввода текстового описания изображения в специальное поле и нажатия соотвествующей кнопки.

#### Краткая информация о модели

Даёт краткую справку о модели, выводит интерактивные графики для значений потерь и метрики FID, даётся ссылка на репозиторий для ознакомления с боле подробной информацией и кодом проекта. Пользователь мложет выбрать цветовую тему для интерактивных графиков из выпаждающего меню в данном разделе.

#### Загрузка датасета

Позволяет загружать датасет формата `parquet`, аналогичный тому, что использовался при обучении моделей, в Streamlit-приложение для анализа данных. Датасет должен состоять минимум из двух колонок с названиями `image` и `text`, в которых ханятся изображения и их текстовое описание соответственно. При этом изображение в каждом объекте должно быть представлено в виде словаря с ключом `'bytes'`, в значении которого содержится изображение в байтовом представлении. 

<div align="center">
  
##### Описание формата датасета

| | image | text |
|-|-------------|-------------|
|456| {'bytes': b'\x89PNG\r\n\x1a\n\x00\...'}    | Simple elegant logo for Concept, love orange ... |

</div>

После загрузки датасета появляется кнопка `Получите случайный элемент датасета`, при нажатии на которую выводится случайный логотип из датасета и его текстовое описание с указанием индекса.

Пример датасета, который представляет собой срез обучающей выборки для нашей модели, представленный двумя тысячами его последних элеметов, можно скачать по [ссылке](https://drive.google.com/file/d/1BiUi9TOVgIjEggFQHb9d49Dp-z0pgIvI/view?usp=sharing). Ссылка на пример датасета и описание требований к нему даны и в самом web-приложении.

#### EDA в web-приложении

Выводит анализ данных загруженного датасета. Кнопка `EDA` появляется только после загрузки датасета в Streamlit-приложение. Выводит следующие графики:
* Анализ текстовых данных
  * Облако слов
  * Гистограмма распределения длин описаний (в символах)
  * Boxplot для длин описаний
  * Гистограмма количества эпитетов в описании
  * Boxplot для количества эпитетов
* Анализ изображений
  * Соотношение RGB и чёрно-белых логотипов (круговая диаграмма)
  * Высота изображений (гистограмма)
  * Ширина изображений (гистограмма)
  * Соотношение сторон (гистограмма)
  * Соотношение сторон (без квадратных изображений, гистограмма)
  * Количество пикселей (гистограмма)

Данныне графики дают начальное представление о данных и позволяют наметить первые шаги к их предобработке. Каждый график, кроме облака слов, является интерактивным. Для каждой гистограммы можно настроить количество бинов для удобства восприятия пользователя. Помимо этого, предоставляется выподающее меню, позволяющее выбрать цветовую тему для интерактивных графиков из списка тем библиотеки `plotly`.

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>


<!-- Раздел ссылок на сайты и миниатюры -->

[Python-url]: https://python.org/
[Python.org]: https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue

[Pandas-url]: https://pandas.pydata.org/
[Рandas.pydata.org]: https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white

[Numpy-url]: https://numpy.org/
[Numpy.org]: https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white

[Colab-url]: https://colab.research.google.com/
[Colab]: https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252

[scikit-learn-url]: https://scikit-learn.org/
[scikit-learn]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white

[telegram-url]: https://telegram.org/
[telegram]: https://img.shields.io/badge/Telegram-grey?style=for-the-badge&logo=telegram

[AIOgram-url]: https://aiogram.dev/
[AIOgram]: https://img.shields.io/badge/AIOgram-blue?style=for-the-badge&logo=aiogram

[Kaggle-url]: https://www.kaggle.com/
[Kaggle]: https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white

[license-shield]: https://img.shields.io/github/license/HerrVonBeloff/AI-YP_24-team-42.svg?style=for-the-badge
[license-url]: https://github.com/HerrVonBeloff/AI-YP_24-team-42/blob/main/LICENSE

[Matplotlib-url]: https://matplotlib.org/
[Matplotlib.org]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black

[Spacy-url]: https://spacy.io/
[Spacy]: https://img.shields.io/badge/-spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white

[Pytorch-url]: https://pytorch.org/
[Pytorch]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white

[Seaborn-url]: https://seaborn.pydata.org/
[Seaborn-badge]: https://img.shields.io/badge/Seaborn-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=blue

[StreamlitBadge]: https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white
[Streamlit-url]: https://streamlit.io/

