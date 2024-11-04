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
  </ol>
</details>

## Описание проекта

Данный проект предназначен для автоматической генерации уникальных логотипов на основе текстовых описаний, предоставленных пользователем. Используя классичиеские ML-подходы и нейросетевые модели, проект будет создавать графические изображения, соответствующие заданным ключевым словам, указанным в описании.

## Технологии

Для реализации проекта используются следующие технологии:

* [![Colab][Colab]][Colab-url]
* [![Python][Python.org]][Python-url]
  * [![AIOgram][AIOgram]][AIOgram-url]
  * [![Numpy][Numpy.org]][Numpy-url]
  * [![Pandas][Рandas.pydata.org]][Pandas-url]
  * [![scikit-learn][scikit-learn]][scikit-learn-url]
  * ...
* [![telegram][telegram]][telegram-url]
* ...

## Участники проекта

|          Имя          |      tg       |GitHub            |
|-----------------------|---------------|------------------|
|**Ева Неудачина (куратор)**|[**@cocosinca**](http://t.me/cocosinca)  |[**neudachina**](https://github.com/neudachina)|
|        Богдан         |[@ghfsbdns](http://t.me/ghfsbdns)            |[HerrVonBeloff](https://github.com/HerrVonBeloff)|
|        Антон          |[@beda3113](http://t.me/beda3113)            |[Beda3113](https://github.com/Beda3113)|
|      Александр        |[@AlexAetM](http://t.me/AlexAetM)            |[GandlinAlexandr](https://github.com/GandlinAlexandr)|
|        Роман          |[@OleynikovRoman](http://t.me/OleynikovRoman)|[roleynikov](https://github.com/roleynikov)|

## Данные
Первичные данные были взяты из свободного доступа на сайте [Hugging Face]([https://huggingface.co/datasets/logo-wizard/modern-logo-dataset](https://huggingface.co/datasets/iamkaikai/amazing_logos_v4/viewer/default/train?p=3972)). Датасет представлен в формате `parquet` и содержит изображения логотипов и их текстовое описание на английском языке. В датасете содержится 397 000 объектов.

## EDA
Анализ первичных данных проводили по двум направлениям: анализ текста описания логотипов и анализ информации об изображениях. Для текстовых данных строили графики распределения по символам, словосочетаниям, а также использовали график облако слов. В случае графических данных изучали высоту, ширину изображений, их распределения в выборке, подсчитывали количество разных форм изображений и др.


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

