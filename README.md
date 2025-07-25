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
        <li><a href="#docker">Docker</a></li>
        <li><a href="#api">API</a></li>
        <li><a href="#streamlit">Streamlit</a></li>
            <ul><li><a href="#генерация-изображений">Генерация изображений</a></li></ul>
            <ul><li><a href="#краткая-информация-о-модели">Краткая информация о модели</a></li></ul>
            <ul><li><a href="#загрузка-датасета">Загрузка датасета</a></li></ul>
            <ul><li><a href="#eda-в-web-приложении">EDA в web-приложении</a></li></ul>
        <li><a href="#реализация-логирования-в-приложении">Реализация логирования в приложении</a></li>
            <ul><li><a href="#доступ-к-сервисам-elk">Доступ к сервисам ELK</a></li></ul>
            <ul><li><a href="#логирование-в-приложении">Логирование в приложении</a></li></ul>
        <li><a href="#развертывание-сервера-на-платформе-selectelru">Развертывание сервера на платформе Selectel.ru</a></li>
            <ul><li><a href="#этапы-развертывания-сервера">Этапы развертывания сервера</a></li>
                <ul><li><a href="#запуск-сервера">Запуск сервера</a></li></ul>
                <ul><li><a href="#установка-необходимых-инструментов">Установка необходимых инструментов</a></li></ul>
                <ul><li><a href="#клонирование-проекта-из-git-на-сервер">Клонирование проекта из Git на сервер</a></li></ul>
                <ul><li><a href="#запуск-docker-билда">Запуск Docker-билда</a></li></ul>
                <ul><li><a href="#готово">Готово!</a></li></ul>
            </ul>
        </ul>
    <li>
      <a href="#улучшение-baseline">Улучшение Baseline</a>
    </li>
        <ul>
        <li><a href="#направления-улучшения">Направления улучшения</a></li>
        <li><a href="#результаты-улучшения">Результаты улучшения</a></li>
        </ul>
   <li>
    <a href="#дальнейшее-улучшение-модели-и-эксперименты">Дальнейшее улучшение модели и эксперименты</a>
  </li>
    <ul>
        <li><a href="#результаты-экспериментов">Результаты экспериментов</a></li>
        </ul>
    <li>
    <a href="#использование-диффузионной-модели-и-гибридный-подход">Использование диффузионной модели и гибридный подход</a>
  </li>
    <ul>
        <li><a href="#пример-работы">Пример работы</a></li>
        </ul>
    <ul>
        <li><a href="#ограничения">Ограничения</a></li>
        </ul>
    <ul>
        <li><a href="#stable-diffusion-v15--lora">Stable Diffusion v1.5 + LoRA</a></li>
        </ul>
    <li>
    <a href="#ai-logo-generator-system">AI Logo Generator System</a>
  </li>
     <ul>
        <li><a href="#структура">Структура</a></li>
        <ul><li><a href="#основные-директории">Основные директории</a></li></ul>
        </ul>
    <ul>
        <li><a href="#технологический-стек">Технологический стек</a></li>
        </ul>
    <ul>
        <li><a href="#быстрый-старт">Быстрый старт</a></li>
             <ul><li><a href="#установка">Установка</a></li></ul>
     </ul>
    <ul>
        <li><a href="#ключевые-возможности">Ключевые возможности</a></li>
             <ul><li><a href="#генерация-логотипов">Генерация логотипов</a></li></ul>
              <ul><li><a href="#особенности-системы">Особенности системы</a></li></ul>
      </ul>
    </ol>
</details>


## Описание проекта

Данный проект предназначен для автоматической генерации уникальных логотипов на основе текстовых описаний, предоставленных пользователем. Используя классичиеские ML-подходы и нейросетевые модели, проект будет создавать графические изображения, соответствующие заданным ключевым словам, указанным в описании.

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

## Технологии

Для реализации проекта используются следующие технологии:

* [![Colab][Colab]][Colab-url]
* [![Docker][DockerBadge]][Docker-url]
* [![Kaggle][Kaggle]][Kaggle-url]
* [![Python][Python.org]][Python-url]
  * [![Matplotlib][Matplotlib.org]][Matplotlib-url]
  * [![Numpy][Numpy.org]][Numpy-url]
  * [![Pandas][Рandas.pydata.org]][Pandas-url]
  * [![Pytorch][Pytorch]][Pytorch-url]
  * [![scikit-learn][scikit-learn]][scikit-learn-url]
  * [![Spacy][SpaCy]][Spacy-url]
  * [![Seaborn][Seaborn-badge]][Seaborn-url]
  * [![Streamlit][StreamlitBadge]][Streamlit-url]

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

## Участники проекта

<div align="center">

##### Список участников проекта
  
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
  <img src="https://github.com/user-attachments/assets/5f9f4878-81cb-4eec-97eb-b663bf4ab77e" alt="Значения потерь">
  <p><i>Значения потерь</i></p>
</div>

Как и ожидается, с ходом эпох потери для обеих нейросетей снижаются. Заметен большой, но кратковременный скачок для генератора в первой эпохе, что связано с началом обучения. В дальнейшем никаких крупных скачков потерь не наблюдается.

Ниже представлен график для метрик.

<div align="center">
  <img src="https://github.com/user-attachments/assets/49b05c68-4fcd-45f4-b028-036606148dcf" alt="Описание изображения">
  <p><i>Метрики качества</i></p>
</div>

Значение метрики FID было большим, но при этом заметен тренд на уменьшнение, что свидетельствует о медленном улучшении модели. Однако, значение IS на всём протяжении обучения оставалось равным единице. Такое поведение метрики пока остаётся для нас непонятным: является ли это обычным поведением при услолвиях обучения, подобных нашим (малое количество эпох, относительно простая структура нейросетей, небольшой размер изображений), или же в наших расчётах содержится ошибка. Нам следует прояснить этот вопрос в дальнейшем.

Ниже представлен пример генераций изображений с одними и теми же векторами описаний на разных этапах обучения.

<div align="center">
  <img src="https://github.com/user-attachments/assets/11c423c8-6d2a-4c2e-b24b-5b8417a2c3cb" alt="Прогресс в ходе обучения">
  <p><i>Прогресс в ходе обучения</i></p>
</div>

Так как обучение длилось недолго - всего 16 эпох, а генератор и дискриминатор бьыли устроены достаточно просто, что также было ограничено имеющимися вычислительными мощностями, качество модели оставляет желать лучшего. Тем не менее, на этом примере мы ознакомились с GAN как простой генеративной моделью на практике.

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

## Сервис

### Docker

Для локального использования и тестирования приложения необхходимо склонировать репозиторий, затем локально в папке с репозиторием выполнить следующую команду:
```
sudo docker-compose up -d --build
```
После того, как данная команда будет выполнена, необходимо выполнить:
```
sudo docker-compose logs
```
В ходе выполнения команды будет выведена информация с ссылками на streamlit приложение:

```
streamlit   |  You can now view your Streamlit app in your browser.
streamlit   |
streamlit   |  Local URL: http://localhost:8501
streamlit   |  Network URL: http://172.19.0.3:8501
streamlit   |  External URL: http://95.24.77.148:8501
```
Документацию FastApi можно будет посмотреть по адресу: <http://localhost:8000/docs>

   
### API
Эта часть проекта представляет собой API (Application Programming Interface), который выступает в роли прослойки между Streamlit-интерфейсом и сервером. Основная задача API — принимать запросы от пользователя, обрабатывать данные на сервере и возвращать результат обратно в приложение. В данном случае API реализует одну ключевую функцию: генерацию логотипа по текстовому описанию

1. Принимает запросы от приложения
   - Пользователь через Streamlit-приложение отправляет POST-запрос с текстовым описанием логотипа
   - API принимает этот запрос и извлекает данные для дальнейшей обработки
3. Обрабатывает данные на сервере
   - Текстовое описание обрабатывается
   - Данные передаются в генеративную модель, которая генерирует логотип 
5. Возвращает результат приложению
   - API возвращает сгенерированное изображение в формате JSON, которое может быть отображено в Streamlit-приложении
   - Streamlit-приложение получает изображение и отображает его пользователю

### Streamlit

Streamlit-приложение выполняет следующие функции: проверяет работу нашей модели, генерируя четыре изображения по текстовому запросу пользователя; выдаёт краткую информацию о модели и её обучении; позволяет загружать датасет, аналгогичный тому, который использовался при обуучении для его анализа. Любую из секций в приложении можно свернуть и развернуть, чтобы пользователю не приходилось прокручивать длинную ленту web-приложения. При нажатии любой из кнопок в приложении осталльные секции не обновляются, не сворачиваются и не откатываются, сохраняя предыдущее состояение, чтобы пользователю не приходилось перезапускать каждую из них заново.

#### Генерация изображений

Осуществляется после ввода текстового описания изображения в специальное поле и нажатия соотвествующей кнопки.

<div align="center">
  <img src="https://github.com/user-attachments/assets/0dfc9f6f-a3b2-4960-825f-642a4fcc36cf" alt="Процесс генерации">
  <p><i>Процесс генерации</i></p>
</div>

#### Краткая информация о модели

Даёт краткую справку о модели, выводит интерактивные графики для значений потерь и метрики FID, даётся ссылка на репозиторий для ознакомления с боле подробной информацией и кодом проекта. Пользователь мложет выбрать цветовую тему для интерактивных графиков из выпаждающего меню в данном разделе.

<div align="center">
  <img src="https://github.com/user-attachments/assets/abfe124b-e84c-46a9-a6b6-8261dc63fffa" alt="Вывод краткой информации о модели">
  <p><i>Вывод краткой информации о модели</i></p>
</div>

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

<div align="center">
  <img src="https://github.com/user-attachments/assets/50d38fe1-a8ea-4158-bb76-d57a5c16c9cf" alt="Процесс загрузки датасета и получения объекта">
  <p><i>Процесс загрузки датасета и получения объекта</i></p>
</div>

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

<div align="center">
  <img src="https://github.com/user-attachments/assets/39b6764f-a032-4a1f-997f-db6727747d87" alt="Демонстрация вывода EDA">
  <p><i>Демонстрация вывода EDA</i></p>
</div>

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

### Реализация логирования в приложении

Логирование — это ключевой процесс, который включает запись событий, сообщений и ошибок, происходящих в приложении или системе. В данном проекте реализована система логирования как для основного приложения, так и для веб-приложения, использующего Streamlit для анализа данных. Логирование выполнено с использованием стека ELK (Elasticsearch, Logstash, Kibana), что позволяет эффективно собирать, обрабатывать и визуализировать логи приложения.

* Составляющие стека ELK
  
  * Elasticsearch: Это сервис для хранения и поиска данных логов. Он обеспечивает быстрый доступ к информации и возможность выполнения сложных запросов, что делает его идеальным для работы с большими объемами логов.
  * Logstash: Инструмент для обработки логов, который собирает данные из различных источников и отправляет их в Elasticsearch для хранения и анализа. Logstash позволяет обрабатывать логи в реальном времени, обеспечивая высокую гибкость и масштабируемость.
  * Kibana: Веб-интерфейс для визуализации данных, хранящихся в Elasticsearch. Kibana позволяет создавать интерактивные дашборды и графики, что облегчает анализ логов и мониторинг состояния приложения.
   
* Преимущества использования стека ELK

  Интеграция компонентов ELK позволяет сервисам взаимодействовать друг с другом через общую сеть, что улучшает архитектуру приложения и упрощает процесс сбора и анализа логов. Это обеспечивает более глубокое понимание работы приложения и упрощает выявление проблем.

#### Доступ к сервисам ELK

[Elasticsearch](http://46.161.52.173:9200)

[Elastic](http://46.161.52.173:5601)


#### Логирование в приложении
В проекте реализовано полноценное логирование с механизмом ротации логов. Это позволяет управлять объемом хранимых логов и предотвращает переполнение дискового пространства.

Реализация системы логирования с использованием стека ELK значительно улучшает возможности мониторинга и анализа работы приложения. Это позволяет разработчикам быстро реагировать на возникающие проблемы и поддерживать высокое качество работы приложения.

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>


### Развертывание сервера на платформе [Selectel.ru](https://my.selectel.ru/vpc)

Проект был успешно развернут на созданном сервере компании Selectel, предоставляющей облачные инфраструктурные сервисы и услуги дата-центров. Ниже представлены основные этапы развертывания сервера.

#### Этапы развертывания сервера

##### Разработка онлайн-сервера в конструкторе, подбор оптимальных параметров сервера.
Используя конструктор серверов от Selectel, был разработан сервер, соответствующий требованиям проекта.
 Проведен анализ и выбор оптимальных параметров сервера с учетом соотношения цена/качество. Обратите внимание, что реализация сервера является платной.

##### Запуск сервера
После завершения настройки параметров, сервер был запущен и готов к дальнейшим действиям.

##### Установка необходимых инструментов
• Установка Git:
```
sudo apt update
sudo apt install git    
```
 • Установка Docker:
```
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update   
``` 
##### Клонирование проекта из Git на сервер 
``` 
git clone https://github.com/HerrVonBeloff/AI-YP_24
```
##### Запуск Docker-билда
``` 
cd /AI-YP_24
docker-compose up -d --build  
```
##### Готово!

Теперь к серверу можно подключиться по следующему адресу:

[http://46.161.52.173:8501/](http://46.161.52.173:8501/)


<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

## Улучшение Baseline

### Направления улучшения

Были выбраны следующие пути для улучшения Baseline:
* Используется более сложная архитектура cGAN с свёрточными слоями. Изначально в Baseline нейросети имели очень простую структуру. Вероятно, это одна из причин, по которой потери для обеих быстро выходили на плато.
* Увеличение размера изображений до (32, 32).

### Результаты улучшения

Потери для генератора и дискриминатора представлены на графике.

<div align="center">
  <img src="https://github.com/user-attachments/assets/0795e646-fd32-4831-8e78-d0235135733a" alt="Значения потерь">
  <p><i>Значения потерь</i></p>
</div>

Особенности динамики потерь улучшенного Baseline:
* Вначале loss генератора быстро уменьшается, далее происходит рост, который замедляется.
* У дискриминатора loss сначала быстро возрастает, потом чуть медленнее уменьшается и держится в основном около нуля. 
* Это говорит о том, что обучение генератора не поспевает за обучением дискриминатора.
* У генератора более нестабильный и изменчивый loss, чем у дискриминатора.

<div align="center">
  <img src="https://github.com/user-attachments/assets/16f8e8bb-54d5-4998-bd2e-73ede7982a09" alt="Прогресс в ходе обучения">
  <p><i>Обучение на последних эпохах</i></p>
</div>

Модель демонстрирует значительный прогресс. Изображения от эпохи к эпохе в каждой ячейке схожи, что говорит о том, что модель действительно принимает во внимание текстовые метки. Также заметно большее разнообразие сложных форм и цветовых сочетаний. До улучшения Baseline мог распознавать лишь отдельные цвета и простые фигуры (круг, квадрат). Обновлённая версия не только лучше различает сложные формы, но и генерирует более детализированные изображения, даже без учёта различий в разрешении.

Ниже представлен график для метрик.

<div align="center">
  <img src="https://github.com/user-attachments/assets/b96b3178-084f-4aa8-a316-c1cb87558811" alt="Описание изображения">
  <p><i>Метрики качества</i></p>
</div>

* Динамика метрик и их величины в случае Baseline и улучшенной модели различаются.
* Метрика FID очень резко уменьшается, а потом немного растет. Это может отражать нестабильность генератора. Несмотря на это, значения метрики сильно улучшились  по сравнению с изначальной моделью.
* Метрика IS растет, что указывает на улучшение качества модели.

Таким образом улучшение Baseline заметно как визуально по качеству генераций, так и по метрикам. 


<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

## Дальнейшее улучшение модели и эксперименты

В соотвествии с пожеланиями куратора было решено продолжать работу с нашей текущей моделью GAN. Эксперименты и работа с моделью включали в себя следующее:

* Было увеличено разрешение выходного изображения до $256\times256$ пикселей.
* Все conv transpose слои были заменены на сочетание up scale + обычных conv слоёв.
* Добавление нормализации по батчам и дропаут для уменьшения переобучения.
* Добавлена возможность определять количество циклов обучения генератора на цикл обучения дискриминатора.
* Внедрен label smoothing - сглаживание меток, чтобы прредотвратить черезмерную уверенность модели в прогнозах.
* В обработке текста используется spatial-структура, создаётся пространственная сетка признаков.
* Добавлен еще один слой на вход дискриминатора, чтобы он следил за разнообразием генераций.
* Добавлено экспоненциальное сглаживание весов генератора (EMA), чтобы получить более стабильную и качественную версию модели на inference.
* Подключено логирование через wandb.
* Добавлена метрика CLIP.
* Эксперименты с функцией потерь. Помимо уже использованноый бинарной кросс-энтропии (`BCELoss`) использована также:
  * Комбинированная из `Sigmoid layer` и `BCELoss` функция потерь (`BCEWithLogitsLoss`).
  * `wasserstein_loss` (минимизация расстояние между реальным и сгенерированным изображением) для уменьшения mode collapse.
  * Изменение функции потерь в соответствии с WGAN-GP.
  * CLIP-guided функция потерь для усиления смысла.
* Добавлен шедулер для `learning rate`.
* Реализован `weight clipping` для гарантии липшицевости дискриминатора - принудительное ограничение весов.
* WGAN-GP вместо `weight clipping` - менее жесткое ограничение для гарантии липшицевости с использованием градиентного штрафа (gradient penalty).

Все эксперименты с кодом содержатся в файле [GAN_experiments.ipynb](https://github.com/HerrVonBeloff/AI-YP_24-team-42/blob/main/GAN_experiments.ipynb).

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

### Результаты экспериментов

Все изменения были введены постепенно после увеличения размера изображения. Ввиду ограничения по времени и ресурсам, проводилось текстовое обучение на всём датасете в течение 2-10 эпох (в случаях очевидного провала модели или чрезмерной длительности расчётов останавливались на 2 эпохах). Было замечено, что дискриминатор почти сразу становился сильнее генератора, генератор в свою очередь застревал на некоем паттерне, который был склонен повторять для разных изображений (mode collapse). По этой причине были попытки учесть разнообразие генераций, ввод дропаута, увеличение циклов обучения для генератора и т.д. Однако ни один из подходов никак не мог решить этой проблемы. Потери для генератора были склонны расти или рано выходить на плато, а метрики качества обычно выходили на плато и колебались вокруг одного значения. Попытки изменить обработку текстовых данных также не привели к значительным улучшениям.

Основные характеристики генераций практически не менялись: отсутствие чётких паттернов и цветовых гамм - (на разных генерациях для одного описания цвета сильно менялись), однообразность генераций для разных описаний. Иногда были видны различия между текстовыми логотипами и собственно логотипами из абстрактных и иных фигур.

Метрики, как правило, выходили на плато в диапозоне одних и тех же значений на протяжении всех экспериментов. Метрики достигали следующих уровней:

* IS колебался около 1.8.
* CLIP-T - около 0.18-0.20.
* CLIP-I - около 0.5-0.6.
* FID - в пределах 1000–1200.

Важно отметить, что по сравнению с более простой ранней реализацией IS и FID (ранее мы не использовали CLIP) стали значительно хуже (ранее $\mathrm{IS}\approx0.20$, $\mathrm{FID}\approx$ 3.3). Такое постоянство метрик дополнительно показывает, что своими улучшениями мы не затронули главную первопричину низкого качества модели. Также метрики указывают на слабое влияние текста (значения CLIP-T). И увы, даже CLIP-guided loss не помог решить эту проблему. В целом метрики CLIP были аномальными. Было проверено на шуме, при сраавнении с которым CLIP-I была 0.5 на малых разрешениях (на изображениях низкого разрешения 0.8). Вероятно, модель clip очень плохо работает на малых разрешениях. При повышении разрешения CLIP-I снизилась до 0.5, что более адекватно, но всё же кажется завышенным.

Предположений о том, как это исправить, ещё много. Вот ряд гипотез:

* Возможно, разрешение $256\times256$ слишком велико для относительно небольших входных изображений (в датасете преобладает $180\times180$),
* Возможно, нормализация по батчам мешает обучению пространственной текстовой сетки,
* Дискриминатор мог быть переусложнён и "сломал" баланс,
* Возможно, стоит иначе настроить коэффициент $\lambda$ в CLIP-guided функции потерь.

Также стоит отметить, что после внедрения сложных функций потерь и CLIP-метрик значительно увеличилось время обучения: даже на GPU Kaggle лимит в 12 часов позволил провести лишь 2 эпохи. Интерпретация поведения сложных функций потерь оказалась нетривиальной - диапазон значений и поведение отличались от стандартных BCE-лоссов, что также затрудняло отладку.

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

## Использование диффузионной модели и гибридный подход 
Диффузионные модели – это современный инструмент для генерации изображений, который работает по принципу постепенного преобразования шума в четкую картинку. Они позволяют получать высококачественные результаты, однако требуют больших вычислительных мощностей и времени.

### Чтобы объединить скорость GAN и эффективность диффузионной модели, мы использовали гибридную GAN-диффузионную модель. 

###  Общий принцип работы
система использует каскадный подход, сочетающий преимущества трех технологий:
1. **Conditional GAN** (cGAN) - быстрая генерация базовых вариантов
2. **Real-ESRGAN** - улучшение качества изображения
3. **Stable Diffusion** - финальная стилизация и детализация

###  Пример работы 
!['Black'](https://github.com/HerrVonBeloff/AI-YP_24-team-42/blob/main/hybrid%20GAN-diffusion%20model/12.gif)
### Преимущества:

* Быстрее чистой диффузии – за счет сокращения числа шагов.
* Готова к использованию – не требует сложных доработок.

### Ограничения:

* Нужна предобученная cGAN.
* Качество итогового изображения зависит от выходных данных GAN.
* Менее гибкая, чем "чистая" диффузия.


### Stable Diffusion v1.5 + LoRA
Модель использует технику LoRA (Low-Rank Adaptation) для тонкой настройки, что позволяет обучать только небольшую часть параметров, не изменяя оригинальные веса. LoRA вставляет в модель дополнительные обучаемые адаптеры (тонкие слои) внутрь определённых блоков (обычно `attention`), не изменяя при этом сами веса модели. Это позволяет использовать меньше памяти, обучать модель быстрее, сохранять качество базовой модели. Stable Diffusion использует UNet как основной генератор, где текстовые подсказки подаются как `encoder_hidden_states` и управляют генерацией на всех слоях через `cross-attention`. Обучение включает:

* Кэширование латентов через VAE
* Токенизация описаний и генерация text embeddings
* Добавление шума в латенты (сжатые представления изображений) по диффузионному расписанию 
* LoRA-модель (UNet) предсказывает шум
* Loss = MSE между предсказанным и настоящим шумом
* Backprop + AdamW + GradScaler
* Логирование и валидация после каждой эпохи (`wandb`)

Применялись следующие метрики:
|Метрика|Что показывает|Интерпретация|
|-------|--------------|-------------|
|FID|Сравнение статистик реального и сгенерированного изображений|Чем ниже, тем лучше|
|IS|Качество и разнообразие генераций|Чем выше, тем лучше|
|LPIPS|Перцептуальное различие (насколько два изображения отличаются с точки зрения человеческого восприятия)|Чем ближе к 0, тем лучше|

**Генерации**. Как правило, до адаптации для генерации логотипов, модель выдавала далёкие от логотипов изображения. Обычно это были реалистичные изображения чего-либо. После адаптации модель уже выдавала нечто смутно напоминающее логотипы, иногда с внедрением отголосков фотореалистичных изображений.

<div align="center">
  <img src="https://github.com/user-attachments/assets/88d1b316-5794-4d49-80bf-bd4e57280c62" alt="Генерация изображений по запросу A futuristic logo with glowing lines and abstract shapes после адаптации модели" width="1000">
  <p><i>Пример генерации по запросу **A futuristic logo with glowing lines and abstract shapes** до адаптации модели</i></p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/7f38e181-4e32-4759-9150-9c1213a96244" alt="Генерация изображений по запросу A futuristic logo with glowing lines and abstract shapes после адаптации модели" width="1000">
  <p><i>Примеры некоторых генераций по запросу **A futuristic logo with glowing lines and abstract shapes** после адаптации модели</i></p>
</div>

Ниже представлена динамика потерь.

<div align="center">
  <img src="https://github.com/user-attachments/assets/093045fe-5135-4fcc-ab38-8d31f3cce4b0" alt="Потери диффузия" width="1000">
  <p><i>Потери на валидации и трейне</i></p>
</div>

**Loss validation**. Колебания от 0.21 до 0.35, нет чёткой нисходящей динамики, наблюдаются скачки (на эпохах 11, 14) — это признак нестабильной генерации. Неплохой, но не идеальный результат. Возможна переобученность или плохая генерализация.

**Average loss train**. Хорошо видно плавное снижение train loss до эпохи 10. После - появляется шум. Это говорит о том, что модель всё же учится, хотя после середины обучающей динамики - возможный выход на плато.

Метрики представлены ниже.

<div align="center">
  <img src="https://github.com/user-attachments/assets/b1b671b1-d8a2-41ee-adac-4f31cb0ea24d" alt="Метрики диффузия"  width="1000">
  <p><i>Метрики</i></p>
</div>

**IS**. По какой-то причине метрикм `IS` не сработало корректно и оказалась аномально низкой. Или же, что также вероятно, модель генерирует одинаковые изображения с точки зрения Inception (низкое разнообразие).

**LPIPS**.  Колебания от 0.22 до 0.5, нет стабильной тенденции к снижению, всплески и падения говорят о нестабильности качества восприятия. Чем ближе к 0, тем лучше. А здесь он в среднем в диапазоне 0.3–0.4, что средне.

**FID**. FID колеблется от 430 до 580, нет устойчивого улучшения (снижения FID), высокие значения. Данные слишком разные от предобученной базы SD либо модель пока не научилась стабильной генерации.

В целом на не хватило времени и ресурсов для дальнейших экспериментов. Несмотря на то, что эта модель в принципе требует меньше данных и ресурсов для обучения.

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

## AI Logo Generator System

Генерация логотипов с использованием современных технологий искусственного интеллекта.

!['Black'](https://github.com/HerrVonBeloff/AI-YP_24-team-42/blob/main/hybrid%20GAN-diffusion%20model/11.gif)

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

##  Структура 

### Основные директории

```

hybrid GAN-diffusion model/
├── backend/ # Серверная часть
│ ├── api/ # API endpoints
│ ├── models/ # Модели ИИ
│ │ ├── cgan.py # Conditional GAN
│ │ ├── esrgan.py # Real-ESRGAN
│ │ └── diffusion.py # Stable Diffusion
│ ├── utils/ # Вспомогательные утилиты
│ └── main.py # Точка входа FastAPI
│
├── frontend/ # Клиентская часть
│ └── app.py # Streamlit интерфейс
│
├── data/ # Датасеты и примеры
├── weights/ # Веса моделей
├── lora/ # Пользовательские стили
└── output/ # Результаты генерации
```

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

##  Технологический стек

| Компонент       | Технология         | Описание |
|----------------|--------------------|----------|
| Генерация      | Conditional GAN    | Базовые варианты логотипов |
| Апскейлинг     | Real-ESRGAN        | Улучшение качества 4x |
| Стилизация     | Stable Diffusion   | Применение стилей + LoRA |
| Бэкенд         | FastAPI            | REST API сервер |
| Фронтенд       | Streamlit          | Веб-интерфейс |
| Аналитика      | Plotly/WordCloud   | Визуализация данных |


<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

##  Быстрый старт


<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

### Установка
```bash
git clone 
cd logo-generator
pip install -r requirements.txt
```
<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

##  Ключевые возможности

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

### Генерация логотипов
1. **Ввод текстового описания** - Просто укажите название и концепцию вашего бренда
2. **Генерация 5 вариантов cGAN** - Нейросеть создаст несколько уникальных вариантов
3. **Выбор лучшего варианта** - Выберите наиболее подходящую базовую версию
4. **Настройка стилизации**:
   - **Интенсивность** - Контроль силы применения стиля (0.3-0.9)
   - **Количество шагов** - Точность стилизации (10-100 шагов)
   - **LoRA стили** - Применение пользовательских стилей из папки `lora/`
  
<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

### Особенности системы
- **Автоматический апскейл** - Качественное увеличение разрешения через Real-ESRGAN
- **Поддержка пользовательских LoRA адаптеров** - Возможность загружать свои стили (.safetensors)
- **Оптимизация под разные GPU/CPU** - Работает как на мощных GPU, так и на обычных процессорах
- **Встроенный анализ датасетов (EDA)** - Инструменты для анализа ваших данных о логотипах
- **Экспорт в PNG** - Скачивание готовых логотипов в высоком качестве
- **Адаптивный интерфейс** - Удобное управление как через веб-интерфейс, так и через API


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

[DockerBadge]: https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white
[Docker-url]: https://www.docker.com/
