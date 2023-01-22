# Тестовое задание

Реализовать и обучить модель, позволяющую выполнять иерархическую семантическую сегментацию.

Обучать модель предлагается на датасете Pascal-part.

### Данные

Обучение модели предлагается производить на датасете Pascal-part.
Подготовленные данные и разметку можно загрузить по следующей [ссылке](https://drive.google.com/file/d/1unIkraozhmsFtkfneZVhw8JMOQ8jv78J/view?usp=sharing).

В папке `JPEGImages` находятся исходные изображения в формате jpeg. В папке `gt_masks` находятся маски сегментации в формате numpy.
Загрузить маски можно при помощи функции `numpy.load()`.

В датасете присутствуют 7 классов, обладающих следующей иерархической структурой (в скобках указан индекс класса):

```
├── (0) background
└── body
    ├── upper_body
    |   ├── (1) low_hand
    |   ├── (6) up_hand
    |   ├── (2) torso
    |   └── (4) head
    └── lower_body
        ├── (3) low_leg
        └── (5) up_leg
```

### Метрики

В качестве основной метрики предлагается использовать mean Intersection over Union (mIoU).
Для каждого из уровней вложенности предлагается вычислять метрику отдельно.
При этом background класс не учитывается при рассчете метрики.
Таким образом, для полученной модели необходимо оценить 3 значения метрик по следующим категориям:

* mIoU`<sup>`0`</sup>` - `body`
* mIou`<sup>`1`</sup>` - `upper_body`, `lower_body`
* mIoU`<sup>`2`</sup>` - `low_hand`, `up_hand`, `torso`, `head`, `low_leg`, `up_leg`

### Что будет оцениваться?

1. Оформление кода на github.
2. Оформление результатов.
3. Структура репозитория.
4. Соответствие решения тестовому заданию.
5. Любые релевантные теме мысли, идеи и соображения.
