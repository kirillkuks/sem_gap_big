# Intelligent Placer

По поданной на вход фотографии нескольких предметов на светлой горизонтальной поверхности и многоугольнику понимать, можно ли расположить одновременно все эти предметы на плоскости так, чтобы они влезли в этот многоугольник.

## Постановка задачи
* На вход алгоритму поступают фотография горизонтальной светлой поверхности, на которой находится какое-то количество предметов (из заранее извсестного набора), и многоугольник.

* Алгоритм определяет, возможно ли на поверхности, ограниченной многоугольником, расположить все предметы с фотографии (все предметы должны находиться на поверхности и не накладываться друг на друга).

* Ны выход алгоритм даёт ответ "да", если предметы можно поместить в многоугольник, ответ "нет" иначе

## Ограничение на входные данные
* Многоугольник задаётся фигурой, нарисованной тёмным маркером на белом листе бумаги, сфотагрофированной вместе с предметами.

* Требования к входным данным:
    * Разрешение фотографии 1200x1600. Напревление камеры перпендекулярно плоскости, на которой расположены предеметы.

    * На фотографии не должно быть пересвеченных и тёмных областей.

    * Предметы на фотографии не могут перекрывать друг друга.
    Высота предметов не более 4 см.

    * Многоугольник может иметь не более 10 вершин.

    * Каждый предмет может нахоться на фотографии не более чем в одном экземпляре.

## Фотографии предметов и поверхности
![Предмет 1](https://github.com/kirillkuks/sem_gap_big/blob/develop/images/objects/object1.jpg)

![Предмет 2](https://github.com/kirillkuks/sem_gap_big/blob/develop/images/objects/object2.jpg)

![Предмет 3](https://github.com/kirillkuks/sem_gap_big/blob/develop/images/objects/object3.jpg)

![Предмет 4](https://github.com/kirillkuks/sem_gap_big/blob/develop/images/objects/object4.jpg)

![Предмет 5](https://github.com/kirillkuks/sem_gap_big/blob/develop/images/objects/object5.jpg)

![Предмет 6](https://github.com/kirillkuks/sem_gap_big/blob/develop/images/objects/object6.jpg)

![Предмет 7](https://github.com/kirillkuks/sem_gap_big/blob/develop/images/objects/object7.jpg)

![Предмет 8](https://github.com/kirillkuks/sem_gap_big/blob/develop/images/objects/object8.jpg)

![Предмет 9](https://github.com/kirillkuks/sem_gap_big/blob/develop/images/objects/object9.jpg)

![Предмет 10](https://github.com/kirillkuks/sem_gap_big/blob/develop/images/objects/object10.jpg)

![Поверхность](https://github.com/kirillkuks/sem_gap_big/blob/develop/images/objects/surface.jpg)
