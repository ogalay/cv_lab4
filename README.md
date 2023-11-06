# Улучшение яркости изображения
## Описание задачи
* Задача - улучшение яркости изображения
* Язык программирования Python
* Ограничений по использованию библиотек и сторонних функций нет

## Датасет
Набор данных содержит содержит 745 примеров тёмных картинок. В каталоге low хранятся тёмные изображения, в каталоге high - соответсвующие им яркие изображения. Пример загрузки датасета в PyTorch приведён в файле dataset.py       
[Ссылка на датасет](https://drive.google.com/file/d/1ThoPb1flnfXDpRIytgBd7_e9Kv_lPnbo/view) 

## Решение
* Архитектура - `Zero-DCE network`, 7 слоев свёртки, оптимизатор Adam, 23 эпохи, batch_size = 8

## Результаты на исходном датасете

![image](https://github.com/ogalay/cv_lab4/assets/43163344/dabb1cd8-9469-4be5-89cc-6210117ecaea)

![image](https://github.com/ogalay/cv_lab4/assets/43163344/b1983c33-978b-432a-b7b7-79b7e3e4a265)

![image](https://github.com/ogalay/cv_lab4/assets/43163344/496958e6-d50a-4089-9189-fd4ac2c427fa)


### Метрики
* PSNR - `19.09`
* SSIM - `0.71`
* LPIPS - `0.24`
* Время инференса - `1.3 мс`

## Результаты на тестовом датасете

![image](https://github.com/ogalay/cv_lab4/assets/43163344/7894a95b-bee7-4817-90ed-de8ad8a3a952)

![image](https://github.com/ogalay/cv_lab4/assets/43163344/44c818b9-9124-47d3-81c5-51427e5fec29)

![image](https://github.com/ogalay/cv_lab4/assets/43163344/285d86f5-eb3d-4269-b07a-8ea7cd4927c4)

### Метрики
* PSNR - `19.5`
* SSIM - `0.72`
* LPIPS - `0.23`
* Время инференса - `1.3 мс`






