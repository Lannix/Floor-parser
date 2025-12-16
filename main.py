import cv2
import numpy as np
import json
import os
from skimage.morphology import skeletonize


class FloorPlanParser:
    def process(self, image_path):
        """
        Основной пайплайн обработки:
        1. Чтение -> 2. Бинаризация -> 3. Удаление шума -> 4. Скелетизация -> 5. Векторизация
        """
        # 1. Чтение
        img = cv2.imread(image_path)
        if img is None:
            print(f"Ошибка: Не удалось открыть {image_path}")
            return None, None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Бинаризация
        # Инвертируем, чтобы стены стали белыми (255) на черном фоне (0) и отрезаем по порогу в яркости в 200
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # 3. Морфология: удаление шума и мелких деталей
        # ядро зависит от разрешения картинки
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Дополнительное уберем дыры внутри стен
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 4. Скелетизация с помощью Scikit-image
        # Приводим к булевому типу, скелетизируем, возвращаем в uint8
        skeleton = skeletonize(mask > 0).astype(np.uint8) * 255
        
        # 5. Преобразование Хафа
        # minLineLength - минимальная длина стены в пикселях
        # maxLineGap - разрыв, который можно считать одной линией
        lines = cv2.HoughLinesP(skeleton, 1, np.pi/180, threshold=20, minLineLength=25, maxLineGap=10)
        
        walls = []
        if lines is not None:
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                walls.append({
                    "id": f"wall {i}",
                    "points": [[int(x1), int(y1)], [int(x2), int(y2)]]
                })

        result_json = {
            "meta": {"source": os.path.basename(image_path)},
            "walls": walls
        }
        
        return result_json, img

    def visualize_result(self, original_img, walls, output_path):
        """
        Рисует найденные векторы поверх исходного изображения
        """
        vis_img = original_img.copy()
        
        for wall in walls:
            pts = wall["points"]
            p1 = tuple(pts[0])
            p2 = tuple(pts[1])
            
            # Рисуем линию стены (Красный, толщина 3)
            cv2.line(vis_img, p1, p2, (0, 0, 255), 3)
            
            # Рисуем узлы (Синий, радиус 3)
            cv2.circle(vis_img, p1, 3, (255, 0, 0), -1)
            cv2.circle(vis_img, p2, 3, (255, 0, 0), -1)
            
        cv2.imwrite(output_path, vis_img)

    def save_json(self, data, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    test_images = ["images\image_2025-12-10_14-48-30.png", "images\image_2025-12-10_14-48-29.png"] 
    
    parser = FloorPlanParser()
    
    print(f"Запуск обработки. Результаты будут в папке: {output_dir}/")
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"Файл не найден: {img_path}")
            continue
            
        # Обработка
        json_data, original_img = parser.process(img_path)
        
        if json_data == None:
            continue

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Сохранение JSON
        json_out = os.path.join(output_dir, f"{base_name}_walls.json")
        parser.save_json(json_data, json_out)
        
        # Визуализация и сохранение картинки
        img_out = os.path.join(output_dir, f"{base_name}_visualized.jpg")
        parser.visualize_result(original_img, json_data['walls'], img_out)
        
        count = len(json_data['walls'])
        print(f"[OK] {img_path}: Найдено {count} сегментов стен -> {img_out}")