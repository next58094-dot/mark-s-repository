import sys
import os
import json
import numpy as np
from datetime import datetime
from math import log, tan, radians, cos, pi, sqrt, atan2, degrees, sin
from PIL import Image, ImageDraw
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QFileDialog, QMessageBox, QProgressBar,
                             QSlider, QCheckBox, QComboBox, QGroupBox,
                             QSpinBox, QDoubleSpinBox, QListWidget, QListWidgetItem,
                             QSplitter, QTabWidget, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QRectF
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QImage, QFont, QBrush, QPolygonF
import requests
from io import BytesIO
import tempfile

# Константы
OPENTOPOGRAPHY_API_URL = "https://portal.opentopography.org/API/globaldem"
DEM_DATASETS = {
    "SRTMGL3": "SRTM GL3 90m",
    "SRTMGL1": "SRTM GL1 30m", 
    "NASADEM": "NASADEM Global DEM 30m",
}

def lat_lon_to_pixel(lat, lon, bounds, img_size):
    """Конвертация координат в пиксели"""
    min_lon, max_lon, min_lat, max_lat = bounds
    x = (lon - min_lon) / (max_lon - min_lon) * img_size[0]
    y = (max_lat - lat) / (max_lat - min_lat) * img_size[1]
    return int(x), int(y)

def pixel_to_lat_lon(x, y, bounds, img_size):
    """Конвертация пикселей в координаты"""
    min_lon, max_lon, min_lat, max_lat = bounds
    lon = min_lon + (x / img_size[0]) * (max_lon - min_lon)
    lat = max_lat - (y / img_size[1]) * (max_lat - min_lat)
    return lat, lon

def calculate_distance(lat1, lon1, lat2, lon2):
    """Расчет расстояния между двумя точками в метрах (гаверсинус)"""
    R = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    c = 2*atan2(sqrt(a), sqrt(1-a))
    return R * c

class MapWidget(QWidget):
    """Виджет карты с поддержкой кликов и drag&drop"""
    point_clicked = pyqtSignal(float, float)  # lat, lon
    point_moved = pyqtSignal(int, float, float)  # index, lat, lon
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 500)
        self.setStyleSheet("background-color: #2b2b2b;")
        
        self.satellite_image = None
        self.dem_array = None
        self.bounds = None  # (min_lon, max_lon, min_lat, max_lat)
        self.waypoints = []  # [(lat, lon, altitude_agl), ...]
        self.selected_point = -1
        self.dragging = False
        self.drag_start = None
        self.unsafe_points = set()
        
    def set_map(self, pixmap, dem_array, bounds):
        """Установка карты и DEM"""
        self.satellite_image = pixmap
        self.dem_array = dem_array
        self.bounds = bounds
        self.update()
    
    def add_waypoint(self, lat, lon, altitude_agl=100):
        """Добавление точки маршрута"""
        self.waypoints.append([lat, lon, altitude_agl])
        self.update()
    
    def update_waypoint(self, index, lat, lon):
        """Обновление координат точки"""
        if 0 <= index < len(self.waypoints):
            self.waypoints[index][0] = lat
            self.waypoints[index][1] = lon
            self.update()
    
    def remove_waypoint(self, index):
        """Удаление точки"""
        if 0 <= index < len(self.waypoints):
            self.waypoints.pop(index)
            self.update()
    
    def get_elevation_at(self, lat, lon):
        """Получение высоты земли в точке"""
        if self.dem_array is None or self.bounds is None:
            return 0
        
        min_lon, max_lon, min_lat, max_lat = self.bounds
        h, w = self.dem_array.shape
        
        # Нормализация координат
        x = (lon - min_lon) / (max_lon - min_lon) * (w - 1)
        y = (max_lat - lat) / (max_lat - min_lat) * (h - 1)
        
        # Билинейная интерполяция
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        
        if x0 < 0 or x0 >= w or y0 < 0 or y0 >= h:
            return 0
        
        dx, dy = x - x0, y - y0
        elev = (1 - dx) * (1 - dy) * self.dem_array[y0, x0] + \
               dx * (1 - dy) * self.dem_array[y0, x1] + \
               (1 - dx) * dy * self.dem_array[y1, x0] + \
               dx * dy * self.dem_array[y1, x1]
        
        return float(elev)
    
    def get_trajectory_profile(self):
        """Получение профиля траектории между точками"""
        if len(self.waypoints) < 2:
            return [], [], []
        
        # Дискретизация траектории
        distances = [0.0]
        elevations = []
        waypoint_indices = [0]
        
        for i in range(len(self.waypoints) - 1):
            lat1, lon1, _ = self.waypoints[i]
            lat2, lon2, _ = self.waypoints[i + 1]
            
            steps = max(10, int(calculate_distance(lat1, lon1, lat2, lon2) / 50))
            
            for j in range(steps + 1):
                t = j / steps
                lat = lat1 + (lat2 - lat1) * t
                lon = lon1 + (lon2 - lon1) * t
                
                if j == 0 and i > 0:
                    continue
                
                elev = self.get_elevation_at(lat, lon)
                elevations.append(elev)
                
                if j < steps:
                    dist = calculate_distance(lat, lon, lat2, lon2) / 1000
                    distances.append(distances[-1] + dist)
            
            waypoint_indices.append(len(elevations) - 1)
        
        return distances, elevations, waypoint_indices
    
    def paintEvent(self, event):
        """Отрисовка карты и маршрута"""
        if self.satellite_image is None:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(43, 43, 43))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(self.rect(), Qt.AlignCenter, "Загрузите карту")
            return
        
        painter = QPainter(self)
        
        # Масштабирование карты
        scaled = self.satellite_image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x_offset = (self.width() - scaled.width()) // 2
        y_offset = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x_offset, y_offset, scaled)
        
        # Коэффициенты преобразования
        img_w, img_h = scaled.width(), scaled.height()
        
        def map_point(lat, lon):
            min_lon, max_lon, min_lat, max_lat = self.bounds
            x = (lon - min_lon) / (max_lon - min_lon) * img_w + x_offset
            y = (max_lat - lat) / (max_lat - min_lat) * img_h + y_offset
            return int(x), int(y)
        
        # Рисуем траекторию
        if len(self.waypoints) >= 2:
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)
            
            for i in range(len(self.waypoints) - 1):
                x1, y1 = map_point(self.waypoints[i][0], self.waypoints[i][1])
                x2, y2 = map_point(self.waypoints[i+1][0], self.waypoints[i+1][1])
                painter.drawLine(x1, y1, x2, y2)
        
        # Рисуем точки маршрута
        for i, (lat, lon, alt) in enumerate(self.waypoints):
            x, y = map_point(lat, lon)
            
            # Цвет точки (красный если нарушена безопасность)
            color = QColor(255, 0, 0) if i in self.unsafe_points else QColor(0, 255, 0)
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawEllipse(x - 8, y - 8, 16, 16)
            
            # Номер точки
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.drawText(x - 5, y - 10, str(i + 1))
    
    def mousePressEvent(self, event):
        if self.satellite_image is None:
            return
        
        # Находим ближайшую точку для перетаскивания
        scaled = self.satellite_image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x_offset = (self.width() - scaled.width()) // 2
        y_offset = (self.height() - scaled.height()) // 2
        img_w, img_h = scaled.width(), scaled.height()
        
        def map_point_reverse(px, py):
            if not (x_offset <= px < x_offset + img_w and y_offset <= py < y_offset + img_h):
                return None, None
            x = (px - x_offset) / img_w
            y = (py - y_offset) / img_h
            min_lon, max_lon, min_lat, max_lat = self.bounds
            lon = min_lon + x * (max_lon - min_lon)
            lat = max_lat - y * (max_lat - min_lat)
            return lat, lon
        
        # Проверяем попадание в существующие точки
        for i, (lat, lon, alt) in enumerate(self.waypoints):
            min_lon, max_lon, min_lat, max_lat = self.bounds
            x = (lon - min_lon) / (max_lon - min_lon) * img_w + x_offset
            y = (max_lat - lat) / (max_lat - min_lat) * img_h + y_offset
            
            dist = sqrt((event.x() - x)**2 + (event.y() - y)**2)
            if dist < 15:
                self.selected_point = i
                self.dragging = True
                self.drag_start = (event.x(), event.y())
                self.setCursor(Qt.ClosedHandCursor)
                return
        
        # Новая точка
        lat, lon = map_point_reverse(event.x(), event.y())
        if lat is not None:
            self.point_clicked.emit(lat, lon)
    
    def mouseMoveEvent(self, event):
        if self.dragging and self.selected_point >= 0:
            scaled = self.satellite_image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x_offset = (self.width() - scaled.width()) // 2
            y_offset = (self.height() - scaled.height()) // 2
            img_w, img_h = scaled.width(), scaled.height()
            
            x = event.x() - x_offset
            y = event.y() - y_offset
            
            if 0 <= x < img_w and 0 <= y < img_h:
                min_lon, max_lon, min_lat, max_lat = self.bounds
                lon = min_lon + (x / img_w) * (max_lon - min_lon)
                lat = max_lat - (y / img_h) * (max_lat - min_lat)
                self.point_moved.emit(self.selected_point, lat, lon)
                self.update()
    
    def mouseReleaseEvent(self, event):
        if self.dragging:
            self.dragging = False
            self.selected_point = -1
            self.setCursor(Qt.ArrowCursor)
            self.update()
    
    def set_unsafe_points(self, indices):
        """Установка проблемных точек"""
        self.unsafe_points = set(indices)
        self.update()

class ProfileWidget(QWidget):
    """Виджет для отображения профиля высот"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(250)
        self.setStyleSheet("background-color: white;")
        
        self.distances = []
        self.elevations = []
        self.flight_altitudes = []
        self.waypoint_indices = []
        self.min_safe_altitude = []
        self.unsafe_indices = []
        
    def set_data(self, distances, elevations, flight_altitudes, min_safe_altitude, waypoint_indices, unsafe_indices):
        self.distances = distances
        self.elevations = elevations
        self.flight_altitudes = flight_altitudes
        self.min_safe_altitude = min_safe_altitude
        self.waypoint_indices = waypoint_indices
        self.unsafe_indices = unsafe_indices
        self.update()
    
    def paintEvent(self, event):
        if not self.distances or not self.elevations:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(255, 255, 255))
            painter.setPen(QColor(128, 128, 128))
            painter.drawText(self.rect(), Qt.AlignCenter, "Постройте маршрут для отображения профиля")
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        margins = QRectF(50, 20, self.width() - 70, self.height() - 60)
        
        # Определяем диапазоны
        max_dist = max(self.distances) if self.distances else 1
        all_heights = self.elevations + self.flight_altitudes + self.min_safe_altitude
        max_height = max(all_heights) if all_heights else 100
        min_height = min(self.elevations) if self.elevations else 0
        height_range = max_height - min_height
        if height_range < 1:
            height_range = 1
        
        # Отрисовка сетки
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        for i in range(5):
            y = margins.top() + (margins.height() * i / 4)
            painter.drawLine(int(margins.left()), int(y), int(margins.right()), int(y))
            
            h = max_height - (i / 4) * height_range
            painter.setPen(QPen(QColor(100, 100, 100), 1))
            painter.drawText(5, int(y) + 3, f"{h:.0f} м")
        
        for i in range(5):
            x = margins.left() + (margins.width() * i / 4)
            painter.drawLine(int(x), int(margins.top()), int(x), int(margins.bottom()))
            d = (i / 4) * max_dist
            painter.drawText(int(x) - 15, int(margins.bottom()) + 15, f"{d:.1f} км")
        
        # Функция для преобразования координат
        def map_point(dist, height):
            x = margins.left() + (dist / max_dist) * margins.width()
            y = margins.bottom() - ((height - min_height) / height_range) * margins.height()
            return int(x), int(y)
        
        # Рисуем гистограмму рельефа
        pen = QPen(QColor(150, 150, 200), 2)
        painter.setPen(pen)
        
        for i in range(len(self.distances) - 1):
            x1, y1 = map_point(self.distances[i], self.elevations[i])
            x2, y2 = map_point(self.distances[i+1], self.elevations[i+1])
            painter.drawLine(x1, y1, x2, y2)
        
        # Рисуем минимальную безопасную высоту
        if self.min_safe_altitude:
            pen = QPen(QColor(255, 100, 100), 2, Qt.DashLine)
            painter.setPen(pen)
            for i in range(len(self.distances) - 1):
                x1, y1 = map_point(self.distances[i], self.min_safe_altitude[i])
                x2, y2 = map_point(self.distances[i+1], self.min_safe_altitude[i+1])
                painter.drawLine(x1, y1, x2, y2)
        
        # Рисуем траекторию полета
        if self.flight_altitudes:
            pen = QPen(QColor(0, 200, 0), 3)
            painter.setPen(pen)
            for i in range(len(self.distances) - 1):
                x1, y1 = map_point(self.distances[i], self.flight_altitudes[i])
                x2, y2 = map_point(self.distances[i+1], self.flight_altitudes[i+1])
                painter.drawLine(x1, y1, x2, y2)
        
        # Отмечаем точки маршрута
        painter.setBrush(QBrush(QColor(0, 255, 0)))
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        
        for idx in self.waypoint_indices:
            if idx < len(self.distances):
                x, y = map_point(self.distances[idx], self.flight_altitudes[idx] if idx < len(self.flight_altitudes) else self.elevations[idx])
                painter.drawEllipse(x - 5, y - 5, 10, 10)
                painter.drawText(x + 5, y - 5, str(self.waypoint_indices.index(idx) + 1))
        
        # Отмечаем проблемные участки
        painter.setPen(QPen(QColor(255, 0, 0), 3))
        for idx in self.unsafe_indices:
            if idx < len(self.distances):
                x, y = map_point(self.distances[idx], self.flight_altitudes[idx] if idx < len(self.flight_altitudes) else self.elevations[idx])
                painter.drawEllipse(x - 8, y - 8, 16, 16)
        
        # Подписи
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.setFont(QFont("Arial", 9))
        painter.drawText(10, 20, "Легенда:")
        painter.setPen(QPen(QColor(150, 150, 200), 2))
        painter.drawLine(70, 15, 100, 15)
        painter.drawText(105, 19, "Рельеф")
        painter.setPen(QPen(QColor(0, 200, 0), 3))
        painter.drawLine(70, 30, 100, 30)
        painter.drawText(105, 34, "Траектория БПЛА")
        painter.setPen(QPen(QColor(255, 100, 100), 2, Qt.DashLine))
        painter.drawLine(70, 45, 100, 45)
        painter.drawText(105, 49, "Мин. безопасная высота")

class RoutePlanner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Планировщик маршрута БПЛА")
        self.setGeometry(100, 100, 1300, 800)
        
        self.satellite_image = None
        self.dem_array = None
        self.bounds = None
        self.waypoints = []
        
        self.setup_ui()
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # Левая панель - управление
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        
        # Загрузка карты
        load_group = QGroupBox("Загрузка карты")
        load_layout = QVBoxLayout()
        
        self.load_btn = QPushButton("Загрузить сохранённую карту")
        self.load_btn.clicked.connect(self.load_map)
        load_layout.addWidget(self.load_btn)
        
        load_group.setLayout(load_layout)
        left_layout.addWidget(load_group)
        
        # Параметры полета
        flight_group = QGroupBox("Параметры полета")
        flight_layout = QVBoxLayout()
        
        flight_layout.addWidget(QLabel("Высота полета (AGL), м:"))
        self.altitude_spin = QDoubleSpinBox()
        self.altitude_spin.setRange(10, 1000)
        self.altitude_spin.setValue(100)
        self.altitude_spin.setSuffix(" м")
        flight_layout.addWidget(self.altitude_spin)
        
        flight_layout.addWidget(QLabel("Минимальная высота над рельефом, м:"))
        self.min_altitude_spin = QDoubleSpinBox()
        self.min_altitude_spin.setRange(0, 500)
        self.min_altitude_spin.setValue(50)
        self.min_altitude_spin.setSuffix(" м")
        flight_layout.addWidget(self.min_altitude_spin)
        
        flight_group.setLayout(flight_layout)
        left_layout.addWidget(flight_group)
        
        # Список точек
        points_group = QGroupBox("Точки маршрута")
        points_layout = QVBoxLayout()
        
        self.points_list = QListWidget()
        self.points_list.itemSelectionChanged.connect(self.on_point_selected)
        points_layout.addWidget(self.points_list)
        
        points_btn_layout = QHBoxLayout()
        self.delete_point_btn = QPushButton("Удалить точку")
        self.delete_point_btn.clicked.connect(self.delete_selected_point)
        self.clear_all_btn = QPushButton("Очистить все")
        self.clear_all_btn.clicked.connect(self.clear_all_points)
        points_btn_layout.addWidget(self.delete_point_btn)
        points_btn_layout.addWidget(self.clear_all_btn)
        points_layout.addLayout(points_btn_layout)
        
        points_group.setLayout(points_layout)
        left_layout.addWidget(points_group)
        
        # Кнопки проверки
        check_group = QGroupBox("Проверка маршрута")
        check_layout = QVBoxLayout()
        
        self.check_btn = QPushButton("Проверить маршрут")
        self.check_btn.clicked.connect(self.check_route)
        self.check_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; padding: 8px; }")
        check_layout.addWidget(self.check_btn)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(100)
        check_layout.addWidget(self.result_text)
        
        check_group.setLayout(check_layout)
        left_layout.addWidget(check_group)
        
        left_layout.addStretch()
        left_panel.setMaximumWidth(300)
        
        # Правая панель - карта и профиль
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.map_widget = MapWidget()
        self.map_widget.point_clicked.connect(self.add_waypoint)
        self.map_widget.point_moved.connect(self.move_waypoint)
        right_layout.addWidget(self.map_widget)
        
        self.profile_widget = ProfileWidget()
        right_layout.addWidget(self.profile_widget)
        
        # Добавляем панели
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 1000])
        
        main_layout.addWidget(splitter)
        
        self.statusBar().showMessage("Загрузите карту для начала работы")
    
    def load_map(self):
        """Загрузка сохраненной карты"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите комбинированную карту", 
            os.path.dirname(os.path.abspath(__file__)),
            "PNG (*.png);;Все файлы (*.*)"
        )
        
        if not path:
            return
        
        try:
            pixmap = QPixmap(path)
            if pixmap.isNull():
                QMessageBox.warning(self, "Ошибка", "Не удалось загрузить изображение")
                return
            
            # Ищем связанные файлы
            base = path.replace('_combined.png', '').replace('.png', '')
            dem_path = f"{base}_dem.npy"
            json_path = f"{base}.json"
            
            dem_array = None
            bounds = None
            
            if os.path.exists(dem_path):
                dem_array = np.load(dem_path)
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    meta = json.load(f)
                
                size_km = meta.get('size_km', 10)
                lat = meta.get('latitude', 48.6337)
                lon = meta.get('longitude', 38.3765)
                
                # Вычисляем границы
                size_deg_lat = (size_km / 2) / 111.32
                size_deg_lon = (size_km / 2) / (111.32 * cos(radians(lat)))
                
                bounds = (lon - size_deg_lon, lon + size_deg_lon, 
                         lat - size_deg_lat, lat + size_deg_lat)
            
            if dem_array is None:
                QMessageBox.warning(self, "Предупреждение", 
                                   "DEM файл не найден. Рельеф будет недоступен.")
            
            self.satellite_image = pixmap
            self.dem_array = dem_array
            self.bounds = bounds
            
            self.map_widget.set_map(pixmap, dem_array, bounds)
            self.clear_all_points()
            
            self.statusBar().showMessage(f"Загружена карта: {os.path.basename(path)}")
            QMessageBox.information(self, "Успех", "Карта загружена. Добавляйте точки маршрута кликом мыши.")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить карту:\n{str(e)}")
    
    def add_waypoint(self, lat, lon):
        """Добавление точки маршрута"""
        elev = self.map_widget.get_elevation_at(lat, lon) if self.dem_array is not None else 0
        
        self.waypoints.append([lat, lon, self.altitude_spin.value()])
        self.map_widget.waypoints.append([lat, lon, self.altitude_spin.value()])
        
        # Обновляем список
        item = QListWidgetItem(f"Точка {len(self.waypoints)}: {lat:.5f}, {lon:.5f} | Высота: {self.altitude_spin.value()} м AGL")
        self.points_list.addItem(item)
        
        self.statusBar().showMessage(f"Добавлена точка {len(self.waypoints)}: широта={lat:.5f}, долгота={lon:.5f}, высота земли={elev:.0f} м")
        self.update_profile()
    
    def move_waypoint(self, index, lat, lon):
        """Перемещение точки маршрута"""
        if index < len(self.waypoints):
            self.waypoints[index][0] = lat
            self.waypoints[index][1] = lon
            
            # Обновляем список
            self.points_list.item(index).setText(
                f"Точка {index+1}: {lat:.5f}, {lon:.5f} | Высота: {self.waypoints[index][2]} м AGL"
            )
            
            self.update_profile()
    
    def delete_selected_point(self):
        """Удаление выбранной точки"""
        selected = self.points_list.currentRow()
        if selected >= 0 and selected < len(self.waypoints):
            self.waypoints.pop(selected)
            self.map_widget.waypoints.pop(selected)
            self.points_list.takeItem(selected)
            
            # Перенумеровываем точки
            for i in range(self.points_list.count()):
                self.points_list.item(i).setText(
                    f"Точка {i+1}: {self.waypoints[i][0]:.5f}, {self.waypoints[i][1]:.5f} | Высота: {self.waypoints[i][2]} м AGL"
                )
            
            self.update_profile()
    
    def clear_all_points(self):
        """Очистка всех точек"""
        self.waypoints.clear()
        self.map_widget.waypoints.clear()
        self.points_list.clear()
        self.update_profile()
    
    def on_point_selected(self):
        """Выбор точки в списке"""
        selected = self.points_list.currentRow()
        if selected >= 0:
            self.map_widget.selected_point = selected
            self.map_widget.update()
    
    def update_profile(self):
        """Обновление графика профиля"""
        if len(self.waypoints) < 2:
            self.profile_widget.set_data([], [], [], [], [], [])
            return
        
        # Получаем профиль траектории
        distances, elevations, waypoint_indices = self.map_widget.get_trajectory_profile()
        
        # Рассчитываем высоту полета и минимальную безопасную высоту
        flight_altitudes = []
        min_safe_altitude = []
        
        for i, elev in enumerate(elevations):
            # Интерполяция заданной высоты между точками
            alt = self.altitude_spin.value()
            flight_altitudes.append(elev + alt)
            min_safe_altitude.append(elev + self.min_altitude_spin.value())
        
        self.profile_widget.set_data(distances, elevations, flight_altitudes, 
                                     min_safe_altitude, waypoint_indices, [])
    
    def check_route(self):
        """Проверка безопасности маршрута"""
        if len(self.waypoints) < 2:
            QMessageBox.warning(self, "Ошибка", "Добавьте минимум 2 точки маршрута")
            return
        
        # Получаем профиль
        distances, elevations, waypoint_indices = self.map_widget.get_trajectory_profile()
        
        flight_altitudes = [elev + self.altitude_spin.value() for elev in elevations]
        min_safe_altitude = [elev + self.min_altitude_spin.value() for elev in elevations]
        
        # Находим нарушения
        unsafe_indices = []
        problem_points = []
        
        for i in range(len(flight_altitudes)):
            if flight_altitudes[i] < min_safe_altitude[i]:
                unsafe_indices.append(i)
                
                # Находим ближайшую точку маршрута
                for j, wp_idx in enumerate(waypoint_indices):
                    if i <= wp_idx:
                        problem_points.append({
                            'point': j + 1,
                            'distance': distances[i] if i < len(distances) else 0,
                            'flight_alt': flight_altitudes[i],
                            'min_alt': min_safe_altitude[i]
                        })
                        break
        
        # Обновляем отображение
        self.profile_widget.set_data(distances, elevations, flight_altitudes,
                                     min_safe_altitude, waypoint_indices, unsafe_indices)
        
        # Находим проблемные точки маршрута
        unsafe_waypoints = set()
        for idx in unsafe_indices:
            for i, wp_idx in enumerate(waypoint_indices):
                if idx <= wp_idx:
                    unsafe_waypoints.add(i)
                    break
        
        self.map_widget.set_unsafe_points(unsafe_waypoints)
        
        # Выводим результат
        if unsafe_indices:
            self.result_text.setStyleSheet("color: red;")
            msg = f"❌ МАРШРУТ НЕБЕЗОПАСЕН!\n\n"
            msg += f"Найдено {len(unsafe_indices)} нарушений минимальной высоты.\n\n"
            msg += "Проблемные участки:\n"
            for prob in problem_points[:10]:  # Показываем первые 10
                msg += f"• У точки {prob['point']}: высота БПЛА {prob['flight_alt']:.0f} м, требуется {prob['min_alt']:.0f} м\n"
            
            self.result_text.setText(msg)
            self.statusBar().showMessage(f"Найдено {len(unsafe_indices)} нарушений безопасности маршрута")
            QMessageBox.warning(self, "Результат проверки", msg)
        else:
            self.result_text.setStyleSheet("color: green;")
            msg = "✅ МАРШРУТ БЕЗОПАСЕН!\n\n"
            msg += f"Все точки маршрута соблюдают минимальную высоту {self.min_altitude_spin.value()} м над рельефом."
            self.result_text.setText(msg)
            self.statusBar().showMessage("Маршрут безопасен")
            QMessageBox.information(self, "Результат проверки", msg)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RoutePlanner()
    window.show()
    sys.exit(app.exec_())