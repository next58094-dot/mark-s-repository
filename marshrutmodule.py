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
        self.bounds = None
        self.waypoints = []  # [(lat, lon, relative_altitude), ...]
        self.start_point = None  # (lat, lon, abs_elevation)
        self.selected_point = -1
        self.dragging = False
        self.unsafe_points = set()
        
    def set_map(self, pixmap, dem_array, bounds):
        self.satellite_image = pixmap
        self.dem_array = dem_array
        self.bounds = bounds
        self.update()
    
    def set_start_point(self, lat, lon):
        """Установка точки старта (обнуление барометра)"""
        abs_elevation = self.get_elevation_at(lat, lon)
        self.start_point = (lat, lon, abs_elevation)
        self.update()
        return abs_elevation
    
    def add_waypoint(self, lat, lon, relative_altitude=0):
        """Добавление точки маршрута с относительной высотой"""
        self.waypoints.append([lat, lon, relative_altitude])
        self.update()
    
    def get_absolute_elevation(self, lat, lon):
        """Получение абсолютной высоты над уровнем моря"""
        return self.get_elevation_at(lat, lon)
    
    def get_relative_ground_elevation(self, lat, lon):
        """Получение высоты земли относительно точки старта"""
        if self.start_point is None:
            return 0
        abs_ground = self.get_elevation_at(lat, lon)
        start_abs = self.start_point[2]
        return abs_ground - start_abs
    
    def get_flight_absolute_altitude(self, lat, lon, relative_flight_alt):
        """Получение абсолютной высоты полета БПЛА"""
        if self.start_point is None:
            return self.get_elevation_at(lat, lon) + relative_flight_alt
        start_abs = self.start_point[2]
        return start_abs + relative_flight_alt
    
    def get_clearance_above_ground(self, lat, lon, relative_flight_alt):
        """Получение расстояния от БПЛА до земли (положительное = безопасно)"""
        ground_rel = self.get_relative_ground_elevation(lat, lon)
        return relative_flight_alt - ground_rel
    
    def get_elevation_at(self, lat, lon):
        """Получение абсолютной высоты земли из DEM"""
        if self.dem_array is None or self.bounds is None:
            return 0
        
        min_lon, max_lon, min_lat, max_lat = self.bounds
        h, w = self.dem_array.shape
        
        x = (lon - min_lon) / (max_lon - min_lon) * (w - 1)
        y = (max_lat - lat) / (max_lat - min_lat) * (h - 1)
        
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
        """Получение профиля траектории (все высоты - относительные)"""
        if len(self.waypoints) < 2 or self.start_point is None:
            return [], [], [], []
        
        distances = [0.0]
        ground_rel = []
        flight_rel = []
        waypoint_indices = [0]
        start_abs = self.start_point[2]
        
        for i in range(len(self.waypoints) - 1):
            lat1, lon1, alt_rel1 = self.waypoints[i]
            lat2, lon2, alt_rel2 = self.waypoints[i + 1]
            
            steps = max(20, int(calculate_distance(lat1, lon1, lat2, lon2) / 30))
            
            for j in range(steps + 1):
                t = j / steps
                lat = lat1 + (lat2 - lat1) * t
                lon = lon1 + (lon2 - lon1) * t
                
                if j == 0 and i > 0:
                    continue
                
                # Относительная высота земли
                ground_abs = self.get_elevation_at(lat, lon)
                ground_rel_val = ground_abs - start_abs
                ground_rel.append(ground_rel_val)
                
                # Относительная высота полета (линейная интерполяция)
                flight_rel_val = alt_rel1 + (alt_rel2 - alt_rel1) * t
                flight_rel.append(flight_rel_val)
                
                if j < steps:
                    dist = calculate_distance(lat, lon, lat2, lon2) / 1000
                    distances.append(distances[-1] + dist)
            
            waypoint_indices.append(len(ground_rel) - 1)
        
        return distances, ground_rel, flight_rel, waypoint_indices
    
    def paintEvent(self, event):
        if self.satellite_image is None:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(43, 43, 43))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(self.rect(), Qt.AlignCenter, "Загрузите карту")
            return
        
        painter = QPainter(self)
        
        scaled = self.satellite_image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x_offset = (self.width() - scaled.width()) // 2
        y_offset = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x_offset, y_offset, scaled)
        
        img_w, img_h = scaled.width(), scaled.height()
        
        def map_point(lat, lon):
            min_lon, max_lon, min_lat, max_lat = self.bounds
            x = (lon - min_lon) / (max_lon - min_lon) * img_w + x_offset
            y = (max_lat - lat) / (max_lat - min_lat) * img_h + y_offset
            return int(x), int(y)
        
        # Рисуем точку старта
        if self.start_point:
            x, y = map_point(self.start_point[0], self.start_point[1])
            painter.setBrush(QBrush(QColor(0, 255, 255)))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawEllipse(x - 10, y - 10, 20, 20)
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            painter.setFont(QFont("Arial", 9, QFont.Bold))
            painter.drawText(x - 15, y - 15, "СТАРТ")
        
        # Рисуем траекторию
        if len(self.waypoints) >= 2:
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)
            
            for i in range(len(self.waypoints) - 1):
                x1, y1 = map_point(self.waypoints[i][0], self.waypoints[i][1])
                x2, y2 = map_point(self.waypoints[i+1][0], self.waypoints[i+1][1])
                painter.drawLine(x1, y1, x2, y2)
        
        # Рисуем точки маршрута
        for i, (lat, lon, alt_rel) in enumerate(self.waypoints):
            x, y = map_point(lat, lon)
            
            color = QColor(255, 0, 0) if i in self.unsafe_points else QColor(0, 255, 0)
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawEllipse(x - 8, y - 8, 16, 16)
            
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.drawText(x - 5, y - 10, str(i + 1))
    
    def mousePressEvent(self, event):
        if self.satellite_image is None:
            return
        
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
                self.setCursor(Qt.ClosedHandCursor)
                return
        
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
        self.unsafe_points = set(indices)
        self.update()

class ProfileWidget(QWidget):
    """Виджет для отображения профиля высот"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)
        self.setStyleSheet("background-color: white;")
        
        self.distances = []
        self.ground_rel = []
        self.flight_rel = []
        self.waypoint_indices = []
        self.unsafe_indices = []
        self.min_safe_relative = []
        self.start_abs_elev = 0
        
    def set_data(self, distances, ground_rel, flight_rel, waypoint_indices, unsafe_indices, start_abs_elev, min_clearance):
        self.distances = distances
        self.ground_rel = ground_rel
        self.flight_rel = flight_rel
        self.waypoint_indices = waypoint_indices
        self.unsafe_indices = unsafe_indices
        self.start_abs_elev = start_abs_elev
        self.min_safe_relative = [g + min_clearance for g in ground_rel]
        self.update()
    
    def paintEvent(self, event):
        if not self.distances or not self.ground_rel:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(255, 255, 255))
            painter.setPen(QColor(128, 128, 128))
            painter.drawText(self.rect(), Qt.AlignCenter, 
                           "Установите точку старта и постройте маршрут")
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        margins = QRectF(60, 30, self.width() - 80, self.height() - 80)
        
        # Диапазоны
        max_dist = max(self.distances) if self.distances else 1
        all_heights = self.ground_rel + self.flight_rel + self.min_safe_relative
        max_height = max(all_heights) if all_heights else 100
        min_height = min(self.ground_rel) if self.ground_rel else 0
        
        if min_height > 0:
            min_height = 0
        
        height_range = max_height - min_height
        if height_range < 1:
            height_range = 1
        
        # Отрисовка сетки
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        for i in range(6):
            y = margins.top() + (margins.height() * i / 5)
            painter.drawLine(int(margins.left()), int(y), int(margins.right()), int(y))
            
            h = max_height - (i / 5) * height_range
            painter.setPen(QPen(QColor(100, 100, 100), 1))
            painter.drawText(5, int(y) + 3, f"{h:.0f} м")
        
        for i in range(5):
            x = margins.left() + (margins.width() * i / 4)
            painter.drawLine(int(x), int(margins.top()), int(x), int(margins.bottom()))
            d = (i / 4) * max_dist
            painter.drawText(int(x) - 15, int(margins.bottom()) + 15, f"{d:.1f} км")
        
        # Горизонтальная линия уровня старта (0)
        painter.setPen(QPen(QColor(100, 100, 255), 2, Qt.DashLine))
        y0 = margins.bottom() - ((0 - min_height) / height_range) * margins.height()
        painter.drawLine(int(margins.left()), int(y0), int(margins.right()), int(y0))
        painter.drawText(int(margins.right()) + 5, int(y0) + 3, "Уровень старта (0)")
        
        def map_point(dist, height):
            x = margins.left() + (dist / max_dist) * margins.width()
            y = margins.bottom() - ((height - min_height) / height_range) * margins.height()
            return int(x), int(y)
        
        # Рельеф (залитый)
        ground_points = []
        for i, dist in enumerate(self.distances):
            x, y = map_point(dist, self.ground_rel[i])
            ground_points.append(QPoint(x, y))
        
        painter.setBrush(QBrush(QColor(150, 150, 200, 100)))
        painter.setPen(QPen(QColor(150, 150, 200), 1))
        
        # Заливка под рельефом
        for i in range(len(ground_points) - 1):
            bottom_left = QPoint(ground_points[i].x(), int(margins.bottom()))
            bottom_right = QPoint(ground_points[i+1].x(), int(margins.bottom()))
            polygon = QPolygonF([ground_points[i], ground_points[i+1], bottom_right, bottom_left])
            painter.drawPolygon(polygon)
        
        # Линия рельефа
        pen = QPen(QColor(100, 100, 200), 2)
        painter.setPen(pen)
        for i in range(len(ground_points) - 1):
            painter.drawLine(ground_points[i], ground_points[i+1])
        
        # Минимальная безопасная высота
        pen = QPen(QColor(255, 100, 100), 2, Qt.DashLine)
        painter.setPen(pen)
        for i in range(len(self.distances) - 1):
            x1, y1 = map_point(self.distances[i], self.min_safe_relative[i])
            x2, y2 = map_point(self.distances[i+1], self.min_safe_relative[i+1])
            painter.drawLine(x1, y1, x2, y2)
        
        # Траектория полета
        flight_points = []
        for i, dist in enumerate(self.distances):
            x, y = map_point(dist, self.flight_rel[i])
            flight_points.append(QPoint(x, y))
        
        pen = QPen(QColor(0, 200, 0), 3)
        painter.setPen(pen)
        for i in range(len(flight_points) - 1):
            painter.drawLine(flight_points[i], flight_points[i+1])
        
        # Точки маршрута
        painter.setBrush(QBrush(QColor(0, 255, 0)))
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        for idx in self.waypoint_indices:
            if idx < len(flight_points):
                painter.drawEllipse(flight_points[idx].x() - 5, flight_points[idx].y() - 5, 10, 10)
                painter.drawText(flight_points[idx].x() + 5, flight_points[idx].y() - 5, 
                               str(self.waypoint_indices.index(idx) + 1))
        
        # Проблемные участки
        painter.setPen(QPen(QColor(255, 0, 0), 3))
        for idx in self.unsafe_indices:
            if idx < len(flight_points):
                painter.drawEllipse(flight_points[idx].x() - 8, flight_points[idx].y() - 8, 16, 16)
        
        # Легенда
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.setFont(QFont("Arial", 8))
        painter.drawText(10, 20, "Легенда:")
        painter.fillRect(70, 12, 20, 10, QBrush(QColor(150, 150, 200)))
        painter.drawText(95, 20, "Рельеф (относительно старта)")
        
        painter.setPen(QPen(QColor(0, 200, 0), 3))
        painter.drawLine(70, 30, 90, 30)
        painter.drawText(95, 34, "Траектория БПЛА")
        
        painter.setPen(QPen(QColor(255, 100, 100), 2, Qt.DashLine))
        painter.drawLine(70, 45, 90, 45)
        painter.drawText(95, 49, "Мин. безопасная высота")
        
        painter.setPen(QPen(QColor(100, 100, 255), 2, Qt.DashLine))
        painter.drawLine(70, 60, 90, 60)
        painter.drawText(95, 64, "Уровень старта (0)")
        
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawText(10, self.height() - 15, 
                        f"Старт: {self.start_abs_elev:.0f} м над уровнем моря")

class RoutePlanner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Планировщик маршрута БПЛА (относительные высоты)")
        self.setGeometry(100, 100, 1400, 900)
        
        self.satellite_image = None
        self.dem_array = None
        self.bounds = None
        self.start_point = None
        self.waypoints = []
        self.start_abs_elev = 0
        
        self.setup_ui()
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # Левая панель
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
        
        # Точка старта
        start_group = QGroupBox("Точка старта (обнуление барометра)")
        start_layout = QVBoxLayout()
        
        start_info = QLabel("Кликните по карте для установки точки старта\n(абсолютная высота берется из DEM)")
        start_info.setWordWrap(True)
        start_info.setStyleSheet("color: gray; font-size: 9pt;")
        start_layout.addWidget(start_info)
        
        self.start_btn = QPushButton("Установить точку старта")
        self.start_btn.clicked.connect(self.set_start_point_mode)
        self.start_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")
        start_layout.addWidget(self.start_btn)
        
        self.start_info_label = QLabel("Старт не установлен")
        self.start_info_label.setStyleSheet("color: orange; font-weight: bold;")
        start_layout.addWidget(self.start_info_label)
        
        start_group.setLayout(start_layout)
        left_layout.addWidget(start_group)
        
        # Параметры полета
        flight_group = QGroupBox("Параметры полета")
        flight_layout = QVBoxLayout()
        
        flight_layout.addWidget(QLabel("Высота полета (относительно старта), м:"))
        self.altitude_spin = QDoubleSpinBox()
        self.altitude_spin.setRange(0, 2000)
        self.altitude_spin.setValue(200)
        self.altitude_spin.setSuffix(" м")
        self.altitude_spin.valueChanged.connect(self.on_flight_params_changed)
        flight_layout.addWidget(self.altitude_spin)
        
        flight_layout.addWidget(QLabel("Минимальное расстояние до земли, м:"))
        self.min_clearance_spin = QDoubleSpinBox()
        self.min_clearance_spin.setRange(0, 500)
        self.min_clearance_spin.setValue(50)
        self.min_clearance_spin.setSuffix(" м")
        self.min_clearance_spin.valueChanged.connect(self.on_flight_params_changed)
        flight_layout.addWidget(self.min_clearance_spin)
        
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
        
        # Проверка маршрута
        check_group = QGroupBox("Проверка маршрута")
        check_layout = QVBoxLayout()
        
        self.check_btn = QPushButton("Проверить маршрут")
        self.check_btn.clicked.connect(self.check_route)
        self.check_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; padding: 8px; }")
        check_layout.addWidget(self.check_btn)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(120)
        check_layout.addWidget(self.result_text)
        
        check_group.setLayout(check_layout)
        left_layout.addWidget(check_group)
        
        left_layout.addStretch()
        left_panel.setMaximumWidth(320)
        
        # Правая панель
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.map_widget = MapWidget()
        self.map_widget.point_clicked.connect(self.on_map_click)
        self.map_widget.point_moved.connect(self.move_waypoint)
        right_layout.addWidget(self.map_widget)
        
        self.profile_widget = ProfileWidget()
        right_layout.addWidget(self.profile_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([320, 1080])
        
        main_layout.addWidget(splitter)
        
        self.statusBar().showMessage("Загрузите карту, затем установите точку старта")
        self.waiting_for_start = False
    
    def on_flight_params_changed(self):
        """При изменении параметров полета обновляем профиль"""
        if self.start_point and len(self.waypoints) >= 2:
            self.update_profile()
    
    def load_map(self):
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
                
                size_deg_lat = (size_km / 2) / 111.32
                size_deg_lon = (size_km / 2) / (111.32 * cos(radians(lat)))
                
                bounds = (lon - size_deg_lon, lon + size_deg_lon, 
                         lat - size_deg_lat, lat + size_deg_lat)
            
            if dem_array is None:
                QMessageBox.warning(self, "Предупреждение", "DEM файл не найден")
            
            self.satellite_image = pixmap
            self.dem_array = dem_array
            self.bounds = bounds
            
            self.map_widget.set_map(pixmap, dem_array, bounds)
            self.clear_all_points()
            self.start_point = None
            self.start_abs_elev = 0
            self.start_info_label.setText("Старт не установлен")
            self.start_info_label.setStyleSheet("color: orange; font-weight: bold;")
            
            self.statusBar().showMessage(f"Загружена карта: {os.path.basename(path)}")
            QMessageBox.information(self, "Успех", "Карта загружена. Установите точку старта.")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить карту:\n{str(e)}")
    
    def set_start_point_mode(self):
        if self.satellite_image is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите карту")
            return
        
        self.waiting_for_start = True
        self.statusBar().showMessage("Кликните по карте для установки точки старта")
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
    
    def on_map_click(self, lat, lon):
        if self.waiting_for_start:
            self.set_start_point(lat, lon)
            self.waiting_for_start = False
            self.start_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")
        else:
            self.add_waypoint(lat, lon)
    
    def set_start_point(self, lat, lon):
        """Установка точки старта с получением абсолютной высоты из DEM"""
        abs_elev = self.map_widget.get_elevation_at(lat, lon)
        self.start_point = (lat, lon)
        self.start_abs_elev = abs_elev
        self.map_widget.set_start_point(lat, lon)
        
        self.start_info_label.setText(f"Старт: {lat:.5f}, {lon:.5f} | Высота: {abs_elev:.0f} м над уровнем моря")
        self.start_info_label.setStyleSheet("color: green; font-weight: bold;")
        
        self.statusBar().showMessage(f"Точка старта установлена. Абсолютная высота: {abs_elev:.0f} м. Теперь добавляйте точки маршрута.")
        
        QMessageBox.information(self, "Старт установлен", 
                               f"Точка старта:\n"
                               f"Широта: {lat:.5f}\n"
                               f"Долгота: {lon:.5f}\n"
                               f"Абсолютная высота: {abs_elev:.0f} м над уровнем моря\n\n"
                               f"Барометр обнулен. Теперь все высоты будут отсчитываться от этой точки.")
    
    def add_waypoint(self, lat, lon):
        if self.start_point is None:
            QMessageBox.warning(self, "Ошибка", "Сначала установите точку старта")
            return
        
        ground_rel = self.map_widget.get_relative_ground_elevation(lat, lon)
        flight_rel = self.altitude_spin.value()
        clearance = flight_rel - ground_rel
        
        self.waypoints.append([lat, lon, flight_rel])
        self.map_widget.add_waypoint(lat, lon, flight_rel)
        
        status_color = "green" if clearance >= self.min_clearance_spin.value() else "red"
        
        item = QListWidgetItem(f"Точка {len(self.waypoints)}: {lat:.5f}, {lon:.5f}")
        item.setToolTip(f"Высота земли: {ground_rel:.0f} м отн.\nВысота полета: {flight_rel:.0f} м отн.\nЗазор: {clearance:.0f} м")
        self.points_list.addItem(item)
        
        self.statusBar().showMessage(f"Добавлена точка {len(self.waypoints)}: высота земли={ground_rel:.0f} м, зазор={clearance:.0f} м")
        self.update_profile()
    
    def move_waypoint(self, index, lat, lon):
        if index < len(self.waypoints):
            self.waypoints[index][0] = lat
            self.waypoints[index][1] = lon
            self.map_widget.waypoints[index][0] = lat
            self.map_widget.waypoints[index][1] = lon
            
            ground_rel = self.map_widget.get_relative_ground_elevation(lat, lon)
            flight_rel = self.waypoints[index][2]
            
            self.points_list.item(index).setText(f"Точка {index+1}: {lat:.5f}, {lon:.5f}")
            self.points_list.item(index).setToolTip(f"Высота земли: {ground_rel:.0f} м отн.\nВысота полета: {flight_rel:.0f} м отн.")
            
            self.update_profile()
    
    def delete_selected_point(self):
        selected = self.points_list.currentRow()
        if selected >= 0 and selected < len(self.waypoints):
            self.waypoints.pop(selected)
            self.map_widget.waypoints.pop(selected)
            self.points_list.takeItem(selected)
            
            for i in range(self.points_list.count()):
                self.points_list.item(i).setText(f"Точка {i+1}: {self.waypoints[i][0]:.5f}, {self.waypoints[i][1]:.5f}")
            
            self.update_profile()
    
    def clear_all_points(self):
        self.waypoints.clear()
        self.map_widget.waypoints.clear()
        self.points_list.clear()
        self.update_profile()
    
    def on_point_selected(self):
        selected = self.points_list.currentRow()
        if selected >= 0:
            self.map_widget.selected_point = selected
            self.map_widget.update()
    
    def update_profile(self):
        if len(self.waypoints) < 2 or self.start_point is None:
            self.profile_widget.set_data([], [], [], [], [], 0, 0)
            return
        
        distances, ground_rel, flight_rel, waypoint_indices = self.map_widget.get_trajectory_profile()
        
        unsafe_indices = []
        for i in range(len(flight_rel)):
            if flight_rel[i] - ground_rel[i] < self.min_clearance_spin.value():
                unsafe_indices.append(i)
        
        self.profile_widget.set_data(distances, ground_rel, flight_rel, 
                                     waypoint_indices, unsafe_indices, 
                                     self.start_abs_elev, self.min_clearance_spin.value())
        
        unsafe_waypoints = set()
        for idx in unsafe_indices:
            for i, wp_idx in enumerate(waypoint_indices):
                if idx <= wp_idx:
                    unsafe_waypoints.add(i)
                    break
        
        self.map_widget.set_unsafe_points(unsafe_waypoints)
    
    def check_route(self):
        if self.start_point is None:
            QMessageBox.warning(self, "Ошибка", "Сначала установите точку старта")
            return
        
        if len(self.waypoints) < 2:
            QMessageBox.warning(self, "Ошибка", "Добавьте минимум 2 точки маршрута")
            return
        
        distances, ground_rel, flight_rel, waypoint_indices = self.map_widget.get_trajectory_profile()
        
        violations = []
        for i in range(len(flight_rel)):
            clearance = flight_rel[i] - ground_rel[i]
            if clearance < self.min_clearance_spin.value():
                dist = distances[i] if i < len(distances) else 0
                violations.append({
                    'dist': dist,
                    'ground': ground_rel[i],
                    'flight': flight_rel[i],
                    'clearance': clearance,
                    'required': self.min_clearance_spin.value(),
                    'deficit': self.min_clearance_spin.value() - clearance
                })
        
        unsafe_indices = [i for i in range(len(flight_rel)) 
                         if flight_rel[i] - ground_rel[i] < self.min_clearance_spin.value()]
        
        self.profile_widget.set_data(distances, ground_rel, flight_rel, 
                                     waypoint_indices, unsafe_indices, 
                                     self.start_abs_elev, self.min_clearance_spin.value())
        
        unsafe_waypoints = set()
        for idx in unsafe_indices:
            for i, wp_idx in enumerate(waypoint_indices):
                if idx <= wp_idx:
                    unsafe_waypoints.add(i)
                    break
        
        self.map_widget.set_unsafe_points(unsafe_waypoints)
        
        if violations:
            self.result_text.setStyleSheet("color: red;")
            msg = f"❌ МАРШРУТ НЕБЕЗОПАСЕН!\n\n"
            msg += f"Найдено {len(violations)} нарушений минимального расстояния до земли.\n"
            msg += f"Требуется зазор: {self.min_clearance_spin.value()} м\n\n"
            msg += "Проблемные участки:\n"
            for v in violations[:10]:
                msg += f"• На {v['dist']:.1f} км: земля={v['ground']:.0f} м, "
                msg += f"БПЛА={v['flight']:.0f} м, зазор={v['clearance']:.0f} м "
                msg += f"(не хватает {v['deficit']:.0f} м)\n"
            
            self.result_text.setText(msg)
            self.statusBar().showMessage(f"Найдено {len(violations)} нарушений безопасности")
            QMessageBox.warning(self, "Результат проверки", msg)
        else:
            self.result_text.setStyleSheet("color: green;")
            min_clearance = min([flight_rel[i] - ground_rel[i] for i in range(len(flight_rel))])
            msg = f"✅ МАРШРУТ БЕЗОПАСЕН!\n\n"
            msg += f"Минимальный зазор: {min_clearance:.0f} м\n"
            msg += f"Требуемый зазор: {self.min_clearance_spin.value()} м\n"
            msg += f"Высота старта: {self.start_abs_elev:.0f} м над уровнем моря"
            self.result_text.setText(msg)
            self.statusBar().showMessage("Маршрут безопасен")
            QMessageBox.information(self, "Результат проверки", msg)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RoutePlanner()
    window.show()
    sys.exit(app.exec_())