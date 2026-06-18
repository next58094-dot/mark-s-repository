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
                             QSplitter, QTabWidget, QTextEdit, QProgressDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QRectF, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QImage, QFont, QBrush, QPolygonF
import traceback

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

def interpolate_points(lat1, lon1, lat2, lon2, num_points):
    """Интерполяция точек между двумя координатами"""
    result = []
    for i in range(num_points + 1):
        t = i / num_points
        lat = lat1 + (lat2 - lat1) * t
        lon = lon1 + (lon2 - lon1) * t
        result.append((lat, lon))
    return result

class RelaySearchThread(QThread):
    """Поток для поиска ретранслятора"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, map_widget, start_point, start_abs_elev, waypoints, 
                 operator_antenna_height, relay_antenna_height):
        super().__init__()
        self.map_widget = map_widget
        self.start_point = start_point
        self.start_abs_elev = start_abs_elev
        self.waypoints = waypoints
        self.operator_antenna_height = operator_antenna_height
        self.relay_antenna_height = relay_antenna_height
        self._is_running = True
        
        # Копируем DEM данные для работы в потоке
        self.dem_array = map_widget.dem_array
        self.bounds = map_widget.bounds
        
    def stop(self):
        self._is_running = False
    
    def get_elevation_at(self, lat, lon):
        """Получение высоты из DEM (без кэша для потока)"""
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
    
    def check_line_of_sight(self, lat1, lon1, alt1, lat2, lon2, alt2, antenna_height=2):
        """Проверка прямой видимости между двумя точками"""
        if self.dem_array is None or self.bounds is None:
            return True
        
        h1 = alt1 + antenna_height
        h2 = alt2 + antenna_height
        
        distance = calculate_distance(lat1, lon1, lat2, lon2)
        num_points = max(20, int(distance / 10))
        points = interpolate_points(lat1, lon1, lat2, lon2, num_points)
        
        for i in range(1, len(points) - 1):
            lat, lon = points[i]
            ground_alt = self.get_elevation_at(lat, lon)
            
            t = i / num_points
            line_alt = h1 + (h2 - h1) * t
            
            if ground_alt > line_alt:
                return False
        
        return True
    
    def run(self):
        try:
            if self.start_point is None or len(self.waypoints) < 2:
                self.finished.emit(None)
                return
            
            start_lat, start_lon = self.start_point
            start_abs = self.start_abs_elev
            
            # Получаем точки маршрута
            route_points = []
            self.status.emit("Генерация точек маршрута...")
            
            total_steps = 0
            for i in range(len(self.waypoints) - 1):
                lat1, lon1, alt1 = self.waypoints[i]
                lat2, lon2, alt2 = self.waypoints[i+1]
                steps = max(10, int(calculate_distance(lat1, lon1, lat2, lon2) / 20))
                total_steps += steps
            
            progress_counter = 0
            
            for i in range(len(self.waypoints) - 1):
                if not self._is_running:
                    self.finished.emit(None)
                    return
                    
                lat1, lon1, alt_rel1 = self.waypoints[i]
                lat2, lon2, alt_rel2 = self.waypoints[i+1]
                steps = max(10, int(calculate_distance(lat1, lon1, lat2, lon2) / 20))
                
                for j in range(steps + 1):
                    if not self._is_running:
                        self.finished.emit(None)
                        return
                        
                    t = j / steps
                    lat = lat1 + (lat2 - lat1) * t
                    lon = lon1 + (lon2 - lon1) * t
                    alt_abs = start_abs + alt_rel1 + (alt_rel2 - alt_rel1) * t
                    route_points.append((lat, lon, alt_abs))
                    
                    progress_counter += 1
                    if progress_counter % 10 == 0 and total_steps > 0:
                        self.progress.emit(int(progress_counter / total_steps * 30))
            
            if not route_points:
                self.finished.emit(None)
                return
            
            self.status.emit("Проверка видимости от оператора...")
            
            # Ищем точки с плохой видимостью
            shadow_points = []
            total_points = len(route_points)
            
            for idx, (lat, lon, alt) in enumerate(route_points):
                if not self._is_running:
                    self.finished.emit(None)
                    return
                    
                if not self.check_line_of_sight(start_lat, start_lon, start_abs, 
                                               lat, lon, alt, self.operator_antenna_height):
                    shadow_points.append((lat, lon, alt))
                
                if idx % 5 == 0 and total_points > 0:
                    self.progress.emit(30 + int(idx / total_points * 30))
            
            if not shadow_points:
                self.status.emit("Зона радиотени не обнаружена")
                self.finished.emit(None)
                return
            
            self.status.emit(f"Найдено {len(shadow_points)} точек в зоне тени. Поиск места для ретранслятора...")
            
            # Создаем сетку кандидатов
            candidates = []
            step = max(5, int(len(route_points) / 20))
            
            for idx in range(0, len(route_points), step):
                if not self._is_running:
                    self.finished.emit(None)
                    return
                    
                lat, lon, alt = route_points[idx]
                
                if calculate_distance(start_lat, start_lon, lat, lon) < 100:
                    continue
                
                if not self.check_line_of_sight(start_lat, start_lon, start_abs, 
                                               lat, lon, alt, self.operator_antenna_height):
                    continue
                
                candidates.append((lat, lon, alt))
                if len(route_points) > 0:
                    self.progress.emit(60 + int(idx / len(route_points) * 20))
            
            if not candidates:
                self.status.emit("Не найдено подходящих кандидатов")
                self.finished.emit(None)
                return
            
            # Оцениваем каждого кандидата
            self.status.emit("Оценка кандидатов...")
            best_relay = None
            best_coverage = 0
            
            total_candidates = len(candidates)
            for idx, (lat, lon, alt) in enumerate(candidates):
                if not self._is_running:
                    self.finished.emit(None)
                    return
                
                visible_count = 0
                for shadow_lat, shadow_lon, shadow_alt in shadow_points:
                    if self.check_line_of_sight(lat, lon, alt, 
                                               shadow_lat, shadow_lon, shadow_alt, 
                                               self.relay_antenna_height):
                        visible_count += 1
                
                coverage = visible_count / len(shadow_points) if shadow_points else 0
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_relay = (lat, lon)
                
                if idx % 2 == 0 and total_candidates > 0:
                    self.progress.emit(80 + int(idx / total_candidates * 20))
            
            self.status.emit(f"Поиск завершен. Покрытие: {best_coverage*100:.1f}%")
            
            if best_relay and best_coverage > 0.2:
                self.finished.emit(best_relay)
            else:
                self.finished.emit(None)
                
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)
            self.finished.emit(None)

class MapWidget(QWidget):
    """Виджет карты"""
    point_clicked = pyqtSignal(float, float)
    point_moved = pyqtSignal(int, float, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 500)
        self.setStyleSheet("background-color: #2b2b2b;")
        
        self.satellite_image = None
        self.dem_array = None
        self.bounds = None
        self.waypoints = []
        self.start_point = None
        self.selected_point = -1
        self.dragging = False
        self.unsafe_points = set()
        self.shadow_points = set()
        self.relay_point = None
        
    def set_map(self, pixmap, dem_array, bounds):
        self.satellite_image = pixmap
        self.dem_array = dem_array
        self.bounds = bounds
        self.update()
    
    def set_start_point(self, lat, lon):
        abs_elevation = self.get_elevation_at(lat, lon)
        self.start_point = (lat, lon, abs_elevation)
        self.update()
        return abs_elevation
    
    def add_waypoint(self, lat, lon, relative_altitude=0):
        self.waypoints.append([lat, lon, relative_altitude])
        self.update()
    
    def get_elevation_at(self, lat, lon):
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
    
    def get_relative_ground_elevation(self, lat, lon):
        if self.start_point is None:
            return 0
        abs_ground = self.get_elevation_at(lat, lon)
        start_abs = self.start_point[2]
        return abs_ground - start_abs
    
    def check_line_of_sight(self, lat1, lon1, alt1, lat2, lon2, alt2, antenna_height=2):
        if self.dem_array is None or self.bounds is None:
            return True
        
        h1 = alt1 + antenna_height
        h2 = alt2 + antenna_height
        
        distance = calculate_distance(lat1, lon1, lat2, lon2)
        num_points = max(20, int(distance / 10))
        points = interpolate_points(lat1, lon1, lat2, lon2, num_points)
        
        for i in range(1, len(points) - 1):
            lat, lon = points[i]
            ground_alt = self.get_elevation_at(lat, lon)
            
            t = i / num_points
            line_alt = h1 + (h2 - h1) * t
            
            if ground_alt > line_alt:
                return False
        
        return True
    
    def check_shadow_zone(self, operator_antenna_height=2):
        if self.start_point is None or len(self.waypoints) < 2:
            return []
        
        shadow_indices = []
        start_lat, start_lon, start_abs = self.start_point
        
        distances, ground_rel, flight_rel, waypoint_indices = self.get_trajectory_profile()
        
        for i in range(len(flight_rel)):
            if len(self.waypoints) > 1:
                total_segments = len(self.waypoints) - 1
                segment_idx = min(i * total_segments // max(1, len(flight_rel)), total_segments - 1)
                
                if segment_idx < len(self.waypoints) - 1:
                    lat1, lon1, alt1 = self.waypoints[segment_idx]
                    lat2, lon2, alt2 = self.waypoints[segment_idx + 1]
                    
                    points_in_segment = len(flight_rel) // total_segments
                    if points_in_segment > 0:
                        t = (i % points_in_segment) / points_in_segment
                    else:
                        t = 0
                    
                    lat = lat1 + (lat2 - lat1) * t
                    lon = lon1 + (lon2 - lon1) * t
                    alt = start_abs + flight_rel[i]
                else:
                    lat, lon, alt = self.waypoints[-1][0], self.waypoints[-1][1], start_abs + flight_rel[i]
            else:
                continue
            
            if not self.check_line_of_sight(start_lat, start_lon, start_abs, 
                                           lat, lon, alt, operator_antenna_height):
                shadow_indices.append(i)
        
        return shadow_indices
    
    def get_trajectory_profile(self):
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
                
                ground_abs = self.get_elevation_at(lat, lon)
                ground_rel_val = ground_abs - start_abs
                ground_rel.append(ground_rel_val)
                
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
        
        if self.start_point:
            x, y = map_point(self.start_point[0], self.start_point[1])
            painter.setBrush(QBrush(QColor(0, 255, 255)))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawEllipse(x - 10, y - 10, 20, 20)
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            painter.setFont(QFont("Arial", 9, QFont.Bold))
            painter.drawText(x - 15, y - 15, "СТАРТ")
        
        if self.shadow_points and len(self.waypoints) >= 2:
            painter.setPen(QPen(QColor(255, 0, 0, 100), 4, Qt.DashLine))
            for i in range(len(self.waypoints) - 1):
                x1, y1 = map_point(self.waypoints[i][0], self.waypoints[i][1])
                x2, y2 = map_point(self.waypoints[i+1][0], self.waypoints[i+1][1])
                if i in self.shadow_points or (i+1) in self.shadow_points:
                    painter.drawLine(x1, y1, x2, y2)
        
        if len(self.waypoints) >= 2:
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)
            for i in range(len(self.waypoints) - 1):
                x1, y1 = map_point(self.waypoints[i][0], self.waypoints[i][1])
                x2, y2 = map_point(self.waypoints[i+1][0], self.waypoints[i+1][1])
                painter.drawLine(x1, y1, x2, y2)
        
        if self.relay_point:
            x, y = map_point(self.relay_point[0], self.relay_point[1])
            painter.setBrush(QBrush(QColor(255, 165, 0)))
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.drawEllipse(x - 12, y - 12, 24, 24)
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            painter.setFont(QFont("Arial", 9, QFont.Bold))
            painter.drawText(x - 20, y - 20, "РЕТРАНСЛЯТОР")
            
            if self.start_point:
                x_start, y_start = map_point(self.start_point[0], self.start_point[1])
                painter.setPen(QPen(QColor(255, 165, 0, 150), 2, Qt.DashLine))
                painter.drawLine(x_start, y_start, x, y)
                
                for lat, lon, alt in self.waypoints:
                    x_wp, y_wp = map_point(lat, lon)
                    painter.setPen(QPen(QColor(255, 165, 0, 100), 1, Qt.DashLine))
                    painter.drawLine(x, y, x_wp, y_wp)
        
        for i, (lat, lon, alt_rel) in enumerate(self.waypoints):
            x, y = map_point(lat, lon)
            
            if i in self.shadow_points:
                color = QColor(255, 0, 0)
            elif i in self.unsafe_points:
                color = QColor(255, 165, 0)
            else:
                color = QColor(0, 255, 0)
            
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
    
    def set_shadow_points(self, indices):
        self.shadow_points = set(indices)
        self.update()
    
    def set_relay_point(self, lat, lon):
        self.relay_point = (lat, lon)
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
        self.shadow_indices = []
        self.min_safe_relative = []
        self.start_abs_elev = 0
        self.relay_point = None  # (dist, height) - расстояние и высота ретранслятора
        self.start_point_abs = 0  # Абсолютная высота старта
        
    def set_data(self, distances, ground_rel, flight_rel, waypoint_indices, 
                 unsafe_indices, shadow_indices, start_abs_elev, min_clearance,
                 relay_point=None):
        self.distances = distances
        self.ground_rel = ground_rel
        self.flight_rel = flight_rel
        self.waypoint_indices = waypoint_indices
        self.unsafe_indices = unsafe_indices
        self.shadow_indices = shadow_indices
        self.start_abs_elev = start_abs_elev
        self.min_safe_relative = [g + min_clearance for g in ground_rel]
        self.relay_point = relay_point  # (dist, height) - расстояние в км и абсолютная высота
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
        
        max_dist = max(self.distances) if self.distances else 1
        all_heights = self.ground_rel + self.flight_rel + self.min_safe_relative
        max_height = max(all_heights) if all_heights else 100
        min_height = min(self.ground_rel) if self.ground_rel else 0
        
        if min_height > 0:
            min_height = 0
        
        height_range = max_height - min_height
        if height_range < 1:
            height_range = 1
        
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
        
        for i in range(len(flight_points) - 1):
            if i in self.shadow_indices or (i+1) in self.shadow_indices:
                pen = QPen(QColor(255, 0, 0), 4)
            else:
                pen = QPen(QColor(0, 200, 0), 3)
            painter.setPen(pen)
            painter.drawLine(flight_points[i], flight_points[i+1])
        
        # Рисуем линии связи от старта и ретранслятор
        if self.relay_point is not None:
            # relay_point = (dist, height_abs) 
            relay_dist = self.relay_point[0]  # расстояние от старта в км
            relay_height_abs = self.relay_point[1]  # абсолютная высота
            relay_height_rel = relay_height_abs - self.start_abs_elev  # относительная высота
            
            # Находим позицию на графике
            x_relay, y_relay = map_point(relay_dist, relay_height_rel)
            x_start, y_start = map_point(0, 0)
            
            # Рисуем линию от старта к ретранслятору
            painter.setPen(QPen(QColor(255, 165, 0, 180), 2, Qt.DashLine))
            painter.drawLine(x_start, y_start, x_relay, y_relay)
            
            # Рисуем точку ретранслятора на графике
            painter.setBrush(QBrush(QColor(255, 165, 0)))
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            painter.drawEllipse(x_relay - 8, y_relay - 8, 16, 16)
            painter.drawText(x_relay + 10, y_relay, "Ретранслятор")
            
            # Линии от ретранслятора к точкам маршрута
            for i, wp_idx in enumerate(self.waypoint_indices):
                if wp_idx < len(flight_points):
                    x_wp, y_wp = flight_points[wp_idx].x(), flight_points[wp_idx].y()
                    painter.setPen(QPen(QColor(255, 165, 0, 100), 1, Qt.DashLine))
                    painter.drawLine(x_relay, y_relay, x_wp, y_wp)
        
        # Точки маршрута
        for idx in self.waypoint_indices:
            if idx < len(flight_points):
                x, y = flight_points[idx].x(), flight_points[idx].y()
                
                if idx in self.shadow_indices:
                    color = QColor(255, 0, 0)
                elif idx in self.unsafe_indices:
                    color = QColor(255, 165, 0)
                else:
                    color = QColor(0, 255, 0)
                
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(0, 0, 0), 2))
                painter.drawEllipse(x - 5, y - 5, 10, 10)
                
                painter.setPen(QPen(QColor(0, 0, 0), 2))
                painter.setFont(QFont("Arial", 8, QFont.Bold))
                painter.drawText(x + 5, y - 5, str(self.waypoint_indices.index(idx) + 1))
        
        # Проблемные участки (подсветка)
        for idx in self.shadow_indices:
            if idx < len(flight_points):
                x, y = flight_points[idx].x(), flight_points[idx].y()
                painter.setPen(QPen(QColor(255, 0, 0, 150), 6))
                painter.drawEllipse(x - 5, y - 5, 10, 10)
        
        # Легенда
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.setFont(QFont("Arial", 8))
        legend_y = 20
        painter.drawText(10, legend_y, "Легенда:")
        
        painter.fillRect(70, legend_y - 8, 20, 10, QBrush(QColor(150, 150, 200)))
        painter.drawText(95, legend_y, "Рельеф (относительно старта)")
        legend_y += 15
        
        painter.setPen(QPen(QColor(0, 200, 0), 3))
        painter.drawLine(70, legend_y - 5, 90, legend_y - 5)
        painter.drawText(95, legend_y, "Траектория БПЛА (видимость)")
        legend_y += 15
        
        painter.setPen(QPen(QColor(255, 0, 0), 3))
        painter.drawLine(70, legend_y - 5, 90, legend_y - 5)
        painter.drawText(95, legend_y, "Траектория БПЛА (зона радиотени)")
        legend_y += 15
        
        painter.setPen(QPen(QColor(255, 100, 100), 2, Qt.DashLine))
        painter.drawLine(70, legend_y - 5, 90, legend_y - 5)
        painter.drawText(95, legend_y, "Мин. безопасная высота")
        legend_y += 15
        
        painter.setPen(QPen(QColor(100, 100, 255), 2, Qt.DashLine))
        painter.drawLine(70, legend_y - 5, 90, legend_y - 5)
        painter.drawText(95, legend_y, "Уровень старта (0)")
        legend_y += 15
        
        painter.setPen(QPen(QColor(255, 165, 0), 2, Qt.DashLine))
        painter.drawLine(70, legend_y - 5, 90, legend_y - 5)
        painter.drawText(95, legend_y, "Линии связи")
        
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
        self.shadow_indices = []
        self.relay_point = None
        
        self.search_thread = None
        self.progress_dialog = None
        
        self.setup_ui()
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        
        load_group = QGroupBox("Загрузка карты")
        load_layout = QVBoxLayout()
        self.load_btn = QPushButton("Загрузить сохранённую карту")
        self.load_btn.clicked.connect(self.load_map)
        load_layout.addWidget(self.load_btn)
        load_group.setLayout(load_layout)
        left_layout.addWidget(load_group)
        
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
        
        radio_group = QGroupBox("Параметры радиосвязи")
        radio_layout = QVBoxLayout()
        
        radio_layout.addWidget(QLabel("Высота антенны оператора, м:"))
        self.operator_antenna_spin = QDoubleSpinBox()
        self.operator_antenna_spin.setRange(0, 5)
        self.operator_antenna_spin.setValue(2)
        self.operator_antenna_spin.setSuffix(" м")
        self.operator_antenna_spin.setSingleStep(0.1)
        self.operator_antenna_spin.valueChanged.connect(self.on_flight_params_changed)
        radio_layout.addWidget(self.operator_antenna_spin)
        
        radio_layout.addWidget(QLabel("Высота антенны ретранслятора, м:"))
        self.relay_antenna_spin = QDoubleSpinBox()
        self.relay_antenna_spin.setRange(0, 5)
        self.relay_antenna_spin.setValue(2)
        self.relay_antenna_spin.setSuffix(" м")
        self.relay_antenna_spin.setSingleStep(0.1)
        self.relay_antenna_spin.valueChanged.connect(self.on_flight_params_changed)
        radio_layout.addWidget(self.relay_antenna_spin)
        
        radio_group.setLayout(radio_layout)
        left_layout.addWidget(radio_group)
        
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
        
        relay_group = QGroupBox("Поиск ретранслятора")
        relay_layout = QVBoxLayout()
        
        self.find_relay_btn = QPushButton("Найти место для ретранслятора")
        self.find_relay_btn.clicked.connect(self.find_relay)
        self.find_relay_btn.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; padding: 8px; }")
        relay_layout.addWidget(self.find_relay_btn)
        
        self.relay_info_label = QLabel("Ретранслятор не найден")
        self.relay_info_label.setStyleSheet("color: gray; font-weight: bold;")
        self.relay_info_label.setWordWrap(True)
        relay_layout.addWidget(self.relay_info_label)
        
        relay_group.setLayout(relay_layout)
        left_layout.addWidget(relay_group)
        
        left_layout.addStretch()
        left_panel.setMaximumWidth(320)
        
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
        abs_elev = self.map_widget.get_elevation_at(lat, lon)
        self.start_point = (lat, lon)
        self.start_abs_elev = abs_elev
        self.map_widget.set_start_point(lat, lon)
        
        self.start_info_label.setText(f"Старт: {lat:.5f}, {lon:.5f} | Высота: {abs_elev:.0f} м над уровнем моря")
        self.start_info_label.setStyleSheet("color: green; font-weight: bold;")
        
        self.statusBar().showMessage(f"Точка старта установлена. Абсолютная высота: {abs_elev:.0f} м")
        
        QMessageBox.information(self, "Старт установлен", 
                               f"Точка старта:\n"
                               f"Широта: {lat:.5f}\n"
                               f"Долгота: {lon:.5f}\n"
                               f"Абсолютная высота: {abs_elev:.0f} м над уровнем моря")
    
    def add_waypoint(self, lat, lon):
        if self.start_point is None:
            QMessageBox.warning(self, "Ошибка", "Сначала установите точку старта")
            return
        
        ground_rel = self.map_widget.get_relative_ground_elevation(lat, lon)
        flight_rel = self.altitude_spin.value()
        clearance = flight_rel - ground_rel
        
        self.waypoints.append([lat, lon, flight_rel])
        self.map_widget.add_waypoint(lat, lon, flight_rel)
        
        item = QListWidgetItem(f"Точка {len(self.waypoints)}: {lat:.5f}, {lon:.5f}")
        item.setToolTip(f"Высота земли: {ground_rel:.0f} м отн.\nВысота полета: {flight_rel:.0f} м отн.\nЗазор: {clearance:.0f} м")
        self.points_list.addItem(item)
        
        self.statusBar().showMessage(f"Добавлена точка {len(self.waypoints)}")
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
        self.relay_point = None
        self.map_widget.relay_point = None
        self.shadow_indices = []
        self.relay_info_label.setText("Ретранслятор не найден")
        self.relay_info_label.setStyleSheet("color: gray; font-weight: bold;")
        self.update_profile()
    
    def on_point_selected(self):
        selected = self.points_list.currentRow()
        if selected >= 0:
            self.map_widget.selected_point = selected
            self.map_widget.update()
    
    def update_profile(self):
        if len(self.waypoints) < 2 or self.start_point is None:
            self.profile_widget.set_data([], [], [], [], [], [], 0, 0)
            return
        
        distances, ground_rel, flight_rel, waypoint_indices = self.map_widget.get_trajectory_profile()
        
        unsafe_indices = []
        for i in range(len(flight_rel)):
            if flight_rel[i] - ground_rel[i] < self.min_clearance_spin.value():
                unsafe_indices.append(i)
        
        shadow_indices = self.map_widget.check_shadow_zone(self.operator_antenna_spin.value())
        self.shadow_indices = shadow_indices
        
        relay_info = None
        if self.relay_point is not None:
            start_lat, start_lon = self.start_point
            relay_lat, relay_lon = self.relay_point
            relay_dist = calculate_distance(start_lat, start_lon, relay_lat, relay_lon) / 1000
            relay_height = self.map_widget.get_elevation_at(relay_lat, relay_lon)
            relay_info = (relay_dist, relay_height)
        
        self.profile_widget.set_data(distances, ground_rel, flight_rel, 
                                     waypoint_indices, unsafe_indices, shadow_indices,
                                     self.start_abs_elev, self.min_clearance_spin.value(),
                                     relay_info)
        
        shadow_waypoints = set()
        for idx in shadow_indices:
            for i, wp_idx in enumerate(waypoint_indices):
                if idx <= wp_idx:
                    shadow_waypoints.add(i)
                    break
        
        self.map_widget.set_shadow_points(shadow_waypoints)
        
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
        
        alt_violations = []
        for i in range(len(flight_rel)):
            clearance = flight_rel[i] - ground_rel[i]
            if clearance < self.min_clearance_spin.value():
                dist = distances[i] if i < len(distances) else 0
                alt_violations.append({
                    'dist': dist,
                    'ground': ground_rel[i],
                    'flight': flight_rel[i],
                    'clearance': clearance,
                    'required': self.min_clearance_spin.value(),
                    'deficit': self.min_clearance_spin.value() - clearance
                })
        
        shadow_indices = self.map_widget.check_shadow_zone(self.operator_antenna_spin.value())
        self.shadow_indices = shadow_indices
        self.update_profile()
        
        if alt_violations or shadow_indices:
            self.result_text.setStyleSheet("color: red;")
            msg = "❌ МАРШРУТ НЕБЕЗОПАСЕН!\n\n"
            
            if alt_violations:
                msg += f"Найдено {len(alt_violations)} нарушений минимального расстояния до земли.\n"
                msg += f"Требуется зазор: {self.min_clearance_spin.value()} м\n\n"
            
            if shadow_indices:
                msg += f"🔴 НАЙДЕНА ЗОНА РАДИОТЕНИ!\n"
                msg += f"На {len(shadow_indices)} участках траектории теряется прямая видимость.\n\n"
            
            msg += "Проблемные участки:\n"
            if alt_violations:
                for v in alt_violations[:5]:
                    msg += f"• Высота: на {v['dist']:.1f} км, земля={v['ground']:.0f} м, "
                    msg += f"БПЛА={v['flight']:.0f} м, зазор={v['clearance']:.0f} м "
                    msg += f"(не хватает {v['deficit']:.0f} м)\n"
            
            if shadow_indices:
                msg += f"\n• Зона радиотени обнаружена на траектории.\n"
                if self.relay_point:
                    msg += f"✅ Предлагаемое место для ретранслятора:\n"
                    msg += f"  Широта: {self.relay_point[0]:.5f}\n"
                    msg += f"  Долгота: {self.relay_point[1]:.5f}\n"
                    msg += f"  Расстояние от старта: {calculate_distance(self.start_point[0], self.start_point[1], self.relay_point[0], self.relay_point[1]):.0f} м\n"
            
            self.result_text.setText(msg)
            self.statusBar().showMessage("Маршрут небезопасен")
            QMessageBox.warning(self, "Результат проверки", msg)
        else:
            self.result_text.setStyleSheet("color: green;")
            min_clearance = min([flight_rel[i] - ground_rel[i] for i in range(len(flight_rel))])
            msg = f"✅ МАРШРУТ БЕЗОПАСЕН!\n\n"
            msg += f"Минимальный зазор: {min_clearance:.0f} м\n"
            msg += f"Требуемый зазор: {self.min_clearance_spin.value()} м\n"
            msg += f"Радиосвязь: прямая видимость на всей траектории\n"
            self.result_text.setText(msg)
            self.statusBar().showMessage("Маршрут безопасен")
            QMessageBox.information(self, "Результат проверки", msg)
    
    def find_relay(self):
        """Поиск оптимального места для ретранслятора"""
        try:
            if self.start_point is None:
                QMessageBox.warning(self, "Ошибка", "Сначала установите точку старта")
                return
            
            if len(self.waypoints) < 2:
                QMessageBox.warning(self, "Ошибка", "Добавьте минимум 2 точки маршрута")
                return
            
            if self.search_thread and self.search_thread.isRunning():
                QMessageBox.warning(self, "Информация", "Поиск уже выполняется")
                return
            
            self.find_relay_btn.setEnabled(False)
            self.find_relay_btn.setText("Поиск...")
            
            self.progress_dialog = QProgressDialog("Поиск места для ретранслятора...", "Отмена", 0, 100, self)
            self.progress_dialog.setWindowTitle("Поиск ретранслятора")
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.canceled.connect(self.cancel_relay_search)
            
            self.search_thread = RelaySearchThread(
                self.map_widget,
                self.start_point,
                self.start_abs_elev,
                self.waypoints,
                self.operator_antenna_spin.value(),
                self.relay_antenna_spin.value()
            )
            
            self.search_thread.progress.connect(self.update_progress)
            self.search_thread.status.connect(self.update_status)
            self.search_thread.finished.connect(self.on_relay_search_finished)
            self.search_thread.error.connect(self.on_relay_search_error)
            
            self.search_thread.start()
            
        except Exception as e:
            error_msg = f"Ошибка при запуске поиска:\n{str(e)}\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Ошибка", error_msg)
            self.find_relay_btn.setEnabled(True)
            self.find_relay_btn.setText("Найти место для ретранслятора")
    
    def update_progress(self, value):
        if self.progress_dialog:
            self.progress_dialog.setValue(value)
    
    def update_status(self, text):
        if self.progress_dialog:
            self.progress_dialog.setLabelText(text)
    
    def cancel_relay_search(self):
        if self.search_thread and self.search_thread.isRunning():
            self.search_thread.stop()
            self.search_thread.wait()
        
        self.find_relay_btn.setEnabled(True)
        self.find_relay_btn.setText("Найти место для ретранслятора")
        
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
    
    def on_relay_search_finished(self, relay_pos):
        try:
            self.find_relay_btn.setEnabled(True)
            self.find_relay_btn.setText("Найти место для ретранслятора")
            
            if self.progress_dialog:
                self.progress_dialog.close()
                self.progress_dialog = None
            
            if relay_pos:
                self.relay_point = relay_pos
                self.map_widget.set_relay_point(relay_pos[0], relay_pos[1])
                
                start_lat, start_lon = self.start_point
                dist = calculate_distance(start_lat, start_lon, relay_pos[0], relay_pos[1])
                elev = self.map_widget.get_elevation_at(relay_pos[0], relay_pos[1])
                
                self.relay_info_label.setText(
                    f"✅ Ретранслятор найден!\n"
                    f"Широта: {relay_pos[0]:.5f}\n"
                    f"Долгота: {relay_pos[1]:.5f}\n"
                    f"Расстояние: {dist:.0f} м\n"
                    f"Высота: {elev:.0f} м"
                )
                self.relay_info_label.setStyleSheet("color: green; font-weight: bold;")
                
                self.update_profile()
                
                QMessageBox.information(self, "Ретранслятор найден",
                                       f"Оптимальное место для ретранслятора:\n\n"
                                       f"Широта: {relay_pos[0]:.5f}\n"
                                       f"Долгота: {relay_pos[1]:.5f}\n"
                                       f"Расстояние от старта: {dist:.0f} м\n"
                                       f"Высота над уровнем моря: {elev:.0f} м")
            else:
                self.relay_info_label.setText("❌ Не удалось найти подходящее место для ретранслятора\nПопробуйте изменить параметры антенн")
                self.relay_info_label.setStyleSheet("color: red; font-weight: bold;")
                QMessageBox.warning(self, "Ретранслятор не найден",
                                   "Не удалось найти подходящее место для установки ретранслятора.\n"
                                   "Попробуйте:\n"
                                   "- Увеличить высоту антенн\n"
                                   "- Изменить маршрут\n"
                                   "- Сократить расстояние")
        except Exception as e:
            error_msg = f"Ошибка при обработке результата:\n{str(e)}\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Ошибка", error_msg)
    
    def on_relay_search_error(self, error_msg):
        self.find_relay_btn.setEnabled(True)
        self.find_relay_btn.setText("Найти место для ретранслятора")
        
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        QMessageBox.critical(self, "Ошибка", f"Ошибка при поиске ретранслятора:\n{error_msg}")
    
    def closeEvent(self, event):
        if self.search_thread and self.search_thread.isRunning():
            self.search_thread.stop()
            self.search_thread.wait()
        
        try:
            if self.start_point and self.waypoints:
                config = {
                    'start_point': self.start_point,
                    'waypoints': self.waypoints,
                    'start_abs_elev': self.start_abs_elev,
                    'relay_point': self.relay_point,
                    'altitude': self.altitude_spin.value(),
                    'min_clearance': self.min_clearance_spin.value(),
                    'operator_antenna': self.operator_antenna_spin.value(),
                    'relay_antenna': self.relay_antenna_spin.value()
                }
                with open('route_config.json', 'w') as f:
                    json.dump(config, f, indent=2)
        except:
            pass
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RoutePlanner()
    window.show()
    sys.exit(app.exec_())