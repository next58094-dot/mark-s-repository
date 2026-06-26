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
                             QSplitter, QTabWidget, QTextEdit, QProgressDialog,
                             QDialog, QDialogButtonBox, QFormLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QRectF, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QImage, QFont, QBrush, QPolygonF
import traceback
import math

# Константы
OPENTOPOGRAPHY_API_URL = "https://portal.opentopography.org/API/globaldem"
DEM_DATASETS = {
    "SRTMGL3": "SRTM GL3 90m",
    "SRTMGL1": "SRTM GL1 30m", 
    "NASADEM": "NASADEM Global DEM 30m",
}
R_EARTH = 6371000  # Радиус Земли в метрах

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

def smoothstep(t):
    """Smoothstep интерполяция для плавного перехода"""
    return t * t * (3 - 2 * t)

def smooth_interpolate(v1, v2, t):
    """Плавная интерполяция между двумя значениями"""
    t_smooth = smoothstep(t)
    return v1 + (v2 - v1) * t_smooth

class AltitudeInputDialog(QDialog):
    """Диалог для ввода высоты полета для точки маршрута"""
    def __init__(self, point_index, default_altitude=100, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Высота для точки {point_index + 1}")
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        info_label = QLabel(f"Введите высоту полета для точки {point_index + 1}\n(относительно точки старта, где 0 = высота старта)")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        form_layout = QFormLayout()
        self.altitude_spin = QDoubleSpinBox()
        self.altitude_spin.setRange(0, 2000)
        self.altitude_spin.setValue(default_altitude)
        self.altitude_spin.setSuffix(" м")
        form_layout.addRow("Высота полета:", self.altitude_spin)
        layout.addLayout(form_layout)
        
        # Добавляем информацию о текущей высоте земли
        self.ground_info = QLabel()
        self.ground_info.setStyleSheet("color: gray; font-size: 9pt;")
        layout.addWidget(self.ground_info)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def get_altitude(self):
        return self.altitude_spin.value()
    
    def set_ground_info(self, ground_abs, start_abs):
        """Установка информации о высоте земли"""
        rel_ground = ground_abs - start_abs
        self.ground_info.setText(f"Высота земли в этой точке: {rel_ground:.0f} м относительно старта ({ground_abs:.0f} м над уровнем моря)")

class RelaySearchThread(QThread):
    """Поток для поиска ретранслятора на наивысшей точке"""
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
        
        # Кэш для высот
        self.elevation_cache = {}
        
    def stop(self):
        self._is_running = False
    
    def get_elevation_at(self, lat, lon):
        """Получение высоты из DEM с кэшированием и проверкой границ"""
        cache_key = (round(lat, 6), round(lon, 6))
        if cache_key in self.elevation_cache:
            return self.elevation_cache[cache_key]
        
        if self.dem_array is None or self.bounds is None:
            return 0
        
        min_lon, max_lon, min_lat, max_lat = self.bounds
        h, w = self.dem_array.shape
        
        # Проверка границ
        if lat < min_lat or lat > max_lat or lon < min_lon or lon > max_lon:
            self.elevation_cache[cache_key] = 0
            return 0
        
        x = (lon - min_lon) / (max_lon - min_lon) * (w - 1)
        y = (max_lat - lat) / (max_lat - min_lat) * (h - 1)
        
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        
        # Проверка выхода за границы массива
        if x0 < 0 or x0 >= w or y0 < 0 or y0 >= h:
            self.elevation_cache[cache_key] = 0
            return 0
        
        dx, dy = x - x0, y - y0
        elev = (1 - dx) * (1 - dy) * self.dem_array[y0, x0] + \
               dx * (1 - dy) * self.dem_array[y0, x1] + \
               (1 - dx) * dy * self.dem_array[y1, x0] + \
               dx * dy * self.dem_array[y1, x1]
        
        result = float(elev)
        self.elevation_cache[cache_key] = result
        return result
    
    def check_line_of_sight_fast(self, lat1, lon1, alt1, lat2, lon2, alt2, antenna_height=2):
        """Быстрая проверка прямой видимости"""
        if self.dem_array is None or self.bounds is None:
            return True
        
        h1 = alt1 + antenna_height
        h2 = alt2 + antenna_height
        
        distance = calculate_distance(lat1, lon1, lat2, lon2)
        num_points = max(10, int(distance / 20))
        points = interpolate_points(lat1, lon1, lat2, lon2, num_points)
        
        for i in range(1, len(points) - 1, 2):
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
            
            self.status.emit("Генерация траектории...")
            self.progress.emit(10)
            
            # Получаем все точки траектории с детализацией
            route_points = []
            for i in range(len(self.waypoints) - 1):
                lat1, lon1, alt_rel1 = self.waypoints[i]
                lat2, lon2, alt_rel2 = self.waypoints[i+1]
                steps = max(20, int(calculate_distance(lat1, lon1, lat2, lon2) / 10))
                
                for j in range(steps + 1):
                    t = j / steps
                    lat = lat1 + (lat2 - lat1) * t
                    lon = lon1 + (lon2 - lon1) * t
                    # Используем плавную интерполяцию высоты
                    alt_rel = smooth_interpolate(alt_rel1, alt_rel2, t)
                    alt_abs = start_abs + alt_rel
                    dist_from_start = calculate_distance(start_lat, start_lon, lat, lon)
                    route_points.append((lat, lon, alt_abs, dist_from_start))
            
            if not route_points:
                self.finished.emit(None)
                return
            
            self.status.emit("Поиск зон радиотени...")
            self.progress.emit(20)
            
            # Находим все зоны радиотени
            shadow_zones = []
            current_zone = []
            in_shadow = False
            
            for idx in range(0, len(route_points), 3):
                if not self._is_running:
                    self.finished.emit(None)
                    return
                
                lat, lon, alt_abs, dist_from_start = route_points[idx]
                
                has_los = self.check_line_of_sight_fast(start_lat, start_lon, start_abs, 
                                                       lat, lon, alt_abs, self.operator_antenna_height)
                
                if not has_los and not in_shadow:
                    in_shadow = True
                    current_zone = [(lat, lon, alt_abs, dist_from_start)]
                elif not has_los and in_shadow:
                    current_zone.append((lat, lon, alt_abs, dist_from_start))
                elif has_los and in_shadow:
                    in_shadow = False
                    if len(current_zone) > 5:
                        shadow_zones.append(current_zone)
                    current_zone = []
                
                if idx % 20 == 0:
                    self.progress.emit(20 + int(idx / len(route_points) * 30))
            
            if in_shadow and len(current_zone) > 5:
                shadow_zones.append(current_zone)
            
            if not shadow_zones:
                self.status.emit("Зона радиотени не обнаружена")
                self.finished.emit(None)
                return
            
            self.status.emit(f"Найдено {len(shadow_zones)} зон радиотени")
            self.progress.emit(50)
            
            # Выбираем первую зону радиотени (ближайшую к старту)
            first_shadow_zone = shadow_zones[0]
            
            zone_start = first_shadow_zone[0]
            zone_end = first_shadow_zone[-1]
            
            start_lat_shadow, start_lon_shadow, start_alt_shadow, start_dist = zone_start
            end_lat_shadow, end_lon_shadow, end_alt_shadow, end_dist = zone_end
            
            self.status.emit(f"Первая зона радиотени: от {start_dist/1000:.1f} до {end_dist/1000:.1f} км")
            self.progress.emit(60)
            
            # Ищем наивысшую точку в районе зоны радиотени
            best_relay = None
            best_score = -1
            
            # Расширяем область поиска
            search_start = max(0, start_dist - 300)
            search_end = end_dist + 300
            
            # Находим все кандидаты в области
            candidates = []
            for idx in range(0, len(route_points), 2):
                if not self._is_running:
                    self.finished.emit(None)
                    return
                
                lat, lon, alt_abs, dist_from_start = route_points[idx]
                
                if dist_from_start < search_start or dist_from_start > search_end:
                    continue
                
                # Проверяем видимость от старта до этой точки
                if not self.check_line_of_sight_fast(start_lat, start_lon, start_abs, 
                                                    lat, lon, alt_abs, self.operator_antenna_height):
                    continue
                
                # Получаем высоту земли
                ground_alt = self.get_elevation_at(lat, lon)
                # Ретранслятор должен быть над землей, а не в ней
                relay_height_above_ground = alt_abs - ground_alt
                if relay_height_above_ground < 0:
                    continue  # Точка под землей - пропускаем
                candidates.append((lat, lon, ground_alt, relay_height_above_ground, dist_from_start))
                
                if idx % 10 == 0:
                    self.progress.emit(60 + int(idx / len(route_points) * 20))
            
            if not candidates:
                self.status.emit("Не найдено подходящих кандидатов")
                self.finished.emit(None)
                return
            
            # Сортируем кандидатов по высоте над землей (от самой высокой)
            candidates.sort(key=lambda x: x[3], reverse=True)
            
            self.status.emit(f"Найдено {len(candidates)} кандидатов. Выбор наивысшей точки...")
            self.progress.emit(80)
            
            # Проверяем кандидатов от самой высокой к низкой
            for lat, lon, ground_alt, relay_height, dist_from_start in candidates[:50]:
                if not self._is_running:
                    self.finished.emit(None)
                    return
                
                # Высота ретранслятора = высота земли + высота над землей + антенна
                relay_abs_alt = ground_alt + relay_height + self.relay_antenna_height
                
                # Проверяем видимость от этой точки до зоны тени
                visible_count = 0
                check_count = min(30, len(first_shadow_zone))
                for shadow_idx in range(0, len(first_shadow_zone), max(1, len(first_shadow_zone) // check_count)):
                    shadow_lat, shadow_lon, shadow_alt, _ = first_shadow_zone[shadow_idx]
                    if self.check_line_of_sight_fast(lat, lon, relay_abs_alt, 
                                                    shadow_lat, shadow_lon, shadow_alt, 
                                                    0):  # Антенна ретранслятора уже учтена
                        visible_count += 1
                
                coverage = visible_count / check_count if check_count > 0 else 0
                
                # Оценка: высота над землей + покрытие
                height_factor = relay_height / 50  # Нормализуем
                score = coverage * 0.6 + height_factor * 0.4
                
                if score > best_score:
                    best_score = score
                    best_relay = (lat, lon)
                
                # Если покрытие > 80%, останавливаемся
                if coverage > 0.8:
                    break
                
                if idx % 5 == 0:
                    self.progress.emit(80 + int(idx / len(candidates) * 20))
            
            # Если не нашли, ставим на самую высокую точку
            if best_relay is None:
                self.status.emit("Установка ретранслятора на наивысшую точку...")
                if candidates:
                    best_relay = (candidates[0][0], candidates[0][1])
                else:
                    best_relay = (start_lat_shadow, start_lon_shadow)
            
            self.status.emit(f"Поиск завершен. Наивысшая точка найдена на {calculate_distance(start_lat, start_lon, best_relay[0], best_relay[1])/1000:.1f} км")
            self.progress.emit(100)
            
            self.finished.emit(best_relay)
                
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)
            self.finished.emit(None)

class MapWidget(QWidget):
    """Виджет карты с поддержкой перетаскивания точек и ретранслятора"""
    point_clicked = pyqtSignal(float, float)
    point_moved = pyqtSignal(int, float, float)
    relay_moved = pyqtSignal(float, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setMinimumSize(600, 500)
        self.setStyleSheet("background-color: #2b2b2b;")
        
        self.satellite_image = None
        self.dem_array = None
        self.bounds = None
        self.waypoints = []
        self.start_point = None
        self.selected_point = -1
        self.dragging = False
        self.dragging_relay = False
        self.unsafe_points = set()
        self.shadow_points = set()
        self.shadow_after_relay_points = set()
        self.relay_point = None
        self.first_waypoint_distance = 0
        self.current_distances = []  # Расстояния вдоль траектории
        self.manual_relay_mode = False
        self.trajectory_distances = []  # Накопленные расстояния вдоль траектории
        
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
        if len(self.waypoints) == 1 and self.start_point is not None:
            start_lat, start_lon, _ = self.start_point
            self.first_waypoint_distance = calculate_distance(start_lat, start_lon, lat, lon)
        self.update()
    
    def get_elevation_at(self, lat, lon):
        """Получение высоты из DEM с проверкой границ"""
        if self.dem_array is None or self.bounds is None:
            return 0
        
        min_lon, max_lon, min_lat, max_lat = self.bounds
        
        # Проверка границ
        if lat < min_lat or lat > max_lat or lon < min_lon or lon > max_lon:
            return 0
        
        h, w = self.dem_array.shape
        
        x = (lon - min_lon) / (max_lon - min_lon) * (w - 1)
        y = (max_lat - lat) / (max_lat - min_lat) * (h - 1)
        
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        
        # Проверка выхода за границы массива
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
        """Проверка прямой видимости с учетом рельефа"""
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
    
    def calculate_relay_radius(self, relay_lat, relay_lon, relay_abs_alt, start_lat, start_lon):
        """
        Расчет радиуса действия ретранслятора с учетом высоты антенны.
        Использует формулу оптической дальности.
        """
        # Высота ретранслятора над землей
        ground_alt = self.get_elevation_at(relay_lat, relay_lon)
        height_above_ground = relay_abs_alt - ground_alt
        
        # Убеждаемся что высота не отрицательная
        if height_above_ground < 0:
            height_above_ground = 0
        
        # Общая высота с учетом антенны
        antenna_height = self.parent_window.relay_antenna_spin.value() if self.parent_window else 2
        total_height = height_above_ground + antenna_height
        
        # Расстояние до старта в метрах
        dist_to_start = calculate_distance(start_lat, start_lon, relay_lat, relay_lon)
        
        # Формула оптической дальности с учетом кривизны Земли
        optical_radius = sqrt(2 * R_EARTH * total_height)
        
        # Ограничиваем радиус: не менее расстояния до старта * 0.5 и не более расстояния * 2
        relay_radius = max(dist_to_start * 0.5, min(optical_radius, dist_to_start * 2))
        
        return relay_radius
    
    def check_visibility_with_relay(self, operator_antenna_height=2, relay_antenna_height=2):
        """Проверка видимости с учетом ретранслятора по радиусу"""
        if self.start_point is None or len(self.waypoints) < 2:
            return [], []
        
        shadow_indices = []
        shadow_after_relay = []
        start_lat, start_lon, start_abs = self.start_point
        
        distances, ground_rel, flight_rel, waypoint_indices = self.get_trajectory_profile()
        
        # Если ретранслятор не установлен - проверяем только от старта
        if self.relay_point is None:
            for i in range(len(flight_rel)):
                if len(self.waypoints) > 1:
                    total_points = len(flight_rel)
                    total_segments = len(self.waypoints) - 1
                    progress = i / total_points if total_points > 0 else 0
                    segment_idx = min(int(progress * total_segments), total_segments - 1)
                    
                    if segment_idx < len(self.waypoints) - 1:
                        lat1, lon1, alt1 = self.waypoints[segment_idx]
                        lat2, lon2, alt2 = self.waypoints[segment_idx + 1]
                        
                        segment_start = segment_idx / total_segments if total_segments > 0 else 0
                        segment_end = (segment_idx + 1) / total_segments if total_segments > 0 else 0
                        segment_duration = segment_end - segment_start
                        
                        if segment_duration > 0:
                            t = (progress - segment_start) / segment_duration
                            t = max(0, min(1, t))
                        else:
                            t = 0
                        
                        lat = lat1 + (lat2 - lat1) * t
                        lon = lon1 + (lon2 - lon1) * t
                        alt = start_abs + smooth_interpolate(alt1, alt2, t)
                    else:
                        lat, lon, alt = self.waypoints[-1][0], self.waypoints[-1][1], start_abs + flight_rel[i]
                else:
                    continue
                
                if not self.check_line_of_sight(start_lat, start_lon, start_abs, 
                                               lat, lon, alt, operator_antenna_height):
                    shadow_indices.append(i)
            return shadow_indices, []
        
        # Если ретранслятор установлен - проверяем видимость от старта И от ретранслятора
        relay_lat, relay_lon = self.relay_point
        
        # Получаем абсолютную высоту земли под ретранслятором
        relay_ground_alt = self.get_elevation_at(relay_lat, relay_lon)
        
        # Высота ретранслятора = высота земли + высота над землей + антенна
        # Используем высоту ретранслятора, переданную из UI
        relay_height_above_ground = 0  # Будет вычислена из относительной высоты
        
        # Ищем относительную высоту ретранслятора из точек маршрута или используем высоту над землей
        # По умолчанию используем высоту ретранслятора как высоту земли + 50м (минимальная высота)
        if self.parent_window and hasattr(self.parent_window, 'relay_altitude'):
            relay_height_above_ground = self.parent_window.relay_altitude
        else:
            # Если высота не сохранена, используем высоту земли + 50м
            relay_height_above_ground = 50
        
        # Убеждаемся что ретранслятор находится над землей
        if relay_height_above_ground < 0:
            relay_height_above_ground = 0
        
        # Абсолютная высота ретранслятора (с учетом антенны)
        relay_abs = relay_ground_alt + relay_height_above_ground + relay_antenna_height
        
        # Проверяем видимость от старта до ретранслятора (ретранслятор находится над землей)
        relay_visible_from_start = self.check_line_of_sight(
            start_lat, start_lon, start_abs,
            relay_lat, relay_lon, relay_abs,  # relay_abs уже включает высоту над землей и антенну
            operator_antenna_height
        )
        
        # Если ретранслятор не виден от старта - он бесполезен
        if not relay_visible_from_start:
            # Используем обычную проверку без ретранслятора
            for i in range(len(flight_rel)):
                if len(self.waypoints) > 1:
                    total_points = len(flight_rel)
                    total_segments = len(self.waypoints) - 1
                    progress = i / total_points if total_points > 0 else 0
                    segment_idx = min(int(progress * total_segments), total_segments - 1)
                    
                    if segment_idx < len(self.waypoints) - 1:
                        lat1, lon1, alt1 = self.waypoints[segment_idx]
                        lat2, lon2, alt2 = self.waypoints[segment_idx + 1]
                        
                        segment_start = segment_idx / total_segments if total_segments > 0 else 0
                        segment_end = (segment_idx + 1) / total_segments if total_segments > 0 else 0
                        segment_duration = segment_end - segment_start
                        
                        if segment_duration > 0:
                            t = (progress - segment_start) / segment_duration
                            t = max(0, min(1, t))
                        else:
                            t = 0
                        
                        lat = lat1 + (lat2 - lat1) * t
                        lon = lon1 + (lon2 - lon1) * t
                        alt = start_abs + smooth_interpolate(alt1, alt2, t)
                    else:
                        lat, lon, alt = self.waypoints[-1][0], self.waypoints[-1][1], start_abs + flight_rel[i]
                else:
                    continue
                
                if not self.check_line_of_sight(start_lat, start_lon, start_abs, 
                                               lat, lon, alt, operator_antenna_height):
                    shadow_indices.append(i)
            return shadow_indices, []
        
        dist_to_relay = calculate_distance(start_lat, start_lon, relay_lat, relay_lon)
        
        # Расчет радиуса действия ретранслятора
        relay_radius = self.calculate_relay_radius(relay_lat, relay_lon, relay_abs, start_lat, start_lon)
        relay_radius_km = relay_radius / 1000
        
        for i in range(len(flight_rel)):
            if len(self.waypoints) > 1:
                total_points = len(flight_rel)
                total_segments = len(self.waypoints) - 1
                progress = i / total_points if total_points > 0 else 0
                segment_idx = min(int(progress * total_segments), total_segments - 1)
                
                if segment_idx < len(self.waypoints) - 1:
                    lat1, lon1, alt1 = self.waypoints[segment_idx]
                    lat2, lon2, alt2 = self.waypoints[segment_idx + 1]
                    
                    segment_start = segment_idx / total_segments if total_segments > 0 else 0
                    segment_end = (segment_idx + 1) / total_segments if total_segments > 0 else 0
                    segment_duration = segment_end - segment_start
                    
                    if segment_duration > 0:
                        t = (progress - segment_start) / segment_duration
                        t = max(0, min(1, t))
                    else:
                        t = 0
                    
                    lat = lat1 + (lat2 - lat1) * t
                    lon = lon1 + (lon2 - lon1) * t
                    alt = start_abs + smooth_interpolate(alt1, alt2, t)
                else:
                    lat, lon, alt = self.waypoints[-1][0], self.waypoints[-1][1], start_abs + flight_rel[i]
            else:
                continue
            
            dist_from_start = calculate_distance(start_lat, start_lon, lat, lon)
            dist_from_relay = calculate_distance(relay_lat, relay_lon, lat, lon)
            
            # Проверяем видимость от старта
            has_los_from_start = self.check_line_of_sight(start_lat, start_lon, start_abs, 
                                                         lat, lon, alt, operator_antenna_height)
            
            # Проверяем видимость от ретранслятора (ретранслятор уже над землей)
            has_los_from_relay = self.check_line_of_sight(relay_lat, relay_lon, relay_abs, 
                                                         lat, lon, alt, 0)  # Антенна уже учтена в relay_abs
            
            # Точка видна, если есть видимость от старта ИЛИ от ретранслятора
            # И точка находится в радиусе действия ретранслятора
            if dist_from_relay <= relay_radius:
                has_visibility = has_los_from_start or has_los_from_relay
            else:
                has_visibility = has_los_from_start
            
            # Определяем зону тени
            if dist_from_start <= dist_to_relay and not has_visibility:
                shadow_indices.append(i)
            elif dist_from_start > dist_to_relay and not has_visibility:
                shadow_after_relay.append(i)
        
        return shadow_indices, shadow_after_relay
    
    def get_trajectory_profile(self):
        """Получение профиля траектории с корректным расчетом расстояний"""
        if len(self.waypoints) < 2 or self.start_point is None:
            return [], [], [], []
        
        distances = [0.0]
        ground_rel = [0.0]
        flight_rel = [0.0]
        waypoint_indices = [0]
        start_abs = self.start_point[2]
        start_lat, start_lon, _ = self.start_point
        
        # Массив для хранения расстояний вдоль траектории
        self.trajectory_distances = [0.0]
        
        # Расстояние до первой точки маршрута
        if len(self.waypoints) >= 1:
            first_lat, first_lon, _ = self.waypoints[0]
            self.first_waypoint_distance = calculate_distance(start_lat, start_lon, first_lat, first_lon)
        
        # Переменная для накопления расстояния
        accumulated_dist = 0.0
        
        for i in range(len(self.waypoints) - 1):
            lat1, lon1, alt_rel1 = self.waypoints[i]
            lat2, lon2, alt_rel2 = self.waypoints[i + 1]
            
            total_segment_dist = calculate_distance(lat1, lon1, lat2, lon2)
            steps = max(20, int(total_segment_dist / 30))
            
            # Сохраняем предыдущие координаты для расчета расстояния
            prev_lat, prev_lon = lat1, lon1
            
            for j in range(1, steps + 1):
                t = j / steps
                lat = lat1 + (lat2 - lat1) * t
                lon = lon1 + (lon2 - lon1) * t
                
                # Относительная высота земли
                ground_abs = self.get_elevation_at(lat, lon)
                ground_rel_val = ground_abs - start_abs
                ground_rel.append(ground_rel_val)
                
                # Относительная высота полета (плавная интерполяция)
                flight_rel_val = smooth_interpolate(alt_rel1, alt_rel2, t)
                flight_rel.append(flight_rel_val)
                
                # Расчет расстояния от предыдущей точки в метрах
                segment_dist = calculate_distance(prev_lat, prev_lon, lat, lon)
                accumulated_dist += segment_dist
                distances.append(accumulated_dist / 1000)  # В километрах
                self.trajectory_distances.append(accumulated_dist / 1000)  # В километрах
                
                prev_lat, prev_lon = lat, lon
            
            waypoint_indices.append(len(ground_rel) - 1)
        
        return distances, ground_rel, flight_rel, waypoint_indices
    
    def get_safe_indices(self, flight_rel, ground_rel, min_clearance):
        """Получение индексов безопасных точек (исключая взлетный участок)"""
        unsafe_indices = []
        
        # Расстояние до первой точки маршрута в км
        first_waypoint_dist_km = self.first_waypoint_distance / 1000
        
        for i in range(len(flight_rel)):
            # Пропускаем взлетный участок (от старта до первой точки)
            if i < len(self.trajectory_distances) and self.trajectory_distances[i] <= first_waypoint_dist_km:
                continue
            
            if flight_rel[i] - ground_rel[i] < min_clearance:
                unsafe_indices.append(i)
        
        return unsafe_indices
    
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
        
        # Рисуем зону радиотени ДО ретранслятора (красным)
        if self.shadow_points and len(self.waypoints) >= 2:
            painter.setPen(QPen(QColor(255, 0, 0, 150), 4, Qt.DashLine))
            for i in range(len(self.waypoints) - 1):
                x1, y1 = map_point(self.waypoints[i][0], self.waypoints[i][1])
                x2, y2 = map_point(self.waypoints[i+1][0], self.waypoints[i+1][1])
                if i in self.shadow_points or (i+1) in self.shadow_points:
                    painter.drawLine(x1, y1, x2, y2)
        
        # Рисуем зону радиотени ПОСЛЕ ретранслятора (фиолетовым)
        if self.shadow_after_relay_points and len(self.waypoints) >= 2:
            painter.setPen(QPen(QColor(255, 0, 255, 150), 3, Qt.DashDotLine))
            for i in range(len(self.waypoints) - 1):
                x1, y1 = map_point(self.waypoints[i][0], self.waypoints[i][1])
                x2, y2 = map_point(self.waypoints[i+1][0], self.waypoints[i+1][1])
                if i in self.shadow_after_relay_points or (i+1) in self.shadow_after_relay_points:
                    painter.drawLine(x1, y1, x2, y2)
        
        # Рисуем траекторию (зеленым)
        if len(self.waypoints) >= 2:
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)
            for i in range(len(self.waypoints) - 1):
                x1, y1 = map_point(self.waypoints[i][0], self.waypoints[i][1])
                x2, y2 = map_point(self.waypoints[i+1][0], self.waypoints[i+1][1])
                painter.drawLine(x1, y1, x2, y2)
        
        # Рисуем ретранслятор
        if self.relay_point:
            x, y = map_point(self.relay_point[0], self.relay_point[1])
            
            # Проверяем, находится ли ретранслятор над землей
            ground_alt = self.get_elevation_at(self.relay_point[0], self.relay_point[1])
            relay_abs_alt = self.get_elevation_at(self.relay_point[0], self.relay_point[1])
            if self.parent_window and hasattr(self.parent_window, 'relay_altitude'):
                relay_abs_alt = ground_alt + self.parent_window.relay_altitude
            
            is_above_ground = relay_abs_alt > ground_alt + 1  # хотя бы 1м над землей
            
            if is_above_ground:
                painter.setBrush(QBrush(QColor(255, 165, 0)))
            else:
                painter.setBrush(QBrush(QColor(255, 0, 0, 150)))  # Красный - ретранслятор в земле!
            
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.drawEllipse(x - 12, y - 12, 24, 24)
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            painter.setFont(QFont("Arial", 9, QFont.Bold))
            painter.drawText(x - 20, y - 20, "РЕТРАНСЛЯТОР")
            
            # Добавляем подпись с высотой
            relay_alt = relay_abs_alt
            rel_alt = relay_alt - self.start_point[2]
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setFont(QFont("Arial", 7))
            painter.drawText(x - 20, y + 30, f"H={rel_alt:.0f}м")
            
            # Если ретранслятор в земле - показываем предупреждение
            if not is_above_ground:
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                painter.setFont(QFont("Arial", 8, QFont.Bold))
                painter.drawText(x - 30, y + 50, "⚠️ В ЗЕМЛЕ!")
            
            # Рисуем радиус действия ретранслятора
            if self.start_point:
                start_lat, start_lon, _ = self.start_point
                relay_radius = self.calculate_relay_radius(
                    self.relay_point[0], self.relay_point[1], 
                    relay_abs_alt, start_lat, start_lon
                )
                
                # Конвертируем радиус в пиксели
                radius_deg = relay_radius / 111000
                center_lat = self.relay_point[0]
                center_lon = self.relay_point[1]
                
                # Рисуем окружность радиуса действия ретранслятора
                painter.setPen(QPen(QColor(255, 165, 0, 80), 1, Qt.DashLine))
                points = []
                for angle in range(0, 360, 10):
                    lat = center_lat + radius_deg * cos(radians(angle))
                    lon = center_lon + radius_deg * sin(radians(angle)) / cos(radians(center_lat))
                    px, py = map_point(lat, lon)
                    points.append((px, py))
                
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        painter.drawLine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
            
            if self.start_point:
                x_start, y_start = map_point(self.start_point[0], self.start_point[1])
                painter.setPen(QPen(QColor(255, 165, 0, 150), 2, Qt.DashLine))
                painter.drawLine(x_start, y_start, x, y)
                
                for lat, lon, alt in self.waypoints:
                    x_wp, y_wp = map_point(lat, lon)
                    painter.setPen(QPen(QColor(255, 165, 0, 100), 1, Qt.DashLine))
                    painter.drawLine(x, y, x_wp, y_wp)
        
        # Рисуем точки маршрута
        for i, (lat, lon, alt_rel) in enumerate(self.waypoints):
            x, y = map_point(lat, lon)
            
            # Определяем цвет точки
            if i in self.shadow_points:
                color = QColor(255, 0, 0)  # Красный - тень до ретранслятора
            elif i in self.shadow_after_relay_points:
                color = QColor(255, 0, 255)  # Фиолетовый - тень после ретранслятора
            elif i in self.unsafe_points:
                color = QColor(255, 165, 0)  # Оранжевый - опасная высота
            else:
                color = QColor(0, 255, 0)  # Зеленый - безопасно
            
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawEllipse(x - 8, y - 8, 16, 16)
            
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.drawText(x - 5, y - 10, str(i + 1))
            
            painter.setPen(QPen(QColor(255, 255, 0), 1))
            painter.setFont(QFont("Arial", 8))
            painter.drawText(x - 20, y + 25, f"H={alt_rel:.0f}м")
    
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
        
        # Проверяем попадание в ретранслятор
        if self.relay_point is not None:
            min_lon, max_lon, min_lat, max_lat = self.bounds
            x = (self.relay_point[1] - min_lon) / (max_lon - min_lon) * img_w + x_offset
            y = (max_lat - self.relay_point[0]) / (max_lat - min_lat) * img_h + y_offset
            
            dist = sqrt((event.x() - x)**2 + (event.y() - y)**2)
            if dist < 20:
                self.dragging_relay = True
                self.setCursor(Qt.ClosedHandCursor)
                return
        
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
            # Если включен режим ручной установки ретранслятора
            if self.manual_relay_mode:
                self.set_relay_point(lat, lon)
                self.manual_relay_mode = False
                if self.parent_window:
                    self.parent_window.manual_relay_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; padding: 8px; }")
                    self.parent_window.manual_relay_btn.setText("Ручная установка ретранслятора")
                QMessageBox.information(self, "Ретранслятор установлен", 
                                       f"Ретранслятор установлен в точке:\nШирота: {lat:.5f}\nДолгота: {lon:.5f}")
            else:
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
        
        elif self.dragging_relay and self.relay_point is not None:
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
                self.relay_moved.emit(lat, lon)
                self.update()
    
    def mouseReleaseEvent(self, event):
        if self.dragging:
            self.dragging = False
            self.selected_point = -1
            self.setCursor(Qt.ArrowCursor)
            self.update()
        
        if self.dragging_relay:
            self.dragging_relay = False
            self.setCursor(Qt.ArrowCursor)
            self.update()
    
    def set_unsafe_points(self, indices):
        self.unsafe_points = set(indices)
        self.update()
    
    def set_shadow_points(self, indices):
        self.shadow_points = set(indices)
        self.update()
    
    def set_shadow_after_relay_points(self, indices):
        self.shadow_after_relay_points = set(indices)
        self.update()
    
    def set_relay_point(self, lat, lon):
        self.relay_point = (lat, lon) if lat is not None and lon is not None else None
        self.update()
        if self.parent_window and hasattr(self.parent_window, 'update_profile'):
            self.parent_window.update_profile()

class ProfileWidget(QWidget):
    """Виджет для отображения профиля высот"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(350)
        self.setStyleSheet("background-color: white;")
        
        self.distances = []
        self.ground_rel = []
        self.flight_rel = []
        self.waypoint_indices = []
        self.unsafe_indices = []
        self.shadow_indices = []
        self.shadow_after_relay = []
        self.min_safe_relative = []
        self.start_abs_elev = 0
        self.relay_point = None
        self.relay_position_km = None
        self.first_waypoint_dist_km = 0
        self.shadow_zones = []
        self.relay_altitude = 0
        self.relay_radius_km = 0
        
    def set_data(self, distances, ground_rel, flight_rel, waypoint_indices, 
                 unsafe_indices, shadow_indices, shadow_after_relay,
                 start_abs_elev, min_clearance, first_waypoint_dist_km=0,
                 relay_point=None, relay_position_km=None, shadow_zones=None,
                 relay_altitude=0, relay_radius_km=0):
        self.distances = distances
        self.ground_rel = ground_rel
        self.flight_rel = flight_rel
        self.waypoint_indices = waypoint_indices
        self.unsafe_indices = unsafe_indices
        self.shadow_indices = shadow_indices
        self.shadow_after_relay = shadow_after_relay
        self.start_abs_elev = start_abs_elev
        self.min_safe_relative = [g + min_clearance for g in ground_rel]
        self.relay_point = relay_point
        self.relay_position_km = relay_position_km
        self.first_waypoint_dist_km = first_waypoint_dist_km
        self.shadow_zones = shadow_zones if shadow_zones else []
        self.relay_altitude = relay_altitude
        self.relay_radius_km = relay_radius_km
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
        
        pen = QPen(QColor(100, 100, 200), 2)
        painter.setPen(pen)
        for i in range(len(ground_points) - 1):
            painter.drawLine(ground_points[i], ground_points[i+1])
        
        pen = QPen(QColor(255, 100, 100), 2, Qt.DashLine)
        painter.setPen(pen)
        for i in range(len(self.distances) - 1):
            x1, y1 = map_point(self.distances[i], self.min_safe_relative[i])
            x2, y2 = map_point(self.distances[i+1], self.min_safe_relative[i+1])
            painter.drawLine(x1, y1, x2, y2)
        
        flight_points = []
        for i, dist in enumerate(self.distances):
            x, y = map_point(dist, self.flight_rel[i])
            flight_points.append(QPoint(x, y))
        
        # Рисуем траекторию - красным участки тени ДО ретранслятора
        for i in range(len(flight_points) - 1):
            if i in self.shadow_indices or (i+1) in self.shadow_indices:
                pen = QPen(QColor(255, 0, 0), 4)
            else:
                pen = QPen(QColor(0, 200, 0), 3)
            painter.setPen(pen)
            painter.drawLine(flight_points[i], flight_points[i+1])
        
        # Рисуем участки тени ПОСЛЕ ретранслятора (фиолетовым пунктиром)
        for i in range(len(flight_points) - 1):
            if i in self.shadow_after_relay or (i+1) in self.shadow_after_relay:
                if not (i in self.shadow_indices or (i+1) in self.shadow_indices):
                    pen = QPen(QColor(255, 0, 255, 150), 3, Qt.DashDotLine)
                    painter.setPen(pen)
                    painter.drawLine(flight_points[i], flight_points[i+1])
        
        if len(flight_points) > 0:
            x_start, y_start = flight_points[0].x(), flight_points[0].y()
            painter.setBrush(QBrush(QColor(0, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            painter.drawEllipse(x_start - 6, y_start - 6, 12, 12)
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.drawText(x_start - 15, y_start - 15, "СТАРТ")
        
        if self.first_waypoint_dist_km > 0:
            x_boundary, _ = map_point(self.first_waypoint_dist_km, 0)
            painter.setPen(QPen(QColor(0, 255, 255, 150), 1, Qt.DashDotLine))
            painter.drawLine(x_boundary, int(margins.top()), x_boundary, int(margins.bottom()))
            painter.setPen(QPen(QColor(0, 255, 255), 1))
            painter.setFont(QFont("Arial", 7))
            painter.drawText(x_boundary - 25, int(margins.bottom()) + 5, "Взлетный")
            painter.drawText(x_boundary - 25, int(margins.bottom()) + 15, "участок")
        
        if self.relay_point is not None:
            relay_dist = self.relay_point[0]
            relay_height_abs = self.relay_point[1]
            relay_height_rel = relay_height_abs - self.start_abs_elev
            
            x_relay, y_relay = map_point(relay_dist, relay_height_rel)
            
            painter.setPen(QPen(QColor(255, 165, 0), 2, Qt.DashLine))
            painter.drawLine(x_relay, int(margins.top()), x_relay, int(margins.bottom()))
            
            painter.setPen(QPen(QColor(255, 165, 0, 180), 2, Qt.DashLine))
            painter.drawLine(x_start, y_start, x_relay, y_relay)
            
            painter.setBrush(QBrush(QColor(255, 165, 0)))
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            painter.drawEllipse(x_relay - 8, y_relay - 8, 16, 16)
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.drawText(x_relay + 10, y_relay - 5, "Ретранслятор")
            painter.drawText(x_relay + 10, y_relay + 10, f"({relay_dist:.1f} км)")
            
            painter.setPen(QPen(QColor(255, 165, 0), 1))
            painter.setFont(QFont("Arial", 7))
            painter.drawText(x_relay + 10, y_relay + 20, f"Высота: {self.relay_altitude:.0f}м")
            
            # Рисуем радиус действия ретранслятора на графике
            if self.relay_radius_km > 0:
                x_radius_end, _ = map_point(self.relay_radius_km, 0)
                painter.setPen(QPen(QColor(255, 165, 0, 80), 1, Qt.DashLine))
                painter.drawLine(x_relay, int(margins.top()), x_radius_end, int(margins.bottom()))
                painter.setPen(QPen(QColor(255, 165, 0), 1))
                painter.setFont(QFont("Arial", 7))
                painter.drawText(x_radius_end + 5, int(margins.top()) + 15, f"Радиус {self.relay_radius_km:.1f} км")
            
            if self.shadow_zones:
                first_zone = self.shadow_zones[0]
                if first_zone:
                    zone_start_dist = first_zone[0][3] / 1000
                    x_zone_start, _ = map_point(zone_start_dist, 0)
                    painter.setPen(QPen(QColor(255, 0, 0, 150), 1, Qt.DashLine))
                    painter.drawLine(x_zone_start, int(margins.top()), x_zone_start, int(margins.bottom()))
                    painter.setPen(QPen(QColor(255, 0, 0), 1))
                    painter.setFont(QFont("Arial", 7))
                    painter.drawText(x_zone_start - 20, int(margins.top()) + 15, "Начало")
                    painter.drawText(x_zone_start - 20, int(margins.top()) + 25, "зоны тени")
        
        for idx in self.waypoint_indices:
            if idx < len(flight_points) and idx > 0:
                x, y = flight_points[idx].x(), flight_points[idx].y()
                
                if idx in self.shadow_indices:
                    color = QColor(255, 0, 0)
                elif idx in self.shadow_after_relay:
                    color = QColor(255, 0, 255)
                elif idx in self.unsafe_indices:
                    color = QColor(255, 165, 0)
                else:
                    color = QColor(0, 255, 0)
                
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(0, 0, 0), 2))
                painter.drawEllipse(x - 5, y - 5, 10, 10)
                
                painter.setPen(QPen(QColor(0, 0, 0), 2))
                painter.setFont(QFont("Arial", 8, QFont.Bold))
                painter.drawText(x + 5, y - 5, str(self.waypoint_indices.index(idx)))
        
        for idx in self.shadow_indices:
            if idx < len(flight_points):
                x, y = flight_points[idx].x(), flight_points[idx].y()
                painter.setPen(QPen(QColor(255, 0, 0, 150), 6))
                painter.drawEllipse(x - 5, y - 5, 10, 10)
        
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
        painter.drawText(95, legend_y, "Зона радиотени (ДО ретранслятора)")
        legend_y += 15
        
        if self.shadow_after_relay:
            painter.setPen(QPen(QColor(255, 0, 255), 3, Qt.DashDotLine))
            painter.drawLine(70, legend_y - 5, 90, legend_y - 5)
            painter.drawText(95, legend_y, "Зона радиотени (ПОСЛЕ ретранслятора)")
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
        painter.drawText(95, legend_y, "Линии связи / Ретранслятор")
        legend_y += 15
        
        painter.setPen(QPen(QColor(255, 165, 0), 1, Qt.DashLine))
        painter.drawLine(70, legend_y - 5, 90, legend_y - 5)
        painter.drawText(95, legend_y, "Радиус действия ретранслятора")
        legend_y += 15
        
        painter.setPen(QPen(QColor(0, 255, 255), 1, Qt.DashDotLine))
        painter.drawLine(70, legend_y - 5, 90, legend_y - 5)
        painter.drawText(95, legend_y, "Граница взлетного участка")
        legend_y += 15
        
        painter.setPen(QPen(QColor(255, 0, 0), 1, Qt.DashLine))
        painter.drawLine(70, legend_y - 5, 90, legend_y - 5)
        painter.drawText(95, legend_y, "Начало зоны радиотени")
        
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawText(10, self.height() - 15, 
                        f"Старт: {self.start_abs_elev:.0f} м над уровнем моря")

class RoutePlanner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Планировщик маршрута БПЛА (индивидуальные высоты)")
        self.setGeometry(100, 100, 1400, 900)
        
        self.satellite_image = None
        self.dem_array = None
        self.bounds = None
        self.start_point = None
        self.waypoints = []
        self.start_abs_elev = 0
        self.shadow_indices = []
        self.shadow_after_relay = []
        self.shadow_zones = []
        self.relay_point = None
        self.default_altitude = 100
        self.relay_position_km = None
        self.first_waypoint_dist_km = 0
        self.relay_altitude = 0  # Высота ретранслятора над землей в метрах
        self.relay_radius_km = 0
        
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
        
        default_group = QGroupBox("Высота по умолчанию")
        default_layout = QVBoxLayout()
        default_layout.addWidget(QLabel("Высота полета для новых точек (отн. старта), м:"))
        self.default_altitude_spin = QDoubleSpinBox()
        self.default_altitude_spin.setRange(0, 2000)
        self.default_altitude_spin.setValue(100)
        self.default_altitude_spin.setSuffix(" м")
        self.default_altitude_spin.valueChanged.connect(self.on_default_altitude_changed)
        default_layout.addWidget(self.default_altitude_spin)
        default_group.setLayout(default_layout)
        left_layout.addWidget(default_group)
        
        flight_group = QGroupBox("Параметры безопасности")
        flight_layout = QVBoxLayout()
        
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
        self.points_list.itemDoubleClicked.connect(self.edit_point_altitude)
        points_layout.addWidget(self.points_list)
        
        points_btn_layout = QHBoxLayout()
        self.edit_alt_btn = QPushButton("Изменить высоту")
        self.edit_alt_btn.clicked.connect(self.edit_selected_point_altitude)
        self.delete_point_btn = QPushButton("Удалить точку")
        self.delete_point_btn.clicked.connect(self.delete_selected_point)
        self.clear_all_btn = QPushButton("Очистить все")
        self.clear_all_btn.clicked.connect(self.clear_all_points)
        points_btn_layout.addWidget(self.edit_alt_btn)
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
        
        # Группа для ретранслятора
        relay_group = QGroupBox("Ретранслятор")
        relay_layout = QVBoxLayout()
        
        # Кнопка автоматического поиска
        self.find_relay_btn = QPushButton("Автоматический поиск")
        self.find_relay_btn.clicked.connect(self.find_relay)
        self.find_relay_btn.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; padding: 8px; }")
        relay_layout.addWidget(self.find_relay_btn)
        
        # Кнопка ручной установки
        self.manual_relay_btn = QPushButton("Ручная установка ретранслятора")
        self.manual_relay_btn.clicked.connect(self.enable_manual_relay)
        self.manual_relay_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; padding: 8px; }")
        relay_layout.addWidget(self.manual_relay_btn)
        
        # Кнопка удаления ретранслятора
        self.remove_relay_btn = QPushButton("Удалить ретранслятор")
        self.remove_relay_btn.clicked.connect(self.remove_relay)
        self.remove_relay_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; padding: 8px; }")
        relay_layout.addWidget(self.remove_relay_btn)
        
        self.relay_info_label = QLabel("Ретранслятор не установлен")
        self.relay_info_label.setStyleSheet("color: gray; font-weight: bold;")
        self.relay_info_label.setWordWrap(True)
        relay_layout.addWidget(self.relay_info_label)
        
        relay_group.setLayout(relay_layout)
        left_layout.addWidget(relay_group)
        
        left_layout.addStretch()
        left_panel.setMaximumWidth(320)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.map_widget = MapWidget(self)
        self.map_widget.point_clicked.connect(self.on_map_click)
        self.map_widget.point_moved.connect(self.move_waypoint)
        self.map_widget.relay_moved.connect(self.move_relay)
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
    
    def on_default_altitude_changed(self):
        self.default_altitude = self.default_altitude_spin.value()
    
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
            self.add_waypoint_with_dialog(lat, lon)
    
    def enable_manual_relay(self):
        """Включение режима ручной установки ретранслятора"""
        if self.start_point is None:
            QMessageBox.warning(self, "Ошибка", "Сначала установите точку старта")
            return
        
        self.map_widget.manual_relay_mode = True
        self.manual_relay_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 8px; }")
        self.manual_relay_btn.setText("Кликните по карте для установки")
        self.statusBar().showMessage("Кликните по карте для установки ретранслятора")
    
    def remove_relay(self):
        """Удаление ретранслятора"""
        self.relay_point = None
        self.relay_altitude = 0
        self.map_widget.set_relay_point(None, None)
        self.relay_info_label.setText("Ретранслятор не установлен")
        self.relay_info_label.setStyleSheet("color: gray; font-weight: bold;")
        self.update_profile()
        self.statusBar().showMessage("Ретранслятор удален")
    
    def move_relay(self, lat, lon):
        """Перемещение ретранслятора"""
        if self.relay_point is not None:
            self.relay_point = (lat, lon)
            self.map_widget.set_relay_point(lat, lon)
            self.update_profile()
            self.statusBar().showMessage(f"Ретранслятор перемещен в: {lat:.5f}, {lon:.5f}")
    
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
                               f"Абсолютная высота: {abs_elev:.0f} м над уровнем моря\n\n"
                               f"Барометр обнулен. Теперь все высоты отсчитываются от этой точки.")
    
    def add_waypoint_with_dialog(self, lat, lon):
        if self.start_point is None:
            QMessageBox.warning(self, "Ошибка", "Сначала установите точку старта")
            return
        
        dialog = AltitudeInputDialog(len(self.waypoints), self.default_altitude, self)
        
        ground_abs = self.map_widget.get_elevation_at(lat, lon)
        dialog.set_ground_info(ground_abs, self.start_abs_elev)
        
        if dialog.exec_() == QDialog.Accepted:
            altitude = dialog.get_altitude()
            self.add_waypoint(lat, lon, altitude)
    
    def add_waypoint(self, lat, lon, relative_altitude=None):
        if relative_altitude is None:
            relative_altitude = self.default_altitude
            
        ground_rel = self.map_widget.get_relative_ground_elevation(lat, lon)
        
        self.waypoints.append([lat, lon, relative_altitude])
        self.map_widget.add_waypoint(lat, lon, relative_altitude)
        
        clearance = relative_altitude - ground_rel
        status = "🔴" if clearance < self.min_clearance_spin.value() else "✅"
        
        item = QListWidgetItem(f"{status} Точка {len(self.waypoints)}: H={relative_altitude:.0f}м")
        item.setToolTip(f"Широта: {lat:.5f}\nДолгота: {lon:.5f}\nВысота земли: {ground_rel:.0f} м отн.\nВысота полета: {relative_altitude:.0f} м отн.\nЗазор: {clearance:.0f} м")
        self.points_list.addItem(item)
        
        self.statusBar().showMessage(f"Добавлена точка {len(self.waypoints)}: высота={relative_altitude:.0f} м, зазор={clearance:.0f} м")
        self.update_profile()
    
    def edit_selected_point_altitude(self):
        selected = self.points_list.currentRow()
        if selected < 0 or selected >= len(self.waypoints):
            QMessageBox.warning(self, "Ошибка", "Выберите точку для редактирования")
            return
        
        lat, lon, current_alt = self.waypoints[selected]
        
        dialog = AltitudeInputDialog(selected, current_alt, self)
        ground_abs = self.map_widget.get_elevation_at(lat, lon)
        dialog.set_ground_info(ground_abs, self.start_abs_elev)
        
        if dialog.exec_() == QDialog.Accepted:
            new_altitude = dialog.get_altitude()
            self.waypoints[selected][2] = new_altitude
            self.map_widget.waypoints[selected][2] = new_altitude
            
            ground_rel = self.map_widget.get_relative_ground_elevation(lat, lon)
            clearance = new_altitude - ground_rel
            status = "🔴" if clearance < self.min_clearance_spin.value() else "✅"
            
            self.points_list.item(selected).setText(f"{status} Точка {selected+1}: H={new_altitude:.0f}м")
            self.points_list.item(selected).setToolTip(f"Широта: {lat:.5f}\nДолгота: {lon:.5f}\nВысота земли: {ground_rel:.0f} м отн.\nВысота полета: {new_altitude:.0f} м отн.\nЗазор: {clearance:.0f} м")
            
            self.update_profile()
    
    def edit_point_altitude(self, index):
        self.edit_selected_point_altitude()
    
    def move_waypoint(self, index, lat, lon):
        if index < len(self.waypoints):
            self.waypoints[index][0] = lat
            self.waypoints[index][1] = lon
            self.map_widget.waypoints[index][0] = lat
            self.map_widget.waypoints[index][1] = lon
            
            ground_rel = self.map_widget.get_relative_ground_elevation(lat, lon)
            flight_rel = self.waypoints[index][2]
            clearance = flight_rel - ground_rel
            status = "🔴" if clearance < self.min_clearance_spin.value() else "✅"
            
            self.points_list.item(index).setText(f"{status} Точка {index+1}: H={flight_rel:.0f}м")
            self.points_list.item(index).setToolTip(f"Широта: {lat:.5f}\nДолгота: {lon:.5f}\nВысота земли: {ground_rel:.0f} м отн.\nВысота полета: {flight_rel:.0f} м отн.\nЗазор: {clearance:.0f} м")
            
            self.update_profile()
    
    def delete_selected_point(self):
        selected = self.points_list.currentRow()
        if selected >= 0 and selected < len(self.waypoints):
            self.waypoints.pop(selected)
            self.map_widget.waypoints.pop(selected)
            self.points_list.takeItem(selected)
            
            for i in range(self.points_list.count()):
                lat, lon, alt = self.waypoints[i]
                ground_rel = self.map_widget.get_relative_ground_elevation(lat, lon)
                clearance = alt - ground_rel
                status = "🔴" if clearance < self.min_clearance_spin.value() else "✅"
                self.points_list.item(i).setText(f"{status} Точка {i+1}: H={alt:.0f}м")
            
            self.update_profile()
    
    def clear_all_points(self):
        self.waypoints.clear()
        self.map_widget.waypoints.clear()
        self.points_list.clear()
        self.relay_point = None
        self.relay_altitude = 0
        self.map_widget.relay_point = None
        self.shadow_indices = []
        self.shadow_after_relay = []
        self.shadow_zones = []
        self.relay_position_km = None
        self.relay_altitude = 0
        self.relay_radius_km = 0
        self.relay_info_label.setText("Ретранслятор не установлен")
        self.relay_info_label.setStyleSheet("color: gray; font-weight: bold;")
        self.update_profile()
    
    def on_point_selected(self):
        selected = self.points_list.currentRow()
        if selected >= 0:
            self.map_widget.selected_point = selected
            self.map_widget.update()
    
    def update_profile(self):
        if len(self.waypoints) < 2 or self.start_point is None:
            self.profile_widget.set_data([], [], [], [], [], [], [], 0, 0)
            return
        
        distances, ground_rel, flight_rel, waypoint_indices = self.map_widget.get_trajectory_profile()
        
        unsafe_indices = self.map_widget.get_safe_indices(flight_rel, ground_rel, self.min_clearance_spin.value())
        
        # Используем метод с учетом радиуса ретранслятора
        shadow_indices, shadow_after_relay = self.map_widget.check_visibility_with_relay(
            self.operator_antenna_spin.value(),
            self.relay_antenna_spin.value()
        )
        
        self.shadow_indices = shadow_indices
        self.shadow_after_relay = shadow_after_relay
        
        relay_info = None
        relay_position_km = None
        relay_altitude = 0
        relay_radius_km = 0
        
        if self.relay_point is not None:
            start_lat, start_lon = self.start_point
            relay_lat, relay_lon = self.relay_point
            
            # Получаем высоту земли под ретранслятором
            ground_alt = self.map_widget.get_elevation_at(relay_lat, relay_lon)
            
            # Вычисляем высоту ретранслятора над землей
            # По умолчанию используем 50м, если не сохранена другая высота
            if self.relay_altitude <= 0:
                self.relay_altitude = 50  # Минимальная высота над землей
            
            # Абсолютная высота ретранслятора
            relay_abs = ground_alt + self.relay_altitude
            
            relay_dist = calculate_distance(start_lat, start_lon, relay_lat, relay_lon) / 1000
            relay_info = (relay_dist, relay_abs)
            relay_position_km = relay_dist
            relay_altitude = self.relay_altitude  # Относительная высота над землей
            
            # Расчет радиуса действия ретранслятора
            relay_radius = self.map_widget.calculate_relay_radius(
                relay_lat, relay_lon, relay_abs, start_lat, start_lon
            )
            relay_radius_km = relay_radius / 1000
            self.relay_radius_km = relay_radius_km
            
            # Проверяем видимость ретранслятора от старта
            relay_visible = self.map_widget.check_line_of_sight(
                start_lat, start_lon, self.start_abs_elev,
                relay_lat, relay_lon, relay_abs,
                self.operator_antenna_spin.value()
            )
            
            # Проверяем, находится ли ретранслятор над землей
            is_above_ground = self.relay_altitude > 1
            
            if not is_above_ground:
                self.relay_info_label.setText(
                    f"⚠️ РЕТРАНСЛЯТОР В ЗЕМЛЕ!\n"
                    f"Высота над землей: {self.relay_altitude:.0f} м\n"
                    f"💡 Установите высоту больше 1м\n"
                    f"Расстояние: {relay_position_km:.1f} км"
                )
                self.relay_info_label.setStyleSheet("color: red; font-weight: bold;")
            elif not relay_visible:
                self.relay_info_label.setText(
                    f"⚠️ Ретранслятор НЕ ВИДЕН ОТ СТАРТА!\n"
                    f"Расстояние: {relay_position_km:.1f} км\n"
                    f"Высота над землей: {self.relay_altitude:.0f} м\n"
                    f"💡 Переместите ретранслятор на линию видимости"
                )
                self.relay_info_label.setStyleSheet("color: red; font-weight: bold;")
            elif shadow_after_relay:
                self.relay_info_label.setText(
                    f"✅ Ретранслятор установлен\n"
                    f"Расстояние: {relay_position_km:.1f} км\n"
                    f"Высота над землей: {self.relay_altitude:.0f} м\n"
                    f"Широта: {self.relay_point[0]:.5f}\n"
                    f"Долгота: {self.relay_point[1]:.5f}\n"
                    f"Радиус: {relay_radius_km:.1f} км\n\n"
                    f"⚠️ Осталось {len(shadow_after_relay)} участков тени\n"
                    f"💡 Переместите ретранслятор для улучшения"
                )
                self.relay_info_label.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self.relay_info_label.setText(
                    f"✅ Ретранслятор установлен\n"
                    f"Расстояние: {relay_position_km:.1f} км\n"
                    f"Высота над землей: {self.relay_altitude:.0f} м\n"
                    f"Широта: {self.relay_point[0]:.5f}\n"
                    f"Долгота: {self.relay_point[1]:.5f}\n"
                    f"Радиус: {relay_radius_km:.1f} км\n\n"
                    f"✅ Полное покрытие!\n"
                    f"💡 Перетащите для точной настройки"
                )
                self.relay_info_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.relay_info_label.setText("Ретранслятор не установлен")
            self.relay_info_label.setStyleSheet("color: gray; font-weight: bold;")
        
        first_waypoint_dist_km = 0
        if len(self.waypoints) >= 1:
            start_lat, start_lon = self.start_point
            first_lat, first_lon, _ = self.waypoints[0]
            first_waypoint_dist_km = calculate_distance(start_lat, start_lon, first_lat, first_lon) / 1000
        
        self.first_waypoint_dist_km = first_waypoint_dist_km
        
        shadow_zones = []
        start_lat, start_lon = self.start_point
        start_abs = self.start_abs_elev
        
        in_shadow = False
        current_zone = []
        
        for i in range(len(flight_rel)):
            if i < len(distances):
                # Используем корректный расчет координат
                total_points = len(flight_rel)
                total_segments = len(self.waypoints) - 1
                progress = i / total_points if total_points > 0 else 0
                segment_idx = min(int(progress * total_segments), total_segments - 1)
                
                if segment_idx < len(self.waypoints) - 1:
                    lat1, lon1, alt1 = self.waypoints[segment_idx]
                    lat2, lon2, alt2 = self.waypoints[segment_idx + 1]
                    
                    segment_start = segment_idx / total_segments if total_segments > 0 else 0
                    segment_end = (segment_idx + 1) / total_segments if total_segments > 0 else 0
                    segment_duration = segment_end - segment_start
                    
                    if segment_duration > 0:
                        t = (progress - segment_start) / segment_duration
                        t = max(0, min(1, t))
                    else:
                        t = 0
                    
                    lat = lat1 + (lat2 - lat1) * t
                    lon = lon1 + (lon2 - lon1) * t
                else:
                    lat, lon = self.waypoints[-1][0], self.waypoints[-1][1]
                
                alt = start_abs + flight_rel[i]
                
                has_los = self.map_widget.check_line_of_sight(start_lat, start_lon, start_abs, 
                                                              lat, lon, alt, self.operator_antenna_spin.value())
                
                if not has_los and not in_shadow:
                    in_shadow = True
                    current_zone = [(lat, lon, alt, distances[i] * 1000)]
                elif not has_los and in_shadow:
                    current_zone.append((lat, lon, alt, distances[i] * 1000))
                elif has_los and in_shadow:
                    in_shadow = False
                    if len(current_zone) > 5:
                        shadow_zones.append(current_zone)
                    current_zone = []
        
        if in_shadow and len(current_zone) > 5:
            shadow_zones.append(current_zone)
        
        self.shadow_zones = shadow_zones
        
        self.profile_widget.set_data(distances, ground_rel, flight_rel, 
                                     waypoint_indices, unsafe_indices, 
                                     shadow_indices, shadow_after_relay,
                                     self.start_abs_elev, self.min_clearance_spin.value(),
                                     first_waypoint_dist_km,
                                     relay_info, relay_position_km, shadow_zones,
                                     relay_altitude, relay_radius_km)
        
        # Обновляем карту
        shadow_waypoints = set()
        for idx in shadow_indices:
            for i, wp_idx in enumerate(waypoint_indices):
                if idx <= wp_idx:
                    shadow_waypoints.add(i)
                    break
        
        self.map_widget.set_shadow_points(shadow_waypoints)
        
        shadow_after_waypoints = set()
        for idx in shadow_after_relay:
            for i, wp_idx in enumerate(waypoint_indices):
                if idx <= wp_idx:
                    shadow_after_waypoints.add(i)
                    break
        
        self.map_widget.set_shadow_after_relay_points(shadow_after_waypoints)
        
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
        first_waypoint_dist = 0
        if len(self.waypoints) >= 1:
            start_lat, start_lon = self.start_point
            first_lat, first_lon, _ = self.waypoints[0]
            first_waypoint_dist = calculate_distance(start_lat, start_lon, first_lat, first_lon) / 1000
        
        for i in range(len(flight_rel)):
            if i < len(distances) and distances[i] <= first_waypoint_dist:
                continue
                
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
        
        # Используем метод с учетом радиуса ретранслятора
        shadow_indices, shadow_after_relay = self.map_widget.check_visibility_with_relay(
            self.operator_antenna_spin.value(),
            self.relay_antenna_spin.value()
        )
        
        self.shadow_indices = shadow_indices
        self.shadow_after_relay = shadow_after_relay
        
        self.update_profile()
        
        if alt_violations or shadow_indices or shadow_after_relay:
            self.result_text.setStyleSheet("color: red;")
            msg = "❌ МАРШРУТ НЕБЕЗОПАСЕН!\n\n"
            
            if alt_violations:
                msg += f"🔴 НАРУШЕНИЕ ВЫСОТЫ!\n"
                msg += f"Найдено {len(alt_violations)} участков, где БПЛА врежется в землю!\n"
                msg += f"Требуется зазор: {self.min_clearance_spin.value()} м\n"
                msg += f"(Взлетный участок до первой точки маршрута не учитывается)\n\n"
            
            if shadow_indices:
                msg += f"🔴 ЗОНА РАДИОТЕНИ ДО РЕТРАНСЛЯТОРА!\n"
                msg += f"На {len(shadow_indices)} участках траектории теряется прямая видимость от старта.\n"
                msg += f"Установите ретранслятор перед этой зоной.\n\n"
            
            if shadow_after_relay:
                msg += f"🟣 ЗОНА РАДИОТЕНИ ПОСЛЕ РЕТРАНСЛЯТОРА!\n"
                msg += f"На {len(shadow_after_relay)} участках все еще есть проблемы с видимостью.\n"
                msg += f"💡 Переместите ретранслятор для лучшего покрытия.\n\n"
            
            if self.relay_point:
                # Проверяем видимость ретранслятора
                start_lat, start_lon = self.start_point
                relay_lat, relay_lon = self.relay_point
                ground_alt = self.map_widget.get_elevation_at(relay_lat, relay_lon)
                relay_abs = ground_alt + self.relay_altitude
                relay_visible = self.map_widget.check_line_of_sight(
                    start_lat, start_lon, self.start_abs_elev,
                    relay_lat, relay_lon, relay_abs,
                    self.operator_antenna_spin.value()
                )
                if not relay_visible:
                    msg += f"⚠️ РЕТРАНСЛЯТОР НЕ ВИДЕН ОТ СТАРТА!\n"
                    msg += f"💡 Переместите ретранслятор на линию видимости.\n\n"
                elif not shadow_indices and not shadow_after_relay:
                    msg += "✅ Ретранслятор обеспечивает полное покрытие!\n\n"
            
            msg += "Проблемные участки:\n"
            if alt_violations:
                for v in alt_violations[:5]:
                    msg += f"💥 Столкновение: на {v['dist']:.1f} км, земля={v['ground']:.0f} м, "
                    msg += f"БПЛА={v['flight']:.0f} м (не хватает {v['deficit']:.0f} м высоты!)\n"
            
            self.result_text.setText(msg)
            self.statusBar().showMessage("Маршрут небезопасен!")
            QMessageBox.critical(self, "Результат проверки", msg)
        else:
            self.result_text.setStyleSheet("color: green;")
            min_clearance = 9999
            for i in range(len(flight_rel)):
                if i < len(distances) and distances[i] <= first_waypoint_dist:
                    continue
                clearance = flight_rel[i] - ground_rel[i]
                if clearance < min_clearance:
                    min_clearance = clearance
            
            if min_clearance == 9999:
                min_clearance = 0
                
            msg = f"✅ МАРШРУТ БЕЗОПАСЕН!\n\n"
            msg += f"Минимальный зазор: {min_clearance:.0f} м\n"
            msg += f"Требуемый зазор: {self.min_clearance_spin.value()} м\n"
            msg += f"Взлетный участок до первой точки ({first_waypoint_dist:.1f} км) исключен из проверки\n"
            if self.relay_point:
                start_lat, start_lon = self.start_point
                relay_lat, relay_lon = self.relay_point
                ground_alt = self.map_widget.get_elevation_at(relay_lat, relay_lon)
                relay_abs = ground_alt + self.relay_altitude
                relay_visible = self.map_widget.check_line_of_sight(
                    start_lat, start_lon, self.start_abs_elev,
                    relay_lat, relay_lon, relay_abs,
                    self.operator_antenna_spin.value()
                )
                if relay_visible:
                    msg += f"📡 Ретранслятор обеспечивает полное покрытие по радиусу!\n"
                    msg += f"  Расстояние от старта: {self.relay_position_km:.1f} км\n"
                    msg += f"  Высота над землей: {self.relay_altitude:.0f} м\n"
                    msg += f"  Радиус действия: {self.relay_radius_km:.1f} км\n"
                    msg += f"💡 Перетащите ретранслятор для точной настройки\n"
                else:
                    msg += f"⚠️ Ретранслятор не виден от старта!\n"
                    msg += f"  Расстояние от старта: {self.relay_position_km:.1f} км\n"
                    msg += f"  Высота над землей: {self.relay_altitude:.0f} м\n"
                    msg += f"💡 Переместите ретранслятор на линию видимости\n"
            else:
                msg += f"Радиосвязь: прямая видимость на всей траектории (ретранслятор не требуется)\n"
            msg += f"Высота старта: {self.start_abs_elev:.0f} м над уровнем моря"
            self.result_text.setText(msg)
            self.statusBar().showMessage("Маршрут безопасен")
            QMessageBox.information(self, "Результат проверки", msg)
    
    def find_relay(self):
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
            
            self.progress_dialog = QProgressDialog("Поиск наивысшей точки для ретранслятора...", "Отмена", 0, 100, self)
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
            self.find_relay_btn.setText("Автоматический поиск")
    
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
        self.find_relay_btn.setText("Автоматический поиск")
        
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
    
    def on_relay_search_finished(self, relay_pos):
        try:
            self.find_relay_btn.setEnabled(True)
            self.find_relay_btn.setText("Автоматический поиск")
            
            if self.progress_dialog:
                self.progress_dialog.close()
                self.progress_dialog = None
            
            if relay_pos:
                self.relay_point = relay_pos
                
                # Устанавливаем высоту ретранслятора над землей
                # Находим высоту ретранслятора из профиля траектории
                start_lat, start_lon = self.start_point
                ground_alt = self.map_widget.get_elevation_at(relay_pos[0], relay_pos[1])
                
                # Ищем высоту ретранслятора в точках маршрута
                relay_height = 50  # По умолчанию 50м над землей
                for i in range(len(self.waypoints) - 1):
                    lat1, lon1, alt1 = self.waypoints[i]
                    lat2, lon2, alt2 = self.waypoints[i + 1]
                    
                    # Проверяем, находится ли точка ретранслятора на этом сегменте
                    dist1 = calculate_distance(lat1, lon1, relay_pos[0], relay_pos[1])
                    dist2 = calculate_distance(lat2, lon2, relay_pos[0], relay_pos[1])
                    total_dist = calculate_distance(lat1, lon1, lat2, lon2)
                    
                    if dist1 + dist2 <= total_dist * 1.01:  # Точка на сегменте
                        # Интерполируем высоту
                        t = dist1 / total_dist if total_dist > 0 else 0
                        relay_rel_alt = smooth_interpolate(alt1, alt2, t)
                        relay_height = relay_rel_alt - (ground_alt - self.start_abs_elev)
                        break
                
                if relay_height < 10:
                    relay_height = 50  # Минимальная высота над землей
                
                self.relay_altitude = relay_height
                
                self.map_widget.set_relay_point(relay_pos[0], relay_pos[1])
                
                dist = calculate_distance(start_lat, start_lon, relay_pos[0], relay_pos[1])
                self.relay_position_km = dist / 1000
                relay_abs_alt = ground_alt + relay_height
                relay_rel_alt = relay_abs_alt - self.start_abs_elev
                
                # Расчет радиуса действия ретранслятора
                relay_radius = self.map_widget.calculate_relay_radius(
                    relay_pos[0], relay_pos[1], relay_abs_alt, start_lat, start_lon
                )
                self.relay_radius_km = relay_radius / 1000
                
                # Проверяем видимость ретранслятора от старта
                relay_visible = self.map_widget.check_line_of_sight(
                    start_lat, start_lon, self.start_abs_elev,
                    relay_pos[0], relay_pos[1], relay_abs_alt,
                    self.operator_antenna_spin.value()
                )
                
                if not relay_visible:
                    self.relay_info_label.setText(
                        f"⚠️ Ретранслятор НЕ ВИДЕН ОТ СТАРТА!\n"
                        f"Расстояние: {self.relay_position_km:.1f} км\n"
                        f"Высота над землей: {relay_height:.0f} м\n"
                        f"💡 Переместите ретранслятор на линию видимости"
                    )
                    self.relay_info_label.setStyleSheet("color: red; font-weight: bold;")
                else:
                    self.relay_info_label.setText(
                        f"✅ Ретранслятор найден!\n"
                        f"Расстояние от старта: {self.relay_position_km:.1f} км\n"
                        f"Высота над землей: {relay_height:.0f} м\n"
                        f"Широта: {relay_pos[0]:.5f}\n"
                        f"Долгота: {relay_pos[1]:.5f}\n"
                        f"Радиус: {self.relay_radius_km:.1f} км\n\n"
                        f"💡 Перетащите ретранслятор для точной настройки"
                    )
                    self.relay_info_label.setStyleSheet("color: green; font-weight: bold;")
                
                self.update_profile()
                
                # Проверяем эффективность ретранслятора
                _, shadow_after = self.map_widget.check_visibility_with_relay(
                    self.operator_antenna_spin.value(),
                    self.relay_antenna_spin.value()
                )
                
                if not relay_visible:
                    QMessageBox.warning(self, "Ретранслятор не виден",
                                      f"⚠️ Ретранслятор НЕ ВИДЕН ОТ СТАРТА!\n\n"
                                      f"Расстояние от старта: {self.relay_position_km:.1f} км\n"
                                      f"Высота над землей: {relay_height:.0f} м\n"
                                      f"Широта: {relay_pos[0]:.5f}\n"
                                      f"Долгота: {relay_pos[1]:.5f}\n\n"
                                      f"💡 Переместите ретранслятор на линию видимости")
                elif shadow_after:
                    QMessageBox.information(self, "Ретранслятор установлен",
                                          f"✅ Ретранслятор установлен!\n\n"
                                          f"Расстояние от старта: {self.relay_position_km:.1f} км\n"
                                          f"Высота над землей: {relay_height:.0f} м\n"
                                          f"Широта: {relay_pos[0]:.5f}\n"
                                          f"Долгота: {relay_pos[1]:.5f}\n"
                                          f"Радиус действия: {self.relay_radius_km:.1f} км\n\n"
                                          f"⚠️ Однако {len(shadow_after)} участков все еще в зоне радиотени.\n"
                                          f"💡 Попробуйте переместить ретранслятор перетаскиванием\n"
                                          f"для полного покрытия.")
                else:
                    QMessageBox.information(self, "Ретранслятор установлен",
                                          f"✅ Ретранслятор успешно установлен!\n\n"
                                          f"Расстояние от старта: {self.relay_position_km:.1f} км\n"
                                          f"Высота над землей: {relay_height:.0f} м\n"
                                          f"Широта: {relay_pos[0]:.5f}\n"
                                          f"Долгота: {relay_pos[1]:.5f}\n"
                                          f"Радиус действия: {self.relay_radius_km:.1f} км\n\n"
                                          f"📡 Полное покрытие радиосигналом!\n"
                                          f"💡 Перетащите ретранслятор для точной настройки")
            else:
                self.relay_info_label.setText("❌ Не удалось найти подходящее место\nПопробуйте увеличить высоту антенн или установить вручную")
                self.relay_info_label.setStyleSheet("color: red; font-weight: bold;")
                QMessageBox.warning(self, "Ретранслятор не найден",
                                   "Не удалось найти подходящее место для ретранслятора.\n\n"
                                   "Возможные причины:\n"
                                   "1. Зона радиотени слишком большая\n"
                                   "2. Высота антенн недостаточна\n"
                                   "3. Рельеф слишком сложный\n\n"
                                   "Попробуйте:\n"
                                   "- Увеличить высоту антенны оператора (сейчас {:.1f} м)\n"
                                   "- Увеличить высоту антенны ретранслятора (сейчас {:.1f} м)\n"
                                   "- Установить ретранслятор вручную\n"
                                   "- Изменить маршрут".format(
                                       self.operator_antenna_spin.value(),
                                       self.relay_antenna_spin.value()
                                   ))
        except Exception as e:
            error_msg = f"Ошибка при обработке результата:\n{str(e)}\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Ошибка", error_msg)
    
    def on_relay_search_error(self, error_msg):
        self.find_relay_btn.setEnabled(True)
        self.find_relay_btn.setText("Автоматический поиск")
        
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
                    'relay_altitude': self.relay_altitude,
                    'default_altitude': self.default_altitude,
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