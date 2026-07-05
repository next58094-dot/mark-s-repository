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
    min_lon, max_lon, min_lat, max_lat = bounds
    x = (lon - min_lon) / (max_lon - min_lon) * img_size[0]
    y = (max_lat - lat) / (max_lat - min_lat) * img_size[1]
    return int(x), int(y)

def pixel_to_lat_lon(x, y, bounds, img_size):
    min_lon, max_lon, min_lat, max_lat = bounds
    lon = min_lon + (x / img_size[0]) * (max_lon - min_lon)
    lat = max_lat - (y / img_size[1]) * (max_lat - min_lat)
    return lat, lon

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    c = 2*atan2(sqrt(a), sqrt(1-a))
    return R * c

def interpolate_points(lat1, lon1, lat2, lon2, num_points):
    result = []
    for i in range(num_points + 1):
        t = i / num_points
        lat = lat1 + (lat2 - lat1) * t
        lon = lon1 + (lon2 - lon1) * t
        result.append((lat, lon))
    return result

def smoothstep(t):
    return t * t * (3 - 2 * t)

def smooth_interpolate(v1, v2, t):
    t_smooth = smoothstep(t)
    return v1 + (v2 - v1) * t_smooth

class AltitudeInputDialog(QDialog):
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
        rel_ground = ground_abs - start_abs
        self.ground_info.setText(f"Высота земли в этой точке: {rel_ground:.0f} м относительно старта ({ground_abs:.0f} м над уровнем моря)")

class RelaySearchThread(QThread):
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
        self.dem_array = map_widget.dem_array
        self.bounds = map_widget.bounds
        self.elevation_cache = {}
        
    def stop(self):
        self._is_running = False
    
    def get_elevation_at(self, lat, lon):
        cache_key = (round(lat, 6), round(lon, 6))
        if cache_key in self.elevation_cache:
            return self.elevation_cache[cache_key]
        if self.dem_array is None or self.bounds is None:
            return 0
        min_lon, max_lon, min_lat, max_lat = self.bounds
        h, w = self.dem_array.shape
        if lat < min_lat or lat > max_lat or lon < min_lon or lon > max_lon:
            self.elevation_cache[cache_key] = 0
            return 0
        x = (lon - min_lon) / (max_lon - min_lon) * (w - 1)
        y = (max_lat - lat) / (max_lat - min_lat) * (h - 1)
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
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
    
    def check_line_of_sight_fast(self, lat1, lon1, alt1, lat2, lon2, alt2, antenna_height=0):
        if self.dem_array is None or self.bounds is None:
            return True
        h1 = alt1 + antenna_height
        h2 = alt2 + antenna_height
        distance = calculate_distance(lat1, lon1, lat2, lon2)
        if distance < 1: return True
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
            
            route_points = []
            for i in range(len(self.waypoints) - 1):
                lat1, lon1, alt_rel1 = self.waypoints[i]
                lat2, lon2, alt_rel2 = self.waypoints[i+1]
                steps = max(20, int(calculate_distance(lat1, lon1, lat2, lon2) / 10))
                for j in range(steps + 1):
                    t = j / steps
                    lat = lat1 + (lat2 - lat1) * t
                    lon = lon1 + (lon2 - lon1) * t
                    alt_rel = smooth_interpolate(alt_rel1, alt_rel2, t)
                    alt_abs = start_abs + alt_rel
                    dist_from_start = calculate_distance(start_lat, start_lon, lat, lon)
                    route_points.append((lat, lon, alt_abs, dist_from_start))
            
            if not route_points:
                self.finished.emit(None)
                return
            
            self.status.emit("Поиск зон радиотени...")
            self.progress.emit(20)
            
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
            
            first_shadow_zone = shadow_zones[0]
            zone_start = first_shadow_zone[0]
            zone_end = first_shadow_zone[-1]
            start_lat, start_lon, _, start_dist = zone_start
            end_lat, end_lon, _, end_dist = zone_end
            self.status.emit(f"Первая зона: {start_dist/1000:.1f} - {end_dist/1000:.1f} км")
            self.progress.emit(60)
            
            best_relay = None
            max_ground_alt = -9999
            best_dist = 0
            
            for idx in range(0, len(route_points), 2):
                if not self._is_running:
                    self.finished.emit(None)
                    return
                lat, lon, _, dist_from_start = route_points[idx]
                if dist_from_start < start_dist or dist_from_start > end_dist:
                    continue
                ground_alt = self.get_elevation_at(lat, lon)
                if ground_alt > max_ground_alt:
                    max_ground_alt = ground_alt
                    best_relay = (lat, lon)
                    best_dist = dist_from_start
                if idx % 10 == 0:
                    self.progress.emit(60 + int(idx / len(route_points) * 20))
            
            if best_relay is None:
                self.status.emit("Внутри зоны тени нет возвышенностей, ищем перед ней...")
                for idx in range(0, len(route_points), 2):
                    if not self._is_running:
                        self.finished.emit(None)
                        return
                    lat, lon, _, dist_from_start = route_points[idx]
                    if dist_from_start < start_dist and dist_from_start > start_dist - 1500:
                        ground_alt = self.get_elevation_at(lat, lon)
                        if ground_alt > max_ground_alt:
                            max_ground_alt = ground_alt
                            best_relay = (lat, lon)
                            best_dist = dist_from_start
            
            if best_relay is None:
                best_relay = (start_lat, start_lon)
                best_dist = start_dist
                max_ground_alt = self.get_elevation_at(start_lat, start_lon)
            
            self.status.emit(f"Поиск завершен. Ретранслятор найден на {best_dist/1000:.1f} км")
            self.progress.emit(100)
            self.finished.emit(best_relay)
                
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)
            self.finished.emit(None)

class MapWidget(QWidget):
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
        self.current_distances = []
        self.manual_relay_mode = False
        self.trajectory_distances = []
        
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
        if self.dem_array is None or self.bounds is None:
            return 0
        min_lon, max_lon, min_lat, max_lat = self.bounds
        if lat < min_lat or lat > max_lat or lon < min_lon or lon > max_lon:
            return 0
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
    
    def calculate_relay_radius(self, relay_lat, relay_lon, relay_abs_alt, start_lat, start_lon):
        ground_alt = self.get_elevation_at(relay_lat, relay_lon)
        height_above_ground = relay_abs_alt - ground_alt
        if height_above_ground < 0:
            height_above_ground = 0
        antenna_height = self.parent_window.relay_antenna_spin.value() if self.parent_window else 2
        total_height = height_above_ground + antenna_height
        optical_radius = sqrt(2 * R_EARTH * total_height)
        relay_radius = max(500, min(optical_radius, 50000))
        return relay_radius
    
    def check_visibility_with_relay(self, operator_antenna_height=2, relay_antenna_height=2):
        if self.start_point is None or len(self.waypoints) < 2:
            return [], []
        shadow_indices = []
        shadow_after_relay = []
        start_lat, start_lon, start_abs = self.start_point
        distances, ground_rel, flight_rel, waypoint_indices = self.get_trajectory_profile()
        
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
        
        relay_lat, relay_lon = self.relay_point
        relay_ground_alt = self.get_elevation_at(relay_lat, relay_lon)
        relay_antenna_abs = relay_ground_alt + relay_antenna_height
        dist_to_relay = calculate_distance(start_lat, start_lon, relay_lat, relay_lon)
        relay_radius = self.calculate_relay_radius(relay_lat, relay_lon, relay_antenna_abs, start_lat, start_lon)
        
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
            has_los_from_start = self.check_line_of_sight(start_lat, start_lon, start_abs, 
                                                         lat, lon, alt, operator_antenna_height)
            has_los_from_relay = self.check_line_of_sight(relay_lat, relay_lon, relay_antenna_abs, 
                                                         lat, lon, alt, 0)
            if dist_from_relay <= relay_radius:
                has_visibility = has_los_from_start or has_los_from_relay
            else:
                has_visibility = has_los_from_start
            if not has_visibility:
                if dist_from_start <= dist_to_relay:
                    shadow_indices.append(i)
                else:
                    shadow_after_relay.append(i)
        return shadow_indices, shadow_after_relay
    
    def get_trajectory_profile(self):
        if len(self.waypoints) < 2 or self.start_point is None:
            return [], [], [], []
        
        distances = [0.0]
        ground_rel = [0.0]
        flight_rel = [0.0]
        waypoint_indices = [0]
        start_abs = self.start_point[2]
        start_lat, start_lon, _ = self.start_point
        
        self.trajectory_distances = [0.0]
        
        if len(self.waypoints) >= 1:
            first_lat, first_lon, _ = self.waypoints[0]
            self.first_waypoint_distance = calculate_distance(start_lat, start_lon, first_lat, first_lon)
        
        accumulated_dist = 0.0
        
        for i in range(len(self.waypoints) - 1):
            lat1, lon1, alt_rel1 = self.waypoints[i]
            lat2, lon2, alt_rel2 = self.waypoints[i + 1]
            
            total_segment_dist = calculate_distance(lat1, lon1, lat2, lon2)
            # Уменьшил шаг для большей точности, чтобы избежать накопления ошибки
            steps = max(20, int(total_segment_dist / 20)) 
            
            prev_lat, prev_lon = lat1, lon1
            
            for j in range(1, steps + 1):
                t = j / steps
                lat = lat1 + (lat2 - lat1) * t
                lon = lon1 + (lon2 - lon1) * t
                
                ground_abs = self.get_elevation_at(lat, lon)
                ground_rel_val = ground_abs - start_abs
                ground_rel.append(ground_rel_val)
                
                flight_rel_val = smooth_interpolate(alt_rel1, alt_rel2, t)
                flight_rel.append(flight_rel_val)
                
                segment_dist = calculate_distance(prev_lat, prev_lon, lat, lon)
                accumulated_dist += segment_dist
                distances.append(accumulated_dist / 1000)
                self.trajectory_distances.append(accumulated_dist / 1000)
                
                prev_lat, prev_lon = lat, lon
            
            waypoint_indices.append(len(ground_rel) - 1)
        
        if len(self.waypoints) > 0 and len(ground_rel) > 0:
            last_wp_idx = len(ground_rel) - 1
            if last_wp_idx not in waypoint_indices:
                waypoint_indices.append(last_wp_idx)

        # ГЛАВНОЕ ИСПРАВЛЕНИЕ (убираем смещение вправо):
        # Мы принудительно обрезаем массив distances ровно по последней точке.
        # Это не дает графику "растягиваться" правее последней точки.
        if len(waypoint_indices) > 0:
            last_idx = waypoint_indices[-1]
            if last_idx < len(distances):
                # Обрезаем всё, что идет после последней точки маршрута
                distances = distances[:last_idx + 1]
                ground_rel = ground_rel[:last_idx + 1]
                flight_rel = flight_rel[:last_idx + 1]
                # Обновляем и внутренний массив, если он используется
                self.trajectory_distances = distances[:] 
        
        return distances, ground_rel, flight_rel, waypoint_indices
    
    def get_safe_indices(self, flight_rel, ground_rel, min_clearance):
        unsafe_indices = []
        first_waypoint_dist_km = self.first_waypoint_distance / 1000 if self.first_waypoint_distance else 0
        for i in range(len(flight_rel)):
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
        
        if len(self.waypoints) >= 2:
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)
            for i in range(len(self.waypoints) - 1):
                x1, y1 = map_point(self.waypoints[i][0], self.waypoints[i][1])
                x2, y2 = map_point(self.waypoints[i+1][0], self.waypoints[i+1][1])
                painter.drawLine(x1, y1, x2, y2)
        
        if self.relay_point:
            x, y = map_point(self.relay_point[0], self.relay_point[1])
            ground_alt = self.get_elevation_at(self.relay_point[0], self.relay_point[1])
            antenna_height = self.parent_window.relay_antenna_spin.value() if self.parent_window else 2
            painter.setBrush(QBrush(QColor(255, 165, 0)))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawRect(x - 10, y - 10, 20, 20)
            antenna_height_px = int(max(antenna_height * 2, 10))
            painter.setPen(QPen(QColor(255, 165, 0), 2))
            painter.drawLine(x, y - 10, x, y - 10 - antenna_height_px)
            painter.setBrush(QBrush(QColor(255, 200, 0)))
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawEllipse(x - 4, y - 14 - antenna_height_px, 8, 8)
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            painter.setFont(QFont("Arial", 9, QFont.Bold))
            painter.drawText(x - 20, y - 20 - antenna_height_px, "РЕТРАНСЛЯТОР")
            if self.start_point:
                start_lat, start_lon, _ = self.start_point
                relay_abs_alt = ground_alt + antenna_height
                relay_radius = self.calculate_relay_radius(
                    self.relay_point[0], self.relay_point[1], 
                    relay_abs_alt, start_lat, start_lon
                )
                radius_deg = relay_radius / 111000
                center_lat = self.relay_point[0]
                center_lon = self.relay_point[1]
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
        
        for i, (lat, lon, alt_rel) in enumerate(self.waypoints):
            x, y = map_point(lat, lon)
            painter.setBrush(QBrush(QColor(0, 255, 0)))
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
        
        if self.relay_point is not None:
            min_lon, max_lon, min_lat, max_lat = self.bounds
            x = (self.relay_point[1] - min_lon) / (max_lon - min_lon) * img_w + x_offset
            y = (max_lat - self.relay_point[0]) / (max_lat - min_lat) * img_h + y_offset
            dist = sqrt((event.x() - x)**2 + (event.y() - y)**2)
            if dist < 20:
                self.dragging_relay = True
                self.setCursor(Qt.ClosedHandCursor)
                return
        
        lat, lon = map_point_reverse(event.x(), event.y())
        if lat is not None:
            if self.manual_relay_mode:
                self.set_relay_point(lat, lon)
                self.manual_relay_mode = False
                if self.parent_window:
                    self.parent_window.manual_relay_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; padding: 8px; }")
                    self.parent_window.manual_relay_btn.setText("Ручная установка ретранслятора")
                    ground_alt = self.get_elevation_at(lat, lon)
                    QMessageBox.information(self, "Ретранслятор установлен", 
                                           f"Ретранслятор установлен НА ЗЕМЛЕ в точке:\n"
                                           f"Широта: {lat:.5f}\n"
                                           f"Долгота: {lon:.5f}\n"
                                           f"Высота земли: {ground_alt:.0f} м над уровнем моря\n"
                                           f"Высота антенны: {self.parent_window.relay_antenna_spin.value():.1f} м")
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
                self.relay_point = (lat, lon)
                self.update()
                if self.parent_window and hasattr(self.parent_window, 'update_profile'):
                    self.parent_window.update_profile()
    
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
            if self.parent_window and hasattr(self.parent_window, 'update_profile'):
                self.parent_window.update_profile()
    
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
    """Виджет профиля с ПРЯМОЙ ГЕОГРАФИЧЕСКОЙ ПРИВЯЗКОЙ (без смещения)"""
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
        self.first_waypoint_dist_km = 0
        self.shadow_zones = []
        self.relay_altitude = 0
        self.relay_radius_km = 0
        
        self.relay_lat_lon = None
        self.map_widget_ref = None
        
    def set_data(self, distances, ground_rel, flight_rel, waypoint_indices, 
                 unsafe_indices, shadow_indices, shadow_after_relay,
                 start_abs_elev, min_clearance, first_waypoint_dist_km=0,
                 relay_lat_lon=None, relay_position_km=None, shadow_zones=None,
                 relay_altitude=0, relay_radius_km=0, map_widget_ref=None):
        self.distances = distances
        self.ground_rel = ground_rel
        self.flight_rel = flight_rel
        self.waypoint_indices = waypoint_indices
        self.unsafe_indices = unsafe_indices
        self.shadow_indices = shadow_indices
        self.shadow_after_relay = shadow_after_relay
        self.start_abs_elev = start_abs_elev
        self.map_widget_ref = map_widget_ref
        
        if ground_rel:
            self.min_safe_relative = [g + min_clearance for g in ground_rel]
        else:
            self.min_safe_relative = []
            
        self.relay_lat_lon = relay_lat_lon
        self.first_waypoint_dist_km = first_waypoint_dist_km
        self.shadow_zones = shadow_zones if shadow_zones else []
        self.relay_altitude = relay_altitude
        self.relay_radius_km = relay_radius_km
        self.update()
    
    def get_elevation_at(self, lat, lon):
        if self.map_widget_ref and hasattr(self.map_widget_ref, 'get_elevation_at'):
            return self.map_widget_ref.get_elevation_at(lat, lon)
        return 0

    def find_exact_x_position_on_route(self, target_lat, target_lon):
        """
        ВАЖНО: Ищет точное положение ретранслятора ВДОЛЬ МАРШРУТА.
        Это полностью устраняет смещение вправо/влево!
        """
        if self.map_widget_ref is None or self.map_widget_ref.start_point is None:
            return 0.0
            
        waypoints = self.map_widget_ref.waypoints
        start_lat, start_lon, _ = self.map_widget_ref.start_point

        # 1. Проверяем, не стоит ли ретранслятор на старте или на точках
        # Сначала ищем среди индексов waypoint_indices, чтобы взять точные значения с графика
        for i, wp_idx in enumerate(self.waypoint_indices):
            if i < len(waypoints):
                wp_lat, wp_lon, _ = waypoints[i]
                # Если ретранслятор совпадает с точкой (в пределах 5 метров) - берем её координату
                if calculate_distance(target_lat, target_lon, wp_lat, wp_lon) < 5:
                    if wp_idx < len(self.distances):
                        return self.distances[wp_idx]

        # 2. Если ретранслятор стоит МЕЖДУ точками (как на вашем скриншоте)
        # Мы проходим по точкам маршрута и находим, между какими двумя он находится.
        for i in range(len(waypoints) - 1):
            lat1, lon1, _ = waypoints[i]
            lat2, lon2, _ = waypoints[i+1]
            
            # Расстояние от ретранслятора до начала и конца сегмента
            d1 = calculate_distance(target_lat, target_lon, lat1, lon1)
            d2 = calculate_distance(target_lat, target_lon, lat2, lon2)
            
            # Если сумма расстояний примерно равна длине сегмента - значит он между ними
            segment_dist = calculate_distance(lat1, lon1, lat2, lon2)
            
            if d1 + d2 <= segment_dist * 1.1:  # Погрешность 10%
                # Находим пропорцию (t) между 0 и 1
                if segment_dist > 0:
                    t = d1 / segment_dist
                else:
                    t = 0
                
                # Получаем километраж начала и конца сегмента на графике
                start_idx = self.waypoint_indices[i]
                end_idx = self.waypoint_indices[i+1]
                
                if start_idx < len(self.distances) and end_idx < len(self.distances):
                    dist_start = self.distances[start_idx]
                    dist_end = self.distances[end_idx]
                    
                    # Интерполируем точное положение по оси X графика
                    exact_dist = dist_start + (dist_end - dist_start) * t
                    return exact_dist

        # 3. Если ничего не нашли (фоллбэк), берем последнюю точку
        if len(self.waypoint_indices) > 0:
            last_idx = self.waypoint_indices[-1]
            if last_idx < len(self.distances):
                return self.distances[last_idx]
                
        return 0.0
    
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
        if self.relay_lat_lon is not None:
            relay_ground_abs = self.get_elevation_at(self.relay_lat_lon[0], self.relay_lat_lon[1])
            all_heights.append(relay_ground_abs - self.start_abs_elev)
        
        max_height = max(all_heights) if all_heights else 100
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
        
        for i in range(len(flight_points) - 1):
            if i in self.shadow_indices or (i+1) in self.shadow_indices:
                pen = QPen(QColor(255, 0, 0), 4)
            else:
                pen = QPen(QColor(0, 200, 0), 3)
            painter.setPen(pen)
            painter.drawLine(flight_points[i], flight_points[i+1])
        
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

        # ===== ОТОБРАЖЕНИЕ РЕТРАНСЛЯТОРА (ГЕОГРАФИЧЕСКАЯ ПРИВЯЗКА) =====
        if self.relay_lat_lon is not None:
            relay_lat, relay_lon = self.relay_lat_lon
            
            # ВЫСОТА: строго по рельефу (убирает парение)
            ground_abs = self.get_elevation_at(relay_lat, relay_lon)
            relay_ground_rel = ground_abs - self.start_abs_elev
            
            # КООРДИНАТА X: Поиск точного места между точками (убирает смещение)
            exact_relay_dist = self.find_exact_x_position_on_route(relay_lat, relay_lon)
            
            x_relay, y_ground = map_point(exact_relay_dist, relay_ground_rel)
            
            painter.setPen(QPen(QColor(255, 165, 0), 2, Qt.DashLine))
            painter.drawLine(x_relay, int(margins.top()), x_relay, y_ground)
            
            square_size = 16
            painter.setBrush(QBrush(QColor(255, 165, 0)))
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            painter.drawRect(
                x_relay - square_size//2,
                y_ground - square_size,
                square_size,
                square_size
            )
            
            antenna_height_px = int(max(self.relay_altitude * 0.75, 10))
            painter.setPen(QPen(QColor(255, 165, 0), 3))
            painter.drawLine(
                x_relay, y_ground - square_size,
                x_relay, y_ground - square_size - antenna_height_px
            )
            
            painter.setBrush(QBrush(QColor(255, 220, 0)))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawEllipse(
                x_relay - 5,
                y_ground - square_size - antenna_height_px - 5,
                10, 10
            )
            
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.drawText(x_relay + 15, y_ground - 25, "РЕТРАНСЛЯТОР")
            painter.drawText(x_relay + 15, y_ground - 10, f"({exact_relay_dist:.1f} км)")
            
            painter.setPen(QPen(QColor(255, 165, 0), 1))
            painter.setFont(QFont("Arial", 7))
            painter.drawText(x_relay + 15, y_ground + 15, f"Антенна: {self.relay_altitude:.0f}м")
            
            painter.setPen(QPen(QColor(90, 90, 90), 1))
            painter.setFont(QFont("Arial", 6))
            painter.drawText(x_relay + 15, y_ground + 30, f"земля: {relay_ground_rel:.0f}м")
            
            if self.relay_radius_km > 0:
                x_radius_end, _ = map_point(self.relay_radius_km, 0)
                painter.setPen(QPen(QColor(255, 165, 0, 80), 1, Qt.DashLine))
                painter.drawLine(x_relay, int(margins.top()), x_radius_end, int(margins.bottom()))
                painter.setPen(QPen(QColor(255, 165, 0), 1))
                painter.setFont(QFont("Arial", 7))
                painter.drawText(x_radius_end + 5, int(margins.top()) + 15, f"Радиус {self.relay_radius_km:.1f} км")
        
        # ===== РИСУЕМ ТОЧКИ МАРШРУТА =====
        for idx in self.waypoint_indices:
            if idx < len(flight_points):
                x, y = flight_points[idx].x(), flight_points[idx].y()
                painter.setBrush(QBrush(QColor(0, 255, 0)))
                painter.setPen(QPen(QColor(0, 0, 0), 2))
                painter.drawEllipse(x - 5, y - 5, 10, 10)
                painter.setPen(QPen(QColor(0, 0, 0), 2))
                painter.setFont(QFont("Arial", 8, QFont.Bold))
                wp_idx = self.waypoint_indices.index(idx) if idx in self.waypoint_indices else -1
                if wp_idx >= 0:
                    painter.drawText(x + 5, y - 5, str(wp_idx + 1))
        
        # ===== ЛЕГЕНДА =====
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.setFont(QFont("Arial", 8))
        legend_y = 20
        painter.drawText(10, legend_y, "Легенда:")
        painter.fillRect(70, legend_y - 8, 20, 10, QBrush(QColor(150, 150, 200)))
        painter.drawText(95, legend_y, "Рельеф (относительно старта)")
        legend_y += 15
        painter.setPen(QPen(QColor(0, 200, 0), 3))
        painter.drawLine(70, legend_y - 5, 90, legend_y - 5)
        painter.drawText(95, legend_y, "Траектория БПЛА")
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawText(10, self.height() - 15, f"Старт: {self.start_abs_elev:.0f} м над уровнем моря")
class RoutePlanner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Планировщик маршрута БПЛА")
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
        self.relay_altitude = 0
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
        
        start_group = QGroupBox("Точка старта")
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
        self.relay_antenna_spin.setRange(0.5, 20)
        self.relay_antenna_spin.setValue(5)
        self.relay_antenna_spin.setSuffix(" м")
        self.relay_antenna_spin.setSingleStep(0.5)
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
        
        relay_group = QGroupBox("Ретранслятор")
        relay_layout = QVBoxLayout()
        self.find_relay_btn = QPushButton("Автоматический поиск")
        self.find_relay_btn.clicked.connect(self.find_relay)
        self.find_relay_btn.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; padding: 8px; }")
        relay_layout.addWidget(self.find_relay_btn)
        self.manual_relay_btn = QPushButton("Ручная установка ретранслятора")
        self.manual_relay_btn.clicked.connect(self.enable_manual_relay)
        self.manual_relay_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; padding: 8px; }")
        relay_layout.addWidget(self.manual_relay_btn)
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
        if self.start_point is None:
            QMessageBox.warning(self, "Ошибка", "Сначала установите точку старта")
            return
        self.map_widget.manual_relay_mode = True
        self.manual_relay_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 8px; }")
        self.manual_relay_btn.setText("Кликните по карте для установки")
        self.statusBar().showMessage("Кликните по карте для установки ретранслятора (на земле)")
    
    def remove_relay(self):
        self.relay_point = None
        self.relay_altitude = 0
        self.map_widget.set_relay_point(None, None)
        self.relay_info_label.setText("Ретранслятор не установлен")
        self.relay_info_label.setStyleSheet("color: gray; font-weight: bold;")
        self.update_profile()
        self.statusBar().showMessage("Ретранслятор удален")
    
    def move_relay(self, lat, lon):
        if self.relay_point is not None:
            self.relay_point = (lat, lon)
            self.map_widget.set_relay_point(lat, lon)
            self.update_profile()
            ground_alt = self.map_widget.get_elevation_at(lat, lon)
            self.statusBar().showMessage(f"Ретранслятор перемещен в: {lat:.5f}, {lon:.5f} (земля: {ground_alt:.0f}м)")
    
    def set_start_point(self, lat, lon):
        abs_elev = self.map_widget.get_elevation_at(lat, lon)
        self.start_point = (lat, lon)
        self.start_abs_elev = abs_elev
        self.map_widget.set_start_point(lat, lon)
        self.start_info_label.setText(f"Старт: {lat:.5f}, {lon:.5f} | Высота: {abs_elev:.0f} м над уровнем моря")
        self.start_info_label.setStyleSheet("color: green; font-weight: bold;")
        self.statusBar().showMessage(f"Точка старта установлена. Абсолютная высота: {abs_elev:.0f} м")
        QMessageBox.information(self, "Старт установлен", 
                               f"Точка старта:\nШирота: {lat:.5f}\nДолгота: {lon:.5f}\nАбсолютная высота: {abs_elev:.0f} м над уровнем моря\n\nБарометр обнулен.")
    
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
    
    # ===== ИСПРАВЛЕННЫЙ update_profile (жесткая привязка к рельефу) =====
    def update_profile(self):
        if len(self.waypoints) < 2 or self.start_point is None:
            self.profile_widget.set_data([], [], [], [], [], [], [], 0, 0)
            return
        
        distances, ground_rel, flight_rel, waypoint_indices = self.map_widget.get_trajectory_profile()
        if not distances or not ground_rel or not flight_rel:
            self.profile_widget.set_data([], [], [], [], [], [], [], 0, 0)
            return
        
        unsafe_indices = self.map_widget.get_safe_indices(flight_rel, ground_rel, self.min_clearance_spin.value())
        shadow_indices, shadow_after_relay = self.map_widget.check_visibility_with_relay(
            self.operator_antenna_spin.value(),
            self.relay_antenna_spin.value()
        )
        self.shadow_indices = shadow_indices
        self.shadow_after_relay = shadow_after_relay
        
        relay_position_km = None
        relay_lat_lon = None
        relay_altitude = 0
        relay_radius_km = 0
        
        if self.map_widget.relay_point is not None:
            relay_lat, relay_lon = self.map_widget.relay_point
            ground_abs = self.map_widget.get_elevation_at(relay_lat, relay_lon)
            ground_rel_height = ground_abs - self.start_abs_elev
            relay_antenna_height = self.relay_antenna_spin.value()
            
            self.relay_altitude = relay_antenna_height
            relay_dist = calculate_distance(self.start_point[0], self.start_point[1], relay_lat, relay_lon) / 1000
            relay_position_km = relay_dist
            relay_lat_lon = (relay_lat, relay_lon) # Передаем координаты, а не высоту!
            
            relay_radius = self.map_widget.calculate_relay_radius(
                relay_lat, relay_lon, ground_abs + relay_antenna_height, 
                self.start_point[0], self.start_point[1]
            )
            relay_radius_km = relay_radius / 1000
            self.relay_radius_km = relay_radius_km
            
            if shadow_after_relay:
                self.relay_info_label.setText(
                    f"✅ Ретранслятор установлен на земле\nРасстояние: {relay_position_km:.1f} км\n"
                    f"Высота земли: {ground_rel_height:.0f} м отн.\nВысота антенны: {relay_altitude:.0f} м\n"
                    f"Широта: {relay_lat:.5f}\nДолгота: {relay_lon:.5f}\nРадиус: {relay_radius_km:.1f} км\n\n"
                    f"⚠️ Осталось {len(shadow_after_relay)} участков тени"
                )
                self.relay_info_label.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self.relay_info_label.setText(
                    f"✅ Ретранслятор установлен на земле\nРасстояние: {relay_position_km:.1f} км\n"
                    f"Высота земли: {ground_rel_height:.0f} м отн.\nВысота антенны: {relay_altitude:.0f} м\n"
                    f"Широта: {relay_lat:.5f}\nДолгота: {relay_lon:.5f}\nРадиус: {relay_radius_km:.1f} км\n\n✅ Полное покрытие!"
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
        # В конце метода update_profile:

        self.profile_widget.set_data(distances, ground_rel, flight_rel, 
                                     waypoint_indices, unsafe_indices, 
                                     shadow_indices, shadow_after_relay,
                                     self.start_abs_elev, self.min_clearance_spin.value(),
                                     first_waypoint_dist_km,
                                     relay_lat_lon,       
                                     relay_position_km,   # <--- ОБЯЗАТЕЛЬНО ПЕРЕДАЙТЕ ЭТО!
                                     shadow_zones,
                                     relay_altitude, 
                                     relay_radius_km,
                                     self.map_widget)
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
                alt_violations.append({'dist': dist, 'ground': ground_rel[i], 'flight': flight_rel[i], 'clearance': clearance, 'required': self.min_clearance_spin.value(), 'deficit': self.min_clearance_spin.value() - clearance})
        shadow_indices, shadow_after_relay = self.map_widget.check_visibility_with_relay(self.operator_antenna_spin.value(), self.relay_antenna_spin.value())
        self.shadow_indices = shadow_indices
        self.shadow_after_relay = shadow_after_relay
        self.update_profile()
        if alt_violations or shadow_indices or shadow_after_relay:
            self.result_text.setStyleSheet("color: red;")
            msg = "❌ МАРШРУТ НЕБЕЗОПАСЕН!\n\n"
            if alt_violations:
                msg += f"🔴 НАРУШЕНИЕ ВЫСОТЫ!\nНайдено {len(alt_violations)} участков, где БПЛА врежется в землю!\nТребуется зазор: {self.min_clearance_spin.value()} м\n"
            if shadow_indices:
                msg += f"🔴 ЗОНА РАДИОТЕНИ ДО РЕТРАНСЛЯТОРА!\nНа {len(shadow_indices)} участках траектории теряется прямая видимость от старта.\n"
            if shadow_after_relay:
                msg += f"🟣 ЗОНА РАДИОТЕНИ ПОСЛЕ РЕТРАНСЛЯТОРА!\nНа {len(shadow_after_relay)} участках все еще есть проблемы с видимостью.\n"
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
            msg = f"✅ МАРШРУТ БЕЗОПАСЕН!\n\nМинимальный зазор: {min_clearance:.0f} м\nТребуемый зазор: {self.min_clearance_spin.value()} м\n"
            if self.relay_point:
                msg += f"📡 Ретранслятор установлен на земле!\nРасстояние от старта: {self.relay_position_km:.1f} км\nРадиус действия: {self.relay_radius_km:.1f} км\n"
            else:
                msg += f"Радиосвязь: прямая видимость на всей траектории\n"
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
            self.progress_dialog = QProgressDialog("Поиск наивысшей точки рельефа для ретранслятора...", "Отмена", 0, 100, self)
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
                start_lat, start_lon = self.start_point
                ground_abs = self.map_widget.get_elevation_at(relay_pos[0], relay_pos[1])
                ground_rel = ground_abs - self.start_abs_elev
                relay_height = self.relay_antenna_spin.value()
                self.relay_altitude = relay_height
                self.map_widget.set_relay_point(relay_pos[0], relay_pos[1])
                dist = calculate_distance(start_lat, start_lon, relay_pos[0], relay_pos[1])
                self.relay_position_km = dist / 1000
                relay_abs_alt = ground_abs + relay_height
                relay_radius = self.map_widget.calculate_relay_radius(relay_pos[0], relay_pos[1], relay_abs_alt, start_lat, start_lon)
                self.relay_radius_km = relay_radius / 1000
                self.relay_info_label.setText(
                    f"✅ Ретранслятор найден на земле!\nРасстояние от старта: {self.relay_position_km:.1f} км\n"
                    f"Высота земли: {ground_rel:.0f} м отн.\nВысота антенны: {relay_height:.0f} м\n"
                    f"Широта: {relay_pos[0]:.5f}\nДолгота: {relay_pos[1]:.5f}\nРадиус: {self.relay_radius_km:.1f} км\n\n💡 Перетащите ретранслятор для точной настройки"
                )
                self.relay_info_label.setStyleSheet("color: green; font-weight: bold;")
                self.update_profile()
                _, shadow_after = self.map_widget.check_visibility_with_relay(self.operator_antenna_spin.value(), self.relay_antenna_spin.value())
                if shadow_after:
                    QMessageBox.information(self, "Ретранслятор установлен",
                                          f"✅ Ретранслятор установлен на земле!\n\nРасстояние от старта: {self.relay_position_km:.1f} км\n"
                                          f"Высота земли: {ground_rel:.0f} м отн.\nВысота антенны: {relay_height:.0f} м\n"
                                          f"Широта: {relay_pos[0]:.5f}\nДолгота: {relay_pos[1]:.5f}\nРадиус действия: {self.relay_radius_km:.1f} км\n\n"
                                          f"⚠️ Однако {len(shadow_after)} участков все еще в зоне радиотени.\n💡 Попробуйте переместить ретранслятор или увеличить антенну")
                else:
                    QMessageBox.information(self, "Ретранслятор установлен",
                                          f"✅ Ретранслятор успешно установлен на земле!\n\nРасстояние от старта: {self.relay_position_km:.1f} км\n"
                                          f"Высота земли: {ground_rel:.0f} м отн.\nВысота антенны: {relay_height:.0f} м\n"
                                          f"Широта: {relay_pos[0]:.5f}\nДолгота: {relay_pos[1]:.5f}\nРадиус действия: {self.relay_radius_km:.1f} км\n\n📡 Полное покрытие радиосигналом!")
            else:
                self.relay_info_label.setText("❌ Не удалось найти подходящее место\nПопробуйте увеличить высоту антенн или установить вручную")
                self.relay_info_label.setStyleSheet("color: red; font-weight: bold;")
                QMessageBox.warning(self, "Ретранслятор не найден", "Не удалось найти подходящее место для ретранслятора.")
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