import sys
import os
import json
import numpy as np
from datetime import datetime
from math import log, tan, radians, cos, pi, sqrt
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QFileDialog, QMessageBox, QProgressBar,
                             QSlider, QCheckBox, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QImage, QFont
import requests
from io import BytesIO
import tempfile
import zipfile

# Константы для API OpenTopography
OPENTOPOGRAPHY_API_URL = "https://portal.opentopography.org/API/globaldem"

# Доступные DEM-датасеты
DEM_DATASETS = {
    "SRTMGL3": "SRTM GL3 90m",
    "SRTMGL1": "SRTM GL1 30m", 
    "NASADEM": "NASADEM Global DEM 30m",
    "COP30": "Copernicus Global DSM 30m",
    "COP90": "Copernicus Global DSM 90m",
    "AW3D30": "ALOS World 3D 30m"
}

def lat_lon_to_pixel_mercator(lat, lon, zoom):
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n * 256
    y = (1.0 - log(tan(radians(lat)) + 1.0 / cos(radians(lat))) / pi) / 2.0 * n * 256
    return x, y

def get_satellite_tile_url(zoom, x, y):
    return f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}"

def pil_image_to_qpixmap(pil_image):
    if pil_image.mode == "RGB":
        pil_image = pil_image.convert("RGBA")
    data = pil_image.tobytes("raw", "RGBA")
    qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimage)

def parse_dem_from_bytes(data):
    """Парсинг DEM данных из байтов (поддержка GeoTIFF и HGT)"""
    import io
    
    # Пробуем прочитать как GeoTIFF через PIL
    try:
        from PIL import Image as PILImage
        import struct
        
        # Сохраняем во временный файл для анализа
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        
        # Читаем с помощью PIL и numpy
        with PILImage.open(tmp_path) as img:
            # Конвертируем в numpy array
            dem_array = np.array(img)
            
            # Для GeoTIFF высоты могут быть в разных форматах
            if dem_array.dtype == np.uint16:
                # Преобразуем в метры (для SRTM)
                dem_array = dem_array.astype(np.float32)
            elif dem_array.dtype == np.int16:
                dem_array = dem_array.astype(np.float32)
            
        os.unlink(tmp_path)
        return dem_array
        
    except Exception as e:
        print(f"Ошибка чтения GeoTIFF: {e}")
        
        # Пробуем как бинарный HGT файл (SRTM)
        try:
            # HGT файлы имеют размер 1201x1201 или 3601x3601
            size = int(sqrt(len(data) / 2))
            if size in [1201, 3601]:
                dem_array = np.frombuffer(data, dtype='>i2').reshape((size, size))
                # Заменяем no-data значения (-32768) на NaN
                dem_array = dem_array.astype(np.float32)
                dem_array[dem_array == -32768] = np.nan
                return dem_array
        except:
            pass
    
    return None

class DEMDownloader(QThread):
    """Поток для загрузки DEM с OpenTopography API"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object, float, float)
    error = pyqtSignal(str)
    
    def __init__(self, lat, lon, size_km, dem_type, api_key):
        super().__init__()
        self.lat = lat
        self.lon = lon
        self.size_km = size_km
        self.dem_type = dem_type
        self.api_key = api_key
    
    def run(self):
        try:
            size_deg_lat = (self.size_km / 2) / 111.32
            size_deg_lon = (self.size_km / 2) / (111.32 * cos(radians(self.lat)))
            
            south = self.lat - size_deg_lat
            north = self.lat + size_deg_lat
            west = self.lon - size_deg_lon
            east = self.lon + size_deg_lon
            
            self.progress.emit(10)
            
            # Пробуем получить DEM в формате GeoTIFF
            params = {
                "demtype": self.dem_type,
                "south": south,
                "north": north,
                "west": west,
                "east": east,
                "outputFormat": "GTiff",
                "API_Key": self.api_key
            }
            
            self.progress.emit(30)
            
            response = requests.get(OPENTOPOGRAPHY_API_URL, params=params, timeout=60)
            
            self.progress.emit(50)
            
            if response.status_code == 401:
                # Пробуем с другим форматом
                params["outputFormat"] = "AAIGrid"
                response = requests.get(OPENTOPOGRAPHY_API_URL, params=params, timeout=60)
                
                if response.status_code == 401:
                    self.error.emit("Ошибка авторизации: неверный API-ключ OpenTopography")
                    return
            
            if response.status_code != 200:
                self.error.emit(f"Ошибка API: {response.status_code}")
                return
            
            self.progress.emit(70)
            
            # Парсим полученные данные
            dem_array = parse_dem_from_bytes(response.content)
            
            if dem_array is None:
                # Пробуем как текстовый AAIGrid
                try:
                    content_str = response.content.decode('utf-8')
                    lines = content_str.strip().split('\n')
                    
                    # Парсим AAIGrid формат
                    header = {}
                    data_start = 0
                    for i, line in enumerate(lines):
                        if line.startswith('ncols'):
                            header['ncols'] = int(line.split()[1])
                        elif line.startswith('nrows'):
                            header['nrows'] = int(line.split()[1])
                        elif line.startswith('xllcorner'):
                            header['xllcorner'] = float(line.split()[1])
                        elif line.startswith('yllcorner'):
                            header['yllcorner'] = float(line.split()[1])
                        elif line.startswith('cellsize'):
                            header['cellsize'] = float(line.split()[1])
                        elif line.startswith('NODATA_value'):
                            header['nodata'] = float(line.split()[1])
                            data_start = i + 1
                            break
                    
                    if header:
                        # Читаем данные
                        data_values = []
                        for line in lines[data_start:]:
                            data_values.extend([float(x) for x in line.split()])
                        
                        dem_array = np.array(data_values[:header['ncols'] * header['nrows']])
                        dem_array = dem_array.reshape((header['nrows'], header['ncols']))
                        
                        # Заменяем no-data значения на NaN
                        if 'nodata' in header:
                            dem_array[dem_array == header['nodata']] = np.nan
                        
                except Exception as e:
                    print(f"Ошибка парсинга AAIGrid: {e}")
            
            self.progress.emit(90)
            
            if dem_array is None or dem_array.size == 0:
                self.error.emit("Не удалось распарсить DEM данные")
                return
            
            self.progress.emit(100)
            self.finished.emit(dem_array, south, north)
            
        except Exception as e:
            self.error.emit(f"Ошибка загрузки DEM: {str(e)}")

class MapDownloader(QThread):
    """Поток для загрузки спутниковой карты"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object, float, float, int, float)
    error = pyqtSignal(str)
    
    def __init__(self, lat, lon, zoom, size_km=10):
        super().__init__()
        self.lat, self.lon, self.zoom, self.size_km = lat, lon, zoom, size_km
    
    def run(self):
        try:
            size_deg_lat = (self.size_km / 2) / 111.32
            size_deg_lon = (self.size_km / 2) / (111.32 * cos(radians(self.lat)))
            min_lat, max_lat = self.lat - size_deg_lat, self.lat + size_deg_lat
            min_lon, max_lon = self.lon - size_deg_lon, self.lon + size_deg_lon
            
            x1, y1 = lat_lon_to_pixel_mercator(max_lat, min_lon, self.zoom)
            x2, y2 = lat_lon_to_pixel_mercator(min_lat, max_lon, self.zoom)
            
            tile_size = 256
            min_tx, max_tx = int(x1 // tile_size), int(x2 // tile_size)
            min_ty, max_ty = int(y1 // tile_size), int(y2 // tile_size)
            
            total = (max_tx - min_tx + 1) * (max_ty - min_ty + 1)
            downloaded = 0
            tiles = {}
            
            for x in range(min_tx, max_tx + 1):
                for y in range(min_ty, max_ty + 1):
                    url = get_satellite_tile_url(self.zoom, x, y)
                    try:
                        headers = {'User-Agent': 'Mozilla/5.0'}
                        response = requests.get(url, timeout=10, headers=headers)
                        if response.status_code == 200:
                            img = Image.open(BytesIO(response.content))
                            tiles[(x, y)] = img
                        else:
                            tiles[(x, y)] = Image.new('RGB', (256, 256), (100, 100, 100))
                    except:
                        tiles[(x, y)] = Image.new('RGB', (256, 256), (100, 100, 100))
                    downloaded += 1
                    self.progress.emit(int(downloaded / total * 100))
            
            width = (max_tx - min_tx + 1) * tile_size
            height = (max_ty - min_ty + 1) * tile_size
            combined = Image.new('RGB', (width, height))
            for (x, y), tile in tiles.items():
                combined.paste(tile, ((x - min_tx) * tile_size, (y - min_ty) * tile_size))
            
            x_start, x_end = int(x1 - min_tx * tile_size), int(x2 - min_tx * tile_size)
            y_start, y_end = int(y1 - min_ty * tile_size), int(y2 - min_ty * tile_size)
            cropped = combined.crop((x_start, y_start, x_end, y_end))
            self.finished.emit(pil_image_to_qpixmap(cropped), self.lat, self.lon, self.zoom, self.size_km)
        except Exception as e:
            self.error.emit(str(e))

def array_to_colored_png(dem_array, min_elev=None, max_elev=None):
    """Конвертация массива DEM в цветное PNG-изображение"""
    if dem_array is None or dem_array.size == 0:
        return None
    
    # Заменяем NaN на среднее значение
    dem_array = np.nan_to_num(dem_array, nan=np.nanmean(dem_array))
    
    if min_elev is None:
        min_elev = np.min(dem_array)
    if max_elev is None:
        max_elev = np.max(dem_array)
    
    if max_elev - min_elev < 0.01:
        return None
    
    normalized = (dem_array - min_elev) / (max_elev - min_elev)
    normalized = np.clip(normalized, 0, 1)
    
    height, width = normalized.shape
    colored = Image.new('RGBA', (width, height))
    pixels = colored.load()
    
    for y in range(height):
        for x in range(width):
            e = normalized[y, x]
            
            if e < 0.1:
                r, g, b = 30, 80, 120
            elif e < 0.25:
                t = (e - 0.1) / 0.15
                r, g, b = 60 + int(t * 40), 100 + int(t * 50), 80 + int(t * 40)
            elif e < 0.5:
                t = (e - 0.25) / 0.25
                r, g, b = 100 + int(t * 60), 150 + int(t * 40), 120 - int(t * 20)
            elif e < 0.75:
                t = (e - 0.5) / 0.25
                r, g, b = 160 + int(t * 60), 190 - int(t * 40), 100 - int(t * 20)
            else:
                t = (e - 0.75) / 0.25
                intensity = 220 + int(t * 35)
                r, g, b = intensity, intensity, intensity
            
            pixels[x, y] = (r, g, b, 255)
    
    return colored

def save_dem_as_png(dem_array, filepath):
    """Сохраняет DEM массив как PNG изображение"""
    colored = array_to_colored_png(dem_array)
    if colored:
        colored.save(filepath, "PNG")
        return True
    return False

def save_dem_as_npy(dem_array, filepath):
    """Сохраняет DEM массив как NPY файл (numpy binary)"""
    np.save(filepath, dem_array)
    return True

class MapPreviewArea(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 500)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid gray; background-color: #2b2b2b;")
        self.setText("Загрузите спутниковую карту")
        self.satellite = None
        self.dem = None
        self.lat = self.lon = None
        self.size_km = 10
        self.opacity = 0.4
        self.show_dem = True
    
    def set_map(self, pixmap, lat, lon, size_km):
        self.satellite = pixmap
        self.lat, self.lon, self.size_km = lat, lon, size_km
        self.update_display()
    
    def set_dem(self, dem_array):
        if dem_array is not None and dem_array.size > 0:
            # Изменяем размер DEM под размер спутниковой карты
            if dem_array.shape != (self.satellite.height(), self.satellite.width()):
                # Ресайзим DEM массив
                from scipy import ndimage
                zoom_y = self.satellite.height() / dem_array.shape[0]
                zoom_x = self.satellite.width() / dem_array.shape[1]
                dem_array = ndimage.zoom(dem_array, (zoom_y, zoom_x), order=1)
            
            self.dem_array = dem_array
            self.dem = array_to_colored_png(dem_array)
        else:
            self.dem = None
            self.dem_array = None
        self.update_display()
    
    def set_opacity(self, opacity):
        self.opacity = opacity
        self.update_display()
    
    def set_show_dem(self, show):
        self.show_dem = show
        self.update_display()
    
    def get_combined_image(self):
        if self.satellite is None:
            return None
        
        result = QPixmap(self.satellite)
        
        if self.show_dem and self.dem is not None:
            painter = QPainter(result)
            painter.setOpacity(self.opacity)
            dem_qimage = QImage(self.dem.tobytes("raw", "RGBA"), 
                               self.dem.width, self.dem.height, 
                               QImage.Format_RGBA8888)
            dem_pixmap = QPixmap.fromImage(dem_qimage)
            dem_scaled = dem_pixmap.scaled(result.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(0, 0, dem_scaled)
            painter.end()
        
        return result
    
    def update_display(self):
        if self.satellite is None:
            return
        result = self.get_combined_image()
        
        painter = QPainter(result)
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        painter.drawRect(0, 0, result.width() - 1, result.height() - 1)
        cx, cy = result.width() // 2, result.height() // 2
        painter.drawLine(cx - 10, cy, cx + 10, cy)
        painter.drawLine(cx, cy - 10, cx, cy + 10)
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.setFont(QFont("Arial", 9))
        painter.drawText(10, 25, f"{self.lat:.4f}, {self.lon:.4f} | {self.size_km}×{self.size_km} км")
        painter.end()
        
        self.setPixmap(result.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def get_dem_array(self):
        return getattr(self, 'dem_array', None)
    
    def resizeEvent(self, event):
        if self.satellite:
            self.update_display()
        super().resizeEvent(event)

class SimpleMapApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Спутниковая карта + DEM (SRTM/NASADEM)")
        self.setGeometry(100, 100, 900, 750)
        
        self.current_satellite = None
        self.current_dem_array = None
        self.current_lat = None
        self.current_lon = None
        self.current_size = 10
        self.save_dir = os.path.dirname(os.path.abspath(__file__))
        self.downloader = None
        self.dem_downloader = None
        
        # API ключ OpenTopography
        self.api_key = "24d6c79232d14e0a759afb5979dbc4ec"
        
        self.setup_ui()
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)
        
        # Панель координат
        coord_widget = QWidget()
        coord_layout = QHBoxLayout(coord_widget)
        coord_layout.setContentsMargins(0, 0, 0, 0)
        
        coord_layout.addWidget(QLabel("Широта:"))
        self.lat_input = QLineEdit()
        self.lat_input.setText("48.6337")
        self.lat_input.setFixedWidth(100)
        coord_layout.addWidget(self.lat_input)
        
        coord_layout.addWidget(QLabel("Долгота:"))
        self.lon_input = QLineEdit()
        self.lon_input.setText("38.3765")
        self.lon_input.setFixedWidth(100)
        coord_layout.addWidget(self.lon_input)
        
        coord_layout.addStretch()
        layout.addWidget(coord_widget)
        
        # Панель масштаба
        scale_widget = QWidget()
        scale_layout = QHBoxLayout(scale_widget)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        
        scale_layout.addWidget(QLabel("Масштаб:"))
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setMinimum(5)
        self.scale_slider.setMaximum(50)
        self.scale_slider.setValue(10)
        self.scale_slider.setFixedWidth(200)
        self.scale_slider.valueChanged.connect(self.on_scale_changed)
        scale_layout.addWidget(self.scale_slider)
        
        self.scale_label = QLabel("10 км")
        self.scale_label.setFixedWidth(50)
        scale_layout.addWidget(self.scale_label)
        
        scale_layout.addStretch()
        layout.addWidget(scale_widget)
        
        # Панель DEM (выбор источника)
        dem_source_widget = QWidget()
        dem_source_layout = QHBoxLayout(dem_source_widget)
        dem_source_layout.setContentsMargins(0, 0, 0, 0)
        
        dem_source_layout.addWidget(QLabel("Источник DEM:"))
        self.dem_combo = QComboBox()
        for key, name in DEM_DATASETS.items():
            self.dem_combo.addItem(name, key)
        dem_source_layout.addWidget(self.dem_combo)
        
        dem_source_layout.addStretch()
        layout.addWidget(dem_source_widget)
        
        # Панель прозрачности
        dem_widget = QWidget()
        dem_layout = QHBoxLayout(dem_widget)
        dem_layout.setContentsMargins(0, 0, 0, 0)
        
        self.show_dem_cb = QCheckBox("Показать рельеф")
        self.show_dem_cb.setChecked(True)
        self.show_dem_cb.stateChanged.connect(self.toggle_dem)
        dem_layout.addWidget(self.show_dem_cb)
        
        dem_layout.addWidget(QLabel("Прозрачность:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(40)
        self.opacity_slider.setFixedWidth(150)
        self.opacity_slider.valueChanged.connect(self.change_opacity)
        dem_layout.addWidget(self.opacity_slider)
        
        self.opacity_label = QLabel("40%")
        self.opacity_label.setFixedWidth(40)
        dem_layout.addWidget(self.opacity_label)
        
        dem_layout.addStretch()
        layout.addWidget(dem_widget)
        
        # Кнопки
        btn_widget = QWidget()
        btn_layout = QHBoxLayout(btn_widget)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        
        self.open_btn = QPushButton("Открыть спутниковую карту + DEM")
        self.open_btn.clicked.connect(self.open_map)
        self.open_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 5px; }")
        
        self.save_btn = QPushButton("Сохранить карту + метаданные")
        self.save_btn.clicked.connect(self.save_map)
        self.save_btn.setEnabled(False)
        
        self.load_btn = QPushButton("Открыть сохранённую")
        self.load_btn.clicked.connect(self.open_saved)
        
        btn_layout.addWidget(self.open_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.load_btn)
        layout.addWidget(btn_widget)
        
        # Прогресс
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Область карты
        self.preview = MapPreviewArea()
        layout.addWidget(self.preview)
        
        # Статус
        self.statusBar().showMessage("Готов. API-ключ OpenTopography установлен")
    
    def on_scale_changed(self, value):
        self.scale_label.setText(f"{value} км")
        self.current_size = value
    
    def toggle_dem(self, state):
        self.preview.set_show_dem(state == Qt.Checked)
    
    def change_opacity(self, value):
        self.preview.set_opacity(value / 100.0)
        self.opacity_label.setText(f"{value}%")
    
    def validate_coords(self):
        try:
            lat = float(self.lat_input.text())
            lon = float(self.lon_input.text())
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
            QMessageBox.warning(self, "Ошибка", "Координаты вне диапазона")
            return None, None
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Введите числа")
            return None, None
    
    def open_map(self):
        lat, lon = self.validate_coords()
        if lat is None:
            return
        
        size = self.current_size
        if size <= 5: zoom = 14
        elif size <= 10: zoom = 13
        elif size <= 20: zoom = 12
        elif size <= 30: zoom = 11
        else: zoom = 10
        
        self.open_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        self.statusBar().showMessage(f"Загрузка спутниковой карты {size}×{size} км...")
        
        self.downloader = MapDownloader(lat, lon, zoom, size)
        self.downloader.progress.connect(self.progress.setValue)
        self.downloader.finished.connect(self.on_satellite_loaded)
        self.downloader.error.connect(self.on_map_error)
        self.downloader.start()
    
    def on_satellite_loaded(self, pixmap, lat, lon, zoom, size):
        self.current_satellite = pixmap
        self.current_lat, self.current_lon, self.current_size = lat, lon, size
        
        self.statusBar().showMessage(f"Загрузка DEM с OpenTopography API...")
        self.progress.setValue(0)
        
        dem_type_key = self.dem_combo.currentData()
        
        self.dem_downloader = DEMDownloader(lat, lon, size, dem_type_key, self.api_key)
        self.dem_downloader.progress.connect(self.progress.setValue)
        self.dem_downloader.finished.connect(self.on_dem_loaded)
        self.dem_downloader.error.connect(self.on_dem_error)
        self.dem_downloader.start()
    
    def on_dem_loaded(self, dem_array, south, north):
        self.current_dem_array = dem_array
        
        self.preview.set_map(self.current_satellite, self.current_lat, self.current_lon, self.current_size)
        self.preview.set_dem(dem_array)
        
        self.progress.setVisible(False)
        self.open_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        
        if dem_array is not None and dem_array.size > 0:
            min_elev = np.nanmin(dem_array)
            max_elev = np.nanmax(dem_array)
            mean_elev = np.nanmean(dem_array)
            self.statusBar().showMessage(
                f"Готово! {self.current_size}×{self.current_size} км, "
                f"DEM: {min_elev:.0f}-{max_elev:.0f} м, средняя {mean_elev:.0f} м"
            )
            QMessageBox.information(self, "Успех", 
                f"Загружено:\n"
                f"Спутниковая карта: {self.current_size}×{self.current_size} км\n"
                f"DEM: {DEM_DATASETS[self.dem_combo.currentData()]}\n"
                f"Высоты: от {min_elev:.0f} до {max_elev:.0f} м")
        else:
            self.statusBar().showMessage(f"Готово! {self.current_size}×{self.current_size} км")
    
    def on_dem_error(self, error):
        self.progress.setVisible(False)
        self.open_btn.setEnabled(True)
        self.preview.set_map(self.current_satellite, self.current_lat, self.current_lon, self.current_size)
        self.save_btn.setEnabled(True)
        self.statusBar().showMessage("Готово (DEM не загружен)")
        QMessageBox.warning(self, "Предупреждение", f"Не удалось загрузить DEM:\n{error}")
    
    def on_map_error(self, error):
        self.progress.setVisible(False)
        self.open_btn.setEnabled(True)
        self.statusBar().showMessage("Ошибка загрузки")
        QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить карту:\n{error}")
    
    def save_map(self):
        if self.current_satellite is None:
            QMessageBox.warning(self, "Ошибка", "Нет загруженной карты")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"map_{self.current_size}km_{timestamp}"
        
        # 1. Сохраняем спутниковую карту
        satellite_path = os.path.join(self.save_dir, f"{base}_satellite.png")
        self.current_satellite.save(satellite_path, "PNG")
        
        # 2. Сохраняем DEM как NPY и PNG
        dem_npy_path = None
        dem_png_path = None
        
        if self.current_dem_array is not None and self.current_dem_array.size > 0:
            dem_npy_path = os.path.join(self.save_dir, f"{base}_dem.npy")
            save_dem_as_npy(self.current_dem_array, dem_npy_path)
            
            dem_png_path = os.path.join(self.save_dir, f"{base}_dem.png")
            save_dem_as_png(self.current_dem_array, dem_png_path)
        
        # 3. Сохраняем комбинированное изображение
        combined_path = os.path.join(self.save_dir, f"{base}_combined.png")
        combined_image = self.preview.get_combined_image()
        if combined_image:
            combined_image.save(combined_path, "PNG")
        
        # 4. Метаданные
        metadata = {
            "type": "satellite_with_real_dem",
            "latitude": self.current_lat,
            "longitude": self.current_lon,
            "size_km": self.current_size,
            "download_date": datetime.now().isoformat(),
            "dem_source": self.dem_combo.currentData(),
            "dem_opacity": self.opacity_slider.value() / 100.0,
            "show_dem": self.show_dem_cb.isChecked(),
            "files": {
                "satellite": f"{base}_satellite.png",
                "combined": f"{base}_combined.png"
            }
        }
        
        if dem_npy_path:
            metadata["files"]["dem_npy"] = f"{base}_dem.npy"
        if dem_png_path:
            metadata["files"]["dem_png"] = f"{base}_dem.png"
        
        json_path = os.path.join(self.save_dir, f"{base}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        msg = f"Сохранены файлы:\n• {satellite_path}\n• {combined_path}"
        if dem_png_path:
            msg += f"\n• {dem_png_path}"
        if dem_npy_path:
            msg += f"\n• {dem_npy_path}"
        msg += f"\n• {json_path}"
        
        QMessageBox.information(self, "Сохранено", msg)
        self.statusBar().showMessage("Все файлы сохранены")
    
    def open_saved(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл", self.save_dir, 
            "Изображения (*.png);;NPY файлы (*.npy);;Все файлы (*.*)"
        )
        if not path:
            return
        
        try:
            if path.endswith('.npy'):
                # Загружаем DEM из NPY
                dem_array = np.load(path)
                if dem_array is not None:
                    self.current_dem_array = dem_array
                    self.preview.set_dem(dem_array)
                    
                    # Пытаемся найти соответствующий JSON
                    json_path = path.replace('.npy', '.json')
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            meta = json.load(f)
                        self.current_lat = meta.get('latitude', 0)
                        self.current_lon = meta.get('longitude', 0)
                        self.current_size = meta.get('size_km', 10)
                        self.scale_slider.setValue(self.current_size)
                        self.opacity_slider.setValue(int(meta.get('dem_opacity', 0.4) * 100))
                        self.show_dem_cb.setChecked(meta.get('show_dem', True))
                    
                    self.statusBar().showMessage(f"Загружен DEM: {self.current_size}×{self.current_size} км")
                    self.save_btn.setEnabled(True)
                    return
            else:
                # Загружаем изображение
                pixmap = QPixmap(path)
                if pixmap.isNull():
                    QMessageBox.warning(self, "Ошибка", "Не удалось загрузить")
                    return
                
                json_path = path.replace('_satellite.png', '.json').replace('_combined.png', '.json')
                if not os.path.exists(json_path):
                    json_path = path.replace('.png', '.json')
                
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        meta = json.load(f)
                    self.current_lat = meta.get('latitude', 0)
                    self.current_lon = meta.get('longitude', 0)
                    self.current_size = meta.get('size_km', 10)
                    self.scale_slider.setValue(self.current_size)
                    self.opacity_slider.setValue(int(meta.get('dem_opacity', 0.4) * 100))
                    self.show_dem_cb.setChecked(meta.get('show_dem', True))
                    self.statusBar().showMessage(f"Загружено: {self.current_size}×{self.current_size} км")
                    
                    # Загружаем DEM из NPY если есть
                    dem_npy = meta.get('files', {}).get('dem_npy')
                    if dem_npy:
                        dem_path = os.path.join(os.path.dirname(path), dem_npy)
                        if os.path.exists(dem_path):
                            self.current_dem_array = np.load(dem_path)
                            self.preview.set_dem(self.current_dem_array)
                else:
                    self.current_lat = self.current_lon = 0
                    self.current_size = 10
                
                self.current_satellite = pixmap
                self.preview.set_map(pixmap, self.current_lat, self.current_lon, self.current_size)
                self.save_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimpleMapApp()
    window.show()
    sys.exit(app.exec_())