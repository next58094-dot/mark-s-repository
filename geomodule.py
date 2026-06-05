import sys
import os
import json
import numpy as np
from datetime import datetime
from math import log, tan, radians, cos, pi, sqrt, sin
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QFileDialog, QMessageBox, QProgressBar,
                             QSlider, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QImage, QFont
import requests
from io import BytesIO

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

def qpixmap_to_pil(pixmap):
    """Конвертация QPixmap в PIL Image"""
    qimage = pixmap.toImage()
    width = qimage.width()
    height = qimage.height()
    
    # Получаем данные изображения
    ptr = qimage.bits()
    ptr.setsize(qimage.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)  # RGBA
    return Image.fromarray(arr, 'RGBA')

class DEMGenerator:
    @staticmethod
    def generate_dem(width, height, center_lat, center_lon, scale_km=10):
        dem_array = np.zeros((height, width))
        scale_factor = 10.0 / scale_km
        mountain_scale = 50 * scale_factor
        hill_scale = 20 * scale_factor
        river_scale = 30 * scale_factor
        
        for y in range(height):
            for x in range(width):
                nx, ny = x / width, y / height
                dist_from_center = sqrt((nx - 0.5)**2 + (ny - 0.5)**2)
                mountain = max(0, 1 - dist_from_center * (1.5 / scale_factor)) * 0.8
                hills = (sin(nx * mountain_scale) * cos(ny * mountain_scale * 0.75) + 
                        sin(nx * hill_scale * 2) * 0.3 + 
                        cos(ny * hill_scale * 1.5) * 0.3) * 0.3
                river_valley = abs(sin(nx * river_scale + ny * river_scale * 0.7)) * 0.2
                noise = np.random.random() * (0.1 / sqrt(scale_factor))
                elevation = mountain + hills + river_valley * 0.5 + noise
                lat_factor = 1 - abs(center_lat) / 90
                elevation = elevation * (0.5 + lat_factor * 0.5)
                dem_array[y, x] = max(0, min(1, elevation))
        return dem_array
    
    @staticmethod
    def dem_to_colormap(dem_array):
        height, width = dem_array.shape
        colored = Image.new('RGBA', (width, height))
        pixels = colored.load()
        for y in range(height):
            for x in range(width):
                e = dem_array[y, x]
                if e < 0.2: r, g, b = 50, 100, 200
                elif e < 0.4:
                    t = (e - 0.2) / 0.2
                    r, g, b = int(100 + t * 50), int(150 + t * 50), int(100 + t * 30)
                elif e < 0.6:
                    t = (e - 0.4) / 0.2
                    r, g, b = int(150 + t * 80), int(200 - t * 50), int(130 - t * 50)
                elif e < 0.8:
                    t = (e - 0.6) / 0.2
                    r, g, b = int(230 - t * 50), int(150 - t * 50), int(80 - t * 30)
                else:
                    intensity = int(200 + (e - 0.8) / 0.2 * 55)
                    r, g, b = intensity, intensity, intensity
                pixels[x, y] = (r, g, b, 255)
        return colored

class MapDownloader(QThread):
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

class MapPreviewArea(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 500)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid gray; background-color: #2b2b2b;")
        self.setText("Загрузите спутниковую карту")
        self.satellite = None
        self.dem = None
        self.combined = None
        self.lat = self.lon = None
        self.size_km = 10
        self.opacity = 0.4
        self.show_dem = True
    
    def set_map(self, pixmap, lat, lon, size_km):
        self.satellite = pixmap
        self.lat, self.lon, self.size_km = lat, lon, size_km
        self.update_display()
    
    def set_dem(self, dem_image):
        self.dem = dem_image
        self.update_display()
    
    def set_opacity(self, opacity):
        self.opacity = opacity
        self.update_display()
    
    def set_show_dem(self, show):
        self.show_dem = show
        self.update_display()
    
    def get_combined_image(self):
        """Получить комбинированное изображение (спутник + DEM)"""
        if self.satellite is None:
            return None
        
        # Создаем копию спутниковой карты
        result = QPixmap(self.satellite)
        
        if self.show_dem and self.dem is not None:
            painter = QPainter(result)
            painter.setOpacity(self.opacity)
            dem_qimage = QImage(self.dem.tobytes("raw", "RGBA"), self.dem.width, self.dem.height, QImage.Format_RGBA8888)
            dem_scaled = QPixmap.fromImage(dem_qimage).scaled(result.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(0, 0, dem_scaled)
            painter.end()
        
        return result
    
    def update_display(self):
        if self.satellite is None:
            return
        result = self.get_combined_image()
        
        # Добавляем рамку и координаты
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
        
        self.combined = result
        self.setPixmap(result.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def resizeEvent(self, event):
        if self.satellite:
            self.update_display()
        super().resizeEvent(event)

class SimpleMapApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Спутниковая карта + DEM")
        self.setGeometry(100, 100, 800, 700)
        
        self.current_satellite = None
        self.current_dem = None
        self.current_lat = None
        self.current_lon = None
        self.current_size = 10
        self.save_dir = os.path.dirname(os.path.abspath(__file__))
        self.downloader = None
        
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
        
        # Панель управления масштабом
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
        
        # Панель DEM
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
        
        self.open_btn = QPushButton("Открыть спутниковую карту")
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
        self.statusBar().showMessage("Готов")
    
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
        # Автоподбор зума
        if size <= 5: zoom = 14
        elif size <= 10: zoom = 13
        elif size <= 20: zoom = 12
        elif size <= 30: zoom = 11
        else: zoom = 10
        
        self.open_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.statusBar().showMessage(f"Загрузка {size}×{size} км...")
        
        self.downloader = MapDownloader(lat, lon, zoom, size)
        self.downloader.progress.connect(self.progress.setValue)
        self.downloader.finished.connect(self.on_map_loaded)
        self.downloader.error.connect(self.on_map_error)
        self.downloader.start()
    
    def on_map_loaded(self, pixmap, lat, lon, zoom, size):
        self.current_satellite = pixmap
        self.current_lat, self.current_lon, self.current_size = lat, lon, size
        
        self.statusBar().showMessage("Генерация рельефа...")
        QApplication.processEvents()
        
        dem_array = DEMGenerator.generate_dem(pixmap.width(), pixmap.height(), lat, lon, size)
        self.current_dem = DEMGenerator.dem_to_colormap(dem_array)
        
        self.preview.set_map(pixmap, lat, lon, size)
        self.preview.set_dem(self.current_dem)
        
        self.save_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.open_btn.setEnabled(True)
        self.statusBar().showMessage(f"Готово! {size}×{size} км")
        
        QMessageBox.information(self, "Успех", f"Загружено {size}×{size} км\nЦентр: {lat:.4f}, {lon:.4f}")
    
    def on_map_error(self, error):
        self.progress.setVisible(False)
        self.open_btn.setEnabled(True)
        self.statusBar().showMessage("Ошибка загрузки")
        QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить:\n{error}")
    
    def save_map(self):
        """Сохранение спутниковой карты, DEM и комбинированного изображения"""
        if self.current_satellite is None:
            QMessageBox.warning(self, "Ошибка", "Нет загруженной карты")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"map_{self.current_size}km_{timestamp}"
        
        # 1. Сохраняем спутниковую карту
        satellite_path = os.path.join(self.save_dir, f"{base}_satellite.png")
        self.current_satellite.save(satellite_path, "PNG")
        
        # 2. Сохраняем DEM (рельеф)
        dem_path = os.path.join(self.save_dir, f"{base}_dem.png")
        if self.current_dem:
            self.current_dem.save(dem_path, "PNG")
        
        # 3. Сохраняем комбинированное изображение (с наложением)
        combined_path = os.path.join(self.save_dir, f"{base}_combined.png")
        combined_image = self.preview.get_combined_image()
        if combined_image:
            combined_image.save(combined_path, "PNG")
        
        # 4. Сохраняем метаданные
        metadata = {
            "type": "satellite_with_dem",
            "latitude": self.current_lat,
            "longitude": self.current_lon,
            "size_km": self.current_size,
            "download_date": datetime.now().isoformat(),
            "dem_opacity": self.opacity_slider.value() / 100.0,
            "show_dem": self.show_dem_cb.isChecked(),
            "files": {
                "satellite": f"{base}_satellite.png",
                "dem": f"{base}_dem.png",
                "combined": f"{base}_combined.png"
            }
        }
        
        json_path = os.path.join(self.save_dir, f"{base}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        QMessageBox.information(self, "Сохранено", 
                                f"Сохранены файлы:\n"
                                f"• Спутниковая карта: {satellite_path}\n"
                                f"• DEM рельеф: {dem_path}\n"
                                f"• Комбинированная карта (с наложением): {combined_path}\n"
                                f"• Метаданные: {json_path}")
        self.statusBar().showMessage("Все файлы сохранены")
    
    def open_saved(self):
        """Открытие сохраненной карты"""
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Выберите файл", 
            self.save_dir, 
            "Изображения (*.png);;Все файлы (*.*)"
        )
        if not path:
            return
        
        try:
            # Определяем тип файла по имени
            filename = os.path.basename(path)
            
            if "_satellite" in filename:
                # Загружаем спутниковую карту
                pixmap = QPixmap(path)
                if pixmap.isNull():
                    QMessageBox.warning(self, "Ошибка", "Не удалось загрузить")
                    return
                
                # Ищем соответствующие файлы
                base = filename.replace("_satellite.png", "")
                dem_path = os.path.join(os.path.dirname(path), f"{base}_dem.png")
                json_path = os.path.join(os.path.dirname(path), f"{base}.json")
                
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        meta = json.load(f)
                    self.current_lat = meta.get('latitude', 0)
                    self.current_lon = meta.get('longitude', 0)
                    self.current_size = meta.get('size_km', 10)
                    self.scale_slider.setValue(self.current_size)
                    self.opacity_slider.setValue(int(meta.get('dem_opacity', 0.4) * 100))
                    self.show_dem_cb.setChecked(meta.get('show_dem', True))
                    
                    # Загружаем DEM если есть
                    if os.path.exists(dem_path):
                        self.current_dem = Image.open(dem_path)
                        self.preview.set_dem(self.current_dem)
                    
                    self.statusBar().showMessage(f"Загружено: {self.current_size}×{self.current_size} км")
                else:
                    self.current_lat = self.current_lon = 0
                    self.current_size = 10
                
                self.current_satellite = pixmap
                self.preview.set_map(pixmap, self.current_lat, self.current_lon, self.current_size)
                self.save_btn.setEnabled(True)
                
            elif "_combined" in filename:
                # Загружаем комбинированное изображение
                pixmap = QPixmap(path)
                if pixmap.isNull():
                    QMessageBox.warning(self, "Ошибка", "Не удалось загрузить")
                    return
                
                self.preview.setPixmap(pixmap.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.statusBar().showMessage("Загружено комбинированное изображение")
                self.save_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "Ошибка", "Неизвестный тип файла")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimpleMapApp()
    window.show()
    sys.exit(app.exec_())