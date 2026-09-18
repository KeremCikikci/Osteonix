import sys
import os
import json
import traceback
from functools import partial

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QListWidget, QLabel, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QPushButton, QFileDialog, QSplitter, QListWidgetItem, QMenu, QAbstractItemView,
    QInputDialog, QSizePolicy, QShortcut, QCheckBox, QSpinBox, QGridLayout, QColorDialog,
    QFrame, QRubberBand, QGraphicsRectItem, QGraphicsItem, QProgressDialog
)
from PyQt5.QtGui import (
    QPixmap, QCursor, QImageReader, QKeySequence, QColor, QPainter, QIcon, QPen, QBrush, QPalette, QFont
)
from PyQt5.QtCore import (
    Qt, QRectF, QTimer, QSize, QPoint, QRect, QStandardPaths, QObject, pyqtSignal, QRunnable, QThreadPool
)

# --- Yeni / eklenen importlar için try/except (opsiyonel paketler) ---
# Bu uygulama, kullanıcı tarafında best.pt dosyasıyla çalışacak bir YOLO modeline bağlanmaya çalışır.
# Hem `ultralytics` paketinin yeni YOLO API'sini, hem de torch.hub yoluyla yolov11 'custom' yüklemeyi dener.
try:
    import torch
except Exception:
    torch = None

_ultralytics_available = False
_yolomodel_cls = None
try:
    from ultralytics import YOLO
    _ultralytics_available = True
except Exception:
    _ultralytics_available = False

def resource_path(relative_path):
    """ PyInstaller için: exe içindeyken de doğru yolu döndürür """
    import sys, os
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# -----------------------
# Configuration
# -----------------------
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
CLASS_MAP = {
    0: "osteochondroma",
    1: "simple bone cyst",
    2: "giant cell tumor",
    3: "osteofibroma",
    4: "other bt",
    5: "osteosarcoma",
    6: "other mt",
    # alias
    "multiple osteochondromas": "osteochondroma",
}
THUMB_SIZE = 56
GLOBAL_FONT_POINT = 10  # user requested: make font a bit smaller
APP_NAME = "ImageRectViewer"  # used for config folder
APP_CONFIG_DIR = QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation) or os.path.join(os.path.expanduser("~"), ".config")
APP_CONFIG_DIR = os.path.join(APP_CONFIG_DIR, APP_NAME)
if not os.path.exists(APP_CONFIG_DIR):
    try:
        os.makedirs(APP_CONFIG_DIR, exist_ok=True)
    except Exception:
        pass
GLOBAL_RECTS_FILENAME = os.path.join(APP_CONFIG_DIR, "rects_all.json")

# -------------------------
# Worker for generating thumbnails in background
# (unchanged)
# -------------------------
class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)  # will emit a tuple (index, pixmap, path)

class ThumbnailWorker(QRunnable):
    def __init__(self, index, path, size=THUMB_SIZE):
        super().__init__()
        self.signals = WorkerSignals()
        self.index = index
        self.path = path
        self.size = size

    def run(self):
        try:
            reader = QImageReader(self.path)
            reader.setAutoTransform(True)
            image = reader.read()
            if image.isNull():
                pm = QPixmap(self.size, self.size)
                pm.fill(Qt.transparent)
            else:
                pm = QPixmap.fromImage(image)
                pm = pm.scaled(self.size, self.size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                canvas = QPixmap(self.size, self.size)
                canvas.fill(Qt.transparent)
                painter = QPainter(canvas)
                x = (self.size - pm.width()) // 2
                y = (self.size - pm.height()) // 2
                painter.drawPixmap(x, y, pm)
                painter.end()
                pm = canvas
            self.signals.result.emit((self.index, pm, self.path))
        except Exception:
            import traceback
            tb = traceback.format_exc()
            BEST_MODEL_PATH = r'best.pt'  # Update this path when you have your best.pt file
            self.signals.error.emit((self.path, tb))
        finally:
            self.signals.finished.emit()

# -------------------------
# Small toast notification helper (unchanged)
# -------------------------
class Toast(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.label = QLabel("", self)
        self.label.setStyleSheet("background: rgba(0,0,0,0.78); color: white; padding: 8px; border-radius: 4px;")
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.setContentsMargins(0, 0, 0, 0)
        self.hide_timer = QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self.hide)

    def show_toast(self, message, duration=2200):
        self.label.setText(message)
        self.adjustSize()
        # position bottom-right of parent
        if self.parent():
            pw = self.parent().width()
            ph = self.parent().height()
            sw = self.width()
            sh = self.height()
            x = self.parent().x() + pw - sw - 20
            y = self.parent().y() + ph - sh - 40
            self.move(x, y)
        else:
            self.move(200, 200)
        self.show()
        self.raise_()
        self.hide_timer.start(duration)

# -------------------------
# Draggable & resizable rect (unchanged)
# -------------------------
class DraggableRectItem(QGraphicsRectItem):
    HANDLE_SIZE = 8

    def __init__(self, x, y, w, h, index, color_hex="#ff0000", parent=None, on_update=None, on_select=None):
        super().__init__(0, 0, w, h)
        self.setPos(x, y)
        self.index = index
        self.on_update = on_update
        self.on_select = on_select
        self._color = QColor(color_hex)
        pen = QPen(self._color)
        pen.setWidth(2)
        self.setPen(pen)
        self.setBrush(QBrush())
        self.setZValue(10)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.handles = {}
        self._init_handles()
        self._resizing = None
        self._mouse_press_pos = None
        self.setAcceptHoverEvents(True)

    def _init_handles(self):
        r = self.rect()
        s = self.HANDLE_SIZE
        self.handles = {
            "tl": QRectF(r.left(), r.top(), s, s),
            "tr": QRectF(r.right() - s, r.top(), s, s),
            "bl": QRectF(r.left(), r.bottom() - s, s, s),
            "br": QRectF(r.right() - s, r.bottom() - s, s, s),
            "top": QRectF(r.center().x() - s/2, r.top(), s, s),
            "bottom": QRectF(r.center().x() - s/2, r.bottom() - s, s, s),
            "left": QRectF(r.left(), r.center().y() - s/2, s, s),
            "right": QRectF(r.right() - s, r.center().y() - s/2, s, s),
        }

    def _update_handles(self):
        r = self.rect()
        s = self.HANDLE_SIZE
        self.handles["tl"].moveTo(r.left(), r.top())
        self.handles["tr"].moveTo(r.right() - s, r.top())
        self.handles["bl"].moveTo(r.left(), r.bottom() - s)
        self.handles["br"].moveTo(r.right() - s, r.bottom() - s)
        self.handles["top"].moveTo(r.center().x() - s/2, r.top())
        self.handles["bottom"].moveTo(r.center().x() - s/2, r.bottom() - s)
        self.handles["left"].moveTo(r.left(), r.center().y() - s/2)
        self.handles["right"].moveTo(r.right() - s, r.center().y() - s/2)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        painter.save()
        pen = QPen(self.pen().color())
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(QBrush(self.pen().color()))
        for rect in self.handles.values():
            painter.drawRect(rect)
        painter.restore()

    def hoverMoveEvent(self, event):
        pos = event.pos()
        for key, rect in self.handles.items():
            if rect.contains(pos):
                if key in ("tl", "br"):
                    self.setCursor(Qt.SizeFDiagCursor)
                elif key in ("tr", "bl"):
                    self.setCursor(Qt.SizeBDiagCursor)
                elif key in ("top", "bottom"):
                    self.setCursor(Qt.SizeVerCursor)
                elif key in ("left", "right"):
                    self.setCursor(Qt.SizeHorCursor)
                return
        self.setCursor(Qt.SizeAllCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        pos = event.pos()
        for key, rect in self.handles.items():
            if rect.contains(pos):
                self._resizing = key
                self._mouse_press_pos = pos
                if key in ("tl", "br"):
                    self.setCursor(Qt.SizeFDiagCursor)
                elif key in ("tr", "bl"):
                    self.setCursor(Qt.SizeBDiagCursor)
                elif key in ("top", "bottom"):
                    self.setCursor(Qt.SizeVerCursor)
                elif key in ("left", "right"):
                    self.setCursor(Qt.SizeHorCursor)
                event.accept()
                return
        self._resizing = None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resizing:
            r = self.rect()
            pos = event.pos()
            try:
                if self._resizing == "tl":
                    new_left = min(pos.x(), r.right() - 1)
                    new_top = min(pos.y(), r.bottom() - 1)
                    new_rect = QRectF(new_left, new_top, r.right() - new_left, r.bottom() - new_top)
                    if new_rect.width() < 1: new_rect.setWidth(1)
                    if new_rect.height() < 1: new_rect.setHeight(1)
                    self.setRect(new_rect)
                elif self._resizing == "tr":
                    new_right = max(pos.x(), r.left() + 1)
                    new_top = min(pos.y(), r.bottom() - 1)
                    new_rect = QRectF(r.left(), new_top, new_right - r.left(), r.bottom() - new_top)
                    self.setRect(new_rect)
                elif self._resizing == "bl":
                    new_left = min(pos.x(), r.right() - 1)
                    new_bottom = max(pos.y(), r.top() + 1)
                    new_rect = QRectF(new_left, r.top(), r.right() - new_left, new_bottom - r.top())
                    self.setRect(new_rect)
                elif self._resizing == "br":
                    new_right = max(pos.x(), r.left() + 1)
                    new_bottom = max(pos.y(), r.top() + 1)
                    new_rect = QRectF(r.left(), r.top(), new_right - r.left(), new_bottom - r.top())
                    self.setRect(new_rect)
                elif self._resizing == "top":
                    new_top = min(pos.y(), r.bottom() - 1)
                    new_rect = QRectF(r.left(), new_top, r.width(), r.bottom() - new_top)
                    if new_rect.height() < 1: new_rect.setHeight(1)
                    self.setRect(new_rect)
                elif self._resizing == "bottom":
                    new_bottom = max(pos.y(), r.top() + 1)
                    new_rect = QRectF(r.left(), r.top(), r.width(), new_bottom - r.top())
                    if new_rect.height() < 1: new_rect.setHeight(1)
                    self.setRect(new_rect)
                elif self._resizing == "left":
                    new_left = min(pos.x(), r.right() - 1)
                    new_rect = QRectF(new_left, r.top(), r.right() - new_left, r.height())
                    if new_rect.width() < 1: new_rect.setWidth(1)
                    self.setRect(new_rect)
                elif self._resizing == "right":
                    new_right = max(pos.x(), r.left() + 1)
                    new_rect = QRectF(r.left(), r.top(), new_right - r.left(), r.height())
                    if new_rect.width() < 1: new_rect.setWidth(1)
                    self.setRect(new_rect)
            except Exception:
                pass
            self._update_handles()
            event.accept()
            return
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        rect = self.sceneBoundingRect()
        new_x = int(rect.x())
        new_y = int(rect.y())
        new_w = max(1, int(rect.width()))
        new_h = max(1, int(rect.height()))
        self._update_handles()
        if self.on_update:
            try:
                self.on_update(new_x, new_y, new_w, new_h, self.index)
            except Exception:
                pass
        if self.isSelected() and self.on_select:
            try:
                self.on_select(self.index)
            except Exception:
                pass
        self._resizing = None
        self._mouse_press_pos = None
        self.setCursor(Qt.ArrowCursor)

    def update_color(self, color_hex):
        self._color = QColor(color_hex)
        pen = QPen(self._color)
        pen.setWidth(2)
        self.setPen(pen)

# -------------------------
# ImageViewer (unchanged)
# -------------------------
class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = None
        self._current_scale = 1.0
        self._min_scale = 0.2
        self._max_scale = 10.0
        self.parent = parent
        self.rect_overlays = []
        self._is_panning = False
        self.horizontalScrollBar().valueChanged.connect(self._on_view_changed)
        self.verticalScrollBar().valueChanged.connect(self._on_view_changed)

    def load_image(self, image_path):
        self._scene.clear()
        self.rect_overlays = []
        if not os.path.exists(image_path):
            return
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)
        self.setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self._current_scale = 1.0
        if self.parent:
            self.parent.update_rects_on_view()
            self.parent.update_info()

    def clear_image(self):
        self._scene.clear()
        self._pixmap_item = None
        self.rect_overlays = []
        self._current_scale = 1.0
        if self.parent:
            self.parent.update_info()

    def add_rect_overlay(self, rect, color: QColor, index):
        if not self._pixmap_item:
            return
        x, y, w, h = rect
        color_hex = QColor(color).name()
        def on_update(new_x, new_y, new_w, new_h, idx):
            if self.parent:
                self.parent._on_overlay_moved(idx, new_x, new_y, new_w, new_h)
        def on_select(idx):
            if self.parent:
                self.parent._on_overlay_selected(idx)
        r_item = DraggableRectItem(x, y, w, h, index, color_hex=color_hex, on_update=on_update, on_select=on_select)
        self._scene.addItem(r_item)
        self.rect_overlays.append(r_item)
        return r_item

    def clear_rect_overlays(self):
        for it in list(self.rect_overlays):
            try:
                self._scene.removeItem(it)
            except Exception:
                pass
        self.rect_overlays = []

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        new_scale = self._current_scale * zoom_factor
        if self._min_scale <= new_scale <= self._max_scale:
            self.scale(zoom_factor, zoom_factor)
            self._current_scale = new_scale
            if self.parent:
                self.parent.update_info()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap_item:
            if abs(self._current_scale - 1.0) < 1e-6:
                self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
                self._current_scale = 1.0
        if self.parent:
            self.parent.update_info()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._is_panning = True
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)
        if self.parent:
            self.parent.update_info()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._is_panning = False
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)
        if self.parent:
            self.parent.update_info()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self.parent:
            self.parent.update_info()

    def keyPressEvent(self, event):
        step = 50
        if event.key() in (Qt.Key_W, Qt.Key_Up):
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - step)
            event.accept()
        elif event.key() in (Qt.Key_S, Qt.Key_Down):
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + step)
            event.accept()
        elif event.key() in (Qt.Key_A, Qt.Key_Left):
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - step)
            event.accept()
        elif event.key() in (Qt.Key_D, Qt.Key_Right):
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + step)
            event.accept()
        else:
            super().keyPressEvent(event)
        if self.parent:
            self.parent.update_info()

    def _on_view_changed(self, _val=None):
        if self.parent:
            self.parent.update_info()

# -------------------------
# PhotoList (unchanged)
# -------------------------
class PhotoList(QListWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setAcceptDrops(True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.open_menu)
        self.setMouseTracking(True)

        # alternating rows - more contrast as requested (white / ececec)
        self.setAlternatingRowColors(True)
        pal = self.palette()
        pal.setColor(QPalette.Base, QColor("#ffffff"))
        pal.setColor(QPalette.AlternateBase, QColor("#ececec"))  # slightly darker alt base
        self.setPalette(pal)

        self._base_style = """
            QListWidget {
                outline: none;
            }
            QListWidget::item { padding: 4px; }
            QListWidget::item:selected {
                background: #cfe8ff;
                color: black;
            }
        """
        self.setStyleSheet(self._base_style)

        # rubber-band selection
        self._rubber = QRubberBand(QRubberBand.Rectangle, self.viewport())
        self._rubber_origin = None
        self._rubber_active = False
        self._rubber_moved = False
        self._ctrl_click_candidate = None
        self._ctrl_click_press_pos = None

        # Thumbs flag and threadpool
        self.thumbs_enabled = False
        self.threadpool = QThreadPool.globalInstance()
        self._pending_thumb_jobs = {}

        self.itemSelectionChanged.connect(self._on_selection_changed)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and (event.modifiers() & Qt.ControlModifier):
            self._rubber_origin = event.pos()
            self._rubber.setGeometry(QRect(self._rubber_origin, QSize()))
            self._rubber.show()
            self._rubber_active = True
            self._rubber_moved = False
            self._ctrl_click_candidate = self.itemAt(event.pos())
            self._ctrl_click_press_pos = event.pos()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._rubber_active:
            rect = QRect(self._rubber_origin, event.pos()).normalized()
            if (event.pos() - self._rubber_origin).manhattanLength() > 4:
                self._rubber_moved = True
                self._rubber.setGeometry(rect)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._rubber_active and event.button() == Qt.LeftButton:
            release_pos = event.pos()
            self._rubber.hide()
            self._rubber_active = False
            rect = QRect(self._rubber_origin, release_pos).normalized()
            selected = []
            for i in range(self.count()):
                it = self.item(i)
                r = self.visualItemRect(it)
                if rect.intersects(r):
                    selected.append(it)
            if not self._rubber_moved:
                it = self._ctrl_click_candidate or self.itemAt(release_pos)
                if it:
                    it.setSelected(not it.isSelected())
            else:
                if not (event.modifiers() & Qt.ControlModifier):
                    self.clearSelection()
                for it in selected:
                    it.setSelected(True)
            if self.parent:
                self.parent.update_info()
            self._ctrl_click_candidate = None
            self._ctrl_click_press_pos = None
            return
        super().mouseReleaseEvent(event)
    

    def _on_selection_changed(self):
        if self.parent:
            self.parent.update_file_counts()

    def _make_thumbnail_sync(self, file_path: str) -> QPixmap:
        try:
            reader = QImageReader(file_path)
            reader.setAutoTransform(True)
            img = reader.read()
            if img.isNull():
                pm = QPixmap(THUMB_SIZE, THUMB_SIZE)
                pm.fill(Qt.transparent)
                return pm
            pm = QPixmap.fromImage(img)
            pm = pm.scaled(THUMB_SIZE, THUMB_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            canvas = QPixmap(THUMB_SIZE, THUMB_SIZE)
            canvas.fill(Qt.transparent)
            painter = QPainter(canvas)
            x = (THUMB_SIZE - pm.width()) // 2
            y = (THUMB_SIZE - pm.height()) // 2
            painter.drawPixmap(x, y, pm)
            painter.end()
            return canvas
        except Exception:
            pm = QPixmap(THUMB_SIZE, THUMB_SIZE)
            pm.fill(Qt.transparent)
            return pm

    def _set_thumb_for_item(self, index, pixmap, path):
        try:
            if index < 0 or index >= self.count():
                return
            it = self.item(index)
            if it is None:
                return
            if it.data(Qt.UserRole) != path:
                return
            w = self.itemWidget(it)
            if not w:
                return
            thumb_lbl = w.findChild(QLabel, "thumb_label")
            if thumb_lbl:
                thumb_lbl.setPixmap(pixmap)
                thumb_lbl.setVisible(True)
        except Exception:
            pass

    def _thumb_result_handler(self, payload):
        index, pixmap, path = payload
        self._pending_thumb_jobs.pop(path, None)
        QTimer.singleShot(0, lambda: self._set_thumb_for_item(index, pixmap, path))

    def _thumb_error_handler(self, payload):
        path, tb = payload
        print("Thumbnail worker error for", path, tb)

    def add_photo(self, file_path):
        if not file_path or not file_path.lower().endswith(IMAGE_EXTS):
            return
        for i in range(self.count()):
            if self.item(i).data(Qt.UserRole) == file_path:
                return
        it = QListWidgetItem()
        it.setData(Qt.UserRole, file_path)
        it.setToolTip(file_path)
        it.setFlags(it.flags() | Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsDragEnabled)
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(6, 3, 6, 3)
        lay.setSpacing(8)
        thumb = QLabel()
        thumb.setObjectName("thumb_label")
        thumb.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        thumb.setStyleSheet("border:1px solid rgba(0,0,0,0.06);")
        if self.thumbs_enabled:
            thumb.setVisible(True)
            idx = self.count()
            worker = ThumbnailWorker(idx, file_path, size=THUMB_SIZE)
            worker.signals.result.connect(self._thumb_result_handler)
            worker.signals.error.connect(self._thumb_error_handler)
            self._pending_thumb_jobs[file_path] = worker
            self.threadpool.start(worker)
        else:
            thumb.setVisible(False)
        name_lbl = QLabel(os.path.basename(file_path))
        name_lbl.setObjectName("name_label")
        lay.addWidget(thumb, 0)
        lay.addWidget(name_lbl, 1)
        w.setLayout(lay)
        it.setSizeHint(w.sizeHint())
        self.addItem(it)
        self.setItemWidget(it, w)
        if self.parent:
            self.parent.update_file_counts()

    def toggle_thumbnails(self, enabled: bool, with_busy: bool = True):
        self.thumbs_enabled = bool(enabled)
        dlg = None
        total = self.count()
        if with_busy and total > 40:
            dlg = QProgressDialog("Updating thumbnails...", None, 0, total, self.parent)
            dlg.setWindowTitle("Please wait")
            self.setWindowIcon(QIcon(resource_path("icon.png")))
            dlg.setWindowModality(Qt.ApplicationModal)
            dlg.setCancelButton(None)
            dlg.show()
            QApplication.processEvents()
        if self.thumbs_enabled:
            for i in range(self.count()):
                it = self.item(i)
                w = self.itemWidget(it)
                if not w:
                    continue
                thumb = w.findChild(QLabel, "thumb_label")
                path = it.data(Qt.UserRole)
                if thumb:
                    if thumb.pixmap() is None or thumb.pixmap().isNull():
                        worker = ThumbnailWorker(i, path, size=THUMB_SIZE)
                        worker.signals.result.connect(self._thumb_result_handler)
                        worker.signals.error.connect(self._thumb_error_handler)
                        self._pending_thumb_jobs[path] = worker
                        self.threadpool.start(worker)
                    thumb.setVisible(True)
                if dlg and i % 20 == 0:
                    QApplication.processEvents()
        else:
            for i in range(self.count()):
                it = self.item(i)
                w = self.itemWidget(it)
                if not w:
                    continue
                thumb = w.findChild(QLabel, "thumb_label")
                if thumb:
                    thumb.clear()
                    thumb.setVisible(False)
                if dlg and i % 20 == 0:
                    QApplication.processEvents()
        if dlg:
            dlg.close()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path.lower().endswith(IMAGE_EXTS):
                    event.acceptProposedAction()
                    return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        super().dragMoveEvent(event)

    def dropEvent(self, event):
        handled = False
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path.lower().endswith(IMAGE_EXTS):
                    self.add_photo(path)
                    handled = True
        if handled:
            event.acceptProposedAction()
        else:
            super().dropEvent(event)
        if handled and self.parent:
            self.parent.update_info()

    def open_menu(self, pos):
        items = self.selectedItems()
        if not items:
            return
        menu = QMenu()
        act_rename = menu.addAction("Rename")
        act_remove = menu.addAction("Remove from list")
        chosen = menu.exec_(self.mapToGlobal(pos))
        if chosen == act_rename:
            if len(items) == 1:
                old = items[0].data(Qt.UserRole)
                folder = os.path.dirname(old)
                base, ext = os.path.splitext(os.path.basename(old))
                text, ok = QInputDialog.getText(self, "Rename", "New name (without extension):", text=base)
                if not ok or not text.strip():
                    return
                new_path = os.path.join(folder, text.strip() + ext)
                try:
                    os.rename(old, new_path)
                    items[0].setData(Qt.UserRole, new_path)
                    self._name_label_of(items[0]).setText(os.path.basename(new_path))
                    thumb_lbl = self._thumb_label_of(items[0])
                    if thumb_lbl and self.thumbs_enabled:
                        worker = ThumbnailWorker(self.row(items[0]), new_path, size=THUMB_SIZE)
                        worker.signals.result.connect(self._thumb_result_handler)
                        worker.signals.error.connect(self._thumb_error_handler)
                        self.threadpool.start(worker)
                    if self.parent and self.parent.current_image_path == old:
                        self.parent.current_image_path = new_path
                        self.parent.file_name_edit.setText(os.path.basename(new_path))
                        self.parent.update_info()
                except Exception as e:
                    print("Rename error:", e)
            else:
                folder = os.path.dirname(items[0].data(Qt.UserRole))
                text, ok = QInputDialog.getText(self, "Batch rename", "Base name (extension preserved):", text="newname")
                if not ok or not text.strip():
                    return
                base = text.strip()
                for idx, it in enumerate(items, 1):
                    old = it.data(Qt.UserRole)
                    _, ext = os.path.splitext(old)
                    new_path = os.path.join(folder, f"{base}_{idx}{ext}")
                    try:
                        os.rename(old, new_path)
                        it.setData(Qt.UserRole, new_path)
                        self._name_label_of(it).setText(os.path.basename(new_path))
                        if self.thumbs_enabled:
                            worker = ThumbnailWorker(self.row(it), new_path, size=THUMB_SIZE)
                            worker.signals.result.connect(self._thumb_result_handler)
                            worker.signals.error.connect(self._thumb_error_handler)
                            self.threadpool.start(worker)
                    except Exception as e:
                        print("Batch rename error:", e)
                if self.parent:
                    self.parent.update_info()
        elif chosen == act_remove:
            rows = sorted([self.row(i) for i in items], reverse=True)
            removed = []
            for r in rows:
                it = self.item(r)
                removed.append(it.data(Qt.UserRole))
                self.takeItem(r)
            if self.parent and self.parent.current_image_path in removed:
                self.parent.current_image_path = None
                self.parent.image_viewer.clear_image()
                self.parent.file_name_edit.clear()
            if self.parent:
                self.parent.update_info()

    def _name_label_of(self, item):
        w = self.itemWidget(item)
        return w.findChild(QLabel, "name_label")

    def _thumb_label_of(self, item):
        w = self.itemWidget(item)
        return w.findChild(QLabel, "thumb_label")

# -------------------------
# Rect item widget (unchanged)
# -------------------------
class RectItemWidget(QWidget):
    def __init__(self, color: QColor, x=100, y=100, w=100, h=100, visible=True, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self._x = int(x)
        self._y = int(y)
        self._w = int(w)
        self._h = int(h)
        self._visible = bool(visible)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(6)
        self.color_lbl = QLabel()
        self.color_lbl.setFixedSize(26, 18)
        self.color_lbl.setCursor(Qt.PointingHandCursor)
        self._update_color_label()
        lay.addWidget(self.color_lbl)
        def _color_label_click(ev):
            if ev.button() == Qt.LeftButton:
                self._on_pick_color()
        self.color_lbl.mousePressEvent = _color_label_click
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)
        self.x_spin = QSpinBox(); self.x_spin.setRange(-10000, 10000); self.x_spin.setValue(self._x)
        self.y_spin = QSpinBox(); self.y_spin.setRange(-10000, 10000); self.y_spin.setValue(self._y)
        self.w_spin = QSpinBox(); self.w_spin.setRange(1, 10000); self.w_spin.setValue(self._w)
        self.h_spin = QSpinBox(); self.h_spin.setRange(1, 10000); self.h_spin.setValue(self._h)
        grid.addWidget(QLabel("x"), 0, 0); grid.addWidget(self.x_spin, 0, 1)
        grid.addWidget(QLabel("y"), 0, 2); grid.addWidget(self.y_spin, 0, 3)
        grid.addWidget(QLabel("w"), 1, 0); grid.addWidget(self.w_spin, 1, 1)
        grid.addWidget(QLabel("h"), 1, 2); grid.addWidget(self.h_spin, 1, 3)
        lay.addLayout(grid, 1)
        self.visible_cb = QCheckBox("Visible")
        self.visible_cb.setChecked(self._visible)
        lay.addWidget(self.visible_cb)
        self.del_btn = QPushButton("×")
        self.del_btn.setFixedWidth(30)
        self.del_btn.setToolTip("Delete")
        lay.addWidget(self.del_btn)
        self.x_spin.valueChanged.connect(self._on_change)
        self.y_spin.valueChanged.connect(self._on_change)
        self.w_spin.valueChanged.connect(self._on_change)
        self.h_spin.valueChanged.connect(self._on_change)
        self.visible_cb.stateChanged.connect(self._on_change)
        self.del_btn.clicked.connect(self._on_delete)
        self.on_change = None
        self.on_delete = None

    def _update_color_label(self):
        c = self._color
        self.color_lbl.setStyleSheet(f"background: {c.name()}; border:1px solid rgba(0,0,0,0.12);")

    def _on_change(self, *_):
        self._x = int(self.x_spin.value())
        self._y = int(self.y_spin.value())
        self._w = int(self.w_spin.value())
        self._h = int(self.h_spin.value())
        self._visible = bool(self.visible_cb.isChecked())
        if self.on_change:
            self.on_change(self.get_data())

    def _on_delete(self):
        if self.on_delete:
            self.on_delete()

    def _on_pick_color(self):
        c = QColorDialog.getColor(self._color, self, "Choose color")
        if c.isValid():
            self._color = c
            self._update_color_label()
            if self.on_change:
                self.on_change(self.get_data())

    def get_data(self):
        return {
            "color": self._color.name() if isinstance(self._color, QColor) else str(self._color),
            "x": int(self._x),
            "y": int(self._y),
            "w": int(self._w),
            "h": int(self._h),
            "visible": bool(self._visible)
        }

    def set_data(self, d):
        self._color = QColor(d.get("color", self._color))
        self._x = int(d.get("x", self._x))
        self._y = int(d.get("y", self._y))
        self._w = int(d.get("w", self._w))
        self._h = int(d.get("h", self._h))
        self._visible = bool(d.get("visible", self._visible))
        self.x_spin.blockSignals(True)
        self.y_spin.blockSignals(True)
        self.w_spin.blockSignals(True)
        self.h_spin.blockSignals(True)
        self.visible_cb.blockSignals(True)
        self.x_spin.setValue(self._x)
        self.y_spin.setValue(self._y)
        self.w_spin.setValue(self._w)
        self.h_spin.setValue(self._h)
        self.visible_cb.setChecked(self._visible)
        self.x_spin.blockSignals(False)
        self.y_spin.blockSignals(False)
        self.w_spin.blockSignals(False)
        self.h_spin.blockSignals(False)
        self.visible_cb.blockSignals(False)
        self._update_color_label()

# -------------------------
# MainWindow - English UI (with YOLO buttons & logic added)
# -------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Osteonix")
        self.setWindowIcon(QIcon(resource_path("icon.png")))
        self.resize(1400, 860)
        self.rects_per_image = {}
        central = QWidget()
        self.setCentralWidget(central)
        main_l = QVBoxLayout(central)

        # Toast notification widget
        self.toast = Toast(self)

        # Top: filename edit
        top_row = QHBoxLayout()
        self.file_name_edit = QLineEdit()
        self.file_name_edit.setPlaceholderText("Filename (rename and press Enter)")
        self.file_name_edit.editingFinished.connect(self.rename_file)
        top_row.addWidget(self.file_name_edit)
        main_l.addLayout(top_row)

        splitter = QSplitter(Qt.Horizontal)
        main_l.addWidget(splitter)

        # LEFT: controls + file list
        left_w = QWidget()
        left_l = QVBoxLayout(left_w)
        left_l.setContentsMargins(6, 6, 6, 6)
        btn_row = QHBoxLayout()
        #self.folder_btn = QPushButton("Select Folder")
        self.folder_btn = QPushButton("")
        self.folder_btn.setIcon(QIcon(resource_path('Assets/folder.png')))
        self.folder_btn.setToolTip('Select Folder')
        #self.file_btn = QPushButton("Add Files")
        self.file_btn = QPushButton("")
        self.file_btn.setIcon(QIcon(resource_path('Assets/file.png')))
        self.file_btn.setToolTip('Add Files')
        #self.find_btn = QPushButton("Find")
        self.find_btn = QPushButton("")
        self.find_btn.setIcon(QIcon(resource_path('Assets/find.png')))
        self.find_btn.setToolTip('Find (Ctrl+F)')
        #self.select_all_btn = QPushButton("Select All")
        self.select_all_btn = QPushButton("")
        self.select_all_btn.setIcon(QIcon(resource_path('Assets/select_all.png')))
        self.select_all_btn.setToolTip('Select All (Ctrl+A)')
        # --- yeni eklenen YOLO butonları ---
        #self.detect_current_btn = QPushButton("Detect (current)")
        self.detect_current_btn = QPushButton("")
        self.detect_current_btn.setIcon(QIcon(resource_path('Assets/detect_current.png')))
        self.detect_current_btn.setToolTip('Detect (current)')
        #self.detect_all_btn = QPushButton("Detect All")
        self.detect_all_btn = QPushButton("")
        self.detect_all_btn.setIcon(QIcon(resource_path('Assets/detect_all.png')))
        self.detect_all_btn.setToolTip('Detect All')
        btn_row.addWidget(self.folder_btn)
        btn_row.addWidget(self.file_btn)
        btn_row.addWidget(self.find_btn)
        btn_row.addWidget(self.select_all_btn)
        btn_row.addWidget(self.detect_current_btn)
        btn_row.addWidget(self.detect_all_btn)
        left_l.addLayout(btn_row)
        counts_row = QHBoxLayout()
        self.counts_label = QLabel("Selected 0 / Total 0")
        counts_row.addWidget(self.counts_label)
        counts_row.addStretch(1)
        left_l.addLayout(counts_row)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search filenames… (Esc to hide)")
        self.search_edit.hide()
        left_l.addWidget(self.search_edit)
        sort_row = QHBoxLayout()
        self.sort_btn = QPushButton("A→Z")
        self.sort_btn.setIcon(QIcon(resource_path('Assets/sort.png')))
        self.sort_btn.setToolTip('Sort by name')
        self.sort_count_btn = QPushButton("Mark")
        self.sort_count_btn.setIcon(QIcon(resource_path('Assets/sort.png')))
        self.sort_count_btn.setToolTip('Sort by rectangle count')
        self.toggle_thumbs_btn = QPushButton("")
        self.toggle_thumbs_btn.setIcon(QIcon(resource_path('Assets/thumbnails_on.png')))
        self.toggle_thumbs_btn.setToolTip('Toggle thumbnails')
        self.sort_count_state = None
        sort_row.addWidget(self.sort_btn)
        sort_row.addWidget(self.sort_count_btn)
        sort_row.addWidget(self.toggle_thumbs_btn)
        left_l.addLayout(sort_row)
        self.photo_list = PhotoList(self)
        self.photo_list.itemClicked.connect(self.on_photo_clicked)
        left_l.addWidget(self.photo_list)
        splitter.addWidget(left_w)

        # CENTER: image viewer
        self.image_viewer = ImageViewer(self)
        splitter.addWidget(self.image_viewer)

        # RIGHT: rect list and export / info
        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        right_l.setContentsMargins(6, 6, 6, 6)
        add_rect_row = QHBoxLayout()
        self.add_rect_btn = QPushButton("Add New Lesion")
        self.add_rect_btn.setIcon(QIcon(resource_path('Assets/add_rect.png')))
        self.add_rect_btn.setToolTip('Add New Lesion')
        add_rect_row.addWidget(self.add_rect_btn)
        right_l.addLayout(add_rect_row)

        self.rect_list = QListWidget()
        self.rect_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.rect_list.customContextMenuRequested.connect(self._rect_list_context)
        self.rect_list.setAlternatingRowColors(True)
        pal = self.rect_list.palette()
        pal.setColor(QPalette.Base, QColor("#ffffff"))
        pal.setColor(QPalette.AlternateBase, QColor("#ececec"))
        self.rect_list.setPalette(pal)
        right_l.addWidget(self.rect_list, 2)

        export_row = QHBoxLayout()
        self.export_current_btn = QPushButton("")
        self.export_current_btn.setIcon(QIcon(resource_path('Assets/export_current.png')))
        self.export_current_btn.setToolTip('Export current image')
        self.export_all_btn = QPushButton("")
        self.export_all_btn.setIcon(QIcon(resource_path('Assets/export_all.png')))
        self.export_all_btn.setToolTip('Export all images')
        export_row.addWidget(self.export_current_btn)
        export_row.addWidget(self.export_all_btn)
        right_l.addLayout(export_row)

        self.info_label = QLabel("Info box")
        self.info_label.setStyleSheet("border:1px solid gray; padding:8px;")
        self.info_label.setAlignment(Qt.AlignTop)
        right_l.addWidget(self.info_label, 1)

        splitter.addWidget(right_w)
        left_w.setMinimumWidth(300)
        self.image_viewer.setMinimumWidth(520)
        right_w.setMinimumWidth(360)
        splitter.setSizes([350, 700, 360])

        # Connect signals
        self.folder_btn.clicked.connect(self.select_folder)
        self.file_btn.clicked.connect(self.select_files)
        self.sort_btn.clicked.connect(self.sort_items)
        self.sort_count_btn.clicked.connect(self.sort_by_rect_count)
        self.toggle_thumbs_btn.clicked.connect(self._on_toggle_thumbs_clicked)
        self.sort_asc = True
        self.find_btn.clicked.connect(self._toggle_search)
        self.select_all_btn.clicked.connect(self.photo_list.selectAll)
        self.export_current_btn.clicked.connect(self.export_current_image)
        self.export_all_btn.clicked.connect(self.export_all_images)
        self.add_rect_btn.clicked.connect(self._on_add_rect_clicked)
        self.rect_list.currentRowChanged.connect(self._on_rect_list_selection_changed)
        self.find_shortcut = QShortcut(QKeySequence.Find, self)
        self.find_shortcut.activated.connect(self._toggle_search)
        self.select_shortcut = QShortcut(QKeySequence.SelectAll, self)
        self.select_shortcut.activated.connect(self.photo_list.selectAll)
        self.search_edit.textChanged.connect(self._apply_filter)
        def _search_keypress(e):
            if e.key() == Qt.Key_Escape:
                self._hide_search(clear=True)
            else:
                QLineEdit.keyPressEvent(self.search_edit, e)
        self.search_edit.keyPressEvent = _search_keypress
        self.current_image_path = None

        # YOLO button connections
        self.detect_current_btn.clicked.connect(self.on_detect_current_clicked)
        self.detect_all_btn.clicked.connect(self.on_detect_all_clicked)


        # Update counts timer initially
        QTimer.singleShot(50, self.update_file_counts)

    # -----------------------
    # File counts (unchanged)
    # -----------------------
    def update_file_counts(self):
        total = self.photo_list.count()
        selected = len(self.photo_list.selectedItems())
        self.counts_label.setText(f"Selected {selected} / Total {total}")

    # -----------------------
    # Photo list actions (unchanged)
    # -----------------------
    def on_photo_clicked(self, item):
        path = item.data(Qt.UserRole)
        self.show_image_for_path(path)

    def show_image_for_path(self, path):
        if not path or not os.path.exists(path):
            return
        self.current_image_path = path
        self.file_name_edit.setText(os.path.basename(path))
        self.image_viewer.load_image(path)
        self.rects_per_image.setdefault(self.current_image_path, [])
        self._load_rects_for_current_image()
        self.update_info()

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        filenames = [fn for fn in sorted(os.listdir(folder)) if fn.lower().endswith(IMAGE_EXTS)]
        dlg = None
        if len(filenames) > 80:
            dlg = QProgressDialog("Loading folder...", None, 0, 0, self)
            dlg.setWindowTitle("Please wait")
            dlg.setWindowModality(Qt.ApplicationModal)
            dlg.setCancelButton(None)
            dlg.show()
            QApplication.processEvents()
        migrated = self._migrate_rects_in_folder(folder)
        self.photo_list.clear()
        for i, fn in enumerate(filenames):
            full = os.path.join(folder, fn)
            self.photo_list.add_photo(full)
            if dlg and i % 20 == 0:
                QApplication.processEvents()
        if dlg:
            dlg.close()
        self.update_file_counts()
        self.update_info()
        if migrated:
            self.toast.show_toast("Migrated per-image rect files into global storage", 2600)

    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        if files:
            dlg = None
            if len(files) > 40:
                dlg = QProgressDialog("Adding files...", None, 0, 0, self)
                dlg.setWindowTitle("Please wait")
                dlg.setWindowModality(Qt.ApplicationModal)
                dlg.setCancelButton(None)
                dlg.show()
                QApplication.processEvents()
            for i, f in enumerate(files):
                self.photo_list.add_photo(f)
                if dlg and i % 20 == 0:
                    QApplication.processEvents()
            if dlg:
                dlg.close()
            self.update_file_counts()
            self.update_info()

    def rename_file(self):
        if not self.current_image_path:
            return
        new_name = self.file_name_edit.text().strip()
        if not new_name:
            return
        folder = os.path.dirname(self.current_image_path)
        _, old_ext = os.path.splitext(self.current_image_path)
        if os.path.splitext(new_name)[1] == "":
            new_name = new_name + old_ext
        new_path = os.path.join(folder, new_name)
        if new_path != self.current_image_path:
            try:
                os.rename(self.current_image_path, new_path)
                for i in range(self.photo_list.count()):
                    item = self.photo_list.item(i)
                    if item.data(Qt.UserRole) == self.current_image_path:
                        item.setData(Qt.UserRole, new_path)
                        self.photo_list._name_label_of(item).setText(os.path.basename(new_path))
                        thumb_lbl = self.photo_list._thumb_label_of(item)
                        if thumb_lbl and self.photo_list.thumbs_enabled:
                            worker = ThumbnailWorker(i, new_path, size=THUMB_SIZE)
                            worker.signals.result.connect(self.photo_list._thumb_result_handler)
                            worker.signals.error.connect(self.photo_list._thumb_error_handler)
                            self.photo_list.threadpool.start(worker)
                        break
                if self.current_image_path in self.rects_per_image:
                    self.rects_per_image[new_path] = self.rects_per_image.pop(self.current_image_path)
                    self._save_all_rects()
                self.current_image_path = new_path
                self.update_info()
                self.toast.show_toast("File renamed", 1800)
            except Exception as e:
                print("Rename failed:", e)

    # -----------------------
    # Sorting (unchanged)
    # -----------------------
    def sort_items(self):
        paths = [self.photo_list.item(i).data(Qt.UserRole) for i in range(self.photo_list.count())]
        paths = [p for p in paths if p]
        paths.sort(key=lambda p: os.path.basename(p).lower(), reverse=not self.sort_asc)
        self.sort_asc = not self.sort_asc
        self.sort_btn.setText("A→Z" if self.sort_asc else "Z→A")
        dlg = None
        if len(paths) > 80:
            dlg = QProgressDialog("Sorting...", None, 0, 0, self)
            dlg.setWindowTitle("Please wait")
            dlg.setWindowModality(Qt.ApplicationModal)
            dlg.setCancelButton(None)
            dlg.show()
            QApplication.processEvents()
        self.photo_list.clear()
        for i, p in enumerate(paths):
            self.photo_list.add_photo(p)
            if dlg and i % 20 == 0:
                QApplication.processEvents()
        if dlg:
            dlg.close()
        self.update_file_counts()
        self.update_info()
        self.toast.show_toast("Sorted by name", 1400)

    def sort_by_rect_count(self):
        if self.sort_count_state != 'desc':
            target = 'desc'
        else:
            target = 'asc'
        paths = [self.photo_list.item(i).data(Qt.UserRole) for i in range(self.photo_list.count())]
        annotated = []
        dlg = None
        if len(paths) > 60:
            dlg = QProgressDialog("Sorting by rect count...", None, 0, 0, self)
            dlg.setWindowTitle("Please wait")
            dlg.setWindowModality(Qt.ApplicationModal)
            dlg.setCancelButton(None)
            dlg.show()
            QApplication.processEvents()
        for i, p in enumerate(paths):
            count = 0
            if p in self.rects_per_image:
                count = len(self.rects_per_image.get(p, []))
            annotated.append((p, count))
            if dlg and i % 20 == 0:
                QApplication.processEvents()
        annotated.sort(key=lambda t: t[1], reverse=(target == 'desc'))
        self.photo_list.clear()
        for i, (p, _) in enumerate(annotated):
            self.photo_list.add_photo(p)
            if dlg and i % 20 == 0:
                QApplication.processEvents()
        self.sort_count_state = target
        if target == 'desc':
            self.sort_count_btn.setText("H→L")
        else:
            self.sort_count_btn.setText("L→H")
        if dlg:
            dlg.close()
        self.update_file_counts()
        self.update_info()
        self.toast.show_toast("Sorted by rectangle count", 1400)

    # -----------------------
    # Search control (unchanged)
    # -----------------------
    def _toggle_search(self):
        if self.search_edit.isVisible():
            self._hide_search(clear=False)
        else:
            self.search_edit.show()
            self.search_edit.setFocus()
            self.search_edit.selectAll()

    def _hide_search(self, clear=True):
        if clear:
            self.search_edit.clear()
            self._apply_filter("")
        self.search_edit.hide()
        self.photo_list.setFocus()

    def _apply_filter(self, text):
        q = (text or "").lower().strip()
        for i in range(self.photo_list.count()):
            it = self.photo_list.item(i)
            name = self.photo_list._name_label_of(it).text().lower()
            it.setHidden(False if q == "" else (q not in name))

    # -----------------------
    # Rects: add / load / view / persist (unchanged)
    # -----------------------
    def _on_add_rect_clicked(self):
        if not self.current_image_path:
            return
        default = {"color": "#e74c3c", "x": 100, "y": 100, "w": 100, "h": 100, "visible": True}
        lst = self.rects_per_image.setdefault(self.current_image_path, [])
        lst.append(default)
        self._load_rects_for_current_image()
        self.update_rects_on_view()
        self._save_all_rects()
        self.toast.show_toast("Rectangle added", 1200)

    def _load_rects_for_current_image(self):
        self.rect_list.clear()
        if not self.current_image_path:
            self.image_viewer.clear_rect_overlays()
            return
        items = self.rects_per_image.get(self.current_image_path, [])
        for idx, r in enumerate(items):
            li = QListWidgetItem()
            w = RectItemWidget(r["color"], r["x"], r["y"], r["w"], r["h"], r.get("visible", True))
            def make_on_change(i):
                def on_change(d):
                    if self.current_image_path not in self.rects_per_image:
                        return
                    try:
                        self.rects_per_image[self.current_image_path][i] = {
                            "color": d["color"], "x": d["x"], "y": d["y"], "w": d["w"], "h": d["h"], "visible": d["visible"]
                        }
                        self.update_rects_on_view()
                        self._save_all_rects()
                        if self.rect_list.currentRow() == i:
                            self.update_info()
                    except Exception:
                        pass
                return on_change
            def make_on_delete(i):
                def on_delete():
                    try:
                        del self.rects_per_image[self.current_image_path][i]
                    except Exception:
                        pass
                    self._load_rects_for_current_image()
                    self.update_rects_on_view()
                    self._save_all_rects()
                    if self.rect_list.count() == 0:
                        self._clear_info_label()
                return on_delete
            w.on_change = make_on_change(idx)
            w.on_delete = make_on_delete(idx)
            li.setSizeHint(w.sizeHint())
            self.rect_list.addItem(li)
            self.rect_list.setItemWidget(li, w)
        self.update_info()
        self.update_rects_on_view()

    def _rect_list_context(self, pos):
        it = self.rect_list.itemAt(pos)
        if not it:
            return
        menu = QMenu()
        act_del = menu.addAction("Delete")
        chosen = menu.exec_(self.rect_list.mapToGlobal(pos))
        if chosen == act_del:
            row = self.rect_list.row(it)
            try:
                del self.rects_per_image[self.current_image_path][row]
            except Exception:
                pass
            self._load_rects_for_current_image()
            self.update_rects_on_view()
            self._save_all_rects()
            if self.rect_list.count() == 0:
                self._clear_info_label()
            self.toast.show_toast("Rectangle deleted", 1200)

    def update_rects_on_view(self):
        self.image_viewer.clear_rect_overlays()
        if not self.current_image_path or not self.image_viewer._pixmap_item:
            return
        lst = self.rects_per_image.get(self.current_image_path, [])
        for idx, r in enumerate(lst):
            if not r.get("visible", True):
                continue  # görünmeyen dikdörtgenleri hiç ekleme
            item = self.image_viewer.add_rect_overlay(
                (r["x"], r["y"], r["w"], r["h"]), r["color"], idx
            )
            if self.rect_list.currentRow() == idx:
                item.setSelected(True)
            item.update_color(r.get("color", "#e74c3c"))
        self.update_info()
        self._sync_rect_widgets_from_data()

    # -----------------------
    # Export (unchanged)
    # -----------------------
    def export_current_image(self):
        if not self.current_image_path or not self.image_viewer._pixmap_item:
            return
        self.info_label.setText("Export: exporting current image...")
        QApplication.processEvents()
        original = self.image_viewer._pixmap_item.pixmap()
        export_pixmap = original.copy()
        painter = QPainter(export_pixmap)
        rects = self.rects_per_image.get(self.current_image_path, [])
        for rect in rects:
            if rect.get("visible", True):
                color = QColor(rect["color"])
                pen = QPen(color)
                pen.setWidth(3)
                painter.setPen(pen)
                painter.drawRect(rect["x"], rect["y"], rect["w"], rect["h"])
        painter.end()
        original_path = self.current_image_path
        name, ext = os.path.splitext(original_path)
        export_path = f"{name}_with_rects{ext}"
        export_pixmap.save(export_path)
        self.info_label.setText(f"Export completed: {os.path.basename(export_path)}")
        QApplication.processEvents()
        self.toast.show_toast("Export completed", 1600)

    def export_all_images(self):
        if self.photo_list.count() == 0:
            return
        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not export_dir:
            return
        export_dir = os.path.join(export_dir, "exported_images")
        os.makedirs(export_dir, exist_ok=True)
        total = self.photo_list.count()
        progress = QProgressDialog("Exporting all images...", "Cancel", 0, total, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(200)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        completed = 0
        for i in range(self.photo_list.count()):
            item = self.photo_list.item(i)
            image_path = item.data(Qt.UserRole)
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                completed += 1
                progress.setValue(completed)
                perc = int((completed / total) * 100)
                self.info_label.setText(f"Export: {completed}/{total} ({perc}%)")
                QApplication.processEvents()
                continue
            rects = self.rects_per_image.get(image_path, [])
            export_pixmap = pixmap.copy()
            painter = QPainter(export_pixmap)
            for rect in rects:
                if rect.get("visible", True):
                    color = QColor(rect["color"])
                    pen = QPen(color)
                    pen.setWidth(3)
                    painter.setPen(pen)
                    painter.drawRect(rect["x"], rect["y"], rect["w"], rect["h"])
            painter.end()
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            export_path = os.path.join(export_dir, f"{name}_with_rects{ext}")
            export_pixmap.save(export_path)
            completed += 1
            progress.setValue(completed)
            perc = int((completed / total) * 100)
            self.info_label.setText(f"Export: {completed}/{total} ({perc}%)")
            QApplication.processEvents()
            if progress.wasCanceled():
                self.info_label.setText(f"Export cancelled: {completed}/{total} ({perc}%)")
                progress.close()
                self.toast.show_toast("Export cancelled", 1600)
                return
        progress.close()
        self.info_label.setText(f"All images exported: {export_dir}")
        QApplication.processEvents()
        self.toast.show_toast("All images exported", 1800)

    # -----------------------
    # Overlay callbacks & selections (unchanged)
    # -----------------------
    def _on_overlay_moved(self, idx, new_x, new_y, w, h):
        if not self.current_image_path:
            return
        try:
            lst = self.rects_per_image.setdefault(self.current_image_path, [])
            if idx < 0 or idx >= len(lst):
                return
            lst[idx]["x"] = int(new_x)
            lst[idx]["y"] = int(new_y)
            lst[idx]["w"] = int(w)
            lst[idx]["h"] = int(h)
            self._sync_rect_widgets_from_data()
            self._save_all_rects()
            self.update_info()
        except Exception as e:
            print("overlay move error:", e)

    def _on_overlay_selected(self, idx):
        if idx is None:
            return
        if 0 <= idx < self.rect_list.count():
            self.rect_list.blockSignals(True)
            self.rect_list.setCurrentRow(idx)
            self.rect_list.blockSignals(False)
            self.update_info()

    def _on_rect_list_selection_changed(self, row):
        for overlay in self.image_viewer.rect_overlays:
            overlay.setSelected(False)
        if row >= 0 and row < len(self.image_viewer.rect_overlays):
            try:
                self.image_viewer.rect_overlays[row].setSelected(True)
            except Exception:
                pass
        self.update_info()

    def _sync_rect_widgets_from_data(self):
        if not self.current_image_path:
            return
        items = self.rects_per_image.get(self.current_image_path, [])
        for i in range(self.rect_list.count()):
            li = self.rect_list.item(i)
            w = self.rect_list.itemWidget(li)
            if i < len(items) and w:
                w.set_data(items[i])

    # -----------------------
    # Info / UI updates (unchanged)
    # -----------------------
    def update_info(self):
        info_lines = []
        pv = self.image_viewer
        if self.current_image_path and pv._pixmap_item:
            pixmap = pv._pixmap_item.pixmap()
            info_lines.append(f"Image size: {pixmap.width()}x{pixmap.height()} px")
            vp = pv.viewport().rect()
            tl = pv.mapToScene(vp.topLeft())
            br = pv.mapToScene(vp.bottomRight())
            scene_rect = QRectF(tl, br)
            pix_rect = pv._pixmap_item.boundingRect()
            visible = scene_rect.intersected(pix_rect)
            if visible.isEmpty():
                percent_visible = 0.0
            else:
                percent_visible = (visible.width() * visible.height()) / (pixmap.width() * pixmap.height()) * 100.0
            info_lines.append(f"Visible area: {percent_visible:.1f}%")
            info_lines.append(f"Zoom: x{pv._current_scale:.2f}")
            vpos = pv.mapFromGlobal(QCursor.pos())
            if pv.viewport().rect().contains(vpos):
                scene_pos = pv.mapToScene(vpos)
                if pv._pixmap_item and pv._pixmap_item.boundingRect().contains(scene_pos):
                    x = int(scene_pos.x()); y = int(scene_pos.y())
                    info_lines.append(f"Mouse: ({x}, {y})")
                else:
                    info_lines.append("Mouse: (outside image)")
        else:
            info_lines.append("Image: none")

        sel_row = self.rect_list.currentRow()
        if sel_row is not None and sel_row >= 0:
            items = self.rects_per_image.get(self.current_image_path, [])
            if 0 <= sel_row < len(items):
                r = items[sel_row]
                try:
                    c = QColor(r.get("color", "#e74c3c"))
                    alpha = 130 if not r.get("visible", True) else 255
                    rgba = f"rgba({c.red()},{c.green()},{c.blue()},{alpha})"
                    txt = "<br>".join([
                        f"<span>{info_lines[0]}</span>" if info_lines else "<span>Image: none</span>",
                        f"<span>Visible area: {percent_visible:.1f}%</span>",
                        f"<span>Zoom: x{pv._current_scale:.2f}</span>",
                        "<hr>",
                        "<span>Selected rectangle:</span>",
                        f"<span style='color:{rgba};'> x: {r['x']} &nbsp; y: {r['y']} </span>",
                        f"<span style='color:{rgba};'> w: {r['w']} &nbsp; h: {r['h']} </span>",
                        (f"<span style='color:{rgba};'>Type: {r.get('label')}</span>" if r.get("label") else ""),
                        (f"<span style='color:{rgba};'>Confidence: {r.get('conf')*100:.1f}%</span>" if r.get("conf") is not None else "")
                    ])
                    self.info_label.setText(txt)
                    self.info_label.setStyleSheet("border:1px solid gray; padding:8px;")
                    return
                except Exception:
                    pass
        self.info_label.setText("\n".join(info_lines))
        self.info_label.setStyleSheet("border:1px solid gray; padding:8px;")

    def _clear_info_label(self):
        self.info_label.setText("Info box")
        self.info_label.setStyleSheet("border:1px solid gray; padding:8px;")

    # -----------------------
    # Global JSON persistence (unchanged)
    # -----------------------
    def _save_all_rects(self):
        pass

    def _load_all_rects_if_any(self):
        self.rects_per_image = {}

    def _migrate_rects_in_folder(self, folder):
        return False

    def _on_toggle_thumbs_clicked(self):
        new_state = not self.photo_list.thumbs_enabled
        with_busy = True if self.photo_list.count() > 60 else False
        self.photo_list.toggle_thumbnails(new_state, with_busy=with_busy)
        self.toggle_thumbs_btn.setIcon(QIcon(resource_path("Assets/thumbnails_off.png" if self.photo_list.thumbs_enabled else "Assets/thumbnails_on.png")))
        self.info_label.setText("Thumbnails enabled" if self.photo_list.thumbs_enabled else "Thumbnails disabled")
        self.toast.show_toast("Thumbnails turned " + ("on" if self.photo_list.thumbs_enabled else "off"), 1600)
        QApplication.processEvents()

    # -----------------------
    # YOLO integration helpers (yeni)
    # -----------------------
    def _attempt_load_yolo_model(self, model_path):
        """
        Try to load YOLO model using ultralytics.YOLO (yolov11 supported).
        Returns a callable `predict_func(image_path)` that returns list of boxes [(x,y,w,h,label,conf), ...]
        or raises RuntimeError if loading failed.
        """
        model_path = model_path or "best.pt"
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model file not found: {model_path}")

        # Try ultralytics first
        if _ultralytics_available:
            try:
                ymodel = YOLO(model_path)
                def predict_ultralytics(img_path):
                    results = ymodel.predict(source=img_path, imgsz=1280, device='cpu', verbose=False)  # device default CPU
                    boxes_out = []
                    for res in results:
                        # res.boxes contains xyxy, conf, cls
                        b = getattr(res, "boxes", None)
                        if b is None:
                            continue
                        xyxy = b.xyxy.cpu().numpy() if hasattr(b, "xyxy") else []
                        confs = b.conf.cpu().numpy() if hasattr(b, "conf") else []
                        cls_ids = b.cls.cpu().numpy() if hasattr(b, "cls") else []
                        for i in range(len(xyxy)):
                            x1, y1, x2, y2 = xyxy[i]
                            w = int(round(x2 - x1))
                            h = int(round(y2 - y1))
                            x = int(round(x1))
                            y = int(round(y1))
                            conf = float(confs[i]) if len(confs)>i else 0.0
                            cls_id = int(cls_ids[i]) if len(cls_ids)>i else None
                            label = str(cls_id) if cls_id is not None else ""
                            boxes_out.append((x, y, w, h, label, conf))
                    return boxes_out
                return predict_ultralytics
            except Exception as e:
                # fallback below
                print("Ultralytics load failed:", e)

        # Only ultralytics YOLO (supports yolov8-yolov11)
        raise RuntimeError("Could not load YOLO model. Install 'ultralytics' and ensure model path is correct.")("Could not load YOLO model. Install 'ultralytics' (preferred) or torch + internet for torch.hub.")

    def _run_detection_on_image(self, image_path, model_path="best.pt", min_conf=0.25):
        """
        Run detection on a single image. Returns list of (x,y,w,h,label,conf).
        Raises RuntimeError on failure to load model.
        """
        try:
            predict_fn = self._attempt_load_yolo_model(model_path)
        except Exception as e:
            raise RuntimeError(str(e))
        boxes = []
        try:
            boxes = predict_fn(image_path)
            # filter by conf threshold
            boxes = [b for b in boxes if b[5] is None or b[5] >= min_conf]
            return boxes
        except Exception as e:
            raise RuntimeError(f"Detection failed: {e}")

    # -----------------------
    # YOLO button handlers (yeni)
    # -----------------------
    def on_detect_current_clicked(self):
        """
        1) Remove all rectangles on current image
        2) Run detection on current image using best.pt
        3) Add rectangles from detections
        4) Update info & toast
        """
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            self.toast.show_toast("No image open to detect", 2000)
            return

        model_path = os.path.join(os.getcwd(), "best.pt")  # kullanıcının best.pt'i çalışma dizininde beklenir
        self.info_label.setText("Running detection on current image...")
        QApplication.processEvents()

        # remove existing rects for current image
        self.rects_per_image[self.current_image_path] = []
        self._load_rects_for_current_image()
        self.update_rects_on_view()
        self._save_all_rects()

        try:
            boxes = self._run_detection_on_image(self.current_image_path, model_path=model_path)
        except RuntimeError as e:
            err = str(e)
            self.info_label.setText("Detection error: " + err)
            self.toast.show_toast("Detection failed — check model / deps", 2800)
            return

        # convert boxes -> rect entries and add
        lst = self.rects_per_image.setdefault(self.current_image_path, [])
        for b in boxes:
            x, y, w, h, label, conf = b
            # pick color by label hash to get variety
            try:
                c = QColor("#" + ("%06x" % (abs(hash(label)) & 0xFFFFFF)))
            except Exception:
                c = QColor("#e74c3c")
            lst.append({
                "color": c.name(),
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "visible": True,
                "label": CLASS_MAP.get(int(label), str(label)) if str(label).isdigit() else str(label),
                "conf": conf,
                "label": CLASS_MAP.get(int(label), str(label)) if str(label).isdigit() else str(label),
                "conf": conf
            })
        self._load_rects_for_current_image()
        self.update_rects_on_view()
        self._save_all_rects()
        self.info_label.setText(f"Detection completed: {len(boxes)} objects on current image")
        QApplication.processEvents()
        self.toast.show_toast(f"Detected {len(boxes)} objects", 2000)

    def on_detect_all_clicked(self):
        """
        Iterate through items in photo_list, run detection with model, add rects.
        Show progress dialog (loading animation/ progress).
        """
        total = self.photo_list.count()
        if total == 0:
            self.toast.show_toast("No files to detect", 1800)
            return
        model_path = os.path.join(os.getcwd(), "best.pt")
        progress = QProgressDialog("Detecting objects in files (this may take a while)...", "Cancel", 0, total, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(200)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        # Try load model once to fail fast
        try:
            predict_fn = self._attempt_load_yolo_model(model_path)
        except Exception as e:
            self.info_label.setText("Detection error: " + str(e))
            self.toast.show_toast("Detection failed — model or deps missing", 2800)
            progress.close()
            return

        completed = 0
        total_found = 0
        for i in range(self.photo_list.count()):
            if progress.wasCanceled():
                break
            item = self.photo_list.item(i)
            image_path = item.data(Qt.UserRole)
            self.info_label.setText(f"Detecting: {os.path.basename(image_path)} ({i+1}/{total})")
            QApplication.processEvents()

            # clear existing rects for this image
            self.rects_per_image[image_path] = []
            try:
                boxes = predict_fn(image_path)
            except Exception as e:
                boxes = []
                print("Detection failed for", image_path, e)

            lst = self.rects_per_image.setdefault(image_path, [])
            for b in boxes:
                x, y, w, h, label, conf = b
                try:
                    c = QColor("#" + ("%06x" % (abs(hash(label)) & 0xFFFFFF)))
                except Exception:
                    c = QColor("#e74c3c")
                lst.append({
                    "color": c.name(),
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "visible": True,
                "label": CLASS_MAP.get(int(label), str(label)) if str(label).isdigit() else str(label),
                "conf": conf
                })
            total_found += len(boxes)
            completed += 1
            progress.setValue(completed)
            perc = int((completed / total) * 100)
            self.info_label.setText(f"Detecting: {completed}/{total} ({perc}%) — total found {total_found}")
            QApplication.processEvents()
            # if the currently open image is this one, reload overlays
            if image_path == self.current_image_path:
                self._load_rects_for_current_image()
                self.update_rects_on_view()
            # periodically save progress to avoid data loss
            if completed % 5 == 0:
                self._save_all_rects()
        progress.close()
        # final save
        self._save_all_rects()
        self.info_label.setText(f"Detection finished: processed {completed}/{total}, found {total_found} objects")
        QApplication.processEvents()
        self.toast.show_toast(f"Detect all finished: {total_found} objects", 2200)

# -------------------------
# Start application
# -------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont()
    font.setPointSize(GLOBAL_FONT_POINT)
    app.setFont(font)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())