"""The Wispr-style pill: a small dark rounded overlay at the bottom centre
of the screen. Shows a live waveform while recording and pulsing dots
while transcribing. It never takes focus, so the paste always lands in the
app you were typing in.
"""
import math

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFontMetrics, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QWidget

ACCENT = QColor(62, 165, 175)


class Pill(QWidget):
    WIDTH = 240
    LIVE_WIDTH = 420  # wider while showing live transcription text
    HEIGHT = 44

    def __init__(self, recorder):
        flags = (
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowDoesNotAcceptFocus
        )
        super().__init__(None, flags)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.recorder = recorder
        self.state = "idle"
        self._phase = 0
        self._live_text = ""
        self.resize(self.WIDTH, self.HEIGHT)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def _position(self):
        screen = QApplication.primaryScreen().availableGeometry()
        x = screen.center().x() - self.width() // 2
        y = screen.bottom() - self.HEIGHT - 20
        self.move(x, y)

    def show_recording(self):
        self.state = "recording"
        self._live_text = ""
        self._position()
        self.show()
        self._timer.start(33)

    def show_processing(self):
        self.state = "processing"

    def set_live_text(self, text: str):
        """Streaming ASR draft. Arrives via a queued signal, so always on
        the main thread. Ignored outside the recording state so a late
        draft can never resurrect a dismissed pill."""
        if self.state != "recording":
            return
        self._live_text = (text or "").strip()
        if self._live_text and self.width() != self.LIVE_WIDTH:
            self.resize(self.LIVE_WIDTH, self.HEIGHT)
            self._position()

    def dismiss(self):
        self._timer.stop()
        self.state = "idle"
        self._live_text = ""
        self.hide()
        self.resize(self.WIDTH, self.HEIGHT)

    def _tick(self):
        self._phase += 1
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(18, 18, 20, 235))
        painter.drawRoundedRect(
            QRectF(0, 0, self.width(), self.HEIGHT), self.HEIGHT / 2, self.HEIGHT / 2
        )
        if self.state == "recording":
            if self._live_text:
                self._draw_live_text(painter)
            else:
                self._draw_waveform(painter)
        elif self.state == "processing":
            self._draw_dots(painter)
        painter.end()

    def _draw_live_text(self, painter):
        painter.setBrush(ACCENT)
        painter.drawEllipse(QRectF(16, self.HEIGHT / 2 - 3, 6, 6))
        font = painter.font()
        font.setPixelSize(13)
        painter.setFont(font)
        painter.setPen(QColor(245, 245, 245, 235))
        available = self.width() - 34 - 18
        # Elide on the left so the most recent words stay visible.
        text = QFontMetrics(font).elidedText(
            self._live_text, Qt.ElideLeft, available
        )
        painter.drawText(
            QRectF(34, 0, available, self.HEIGHT),
            Qt.AlignVCenter | Qt.AlignLeft,
            text,
        )
        painter.setPen(Qt.NoPen)

    def _draw_waveform(self, painter):
        bars = 26
        bar_width = 3.0
        gap = 4.0
        total = bars * bar_width + (bars - 1) * gap
        x = (self.width() - total) / 2
        levels = list(self.recorder.levels)[-bars:]
        levels = [0.0] * (bars - len(levels)) + levels
        painter.setBrush(QColor(245, 245, 245, 235))
        for level in levels:
            height = 4 + min(1.0, level * 14.0) * (self.HEIGHT * 0.55)
            rect = QRectF(x, (self.HEIGHT - height) / 2, bar_width, height)
            painter.drawRoundedRect(rect, 1.5, 1.5)
            x += bar_width + gap

    def _draw_dots(self, painter):
        painter.setBrush(ACCENT)
        cx = self.width() / 2
        for i in (-1, 0, 1):
            pulse = 0.5 + 0.5 * math.sin(self._phase / 4.0 + i)
            radius = 3 + 2.5 * pulse
            painter.drawEllipse(
                QRectF(cx + i * 16 - radius, self.HEIGHT / 2 - radius, radius * 2, radius * 2)
            )


def tray_icon() -> QIcon:
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(ACCENT)
    painter.drawEllipse(4, 4, 24, 24)
    painter.setBrush(QColor(255, 255, 255))
    painter.drawRoundedRect(QRectF(13, 9, 6, 12), 3, 3)
    painter.drawRoundedRect(QRectF(14.5, 22, 3, 4), 1.5, 1.5)
    painter.end()
    return QIcon(pixmap)
