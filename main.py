"""
Legenda ao vivo com tradução (EN -> PT)
----------------------------------------
Captura o áudio de saída do PC (loopback), transcreve com faster-whisper
e traduz com deep-translator, exibindo tudo em um overlay flutuante.

Principais correções em relação à versão anterior:
  1. Detecção de silêncio (VAD simples por energia) para COMMITAR e
     RESETAR o buffer de áudio quando a fala para. Isso elimina o
     problema de retranscrever e retraduzir repetidamente o mesmo
     trecho de áudio a cada ciclo do buffer deslizante.
  2. Tradução roda em uma thread própria, consumindo uma fila. Assim,
     uma tradução lenta (rede) nunca trava a captura/transcrição de
     áudio, que é o que realmente precisa rodar em tempo real.
  3. Só envia para tradução o texto NOVO desde o último ciclo (diff
     incremental), reduzindo chamadas repetidas ao GoogleTranslator.
"""

import sys
import logging
import warnings
from queue import Queue, Empty
from dataclasses import dataclass

import numpy as np
import soundcard as sc
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QKeyEvent

warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("subtitle_overlay")


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    step_seconds: float = 1.0          # tamanho de cada chunk capturado
    max_buffer_seconds: float = 8.0    # teto do buffer (fala longa sem pausa)
    min_buffer_seconds: float = 1.0    # não transcreve com menos áudio que isso
    silence_threshold: float = 0.015   # amplitude abaixo disso = silêncio
    silence_hangover_steps: int = 2    # nº de chunks silenciosos p/ commitar

    # Modelo Whisper. Opções (da mais leve pra mais "inteligente"):
    #   "tiny.en"          -> muito rápido, qualidade baixa
    #   "small.en"         -> leve, qualidade razoável
    #   "distil-medium.en" -> qualidade próxima de medium.en, ~6x mais rápido
    #   "medium.en"        -> ótima qualidade, mais pesado em CPU
    #   "distil-large-v3"  -> qualidade quase de large, ainda rápido pra CPU forte
    model_size: str = "distil-large-v3"
    cpu_threads: int = 8       # threads usadas pelo faster-whisper por transcrição
    num_workers: int = 1       # transcrições paralelas (deixe 1 pra esse uso)
    beam_size: int = 5         # antes: 1 — busca mais hipóteses = mais precisão
    best_of: int = 5           # antes: 1

    @property
    def step_size(self) -> int:
        return int(self.sample_rate * self.step_seconds)

    @property
    def max_buffer_size(self) -> int:
        return int(self.sample_rate * self.max_buffer_seconds)

    @property
    def min_buffer_size(self) -> int:
        return int(self.sample_rate * self.min_buffer_seconds)


class TranslatorWorker(QThread):
    """Traduz textos em uma thread separada, consumindo uma fila.

    Isso garante que uma tradução lenta (chamada de rede) nunca
    bloqueie a captura/transcrição de áudio, que roda no AudioWorker.
    """

    translation_ready = Signal(str, str)  # (texto_en, texto_pt)

    def __init__(self, source: str = "en", target: str = "pt"):
        super().__init__()
        self._translator = GoogleTranslator(source=source, target=target)
        self._queue: Queue[str | None] = Queue()
        self._running = True

    def enqueue(self, text: str) -> None:
        self._queue.put(text)

    def run(self) -> None:
        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
            except Empty:
                continue

            if text is None:  # sentinela de parada
                break

            try:
                translated = self._translator.translate(text)
            except Exception as exc:
                log.warning("Falha ao traduzir %r: %s", text, exc)
                translated = "..."

            self.translation_ready.emit(text, translated)

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)
        self.wait()


class AudioWorker(QThread):
    """Captura o áudio do sistema, transcreve e detecta o fim de cada frase.

    Emite:
      - text_updated: texto em inglês da frase atual (chamado a cada ciclo,
        para feedback visual imediato, sem esperar tradução).
      - utterance_finished: quando detecta silêncio suficiente após fala,
        sinaliza que a frase terminou (a UI pode, por exemplo, decidir
        limpar o texto na próxima fala nova).
      - new_fragment: pedaço de texto NOVO desde o último ciclo, que deve
        ser enviado para tradução (evita retraduzir o texto inteiro).
    """

    text_updated = Signal(str)
    new_fragment = Signal(str)
    utterance_finished = Signal()

    def __init__(self, config: AudioConfig | None = None):
        super().__init__()
        self.config = config or AudioConfig()
        self._running = True

        log.info("Carregando o modelo Whisper (%s)...", self.config.model_size)
        self.model = WhisperModel(
            self.config.model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=self.config.cpu_threads,
            num_workers=self.config.num_workers,
        )
        log.info("Modelo pronto!")

    def _extract_new_fragment(self, current_text: str, previous_text: str) -> str:
        """Retorna só a parte nova de current_text em relação a previous_text.

        Se o Whisper reescreveu o início da frase (o que acontece às
        vezes entre um ciclo e outro), não dá pra confiar só no
        startswith — nesse caso, tratamos o texto inteiro como novo
        para não perder conteúdo, mesmo que isso gere uma tradução
        redundante ocasional (raro, e bem menos frequente que antes).
        """
        if not previous_text:
            return current_text
        if current_text.startswith(previous_text):
            return current_text[len(previous_text):].strip()
        return current_text

    def run(self) -> None:
        cfg = self.config
        default_speaker = sc.default_speaker()
        loopback_mic = sc.get_microphone(
            id=str(default_speaker.name), include_loopback=True
        )

        audio_buffer = np.zeros(0, dtype=np.float32)
        last_text = ""
        silence_steps = 0

        with loopback_mic.recorder(samplerate=cfg.sample_rate) as recorder:
            while self._running:
                chunk = recorder.record(numframes=cfg.step_size)
                if chunk.ndim > 1:
                    chunk = np.mean(chunk, axis=1)

                is_silent_chunk = np.max(np.abs(chunk)) < cfg.silence_threshold
                silence_steps = silence_steps + 1 if is_silent_chunk else 0

                audio_buffer = np.append(audio_buffer, chunk)
                if len(audio_buffer) > cfg.max_buffer_size:
                    audio_buffer = audio_buffer[-cfg.max_buffer_size:]

                buffer_has_signal = (
                    np.max(np.abs(audio_buffer)) >= cfg.silence_threshold
                )
                buffer_long_enough = len(audio_buffer) >= cfg.min_buffer_size

                if buffer_has_signal and buffer_long_enough:
                    segments, _ = self.model.transcribe(
                        audio_buffer,
                        language="en",
                        task="transcribe",
                        beam_size=cfg.beam_size,
                        best_of=cfg.best_of,
                        condition_on_previous_text=False,
                        vad_filter=True,
                    )
                    current_text = "".join(s.text for s in segments).strip()

                    if current_text and current_text != last_text:
                        self.text_updated.emit(current_text)

                        fragment = self._extract_new_fragment(current_text, last_text)
                        if fragment:
                            self.new_fragment.emit(fragment)

                        last_text = current_text

                # Silêncio suficiente após ter havido fala -> encerra a
                # frase atual e reinicia o buffer do zero. É isso que
                # evita reprocessar o mesmo áudio indefinidamente.
                if silence_steps >= cfg.silence_hangover_steps and last_text:
                    self.utterance_finished.emit()
                    audio_buffer = np.zeros(0, dtype=np.float32)
                    last_text = ""
                    silence_steps = 0

    def stop(self) -> None:
        self._running = False
        self.wait()


class OverlayWindow(QWidget):
    def __init__(self, audio_worker: AudioWorker, translator_worker: TranslatorWorker):
        super().__init__()
        self.audio_worker = audio_worker
        self.translator_worker = translator_worker
        self._pt_accumulated = ""
        self._drag_pos = None
        self._init_ui()

    def _init_ui(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        layout = QVBoxLayout()

        self.lbl_english = QLabel("Aguardando áudio...")
        self.lbl_english.setStyleSheet(
            """
            QLabel {
                color: #FFFFFF;
                font-size: 20px;
                font-weight: bold;
                background-color: rgba(0, 0, 0, 180);
                border-radius: 8px;
                padding: 10px;
            }
            """
        )
        self.lbl_english.setWordWrap(True)

        self.lbl_portuguese = QLabel(
            "Legenda em português aparecerá aqui. (Pressione ESC para fechar)"
        )
        self.lbl_portuguese.setStyleSheet(
            """
            QLabel {
                color: #FFD700;
                font-size: 16px;
                background-color: rgba(0, 0, 0, 180);
                border-radius: 8px;
                padding: 8px;
            }
            """
        )
        self.lbl_portuguese.setWordWrap(True)

        layout.addWidget(self.lbl_english)
        layout.addWidget(self.lbl_portuguese)
        self.setLayout(layout)

        self.resize(700, 140)
        screen_geometry = QApplication.primaryScreen().geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = screen_geometry.height() - self.height() - 100
        self.move(x, y)

    # --- Slots conectados aos workers -------------------------------

    def on_text_updated(self, en_text: str) -> None:
        self.lbl_english.setText(en_text)

    def on_translation_ready(self, _en_fragment: str, pt_fragment: str) -> None:
        if self._pt_accumulated:
            self._pt_accumulated += " " + pt_fragment
        else:
            self._pt_accumulated = pt_fragment
        self.lbl_portuguese.setText(self._pt_accumulated)

    def on_utterance_finished(self) -> None:
        # A frase terminou (silêncio detectado). Deixa o texto final
        # visível; ele será limpo assim que a próxima frase começar.
        self._pt_accumulated = ""

    # --- Arrastar a janela -------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    # --- Atalho ESC para fechar tudo -----------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._shutdown()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self._shutdown()
        event.accept()

    def _shutdown(self) -> None:
        log.info("Encerrando...")
        self.audio_worker.stop()
        self.translator_worker.stop()
        QApplication.quit()


def main() -> int:
    app = QApplication(sys.argv)

    translator_worker = TranslatorWorker(source="en", target="pt")
    audio_worker = AudioWorker()
    overlay = OverlayWindow(audio_worker, translator_worker)

    audio_worker.text_updated.connect(overlay.on_text_updated)
    audio_worker.new_fragment.connect(translator_worker.enqueue)
    audio_worker.utterance_finished.connect(overlay.on_utterance_finished)
    translator_worker.translation_ready.connect(overlay.on_translation_ready)

    overlay.show()
    translator_worker.start()
    audio_worker.start()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())