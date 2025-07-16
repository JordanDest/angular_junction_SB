# log_setup.py ─────────────────────────────────────────────────────────
import logging, time, threading
try:
    import feedback.texting           # your SMS helper
except ImportError:
    texting = None

PHONE = "3366023500"
DUP_WINDOW = 900*8            # 2 hour window between duplicate SMS
_last = {}
_lock = threading.Lock()

class SMSCritical(logging.Handler):
    def emit(self, record):
        if record.levelno < logging.CRITICAL or texting is None:
            return
        msg = self.format(record)
        with _lock:
            if time.time() - _last.get(msg, 0) < DUP_WINDOW:
                return
            _last[msg] = time.time()
        try:
            texting.send_verizon_text(PHONE, "Botfarm", msg)
        except Exception:
            pass                         # never crash on failed SMS

FMT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
def get_logger(name: str, level=logging.INFO):
    lg = logging.getLogger(name)
    if lg.handlers:                      # already configured
        return lg
    lg.setLevel(level)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(FMT))
    lg.addHandler(sh)
    lg.addHandler(SMSCritical())
    lg.propagate = False
    return lg
