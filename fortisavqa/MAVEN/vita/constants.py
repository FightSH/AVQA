# Model Constants
MAX_IMAGE_LENGTH = 16  # 8#16#32
MIN_IMAGE_LENGTH = 4
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
AUDIO_TOKEN_INDEX = -500
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_AUDIO_TOKEN = "<audio>"
CONTROLLER_HEART_BEAT_EXPIRATION = 30
LOGDIR = "gradio-logs"
WORKER_HEART_BEAT_INTERVAL = 15
DEFAULT_DATA_RATIO = 1.0
GLOBAL_WEIGHTS_PATH = ""
MCCD = {"flag": True,
        "lambda_multifaceted": 0.001,
        "lambda_cycle": 0.005,
        "multifaceted":{"lang": True, "audio": True, "video": True},
        "cycle": False
        }
