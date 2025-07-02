# Configuration template for 3D Model Bot
# Copy this file to 'config.py' and fill in your credentials

# Telegram Bot Configuration
BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"  # Get from @BotFather
ADMIN_ID = YOUR_ADMIN_ID_HERE  # Your Telegram user ID (integer)

# Telethon Configuration (for channel parsing)
API_ID = YOUR_API_ID_HERE  # Get from https://my.telegram.org
API_HASH = "YOUR_API_HASH_HERE"  # Get from https://my.telegram.org

# Target Channel
TARGET_CHANNEL = "@blocks_01"  # Channel to parse for 3D models

# Database Configuration
DATABASE_TYPE = "firebase"  # Options: "sqlite", "firebase"

# Firebase Configuration (if using Firebase)
FIREBASE_CREDENTIALS_PATH = "firebase_credentials.json"  # Path to Firebase service account JSON

# Image Analysis Configuration
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score for results (0.0 to 1.0)
MAX_RESULTS = 10  # Maximum number of results to return

# Search Configuration
DEFAULT_SEARCH_MINUTES = 5  # Default search duration in minutes
MAX_SEARCH_MINUTES = 60  # Maximum allowed search duration 