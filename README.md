# Bayer Time Series Forecasting

Enterprise-grade time series forecasting with automatic checkpoint/resume and container auto-restart.

## 🎯 Key Features

- **🔄 Auto-Resume After Crash**: Container crashes → Auto-restarts → Auto-resumes from last checkpoint
- **💾 Smart Checkpointing**: Progress saved after each combination
- **🌐 Interactive UI**: Web-based interface with real-time progress
- **📊 Multiple Models**: XGBoost, Exponential Smoothing, DLM, TSFresh
- **🚀 Zero Manual Intervention**: Once started, job completes even through multiple crashes

---

## 🚀 Quick Start

### Option 1: Docker (Recommended for Production)

```bash
# 1. Clone repository
git clone <repo-url>
cd bayer_ts_forecasting

# 2. Start with Docker Compose (auto-restart enabled)
docker-compose up -d --build

# 3. Access in browser
http://localhost:8501  # or http://YOUR-SERVER-IP:8501
```

### Option 2: Local Development

```bash
# 1. Install dependencies
uv sync

# 2. Run application
uv run streamlit run streamlit_app.py
```

---

## 💡 How It Works

### Normal Run (No Previous Crashes)

1. **Select filters** in the UI (Country, Categories, Products, etc.)
2. **Click "Run combinations"** to find valid data combinations
3. **Click "Run models"** to start forecasting
4. **Monitor progress** - progress bar shows real-time completion
5. **Get results** - download CSV when complete

### After Container Crash (Auto-Resume)

```
Before crash: Processing combination 900/1300
          ↓
💥 Container crashes
          ↓
🔄 Docker auto-restarts container (5 seconds)
          ↓
📂 App detects checkpoint file
          ↓
🔔 Shows: "Detected Incomplete Job from Previous Session"
          ↓
▶️ Click "Resume Job" button
          ↓
📌 Auto-loads: Same filters, same combinations
⏩ Auto-skips: Combinations 1-900 (already done)
▶️ Auto-continues: From combination 901
          ↓
✅ Completes combinations 901-1300
```

**Result: ZERO data loss, minimal manual intervention (just 1 click)**

---

## 📋 Detailed Usage

### First Time Run

1. **Access the UI**: Open http://localhost:8501 in browser
2. **Select Filters**:
   - Country (single selection)
   - Global CAT (multi-select)
   - Global Segment (multi-select)
   - Bayer (BCH) (multi-select)
   - Product (multi-select)
   - Target(s): Units and/or Euro Value
3. **Run Combinations**: Click button to validate filter combinations
4. **Configure**:
   - Enable/disable hyperparameter tuning
   - Review number of models and combinations
5. **Start Job**: Click "Run models"
6. **Monitor**: Watch progress bar and status updates

### After Crash/Restart

1. **Container auto-restarts** (Docker handles this)
2. **Refresh browser** (if you had it open)
3. **See notification**: Blue box shows "Detected Incomplete Job"
4. **Review saved config**: See what filters and progress were saved
5. **Click "Resume Job"**: Single click to continue
6. **Watch auto-skip**: Skips completed combinations instantly
7. **Completion**: Job finishes and clears checkpoint

### Starting Fresh (Clear Previous Job)

If you see the "Detected Incomplete Job" notification but want to start a new job:
1. Click **"Clear & Start Fresh"** button
2. Select new filters
3. Click "Run models"

---

## 🔧 Configuration

### Docker Configuration

The `docker-compose.yml` includes:
- **`restart: unless-stopped`**: Auto-restart on crash
- **Volume mount**: Persists checkpoints across restarts
- **Port 8501**: Streamlit UI access

### Checkpoint Files

Automatically created in the `./checkpoints/` directory:
- `checkpoint_Units.pkl`: Progress for Units target
- `checkpoint_Euro_Value.pkl`: Progress for Euro Value target
- `job_state.pkl`: Saves your filter configuration

### Memory Management

The app uses `ThreadPoolExecutor` (4 workers default) to:
- Share memory across threads
- Avoid DataFrame pickling overhead  
- Prevent out-of-memory errors

To adjust workers, edit `max_workers` in streamlit_app.py.

---

## 📊 Project Structure

```
bayer_ts_forecasting/
├── streamlit_app.py          # Main application with checkpoint/resume
├── timeseries_utils.py       # ML model implementations
├── file_cache_utils.py       # Caching utilities
├── docker-compose.yml        # Container configuration with auto-restart
├── Dockerfile                # Container image definition
├── pyproject.toml            # Python dependencies
├── checkpoints/              # Progress checkpoints (auto-created)
│   ├── checkpoint_Units.pkl
│   └── job_state.pkl
└── README.md                 # This file
```

---

## 🐳 Docker Commands

```bash
# Start application
docker-compose up -d

# View logs (real-time)
docker-compose logs -f

# Check status
docker-compose ps

# Restart
docker-compose restart

# Stop (keeps checkpoints)
docker-compose down

# Stop and remove all data
docker-compose down -v
```

---

## 🔍 Monitoring

### Check Container Status
```bash
docker ps | grep bayer
```

### View Logs
```bash
docker logs -f bayer_forecasting
```

### Check Checkpoints
```bash
ls -lh checkpoints/
```

### Resource Usage
```bash
docker stats bayer_forecasting
```

---

## ❓ FAQ

**Q: What happens if I lose power during a run?**
A: Docker auto-restarts the container. On startup, click "Resume Job" to continue.

**Q: Can I change filters mid-run?**
A: Click "Clear & Start Fresh", then select new filters and run again.

**Q: How do I know if a checkpoint exists?**
A: You'll see a blue notification box on app startup if an incomplete job is detected.

**Q: What if I don't want to resume?**
A: Click "Clear & Start Fresh" button to delete the checkpoint and start over.

**Q: Does it work without Docker?**
A: Yes, but you need to manually restart the app after crashes. Auto-restart is a Docker feature.

**Q: How much data loss on crash?**
A: Zero. Every completed combination is saved to checkpoint immediately.

**Q: How long does one combination take?**
A: 30-60 seconds typically, depending on data size and models enabled.

**Q: Can I run multiple jobs simultaneously?**
A: No, one job at a time. Complete or clear current job before starting another.

---

## 🛠️ Troubleshooting

### Container won't start
```bash
# Check logs
docker logs bayer_forecasting

# Rebuild from scratch
docker-compose down
docker-compose up -d --build
```

### Checkpoint not working
```bash
# Verify checkpoints directory exists
ls -la checkpoints/

# Clear corrupted checkpoints
rm -rf checkpoints/
mkdir checkpoints/
```

### Out of memory
```bash
# Reduce workers in streamlit_app.py
# Find: max_workers=4
# Change to: max_workers=2

# Or add memory limit to docker-compose.yml:
services:
  bayer_app:
    mem_limit: 8g
```

### Port already in use
```bash
# Check what's using port 8501
sudo lsof -i :8501

# Or change port in docker-compose.yml:
ports:
  - "8502:8501"  # Use 8502 instead
```

---

## 📦 Requirements

- **Docker** 20.10+ and Docker Compose 1.29+ (for container deployment)
- **Python** 3.9+ (for local development)
- **RAM**: 8GB+ recommended
- **Disk**: 10GB+ for data and checkpoints

---

## 📄 License

[Your License Here]

---

## 🤝 Support

For issues or questions:
- Check logs: `docker logs bayer_forecasting`
- Review this README
- Check checkpoint files in `./checkpoints/`

---

## ✨ What Makes This Special

**Traditional approach:**
- Crash = lose all progress
- Must manually track where you left off
- Risky for long-running jobs

**This solution:**
- Crash = auto-restart + auto-resume
- Progress saved after every combination
- Safe for multi-hour/multi-day jobs
- Minimal intervention (1 button click)