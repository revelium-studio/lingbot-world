# LingBot-World Deployment Guide

## 🚀 Deployed Services

### Frontend (Vercel)
- **Production URL**: https://frontend-blond-alpha-20.vercel.app
- **Repository**: https://github.com/revelium-studio/lingbot-world
- **Framework**: React + Vite
- **Auto-deploys**: Pushes to `main` branch

### Backend (Modal)
- **API URL**: https://revelium-studio--lingbot-world-fastapi-app.modal.run
- **GPU**: NVIDIA A100 40GB (serverless)
- **Model**: robbyant/lingbot-world-base-cam (~15GB)
- **Dashboard**: https://modal.com/apps/revelium-studio/main/deployed/lingbot-world

## 💰 Cost Estimates (with $30 credit)

### Modal GPU Usage
- **A100 40GB**: ~$4/hour
- **Est. generation time**: ~2-5 minutes per world (161 frames)
- **Cost per generation**: ~$0.13-$0.33
- **$30 budget**: ~90-230 world generations

### How to Monitor Usage
```bash
python3 -m modal app logs lingbot-world
```

## 🔧 Deployment Commands

### Backend (Modal)

**Deploy API**:
```bash
python3 -m modal deploy modal_app.py
```

**Download model weights** (one-time, ~15GB):
```bash
python3 -m modal run modal_app.py::download_weights
```

**View logs**:
```bash
python3 -m modal app logs lingbot-world --follow
```

### Frontend (Vercel)

**Deploy**:
```bash
cd frontend
npm run build
VERCEL_TOKEN=VkfOh8AxMEAHjQzSh22omPqW vercel deploy --prod --yes
```

**Or push to GitHub** (auto-deploys):
```bash
git push origin main
```

## 📡 API Endpoints

### REST API

**Health Check**:
```bash
curl https://revelium-studio--lingbot-world-fastapi-app.modal.run/health
```

**Generate World**:
```bash
curl -X POST https://revelium-studio--lingbot-world-fastapi-app.modal.run/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A neon cyberpunk alley at night with rain",
    "resolution": [480, 832],
    "num_frames": 161,
    "sampling_steps": 40,
    "guide_scale": 5.0
  }'
```

### WebSocket

Connect to `/ws/{session_id}` for real-time frame streaming.

## 🐛 Troubleshooting

### Frontend not connecting to backend

1. Check Modal API is running:
   ```bash
   curl https://revelium-studio--lingbot-world-fastapi-app.modal.run/health
   ```

2. Check Vercel environment variables:
   - Go to Vercel dashboard → Settings → Environment Variables
   - Ensure `VITE_API_URL` points to Modal URL

3. Check browser console for CORS errors

### Modal deployment failed

1. **Out of credits**: Check Modal dashboard for remaining credits

2. **Model download failed**: Re-run weight download:
   ```bash
   python3 -m modal run modal_app.py::download_weights
   ```

3. **GPU not available**: Modal will queue your request if GPUs are busy

### Generation taking too long

- **Expected time**: 2-5 minutes for 161 frames on A100
- **Reduce frames**: Set `num_frames=81` for faster generation
- **Reduce steps**: Set `sampling_steps=20` (lower quality)

## 🔐 Security Notes

- **Vercel Token**: Already used for deployment (can be rotated in Vercel dashboard)
- **Modal Token**: Stored in `~/.modal.toml`
- **GitHub**: Public repository (code is open-source)

## 📊 Monitoring

### Modal Dashboard
- View GPU usage and costs
- Monitor function invocations
- Check logs and errors
- https://modal.com/revelium-studio

### Vercel Dashboard
- View deployments and analytics
- Monitor frontend performance
- https://vercel.com/revelium-studios

## 🔄 Update Workflow

1. **Make code changes** locally
2. **Test locally**:
   ```bash
   # Backend
   python -m backend.main
   
   # Frontend
   cd frontend && npm run dev
   ```
3. **Commit and push**:
   ```bash
   git add -A
   git commit -m "Your changes"
   git push origin main
   ```
4. **Deploy backend** (if changed):
   ```bash
   python3 -m modal deploy modal_app.py
   ```
5. **Frontend auto-deploys** via GitHub integration

## 📚 Resources

- **LingBot-World GitHub**: https://github.com/Robbyant/lingbot-world
- **Model on HuggingFace**: https://huggingface.co/robbyant/lingbot-world-base-cam
- **Paper**: https://arxiv.org/abs/2601.20540
- **Modal Docs**: https://modal.com/docs
- **Vercel Docs**: https://vercel.com/docs

## 🎯 Next Steps

1. **Test the deployed app**: Visit https://frontend-blond-alpha-20.vercel.app
2. **Monitor usage**: Check Modal dashboard to track GPU costs
3. **Optimize**: Reduce `num_frames` or `sampling_steps` to save costs
4. **Scale**: Upgrade Modal credits when needed

---

**Deployment Date**: February 1, 2026  
**Total Setup Time**: ~15 minutes  
**Budget**: $30 Modal credits
