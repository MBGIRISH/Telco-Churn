# 🔧 Fix Render Deployment Error

## Problem
Your deployed app at https://telco-churn-kehp.onrender.com shows:
```
Error: Model not found: /opt/render/project/src/models/logistic_regression.pkl
```

## Solution Options

### Option 1: Train Models on Render (Recommended)

**Update your Render service:**

1. Go to Render Dashboard → Your Service → Settings
2. Update **Build Command** to:
   ```bash
   pip install -r requirements.txt && python train_models_on_render.py
   ```
3. Click **Save Changes**
4. **Manual Deploy** → **Deploy latest commit**

**Note:** First deployment will take 10-15 minutes to train models.

### Option 2: Upload Models to GitHub

1. **Train models locally:**
   ```bash
   python src/data_cleaning.py
   python src/feature_engineering.py
   python src/model_training.py
   ```

2. **Commit models to GitHub:**
   ```bash
   git add models/*.pkl models/*.pth data/X_train.csv data/X_val.csv data/X_test.csv
   git commit -m "Add trained models"
   git push origin main
   ```

3. **Redeploy on Render** (auto-deploys on push)

### Option 3: Use Render Shell (Quick Fix)

1. Go to Render Dashboard → Your Service → **Shell**
2. Run:
   ```bash
   python src/data_cleaning.py
   python src/feature_engineering.py
   python src/model_training.py
   ```
3. Models will persist on Render disk
4. Restart your service

### Option 4: Update Build Command Only

**In Render Dashboard:**

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
python train_models_on_render.py && streamlit run app_advanced.py --server.port=$PORT --server.address=0.0.0.0
```

## ✅ What Was Fixed

1. ✅ **Better error messages** - Now shows helpful instructions
2. ✅ **Path handling** - Fixed model path resolution for Render
3. ✅ **Auto-training script** - `train_models_on_render.py` trains models if missing
4. ✅ **Graceful handling** - App shows helpful message instead of crashing

## 🚀 Quick Fix Steps

1. **Update Render Build Command:**
   ```
   pip install -r requirements.txt && python train_models_on_render.py
   ```

2. **Or use Shell to train:**
   - Render Dashboard → Shell
   - Run: `python src/model_training.py`

3. **Redeploy** your service

## 📝 Files Changed

- ✅ `src/deployment_streamlit.py` - Better path handling and error messages
- ✅ `app_advanced.py` - Graceful error handling with instructions
- ✅ `train_models_on_render.py` - New script to train models automatically
- ✅ `Procfile` - Updated to train models before starting
- ✅ `render.yaml` - Updated build command

## ⚠️ Important Notes

- **Free tier**: Models may be lost on service restart (use paid plan for persistence)
- **First deploy**: Takes 10-15 minutes to train models
- **Subsequent deploys**: Fast if models are in GitHub
- **Build timeout**: Free tier has 10 min limit (may need paid plan for training)

## 🎯 Recommended Approach

**For Production:**
1. Train models locally
2. Commit to GitHub
3. Deploy on Render (models included)

**For Quick Testing:**
1. Use Option 3 (Render Shell) to train models
2. Service will work until restart

---

**After applying the fix, your app should work!** ✅

