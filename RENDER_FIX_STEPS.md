# 🚀 Step-by-Step: Fix Render Deployment (Option 1)

## 📋 Prerequisites
- Your code is already pushed to GitHub
- You have access to Render Dashboard
- Your service is already created on Render

---

## 🔧 Step-by-Step Instructions

### Step 1: Push Updated Files to GitHub

1. **Open terminal in your project directory:**
   ```bash
   cd /Users/mbgirish/Telco-Churn
   ```

2. **Check what files changed:**
   ```bash
   git status
   ```

3. **Add all changes:**
   ```bash
   git add .
   ```

4. **Commit the changes:**
   ```bash
   git commit -m "Fix Render deployment - add model training script"
   ```

5. **Push to GitHub:**
   ```bash
   git push origin main
   ```
   (Replace `main` with your branch name if different)

---

### Step 2: Access Render Dashboard

1. **Go to Render website:**
   - Open: https://dashboard.render.com
   - Login to your account

2. **Navigate to your service:**
   - Click on **"Web Services"** in the left sidebar
   - Find your service: `telco-churn-kehp` (or similar name)
   - Click on it to open

---

### Step 3: Update Build Command

1. **Go to Settings:**
   - In your service page, click on **"Settings"** tab (top menu)

2. **Find Build Command:**
   - Scroll down to **"Build Command"** section
   - You'll see the current command:
     ```
     pip install -r requirements.txt
     ```

3. **Update Build Command:**
   - Click in the Build Command field
   - Replace with:
     ```bash
     pip install -r requirements.txt && python train_models_on_render.py
     ```
   - Or if you want it to continue even if training fails:
     ```bash
     pip install -r requirements.txt && (python train_models_on_render.py || echo "Training skipped, models may exist")
     ```

4. **Save Changes:**
   - Click **"Save Changes"** button at the bottom

---

### Step 4: Manual Deploy

1. **Go to Manual Deploy:**
   - In your service page, click on **"Manual Deploy"** dropdown (top right)
   - Select **"Deploy latest commit"**

2. **Wait for Deployment:**
   - You'll see build logs in real-time
   - **First deployment will take 10-15 minutes** (training models)
   - Watch the logs to see progress:
     - Installing dependencies
     - Data cleaning
     - Feature engineering
     - Training models
     - Starting Streamlit

3. **Monitor Logs:**
   - Click on **"Logs"** tab to see detailed output
   - Look for messages like:
     - "Step 1/4: Data cleaning..."
     - "Step 2/4: Feature engineering..."
     - "Step 3/4: Training models..."
     - "Model training completed!"

---

### Step 5: Verify Deployment

1. **Check Service Status:**
   - Wait until status shows **"Live"** (green)
   - URL will be shown at the top

2. **Test Your App:**
   - Click on your service URL (e.g., `https://telco-churn-kehp.onrender.com`)
   - The app should load without model errors
   - Try making a prediction to verify

3. **If Still Errors:**
   - Check the Logs tab for any errors
   - Verify models were created (check logs for "Model saved to...")
   - If training failed, try Option 2 (Render Shell)

---

## ⏱️ Expected Timeline

- **Build time (first deploy):** 10-15 minutes
- **Build time (subsequent):** 2-5 minutes (if models exist)
- **Service startup:** 30-60 seconds

---

## 🔍 What to Look For in Logs

### ✅ Success Indicators:
```
Step 1/4: Data cleaning...
Step 2/4: Feature engineering...
Step 3/4: Training models...
Model saved to /opt/render/project/src/models/logistic_regression.pkl
Model training completed!
Starting Streamlit...
```

### ❌ Error Indicators:
```
Error: No module named 'xgboost'
Error: File not found: data/WA_Fn-UseC_-Telco-Customer-Churn.csv
Timeout: Build exceeded 10 minutes
```

---

## 🐛 Troubleshooting

### Issue: Build Times Out
**Solution:** 
- Free tier has 10 min limit
- Upgrade to paid plan OR
- Use Option 2 (train via Shell) OR
- Upload pre-trained models to GitHub

### Issue: Training Fails
**Solution:**
- Check logs for specific error
- Ensure dataset file exists in `data/` directory
- Verify all dependencies in `requirements.txt`

### Issue: Models Still Not Found
**Solution:**
- Check logs to see if models were actually created
- Verify path in logs matches what app expects
- Try Option 2 (Render Shell) as alternative

---

## ✅ Success Checklist

- [ ] Files pushed to GitHub
- [ ] Build command updated in Render
- [ ] Manual deploy initiated
- [ ] Build completed successfully (10-15 min)
- [ ] Service shows "Live" status
- [ ] App loads without errors
- [ ] Predictions work correctly

---

## 📞 Need Help?

If you encounter issues:
1. Check Render logs for specific errors
2. Verify all files are in GitHub
3. Try Option 2 (Render Shell) as backup
4. Check `FIX_RENDER_DEPLOYMENT.md` for alternatives

---

**After completing these steps, your app should work!** 🎉

