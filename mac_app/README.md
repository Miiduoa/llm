# LLM 聊天助手 - Mac 應用程式

這是一個原生 macOS 應用程式，提供美觀的聊天介面來與你的 LLM 模型互動。

## 🚀 快速開始

### 方法一：直接執行 Python 腳本
```bash
cd mac_app
pip install PyQt6 requests
python app.py
```

### 方法二：建置成 Mac 應用程式
```bash
cd mac_app
python build_app.py
```

## 📱 功能特色

- **原生 Mac 介面**：使用 PyQt6 打造的美觀介面
- **自動服務管理**：自動啟動/停止本地 API 服務
- **即時聊天**：流暢的對話體驗
- **狀態監控**：即時顯示服務狀態
- **錯誤處理**：完善的錯誤提示和處理

## 🔧 系統需求

- macOS 10.14 或更高版本
- Python 3.8+
- 至少 4GB RAM（用於模型載入）

## 📦 建置說明

### 1. 安裝依賴
```bash
pip install -r ../requirements.txt
pip install PyQt6 pyinstaller
```

### 2. 建置應用程式
```bash
python build_app.py
```

### 3. 安裝應用程式
```bash
# 方法一：直接複製
cp -r dist/LLMChat.app /Applications/

# 方法二：使用安裝腳本
cd dist
./install.sh
```

## 🎨 自定義

### 修改介面
編輯 `app.py` 中的 `init_ui()` 方法來調整介面佈局和樣式。

### 修改 API 設定
在 `app.py` 中修改 `self.api_url` 變數來指向不同的 API 端點。

### 添加圖標
將你的圖標檔案命名為 `icon.icns` 並放在 `mac_app` 目錄中。

## 🐛 故障排除

### 應用程式無法啟動
1. 檢查 Python 版本：`python --version`
2. 重新安裝依賴：`pip install -r ../requirements.txt`
3. 檢查控制台錯誤訊息

### 服務無法連接
1. 確保本地 API 服務正在運行
2. 檢查防火牆設定
3. 確認端口 8000 未被佔用

### 模型載入失敗
1. 檢查網路連線
2. 確認模型檔案完整
3. 增加記憶體分配

## 📄 授權

此專案使用 MIT 授權條款。
