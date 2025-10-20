#!/usr/bin/env python3
"""
Mac 應用程式建置腳本
使用 PyInstaller 將 Python 應用程式打包成 macOS .app 檔案
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def build_mac_app():
    """建置 Mac 應用程式"""
    
    # 確保在正確的目錄
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print("🚀 開始建置 Mac 應用程式...")
    
    # 檢查 PyInstaller 是否已安裝
    try:
        import PyInstaller
        print("✅ PyInstaller 已安裝")
    except ImportError:
        print("❌ PyInstaller 未安裝，正在安裝...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # 建置參數
    build_args = [
        "pyinstaller",
        "--onefile",  # 單一執行檔
        "--windowed",  # 無控制台視窗
        "--name=LLMChat",  # 應用程式名稱
        "--icon=icon.icns",  # 圖標（如果有的話）
        "--add-data=../src:src",  # 包含 src 目錄
        "--add-data=../configs:configs",  # 包含 configs 目錄
        "--hidden-import=uvicorn",  # 隱藏導入
        "--hidden-import=fastapi",
        "--hidden-import=transformers",
        "--hidden-import=torch",
        "--hidden-import=peft",
        "app.py"
    ]
    
    # 如果沒有圖標檔案，移除圖標參數
    if not os.path.exists("icon.icns"):
        build_args = [arg for arg in build_args if not arg.startswith("--icon")]
    
    print("📦 執行 PyInstaller...")
    print(f"命令: {' '.join(build_args)}")
    
    try:
        # 執行建置
        result = subprocess.run(build_args, cwd=script_dir, check=True)
        print("✅ 建置成功！")
        
        # 移動到 dist 目錄
        dist_dir = script_dir / "dist"
        if dist_dir.exists():
            print(f"📁 應用程式位置: {dist_dir / 'LLMChat.app'}")
            
            # 創建安裝腳本
            create_install_script(dist_dir)
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 建置失敗: {e}")
        return False

def create_install_script(dist_dir):
    """創建安裝腳本"""
    install_script = dist_dir / "install.sh"
    
    with open(install_script, "w") as f:
        f.write("""#!/bin/bash
# LLM 聊天助手安裝腳本

echo "🚀 安裝 LLM 聊天助手..."

# 複製應用程式到應用程式目錄
sudo cp -r LLMChat.app /Applications/

# 設定權限
sudo chmod -R 755 /Applications/LLMChat.app

echo "✅ 安裝完成！"
echo "📱 你可以在 Launchpad 或應用程式資料夾中找到 'LLM 聊天助手'"
echo ""
echo "💡 首次使用時，請確保："
echo "   1. 已安裝 Python 和相關套件"
echo "   2. 網路連線正常（用於下載模型）"
echo "   3. 給予應用程式必要的權限"
""")
    
    # 設定執行權限
    os.chmod(install_script, 0o755)
    print(f"📝 已創建安裝腳本: {install_script}")

def create_dmg():
    """創建 DMG 安裝包"""
    print("📦 創建 DMG 安裝包...")
    
    script_dir = Path(__file__).parent
    dist_dir = script_dir / "dist"
    
    if not (dist_dir / "LLMChat.app").exists():
        print("❌ 找不到應用程式檔案")
        return False
    
    try:
        # 創建 DMG
        dmg_cmd = [
            "hdiutil", "create",
            "-volname", "LLM 聊天助手",
            "-srcfolder", str(dist_dir / "LLMChat.app"),
            "-ov", "-format", "UDZO",
            str(dist_dir / "LLMChat.dmg")
        ]
        
        subprocess.run(dmg_cmd, check=True)
        print(f"✅ DMG 創建成功: {dist_dir / 'LLMChat.dmg'}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ DMG 創建失敗: {e}")
        return False

def main():
    """主函數"""
    print("=" * 50)
    print("🤖 LLM 聊天助手 - Mac 應用程式建置工具")
    print("=" * 50)
    
    # 建置應用程式
    if build_mac_app():
        print("\n🎉 建置完成！")
        
        # 詢問是否創建 DMG
        create_dmg_choice = input("\n是否創建 DMG 安裝包？(y/n): ").lower().strip()
        if create_dmg_choice in ['y', 'yes', '是']:
            create_dmg()
        
        print("\n📋 後續步驟：")
        print("1. 測試應用程式: 雙擊 dist/LLMChat.app")
        print("2. 安裝到系統: 執行 dist/install.sh")
        print("3. 分發應用程式: 使用 dist/LLMChat.dmg")
        
    else:
        print("\n❌ 建置失敗，請檢查錯誤訊息")

if __name__ == "__main__":
    main()
