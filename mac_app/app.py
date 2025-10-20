import sys
import os
import threading
import webbrowser
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QTextEdit, QLineEdit, QPushButton, QLabel,
                             QMessageBox, QProgressBar, QSplitter)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QIcon
import requests
import json
import subprocess
import time


class LLMWorker(QThread):
    response_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    
    def __init__(self, prompt, api_url):
        super().__init__()
        self.prompt = prompt
        self.api_url = api_url
        
    def run(self):
        try:
            self.status_changed.emit("正在思考...")
            response = requests.post(
                f"{self.api_url}/v1/generate",
                json={
                    "prompt": self.prompt,
                    "max_new_tokens": 256,
                    "temperature": 0.7,
                    "top_p": 0.95
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                self.response_received.emit(data.get("output", "無回應"))
            else:
                self.error_occurred.emit(f"API 錯誤: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.error_occurred.emit(f"連線錯誤: {str(e)}")
        except Exception as e:
            self.error_occurred.emit(f"未知錯誤: {str(e)}")


class LLMChatApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.api_url = "http://127.0.0.1:8000"  # 本地 API
        self.server_process = None
        self.init_ui()
        self.start_local_server()
        
    def init_ui(self):
        self.setWindowTitle("LLM 聊天助手")
        self.setGeometry(100, 100, 800, 600)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主佈局
        main_layout = QVBoxLayout(central_widget)
        
        # 標題
        title_label = QLabel("🤖 LLM 聊天助手")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #2c3e50; margin: 10px;")
        main_layout.addWidget(title_label)
        
        # 分割器
        splitter = QSplitter()
        main_layout.addWidget(splitter)
        
        # 聊天區域
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        
        # 聊天記錄
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
            }
        """)
        chat_layout.addWidget(self.chat_display)
        
        # 輸入區域
        input_layout = QHBoxLayout()
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("輸入你的問題...")
        self.message_input.returnPressed.connect(self.send_message)
        self.message_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
            }
        """)
        
        self.send_button = QPushButton("發送")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.send_button)
        chat_layout.addLayout(input_layout)
        
        splitter.addWidget(chat_widget)
        
        # 狀態區域
        status_widget = QWidget()
        status_layout = QVBoxLayout(status_widget)
        
        # 狀態標籤
        self.status_label = QLabel("正在啟動服務...")
        self.status_label.setStyleSheet("color: #6c757d; font-size: 12px;")
        status_layout.addWidget(self.status_label)
        
        # 進度條
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        
        # 控制按鈕
        control_layout = QHBoxLayout()
        
        self.start_button = QPushButton("啟動服務")
        self.start_button.clicked.connect(self.start_local_server)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        
        self.stop_button = QPushButton("停止服務")
        self.stop_button.clicked.connect(self.stop_local_server)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        status_layout.addLayout(control_layout)
        
        splitter.addWidget(status_widget)
        
        # 初始訊息
        self.add_message("系統", "歡迎使用 LLM 聊天助手！正在啟動本地服務...")
        
    def start_local_server(self):
        """啟動本地 API 服務"""
        try:
            if self.server_process is None:
                self.status_label.setText("正在啟動服務...")
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 0)  # 無限進度條
                
                # 啟動 API 服務
                self.server_process = subprocess.Popen([
                    sys.executable, "-m", "uvicorn", 
                    "src.server:app", 
                    "--host", "127.0.0.1", 
                    "--port", "8000"
                ], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
                # 等待服務啟動
                QTimer.singleShot(3000, self.check_server_status)
            else:
                self.add_message("系統", "服務已在運行中")
                
        except Exception as e:
            self.add_message("錯誤", f"無法啟動服務: {str(e)}")
            self.status_label.setText("服務啟動失敗")
            self.progress_bar.setVisible(False)
    
    def check_server_status(self):
        """檢查服務狀態"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=2)
            if response.status_code == 200:
                self.status_label.setText("服務運行中 ✅")
                self.progress_bar.setVisible(False)
                self.add_message("系統", "服務已就緒，可以開始聊天！")
            else:
                self.status_label.setText("服務啟動中...")
                QTimer.singleShot(2000, self.check_server_status)
        except:
            self.status_label.setText("服務啟動中...")
            QTimer.singleShot(2000, self.check_server_status)
    
    def stop_local_server(self):
        """停止本地 API 服務"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process = None
            self.status_label.setText("服務已停止")
            self.add_message("系統", "服務已停止")
    
    def send_message(self):
        """發送訊息"""
        message = self.message_input.text().strip()
        if not message:
            return
            
        self.add_message("你", message)
        self.message_input.clear()
        self.send_button.setEnabled(False)
        
        # 啟動工作線程
        self.worker = LLMWorker(message, self.api_url)
        self.worker.response_received.connect(self.handle_response)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.status_changed.connect(self.status_label.setText)
        self.worker.start()
    
    def handle_response(self, response):
        """處理回應"""
        self.add_message("AI", response)
        self.send_button.setEnabled(True)
        self.status_label.setText("服務運行中 ✅")
    
    def handle_error(self, error):
        """處理錯誤"""
        self.add_message("錯誤", error)
        self.send_button.setEnabled(True)
        self.status_label.setText("服務運行中 ⚠️")
    
    def add_message(self, sender, message):
        """添加訊息到聊天記錄"""
        if sender == "你":
            color = "#007bff"
            alignment = "right"
        elif sender == "AI":
            color = "#28a745"
            alignment = "left"
        elif sender == "系統":
            color = "#6c757d"
            alignment = "center"
        else:
            color = "#dc3545"
            alignment = "center"
            
        html = f"""
        <div style="margin: 5px; text-align: {alignment};">
            <span style="color: {color}; font-weight: bold;">{sender}:</span>
            <span style="color: #2c3e50;">{message}</span>
        </div>
        """
        self.chat_display.append(html)
        
        # 滾動到底部
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def closeEvent(self, event):
        """關閉應用程式時清理資源"""
        if self.server_process:
            self.server_process.terminate()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("LLM 聊天助手")
    app.setApplicationVersion("1.0.0")
    
    # 設定應用程式圖標（如果有）
    # app.setWindowIcon(QIcon("icon.ico"))
    
    window = LLMChatApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
