import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QTextEdit
from PyQt5.QtCore import Qt
from football_analysis import automated_pipeline

class FootballAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle('足球比赛分析')
        self.setGeometry(100, 100, 600, 400)

        # 创建布局
        layout = QVBoxLayout()

        # 文件选择部分
        file_layout = QHBoxLayout()
        self.file_label = QLabel('选择CSV文件:')
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.browse_button = QPushButton('浏览')
        self.browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.browse_button)
        layout.addLayout(file_layout)

        # 主队输入部分
        home_layout = QHBoxLayout()
        self.home_label = QLabel('主队:')
        self.home_edit = QLineEdit()
        home_layout.addWidget(self.home_label)
        home_layout.addWidget(self.home_edit)
        layout.addLayout(home_layout)

        # 客队输入部分
        away_layout = QHBoxLayout()
        self.away_label = QLabel('客队:')
        self.away_edit = QLineEdit()
        away_layout.addWidget(self.away_label)
        away_layout.addWidget(self.away_edit)
        layout.addLayout(away_layout)

        # 分析按钮
        self.analyze_button = QPushButton('分析')
        self.analyze_button.clicked.connect(self.analyze)
        layout.addWidget(self.analyze_button)

        # 结果显示部分
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)

        # 设置布局
        self.setLayout(layout)

    def browse_file(self):
        # 打开文件对话框选择CSV文件
        file_path, _ = QFileDialog.getOpenFileName(self, '选择CSV文件', '', 'CSV文件 (*.csv)')
        if file_path:
            self.file_path_edit.setText(file_path)

    def analyze(self):
        # 获取文件路径、主队和客队名称
        file_path = self.file_path_edit.text()
        home_team = self.home_edit.text()
        away_team = self.away_edit.text()

        if not file_path or not home_team or not away_team:
            self.result_text.setText('请确保文件路径、主队和客队名称都已填写！')
            return

        # 清空之前的输出
        self.result_text.clear()

        # 重定向标准输出到QTextEdit
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        try:
            # 调用自动化分析流程
            automated_pipeline(file_path, home_team, away_team)
        except Exception as e:
            print(f"程序运行出错: {e}")
        finally:
            # 恢复标准输出
            sys.stdout = old_stdout

        # 获取输出并显示在QTextEdit中
        output = mystdout.getvalue()
        self.result_text.setText(output)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FootballAnalysisApp()
    ex.show()
    sys.exit(app.exec_())