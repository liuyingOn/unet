# -*-coding:utf-8 -*-

# @Description: 图形化界面，支持图片检测，并输出对应的占比


import shutil
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
import torch
import os.path as osp
from model.unet_model import UNet
import numpy as np
import time
import threading

# 窗口主类，选择使用GPU（如果可用）或者CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MainWindow(QTabWidget):
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle('基于Unet的眼底血管分割 ')
        self.resize(1200, 800)
        self.setWindowIcon(QIcon("images/UI/lufei.png"))
        # 图片读取进程
        self.output_size = 480
        self.img2predict = ""

        # 初始化UNet模型
        net = UNet(n_channels=1, n_classes=1)
        # 将网络拷贝到deivce中
        net.to(device=device)
        # 加载模型参数
        net.load_state_dict(torch.load('best_model.pth', map_location=device))  # todo 模型位置
        # 测试模式
        net.eval()
        self.model = net
        # 初始化UI
        self.initUI()

    '''
    ***界面初始化***
    '''

    def initUI(self):
        # 图片检测子界面
        # 设置字体
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)
        # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget = QWidget()  # 创建一个新的QWidget作为图片检测界面
        img_detection_layout = QVBoxLayout()  # 创建一个垂直布局管理器
        img_detection_title = QLabel("图片识别功能")  # 创建一个标签显示“图片识别功能”
        img_detection_title.setAlignment(Qt.AlignCenter)  # 设置标签居中对齐
        img_detection_title.setFont(font_title)  # 设置标签字体
        # 中间图片显示区域
        mid_img_widget = QWidget()  # 创建一个新的QWidget作为中间图片显示区域
        mid_img_layout = QHBoxLayout()  # 创建一个水平布局管理器
        self.left_img = QLabel()  # 创建一个标签用于显示左侧图片
        self.right_img = QLabel()  # 创建一个标签用于显示右侧图片
        self.left_img.setPixmap(QPixmap("images/UI/up.jpeg"))  # 设置左侧图片标签的初始显示图片
        self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))  # 设置右侧图片标签的初始显示图片
        self.left_img.setAlignment(Qt.AlignCenter)  # 设置左侧图片标签居中对齐
        self.right_img.setAlignment(Qt.AlignCenter)  # 设置右侧图片标签居中对齐
        mid_img_layout.addWidget(self.left_img)  # 将左侧图片标签添加到水平布局中
        mid_img_layout.addStretch(0)  # 在图片标签之间添加一个弹性空间
        mid_img_layout.addWidget(self.right_img)  # 将右侧图片标签添加到水平布局中
        mid_img_widget.setLayout(mid_img_layout)  # 将水平布局设置为中间图片显示区域的布局
        # 创建上传图片和开始检测按钮
        up_img_button = QPushButton("上传图片")  # 创建一个按钮用于上传图片
        det_img_button = QPushButton("开始检测")  # 创建一个按钮用于开始检测

        # 连接按钮点击事件到相应的处理函数
        up_img_button.clicked.connect(self.upload_img)  # 连接上传图片按钮的点击事件到self.upload_img函数
        det_img_button.clicked.connect(self.detect_img)  # 连接开始检测按钮的点击事件到self.detect_img函数

        # 设置按钮字体
        up_img_button.setFont(font_main)  # 设置上传图片按钮的字体
        det_img_button.setFont(font_main)  # 设置开始检测按钮的字体

        # 设置上传图片按钮的样式
        up_img_button.setStyleSheet(
            "QPushButton{color:white}"  # 按钮文字颜色为白色
            "QPushButton:hover{background-color: rgb(2,110,180);}"  # 鼠标悬停时背景颜色为RGB(2,110,180)
            "QPushButton{background-color:rgb(48,124,208)}"  # 按钮背景颜色为RGB(48,124,208)
            "QPushButton{border:2px}"  # 按钮边框宽度为2px
            "QPushButton{border-radius:5px}"  # 按钮边框圆角半径为5px
            "QPushButton{padding:5px 5px}"  # 按钮内部填充为5px
            "QPushButton{margin:5px 5px}"  # 按钮外部边距为5px
        )

        # 设置开始检测按钮的样式
        det_img_button.setStyleSheet(
            "QPushButton{color:white}"  # 按钮文字颜色为白色
            "QPushButton:hover{background-color: rgb(2,110,180);}"  # 鼠标悬停时背景颜色为RGB(2,110,180)
            "QPushButton{background-color:rgb(48,124,208)}"  # 按钮背景颜色为RGB(48,124,208)
            "QPushButton{border:2px}"  # 按钮边框宽度为2px
            "QPushButton{border-radius:5px}"  # 按钮边框圆角半径为5px
            "QPushButton{padding:5px 5px}"  # 按钮内部填充为5px
            "QPushButton{margin:5px 5px}"  # 按钮外部边距为5px
        )

        # 将组件添加到垂直布局中
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)  # 添加标题标签并居中对齐
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)  # 添加中间图片显示区域并居中对齐
        img_detection_layout.addWidget(up_img_button)  # 添加上传图片按钮
        img_detection_layout.addWidget(det_img_button)  # 添加开始检测按钮

        # 设置图片检测界面的布局
        img_detection_widget.setLayout(img_detection_layout)

        # 将图片检测界面添加到主窗口的标签页中
        self.addTab(img_detection_widget, '图片检测')



        # todo 关于界面
        # 创建关于窗口的小部件
        about_widget = QWidget()  # 创建一个新的QWidget作为关于界面
        about_layout = QVBoxLayout()  # 创建一个垂直布局管理器
        about_title = QLabel('欢迎使用医学影像语义分割系统\n\n ')  # 创建一个标签用于显示欢迎信息（TODO: 修改欢迎词语）
        about_title.setFont(QFont('楷体', 18))  # 设置标签字体为楷体，字号为18
        about_title.setAlignment(Qt.AlignCenter)  # 设置标签居中对齐

        # 添加关于界面中的图片
        about_img = QLabel()  # 创建一个标签用于显示图片
        about_img.setPixmap(QPixmap('images/UI/qq.png'))  # 设置标签的初始显示图片
        about_img.setAlignment(Qt.AlignCenter)  # 设置图片标签居中对齐

        # 添加作者信息标签
        label_super = QLabel()  # 创建一个标签用于显示作者信息（TODO: 添加作者信息）
        label_super.setText("<a href='https://www.yuanshen.com/'>你可以在这里找到我-->启动</a>")  # 设置标签的超链接文本
        label_super.setFont(QFont('楷体', 16))  # 设置标签字体为楷体，字号为16
        label_super.setAlignment(Qt.AlignRight)  # 设置标签右对齐
        label_super.setOpenExternalLinks(True)  # 启用标签的外部链接打开功能
        # 将各个组件添加到垂直布局中
        about_layout.addWidget(about_title)  # 添加欢迎信息标签到布局中
        about_layout.addStretch()  # 添加一个弹性空间
        about_layout.addWidget(about_img)  # 添加图片标签到布局中
        about_layout.addStretch()  # 添加一个弹性空间
        about_layout.addWidget(label_super)  # 添加作者信息标签到布局中
        about_widget.setLayout(about_layout)  # 设置关于界面的布局

        # 设置左侧图片标签居中对齐
        self.left_img.setAlignment(Qt.AlignCenter)

        # 将图片检测和关于界面添加到主窗口的标签页中
        self.addTab(img_detection_widget, '图片检测')  # 添加图片检测标签页
        self.addTab(about_widget, '作者')  # 添加关于标签页

        # 设置标签页的图标
        self.setTabIcon(0, QIcon('images/UI/lufei.png'))  # 设置第一个标签页（图片检测）的图标
        self.setTabIcon(1, QIcon('images/UI/lufei.png'))  # 设置第二个标签页（作者）的图标
        self.setTabIcon(2, QIcon('images/UI/lufei.png'))  # 设置第三个标签页的图标（虽然代码中没有第三个标签页，但这里是为了防止代码报错）

    '''
    ***上传图片***
    '''

    def upload_img(self):
        # 选择图片文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        print('a', fileName)
        if fileName:
            suffix = fileName.split(".")[-1]  # 获取文件后缀名
            filename1 = fileName.split("/")[7::]  # 获取文件路径中的文件名部分
            filename1 = '/'.join(filename1)  # 将路径列表转换为字符串
            print(filename1)
            print('cc', filename1)
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)  # 设置保存路径
            shutil.copy(fileName, save_path)  # 将文件复制到指定路径
            if not osp.exists(fileName):
                print("Error: 文件不存在。")
                return
            # 读取图片并调整大小
            im0 = cv2.imread(save_path)  # 读取图片
            resize_scale = self.output_size / im0.shape[0]  # 计算调整比例
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)  # 调整图片大小
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)  # 保存调整后的图片
            self.img2predict = filename1  # 更新待预测图片路径
            print('c', self.img2predict)
            print("self.img2predict:", self.img2predict)
            self.origin_shape = (im0.shape[1], im0.shape[0])  # 保存原始图片大小
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))  # 在左侧标签上显示调整后的图片
            # 上传图片之后右侧的图片重置
            self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))  # 重置右侧图片
            # 调用 detect_img 方法进行图片检测
            self.detect_img()

    '''
    ***检测图片***
    '''

    def detect_img(self):
        # 视频在这个基础上加入for循环进来
        source = self.img2predict  ##获取待预测的图片路径
        print("Trying to load image from:", source)
        # 读取图片
        img = cv2.imread(source)
        if img is None:
            print("Error: 图像没有成功加载。请检查图像路径是否正确。")
            return
            # 转为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 调整图像大小为512x512
        img = cv2.resize(img, (512, 512))

        # 将图像转为4维数组，batch大小为1，通道数为1，图像大小为512x512
        img = img.reshape(1, 1, img.shape[0], img.shape[1])

        # 将numpy数组转为tensor
        img_tensor = torch.from_numpy(img)

        # 将tensor拷贝到设备中，device可以是CPU或者CUDA
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)

        # 使用模型进行预测
        pred = self.model(img_tensor)

        # 将预测结果从tensor转为numpy数组，并移除batch和通道维度
        pred_np = pred.squeeze(0).squeeze(0).data.cpu().numpy()

        # 将预测结果二值化，阈值为0.5，大于等于0.5的像素设为255（白色），小于0.5的像素设为0（黑色）
        pred_binary = (pred_np >= 0.5).astype(np.uint8) * 255

        # 保存预测结果图片
        cv2.imwrite("images/tmp/single_result.jpg", pred_binary)

        # 在右侧标签上显示预测结果图片
        self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))

        # 界面关闭

    def closeEvent(self, event):
        # 退出确认对话框
        reply = QMessageBox.question(self,
                                     'quit',
                                     "Are you sure?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
