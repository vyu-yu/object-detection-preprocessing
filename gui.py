import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import os
import cv2
import numpy as np
import csv
from PIL import Image, ImageTk
from core.parser import LabelParser
from core.visualizer import Visualizer
from core.iou_calculator import IoUCalculator
from core.overlap_detector import OverlapDetector
from core.negative_sampler import NegativeSampler
from config import VISUALIZATION_DIR, CLASS_MAP


class OBBPreprocessingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("目标检测标签预处理系统")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)

        # 基础变量（原有不变）
        self.image_path = tk.StringVar()
        self.label_path = tk.StringVar()
        self.negative_samples_var = tk.IntVar(value=3)
        self.boxes = []
        self.meta_data = {}
        self.original_pil_image = None
        self.current_scale = 1.0
        self.image_id = None
        self.magnifier_window = None
        self.magnifier_label = None
        self.magnification = 2.5
        self.magnifier_radius = 60
        self.iou_results = []

        # 调用样式配置方法
        self.setup_styles()

        # 预生成按钮背景图片（蓝色，与原设计一致）
        self.button_bg_image = self.create_solid_image(1, 1, "#4a86e8")  # 1x1像素蓝色图片（自动拉伸）
        self.button_bg_active_image = self.create_solid_image(1, 1, "#356ac3")  # hover时背景图（略深的蓝）

        # 调用界面布局方法
        self.create_widgets()

    def setup_styles(self):
        """自定义TTK样式（保留非按钮部分，按钮改用tk.Button）"""
        self.style = ttk.Style()
        # 1. 全局默认样式（非按钮组件用）
        self.style.configure(
            ".",
            font=("Segoe UI", 10),
            foreground="#333333",
            background="#f5f5f5"
        )

        # 2. 其他组件样式（原有不变，仅移除按钮样式）
        self.style.configure("Title.TLabel", font=("Segoe UI", 11, "bold"), foreground="#2c3e50")
        self.style.configure(
            "Custom.Treeview",
            font=("Segoe UI", 9),
            rowheight=22,
            fieldbackground="#ffffff",
            background="#f9f9f9"
        )
        self.style.configure(
            "Custom.Treeview.Heading",
            font=("Segoe UI", 10, "bold"),
            padding=10,
            background="#e8e8e8",
            foreground="#2c3e50",
            relief=tk.FLAT
        )
        self.style.map("Custom.Treeview.Heading", background=[("active", "#d0d0d0")])
        self.style.configure("Custom.TEntry", padding=8, borderwidth=1, relief=tk.SUNKEN)
        self.style.configure("Custom.TCheckbutton", font=("Segoe UI", 10), padding=5)
        self.style.configure("Custom.TFrame", background="#f5f5f5")

        # 3. 状态栏样式（原有不变）
        self.style.configure("Status.TLabel", borderwidth=1, relief=tk.SUNKEN, background="#e8e8e8")

    def create_solid_image(self, width, height, color):
        """生成纯色图片（用于tk.Button的背景）"""
        # 创建PIL纯色图像（模式RGB，颜色为传入的hex色）
        pil_image = Image.new("RGB", (width, height), color=color)
        # 转换为Tkinter可用的PhotoImage
        return ImageTk.PhotoImage(pil_image)

    def create_widgets(self):
        # 顶部文件选择区
        top_frame = ttk.Frame(self.root, padding=(15, 10), style="Custom.TFrame")
        top_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))

        ttk.Label(top_frame, text="图像文件:", style="Title.TLabel").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Entry(
            top_frame,
            textvariable=self.image_path,
            width=60,
            style="Custom.TEntry"
        ).grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        # ---------------------- 修改1：替换为tk.Button，用图片背景 ----------------------
        tk.Button(
            top_frame,
            text="浏览",
            command=self.browse_image,
            image=self.button_bg_image,  # 背景图片（自动拉伸）
            compound=tk.CENTER,  # 文字在图片中央
            font=("Segoe UI", 10, "bold"),  # 字体与原设计一致
            fg="#ffffff",  # 固定字体色（亮白色）
            bg="#4a86e8",  # 兼容旧系统的背景色（图片不生效时备用）
            activebackground="#356ac3",  # hover时背景色
            activeforeground="#ffffff",  # hover时字体色
            bd=1,  # 边框宽度
            relief=tk.FLAT,  # 扁平化
            padx=10,  # 水平内边距
            pady=5  # 垂直内边距
        ).grid(row=0, column=2, padx=5)

        ttk.Label(top_frame, text="标签文件:", style="Title.TLabel").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(8, 0))
        ttk.Entry(
            top_frame,
            textvariable=self.label_path,
            width=60,
            style="Custom.TEntry"
        ).grid(row=1, column=1, padx=5, pady=(8, 0), sticky=(tk.W, tk.E))
        # ---------------------- 修改2：标签文件浏览按钮，同样替换 ----------------------
        tk.Button(
            top_frame,
            text="浏览",
            command=self.browse_label,
            image=self.button_bg_image,
            compound=tk.CENTER,
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg="#4a86e8",
            activebackground="#356ac3",
            activeforeground="#ffffff",
            bd=1,
            relief=tk.FLAT,
            padx=10,
            pady=5
        ).grid(row=1, column=2, padx=5, pady=(8, 0))

        top_frame.columnconfigure(1, weight=1)

        # 中间左右分栏（原有不变）
        main_middle_frame = ttk.Frame(self.root, padding=(15, 5), style="Custom.TFrame")
        main_middle_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_middle_frame.columnconfigure(0, weight=2)
        main_middle_frame.columnconfigure(1, weight=1)
        main_middle_frame.rowconfigure(0, weight=1)

        # 左侧图像显示区（原有不变）
        image_frame = ttk.LabelFrame(main_middle_frame, text="图像预览", style="Title.TLabel", padding=10)
        image_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        canvas_frame = ttk.Frame(image_frame, style="Custom.TFrame")
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=800,
            height=500,
            bg="white",
            bd=0,
            highlightthickness=0,
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        v_scroll.config(command=self.canvas.yview)
        h_scroll.config(command=self.canvas.xview)

        # 图像缩放按钮（应用修改后的tk.Button）
        scale_frame = ttk.Frame(image_frame, padding=5, style="Custom.TFrame")
        scale_frame.pack(fill=tk.X, anchor=tk.CENTER)
        # ---------------------- 修改3：放大按钮替换 ----------------------
        tk.Button(
            scale_frame,
            text="放大(+)",
            command=self.zoom_in,
            image=self.button_bg_image,
            compound=tk.CENTER,
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg="#4a86e8",
            activebackground="#356ac3",
            activeforeground="#ffffff",
            bd=1,
            relief=tk.FLAT,
            padx=10,
            pady=5
        ).pack(side=tk.LEFT, padx=5)
        # ---------------------- 修改4：缩小按钮替换 ----------------------
        tk.Button(
            scale_frame,
            text="缩小(-)",
            command=self.zoom_out,
            image=self.button_bg_image,
            compound=tk.CENTER,
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg="#4a86e8",
            activebackground="#356ac3",
            activeforeground="#ffffff",
            bd=1,
            relief=tk.FLAT,
            padx=10,
            pady=5
        ).pack(side=tk.LEFT, padx=5)
        self.magnifier_switch = ttk.Checkbutton(
            scale_frame,
            text="开启放大镜",
            command=self.toggle_magnifier,
            style="Custom.TCheckbutton"
        )
        self.magnifier_switch.pack(side=tk.LEFT, padx=10)

        # 右侧IoU结果区（原有不变）
        iou_frame = ttk.LabelFrame(main_middle_frame, text="IoU计算结果", style="Title.TLabel", padding=10)
        iou_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        iou_frame.rowconfigure(0, weight=1)
        iou_frame.columnconfigure(0, weight=1)

        tree_frame = ttk.Frame(iou_frame, style="Custom.TFrame")
        tree_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_v_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        tree_v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.iou_tree = ttk.Treeview(
            tree_frame,
            columns=("box1_idx", "box1_cls", "box2_idx", "box2_cls", "iou"),
            show="headings",
            yscrollcommand=tree_v_scroll.set,
            style="Custom.Treeview"
        )
        self.iou_tree.heading("box1_idx", text="框1索引", anchor=tk.CENTER)
        self.iou_tree.heading("box1_cls", text="框1类别", anchor=tk.CENTER)
        self.iou_tree.heading("box2_idx", text="框2索引", anchor=tk.CENTER)
        self.iou_tree.heading("box2_cls", text="框2类别", anchor=tk.CENTER)
        self.iou_tree.heading("iou", text="IoU值", anchor=tk.CENTER)
        self.iou_tree.column("box1_idx", width=80, anchor=tk.CENTER)
        self.iou_tree.column("box1_cls", width=120, anchor=tk.CENTER)
        self.iou_tree.column("box2_idx", width=80, anchor=tk.CENTER)
        self.iou_tree.column("box2_cls", width=120, anchor=tk.CENTER)
        self.iou_tree.column("iou", width=100, anchor=tk.CENTER)
        tree_v_scroll.config(command=self.iou_tree.yview)
        self.iou_tree.pack(fill=tk.BOTH, expand=True)

        # IoU操作按钮（替换为tk.Button）
        iou_btn_frame = ttk.Frame(iou_frame, padding=5, style="Custom.TFrame")
        iou_btn_frame.grid(row=1, column=0, sticky=tk.E)
        # ---------------------- 修改5：导出IoU结果按钮 ----------------------
        tk.Button(
            iou_btn_frame,
            text="导出IoU结果",
            command=self.export_iou_results,
            image=self.button_bg_image,
            compound=tk.CENTER,
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg="#4a86e8",
            activebackground="#356ac3",
            activeforeground="#ffffff",
            bd=1,
            relief=tk.FLAT,
            padx=10,
            pady=5
        ).pack(side=tk.RIGHT, padx=5)
        # ---------------------- 修改6：清空结果按钮 ----------------------
        tk.Button(
            iou_btn_frame,
            text="清空结果",
            command=self.clear_iou_results,
            image=self.button_bg_image,
            compound=tk.CENTER,
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg="#4a86e8",
            activebackground="#356ac3",
            activeforeground="#ffffff",
            bd=1,
            relief=tk.FLAT,
            padx=10,
            pady=5
        ).pack(side=tk.RIGHT)

        # 底部功能按钮区（所有按钮替换为tk.Button）
        bottom_frame = ttk.Frame(self.root, padding=(15, 10), style="Custom.TFrame")
        bottom_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # ---------------------- 修改7：解析标签按钮 ----------------------
        tk.Button(
            bottom_frame,
            text="解析标签",
            command=self.parse_labels,
            image=self.button_bg_image,
            compound=tk.CENTER,
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg="#4a86e8",
            activebackground="#356ac3",
            activeforeground="#ffffff",
            bd=1,
            relief=tk.FLAT,
            padx=10,
            pady=5
        ).pack(side=tk.LEFT, padx=5)
        # ---------------------- 修改8：可视化按钮 ----------------------
        tk.Button(
            bottom_frame,
            text="可视化",
            command=self.visualize,
            image=self.button_bg_image,
            compound=tk.CENTER,
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg="#4a86e8",
            activebackground="#356ac3",
            activeforeground="#ffffff",
            bd=1,
            relief=tk.FLAT,
            padx=10,
            pady=5
        ).pack(side=tk.LEFT, padx=5)
        # ---------------------- 修改9：可视化重叠按钮 ----------------------
        tk.Button(
            bottom_frame,
            text="可视化重叠",
            command=self.visualize_overlaps,
            image=self.button_bg_image,
            compound=tk.CENTER,
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg="#4a86e8",
            activebackground="#356ac3",
            activeforeground="#ffffff",
            bd=1,
            relief=tk.FLAT,
            padx=10,
            pady=5
        ).pack(side=tk.LEFT, padx=5)
        # ---------------------- 修改10：计算所有IoU按钮 ----------------------
        tk.Button(
            bottom_frame,
            text="计算所有IoU",
            command=self.calculate_all_iou,
            image=self.button_bg_image,
            compound=tk.CENTER,
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg="#4a86e8",
            activebackground="#356ac3",
            activeforeground="#ffffff",
            bd=1,
            relief=tk.FLAT,
            padx=10,
            pady=5
        ).pack(side=tk.LEFT, padx=5)
        # ---------------------- 修改11：检测重叠按钮 ----------------------
        tk.Button(
            bottom_frame,
            text="检测重叠",
            command=self.detect_overlaps,
            image=self.button_bg_image,
            compound=tk.CENTER,
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg="#4a86e8",
            activebackground="#356ac3",
            activeforeground="#ffffff",
            bd=1,
            relief=tk.FLAT,
            padx=10,
            pady=5
        ).pack(side=tk.LEFT, padx=5)

        negative_frame = ttk.Frame(bottom_frame, style="Custom.TFrame")
        negative_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(negative_frame, text="负样本数量:", style="Title.TLabel").pack(side=tk.LEFT, padx=5)
        ttk.Entry(
            negative_frame,
            textvariable=self.negative_samples_var,
            width=5,
            style="Custom.TEntry",
            justify=tk.CENTER
        ).pack(side=tk.LEFT, padx=5)
        # ---------------------- 修改12：生成负样本按钮 ----------------------
        tk.Button(
            negative_frame,
            text="生成负样本",
            command=self.generate_negatives,
            image=self.button_bg_image,
            compound=tk.CENTER,
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg="#4a86e8",
            activebackground="#356ac3",
            activeforeground="#ffffff",
            bd=1,
            relief=tk.FLAT,
            padx=10,
            pady=5
        ).pack(side=tk.LEFT)

        # ---------------------- 修改13：保存标签结果按钮 ----------------------
        tk.Button(
            bottom_frame,
            text="保存标签结果",
            command=self.save_results,
            image=self.button_bg_image,
            compound=tk.CENTER,
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg="#4a86e8",
            activebackground="#356ac3",
            activeforeground="#ffffff",
            bd=1,
            relief=tk.FLAT,
            padx=10,
            pady=5
        ).pack(side=tk.RIGHT, padx=5)

        # 状态栏（原有不变，应用样式）
        self.status_var = tk.StringVar()
        self.status_var.set("就绪 | 未加载图像和标签")
        self.status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=8,
            style="Status.TLabel"
        )
        self.status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # 窗口权重配置（原有不变）
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

    # ---------------------- 以下所有功能逻辑（原有不变，无需修改） ----------------------
    def calculate_all_iou(self):
        if len(self.boxes) < 2:
            messagebox.showerror("错误", "需要至少两个边界框才能计算IoU", parent=self.root)
            return
        if self.boxes[0].is_normalized != self.boxes[1].is_normalized:
            messagebox.showerror("错误", "所有边界框必须使用相同的坐标格式（均为绝对/归一化）", parent=self.root)
            return

        self.clear_iou_results()
        total_pairs = len(self.boxes) * (len(self.boxes) - 1) // 2
        self.status_var.set(f"正在计算IoU... 共{total_pairs}对框，请勿关闭窗口")
        self.root.update_idletasks()

        try:
            for i in range(len(self.boxes)):
                for j in range(i + 1, len(self.boxes)):
                    box1 = self.boxes[i]
                    box2 = self.boxes[j]
                    iou = IoUCalculator.calculate_iou(box1, box2)
                    cls1 = CLASS_MAP.get(box1.class_id, f"未知类_{box1.class_id}")
                    cls2 = CLASS_MAP.get(box2.class_id, f"未知类_{box2.class_id}")
                    self.iou_results.append((i, j, cls1, cls2, round(iou, 4)))

            self.update_iou_tree()
            max_iou = max([r[4] for r in self.iou_results]) if self.iou_results else 0
            self.status_var.set(f"IoU计算完成 | 共{len(self.iou_results)}对框，最大IoU: {max_iou:.4f}")
            messagebox.showinfo(
                "IoU计算完成",
                f"已计算所有{len(self.iou_results)}对边界框的IoU\n最大IoU值: {max_iou:.4f}",
                parent=self.root
            )

        except Exception as e:
            messagebox.showerror("IoU计算错误", f"计算过程中出错: {str(e)}", parent=self.root)
            self.status_var.set(f"IoU计算失败: {str(e)}")

    def update_iou_tree(self):
        for item in self.iou_tree.get_children():
            self.iou_tree.delete(item)
        sorted_results = sorted(self.iou_results, key=lambda x: x[4], reverse=True)
        for result in sorted_results:
            box1_idx, box2_idx, box1_cls, box2_cls, iou = result
            self.iou_tree.insert(
                "",
                tk.END,
                values=(box1_idx, box1_cls, box2_idx, box2_cls, f"{iou:.4f}"),
                tags=("high_iou",) if iou > 0.5 else ("low_iou",)
            )
        self.iou_tree.tag_configure("high_iou", background="#ffebee")
        self.iou_tree.tag_configure("low_iou", background="#ffffff")

    def export_iou_results(self):
        if not self.iou_results:
            messagebox.showwarning("无结果可导出", "请先计算IoU再导出", parent=self.root)
            return

        save_path = filedialog.asksaveasfilename(
            title="导出IoU结果",
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")],
            parent=self.root
        )
        if not save_path:
            return

        try:
            with open(save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["框1索引", "框1类别", "框2索引", "框2类别", "IoU值"])
                for result in self.iou_results:
                    box1_idx, box2_idx, box1_cls, box2_cls, iou = result
                    writer.writerow([box1_idx, box1_cls, box2_idx, box2_cls, round(iou, 4)])

            self.status_var.set(f"IoU结果已导出到: {os.path.basename(save_path)}")
            messagebox.showinfo("导出成功", f"IoU结果已保存到:\n{save_path}", parent=self.root)

        except Exception as e:
            messagebox.showerror("导出错误", f"保存IoU结果时出错: {str(e)}", parent=self.root)

    def clear_iou_results(self):
        self.iou_results.clear()
        for item in self.iou_tree.get_children():
            self.iou_tree.delete(item)
        self.status_var.set("IoU结果已清空 | 就绪")

    def browse_image(self):
        filename = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff")],
            parent=self.root
        )
        if filename:
            filename = os.path.normpath(filename)
            self.image_path.set(filename)
            if not os.path.exists(filename):
                messagebox.showerror("错误", f"文件不存在: {filename}", parent=self.root)
                return
            self.display_image(filename)

    def browse_label(self):
        filename = filedialog.askopenfilename(
            title="选择标签文件",
            filetypes=[("文本文件", "*.txt")],
            parent=self.root
        )
        if filename:
            filename = os.path.normpath(filename)
            if not os.path.exists(filename):
                messagebox.showerror("错误", f"文件不存在: {filename}", parent=self.root)
                return
            self.label_path.set(filename)

    def display_image(self, image_path):
        try:
            if not os.path.exists(image_path):
                messagebox.showerror("错误", f"文件不存在: {image_path}", parent=self.root)
                return
            try:
                self.original_pil_image = Image.open(image_path).convert("RGB")
            except Exception as e:
                img_cv = cv2.imread(image_path)
                if img_cv is None:
                    raise Exception(f"OpenCV也无法读取图像: {str(e)}")
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                self.original_pil_image = Image.fromarray(img_rgb)
            self.current_scale = 1.0
            self.update_scaled_image()
            self.status_var.set(f"已加载图像: {os.path.basename(image_path)} | 缩放比例: 1.0倍")

        except Exception as e:
            messagebox.showerror("图像显示错误", f"显示图像时出错: {str(e)}", parent=self.root)

    def zoom_in(self):
        if self.original_pil_image is None:
            messagebox.showwarning("提示", "请先选择并显示图像", parent=self.root)
            return
        if self.current_scale < 3.0:
            self.current_scale += 0.2
            self.update_scaled_image()
            self.status_var.set(f"图像已缩放至 {self.current_scale:.1f} 倍 | 就绪")

    def zoom_out(self):
        if self.original_pil_image is None:
            messagebox.showwarning("提示", "请先选择并显示图像", parent=self.root)
            return
        if self.current_scale > 0.5:
            self.current_scale -= 0.2
            self.update_scaled_image()
            self.status_var.set(f"图像已缩放至 {self.current_scale:.1f} 倍 | 就绪")

    def update_scaled_image(self):
        if not self.original_pil_image:
            return
        new_width = int(self.original_pil_image.width * self.current_scale)
        new_height = int(self.original_pil_image.height * self.current_scale)
        scaled_pil_image = self.original_pil_image.resize((new_width, new_height), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(scaled_pil_image)
        if self.image_id:
            self.canvas.delete(self.image_id)
        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def toggle_magnifier(self):
        if self.magnifier_switch.instate(['selected']):
            if self.magnifier_window is None:
                self.magnifier_window = tk.Toplevel(self.root)
                self.magnifier_window.overrideredirect(True)
                self.magnifier_window.attributes("-topmost", True)
                self.magnifier_label = ttk.Label(self.magnifier_window)
                self.magnifier_label.pack()
            self.canvas.bind("<Motion>", self.update_magnifier)
            self.status_var.set("已开启放大镜 | 鼠标移动到图像上查看细节")
        else:
            if self.magnifier_window:
                self.magnifier_window.destroy()
                self.magnifier_window = None
            self.canvas.unbind("<Motion>")
            self.status_var.set("已关闭放大镜 | 就绪")

    def update_magnifier(self, event):
        if self.original_pil_image is None or self.magnifier_window is None:
            return
        original_x = int(event.x / self.current_scale)
        original_y = int(event.y / self.current_scale)
        crop_left = max(0, original_x - self.magnifier_radius // self.magnification)
        crop_top = max(0, original_y - self.magnifier_radius // self.magnification)
        crop_right = min(self.original_pil_image.width, original_x + self.magnifier_radius // self.magnification)
        crop_bottom = min(self.original_pil_image.height, original_y + self.magnifier_radius // self.magnification)
        cropped_image = self.original_pil_image.crop((crop_left, crop_top, crop_right, crop_bottom))
        magnified_image = cropped_image.resize((self.magnifier_radius * 2, self.magnifier_radius * 2), Image.LANCZOS)
        self.magnifier_photo = ImageTk.PhotoImage(magnified_image)
        self.magnifier_label.config(image=self.magnifier_photo)
        self.magnifier_label.image = self.magnifier_photo
        self.magnifier_window.geometry(f"{self.magnifier_radius * 2}x{self.magnifier_radius * 2}+{event.x_root + 10}+{event.y_root + 10}")

    def parse_labels(self):
        if not self.label_path.get():
            messagebox.showerror("错误", "请先选择标签文件", parent=self.root)
            return
        try:
            self.meta_data, self.boxes = LabelParser.parse_obb_txt(self.label_path.get())
            print(f"解析到的边界框数量: {len(self.boxes)}")
            if self.boxes:
                print(f"第一个边界框的类别ID: {self.boxes[0].class_id}")
                print(f"第一个边界框的类别名称: {CLASS_MAP.get(self.boxes[0].class_id, '未知')}")
            self.status_var.set(f"解析成功: 找到 {len(self.boxes)} 个边界框")
            if self.image_path.get():
                self.visualize()
        except Exception as e:
            messagebox.showerror("错误", f"解析标签时出错: {str(e)}", parent=self.root)

    def detect_overlaps(self):
        if len(self.boxes) < 2:
            messagebox.showerror("错误", "需要至少两个边界框来检测重叠", parent=self.root)
            return
        try:
            detector = OverlapDetector(iou_threshold=0.5)
            overlaps = detector.detect_overlaps(self.boxes)
            if overlaps:
                message = f"检测到 {len(overlaps)} 对重叠边界框:\n"
                for i, j, iou in overlaps:
                    message += f"边界框 {i} 和 {j} 的IoU: {iou:.3f}\n"
                messagebox.showinfo("重叠检测结果", message, parent=self.root)
            else:
                messagebox.showinfo("重叠检测结果", "未检测到重叠边界框", parent=self.root)
        except Exception as e:
            messagebox.showerror("错误", f"检测重叠时出错: {str(e)}", parent=self.root)

    def generate_negatives(self):
        if not self.image_path.get():
            messagebox.showerror("错误", "请先选择图像文件", parent=self.root)
            return
        if not self.boxes:
            messagebox.showerror("错误", "请先解析标签文件", parent=self.root)
            return
        try:
            img = cv2.imread(self.image_path.get())
            if img is None:
                messagebox.showerror("错误", "无法读取图像文件", parent=self.root)
                return
            img_height, img_width = img.shape[:2]
            num_negatives = self.negative_samples_var.get()
            sampler = NegativeSampler(seed=42)
            negative_boxes = sampler.generate_negative_boxes(
                img_width, img_height, self.boxes, num_boxes=num_negatives
            )
            all_boxes = self.boxes + negative_boxes
            visualizer = Visualizer()
            visualizer.visualize_and_save(
                self.image_path.get(),
                all_boxes,
                "with_negatives"
            )
            self.status_var.set(f"已生成 {len(negative_boxes)} 个负样本，结果已保存到 {VISUALIZATION_DIR}")
            messagebox.showinfo("成功", f"已生成 {len(negative_boxes)} 个负样本", parent=self.root)
        except Exception as e:
            messagebox.showerror("错误", f"生成负样本时出错: {str(e)}", parent=self.root)

    def visualize(self):
        if not self.image_path.get():
            messagebox.showerror("错误", "请先选择图像文件", parent=self.root)
            return
        if not self.boxes:
            messagebox.showerror("错误", "请先解析标签文件", parent=self.root)
            return
        try:
            visualizer = Visualizer()
            output_path = os.path.join(VISUALIZATION_DIR, "visualization.jpg")
            visualizer.visualize_and_save(
                self.image_path.get(),
                self.boxes,
                "visualization"
            )
            self.display_image(output_path)
            self.status_var.set(f"可视化结果已保存到 {VISUALIZATION_DIR}")
        except Exception as e:
            messagebox.showerror("错误", f"可视化时出错: {str(e)}", parent=self.root)

    def visualize_overlaps(self):
        if not self.image_path.get():
            messagebox.showerror("错误", "请先选择图像文件", parent=self.root)
            return
        if not self.boxes:
            messagebox.showerror("错误", "请先解析标签文件", parent=self.root)
            return
        try:
            visualizer = Visualizer()
            visualizer.visualize_overlaps(
                self.image_path.get(),
                self.boxes,
                "overlaps"
            )
            self.status_var.set(f"重叠可视化结果已保存到 {VISUALIZATION_DIR}")
        except Exception as e:
            messagebox.showerror("错误", f"可视化重叠时出错: {str(e)}", parent=self.root)

    def save_results(self):
        if not self.label_path.get():
            messagebox.showerror("错误", "请先选择标签文件", parent=self.root)
            return
        if not self.boxes:
            messagebox.showerror("错误", "请先解析标签文件", parent=self.root)
            return
        try:
            filename = filedialog.asksaveasfilename(
                title="保存处理后的标签",
                defaultextension=".txt",
                filetypes=[("文本文件", "*.txt")],
                parent=self.root
            )
            if filename:
                LabelParser.save_obb_txt(self.meta_data, self.boxes, filename)
                self.status_var.set(f"处理后的标签已保存到 {filename}")
                messagebox.showinfo("成功", "处理后的标签已保存", parent=self.root)
        except Exception as e:
            messagebox.showerror("错误", f"保存结果时出错: {str(e)}", parent=self.root)


def main():
    root = tk.Tk()
    root.configure(bg="#f5f5f5")
    app = OBBPreprocessingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
