#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

from PIL import Image, ImageTk, ImageDraw


# ============================================================
# 設定
# ============================================================

IMAGES_PER_PAGE = 4

# 写真を大きく表示するためのサイズ
DISPLAY_WIDTH = 520
DISPLAY_HEIGHT = 360


class AnnotationViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("COCO Annotation Review")
        self.root.geometry("1120x760")

        self.json_path = None
        self.image_dir = None
        self.data = None

        self.current_page = 0
        self.page_items = []

        self.checkbox_vars = []
        self.image_labels = []
        self.image_names = []
        self.tk_images = []

        self.build_ui()

        self.root.after(100, self.open_json)

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def build_ui(self):
        # 上部：OPEN JSON
        top_frame = tk.Frame(self.root)
        top_frame.pack(
            fill="x",
            padx=10,
            pady=(10, 5),
        )

        open_button = tk.Button(
            top_frame,
            text="OPEN JSON",
            command=self.open_json,
            width=14,
            height=2,
            font=("Helvetica", 12, "bold"),
        )
        open_button.pack(side="left")

        self.info_label = tk.Label(
            top_frame,
            text="annotations.json を選択してください",
            anchor="w",
            font=("Helvetica", 12),
        )
        self.info_label.pack(
            side="left",
            fill="x",
            expand=True,
            padx=12,
        )

        # OPEN JSONのすぐ下：NEXT
        control_frame = tk.Frame(self.root)
        control_frame.pack(
            fill="x",
            padx=10,
            pady=(0, 5),
        )

        self.page_label = tk.Label(
            control_frame,
            text="",
            anchor="w",
            font=("Helvetica", 11),
        )
        self.page_label.pack(side="left")

        next_button = tk.Button(
            control_frame,
            text="NEXT →",
            command=self.next_page,
            width=14,
            height=2,
            font=("Helvetica", 12, "bold"),
        )
        next_button.pack(side="right")

        # 2 x 2 の画像表示領域
        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5,
        )

        for row in range(2):
            self.grid_frame.rowconfigure(row, weight=1)

        for col in range(2):
            self.grid_frame.columnconfigure(col, weight=1)

        for i in range(4):
            cell = tk.Frame(
                self.grid_frame,
                relief="groove",
                borderwidth=1,
            )
            cell.grid(
                row=i // 2,
                column=i % 2,
                sticky="nsew",
                padx=5,
                pady=5,
            )

            # 写真とDELETEを横並びにする
            content_frame = tk.Frame(cell)
            content_frame.pack(
                fill="both",
                expand=True,
                padx=5,
                pady=5,
            )

            image_frame = tk.Frame(content_frame)
            image_frame.pack(
                side="left",
                fill="both",
                expand=True,
            )

            image_label = tk.Label(
                image_frame,
                text="",
                bg="#eeeeee",
            )
            image_label.pack(
                fill="both",
                expand=True,
            )

            side_frame = tk.Frame(
                content_frame,
                width=90,
            )
            side_frame.pack(
                side="right",
                fill="y",
                padx=(8, 0),
            )
            side_frame.pack_propagate(False)

            var = tk.BooleanVar(value=False)

            check = tk.Checkbutton(
                side_frame,
                text="DELETE",
                variable=var,
                font=("Helvetica", 11),
                justify="left",
                anchor="w",
            )
            check.pack(
                anchor="w",
                pady=(10, 0),
            )

            name_label = tk.Label(
                cell,
                text="",
                anchor="w",
                font=("Helvetica", 10),
            )
            name_label.pack(
                fill="x",
                padx=8,
                pady=(0, 5),
            )

            self.image_labels.append(image_label)
            self.image_names.append(name_label)
            self.checkbox_vars.append(var)

    # --------------------------------------------------------
    # JSONを開く
    # --------------------------------------------------------

    def open_json(self):
        path = filedialog.askopenfilename(
            title="COCO annotations.json を選択",
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )

        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"JSONを読み込めませんでした。\n\n{e}",
            )
            return

        if "images" not in data or "annotations" not in data:
            messagebox.showerror(
                "Error",
                "COCO形式ではありません。\n"
                "\"images\" と \"annotations\" が必要です。",
            )
            return

        self.json_path = Path(path)
        self.image_dir = self.json_path.parent / "images"
        self.data = data
        self.current_page = 0

        if not self.image_dir.exists():
            self.image_dir = self.json_path.parent

        self.info_label.config(
            text=f"{self.json_path}    |    "
                 f"images: {len(self.data['images'])}"
        )

        self.show_page()

    # --------------------------------------------------------
    # 現在のページを表示
    # --------------------------------------------------------

    def show_page(self):
        if self.data is None:
            return

        images = self.data["images"]

        start = self.current_page * IMAGES_PER_PAGE
        end = min(
            start + IMAGES_PER_PAGE,
            len(images),
        )

        self.page_items = images[start:end]
        self.tk_images = []

        for i in range(IMAGES_PER_PAGE):
            self.checkbox_vars[i].set(False)
            self.image_labels[i].config(
                image="",
                text="",
            )
            self.image_names[i].config(text="")

        for i, image_info in enumerate(self.page_items):
            file_name = image_info["file_name"]

            image_path = self.json_path.parent / file_name

            if not image_path.exists():
                image_path = self.image_dir / Path(file_name).name

            if not image_path.exists():
                self.image_labels[i].config(
                    text=f"IMAGE NOT FOUND\n{file_name}",
                    fg="red",
                )
                self.image_names[i].config(
                    text=file_name,
                )
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                self.image_labels[i].config(
                    text=f"IMAGE ERROR\n{e}",
                    fg="red",
                )
                continue

            image_id = image_info["id"]

            annotations = [
                ann
                for ann in self.data["annotations"]
                if ann["image_id"] == image_id
            ]

            draw = ImageDraw.Draw(image)

            for ann in annotations:
                bbox = ann.get("bbox")

                if not bbox or len(bbox) != 4:
                    continue

                x, y, w, h = bbox

                x1 = int(x)
                y1 = int(y)
                x2 = int(x + w)
                y2 = int(y + h)

                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline="red",
                    width=max(3, image.width // 500),
                )

            image.thumbnail(
                (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                Image.Resampling.LANCZOS,
            )

            tk_image = ImageTk.PhotoImage(image)
            self.tk_images.append(tk_image)

            self.image_labels[i].config(
                image=tk_image,
                text="",
                fg="black",
            )

            self.image_names[i].config(
                text=file_name,
            )

        total_pages = max(
            1,
            (len(images) + IMAGES_PER_PAGE - 1)
            // IMAGES_PER_PAGE,
        )

        current_page_number = self.current_page + 1

        self.page_label.config(
            text=(
                f"Page {current_page_number} / {total_pages}"
                f"    "
                f"Images {start + 1} - {end} / {len(images)}"
            )
        )

    # --------------------------------------------------------
    # NEXT
    # --------------------------------------------------------

    def next_page(self):
        if self.data is None:
            return

        selected_indices = []

        for i, image_info in enumerate(self.page_items):
            if self.checkbox_vars[i].get():
                selected_indices.append(i)

        selected_images = [
            self.page_items[i]
            for i in selected_indices
        ]

        deleted_any = False

        if selected_images:
            names = "\n".join(
                image_info["file_name"]
                for image_info in selected_images
            )

            answer = messagebox.askyesno(
                "Delete annotations?",
                "以下の画像をCOCO JSONから削除します。\n\n"
                f"{names}\n\n"
                "画像ファイル自体は削除しません。\n"
                "よろしいですか？",
            )

            if not answer:
                return

            self.delete_images(selected_images)
            deleted_any = True

        # 削除した場合は、同じページ番号を再表示。
        # 後ろの画像が前に詰まるため、ここでページを飛ばさない。
        total_images = len(self.data["images"])

        if total_images == 0:
            self.current_page = 0
            self.show_page()
            self.page_label.config(
                text="Images 0 / 0    ✓ COMPLETE"
            )
            return

        total_pages = max(
            1,
            (total_images + IMAGES_PER_PAGE - 1)
            // IMAGES_PER_PAGE,
        )

        if self.current_page >= total_pages:
            self.current_page = total_pages - 1

        if deleted_any:
            self.show_page()
            return

        # 削除していない場合だけ次の4枚へ進む
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.show_page()
        else:
            self.show_page()
            self.page_label.config(
                text=self.page_label.cget("text")
                + "    ✓ LAST PAGE"
            )

    # --------------------------------------------------------
    # COCO JSONから画像とannotationを削除
    # --------------------------------------------------------

    def delete_images(self, selected_images):
        delete_ids = {
            image_info["id"]
            for image_info in selected_images
        }

        backup_path = self.json_path.with_suffix(
            self.json_path.suffix + ".bak"
        )

        if not backup_path.exists():
            shutil.copy2(
                self.json_path,
                backup_path,
            )

        self.data["images"] = [
            image_info
            for image_info in self.data["images"]
            if image_info["id"] not in delete_ids
        ]

        self.data["annotations"] = [
            ann
            for ann in self.data["annotations"]
            if ann["image_id"] not in delete_ids
        ]

        with open(
            self.json_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                self.data,
                f,
                ensure_ascii=False,
                indent=2,
            )

        self.info_label.config(
            text=(
                f"{self.json_path}    |    "
                f"images: {len(self.data['images'])}"
            )
        )

        messagebox.showinfo(
            "Deleted",
            f"{len(selected_images)}枚をJSONから削除しました。\n\n"
            f"バックアップ:\n{backup_path.name}",
        )


def main():
    root = tk.Tk()
    app = AnnotationViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
