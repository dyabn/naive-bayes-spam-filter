"""Tkinter graphical interface for the spam mail demo system."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import database
import mail_service


METHOD_LABELS = {
    "rule": "固定关键词规则",
    "bayes": "朴素贝叶斯",
}


class SpamMailApp:
    """Main Tkinter application window."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("基于朴素贝叶斯的垃圾邮件过滤与纠错系统")
        self.root.geometry("1200x720")
        self.root.minsize(1000, 650)

        self.users = database.list_users()
        if not self.users:
            raise RuntimeError("数据库中没有可用用户")

        self.user_by_name = {user["username"]: user for user in self.users}
        self.user_by_id = {user["id"]: user for user in self.users}
        self.current_user_id = self.users[0]["id"]
        self.current_folder = "inbox"
        self.current_email_id: int | None = None
        self.mail_records = []

        self.user_var = tk.StringVar(value=self.users[0]["username"])
        self.folder_title_var = tk.StringVar(value="收件箱")
        self.status_var = tk.StringVar(value="")

        self.sender_value = tk.StringVar(value="-")
        self.receiver_value = tk.StringVar(value="-")
        self.subject_value = tk.StringVar(value="-")
        self.time_value = tk.StringVar(value="-")
        self.spam_probability_value = tk.StringVar(value="-")
        self.normal_probability_value = tk.StringVar(value="-")
        self.method_value = tk.StringVar(value="-")
        self.reason_value = tk.StringVar(value="-")
        self.keywords_value = tk.StringVar(value="-")

        self.build_layout()
        self.show_inbox()

    def build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self.build_header()

        content = ttk.Frame(self.root, padding=(12, 8, 12, 12))
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(1, weight=2)
        content.columnconfigure(2, weight=3)
        content.rowconfigure(0, weight=1)

        self.build_sidebar(content)
        self.build_mail_list(content)
        self.build_detail_panel(content)

        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w", padding=(12, 4))
        status.grid(row=2, column=0, sticky="ew")

    def build_header(self) -> None:
        header = ttk.Frame(self.root, padding=(12, 12, 12, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        title = ttk.Label(
            header,
            text="基于朴素贝叶斯的垃圾邮件过滤与纠错系统",
            font=("Microsoft YaHei UI", 16, "bold"),
        )
        title.grid(row=0, column=0, sticky="w")

        user_frame = ttk.Frame(header)
        user_frame.grid(row=0, column=1, sticky="e")
        ttk.Label(user_frame, text="当前用户").grid(row=0, column=0, padx=(0, 8))
        user_box = ttk.Combobox(
            user_frame,
            textvariable=self.user_var,
            values=[user["username"] for user in self.users],
            state="readonly",
            width=14,
        )
        user_box.grid(row=0, column=1)
        user_box.bind("<<ComboboxSelected>>", self.switch_user)

    def build_sidebar(self, parent: ttk.Frame) -> None:
        sidebar = ttk.Frame(parent, padding=(0, 0, 12, 0))
        sidebar.grid(row=0, column=0, sticky="ns")

        compose = ttk.Button(sidebar, text="+ 写邮件", command=self.open_compose_window)
        compose.grid(row=0, column=0, sticky="ew", pady=(0, 14))

        ttk.Button(sidebar, text="收件箱", command=self.show_inbox).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=4,
        )
        ttk.Button(sidebar, text="已发送", command=self.show_sent).grid(
            row=2,
            column=0,
            sticky="ew",
            pady=4,
        )
        ttk.Button(sidebar, text="垃圾回收站", command=self.show_trash).grid(
            row=3,
            column=0,
            sticky="ew",
            pady=4,
        )

    def build_mail_list(self, parent: ttk.Frame) -> None:
        list_frame = ttk.Frame(parent)
        list_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 12))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(1, weight=1)

        ttk.Label(
            list_frame,
            textvariable=self.folder_title_var,
            font=("Microsoft YaHei UI", 12, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        columns = ("party", "subject", "time")
        self.mail_tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="browse")
        self.mail_tree.heading("party", text="发件人")
        self.mail_tree.heading("subject", text="主题")
        self.mail_tree.heading("time", text="时间")
        self.mail_tree.column("party", width=110, anchor="w")
        self.mail_tree.column("subject", width=260, anchor="w")
        self.mail_tree.column("time", width=150, anchor="w")
        self.mail_tree.grid(row=1, column=0, sticky="nsew")
        self.mail_tree.bind("<<TreeviewSelect>>", self.show_mail_detail)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.mail_tree.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.mail_tree.configure(yscrollcommand=scrollbar.set)

    def build_detail_panel(self, parent: ttk.Frame) -> None:
        detail = ttk.Frame(parent)
        detail.grid(row=0, column=2, sticky="nsew")
        detail.columnconfigure(1, weight=1)
        detail.rowconfigure(5, weight=1)

        ttk.Label(detail, text="邮件详情", font=("Microsoft YaHei UI", 12, "bold")).grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(0, 8),
        )

        rows = [
            ("发件人", self.sender_value),
            ("收件人", self.receiver_value),
            ("主题", self.subject_value),
            ("时间", self.time_value),
        ]
        for index, (label, variable) in enumerate(rows, start=1):
            ttk.Label(detail, text=f"{label}：").grid(row=index, column=0, sticky="nw", pady=2)
            ttk.Label(detail, textvariable=variable, wraplength=430).grid(
                row=index,
                column=1,
                sticky="ew",
                pady=2,
            )

        ttk.Label(detail, text="正文：").grid(row=5, column=0, sticky="nw", pady=(8, 2))
        self.content_text = tk.Text(detail, height=9, wrap="word", state="disabled")
        self.content_text.grid(row=5, column=1, sticky="nsew", pady=(8, 8))

        info_rows = [
            ("垃圾概率", self.spam_probability_value),
            ("正常概率", self.normal_probability_value),
            ("判定方式", self.method_value),
            ("判定原因", self.reason_value),
            ("检测特征", self.keywords_value),
        ]
        for index, (label, variable) in enumerate(info_rows, start=6):
            ttk.Label(detail, text=f"{label}：").grid(row=index, column=0, sticky="nw", pady=2)
            ttk.Label(detail, textvariable=variable, wraplength=430).grid(
                row=index,
                column=1,
                sticky="ew",
                pady=2,
            )

        actions = ttk.Frame(detail)
        actions.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        self.mark_spam_button = ttk.Button(
            actions,
            text="标记为垃圾邮件",
            command=self.mark_current_as_spam,
        )
        self.mark_normal_button = ttk.Button(
            actions,
            text="这不是垃圾邮件",
            command=self.mark_current_as_normal,
        )

    def switch_user(self, _event=None) -> None:
        selected = self.user_by_name[self.user_var.get()]
        self.current_user_id = selected["id"]
        self.clear_detail()
        self.refresh_mail_list()

    def show_inbox(self) -> None:
        self.current_folder = "inbox"
        self.folder_title_var.set("收件箱")
        self.mail_tree.heading("party", text="发件人")
        self.refresh_mail_list()

    def show_sent(self) -> None:
        self.current_folder = "sent"
        self.folder_title_var.set("已发送")
        self.mail_tree.heading("party", text="收件人")
        self.refresh_mail_list()

    def show_trash(self) -> None:
        self.current_folder = "trash"
        self.folder_title_var.set("垃圾回收站")
        self.mail_tree.heading("party", text="发件人")
        self.refresh_mail_list()

    def load_current_folder_records(self):
        if self.current_folder == "sent":
            return mail_service.get_sent(self.current_user_id)
        if self.current_folder == "trash":
            return mail_service.get_trash(self.current_user_id)
        return mail_service.get_inbox(self.current_user_id)

    def refresh_mail_list(self) -> None:
        self.mail_records = self.load_current_folder_records()
        self.current_email_id = None

        for item in self.mail_tree.get_children():
            self.mail_tree.delete(item)

        for record in self.mail_records:
            party_id = record["receiver_id"] if self.current_folder == "sent" else record["sender_id"]
            party_name = self.username_for_id(party_id)
            self.mail_tree.insert(
                "",
                "end",
                iid=str(record["id"]),
                values=(party_name, record["subject"], record["send_time"]),
            )

        self.clear_detail()
        self.status_var.set(f"{self.folder_title_var.get()}：{len(self.mail_records)} 封邮件")

    def show_mail_detail(self, _event=None) -> None:
        selection = self.mail_tree.selection()
        if not selection:
            return

        email_id = int(selection[0])
        record = next((mail for mail in self.mail_records if mail["id"] == email_id), None)
        if record is None:
            return

        self.current_email_id = email_id
        self.sender_value.set(self.username_for_id(record["sender_id"]))
        self.receiver_value.set(self.username_for_id(record["receiver_id"]))
        self.subject_value.set(record["subject"])
        self.time_value.set(record["send_time"])
        self.spam_probability_value.set(self.format_probability(record["spam_probability"]))
        self.normal_probability_value.set(self.format_probability(record["normal_probability"]))
        self.method_value.set(METHOD_LABELS.get(record["classification_method"], "-"))
        self.reason_value.set(record["classification_reason"] or "-")

        text = mail_service.combine_mail_text(record["subject"], record["content"])
        if record["classification_method"] == "rule":
            keywords = mail_service.get_matched_hard_spam_words(text)
        else:
            keywords = mail_service.get_matched_feature_words(text)
        self.keywords_value.set("、".join(keywords) if keywords else "-")

        self.set_content_text(record["content"])
        self.update_action_buttons()

    def clear_detail(self) -> None:
        self.current_email_id = None
        for variable in (
            self.sender_value,
            self.receiver_value,
            self.subject_value,
            self.time_value,
            self.spam_probability_value,
            self.normal_probability_value,
            self.method_value,
            self.reason_value,
            self.keywords_value,
        ):
            variable.set("-")
        self.set_content_text("")
        self.update_action_buttons()

    def set_content_text(self, content: str) -> None:
        self.content_text.configure(state="normal")
        self.content_text.delete("1.0", "end")
        self.content_text.insert("1.0", content or "")
        self.content_text.configure(state="disabled")

    def update_action_buttons(self) -> None:
        self.mark_spam_button.grid_forget()
        self.mark_normal_button.grid_forget()
        if self.current_email_id is None:
            return
        if self.current_folder == "inbox":
            self.mark_spam_button.grid(row=0, column=0, sticky="w")
        elif self.current_folder == "trash":
            self.mark_normal_button.grid(row=0, column=0, sticky="w")

    def open_compose_window(self) -> None:
        window = tk.Toplevel(self.root)
        window.title("写邮件")
        window.geometry("520x460")
        window.transient(self.root)
        window.grab_set()
        window.columnconfigure(1, weight=1)
        window.rowconfigure(3, weight=1)

        current_user = self.user_by_id[self.current_user_id]
        receiver_names = [user["username"] for user in self.users if user["id"] != self.current_user_id]
        receiver_var = tk.StringVar(value=receiver_names[0] if receiver_names else "")
        subject_var = tk.StringVar()

        ttk.Label(window, text="发件人：").grid(row=0, column=0, sticky="w", padx=12, pady=(14, 8))
        ttk.Label(window, text=current_user["username"]).grid(row=0, column=1, sticky="ew", padx=12, pady=(14, 8))

        ttk.Label(window, text="收件人：").grid(row=1, column=0, sticky="w", padx=12, pady=8)
        receiver_box = ttk.Combobox(
            window,
            textvariable=receiver_var,
            values=receiver_names,
            state="readonly",
        )
        receiver_box.grid(row=1, column=1, sticky="ew", padx=12, pady=8)

        ttk.Label(window, text="主题：").grid(row=2, column=0, sticky="w", padx=12, pady=8)
        ttk.Entry(window, textvariable=subject_var).grid(row=2, column=1, sticky="ew", padx=12, pady=8)

        ttk.Label(window, text="正文：").grid(row=3, column=0, sticky="nw", padx=12, pady=8)
        body_text = tk.Text(window, height=12, wrap="word")
        body_text.grid(row=3, column=1, sticky="nsew", padx=12, pady=8)

        actions = ttk.Frame(window)
        actions.grid(row=4, column=0, columnspan=2, sticky="e", padx=12, pady=(8, 14))

        def send_current_mail() -> None:
            receiver_name = receiver_var.get()
            if not receiver_name:
                messagebox.showwarning("无法发送", "请选择收件人")
                return

            subject = subject_var.get().strip()
            content = body_text.get("1.0", "end").strip()
            if not subject and not content:
                messagebox.showwarning("无法发送", "主题和正文不能同时为空")
                return

            receiver = self.user_by_name[receiver_name]
            result = mail_service.send_mail(
                sender_id=self.current_user_id,
                receiver_id=receiver["id"],
                subject=subject,
                content=content,
            )
            messagebox.showinfo("发送成功", f"邮件已发送，收件人文件夹：{self.folder_label(result['receiver_folder'])}")
            window.destroy()
            self.refresh_mail_list()

        ttk.Button(actions, text="发送", command=send_current_mail).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(actions, text="取消", command=window.destroy).grid(row=0, column=1)

    def mark_current_as_spam(self) -> None:
        if self.current_email_id is None:
            return
        mail_service.mark_as_spam(self.current_email_id)
        messagebox.showinfo("已更新", "邮件已标记为垃圾邮件")
        self.refresh_mail_list()

    def mark_current_as_normal(self) -> None:
        if self.current_email_id is None:
            return
        mail_service.mark_as_normal(self.current_email_id)
        messagebox.showinfo("已更新", "邮件已移回收件箱")
        self.refresh_mail_list()

    def username_for_id(self, user_id: int) -> str:
        user = self.user_by_id.get(user_id)
        return user["username"] if user else f"用户 {user_id}"

    @staticmethod
    def format_probability(value) -> str:
        if value is None:
            return "-"
        return f"{float(value):.2%}"

    @staticmethod
    def folder_label(folder: str) -> str:
        return {
            "inbox": "收件箱",
            "sent": "已发送",
            "trash": "垃圾回收站",
        }.get(folder, folder)
