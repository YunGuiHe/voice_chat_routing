import os
import queue
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk


ROOT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = ROOT_DIR / "skills" / "voice-chat-routing"
SKILL_SCRIPTS = SKILL_ROOT / "scripts"
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))

from voice_chat_runtime import VoiceChatSkill


APP_BG = "#f6f7fb"
SIDEBAR_BG = "#eef2f7"
SIDEBAR_BORDER = "#d7dde8"
TEXT_PRIMARY = "#1f2937"
TEXT_SECONDARY = "#64748b"
BUTTON_BG = "#ffffff"
BUTTON_ACTIVE_BG = "#dbeafe"
BUTTON_PRIMARY_BG = "#2563eb"
BUTTON_PRIMARY_ACTIVE_BG = "#1d4ed8"
BUTTON_PRIMARY_TEXT = "#ffffff"
INPUT_BG = "#ffffff"


class MultiTurnDebugGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Voice Chat Skill 调试")
        self.root.geometry("1120x760")
        self.root.minsize(920, 620)
        self.result_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.nav_buttons: dict[str, tk.Button] = {}
        self.pages: dict[str, tk.Frame] = {}
        self.current_page = "chat"
        self.advanced_visible = False
        self.streaming_response_active = False
        self._skill_instance: VoiceChatSkill | None = None
        self._skill_config: tuple[int, int] | None = None
        self._skill_lock = threading.Lock()
        self._configure_style()
        self._build_ui()
        self._show_page("chat")
        self._poll_queue()
        self._warm_local_classifier()

    def _configure_style(self) -> None:
        style = ttk.Style()
        style.configure("TButton", padding=6)
        style.configure("TLabel", font=("Arial", 12))
        style.configure("TEntry", padding=4)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, minsize=270)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.sidebar = tk.Frame(self.root, bg=SIDEBAR_BG, padx=14, pady=14)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.main = tk.Frame(self.root, bg=APP_BG)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.rowconfigure(0, weight=1)
        self.main.columnconfigure(0, weight=1)

        self._build_sidebar()
        self._build_pages()

    def _build_sidebar(self) -> None:
        title = tk.Label(
            self.sidebar,
            text="Voice Chat",
            bg=SIDEBAR_BG,
            fg=TEXT_PRIMARY,
            font=("Arial", 20, "bold"),
            anchor="w",
        )
        title.pack(fill=tk.X)

        subtitle = tk.Label(
            self.sidebar,
            text="独立 Skill 调试",
            bg=SIDEBAR_BG,
            fg=TEXT_SECONDARY,
            font=("Arial", 11),
            anchor="w",
        )
        subtitle.pack(fill=tk.X, pady=(2, 18))

        self.session_var = tk.StringVar(value="debug")
        self.user_var = tk.StringVar(value="default")
        self.threshold_var = tk.IntVar(value=6)
        self.recent_var = tk.IntVar(value=3)

        self._add_nav_button("chat", "对话")
        self._add_nav_button("memory", "记忆")

        separator = tk.Frame(self.sidebar, bg=SIDEBAR_BORDER, height=1)
        separator.pack(fill=tk.X, pady=18)

        tk.Label(
            self.sidebar,
            text="当前状态",
            bg=SIDEBAR_BG,
            fg=TEXT_PRIMARY,
            font=("Arial", 13, "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 8))

        self.context_var = tk.StringVar()
        self.session_var.trace_add("write", self._update_context_labels)
        self.user_var.trace_add("write", self._update_context_labels)
        self._update_context_labels()
        tk.Label(
            self.sidebar,
            textvariable=self.context_var,
            bg=SIDEBAR_BG,
            fg=TEXT_SECONDARY,
            font=("Arial", 11),
            anchor="w",
            justify=tk.LEFT,
        ).pack(fill=tk.X)

        self.advanced_button = tk.Button(
            self.sidebar,
            text="高级参数 >",
            command=self._toggle_advanced,
            bg=SIDEBAR_BG,
            fg=TEXT_PRIMARY,
            activebackground=BUTTON_ACTIVE_BG,
            activeforeground=TEXT_PRIMARY,
            relief=tk.FLAT,
            padx=10,
            pady=8,
            anchor="w",
            font=("Arial", 11),
        )
        self.advanced_button.pack(fill=tk.X, pady=(14, 0))

        self.advanced_frame = tk.Frame(self.sidebar, bg=SIDEBAR_BG)
        self._add_sidebar_entry("session_id", self.session_var, self.advanced_frame)
        self._add_sidebar_entry("user_id", self.user_var, self.advanced_frame)
        self._add_sidebar_spinbox("摘要阈值轮数", self.threshold_var, 1, 50, self.advanced_frame)
        self._add_sidebar_spinbox("保留最近轮数", self.recent_var, 1, 20, self.advanced_frame)

        tk.Button(
            self.advanced_frame,
            text="新建会话",
            command=self.new_session,
            bg=BUTTON_BG,
            fg=TEXT_PRIMARY,
            activebackground=BUTTON_ACTIVE_BG,
            activeforeground=TEXT_PRIMARY,
            relief=tk.FLAT,
            padx=10,
            pady=8,
            anchor="w",
        ).pack(fill=tk.X, pady=(16, 6))

        tk.Button(
            self.advanced_frame,
            text="清空当前会话",
            command=self.reset_session,
            bg=BUTTON_BG,
            fg=TEXT_PRIMARY,
            activebackground=BUTTON_ACTIVE_BG,
            activeforeground=TEXT_PRIMARY,
            relief=tk.FLAT,
            padx=10,
            pady=8,
            anchor="w",
        ).pack(fill=tk.X)

        self.status_var = tk.StringVar(value="空闲")
        self.status_label = tk.Label(
            self.sidebar,
            textvariable=self.status_var,
            bg=SIDEBAR_BG,
            fg=TEXT_SECONDARY,
            font=("Arial", 11),
            anchor="w",
            wraplength=230,
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(12, 0))

    def _add_nav_button(self, page: str, text: str) -> None:
        button = tk.Button(
            self.sidebar,
            text=text,
            command=lambda: self._show_page(page),
            bg=SIDEBAR_BG,
            fg=TEXT_PRIMARY,
            activebackground=BUTTON_ACTIVE_BG,
            activeforeground=TEXT_PRIMARY,
            relief=tk.FLAT,
            padx=10,
            pady=10,
            anchor="w",
            font=("Arial", 12),
        )
        button.pack(fill=tk.X, pady=2)
        self.nav_buttons[page] = button

    def _add_sidebar_entry(
        self,
        label: str,
        variable: tk.StringVar,
        parent: tk.Frame | None = None,
    ) -> None:
        container = parent or self.sidebar
        tk.Label(
            container,
            text=label,
            bg=SIDEBAR_BG,
            fg=TEXT_SECONDARY,
            anchor="w",
            font=("Arial", 10),
        ).pack(fill=tk.X, pady=(8, 2))
        tk.Entry(
            container,
            textvariable=variable,
            bg=INPUT_BG,
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief=tk.FLAT,
        ).pack(fill=tk.X, ipady=6)

    def _add_sidebar_spinbox(
        self,
        label: str,
        variable: tk.IntVar,
        start: int,
        end: int,
        parent: tk.Frame | None = None,
    ) -> None:
        container = parent or self.sidebar
        tk.Label(
            container,
            text=label,
            bg=SIDEBAR_BG,
            fg=TEXT_SECONDARY,
            anchor="w",
            font=("Arial", 10),
        ).pack(fill=tk.X, pady=(8, 2))
        tk.Spinbox(
            container,
            from_=start,
            to=end,
            textvariable=variable,
            bg=INPUT_BG,
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            buttonbackground="#e2e8f0",
            relief=tk.FLAT,
        ).pack(fill=tk.X, ipady=5)

    def _build_pages(self) -> None:
        self.pages["chat"] = self._build_chat_page(self.main)
        self.pages["memory"] = self._build_memory_page(self.main)

        for frame in self.pages.values():
            frame.grid(row=0, column=0, sticky="nsew")

    def _update_context_labels(self, *_: object) -> None:
        if not hasattr(self, "context_var"):
            return
        self.context_var.set(f"用户：{self._user_id()}\n会话：{self._session_id()}")

    def _toggle_advanced(self) -> None:
        self.advanced_visible = not self.advanced_visible
        if self.advanced_visible:
            self.advanced_frame.pack(fill=tk.X, pady=(6, 0), before=self.status_label)
            self.advanced_button.configure(text="高级参数 v")
        else:
            self.advanced_frame.pack_forget()
            self.advanced_button.configure(text="高级参数 >")

    def _build_chat_page(self, parent: tk.Frame) -> tk.Frame:
        page = tk.Frame(parent, bg=APP_BG, padx=22, pady=18)
        page.rowconfigure(1, weight=1)
        page.columnconfigure(0, weight=1)

        header = tk.Frame(page, bg=APP_BG)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)

        tk.Label(
            header,
            text="对话调试",
            bg=APP_BG,
            fg=TEXT_PRIMARY,
            font=("Arial", 20, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        tk.Button(
            header,
            text="清空屏幕",
            command=self.clear_chat_display,
            bg="#e2e8f0",
            fg=TEXT_PRIMARY,
            activebackground="#cbd5e1",
            relief=tk.FLAT,
            padx=10,
            pady=6,
        ).grid(row=0, column=1, sticky="e")

        self.chat_text = scrolledtext.ScrolledText(
            page,
            wrap=tk.WORD,
            bg=INPUT_BG,
            fg=TEXT_PRIMARY,
            relief=tk.FLAT,
            padx=18,
            pady=18,
            font=("Arial", 13),
        )
        self.chat_text.grid(row=1, column=0, sticky="nsew")
        self.chat_text.tag_configure("user", foreground=TEXT_PRIMARY, font=("Arial", 13, "bold"))
        self.chat_text.tag_configure("assistant", foreground=TEXT_PRIMARY)
        self.chat_text.tag_configure("meta", foreground=TEXT_SECONDARY, font=("Arial", 10))

        input_area = tk.Frame(page, bg=APP_BG)
        input_area.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        input_area.columnconfigure(0, weight=1)

        self.query_text = scrolledtext.ScrolledText(
            input_area,
            height=4,
            wrap=tk.WORD,
            bg=INPUT_BG,
            fg=TEXT_PRIMARY,
            relief=tk.FLAT,
            padx=12,
            pady=10,
            font=("Arial", 12),
        )
        self.query_text.grid(row=0, column=0, sticky="ew")

        self.send_button = tk.Button(
            input_area,
            text="发送",
            command=self.send_query,
            bg=BUTTON_PRIMARY_BG,
            fg=BUTTON_PRIMARY_TEXT,
            activebackground=BUTTON_PRIMARY_ACTIVE_BG,
            activeforeground=BUTTON_PRIMARY_TEXT,
            relief=tk.FLAT,
            padx=18,
            pady=12,
            font=("Arial", 12, "bold"),
        )
        self.send_button.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        return page

    def _build_memory_page(self, parent: tk.Frame) -> tk.Frame:
        page = tk.Frame(parent, bg=APP_BG, padx=22, pady=18)
        page.rowconfigure(1, weight=1)
        page.columnconfigure(0, weight=1)

        header = tk.Frame(page, bg=APP_BG)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)

        tk.Label(
            header,
            text="会话记忆",
            bg=APP_BG,
            fg=TEXT_PRIMARY,
            font=("Arial", 20, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        tk.Button(
            header,
            text="刷新",
            command=self.refresh_memory_view,
            bg="#e2e8f0",
            fg=TEXT_PRIMARY,
            activebackground="#cbd5e1",
            relief=tk.FLAT,
            padx=10,
            pady=6,
        ).grid(row=0, column=1, sticky="e")

        panes = ttk.PanedWindow(page, orient=tk.VERTICAL)
        panes.grid(row=1, column=0, sticky="nsew")

        summary_box = ttk.LabelFrame(panes, text="当前摘要", padding=8)
        self.summary_text = scrolledtext.ScrolledText(summary_box, height=8, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        panes.add(summary_box, weight=1)

        long_term_box = ttk.LabelFrame(panes, text="长期记忆", padding=8)
        self.long_term_memory_text = scrolledtext.ScrolledText(long_term_box, height=8, wrap=tk.WORD)
        self.long_term_memory_text.pack(fill=tk.BOTH, expand=True)
        panes.add(long_term_box, weight=1)

        history_box = ttk.LabelFrame(panes, text="最近历史", padding=8)
        self.history_text = scrolledtext.ScrolledText(history_box, height=12, wrap=tk.WORD)
        self.history_text.pack(fill=tk.BOTH, expand=True)
        panes.add(history_box, weight=2)
        return page

    def _show_page(self, page: str) -> None:
        self.current_page = page
        self.pages[page].tkraise()
        for name, button in self.nav_buttons.items():
            if name == page:
                button.configure(bg=BUTTON_ACTIVE_BG, fg=TEXT_PRIMARY)
            else:
                button.configure(bg=SIDEBAR_BG, fg=TEXT_PRIMARY)

    def _make_skill(self) -> VoiceChatSkill:
        config = (self.threshold_var.get(), self.recent_var.get())
        with self._skill_lock:
            if self._skill_instance is None or self._skill_config != config:
                self._skill_instance = VoiceChatSkill.from_env(
                    SKILL_ROOT,
                    summary_threshold_rounds=config[0],
                    recent_rounds=config[1],
                    background_memory_updates=True,
                    background_update_delay_seconds=10.0,
                    background_update_callback=self._on_background_update,
                )
                self._skill_config = config
            return self._skill_instance

    def _on_background_update(self, event: dict[str, object]) -> None:
        self.result_queue.put(("background", event))

    def _warm_local_classifier(self) -> None:
        if os.getenv("VOICE_CHAT_CLASSIFIER_MODE", "local").strip().lower() != "local":
            return

        def worker() -> None:
            try:
                skill = self._make_skill()
                skill.classify("你好")
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def _session_id(self) -> str:
        return self.session_var.get().strip() or "debug"

    def _user_id(self) -> str:
        return self.user_var.get().strip() or "default"

    def new_session(self) -> None:
        self.session_var.set(datetime.now().strftime("chat-%Y%m%d-%H%M%S"))
        self.clear_chat_display()
        self.refresh_memory_view(switch_page=False)
        self.status_var.set("已新建会话")

    def send_query(self) -> None:
        query = self.query_text.get("1.0", tk.END).strip()
        if not query:
            messagebox.showwarning("缺少输入", "请输入一句用户问题。")
            return
        self.query_text.delete("1.0", tk.END)
        self._append_chat("用户", query, "user")
        self.streaming_response_active = False
        self.status_var.set("正在等待模型响应，会消耗 API 额度...")
        self.send_button.configure(state=tk.DISABLED)
        threading.Thread(
            target=self._send_worker,
            args=(query, self._session_id(), self._user_id()),
            daemon=True,
        ).start()

    def _send_worker(self, query: str, session_id: str, user_id: str) -> None:
        try:
            skill = self._make_skill()
            streamed = False

            def on_delta(delta: str) -> None:
                nonlocal streamed
                streamed = True
                self.result_queue.put(("delta", delta))

            result = skill.reply(
                query,
                session_id=session_id,
                user_id=user_id,
                on_delta=on_delta,
            )
            store = skill.memory_store
            summary = store.get_summary(session_id) if store else None
            recent = store.get_recent_messages(session_id, self.recent_var.get()) if store else []
            long_term_memories = store.get_active(user_id, limit=20) if store else []
            self.result_queue.put(
                (
                    "reply",
                    {
                        "result": result.to_dict(),
                        "summary": summary.summary if summary else "",
                        "recent": recent,
                        "long_term_memories": long_term_memories,
                        "streamed": streamed,
                    },
                )
            )
        except Exception as exc:
            self.result_queue.put(("error", str(exc)))

    def reset_session(self) -> None:
        session_id = self._session_id()
        if not messagebox.askyesno("确认重置", f"确定清空会话 {session_id} 吗？"):
            return
        try:
            skill = self._make_skill()
            skill.reset_session(session_id)
            self.clear_chat_display()
            store = skill.memory_store
            long_term_memories = store.get_active(self._user_id(), limit=20) if store else []
            self._render_memory("", [], long_term_memories)
            self.status_var.set(f"已重置会话：{session_id}")
        except Exception as exc:
            messagebox.showerror("重置失败", str(exc))

    def refresh_memory_view(self, switch_page: bool = True) -> None:
        try:
            session_id = self._session_id()
            skill = self._make_skill()
            store = skill.memory_store
            summary = store.get_summary(session_id) if store else None
            recent = store.get_recent_messages(session_id, self.recent_var.get()) if store else []
            long_term_memories = store.get_active(self._user_id(), limit=20) if store else []
            self._render_memory(summary.summary if summary else "", recent, long_term_memories)
            self.status_var.set("已刷新")
            if switch_page:
                self._show_page("memory")
        except Exception as exc:
            messagebox.showerror("刷新失败", str(exc))

    def clear_chat_display(self) -> None:
        self.chat_text.delete("1.0", tk.END)

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.result_queue.get_nowait()
                if kind == "delta":
                    if not self.streaming_response_active:
                        self.chat_text.insert(tk.END, "\n助手\n", "assistant")
                        self.streaming_response_active = True
                    self.chat_text.insert(tk.END, str(payload), "assistant")
                    self.chat_text.see(tk.END)
                    self.status_var.set("正在生成回复")
                    continue

                if kind == "background":
                    event = payload
                    if (
                        event.get("session_id") == self._session_id()
                        and event.get("user_id") == self._user_id()
                    ):
                        self._refresh_memory_without_switch()
                    summary_ms = event.get("summary_latency_ms", 0)
                    memory_ms = event.get("memory_latency_ms", 0)
                    total_ms = event.get("total_latency_ms", 0)
                    self.chat_text.insert(
                        tk.END,
                        (
                            f"后台记忆更新完成：摘要 {summary_ms}ms | "
                            f"长期记忆 {memory_ms}ms | 总耗时 {total_ms}ms\n"
                        ),
                        "meta",
                    )
                    if event.get("error"):
                        self.chat_text.insert(
                            tk.END,
                            f"后台记忆错误：{event.get('error')}\n",
                            "meta",
                        )
                    self.chat_text.see(tk.END)
                    self.status_var.set("完成")
                    continue

                self.send_button.configure(state=tk.NORMAL)
                if kind == "reply":
                    data = payload
                    result = data["result"]
                    answer = result.get("answer") or ""
                    if data.get("streamed"):
                        self.chat_text.insert(tk.END, "\n", "assistant")
                    else:
                        self._append_chat("助手", answer or "[空回复]", "assistant")
                    self.streaming_response_active = False
                    self._append_meta(result)
                    self._render_memory(
                        data["summary"],
                        data["recent"],
                        data["long_term_memories"],
                    )
                    if result.get("summary_pending") or result.get("long_term_memory_pending"):
                        self.status_var.set("回答完成，停止输入 10 秒后更新记忆")
                    else:
                        self.status_var.set("完成")
                else:
                    self.streaming_response_active = False
                    self.status_var.set("失败")
                    messagebox.showerror("调用失败", str(payload))
        except queue.Empty:
            pass
        self.root.after(200, self._poll_queue)

    def _append_chat(self, role: str, content: str, tag: str) -> None:
        self.chat_text.insert(tk.END, f"\n{role}\n", tag)
        self.chat_text.insert(tk.END, f"{content}\n", "assistant")
        self.chat_text.see(tk.END)

    def _append_meta(self, result: dict[str, object]) -> None:
        meta = (
            f"场景：{result.get('scene', '')} | "
            f"模式：{result.get('history_mode', '')} | "
            f"摘要：{'使用' if result.get('summary_used') else '未使用'} | "
            f"长期记忆：{result.get('long_term_memories_used', 0)}条 | "
            f"分类：{result.get('classifier_latency_ms', 0)}ms | "
            f"首字：{result.get('first_token_latency_ms', 0) or 0}ms | "
            f"生成：{result.get('generation_latency_ms', 0)}ms | "
            f"总耗时：{result.get('latency_ms', 0)}ms"
        )
        self.chat_text.insert(tk.END, f"{meta}\n", "meta")
        if result.get("summary_updated"):
            self.chat_text.insert(tk.END, "本轮已更新会话摘要。\n", "meta")
        if result.get("long_term_memory_updated"):
            self.chat_text.insert(tk.END, "本轮已更新长期记忆。\n", "meta")
        if result.get("error"):
            self.chat_text.insert(tk.END, f"错误：{result.get('error')}\n", "meta")
        if result.get("memory_error"):
            self.chat_text.insert(tk.END, f"记忆错误：{result.get('memory_error')}\n", "meta")
        self.chat_text.see(tk.END)

    def _refresh_memory_without_switch(self) -> None:
        try:
            session_id = self._session_id()
            skill = self._make_skill()
            store = skill.memory_store
            summary = store.get_summary(session_id) if store else None
            recent = store.get_recent_messages(session_id, self.recent_var.get()) if store else []
            long_term_memories = store.get_active(self._user_id(), limit=20) if store else []
            self._render_memory(summary.summary if summary else "", recent, long_term_memories)
        except Exception:
            pass

    def _render_memory(
        self,
        summary: str,
        recent: list[object],
        long_term_memories: list[object],
    ) -> None:
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, summary or "暂无摘要")

        self.long_term_memory_text.delete("1.0", tk.END)
        if not long_term_memories:
            self.long_term_memory_text.insert(tk.END, "暂无长期记忆")
        else:
            for item in long_term_memories:
                self.long_term_memory_text.insert(
                    tk.END,
                    (
                        f"[{item.id}] {item.memory_type} | "
                        f"priority={item.priority} | confidence={item.confidence:.2f}\n"
                        f"{item.content}\n"
                        f"证据：{item.evidence or '无'}\n\n"
                    ),
                )

        self.history_text.delete("1.0", tk.END)
        if not recent:
            self.history_text.insert(tk.END, "暂无历史")
            return
        for item in recent:
            role = "用户" if item.role == "user" else "助手"
            self.history_text.insert(tk.END, f"{role}：{item.content}\n\n")


def main() -> None:
    root = tk.Tk()
    MultiTurnDebugGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
