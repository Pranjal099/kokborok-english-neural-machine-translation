import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import time
from pynput import keyboard
from pynput.keyboard import Controller, Key


class AutoTyperGUI:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Ultra Type Pro ⌨️")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        self.root.configure(bg="#f0f4ff")  # Royal white background

        # --- State ---
        self.is_typing = False
        self.stop_event = threading.Event()
        self.typing_speed = 10
        self.keyboard_controller = Controller()
        self.current_text_section = 1

        self.hotkey_listener = None
        self.current_keys = set()

        self.setup_styles()

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        # ---------- Header ----------
        header_frame = ttk.Frame(self.root, style="Header.TFrame")
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        ttk.Label(
            header_frame,
            text="Ultra Type Pro",
            style="Header.TLabel"
        ).pack(anchor="center", pady=8)

        # ---------- Speed Control ----------
        settings_frame = ttk.Frame(self.root, style="Settings.TFrame")
        settings_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        settings_frame.columnconfigure(1, weight=1)

        ttk.Label(settings_frame, text="Typing Delay (ms):").grid(row=0, column=0, sticky="w", padx=5)

        self.speed_var = tk.IntVar(value=self.typing_speed)
        ttk.Scale(
            settings_frame,
            from_=0,
            to=200,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
            command=self.on_speed_change
        ).grid(row=0, column=1, sticky="ew", padx=5)

        self.speed_label = ttk.Label(settings_frame, text=f"{self.typing_speed} ms / char")
        self.speed_label.grid(row=0, column=2, sticky="w", padx=5)

        # ---------- Notebook ----------
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)

        self.text_areas = []
        for i, key in enumerate(["X", "Y", "Z"]):
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=f"Section {i+1} (Cmd/Ctrl+Shift+{key})")

            text_area = scrolledtext.ScrolledText(
                frame,
                wrap=tk.WORD,
                font=("SF Pro Text", 12),
                bg="#ffffff",
                fg="#0f172a",
                insertbackground="#1d4ed8",
                padx=12,
                pady=12,
                relief=tk.FLAT,
                borderwidth=1
            )
            text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            self.text_areas.append(text_area)

        # Default text
        self.text_areas[0].insert(tk.END, "This is sample text for Section 1.")
        self.text_areas[1].insert(tk.END, "Enter text for Section 2 here.")
        self.text_areas[2].insert(tk.END, "Use hotkeys or buttons to auto type this text.")

        # ---------- Buttons ----------
        control_frame = ttk.Frame(self.root)
        control_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=10)

        for i in range(4):
            control_frame.columnconfigure(i, weight=1)

        ttk.Button(control_frame, text="Type Section 1",
                   command=lambda: self.start_typing(1),
                   style="Action.TButton").grid(row=0, column=0, padx=5, sticky="ew")

        ttk.Button(control_frame, text="Type Section 2",
                   command=lambda: self.start_typing(2),
                   style="Action.TButton").grid(row=0, column=1, padx=5, sticky="ew")

        ttk.Button(control_frame, text="Type Section 3",
                   command=lambda: self.start_typing(3),
                   style="Action.TButton").grid(row=0, column=2, padx=5, sticky="ew")

        ttk.Button(control_frame, text="Stop Typing (Esc)",
                   command=self.stop_typing,
                   style="Stop.TButton").grid(row=0, column=3, padx=5, sticky="ew")

        # ---------- Status Bar ----------
        self.status_var = tk.StringVar(
            value="Ready | Start: Cmd/Ctrl+Shift+X/Y/Z | Stop: Esc or Cmd/Ctrl+Shift+0"
        )

        status_bar = ttk.Frame(self.root, style="Status.TFrame")
        status_bar.grid(row=4, column=0, sticky="ew", padx=10, pady=5)
        status_bar.columnconfigure(0, weight=1)

        ttk.Label(status_bar, textvariable=self.status_var,
                  style="Status.TLabel").grid(row=0, column=0, sticky="w", padx=5)

        self.progress = ttk.Progressbar(status_bar, mode="indeterminate")
        self.progress.grid(row=0, column=1, sticky="e", padx=5)

        # ---------- Footer ----------
        footer_frame = ttk.Frame(self.root, style="Footer.TFrame")
        footer_frame.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 6))

        ttk.Label(
            footer_frame,
            text="Developed by Pranjal Gautam",
            font=("SF Pro Text", 9, "italic"),
            foreground="#1d4ed8",
            background="#f0f4ff"
        ).pack(anchor="center")

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.setup_global_hotkeys()
        self.center_window()

    # ---------- Styles ----------
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".", background="#f0f4ff", foreground="#0f172a")

        style.configure("Header.TFrame", background="#1d4ed8")
        style.configure("Header.TLabel",
                        background="#1d4ed8",
                        foreground="white",
                        font=("SF Pro Display", 18, "bold"))

        style.configure("Settings.TFrame", background="#f0f4ff")
        style.configure("Footer.TFrame", background="#f0f4ff")

        style.configure("Action.TButton",
                        background="#2563eb",
                        foreground="white",
                        font=("SF Pro Text", 10, "bold"),
                        padding=10)
        style.map("Action.TButton",
                  background=[("active", "#1e40af")])

        style.configure("Stop.TButton",
                        background="#dc2626",
                        foreground="white",
                        font=("SF Pro Text", 10, "bold"),
                        padding=10)
        style.map("Stop.TButton",
                  background=[("active", "#b91c1c")])

        style.configure("Status.TFrame", background="#e0e7ff")
        style.configure("Status.TLabel",
                        background="#e0e7ff",
                        foreground="#1e3a8a",
                        font=("SF Pro Text", 9))

    # ---------- Logic ----------
    def center_window(self):
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def on_speed_change(self, value):
        self.typing_speed = int(float(value))
        self.speed_label.config(text=f"{self.typing_speed} ms / char")

    def start_typing(self, section):
        if self.is_typing:
            self.update_status("⚠️ Already typing")
            return
        self.current_text_section = section
        threading.Thread(target=self.run_typing_logic, daemon=True).start()

    def stop_typing(self):
        if self.is_typing:
            self.stop_event.set()
            self.update_status("🛑 Stopping...")

    def run_typing_logic(self):
        self.is_typing = True
        self.stop_event.clear()

        text = self.text_areas[self.current_text_section - 1].get("1.0", tk.END).strip()
        if not text:
            self.update_status("Section is empty")
            self.is_typing = False
            return

        self.root.after(0, self.progress.start)
        for i in range(3, 0, -1):
            if self.stop_event.is_set():
                self.cleanup()
                return
            self.update_status(f"Starting in {i}...")
            time.sleep(1)

        self.update_status("⌨️ Typing...")
        for ch in text:
            if self.stop_event.is_set():
                break
            self.keyboard_controller.type(ch)
            time.sleep(self.typing_speed / 1000)

        self.update_status("✅ Completed" if not self.stop_event.is_set() else "🛑 Stopped")
        self.cleanup()

    def cleanup(self):
        self.is_typing = False
        self.stop_event.clear()
        self.root.after(0, self.progress.stop)

    def update_status(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))

    # ---------- Hotkeys ----------
    def setup_global_hotkeys(self):
        self.hotkey_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        self.hotkey_listener.start()

    def on_key_press(self, key):
        if key == Key.esc:
            self.stop_typing()
            return
        self.current_keys.add(key)
        self.check_hotkeys()

    def on_key_release(self, key):
        self.current_keys.discard(key)

    def check_hotkeys(self):
        ctrl = Key.ctrl_l in self.current_keys or Key.ctrl_r in self.current_keys
        cmd = Key.cmd in self.current_keys
        shift = Key.shift_l in self.current_keys or Key.shift_r in self.current_keys

        if (ctrl or cmd) and shift:
            if keyboard.KeyCode.from_char('x') in self.current_keys:
                self.root.after(0, self.start_typing, 1)
            elif keyboard.KeyCode.from_char('y') in self.current_keys:
                self.root.after(0, self.start_typing, 2)
            elif keyboard.KeyCode.from_char('z') in self.current_keys:
                self.root.after(0, self.start_typing, 3)
            elif keyboard.KeyCode.from_char('0') in self.current_keys:
                self.root.after(0, self.stop_typing)

    def on_closing(self):
        if self.is_typing:
            self.stop_event.set()
        if self.hotkey_listener:
            self.hotkey_listener.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = AutoTyperGUI(root)
    root.mainloop()
