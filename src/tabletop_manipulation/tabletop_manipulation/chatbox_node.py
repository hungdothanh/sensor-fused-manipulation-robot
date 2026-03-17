#!/usr/bin/env python3
"""
chatbox_node.py  —  UR5 Pick & Place  ·  Robot Control Interface
"""

import json
import os
import threading
import time
import tkinter as tk

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String

try:
    from PIL import Image as PILImage, ImageTk
    _PIL_OK = True
except ImportError:
    _PIL_OK = False


# ── Layout constants ───────────────────────────────────────────────────────────
COL_W     = 500   # all 3 columns start at this width and grow equally
CAM_COL_W = COL_W
DATA_W    = COL_W
CHAT_MIN  = COL_W

# ── LLM backend defaults ───────────────────────────────────────────────────────
# Override via ROS launch parameters (openai_model / ollama_model / openai_api_key)
LLM_OPENAI_MODEL = 'gpt-4o'
LLM_OLLAMA_URL   = 'http://localhost:11434/api/chat'
LLM_TEMPERATURE  = 0.7
LLM_TIMEOUT_S    = 60

# ── Ground truth centroids (Gazebo spawn poses, world frame) ───────────────────
# Cylinder geometry: radius=0.03 m, length=0.10 m — SDF pose = geometric centre
# TABLE_Z = 0.765 m  →  centroid Z = 0.765 + 0.05 = 0.815 m
GT_CENTROIDS = {
    'red_cylinder':   [0.65,  0.10, 0.815],
    'green_cylinder': [0.75, -0.25, 0.815],
}


# ── LLM system prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a friendly assistant for an industrial conveyor pick-and-place robot system.

The robot is a UR5 arm. Its task is to pick colored cylinders from a conveyor table
and sort them into box containers.

Scene:
  - Conveyor table with colored cylinders: red cylinder, green cylinder
  - Left box  — container on the LEFT side of the table
  - Right box — container on the RIGHT side of the table

Always respond with a JSON object containing exactly these two keys:
  "reply"         – your conversational response shown to the user (always required)
  "robot_command" – if the user wants the robot to pick/place a cylinder,
                    copy their EXACT words verbatim as the command string.
                    For general chat or questions, set this to null.

CRITICAL: robot_command must be the user's EXACT phrasing — never paraphrase,
summarise, or change any word. The downstream planner depends on every word.

Examples
--------
User: "What can you do?"
→ {"reply": "I can pick cylinders from the conveyor and sort them into the left or right box!", "robot_command": null}

User: "Move the red cylinder to the right box."
→ {"reply": "On it!", "robot_command": "move the red cylinder to the right box"}

Output nothing except the JSON object.\
"""


# ── Colour palette — clean light industrial ────────────────────────────────────
C = {
    # Page / panel backgrounds
    'bg':           '#eef2f7',   # light blue-gray page
    'bg_header':    '#0f172a',   # dark navy header (intentional contrast)
    'bg_card':      '#ffffff',   # white card surface
    'bg_card_alt':  '#f8fafc',   # very slightly tinted alternate card
    'bg_entry':     '#f1f5f9',   # input field
    'bg_btn':       '#2563eb',   # primary button
    'bg_btn_hl':    '#1d4ed8',
    'bg_btn_dis':   '#94a3b8',

    # Borders / separators
    'border':       '#d1dae8',
    'sep':          '#e8eef5',
    'sep_hdr':      '#1e3a5f',   # header separator

    # Accent colours (card top-bar + icon colours)
    'blue':         '#2563eb',
    'blue_lt':      '#3b82f6',
    'cyan':         '#0891b2',
    'cyan_lt':      '#0ea5e9',
    'green':        '#16a34a',
    'green_lt':     '#22c55e',
    'yellow':       '#b45309',   # amber — readable on white
    'yellow_lt':    '#d97706',
    'orange':       '#c2410c',   # readable on white
    'orange_lt':    '#ea580c',
    'red':          '#dc2626',
    'teal':         '#0d9488',

    # Text
    'fg':           '#0f172a',   # primary (near-black)
    'fg_sec':       '#374151',   # secondary (dark gray)
    'fg_dim':       '#6b7280',   # muted
    'fg_muted':     '#9ca3af',   # very muted / placeholder
}

ACCENT_CHAT  = C['blue']
ACCENT_SCENE = C['cyan']
ACCENT_WRIST = C['green']
ACCENT_CENT  = C['yellow']
ACCENT_TRAJ  = C['orange']


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class ChatboxNode(Node):

    def __init__(self):
        super().__init__('chatbox_node')

        self.declare_parameter('openai_api_key',
                               os.environ.get('OPENAI_API_KEY', ''))
        self.declare_parameter('openai_model', LLM_OPENAI_MODEL)
        self.declare_parameter('ollama_model', '')

        self._api_key = self.get_parameter('openai_api_key').value
        self._model   = self.get_parameter('openai_model').value
        self._ollama  = self.get_parameter('ollama_model').value

        self._pub_cmd = self.create_publisher(String, '/manipulation/command', 10)

        self.on_status     = None
        self.on_grasp_plan = None

        self.create_subscription(String, '/manipulation/task_status',    self._task_cb, 10)
        self.create_subscription(String, '/manipulation/commander_status', self._cmd_cb,  10)

        self._frame_lock          = threading.Lock()
        self._scene_frame         = None
        self._wrist_frame         = None
        self._fusion_debug_frame  = None

        self.create_subscription(RosImage, '/scene_camera/image_raw',
                                 self._scene_cb, qos_profile_sensor_data)
        self.create_subscription(RosImage, '/camera/color/image_raw',
                                 self._wrist_cb, qos_profile_sensor_data)
        self.create_subscription(RosImage, '/perception/fusion_debug',
                                 self._fusion_debug_cb, qos_profile_sensor_data)

        self._data_lock         = threading.Lock()
        self._fused_results_raw = None
        self._grasp_plan_raw    = None

        self.create_subscription(String, '/perception/lidar_fused_results',
                                 self._fused_results_cb, 10)
        self.create_subscription(String, '/manipulation/grasp_plan',
                                 self._grasp_plan_cb, 10)

        backend = self._model if self._api_key else (self._ollama or 'NOT CONFIGURED')
        self.get_logger().info(f'Chatbox node ready. LLM backend: {backend}')

    def _scene_cb(self, msg):
        with self._frame_lock:
            self._scene_frame = msg

    def _wrist_cb(self, msg):
        with self._frame_lock:
            self._wrist_frame = msg

    def _fusion_debug_cb(self, msg):
        with self._frame_lock:
            self._fusion_debug_frame = msg

    def _fused_results_cb(self, msg):
        with self._data_lock:
            self._fused_results_raw = msg.data

    def _grasp_plan_cb(self, msg):
        with self._data_lock:
            self._grasp_plan_raw = msg.data
        if self.on_grasp_plan:
            self.on_grasp_plan(msg.data)

    def _task_cb(self, msg):
        self._forward_status(msg.data, 'ROBOT')

    def _cmd_cb(self, msg):
        self._forward_status(msg.data, 'PIPELINE')

    def _forward_status(self, raw, prefix):
        if not self.on_status:
            return
        try:
            status = json.loads(raw).get('status', raw)
        except Exception:
            status = raw
        self.on_status(f'[{prefix}]  {status}')

    def publish_command(self, command):
        self._pub_cmd.publish(String(data=command))
        self.get_logger().info(f'Published command: "{command}"')

    def call_llm(self, conversation):
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}] + conversation
        if self._api_key:
            return self._openai(messages)
        if self._ollama:
            return self._ollama_call(messages)
        self.get_logger().error('No LLM backend configured.')
        return None

    def _openai(self, messages):
        try:
            import openai
            client = openai.OpenAI(api_key=self._api_key)
            resp = client.chat.completions.create(
                model=self._model, messages=messages,
                temperature=LLM_TEMPERATURE,
                response_format={'type': 'json_object'})
            return json.loads(resp.choices[0].message.content)
        except Exception as exc:
            self.get_logger().error(f'OpenAI error: {exc}')
            return None

    def _ollama_call(self, messages):
        try:
            import requests
            resp = requests.post(
                LLM_OLLAMA_URL,
                json={'model': self._ollama, 'messages': messages,
                      'stream': False, 'format': 'json'},
                timeout=LLM_TIMEOUT_S)
            resp.raise_for_status()
            return json.loads(resp.json()['message']['content'])
        except Exception as exc:
            self.get_logger().error(f'Ollama error: {exc}')
            return None


# ── GUI ───────────────────────────────────────────────────────────────────────

class ChatboxApp:

    def __init__(self, node: ChatboxNode):
        self._node         = node
        self._conversation = []
        self._busy         = False

        self._root = tk.Tk()
        self._root.title('UR5 Pick & Place  ·  Robot Control Interface')
        self._root.configure(bg=C['bg'])
        self._root.geometry(f'{CHAT_MIN + CAM_COL_W + DATA_W + 40}x940')
        self._root.minsize(960, 680)
        self._root.wm_attributes('-zoomed', True)   # start maximised

        self._scene_photo  = None
        self._wrist_photo  = None
        self._fusion_photo = None

        self._build_ui()

        self._node.on_status     = lambda s: self._root.after(0, self._append_status, s)
        self._node.on_grasp_plan = lambda d: self._root.after(0, self._update_data_panels, d)

        # Delay first camera refresh so widgets are rendered and have real sizes
        self._root.after(300, self._refresh_cameras)
        self._root.after(1000, self._tick_clock)

        self._append_bot(
            "System online.\n"
            "I can move cylinders between the conveyor table and the storage boxes.\n"
            "Tell me which cylinder to pick and where to place it.")

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        root = self._root

        # ── Top header bar (dark) ──────────────────────────────────────────
        header = tk.Frame(root, bg=C['bg_header'], height=46)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        lf = tk.Frame(header, bg=C['bg_header'])
        lf.pack(side=tk.LEFT, padx=(16, 0), fill=tk.Y)
        tk.Label(lf, text='⬡', bg=C['bg_header'], fg=C['cyan_lt'],
                 font=('Helvetica', 14, 'bold')).pack(side=tk.LEFT)
        tk.Label(lf, text='  UR5 PICK & PLACE',
                 bg=C['bg_header'], fg='#f1f5f9',
                 font=('Helvetica', 13, 'bold')).pack(side=tk.LEFT)
        tk.Label(lf, text='  ·  ROBOT CONTROL INTERFACE',
                 bg=C['bg_header'], fg='#64748b',
                 font=('Helvetica', 12)).pack(side=tk.LEFT)

        rf = tk.Frame(header, bg=C['bg_header'])
        rf.pack(side=tk.RIGHT, padx=(0, 18), fill=tk.Y)
        for txt, col in [('● ROS2', C['green_lt']),
                          ('● DINO', C['cyan_lt']),
                          ('● LIDAR', C['blue_lt'])]:
            tk.Label(rf, text=txt, bg=C['bg_header'], fg=col,
                     font=('Helvetica', 12, 'bold')).pack(side=tk.LEFT, padx=(0, 14))
        tk.Frame(rf, bg=C['sep_hdr'], width=1, height=20).pack(
            side=tk.LEFT, padx=(0, 14))
        self._clock_lbl = tk.Label(rf, text='--:--:--',
                                    bg=C['bg_header'], fg='#64748b',
                                    font=('Courier', 11))
        self._clock_lbl.pack(side=tk.LEFT)

        # Thin blue accent rule under header
        tk.Frame(root, bg=C['blue'], height=2).pack(fill=tk.X)

        # ── Bottom input bar ───────────────────────────────────────────────
        inp_wrap = tk.Frame(root, bg=C['border'])
        inp_wrap.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Frame(inp_wrap, bg=C['border'], height=1).pack(fill=tk.X)

        inp = tk.Frame(inp_wrap, bg=C['bg_card'], padx=10, pady=10)
        inp.pack(fill=tk.X)
        inp.columnconfigure(0, weight=1)

        tk.Label(inp, text='▸  COMMAND INPUT',
                 bg=C['bg_card'], fg=C['fg_dim'],
                 font=('Helvetica', 12, 'bold')).grid(
                     row=0, column=0, sticky='w', pady=(0, 4))

        self._entry = tk.Text(
            inp, height=2, width=1,   # width=1 collapses natural minimum
            bg=C['bg_entry'], fg=C['fg'],
            relief=tk.FLAT, bd=1,
            font=('Helvetica', 13),
            insertbackground=C['blue'],
            wrap=tk.WORD, padx=8, pady=6,
        )
        self._entry.grid(row=1, column=0, sticky='ew', padx=(0, 10))
        self._entry.bind('<Return>', self._on_return)

        self._btn = tk.Button(
            inp, text='SEND  ▶',
            bg=C['bg_btn'], fg='#ffffff',
            relief=tk.FLAT, bd=0,
            font=('Helvetica', 12, 'bold'),
            padx=18, pady=10,
            cursor='hand2',
            activebackground=C['bg_btn_hl'],
            activeforeground='#ffffff',
            command=self._send,
        )
        self._btn.grid(row=1, column=1, sticky='ns')

        tk.Label(inp, text='Enter → send   ·   Shift+Enter → new line',
                 bg=C['bg_card'], fg=C['fg_muted'],
                 font=('Helvetica', 12)).grid(
                     row=2, column=0, sticky='w', pady=(4, 0))

        # ── Body: 3-column grid (all same height) ─────────────────────────
        body = tk.Frame(root, bg=C['bg'])
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Grid: 3 columns, 1 row — uniform='cols' forces identical widths regardless of content
        body.columnconfigure(0, weight=1, minsize=CHAT_MIN,  uniform='cols')
        body.columnconfigure(1, weight=1, minsize=CAM_COL_W, uniform='cols')
        body.columnconfigure(2, weight=1, minsize=DATA_W,    uniform='cols')
        body.rowconfigure(0, weight=1)

        # ── LEFT: chat panel ───────────────────────────────────────────────
        left = self._build_chat_column(body)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 5))

        # ── MIDDLE: camera feeds ───────────────────────────────────────────
        mid = tk.Frame(body, bg=C['bg'])
        mid.grid(row=0, column=1, sticky='nsew', padx=(0, 5))
        mid.rowconfigure(0, weight=1)     # scene cam — equal share
        mid.rowconfigure(1, weight=0)     # gap
        mid.rowconfigure(2, weight=1)     # wrist cam — equal share
        mid.columnconfigure(0, weight=1)

        scene_outer, self._scene_label = self._build_camera_card(
            mid, 'SCENE CAM', ACCENT_SCENE)
        scene_outer.grid(row=0, column=0, sticky='nsew')

        tk.Frame(mid, bg=C['bg'], height=6).grid(row=1, column=0)

        wrist_outer, self._wrist_label = self._build_camera_card(
            mid, 'WRIST CAM  ·  RAW RGB', ACCENT_WRIST)
        wrist_outer.grid(row=2, column=0, sticky='nsew')

        # ── RIGHT column: 50/50 vertical split ────────────────────────────
        #   row 0 (weight=1) — scrollable data cards (centroid + trajectory)
        #   row 1 (weight=0) — 6 px gap
        #   row 2 (weight=1) — fusion debug camera  [exact 50 % of body]
        right = tk.Frame(body, bg=C['bg'])
        right.grid(row=0, column=2, sticky='nsew')
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)
        right.rowconfigure(2, weight=1, minsize=30)
        right.columnconfigure(0, weight=1)

        # ── Row 0: scrollable centroid + trajectory cards ──────────────────
        # height=1 collapses the canvas's own minimum so the weight=1 on both
        # rows distributes space equally (50/50) with the fusion camera below.
        data_canvas = tk.Canvas(right, bg=C['bg'], highlightthickness=0, bd=0,
                                height=1)
        data_canvas.grid(row=0, column=0, sticky='nsew')

        vsb = tk.Scrollbar(right, orient='vertical', command=data_canvas.yview)
        vsb.grid(row=0, column=1, sticky='ns')
        data_canvas.configure(yscrollcommand=vsb.set)

        inner = tk.Frame(data_canvas, bg=C['bg'])
        cw_id = data_canvas.create_window((0, 0), window=inner, anchor='nw')

        def _on_canvas_resize(event):
            data_canvas.itemconfig(cw_id, width=event.width)
        data_canvas.bind('<Configure>', _on_canvas_resize)

        def _on_inner_resize(event):
            data_canvas.configure(scrollregion=data_canvas.bbox('all'))
        inner.bind('<Configure>', _on_inner_resize)

        def _wheel(event):
            if event.num == 4:
                data_canvas.yview_scroll(-1, 'units')
            elif event.num == 5:
                data_canvas.yview_scroll(1, 'units')
            else:
                data_canvas.yview_scroll(int(-1 * event.delta / 120), 'units')
        data_canvas.bind('<MouseWheel>', _wheel)
        data_canvas.bind('<Button-4>', _wheel)
        data_canvas.bind('<Button-5>', _wheel)

        self._centroid_text = self._build_data_card(
            inner,
            title='3D CENTROID ANALYSIS', icon='◈', accent=ACCENT_CENT,
            tags={
                'lbl':      {'foreground': C['fg_dim'],    'font': ('Courier', 11, 'bold')},
                'val':      {'foreground': C['fg'],        'font': ('Courier', 12)},
                'dino':     {'foreground': C['yellow'],    'font': ('Courier', 12, 'bold')},
                'fused':    {'foreground': C['green'],     'font': ('Courier', 12, 'bold')},
                'gt':       {'foreground': C['cyan'],      'font': ('Courier', 12, 'bold')},
                'err_base': {'foreground': C['orange'],    'font': ('Courier', 12, 'bold')},
                'err_prop': {'foreground': C['green_lt'],  'font': ('Courier', 12, 'bold')},
                'good':     {'foreground': C['green'],     'font': ('Courier', 13, 'bold')},
                'bad':      {'foreground': C['red'],       'font': ('Courier', 13, 'bold')},
                'dim':      {'foreground': C['fg_muted'],  'font': ('Courier', 11)},
                'tag':      {'foreground': C['cyan'],      'font': ('Courier', 12, 'bold')},
            },
            placeholder='Awaiting pick command …',
        )

        tk.Frame(inner, bg=C['bg'], height=8).pack(fill=tk.X)

        self._moveit_text = self._build_data_card(
            inner,
            title='MOTION TRAJECTORY', icon='◈', accent=ACCENT_TRAJ,
            tags={
                'lbl':   {'foreground': C['fg_dim'],     'font': ('Courier', 11, 'bold')},
                'coord': {'foreground': C['orange'],     'font': ('Courier', 12, 'bold')},
                'step':  {'foreground': C['blue'],       'font': ('Courier', 11, 'bold')},
                'dim':   {'foreground': C['fg_muted'],   'font': ('Courier', 11)},
            },
            placeholder='Awaiting grasp plan …',
        )

        # ── Row 1: gap ─────────────────────────────────────────────────────
        tk.Frame(right, bg=C['bg'], height=6).grid(
            row=1, column=0, columnspan=2, sticky='ew')

        # ── Row 2: fusion debug camera — 50 % of right column ─────────────
        # Built inline (not via _build_camera_card) so we can add
        # pack_propagate(False) on the card, preventing incoming images from
        # ever changing the card's requested size and breaking the 50/50 split.
        fusion_outer = tk.Frame(right, bg=C['border'])
        fusion_outer.grid(row=2, column=0, columnspan=2, sticky='nsew')

        fusion_card = tk.Frame(fusion_outer, bg=C['bg_card'])
        fusion_card.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        fusion_card.pack_propagate(False)   # ← image cannot resize this card

        tk.Frame(fusion_card, bg=C['teal'], height=2).pack(fill=tk.X)
        fhdr = tk.Frame(fusion_card, bg=C['bg_card'])
        fhdr.pack(fill=tk.X, padx=8, pady=(5, 4))
        tk.Label(fhdr, text='◉  WRIST CAM  ·  FUSION INSIGHT',
                 bg=C['bg_card'], fg=C['teal'],
                 font=('Helvetica', 12, 'bold')).pack(side=tk.LEFT)
        tk.Label(fhdr, text='● LIVE',
                 bg=C['bg_card'], fg=C['green'],
                 font=('Helvetica', 9, 'bold')).pack(side=tk.RIGHT)
        tk.Frame(fusion_card, bg=C['sep'], height=1).pack(fill=tk.X)

        self._fusion_label = tk.Label(fusion_card, bg='#1a1a2e')
        self._fusion_label.pack(fill=tk.BOTH, expand=True)

    # ── Widget builders ───────────────────────────────────────────────────

    def _build_chat_column(self, parent):
        outer = tk.Frame(parent, bg=C['border'], bd=0)
        card  = tk.Frame(outer, bg=C['bg_card'])
        card.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        # Accent bar
        tk.Frame(card, bg=ACCENT_CHAT, height=2).pack(fill=tk.X)

        # Header
        hdr = tk.Frame(card, bg=C['bg_card'])
        hdr.pack(fill=tk.X, padx=10, pady=(6, 4))
        tk.Label(hdr, text='◈  COMMAND INTERFACE',
                 bg=C['bg_card'], fg=C['fg_dim'],
                 font=('Helvetica', 12, 'bold')).pack(side=tk.LEFT)
        self._online_dot = tk.Label(hdr, text='● ONLINE',
                                     bg=C['bg_card'], fg=C['green'],
                                     font=('Helvetica', 9, 'bold'))
        self._online_dot.pack(side=tk.RIGHT)
        tk.Frame(card, bg=C['sep'], height=1).pack(fill=tk.X)

        # Chat text area
        chat_frame = tk.Frame(card, bg=C['bg_card'])
        chat_frame.pack(fill=tk.BOTH, expand=True)

        self._chat = tk.Text(
            chat_frame,
            bg=C['bg_card'], fg=C['fg'],
            relief=tk.FLAT, bd=0,
            padx=12, pady=8,
            wrap=tk.WORD,
            width=1,          # collapse natural width; grid/uniform controls the column
            state=tk.DISABLED,
            font=('Helvetica', 13),
            cursor='arrow',
            spacing1=1, spacing3=3,
        )
        sb = tk.Scrollbar(chat_frame, command=self._chat.yview,
                          bg=C['bg_card'], troughcolor=C['sep'],
                          relief=tk.FLAT, bd=0, width=8)
        self._chat['yscrollcommand'] = sb.set
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._chat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._chat.tag_configure('label_user', foreground=C['blue'],
                                  font=('Helvetica', 13, 'bold'))
        self._chat.tag_configure('text_user', foreground=C['fg'],
                                  font=('Helvetica', 13), lmargin1=14, lmargin2=14)
        self._chat.tag_configure('label_bot', foreground=C['cyan'],
                                  font=('Helvetica', 13, 'bold'))
        self._chat.tag_configure('text_bot', foreground=C['fg_sec'],
                                  font=('Helvetica', 13), lmargin1=14, lmargin2=14)
        self._chat.tag_configure('cmd_hint', foreground=C['orange'],
                                  font=('Helvetica', 12, 'italic'),
                                  lmargin1=14, lmargin2=14)
        self._chat.tag_configure('status', foreground=C['fg_dim'],
                                  font=('Helvetica', 11, 'italic'),
                                  lmargin1=14, lmargin2=14)
        self._chat.tag_configure('error', foreground=C['red'],
                                  font=('Helvetica', 12, 'italic'),
                                  lmargin1=14, lmargin2=14)
        return outer

    def _build_camera_card(self, parent, title, accent):
        """
        Returns (outer_frame, image_label).
        The image_label expands dynamically to fill the card's remaining height.
        """
        outer = tk.Frame(parent, bg=C['border'])

        card = tk.Frame(outer, bg=C['bg_card'])
        card.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        card.pack_propagate(False)  # grid controls size, not the image label

        # Accent bar
        tk.Frame(card, bg=accent, height=2).pack(fill=tk.X)

        # Header row
        hdr = tk.Frame(card, bg=C['bg_card'])
        hdr.pack(fill=tk.X, padx=8, pady=(5, 4))
        tk.Label(hdr, text=f'◉  {title}',
                 bg=C['bg_card'], fg=accent,
                 font=('Helvetica', 12, 'bold')).pack(side=tk.LEFT)
        tk.Label(hdr, text='● LIVE',
                 bg=C['bg_card'], fg=C['green'],
                 font=('Helvetica', 9, 'bold')).pack(side=tk.RIGHT)
        tk.Frame(card, bg=C['sep'], height=1).pack(fill=tk.X)

        # Image label — fills remaining card height (no fixed height!)
        img_lbl = tk.Label(card, bg='#1a1a2e')
        img_lbl.pack(fill=tk.BOTH, expand=True)

        return outer, img_lbl

    def _build_data_card(self, parent, title, icon, accent, tags, placeholder):
        """Styled card with a scrollable Text widget. Returns the Text widget."""
        outer = tk.Frame(parent, bg=C['border'])
        outer.pack(fill=tk.X, padx=0, pady=0)

        card = tk.Frame(outer, bg=C['bg_card_alt'])
        card.pack(fill=tk.X, padx=1, pady=1)

        # Accent bar
        tk.Frame(card, bg=accent, height=2).pack(fill=tk.X)

        # Header
        hdr = tk.Frame(card, bg=C['bg_card_alt'])
        hdr.pack(fill=tk.X, padx=10, pady=(6, 4))
        tk.Label(hdr, text=f'{icon}  {title}',
                 bg=C['bg_card_alt'], fg=accent,
                 font=('Helvetica', 12, 'bold')).pack(side=tk.LEFT)
        tk.Frame(card, bg=C['sep'], height=1).pack(fill=tk.X)

        # Text widget (no fixed height — expands with content)
        txt_frame = tk.Frame(card, bg=C['bg_card_alt'])
        txt_frame.pack(fill=tk.X)

        txt = tk.Text(
            txt_frame,
            bg=C['bg_card_alt'], fg=C['fg'],
            relief=tk.FLAT, bd=0,
            font=('Courier', 12),
            state=tk.DISABLED,
            padx=10, pady=8,
            wrap=tk.WORD,
            height=1,          # minimum; expands via _set_text_height
        )
        for tag_name, opts in tags.items():
            txt.tag_configure(tag_name, **opts)
        txt.pack(fill=tk.X, expand=False)

        txt.config(state=tk.NORMAL)
        txt.insert(tk.END, placeholder, 'dim')
        txt.config(state=tk.DISABLED)
        self._auto_height(txt)

        return txt

    @staticmethod
    def _auto_height(txt: tk.Text):
        """Set text widget height to exactly fit its content (no extra blank lines)."""
        lines = int(txt.index('end-1c').split('.')[0])
        txt.config(height=max(lines, 1))

    # ── Clock ─────────────────────────────────────────────────────────────

    def _tick_clock(self):
        self._clock_lbl.config(text=time.strftime('%H:%M:%S'))
        self._root.after(1000, self._tick_clock)

    # ── Camera refresh (dynamic resize to fit card) ───────────────────────

    def _refresh_cameras(self):
        if _PIL_OK:
            with self._node._frame_lock:
                scene_msg  = self._node._scene_frame
                wrist_msg  = self._node._wrist_frame
                fusion_msg = self._node._fusion_debug_frame

            if scene_msg is not None:
                w = self._scene_label.winfo_width()
                h = self._scene_label.winfo_height()
                if w > 10 and h > 10:
                    photo = self._ros_to_photoimage(scene_msg, w, h)
                    if photo:
                        self._scene_photo = photo
                        self._scene_label.config(image=photo)

            if wrist_msg is not None:
                w = self._wrist_label.winfo_width()
                h = self._wrist_label.winfo_height()
                if w > 10 and h > 10:
                    photo = self._ros_to_photoimage(wrist_msg, w, h)
                    if photo:
                        self._wrist_photo = photo
                        self._wrist_label.config(image=photo)

            if fusion_msg is not None:
                w = self._fusion_label.winfo_width()
                h = self._fusion_label.winfo_height()
                if w > 10 and h > 10:
                    photo = self._ros_to_photoimage(fusion_msg, w, h)
                    if photo:
                        self._fusion_photo = photo
                        # Do NOT pass width/height — pack_propagate(False) on
                        # the card keeps the cell size fixed regardless of image.
                        self._fusion_label.config(image=photo)

        self._root.after(100, self._refresh_cameras)

    def _ros_to_photoimage(self, msg: RosImage, w: int, h: int):
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            img  = data.reshape((msg.height, msg.width, -1))
            enc  = msg.encoding.lower()
            if enc in ('bgr8', 'bgr', 'bgra8'):
                img = img[:, :, :3][..., ::-1].copy()
            else:
                img = img[:, :, :3]
            pil = PILImage.fromarray(img, 'RGB')
            pil = pil.resize((w, h), PILImage.LANCZOS)
            return ImageTk.PhotoImage(pil)
        except Exception:
            return None

    # ── Data panel updates ────────────────────────────────────────────────

    def _update_data_panels(self, plan_json: str):
        try:
            plan = json.loads(plan_json)
        except Exception:
            return

        obj_key = plan.get('object', '')

        # ── 3D Centroid ───────────────────────────────────────────────────
        with self._node._data_lock:
            fused_raw = self._node._fused_results_raw

        dino_xyz = fused_xyz = None
        source = 'unknown'

        if fused_raw:
            try:
                for det in json.loads(fused_raw).get('detections', []):
                    if det.get('label', '').strip().replace(' ', '_') == obj_key:
                        dino_xyz  = det.get('dino_centroid_world')
                        fused_xyz = det.get('centroid_3d_world')
                        source    = det.get('source', 'unknown')
                        break
            except Exception:
                pass

        t = self._centroid_text
        t.config(state=tk.NORMAL)
        t.delete('1.0', tk.END)

        t.insert(tk.END, '  TARGET   ', 'lbl')
        t.insert(tk.END, obj_key.upper().replace('_', ' ') + '\n\n', 'val')

        t.insert(tk.END, '  CAMERA   ', 'lbl')
        if dino_xyz:
            t.insert(tk.END,
                f'X {dino_xyz[0]:+.4f}  Y {dino_xyz[1]:+.4f}  Z {dino_xyz[2]:+.4f}\n',
                'dino')
        else:
            t.insert(tk.END, 'unavailable\n', 'dim')

        src_tag = {'lidar_fusion': 'LIDAR-R', 'lidar_depth': 'LIDAR-D'}.get(source, 'LIDAR  ')
        t.insert(tk.END, f'  {src_tag}   ', 'lbl')
        if fused_xyz:
            t.insert(tk.END,
                f'X {fused_xyz[0]:+.4f}  Y {fused_xyz[1]:+.4f}  Z {fused_xyz[2]:+.4f}\n',
                'fused')
        else:
            t.insert(tk.END, 'unavailable\n', 'dim')

        if dino_xyz and fused_xyz:
            def _dist3(a, b):
                return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

            method = {'lidar_fusion': 'RANSAC CIRCLE',
                      'lidar_depth':  'LIDAR DEPTH+PX',
                      'dino_only':    'CAMERA ONLY'}.get(source, source.upper())

            gt = GT_CENTROIDS.get(obj_key)
            if gt:
                baseline_err = _dist3(dino_xyz, gt)
                proposed_err = _dist3(fused_xyz, gt)
                improvement  = ((baseline_err - proposed_err) / baseline_err * 100
                                if baseline_err > 0 else 0.0)

                t.insert(tk.END, '  GROUND T  ', 'lbl')
                t.insert(tk.END,
                    f'X {gt[0]:+.4f}  Y {gt[1]:+.4f}  Z {gt[2]:+.4f}\n\n', 'gt')

                t.insert(tk.END, '  BASE ERR  ', 'lbl')
                t.insert(tk.END, f'{baseline_err * 100:.2f} cm', 'err_base')
                t.insert(tk.END, '  [CAMERA]\n', 'dim')

                t.insert(tk.END, '  PROP ERR  ', 'lbl')
                t.insert(tk.END, f'{proposed_err * 100:.2f} cm', 'err_prop')
                t.insert(tk.END, f'  [{method}]\n\n', 'dim')

                imp_tag = 'good' if improvement >= 0 else 'bad'
                t.insert(tk.END, '  IMPROVE   ', 'lbl')
                t.insert(tk.END, f'{improvement:+.1f} %\n', imp_tag)
            else:
                # No GT entry for this object — fall back to inter-method delta
                dist = _dist3(fused_xyz, dino_xyz)
                t.insert(tk.END, '\n  Δ DIST    ', 'lbl')
                t.insert(tk.END, f'{dist * 100:.1f} cm  ', 'dim')
                t.insert(tk.END, f'[{method}]', 'tag')

        t.config(state=tk.DISABLED)
        self._auto_height(t)

        # ── Trajectory ────────────────────────────────────────────────────
        pick_xyz  = plan.get('pick_xyz',  [0, 0, 0])
        place_xyz = plan.get('place_xyz', [0, 0, 0])
        hover_z   = plan.get('place_approach_z', 0.0)
        APPROACH_H = 0.25

        approach  = [pick_xyz[0],  pick_xyz[1],  round(pick_xyz[2] + APPROACH_H, 4)]
        place_hov = [place_xyz[0], place_xyz[1], round(hover_z, 4)]

        def fmt(xyz):
            return f'[{xyz[0]:+.4f},  {xyz[1]:+.4f},  {xyz[2]:+.4f}]'

        m = self._moveit_text
        m.config(state=tk.NORMAL)
        m.delete('1.0', tk.END)

        m.insert(tk.END, '  TARGET   ', 'lbl')
        m.insert(tk.END, obj_key.upper().replace('_', ' ') + '\n\n', 'dim')

        for num, label, xyz in [
            ('01', 'APPROACH ', approach),
            ('02', 'PICK     ', pick_xyz),
            ('03', 'RETREAT  ', approach),
            ('04', 'HOVER    ', place_hov),
            ('05', 'PLACE    ', place_xyz),
        ]:
            m.insert(tk.END, f'  {num} ', 'step')
            m.insert(tk.END, f'{label} ', 'lbl')
            m.insert(tk.END, fmt(xyz) + '\n', 'coord')

        m.config(state=tk.DISABLED)
        self._auto_height(m)

    # ── User input ────────────────────────────────────────────────────────

    def _on_return(self, event):
        if event.state & 0x1:
            return
        self._send()
        return 'break'

    def _send(self):
        text = self._entry.get('1.0', tk.END).strip()
        if not text or self._busy:
            return
        self._entry.delete('1.0', tk.END)
        self._append_user(text)
        self._conversation.append({'role': 'user', 'content': text})
        self._set_busy(True)
        threading.Thread(target=self._llm_thread, daemon=True).start()

    def _llm_thread(self):
        result = self._node.call_llm(self._conversation)
        self._root.after(0, self._on_llm_result, result)

    def _on_llm_result(self, result):
        self._set_busy(False)
        if result is None:
            self._append_error('LLM call failed — check API key or Ollama connection.')
            return
        reply = result.get('reply', '(no reply)')
        cmd   = result.get('robot_command')
        self._conversation.append({'role': 'assistant', 'content': reply})
        self._append_bot(reply)
        if cmd:
            self._append_cmd(f'Dispatching:  "{cmd}"')
            self._node.publish_command(cmd)

    # ── Chat helpers ──────────────────────────────────────────────────────

    def _write(self, *segments):
        self._chat.config(state=tk.NORMAL)
        for text, tag in segments:
            self._chat.insert(tk.END, text, tag)
        self._chat.config(state=tk.DISABLED)
        self._chat.see(tk.END)

    def _append_user(self, text):
        self._write(('  YOU\n', 'label_user'), (text + '\n\n', 'text_user'))

    def _append_bot(self, text):
        self._write(('  SYSTEM\n', 'label_bot'), (text + '\n\n', 'text_bot'))

    def _append_cmd(self, text):
        self._write(('  ↳ ' + text + '\n\n', 'cmd_hint'))

    def _append_status(self, text):
        self._write((text + '\n', 'status'))

    def _append_error(self, text):
        self._write(('  ⚠  ' + text + '\n\n', 'error'))

    def _set_busy(self, busy: bool):
        self._busy = busy
        if busy:
            self._btn.config(state=tk.DISABLED, text='WAIT  …',
                             bg=C['bg_btn_dis'], fg='#ffffff')
            self._online_dot.config(text='● PROCESSING', fg=C['yellow_lt'])
        else:
            self._btn.config(state=tk.NORMAL, text='SEND  ▶',
                             bg=C['bg_btn'], fg='#ffffff')
            self._online_dot.config(text='● ONLINE', fg=C['green'])

    def run(self):
        self._root.mainloop()


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ChatboxNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        app = ChatboxApp(node)
        app.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
