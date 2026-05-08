import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
from PIL import Image, ImageTk
import base64
import os
import shutil
import threading
import queue
from datetime import datetime
import subprocess
import platform
from openai import OpenAI
import glob
import re
import json
import time
from StateMachine.workflow_engine import EvacWorkflowManager
from RAG.rag_engine import CodeRAG
from config import CHAT_API_KEY, CHAT_API_BASE, CHAT_MODEL, CONCLUSION_MODEL

# ==================== Global Font Settings ====================
FONT_FAMILY = "Arial"
FONT_SIZE_NORMAL = 14    # Normal text size (chat box, input box)
FONT_SIZE_SMALL = 12     # Small text size (status labels, buttons)
FONT_SIZE_LARGE = 14     # Large text size (if needed)

# Font style combinations (ready to use)
FONT_NORMAL = (FONT_FAMILY, FONT_SIZE_NORMAL)
FONT_NORMAL_BOLD = (FONT_FAMILY, FONT_SIZE_NORMAL, "bold")
FONT_SMALL = (FONT_FAMILY, FONT_SIZE_SMALL)
FONT_SMALL_BOLD = (FONT_FAMILY, FONT_SIZE_SMALL, "bold")
FONT_LARGE = (FONT_FAMILY, FONT_SIZE_LARGE)
FONT_LARGE_BOLD = (FONT_FAMILY, FONT_SIZE_LARGE, "bold")
# =====================================================

class ImageChatApp:
    def __init__(self, root=None):
        self.root = root or tk.Tk()
        self.root.title("ChatEvac")
        self.root.geometry("900x1200")
        self.root.minsize(600, 400)

        # OpenAI API key
        # Parameter configuration mapping
        self.parameter_mapping = {
            "space width": "--space-width",
            "space height": "--space-height", 
            "width": "--space-width",
            "height": "--space-height",
            "scene width": "--space-width",
            "scene height": "--space-height",
            "people": "--num-people",
            "pedestrian": "--num-people",
            "person": "--num-people",
            "number of people": "--num-people",
            "pedestrian number": "--num-people",
            "max speed": "--max-speed-factor",
            "maximum speed": "--max-speed-factor",
            "speed factor": "--max-speed-factor",
            "radius": "--people-radius-base",
            "pedestrian radius": "--people-radius-base",
            "people radius": "--people-radius-base",
            "delta time": "--delta-time",
            "time step": "--delta-time",
            "mass": "--people-mass-base",
            "people mass": "--people-mass-base",
            "pedestrian mass": "--people-mass-base",
            "repulsion": "--social-force-a1",
            "people repulsion": "--social-force-a1",
            "pedestrian repulsion": "--social-force-a1",
            "people-people": "--social-force-a1",
            "pedestrian-pedestrian": "--social-force-a1",
            "wall repulsion": "--social-force-a2",
            "people-wall": "--social-force-a2",
            "pedestrian-wall": "--social-force-a2"
        }
        
        # Default parameter values and types
        self.default_parameters = {
            "--space-width": 30.0,
            "--space-height": 30.0,
            "--num-people": 30,
            "--max-speed-factor": 1.2,
            "--people-radius-base": 0.25,  # meters (helbing engine uses meters directly)
            "--delta-time": 0.5,
            "--people-mass-base": 50,
            "--social-force-a1": 2000,
            "--social-force-a2": 2000
        }
        
        # Parameter type definitions (for correct type conversion)
        self.parameter_types = {
            "--space-width": float,
            "--space-height": float,
            "--num-people": int,
            "--max-speed-factor": float,
            "--people-radius-base": float,
            "--delta-time": float,
            "--people-mass-base": int,
            "--social-force-a1": int,
            "--social-force-a2": int
        }
        
        # Current parameter configuration
        self.current_parameters = self.default_parameters.copy()

        # System prompt: describes role and workflow control mechanism.
        # Specific state instructions are dynamically injected by workflow_engine.
        self.system_prompt = """You are an AI assistant specialized in building evacuation safety assessment. You help users conduct simulation evaluations of architectural floor plans through a structured workflow.

WORKFLOW CONTROL MECHANISM:
- The workflow is managed by a state machine. Each state has a single-letter symbol (A through O).
- At the end of each turn, you will receive a [WORKFLOW STATE CONTEXT] block describing your current state and available transitions.
- When the user's intent matches a transition condition, include the corresponding tag in your response: <WORKFLOW_ACTION>SYMBOL</WORKFLOW_ACTION>
- Use ONLY the single-letter symbols listed in the context block. Do not invent new symbols or use old action names.
- Evaluate user intent SEMANTICALLY — do not rely on exact keyword matching. Understand what the user means, not just what they literally say.
- You may include at most ONE workflow action tag per response.

PARAMETER CONFIGURATION:
- Supported parameters: space width (m), space height (m), number of people, max speed factor, pedestrian radius (m), delta time (s), people mass (kg), people-people repulsion, people-wall repulsion.
- Default values: width=30m, height=30m, people=30, speed=1.34, radius=0.25m, dt=0.01s, mass=80kg, repulsion=2000, wall-repulsion=2000.
- When the user provides numbers, extract and apply them. When no numbers are given, proceed with defaults.

GENERAL GUIDELINES:
- Keep responses concise and professional.
- Focus on evacuation safety assessment tasks.
- After the conclusion is generated, answer any follow-up questions about the analysis."""

        # Chat history and threading queue
        self.result_queue = queue.Queue()
        self.chat_history = []
        self.current_image_b64 = None
        self.current_image_path = None
        self.thumbnail_ref = None
        self.chat_image_refs = []
        
        # Counter for tracking images in chat
        self.image_counter = 0
        self._image_tag_paths = {}  # image_tag -> image_path

        # Program path configuration
        _here = os.path.dirname(os.path.abspath(__file__))

        # Workflow state machine engine
        self.workflow_mgr = EvacWorkflowManager(os.path.join(_here, "StateMachine", "workflow_config.json"))

        # RAG building code knowledge base engine
        self.rag_engine = CodeRAG(
            api_key=CHAT_API_KEY,
            api_base=CHAT_API_BASE
        )
        # Only trigger RAG retrieval in these conversational states (execution states D/G/J don't need it)
        self.rag_active_states = {
            "awaiting_parameter_config",   # E — User may ask about parameter standards
            "awaiting_data_analysis",      # I — User may ask about analysis standards
            "data_analysis_completed",     # K — User may ask about result interpretation
            "awaiting_conclusion",         # L — User may ask about report format
            "generating_conclusion",       # M — Generating report needs code references
            "conclusion_ready",            # N — User asks follow-up about analysis results
            "completed",                   # O — Consultation after workflow completion
        }
        self.controlnet_script = os.path.join(_here, "use_controlnet.py")
        self.launch_script = os.path.join(_here, "launch.py")
        self.analysis_script = os.path.join(_here, "analysis_gui.py")
        self.input_image_path = os.path.join(_here, "Material", "Input.png")
        self.output_image_path = os.path.join(_here, "Material", "Output.png")

        # Data analysis related paths
        self.data_dir = os.path.join(_here, "Material", "data")
        self.congestion_image = os.path.join(self.data_dir, "congestion.png")
        self.curve_image = os.path.join(self.data_dir, "curve.png")
        self.evacuation_time_file = os.path.join(self.data_dir, "EvacT.txt")

        # Prepare Material folder
        self.material_dir = os.path.join(_here, "Material")
        if os.path.exists(self.material_dir):
            shutil.rmtree(self.material_dir)
        os.makedirs(self.material_dir)

        # Build GUI
        self._build_ui()
        self.add_message("AI Helper", "Welcome! I will assist you to conduct the evacuation assessment. Please send me the floorplan that you want to assess.")
        self._poll_results()

    @property
    def workflow_state(self):
        return self.workflow_mgr.current_state

    @workflow_state.setter
    def workflow_state(self, value):
        self.workflow_mgr.force_transition_to(value)

    def _encode_image_to_base64(self, image_path):
        """Encode a local image to base64 format"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def _build_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=0)
        main_frame.rowconfigure(2, weight=0)
        main_frame.columnconfigure(0, weight=1)

        # Chat display
        chat_frame = tk.Frame(main_frame)
        chat_frame.grid(row=0, column=0, sticky='nsew', pady=(0,10))
        chat_frame.rowconfigure(0, weight=1)
        chat_frame.columnconfigure(0, weight=1)

        self.chat_box = scrolledtext.ScrolledText(
            chat_frame, 
            state=tk.DISABLED, 
            wrap=tk.WORD,
            bg="#f5f5f5",
            font=FONT_NORMAL
        )
        self.chat_box.grid(row=0, column=0, sticky='nsew')

        # Widget-level click/motion bindings — work even when widget is DISABLED
        self.chat_box.bind("<Button-1>", self._on_chat_click)
        self.chat_box.bind("<Motion>", self._on_chat_motion)

        # Workflow status display
        status_frame = tk.Frame(main_frame)
        status_frame.grid(row=1, column=0, sticky='ew', pady=(0,10))
        
        tk.Label(status_frame, text="Workflow Status:", font=FONT_SMALL_BOLD).pack(side=tk.LEFT)
        self.workflow_status_var = tk.StringVar()
        self.workflow_status_var.set(self.workflow_mgr.get_status_text())
        tk.Label(status_frame, textvariable=self.workflow_status_var, 
                fg="blue", font=FONT_SMALL).pack(side=tk.LEFT, padx=(10,0))

        # Status label
        self.status_label = tk.Label(
            main_frame, 
            text="No image selected", 
            bg="#e0e0e0",
            font=FONT_SMALL,
            height=2
        )
        self.status_label.grid(row=2, column=0, sticky='ew', pady=(0,10))

        # Input area
        input_frame = tk.Frame(main_frame)
        input_frame.grid(row=3, column=0, sticky='ew')
        input_frame.columnconfigure(0, weight=1)

        # Text input
        self.text_input = tk.Text(
            input_frame, 
            height=3,
            wrap=tk.WORD,
            font=FONT_NORMAL
        )
        self.text_input.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 5))

        # Thumbnail preview
        self.thumb_label = tk.Label(input_frame)
        self.thumb_label.grid(row=1, column=0, sticky='w', pady=5)

        # Buttons
        btn_frame = tk.Frame(input_frame)
        btn_frame.grid(row=2, column=0, columnspan=3, sticky='ew')
        
        self.select_btn = tk.Button(
            btn_frame, text="Select Image", command=self.select_image,
            width=12, bg="#FF9800", fg="white", font=FONT_SMALL_BOLD
        )
        self.select_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.clear_btn = tk.Button(
            btn_frame, text="Clear Image", command=self.clear_image,
            width=12, bg="#f44336", fg="white", font=FONT_SMALL_BOLD
        )
        self.clear_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.view_btn = tk.Button(
            btn_frame, text="View Image", command=self.view_image, state=tk.DISABLED,
            width=12, bg="#2196F3", fg="white", font=FONT_SMALL_BOLD
        )
        self.view_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.send_btn = tk.Button(
            btn_frame, text="Send", command=self.send,
            width=12, bg="#4CAF50", fg="white", font=FONT_SMALL_BOLD
        )
        self.send_btn.pack(side=tk.RIGHT)

        # Progress label
        self.progress_var = tk.StringVar()
        self.progress_label = tk.Label(btn_frame, textvariable=self.progress_var, fg="blue", font=FONT_SMALL)
        self.progress_label.pack(side=tk.RIGHT, padx=(0, 10))

        # Bind Ctrl+Enter to send
        self.text_input.bind("<Control-Return>", lambda e: self.send())

    def update_workflow_status(self, new_state):
        """Update workflow status"""
        old_state = self.workflow_mgr.current_state
        if not self.workflow_mgr.transition_to(new_state):
            self.workflow_mgr.force_transition_to(new_state)
        self.workflow_status_var.set(self.workflow_mgr.get_status_text())
        print(f"Workflow state changed: {old_state} -> {new_state}")

    def parse_ai_workflow_actions(self, ai_response):
        """Parse workflow action directives from AI response"""
        actions = []
        pattern = r'<WORKFLOW_ACTION>(.*?)</WORKFLOW_ACTION>'
        matches = re.findall(pattern, ai_response, re.IGNORECASE)
        
        for match in matches:
            action = match.strip().upper()
            actions.append(action)
            print(f"AI requested workflow action: {action}")
        
        return actions

    def parse_parameter_modifications(self, user_input):
        """Parse parameter modification requests from user input"""
        modifications = {}
        text = user_input.lower()

        # First check if the input contains numbers; if not, use defaults
        import re
        has_numbers = bool(re.search(r'\d+', text))
        if not has_numbers:
            return {}  # Return empty dict to indicate using default parameters

        # Find numbers and related parameters
        number_patterns = [
            r'(\w+)\s*[=:]\s*(\d+\.?\d*)',  # param = value
            r'(\d+\.?\d*)\s*(\w+)',         # value unit
            r'(\w+)\s+(\d+\.?\d*)',         # param value
            r'(\d+\.?\d*)\s*for\s+(\w+)',   # value for param
        ]
        
        # Extract all numbers and related words
        found_params = {}
        
        for pattern in number_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 2:
                    word1, word2 = match
                    try:
                        # Try to determine which is the number
                        if '.' in word1 or word1.isdigit():
                            value = float(word1)
                            param_word = word2
                        elif '.' in word2 or word2.isdigit():
                            value = float(word2)
                            param_word = word1
                        else:
                            continue
                        
                        # Find matching parameter
                        for key, param in self.parameter_mapping.items():
                            if key in param_word or param_word in key:
                                found_params[param] = value
                                break
                    except ValueError:
                        continue
        
        # Special handling for common expressions
        special_patterns = {
            r'(\d+\.?\d*)\s*m(?:eter)?s?\s*width': '--space-width',
            r'(\d+\.?\d*)\s*m(?:eter)?s?\s*height': '--space-height',
            r'(\d+)\s*people': '--num-people',
            r'(\d+)\s*person': '--num-people',
            r'(\d+\.?\d*)\s*speed': '--max-speed-factor',
            r'(\d+\.?\d*)\s*radius': '--people-radius-base',
            r'(\d+\.?\d*)\s*mass': '--people-mass-base',
            r'(\d+)\s*kg': '--people-mass-base'
        }
        
        for pattern, param in special_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                try:
                    value = float(matches[0])
                    found_params[param] = value
                except ValueError:
                    continue
        
        # Handle unit conversion
        final_params = {}
        for param, value in found_params.items():
            if param == '--people-radius-base':
                # If user input is in meters, convert to pixels (assuming 0.35m = 35 pixels)
                if value < 5:  # Likely in meters
                    value = value * 100  # Simple conversion
            final_params[param] = value
        
        return final_params

    def execute_workflow_actions(self, actions):
        """Execute symbolic workflow actions (A-O) output by LLM. Z is a null action — no state transition."""
        for symbol in actions:
            symbol = symbol.strip().upper()

            # Z = null action — LLM chooses not to advance workflow, keep current state
            if self.workflow_mgr.is_null_symbol(symbol):
                print(f"[Agent] Null action Z — state unchanged: {self.workflow_mgr.current_state}")
                continue
            target_state = self.workflow_mgr.get_state_by_symbol(symbol)
            if not target_state:
                print(f"[Agent] Unknown workflow symbol: '{symbol}'")
                continue

            print(f"[Agent] Executing symbol {symbol} -> {target_state}")

            if target_state == "running_feature_extraction":
                self.update_workflow_status("running_feature_extraction")
                self.result_queue.put(("start_feature_extraction", ""))

            elif target_state == "awaiting_parameter_config":
                self.update_workflow_status("awaiting_parameter_config")

            elif target_state == "running_simulation":
                # Unified handling: check if the most recent user input contains parameters; if so use them, otherwise use defaults
                last_user_input = ""
                for msg in reversed(self.chat_history):
                    if msg["role"] == "user":
                        last_user_input = msg["content"] if isinstance(msg["content"], str) else ""
                        break
                param_modifications = self.parse_parameter_modifications(last_user_input)
                if param_modifications:
                    self.current_parameters.update(param_modifications)
                    print(f"[Agent] Updated parameters: {param_modifications}")
                self.update_workflow_status("running_simulation")
                self.result_queue.put(("start_simulation", self.current_parameters))

            elif target_state == "awaiting_data_analysis":
                self.update_workflow_status("awaiting_data_analysis")

            elif target_state == "running_data_analysis":
                self.update_workflow_status("running_data_analysis")
                self.result_queue.put(("start_data_analysis", ""))

            elif target_state == "awaiting_conclusion":
                self.update_workflow_status("awaiting_conclusion")

            elif target_state == "generating_conclusion":
                self.update_workflow_status("generating_conclusion")
                self.result_queue.put(("generate_conclusion", ""))

            elif target_state == "completed":
                self.update_workflow_status("completed")
                self.result_queue.put(("workflow_complete", ""))

            else:
                # Pure state update, no subprocess needed
                self.update_workflow_status(target_state)

    def clean_ai_response_for_display(self, ai_response):
        """Clean AI response by removing workflow control tags"""
        # Remove workflow action tags
        cleaned = re.sub(r'<WORKFLOW_ACTION>.*?</WORKFLOW_ACTION>', '', ai_response, flags=re.IGNORECASE)
        # Clean up extra blank lines
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned).strip()
        return cleaned

    def copy_image_to_target_location(self, source_path):
        """Copy image to the designated input location"""
        try:
            target_dir = os.path.dirname(self.input_image_path)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            with Image.open(source_path) as img:
                if 'icc_profile' in img.info:
                    del img.info['icc_profile']
                
                if img.mode in ('RGBA', 'LA', 'P'):
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    if img.mode in ('RGBA', 'LA'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'RGBA':
                            background.paste(img, mask=img.split()[-1])
                        else:
                            background.paste(img)
                        img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img.save(self.input_image_path, 'PNG', optimize=True)
                print(f"Image copied to target location: {self.input_image_path}")
                
        except Exception as e:
            print(f"Error copying image to target location: {e}")
            raise

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files","*.png *.jpg *.jpeg *.bmp *.gif *.tiff")])
        if not path: 
            return
            
        try:
            # Copy to both material folder and target location
            self.copy_image_to_material(path)
            self.copy_image_to_target_location(path)
            
            # Use the local copy for base64 encoding
            local_copy = os.path.join(self.material_dir, "Input.png")
            with open(local_copy, 'rb') as f:
                self.current_image_b64 = base64.b64encode(f.read()).decode()
            
            self.current_image_path = path
            self.status_label.config(text=f"Selected: {os.path.basename(path)}")
            self.view_btn.config(state=tk.NORMAL)
            self._show_thumbnail(local_copy)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.clear_image()

    def copy_image_to_material(self, source_path):
        """Copy the selected image to Material folder as Input.png"""
        try:
            target_path = os.path.join(self.material_dir, "Input.png")
            
            with Image.open(source_path) as img:
                if 'icc_profile' in img.info:
                    del img.info['icc_profile']
                
                if img.mode in ('RGBA', 'LA', 'P'):
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    if img.mode in ('RGBA', 'LA'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'RGBA':
                            background.paste(img, mask=img.split()[-1])
                        else:
                            background.paste(img)
                        img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img.save(target_path, 'PNG', optimize=True)
                print(f"Image copied to: {target_path}")
                
        except Exception as e:
            print(f"Error copying image to material folder: {e}")
            raise

    def _show_thumbnail(self, path):
        try:
            with Image.open(path) as img:
                img.thumbnail((100,100), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
            self.thumbnail_ref = photo
            self.thumb_label.config(image=photo)
        except Exception as e:
            print(f"Error showing thumbnail: {e}")
            self.thumb_label.config(image='')
            self.thumbnail_ref = None

    def clear_image(self):
        self.current_image_b64 = None
        self.current_image_path = None
        self.status_label.config(text="No image selected")
        self.view_btn.config(state=tk.DISABLED)
        self.thumb_label.config(image='')
        self.thumbnail_ref = None

    def view_image(self):
        img_path = os.path.join(self.material_dir, "Input.png")
        if not os.path.exists(img_path): 
            messagebox.showwarning("Warning", "No image found")
            return
            
        self._open_image_viewer(img_path, self.current_image_path)

    def _open_image_viewer(self, img_path, original_path=None):
        """Generic image viewer"""
        try:
            win = tk.Toplevel(self.root)
            win.title("Image Viewer")
            win.geometry("600x500")
            win.configure(bg="white")
            
            with Image.open(img_path) as img:
                img_copy = img.copy()
                original_name = os.path.basename(original_path) if original_path else os.path.basename(img_path)
                original_size = f"{img_copy.width} x {img_copy.height}"
                
                info_label = tk.Label(win, text=f"File: {original_name}\nSize: {original_size} pixels", 
                                    font=FONT_SMALL, bg="white")
                info_label.pack(pady=(10, 5))

                img_frame = tk.Frame(win, bg="white")
                img_frame.pack(expand=True, fill=tk.BOTH, pady=10)

                def resize_image(event=None):
                    frame_width = img_frame.winfo_width()
                    frame_height = img_frame.winfo_height()
                    
                    if frame_width > 1 and frame_height > 1:
                        img_copy = img.copy()
                        img_ratio = img_copy.width / img_copy.height
                        frame_ratio = frame_width / frame_height
                        
                        if img_ratio > frame_ratio:
                            new_width = frame_width - 20
                            new_height = int(new_width / img_ratio)
                        else:
                            new_height = frame_height - 20
                            new_width = int(new_height * img_ratio)
                        
                        if new_width > 0 and new_height > 0:
                            img_copy = img_copy.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img_copy)
                            lbl.configure(image=photo)
                            lbl.image = photo

                lbl = tk.Label(img_frame, bg="white", bd=2, relief="sunken")
                lbl.pack(expand=True, fill=tk.BOTH)

                img_frame.bind('<Configure>', resize_image)
                win.after(100, resize_image)
                
                btn_frame = tk.Frame(win, bg="white")
                btn_frame.pack(pady=10)
                
                close_btn = tk.Button(btn_frame, text="Close", command=win.destroy,
                                    width=10, bg="#f44336", fg="white", font=FONT_SMALL_BOLD)
                close_btn.pack(side=tk.LEFT, padx=5)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")

    def _on_image_click(self, event, image_path):
        """Handle image click events"""
        self._open_image_viewer(image_path)

    def _on_chat_click(self, event):
        """Widget-level click handler, works even when DISABLED"""
        idx = self.chat_box.index(f"@{event.x},{event.y}")
        for tag in self.chat_box.tag_names(idx):
            if tag in self._image_tag_paths:
                self._open_image_viewer(self._image_tag_paths[tag])
                return

    def _on_chat_motion(self, event):
        """Update cursor style on mouse movement"""
        idx = self.chat_box.index(f"@{event.x},{event.y}")
        for tag in self.chat_box.tag_names(idx):
            if tag in self._image_tag_paths:
                self.chat_box.config(cursor="hand2")
                return
        self.chat_box.config(cursor="")

    def add_message(self, sender, text, image_thumbnail=None, image_path=None):
        """Add a message to the chat box, supports clickable image thumbnails"""
        self.chat_box.config(state=tk.NORMAL)
        ts = datetime.now().strftime("%H:%M:%S")
        
        if sender == "You":
            self.chat_box.insert(tk.END, f"\n[{ts}] {sender}: ", "user")
            self.chat_box.tag_configure("user", foreground="blue", font=FONT_NORMAL_BOLD)
        elif sender == "System":
            self.chat_box.insert(tk.END, f"\n[{ts}] {sender}: ", "system")
            self.chat_box.tag_configure("system", foreground="purple", font=FONT_NORMAL_BOLD)
        else:
            self.chat_box.insert(tk.END, f"\n[{ts}] {sender}: ", "ai")
            self.chat_box.tag_configure("ai", foreground="green", font=FONT_NORMAL_BOLD)
        
        self.chat_box.insert(tk.END, f"{text}\n")
        
        if image_thumbnail and image_path:
            # Save thumbnail reference
            self.chat_image_refs.append(image_thumbnail)

            # Create unique image tag
            self.image_counter += 1
            image_tag = f"image_{self.image_counter}"

            # Use "end-1c" to ensure exact alignment of get and insert positions
            img_pos = self.chat_box.index("end-1c")
            self.chat_box.image_create("end-1c", image=image_thumbnail, padx=4, pady=4)
            self.chat_box.tag_add(image_tag, img_pos, f"{img_pos}+1c")
            self.chat_box.tag_configure(image_tag, relief="raised", borderwidth=1, background="#e8f4fd")
            # Store path in dict, looked up by widget-level click handler
            self._image_tag_paths[image_tag] = image_path

            self.chat_box.insert(tk.END, "\n")
        
        self.chat_box.see(tk.END)
        self.chat_box.config(state=tk.DISABLED)

    def send(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text and not self.current_image_b64:
            messagebox.showwarning("Empty", "Add text or image before sending.")
            return

        current_thumbnail = self.thumbnail_ref
        current_image_b64 = self.current_image_b64
        # Use locally copied image path
        current_image_path = os.path.join(self.material_dir, "Input.png") if self.current_image_b64 else None

        display_message = text if text else "[Image]"
        # Pass image path parameter
        self.add_message("You", display_message, current_thumbnail, current_image_path)

        self.text_input.delete("1.0", tk.END)
        self.clear_image()

        self.send_btn.config(state=tk.DISABLED)
        self.progress_var.set("Processing...")

        # Update state to processing
        if self.workflow_state == "idle":
            self.update_workflow_status("processing_initial")

        # Unified AI call
        threading.Thread(target=self._call_ai, args=(text, current_image_b64), daemon=True).start()

    def _call_ai(self, text, image_b64):
        """Call AI and execute workflow actions based on response"""
        try:
            client = OpenAI(base_url=CHAT_API_BASE, api_key=CHAT_API_KEY)

            # Inject current state's workflow context; LLM decides which symbol to output via semantic understanding
            current_prompt = self.system_prompt + self.workflow_mgr.get_prompt_injection(
                user_text=text or ""
            )

            # RAG: Retrieve relevant code provisions in conversational states and append to system prompt
            if self.workflow_mgr.current_state in self.rag_active_states:
                rag_block, _ = self.rag_engine.retrieve_and_format(
                    text if text else "", top_k=3
                )
                if rag_block:
                    current_prompt += rag_block

            messages = [{"role": "system", "content": current_prompt}]

            # Add chat history
            for msg in self.chat_history[-6:]:
                messages.append(msg)

            # Prepare current message
            if image_b64:
                content = [
                    {"type": "text", "text": text if text else "Please analyze this floor plan for evacuation assessment."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            else:
                content = text

            messages.append({"role": "user", "content": content})

            # Call API
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                max_tokens=1000
            )

            ai_message = response.choices[0].message.content
            print(f"DEBUG: AI response: {ai_message}")

            # Parse symbolic workflow actions from AI response
            workflow_actions = self.parse_ai_workflow_actions(ai_message)
            print(f"DEBUG: Parsed workflow symbols: {workflow_actions}")

            # Clean AI response for display
            clean_message = self.clean_ai_response_for_display(ai_message)

            # Update chat history
            self.chat_history.append({"role": "user", "content": text if text else "Please analyze this image"})
            self.chat_history.append({"role": "assistant", "content": clean_message})

            # Return result and actions
            self.result_queue.put(("ai_response", (clean_message, workflow_actions)))

        except Exception as e:
            self.result_queue.put(("error", str(e)))

    def _call_ai_for_conclusion(self):
        """Dedicated AI call for generating conclusions - uses local file reading"""
        try:
            client = OpenAI(base_url=CHAT_API_BASE, api_key=CHAT_API_KEY)

            # Read evacuation time data
            evacuation_time = "Unknown"
            try:
                if os.path.exists(self.evacuation_time_file):
                    with open(self.evacuation_time_file, 'r') as f:
                        evacuation_time = f.read().strip() + ' seconds'
                else:
                    evacuation_time = "File not found"
            except Exception as e:
                print(f"Error reading evacuation time: {e}")
                evacuation_time = f"Error reading file: {e}"
            
            # Build expert analysis prompt
            expert_prompt = f"""You are a professional safety engineer and emergency evacuation expert with years of experience in building safety assessment. Based on the provided evacuation simulation data and images, please provide a comprehensive analysis in the following structure:

(1) BUILDING LAYOUT ANALYSIS:
- Analyze the spatial structure based on the building layout image
- Black pixels represent indoor areas, white pixels represent walls, red pixels represent exits
- Evaluate the distribution and capacity of exits
- Identify potential bottlenecks in the layout

(2) TOTAL EVACUATION TIME:
- The total evacuation time from simulation is: {evacuation_time}
- Evaluate whether this time is acceptable for emergency evacuation
- Compare with safety standards and recommendations

(3) EVACUATION EFFICIENCY:
- Analyze the evacuation efficiency curve showing the relationship between evacuated people and time
- Identify critical time points and bottlenecks during evacuation
- Evaluate the smoothness of the evacuation process

(4) CONGESTION ANALYSIS:
- Analyze the congestion heatmap showing areas with longer congestion times
- Darker colors indicate longer congestion duration
- Identify high-risk areas and potential safety hazards
- Correlate congestion patterns with building layout features

(5) OPTIMIZATION RECOMMENDATIONS:
- Provide specific design improvements to enhance evacuation efficiency and safety
- Focus on structural modifications (exit placement, corridor width, emergency routes)
- Suggest operational improvements (guidance systems, evacuation procedures)
- Prioritize recommendations based on risk level and implementation feasibility

Please be detailed, professional, and provide actionable insights. Focus on practical solutions that can be implemented to improve building safety. Your analysis should be thorough and suitable for presentation to building owners, architects, and safety officials."""

            # Prepare image content
            content = [
                {"type": "text", "text": "Please analyze the following evacuation simulation results as a professional safety engineer:"}
            ]

            # Add four images
            images_to_analyze = [
                (self.output_image_path, "Building layout structure (black=indoor, white=walls, red=exits)"),
                (self.curve_image, "Evacuation efficiency curve (people vs time)"),
                (self.congestion_image, "Congestion heatmap (darker=longer congestion)")
            ]
            
            for img_path, description in images_to_analyze:
                if os.path.exists(img_path):
                    img_b64 = self._encode_image_to_base64(img_path)
                    if img_b64:
                        content.append({
                            "type": "text", 
                            "text": f"\n{description}:"
                        })
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        })
                    else:
                        content.append({
                            "type": "text", 
                            "text": f"\n{description}: [Failed to load image]"
                        })
                else:
                    content.append({
                        "type": "text", 
                        "text": f"\n{description}: [Image not found]"
                    })

            # RAG: Retrieve code provisions relevant to conclusion report (door width, exit capacity, travel distance, etc.)
            rag_query = (
                "door width exit capacity travel distance occupant load "
                "illumination emergency lighting stair width ramp slope corridor width "
                "number of exits exit discharge"
            )
            rag_block, _ = self.rag_engine.retrieve_and_format(rag_query, top_k=5)
            if rag_block:
                expert_prompt += "\n" + rag_block

            messages = [
                {"role": "system", "content": expert_prompt},
                {"role": "user", "content": content}
            ]

            # Call API
            response = client.chat.completions.create(
                model=CONCLUSION_MODEL,
                messages=messages,
                max_tokens=2000
            )

            conclusion = response.choices[0].message.content
            self.result_queue.put(("conclusion_ready", conclusion))

        except Exception as e:
            self.result_queue.put(("error", f"Failed to generate conclusion: {e}"))

    def _run_feature_extraction(self):
        """Run feature extraction program"""
        try:
            print(f"Running feature extraction: {self.controlnet_script}")
            result = subprocess.run(
                ['python', self.controlnet_script],
                capture_output=True,
                text=True,
                timeout=300  # 5-minute timeout
            )
            
            if result.returncode == 0:
                if os.path.exists(self.output_image_path):
                    self.result_queue.put(("feature_extraction_success", "Feature extraction completed successfully!"))
                else:
                    self.result_queue.put(("error", "Feature extraction completed but output image not found."))
            else:
                error_msg = f"Feature extraction failed: {result.stderr}"
                self.result_queue.put(("error", error_msg))
                
        except subprocess.TimeoutExpired:
            self.result_queue.put(("error", "Feature extraction timed out (>5 minutes)"))
        except Exception as e:
            self.result_queue.put(("error", f"Failed to run feature extraction: {e}"))

    def _run_simulation(self, parameters=None):
        """Run simulation program"""
        try:
            print(f"Running simulation: {self.launch_script}")

            # Build command-line arguments
            cmd = ['python', self.launch_script, '--engine', 'helbing']

            if parameters:
                for param, value in parameters.items():
                    # Perform correct type conversion based on parameter type
                    if param in self.parameter_types:
                        param_type = self.parameter_types[param]
                        if param_type == int:
                            # Ensure integer parameters get integer values
                            converted_value = int(float(value))  # Convert to float first then int, handles cases like '40.0'
                        elif param_type == float:
                            converted_value = float(value)
                        else:
                            converted_value = value
                    else:
                        converted_value = value
                    
                    cmd.extend([param, str(converted_value)])
                
                print(f"Using parameters: {parameters}")
                print(f"Command: {' '.join(cmd)}")
            
            # Set environment variables for encoding compatibility
            import os
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5-minute timeout
                env=env,
                encoding='utf-8',
                errors='ignore'
            )
            
            if result.returncode == 0:
                self.result_queue.put(("simulation_success", result.stdout or "Simulation completed successfully!"))
            else:
                error_msg = f"Simulation failed: {result.stderr}"
                self.result_queue.put(("error", error_msg))
                
        except subprocess.TimeoutExpired:
            self.result_queue.put(("error", "Simulation timed out (>5 minutes)"))
        except Exception as e:
            self.result_queue.put(("error", f"Failed to run simulation: {e}"))

    def _run_data_analysis(self):
        """Run data analysis program"""
        try:
            print(f"Running data analysis: {self.analysis_script}")

            # Ensure data directory exists
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)

            # Use Popen to launch subprocess without capturing output to avoid pipe blocking
            self.analysis_process = subprocess.Popen(
                ['python', self.analysis_script],
                # Remove stdout/stderr redirection so subprocess outputs directly to console
                # This avoids pipe buffer filling and causing deadlocks
            )

            # Return success immediately, don't wait for window to close
            self.result_queue.put(("data_analysis_started", "Data analysis GUI has been launched. Please close the window when done."))
                
        except Exception as e:
            self.result_queue.put(("error", f"Failed to run data analysis: {e}"))
            
    def _create_image_thumbnail(self, image_path):
        """Create image thumbnail"""
        try:
            if os.path.exists(image_path):
                with Image.open(image_path) as img:
                    # Create a copy of the image to avoid issues with the context manager
                    img_copy = img.copy()
                    img_copy.thumbnail((100,100), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img_copy)
                return photo
        except Exception as e:
            print(f"Error creating thumbnail for {image_path}: {e}")
        return None

    def _poll_results(self):
        try:
            while True:
                result_type, result = self.result_queue.get_nowait()
                
                if result_type == "ai_response":
                    clean_message, workflow_actions = result
                    self.add_message("AI Helper", clean_message)
                    
                    # Execute workflow actions requested by AI
                    if workflow_actions:
                        self.execute_workflow_actions(workflow_actions)
                    else:
                        # If no workflow actions, reset state to idle
                        if self.workflow_state == "processing_initial":
                            self.update_workflow_status("idle")
                
                elif result_type == "start_feature_extraction":
                    self.add_message("System", "Starting feature extraction...")
                    threading.Thread(target=self._run_feature_extraction, daemon=True).start()
                
                elif result_type == "feature_extraction_success":
                    self.add_message("System", result)
                    self.update_workflow_status("awaiting_parameter_config")

                    # Show output image thumbnail
                    output_thumbnail = self._create_image_thumbnail(self.output_image_path)
                    if output_thumbnail:
                        self.add_message("System", "Output image generated:", output_thumbnail, self.output_image_path)

                    # Pass user-visible default parameters to AI so it can present them naturally in its response
                    p = self.current_parameters
                    param_summary = (
                        f"Space width: {p['--space-width']} m, "
                        f"Space height: {p['--space-height']} m, "
                        f"Number of people: {p['--num-people']}, "
                        f"Max speed factor: {p['--max-speed-factor']} m/s, "
                        f"People mass: {p['--people-mass-base']} kg"
                    )
                    ai_prompt = (
                        f"Feature extraction completed successfully. "
                        f"Now inform the user that feature extraction is done and ask if they want to proceed with the evacuation simulation. "
                        f"Tell them the current default parameters are: {param_summary}. "
                        f"Ask if they want to modify any of these or proceed with defaults."
                    )
                    threading.Thread(target=self._call_ai, args=(ai_prompt, None), daemon=True).start()
                
                elif result_type == "start_simulation":
                    self.add_message("System", "Starting evacuation simulation...")
                    threading.Thread(target=self._run_simulation, args=(result,), daemon=True).start()
                
                elif result_type == "simulation_success":
                    self.add_message("System", f"Simulation completed!")
                    self.update_workflow_status("simulation_completed")
                    # Have AI ask about data analysis
                    threading.Thread(target=self._call_ai, args=("Simulation completed successfully. Ask about data analysis.", None), daemon=True).start()
                
                elif result_type == "start_data_analysis":
                    self.add_message("System", "Starting data analysis...")
                    threading.Thread(target=self._run_data_analysis, daemon=True).start()
                
                elif result_type == "data_analysis_started":
                    self.add_message("System", result)
                    self.add_message("System", "Please close the data analysis window when you're done. Then type 'continue' to proceed.")
                    self.update_workflow_status("data_analysis_completed")
                
                elif result_type == "generate_conclusion":
                    self.add_message("System", "Generating expert analysis conclusion...")
                    
                    # First show congestion analysis chart for user review
                    congestion_thumbnail = self._create_image_thumbnail(self.congestion_image)
                    if congestion_thumbnail:
                        self.add_message("System", "Congestion analysis for your review:", congestion_thumbnail, self.congestion_image)

                    # Start AI analysis
                    threading.Thread(target=self._call_ai_for_conclusion, daemon=True).start()

                elif result_type == "conclusion_ready":
                    # Display AI-generated conclusion text
                    self.add_message("AI Helper", result)
                    self.update_workflow_status("conclusion_ready")
                    self.add_message("System", "Expert analysis completed. You may now ask questions about the evacuation assessment.")

                elif result_type == "workflow_complete":
                    self.add_message("System", "Evacuation assessment workflow completed successfully!")

                elif result_type == "error":
                    self.add_message("System", f"Error: {result}")
                    # Reset state on error
                    self.update_workflow_status("idle")

                # Re-enable send button
                self.send_btn.config(state=tk.NORMAL)
                self.progress_var.set("")
                
        except queue.Empty:
            pass
        self.root.after(100, self._poll_results)

def run_app():
    root = tk.Tk()
    app = ImageChatApp(root)
    root.mainloop()

if __name__ == '__main__':
    run_app()