import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import threading
import time


class EvacuationAnalysisGUI:
    def __init__(self, output_path):
        self.output_path = output_path
        self.fig = None
        self.canvas = None
        self.video_label = None
        self.video_cap = None
        self.video_playing = False
        self.video_thread = None
        self.analysis_window = None
        
        # Video control related
        self.progress_var = tk.DoubleVar()
        self.progress_scale = None
        self.is_dragging = False
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 30
        self.video_length = 0
        
        # Data file paths
        self.evac_flow_path = os.path.join(output_path, "EvacFlow.csv")
        self.video_path = os.path.join(output_path, "Density.mp4")
        self.curve_save_path = os.path.join(output_path, "curve.png")
        
    def create_gui(self):
        """Create main GUI window"""
        self.analysis_window = tk.Toplevel()
        self.analysis_window.title("Evacuation Analysis Dashboard")
        self.analysis_window.geometry("900x1200")
        self.analysis_window.configure(bg="#f0f0f0")
        
        # Set window icon (if exists)
        try:
            icon_path = "Image/Signal.ico"
            if os.path.exists(icon_path):
                self.analysis_window.iconbitmap(icon_path)
        except:
            pass
        
        # Create styles
        self.create_styles()
        
        # Create main frame
        self.create_main_frame()
        
        # Create chart
        self.create_chart()
        
        # Create video player
        self.create_video_player()
        
        # Set window close event handler
        self.analysis_window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Bind window resize event
        self.analysis_window.bind("<Configure>", self.on_resize)
        
        # Start video playback
        self.start_video_playback()
        
        # Delayed layout adjustment
        self.analysis_window.after(100, self.adjust_layout)
        
    def adjust_layout(self):
        """Adjust window layout"""
        # Manually update window task queue to ensure all layout calculations complete
        self.analysis_window.update_idletasks()
        
        # Get current window height
        window_height = self.analysis_window.winfo_height()
        
        # Ensure window is shown and height is sufficient
        if window_height > 100:  
            # Calculate sash position: 40% of window height
            # So upper half (chart area) is 40%, lower half (video area) is 60%
            sash_position = int(window_height * 0.4)
            
            # Set sash position
            self.main_paned.sashpos(0, sash_position)
            
            # Print debug info (optional)
            print(f"Adjusting layout: window_height={window_height}, sash_position={sash_position}")
            
            # Force UI refresh to apply changes
            self.analysis_window.update()
        
    def create_styles(self):
        """Create TTK styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure("Header.TLabel", 
                       font=("Arial", 14, "bold"), 
                       background="#f0f0f0",
                       foreground="#2c3e50")
        
        style.configure("Subheader.TLabel", 
                       font=("Arial", 10), 
                       background="#f0f0f0",
                       foreground="#34495e")
        
        style.configure("Custom.TFrame", 
                       background="#ffffff",
                       relief="solid",
                       borderwidth=1)
        
        style.configure("Main.TFrame", 
                       background="#f0f0f0")
        
    def create_main_frame(self):
        """Create main frame structure"""
        # Main container
        main_frame = ttk.Frame(self.analysis_window, style="Main.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Title area
        title_frame = ttk.Frame(main_frame, style="Main.TFrame")
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = ttk.Label(title_frame, 
                               text="Evacuation Analysis Dashboard",
                               style="Header.TLabel")
        title_label.pack(anchor=tk.W)
        
        subtitle_label = ttk.Label(title_frame,
                                  text="Real-time evacuation flow analysis and density visualization",
                                  style="Subheader.TLabel")
        subtitle_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Use PanedWindow to split upper and lower regions
        self.main_paned = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Upper part: chart area
        self.chart_frame = ttk.Frame(self.main_paned, style="Custom.TFrame")
        
        # Lower part: video area
        self.video_frame = ttk.Frame(self.main_paned, style="Custom.TFrame")
        
        # Add to PanedWindow
        self.main_paned.add(self.chart_frame, weight=1)
        self.main_paned.add(self.video_frame, weight=1)
        
        # Chart area title
        chart_title_frame = ttk.Frame(self.chart_frame, style="Custom.TFrame")
        chart_title_frame.pack(fill=tk.X, padx=15, pady=(15, 5))
        
        ttk.Label(chart_title_frame, 
                 text="Evacuation Flow Over Time",
                 style="Header.TLabel").pack(anchor=tk.W)
        
        ttk.Label(chart_title_frame,
                 text="Number of evacuated people vs. time",
                 style="Subheader.TLabel").pack(anchor=tk.W, pady=(2, 0))
        
        # Video area title
        video_title_frame = ttk.Frame(self.video_frame, style="Custom.TFrame")
        video_title_frame.pack(fill=tk.X, padx=15, pady=(15, 5))
        
        ttk.Label(video_title_frame, 
                 text="Density Visualization",
                 style="Header.TLabel").pack(anchor=tk.W)
        
        ttk.Label(video_title_frame,
                 text="Real-time pedestrian density heatmap",
                 style="Subheader.TLabel").pack(anchor=tk.W, pady=(2, 0))
        
    def create_chart(self):
        """Create and display evacuation flow chart"""
        try:
            # Read data
            if not os.path.exists(self.evac_flow_path):
                print(f"Warning: {self.evac_flow_path} not found")
                return
                
            df = pd.read_csv(self.evac_flow_path)
            
            # Create chart
            self.fig = Figure(figsize=(10, 5), dpi=100, facecolor='white')
            ax = self.fig.add_subplot(111)
            
            # Draw line plot
            line = ax.plot(df['T'], df['N'], 
                          linewidth=3, 
                          color='#3498db', 
                          marker='o', 
                          markersize=4,
                          markerfacecolor='#e74c3c',
                          markeredgecolor='white',
                          markeredgewidth=1,
                          label='Evacuated People')
            
            # Add fill area
            ax.fill_between(df['T'], df['N'], alpha=0.3, color='#3498db')
            
            # Set chart style
            ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Evacuated People', fontsize=12, fontweight='bold')
            ax.set_title('Evacuation Progress Analysis', fontsize=14, fontweight='bold', pad=20)
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#fafafa')
            
            # Set axis style
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#bdc3c7')
            ax.spines['bottom'].set_color('#bdc3c7')
            
            # Add statistics info
            total_people = df['N'].max()
            total_time = df['T'].max()
            avg_rate = total_people / total_time if total_time > 0 else 0
            
            # Add statistics text on chart
            stats_text = f'Total Evacuated: {total_people}\nTotal Time: {total_time}s\nAvg Rate: {avg_rate:.2f} people/s'
            ax.text(0.02, 0.98, stats_text, 
                   transform=ax.transAxes, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                   fontsize=10)
            
            # Adjust layout
            self.fig.tight_layout(pad=3.0)
            
            # Embed into tkinter
            chart_container = ttk.Frame(self.chart_frame)
            chart_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(5, 15))
            
            self.canvas = FigureCanvasTkAgg(self.fig, chart_container)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            print("Chart created successfully")
            
        except Exception as e:
            print(f"Error creating chart: {e}")
            # Display error message
            error_label = ttk.Label(self.chart_frame, 
                                   text=f"Error loading chart data: {e}",
                                   style="Subheader.TLabel")
            error_label.pack(expand=True)
    
    def create_video_player(self):
        """Create enhanced video player (with progress bar)"""
        try:
            # Check if video file exists
            if not os.path.exists(self.video_path):
                no_video_label = ttk.Label(self.video_frame,
                                          text="Video file not found. Please ensure the simulation has completed.",
                                          style="Subheader.TLabel")
                no_video_label.pack(expand=True)
                return
            
            # Video display container (reserve space for control bar)
            video_display_frame = ttk.Frame(self.video_frame)
            video_display_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(5, 0))
            
            # Create video display label - set initial minimum size
            self.video_label = tk.Label(video_display_frame, bg='black', height=10)
            self.video_label.pack(fill=tk.BOTH, expand=True)
            
            # Control panel frame (fixed at bottom)
            control_panel = ttk.Frame(self.video_frame)
            control_panel.pack(fill=tk.X, padx=15, pady=(5, 15))
            
            # Row 1: playback control buttons
            button_frame = ttk.Frame(control_panel)
            button_frame.pack(fill=tk.X, pady=(0, 5))
            
            # Play/Pause button
            self.play_button = ttk.Button(button_frame, 
                                         text="⏸ Pause", 
                                         command=self.toggle_video,
                                         width=12)
            self.play_button.pack(side=tk.LEFT, padx=(0, 5))
            
            # Restart button
            restart_button = ttk.Button(button_frame, 
                                       text="🔄 Restart", 
                                       command=self.restart_video,
                                       width=12)
            restart_button.pack(side=tk.LEFT, padx=5)
            
            # Status label
            self.status_label = ttk.Label(button_frame, 
                                         text="Loading video...",
                                         style="Subheader.TLabel")
            self.status_label.pack(side=tk.RIGHT, padx=5)
            
            # Row 2: progress bar and time display
            progress_frame = ttk.Frame(control_panel)
            progress_frame.pack(fill=tk.X, pady=(5, 0))
            
            # Current time label
            self.current_time_label = ttk.Label(progress_frame, 
                                               text="00:00",
                                               style="Subheader.TLabel")
            self.current_time_label.pack(side=tk.LEFT, padx=(0, 5))
            
            # Progress bar
            self.progress_scale = ttk.Scale(progress_frame,
                                           from_=0,
                                           to=100,
                                           orient=tk.HORIZONTAL,
                                           variable=self.progress_var,
                                           command=self.on_progress_change)
            self.progress_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            # Total time label
            self.total_time_label = ttk.Label(progress_frame, 
                                             text="00:00",
                                             style="Subheader.TLabel")
            self.total_time_label.pack(side=tk.RIGHT, padx=(5, 0))
            
            # Bind progress bar drag events
            self.progress_scale.bind("<Button-1>", self.on_progress_press)
            self.progress_scale.bind("<ButtonRelease-1>", self.on_progress_release)
            
        except Exception as e:
            print(f"Error creating video player: {e}")
    
    def format_time(self, seconds):
        """Format time display (min:sec)"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def on_progress_press(self, event):
        """Progress bar press event"""
        self.is_dragging = True
    
    def on_progress_release(self, event):
        """Progress bar release event"""
        self.is_dragging = False
        if self.video_cap and self.total_frames > 0:
            # Set video frame position based on progress bar position
            progress = self.progress_var.get()
            target_frame = int((progress / 100.0) * self.total_frames)
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            self.current_frame = target_frame
    
    def on_progress_change(self, value):
        """Progress bar value change event (display update only)"""
        if self.is_dragging and self.total_frames > 0:
            progress = float(value)
            current_time = (progress / 100.0) * self.video_length
            self.current_time_label.configure(text=self.format_time(current_time))
            
    def on_resize(self, event):
        """Handle window resize event"""
        if event.widget == self.analysis_window:
            # Only handle size changes, ignore position changes
            if event.width != self.last_width or event.height != self.last_height:
                # Update last recorded size
                self.last_width = 0
                self.last_height = 0
                
                # Adjust layout
                self.adjust_layout(force=True)
                self.main_paned.sashpos(0, window_height // 2)
    
    def start_video_playback(self):
        """Start video playback"""
        if not os.path.exists(self.video_path):
            return
            
        try:
            self.video_cap = cv2.VideoCapture(self.video_path)
            if not self.video_cap.isOpened():
                print("Error: Could not open video file")
                return
            
            # Get video info
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_length = self.total_frames / self.fps if self.fps > 0 else 0
            
            # Update total time display
            if hasattr(self, 'total_time_label'):
                self.total_time_label.configure(text=self.format_time(self.video_length))
            
            # Set progress bar range
            if hasattr(self, 'progress_scale'):
                self.progress_scale.configure(to=100)
            
            # Delay video playback start to ensure window is fully initialized
            self.analysis_window.after(500, self._delayed_video_start)
                
        except Exception as e:
            print(f"Error starting video playback: {e}")
    
    def _delayed_video_start(self):
        """Delayed start of video playback"""
        self.video_playing = True
        self.video_thread = threading.Thread(target=self.play_video_loop, daemon=True)
        self.video_thread.start()
    
    def play_video_loop(self):
        """Video playback loop"""
        delay = 1.0 / self.fps if self.fps > 0 else 0.033  # Default 30fps
        
        while self.video_playing and self.video_cap and self.video_cap.isOpened():
            if not self.is_dragging:  # Only continue playback when not dragging progress bar
                ret, frame = self.video_cap.read()
                
                if not ret:
                    # Video ended, restart
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame = 0
                    continue
                
                self.current_frame = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                if self.video_label and self.analysis_window.winfo_exists():
                    try:
                        # Get label size
                        label_width = self.video_label.winfo_width()
                        label_height = self.video_label.winfo_height()
                        
                        # Wait for label to have reasonable size (avoid starting from very small)
                        if label_width < 100 or label_height < 100:
                            time.sleep(0.1)
                            continue
                        
                        # Convert color format
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        if label_width > 1 and label_height > 1:
                            # Maintain aspect ratio for adaptive scaling
                            frame_height, frame_width = frame_rgb.shape[:2]
                            
                            # Calculate scale ratio
                            width_ratio = label_width / frame_width
                            height_ratio = label_height / frame_height
                            scale_ratio = min(width_ratio, height_ratio)
                            
                            # Calculate new size
                            new_width = int(frame_width * scale_ratio)
                            new_height = int(frame_height * scale_ratio)
                            
                            # Resize frame
                            frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
                            
                            # Create black background
                            background = np.zeros((label_height, label_width, 3), dtype=np.uint8)
                            
                            # Calculate center position
                            y_offset = (label_height - new_height) // 2
                            x_offset = (label_width - new_width) // 2
                            
                            # Place resized frame at background center
                            background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = frame_resized
                            
                            # Convert to PIL image
                            image = Image.fromarray(background)
                            photo = ImageTk.PhotoImage(image)
                            
                            # Update display
                            self.video_label.configure(image=photo)
                            self.video_label.image = photo  # Keep reference
                            
                            # Update progress bar and time display
                            if self.total_frames > 0:
                                progress = (self.current_frame / self.total_frames) * 100
                                self.progress_var.set(progress)
                                
                                current_time = self.current_frame / self.fps if self.fps > 0 else 0
                                if hasattr(self, 'current_time_label'):
                                    self.current_time_label.configure(text=self.format_time(current_time))
                                
                                if hasattr(self, 'status_label'):
                                    status_text = f"Frame {self.current_frame}/{self.total_frames}"
                                    self.status_label.configure(text=status_text)
                        
                    except Exception as e:
                        print(f"Error updating video frame: {e}")
                        break
            
            time.sleep(delay)
        
    def toggle_video(self):
        """Toggle video play/pause"""
        self.video_playing = not self.video_playing
        
        if self.video_playing:
            self.play_button.configure(text="⏸ Pause")
            if not self.video_thread or not self.video_thread.is_alive():
                self.video_thread = threading.Thread(target=self.play_video_loop, daemon=True)
                self.video_thread.start()
        else:
            self.play_button.configure(text="▶ Play")
    
    def restart_video(self):
        """Restart video playback"""
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            self.progress_var.set(0)
            if not self.video_playing:
                self.toggle_video()
    
    def save_chart(self):
        """Save chart as PNG file"""
        if self.fig:
            try:
                self.fig.savefig(self.curve_save_path, 
                                dpi=300, 
                                bbox_inches='tight',
                                facecolor='white',
                                edgecolor='none')
                print(f"Chart saved to: {self.curve_save_path}")
            except Exception as e:
                print(f"Error saving chart: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        self.video_playing = False
        
        if self.video_cap:
            self.video_cap.release()
        
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
    
    def on_closing(self):
        """Window close event handler"""
        print("Saving chart and cleaning up...")
        
        # Save chart
        self.save_chart()
        
        # Clean up resources
        self.cleanup()
        
        # Close window
        if self.analysis_window:
            self.analysis_window.destroy()


def run_analysis_gui(output_path):
    """
    Main function to launch the analysis GUI.

    Args:
    - output_path: Output file path
    """
    print(f"Starting analysis GUI with output path: {output_path}")
    
    try:

        gui = EvacuationAnalysisGUI(output_path)
        
        gui.create_gui()
        
        print("Analysis GUI created successfully")
        
    except Exception as e:
        print(f"Error creating analysis GUI: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Main function for testing
    test_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Material", "data")
    
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    run_analysis_gui(test_output_path)
    
    root.mainloop()