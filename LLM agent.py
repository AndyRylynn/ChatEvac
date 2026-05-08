import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog, ttk
import base64
from openai import OpenAI
from PIL import Image, ImageTk
import io
import os
import sys
import shutil
import glob
import threading
import queue
from datetime import datetime

# API configuration — edit config.py in the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CHAT_API_KEY, CHAT_API_BASE, CHAT_MODEL


class ImageChatWindow:
    def __init__(self, master=None):
        
        # Create a new window for Image Chat
        if master:
            self.window = tk.Toplevel(master)
        else:
            self.window = tk.Tk()
            
        self.window.title("ChatEvac")
        self.window.geometry("800x600")
        self.window.minsize(600, 400)

        # API key and base URL are configured in config.py
        self.openai_api_key = CHAT_API_KEY
        self.openai_api_base = CHAT_API_BASE
        
        # System prompt for image description
        self.system_prompt = "You are an AI assistant that describes images in detail. Please analyze and describe the content of any image provided, including objects, people, scenes, colors, composition, and any other relevant details you can observe."

        # Initialize chat history
        self.chat_history = []
        
        # Current selected image
        self.current_image_path = None
        self.current_image_base64 = None
        
        # Store all widgets and their image references to prevent garbage collection
        self.chat_widgets = []
        
        # Threading for API calls
        self.api_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Material folder path
        self.material_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Material")
        
        # Clear material folder at startup
        self.clear_material_folder()

        # Setup the UI components
        self.setup_ui()

        # Welcome message
        self.add_to_chat("AI Helper", "Hello! I can help you describe images. Please select an image and optionally add text, then click send.")
        
        # Start checking for API results
        self.check_api_results()

    def setup_ui(self):
        # Main frame
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Configure grid
        main_frame.rowconfigure(0, weight=1)  # Chat area
        main_frame.rowconfigure(1, weight=0)  # Image preview area
        main_frame.rowconfigure(2, weight=0)  # Input area
        main_frame.columnconfigure(0, weight=1)

        # Chat display area with scrollbar - using simpler approach
        chat_frame = tk.Frame(main_frame)
        chat_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        chat_frame.rowconfigure(0, weight=1)
        chat_frame.columnconfigure(0, weight=1)

        self.chat_text = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            bg="#f5f5f5",
            font=("Arial", 11),
            state=tk.DISABLED
        )
        self.chat_text.grid(row=0, column=0, sticky="nsew")

        # Image preview area - now just shows status
        preview_frame = tk.Frame(main_frame, bg="#e0e0e0", height=60)
        preview_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        preview_frame.grid_propagate(False)
        
        self.image_label = tk.Label(
            preview_frame, 
            text="No image selected", 
            bg="#e0e0e0",
            wraplength=400,
            font=("Arial", 10)
        )
        self.image_label.pack(expand=True, pady=10)

        # Input area frame
        input_frame = tk.Frame(main_frame)
        input_frame.grid(row=2, column=0, sticky="ew")
        input_frame.columnconfigure(0, weight=1)

        # Text input area
        text_input_frame = tk.Frame(input_frame)
        text_input_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        text_input_frame.columnconfigure(0, weight=1)

        self.input_text = tk.Text(text_input_frame, height=3, wrap=tk.WORD)
        self.input_text.grid(row=0, column=0, sticky="ew")

        # Button frame
        button_frame = tk.Frame(input_frame)
        button_frame.grid(row=1, column=0, columnspan=2, sticky="ew")

        # Select image button
        self.select_image_button = tk.Button(
            button_frame,
            text="Select Image",
            command=self.select_image,
            width=12,
            bg="#FF9800",
            fg="white",
            relief=tk.RAISED
        )
        self.select_image_button.pack(side=tk.LEFT, padx=(0, 5))

        # Clear image button
        self.clear_image_button = tk.Button(
            button_frame,
            text="Clear Image",
            command=self.clear_image,
            width=12,
            bg="#f44336",
            fg="white",
            relief=tk.RAISED
        )
        self.clear_image_button.pack(side=tk.LEFT, padx=(0, 5))

        # View image button (new)
        self.view_image_button = tk.Button(
            button_frame,
            text="View Image",
            command=self.view_current_image,
            width=12,
            bg="#2196F3",
            fg="white",
            relief=tk.RAISED,
            state=tk.DISABLED
        )
        self.view_image_button.pack(side=tk.LEFT, padx=(0, 5))

        # Send button
        self.send_button = tk.Button(
            button_frame,
            text="Send",
            command=self.send_message,
            width=12,
            bg="#4CAF50",
            fg="white",
            relief=tk.RAISED
        )
        self.send_button.pack(side=tk.RIGHT)

        # Progress bar for loading
        self.progress_var = tk.StringVar()
        self.progress_label = tk.Label(button_frame, textvariable=self.progress_var, fg="blue")
        self.progress_label.pack(side=tk.RIGHT, padx=(0, 10))

        # Bind Enter key to send message
        self.input_text.bind("<Control-Return>", self.on_enter_key)

        # Add a status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = tk.Label(
            self.window,
            textvariable=self.status_var,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def clear_material_folder(self):
        """Clear all files in the Material folder"""
        try:
            # Create the directory if it doesn't exist
            if not os.path.exists(self.material_path):
                os.makedirs(self.material_path)
                print(f"Created directory: {self.material_path}")
                return
            
            # Clear all files in the directory
            files = glob.glob(os.path.join(self.material_path, "*"))
            for file in files:
                try:
                    if os.path.isfile(file):
                        os.remove(file)
                        print(f"Removed file: {file}")
                    elif os.path.isdir(file):
                        shutil.rmtree(file)
                        print(f"Removed directory: {file}")
                except Exception as e:
                    print(f"Error removing {file}: {e}")
            
            print(f"Material folder cleared: {self.material_path}")
            
        except Exception as e:
            print(f"Error clearing material folder: {e}")
            messagebox.showwarning("Warning", f"Could not clear material folder: {e}")

    def copy_image_to_material(self, source_path):
        """Copy the selected image to Material folder as Input.png"""
        try:
            if not os.path.exists(self.material_path):
                os.makedirs(self.material_path)
            
            target_path = os.path.join(self.material_path, "Input.png")
            
            # Open image and save as PNG, removing problematic color profiles
            with Image.open(source_path) as img:
                # Remove any color profile to avoid warnings
                if 'icc_profile' in img.info:
                    del img.info['icc_profile']
                
                # Convert to RGB if necessary
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
                
                # Save as PNG without color profile
                img.save(target_path, 'PNG', optimize=True)
                print(f"Image copied to: {target_path}")
                
        except Exception as e:
            print(f"Error copying image to material folder: {e}")
            messagebox.showerror("Error", f"Failed to copy image: {e}")

    def select_image(self):
        """Select an image file"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.gif *.bmp *.tiff"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=file_types
        )
        
        if file_path:
            try:
                # Load and encode image
                with open(file_path, "rb") as image_file:
                    image_data = image_file.read()
                    self.current_image_base64 = base64.b64encode(image_data).decode('utf-8')
                
                self.current_image_path = file_path
                
                # Copy image to Material folder
                self.copy_image_to_material(file_path)
                
                # Show text status instead of image preview
                filename = os.path.basename(file_path)
                self.image_label.config(text=f"Image loaded: {filename}")
                self.status_var.set(f"Image loaded and copied: {filename}")
                
                # Enable view image button
                self.view_image_button.config(state=tk.NORMAL)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.status_var.set("Error loading image")
                self.clear_image()

    def view_current_image(self):
        """Open the current image in a popup window - read from Material folder"""
        # Always read from the Material folder's Input.png
        material_image_path = os.path.join(self.material_path, "Input.png")
        
        if not os.path.exists(material_image_path):
            messagebox.showwarning("Warning", "No image found in Material folder")
            return
        
        try:
            # Create popup window
            popup = tk.Toplevel(self.window)
            popup.title("Image Viewer - Input.png")
            popup.geometry("600x500")
            popup.configure(bg="white")
            
            # Load and display image from Material folder
            with Image.open(material_image_path) as img:
                # Get original filename for display
                original_name = os.path.basename(self.current_image_path) if self.current_image_path else "Input.png"
                
                # Create a copy to avoid file lock issues
                img_copy = img.copy()
                
                # Remove color profile to avoid warnings
                if 'icc_profile' in img_copy.info:
                    del img_copy.info['icc_profile']
                
                # Convert to RGB if necessary
                if img_copy.mode in ('RGBA', 'LA', 'P'):
                    if img_copy.mode == 'P':
                        img_copy = img_copy.convert('RGBA')
                    if img_copy.mode in ('RGBA', 'LA'):
                        background = Image.new('RGB', img_copy.size, (255, 255, 255))
                        if img_copy.mode == 'RGBA':
                            background.paste(img_copy, mask=img_copy.split()[-1])
                        else:
                            background.paste(img_copy)
                        img_copy = background
                elif img_copy.mode != 'RGB':
                    img_copy = img_copy.convert('RGB')
                
                # Get image dimensions for display
                original_size = f"{img_copy.width} x {img_copy.height}"
                
                # Resize to fit popup if necessary
                max_width, max_height = 550, 350
                display_img = img_copy.copy()
                if display_img.width > max_width or display_img.height > max_height:
                    display_img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(display_img)
                
                # Add image info label
                info_label = tk.Label(
                    popup, 
                    text=f"File: {original_name}\nSize: {original_size} pixels", 
                    font=("Arial", 10),
                    bg="white"
                )
                info_label.pack(pady=(10, 5))
                
                # Create label and keep reference
                label = tk.Label(popup, image=photo, bg="white", bd=2, relief="sunken")
                label.image = photo  # Keep a reference to prevent garbage collection
                label.pack(expand=True, pady=10)
                
                # Add button frame
                btn_frame = tk.Frame(popup, bg="white")
                btn_frame.pack(pady=10)
                
                # Add close button
                close_btn = tk.Button(
                    btn_frame, 
                    text="Close", 
                    command=popup.destroy,
                    width=10,
                    bg="#f44336",
                    fg="white"
                )
                close_btn.pack(side=tk.LEFT, padx=5)
                
                # Add open in system viewer button
                def open_in_system():
                    try:
                        import subprocess
                        import platform
                        if platform.system() == 'Windows':
                            os.startfile(material_image_path)
                        elif platform.system() == 'Darwin':  # macOS
                            subprocess.run(['open', material_image_path])
                        else:  # Linux
                            subprocess.run(['xdg-open', material_image_path])
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to open image in system viewer: {str(e)}")
                
                system_btn = tk.Button(
                    btn_frame,
                    text="Open in System Viewer",
                    command=open_in_system,
                    width=20,
                    bg="#2196F3",
                    fg="white"
                )
                system_btn.pack(side=tk.LEFT, padx=5)
                
        except Exception as e:
            print(f"Error viewing image: {e}")
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")

    def clear_image(self):
        """Clear the selected image"""
        # Clear all image references
        self.current_image_path = None
        self.current_image_base64 = None
        
        # Clear the label status text
        self.image_label.config(text="No image selected")
        
        # Disable view image button
        self.view_image_button.config(state=tk.DISABLED)
        
        self.status_var.set("Image cleared")

    def add_to_chat(self, sender, message, image_path=None):
        """Add a message to the chat display - simplified approach"""
        self.chat_text.config(state=tk.NORMAL)

        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Format based on sender
        if sender == "You":
            self.chat_text.insert(tk.END, f"\n[{timestamp}] {sender}: ", "user")
            self.chat_text.tag_configure("user", foreground="blue", font=("Arial", 11, "bold"))
        else:
            self.chat_text.insert(tk.END, f"\n[{timestamp}] {sender}: ", "ai")
            self.chat_text.tag_configure("ai", foreground="green", font=("Arial", 11, "bold"))

        # Insert the message text
        self.chat_text.insert(tk.END, f"{message}\n")

        # Add image indicator if provided (instead of actual image)
        if image_path and os.path.exists(image_path):
            filename = os.path.basename(image_path)
            self.chat_text.insert(tk.END, f"📷 [Image: {filename}]\n", "image_indicator")
            self.chat_text.tag_configure("image_indicator", foreground="purple", font=("Arial", 10, "italic"))

        # Auto-scroll to the bottom
        self.chat_text.see(tk.END)
        self.chat_text.config(state=tk.DISABLED)

        # Update window to show changes
        self.window.update_idletasks()

    def on_enter_key(self, event):
        """Handle Ctrl+Enter key press to send message"""
        self.send_message()
        return "break"

    def send_message(self):
        """Process and send user message"""
        user_message = self.input_text.get("1.0", tk.END).strip()
        
        # Check if we have either text or image
        if not user_message and not self.current_image_base64:
            messagebox.showwarning("Warning", "Please enter text or select an image")
            return

        # Prepare display message
        display_message = user_message if user_message else "[Image only]"

        # Add user message to chat with image indicator if present
        self.add_to_chat("You", display_message, self.current_image_path)

        # Clear input field
        self.input_text.delete("1.0", tk.END)

        # Disable send button and show progress
        self.send_button.config(state=tk.DISABLED)
        self.progress_var.set("Sending...")
        self.status_var.set("Getting response...")

        # Start API call in background thread
        api_thread = threading.Thread(
            target=self.call_openai_api_threaded,
            args=(user_message, self.current_image_base64),
            daemon=True
        )
        api_thread.start()

    def call_openai_api_threaded(self, user_message, image_base64):
        """Call OpenAI API in a separate thread"""
        try:
            client = OpenAI(base_url=CHAT_API_BASE, api_key=CHAT_API_KEY)

            # Prepare messages for the API
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]

            # Add chat history (limit to last 6 messages to avoid token limits)
            for msg in self.chat_history[-6:]:
                messages.append(msg)

            # Prepare the current message
            if image_base64:
                # Message with image
                content = [
                    {
                        "type": "text",
                        "text": user_message if user_message else "Please describe this image in detail."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            else:
                # Text only message
                content = user_message

            messages.append({
                "role": "user",
                "content": content
            })

            # Call the API
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                max_tokens=1000
            )

            # Get the response
            ai_message = response.choices[0].message.content

            # Update chat history
            self.chat_history.append({
                "role": "user",
                "content": user_message if user_message else "Please describe this image"
            })
            self.chat_history.append({
                "role": "assistant", 
                "content": ai_message
            })
            
            # Put result in queue
            self.result_queue.put(("success", ai_message))

        except Exception as e:
            # Put error in queue
            self.result_queue.put(("error", str(e)))

    def check_api_results(self):
        """Check for API results and update UI accordingly"""
        try:
            # Check if there's a result waiting
            result_type, message = self.result_queue.get_nowait()
            
            if result_type == "success":
                self.add_to_chat("AI Helper", message)
                self.status_var.set("Ready")
            else:
                error_msg = f"Sorry, an error occurred: {message}"
                self.add_to_chat("AI Helper", error_msg)
                self.status_var.set("Error occurred")
            
            # Re-enable send button and clear progress
            self.send_button.config(state=tk.NORMAL)
            self.progress_var.set("")
            
        except queue.Empty:
            # No result yet, check again later
            pass
        
        # Schedule next check
        self.window.after(100, self.check_api_results)

        
def run_image_chat():
    """Main function to run the ChatEvac application"""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    app = ImageChatWindow()
    
    def on_closing():
        root.quit()
        root.destroy()
    
    app.window.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


# Run the application if script is executed directly
if __name__ == "__main__":
    run_image_chat()