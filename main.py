import os
 if 'ANDROID_AUGMENT' in os.environ:
     import cv2
     import torch
     import pytz
     import tensorflow.lite as tflite
     import requests
     import numpy as np
     import threading
     import queue
     from datetime import datetime
     from PIL import Image 
     from ultralytics import YOLO
     from kivy.graphics.texture import Texture
     from kivy.core.text import LabelBase
     from kivy.uix.screenmanager import ScreenManager, Screen
     from kivymd.app import MDApp
     from kivy.lang import Builder
     from kivy.clock import Clock
     from kivy.uix.image import Image
     from kivy.uix.boxlayout import BoxLayout
     from kivymd.uix.menu import MDDropdownMenu
     from kivymd.uix.button import MDRaisedButton
     from kivy.core.window import Window
     from kivymd.uix.dialog import MDDialog
     from kivy.uix.screenmanager import NoTransition 
     from kivy.uix.popup import Popup
     from kivy.uix.label import Label
     from kivy.uix.screenmanager import Screen, ScreenManager
     from kivymd.app import MDApp
     from kivy.uix.boxlayout import BoxLayout
     from kivy.metrics import dp
     import matplotlib.pyplot as plt
     import matplotlib.colors as mcolors
     from matplotlib.colors import LinearSegmentedColormap
     from kivy.utils import get_color_from_hex
     from PIL import Image as PILImage, ImageDraw, ImageFont
     from kivy.lang import Builder
 





def scale_bbox_to_thermal(bbox, live_width, live_height, thermal_width, thermal_height):

    #Scale bounding box from live feed resolution to thermal resolution.
    x1, y1, x2, y2 = bbox
    x1_scaled = int(x1 * thermal_width / live_width)
    x2_scaled = int(x2 * thermal_width / live_width)
    y1_scaled = int(y1 * thermal_height / live_height)
    y2_scaled = int(y2 * thermal_height / live_height)
    
    # Ensure the coordinates stay within the thermal resolution bounds
    x1_scaled = max(0, min(thermal_width - 1, x1_scaled))
    x2_scaled = max(0, min(thermal_width - 1, x2_scaled))
    y1_scaled = max(0, min(thermal_height - 1, y1_scaled))
    y2_scaled = max(0, min(thermal_height - 1, y2_scaled))
    
    return (x1_scaled, y1_scaled, x2_scaled, y2_scaled)


Builder.load_file('main.kv')
Builder.load_file('login.kv')
Builder.load_file('signup.kv')
Builder.load_file('profile.kv')
Builder.load_file('pet_profile.kv')
Builder.load_file('dashboard.kv')
Builder.load_file('livefeed.kv') 
Builder.load_file('profile_tab.kv')
Builder.load_file('about_us.kv')
Builder.load_file('help.kv')
Builder.load_file('user_guide.kv')
Builder.load_file('FAQs.kv')
Builder.load_file('edit_profile.kv')
Builder.load_file('pet_tab.kv')
Builder.load_file('activity_logs.kv')
Builder.load_file('thermal.kv')
Builder.load_file('forgot_password.kv')

class LiveFeedScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.dog_model = YOLO('dog.pt')
        self.dog_model = self.load_tflite_model("dog_saved_model/dog_float16.tflite")  # Load TFLite dog model

        #self.behavior_model = YOLO('best1.pt')
        self.behavior_model = self.load_tflite_model("best1_saved_model/best1_float16.tflite")  # Load TFLite behavior model

        ip = '192.168.0.102'
        # RTSP URL
        self.rtsp_url = f'rtsp://admin:L2A51CBA@{ip}:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif'

        self.img_widget = Image()
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.img_widget)

        self.capture = cv2.VideoCapture(self.rtsp_url)
        self.frame_queue = queue.Queue(maxsize=10)
        self.thread = threading.Thread(target=self.read_frames)
        self.thread.daemon = True
        self.thread.start()

        Clock.schedule_interval(self.update_frame, 1/60)

        self.previous_saved_behavior = None 
        self.previous_behavior = None

        self.add_widget(layout)
    
    def load_tflite_model(self, model_path):
        """ Load the TensorFlow Lite model """
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter


    def read_frames(self):
        while True:
            ret, frame = self.capture.read()
            if ret and not self.frame_queue.full():
                self.frame_queue.put(frame)

    def update_frame(self, dt):
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            frame = cv2.flip(frame, 0)
            detected_behaviors, dog_boxes = self.detect_dog_and_behavior(frame)

            # Limit to only 1 dog and 1 bounding box
            if dog_boxes:
                # Choose the largest bounding box (if multiple are detected)
                dog_boxes = [self.select_largest_bbox(dog_boxes)]

            app = PetWatch.get_running_app()
            app.shared_data['dog_bboxes'] = dog_boxes
            print(f"LiveFeedScreen: Updated bounding boxes: {dog_boxes}")

            for box, behavior in zip(dog_boxes, detected_behaviors):
                x1, y1, x2, y2 = box
                behavior_label = behavior["behavior"].capitalize()

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Draw behavior label
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2
                font_thickness = 5
                (text_width, text_height), _ = cv2.getTextSize(behavior_label, font, font_scale, font_thickness)
                text_img = np.full((text_height + 15, text_width + 15, 3), (0, 255, 0), dtype=np.uint8)
                text_color = (255, 255, 255)
                cv2.putText(text_img, behavior_label, (5, text_height + 5), font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)
                flipped_text_img = cv2.flip(text_img, 0)

                text_x = x1
                text_y = min(frame.shape[0] - text_height, y2 + text_height + 5)
                overlay_y1 = max(0, text_y)
                overlay_y2 = min(frame.shape[0], overlay_y1 + flipped_text_img.shape[0])
                overlay_x1 = text_x
                overlay_x2 = min(frame.shape[1], overlay_x1 + flipped_text_img.shape[1])

                alpha = 0.9
                if overlay_y2 > overlay_y1 and overlay_x2 > overlay_x1:
                    frame[overlay_y1:overlay_y2, overlay_x1:overlay_x2] = cv2.addWeighted(
                        frame[overlay_y1:overlay_y2, overlay_x1:overlay_x2],
                        1 - alpha,
                        flipped_text_img[: overlay_y2 - overlay_y1, : overlay_x2 - overlay_x1],
                        alpha,
                        0,
                    )

                # Log behavior if it's new
                if behavior_label != self.previous_behavior:
                    self.send_behavior_log(behavior_label)
                    self.previous_behavior = behavior_label  # Update the previous behavior

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf = frame.tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.img_widget.texture = texture

    def detect_dog_and_behavior(self, frame):
        detected_behaviors = []
        dog_boxes = []

    # Perform dog detection
        dog_results = self.run_tflite_inference(frame, self.dog_model)

        if dog_results is not None and len(dog_results) > 0:
            for dog_bbox in dog_results:
                if len(dog_bbox) >= 6:  # Ensure valid bounding box format
                    x_min, y_min, x_max, y_max, confidence, class_id = dog_bbox[:6]  # Extract bbox values and confidence
                    confidence = float(confidence)  # Ensure it's a float

                if confidence > 0.5:  # Filter low-confidence detections
                    dog_boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))
                    print(f"✅ Detected object {int(class_id)} with confidence {confidence:.2f}")

                    # Crop the image to focus on the dog region
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                    dog_crop = frame[y_min:y_max, x_min:x_max]

                    if dog_crop is None or dog_crop.size == 0:  # Check if the crop is valid
                        print("❌ Warning: dog_crop is empty. Skipping resize.")
                    else:
                        dog_resized = cv2.resize(dog_crop, (224, 224))
                        dog_resized = cv2.cvtColor(dog_resized, cv2.COLOR_BGR2RGB)
                        dog_resized = dog_resized.astype(np.float32) / 255.0
                        dog_resized = np.expand_dims(dog_resized, axis=0)

                        # Run behavior detection on the cropped dog region
                        behavior_results = self.run_tflite_inference(dog_resized, self.behavior_model)

                        # Extract detected behavior labels
                        if behavior_results is not None and len(behavior_results) > 0:
                            for behavior_box in behavior_results:
                                behavior_label = self.behavior_model.names[int(behavior_box[0])]
                                detected_behaviors.append({"behavior": behavior_label})

        return detected_behaviors, dog_boxes
    
    def run_tflite_inference(self, input_image, model):
        """ Run inference using TensorFlow Lite model and extract bounding boxes """
        interpreter = model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Resize image to match model input size
        input_image = cv2.resize(input_image, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
        input_image = np.array(input_image, dtype=np.float32) / 255.0
        input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

        interpreter.set_tensor(input_details[0]['index'], input_image)
        interpreter.invoke()

        # Get output data
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Extract bounding boxes (depends on model output format)
        boxes = output_data[0]  # Adjust if the output format is different

        return boxes




    def select_largest_bbox(self, dog_boxes):
        """Select the largest bounding box based on area (width * height)."""
        largest_bbox = None
        max_area = 0
        for box in dog_boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                largest_bbox = box
        return largest_bbox

    def send_behavior_log(self, behavior):
        behavior = behavior.capitalize()  # Ensure the behavior label is capitalized
        timestamp = datetime.now().isoformat()  # Get the current timestamp in ISO format

        url = "http://127.0.0.1:8000/save-behavior-log/"
        data = {
            "behavior": behavior,  # Behavior field
            "timestamp": timestamp  # Timestamp field
        }

        try:
            response = requests.post(url, json=data)
            print(f"Response status code: {response.status_code}")
            print(f"Response content: {response.text}")

            if response.status_code == 201:
                print(f"Log saved successfully: {behavior}")
                self.fetch_logs_from_backend()  # Fetch logs again to ensure the latest is at the top
            else:
                print(f"Failed to save log: {response.json()}")
        except Exception as e:
            print(f"Error: {e}")


    def on_enter(self):
        print(f'Entering Live Feed Screen. RTSP URL: {self.rtsp_url}')

    def on_leave(self):
        print("Leaving Live Feed Screen. Stream will continue running.")

    def go_to_live_feed(self):
        self.manager.current = "live_feed"  

    def go_to_thermal_feed(self):
        self.manager.current = "thermal_camera"

    def profile_tab(self):
        self.manager.current = "profile_tab"

    def home(self):
        self.manager.current = "dashboard"
    
    def pet_tab(self):
        self.manager.current = "pet_tab"
    
    def activity_logs(self):
        self.manager.current = "activity_logs"

class ThermalCameraScreen(Screen):
    def __init__(self, **kwargs):
        super(ThermalCameraScreen, self).__init__(**kwargs)
        self.dog_bboxes = []  # List to store bounding boxes of detected dogs
        self.image_widget = None  
        self._event = None       
        self.build_ui()          

    def build_ui(self):
        layout = BoxLayout(orientation='vertical')
        self.image_widget = Image()
        layout.add_widget(self.image_widget)
        self.add_widget(layout)

    def update_dog_bboxes(self, bboxes):
        self.dog_bboxes = bboxes 

    def calculate_dog_temperature(self, thermal_data, dog_bbox):
        """
        Calculate the highest temperature inside the bounding box of the dog.
        """
        x1, y1, x2, y2 = dog_bbox
        dog_region = thermal_data[y1:y2, x1:x2]  # Crop the region from the thermal image
        if dog_region.size > 0:
            return np.max(dog_region)  # Return the highest temperature in the region
        return None
    
    def on_enter(self):
        self._event = Clock.schedule_interval(self.update_thermal_view, 1)

    def on_leave(self):
        if self._event:
            self._event.cancel()
            self._event = None

    def fetch_thermal_data_from_server(self):
        try:
            # ESP32 IP address
            ESP32_ip = 'http://192.168.0.200' 
            endpoint = '/thermal'

            # Send HTTP GET request to fetch thermal data
            response = requests.get(ESP32_ip + endpoint)

            if response.status_code == 200:
                # Process and return the thermal data
                data = response.text
                return [float(i) for i in data.split(',')]
            else:
                print(f'Error: {response.status_code}')
        except requests.exceptions.RequestException as e:
            print(f'Error fetching data: {e}')
        return None

    def display_heatmap(self, frame_data):
        try:
            # Reshape the thermal data to match the thermal camera's resolution (24x32)
            data_matrix = np.array(frame_data).reshape((24, 32))

            dog_temp_found = False
            max_dog_temp = None

            # Scale bounding boxes to thermal resolution and store them for drawing
            scaled_bboxes = []
            for dog_bbox in self.dog_bboxes:
                scaled_bbox = scale_bbox_to_thermal(dog_bbox, 2880, 1620, 32, 24)
                scaled_bboxes.append(scaled_bbox)

                # Crop the region from the thermal data based on the bounding box
                x1, y1, x2, y2 = scaled_bbox
                dog_region = data_matrix[y1:y2, x1:x2]
                if dog_region.size > 0:
                    # Find the maximum temperature in the region
                    dog_temp = np.max(dog_region) + 8
                    if max_dog_temp is None or dog_temp > max_dog_temp:
                        max_dog_temp = dog_temp
                        dog_temp_found = True

            # Update dog temperature label
            if dog_temp_found and max_dog_temp is not None:
                self.ids.dog_temperature_label.text = f": {max_dog_temp:.2f}°C"
                PetWatch.get_running_app().dog_body_temp = f"{max_dog_temp:.2f}°C"
            else:
                self.ids.dog_temperature_label.text = ": Undetected"
                PetWatch.get_running_app().dog_body_temp = "Undetected"

            # Calculate environment temperature
            environment_temp = np.mean(data_matrix)
            if self.ids.temperature_label:
                self.ids.temperature_label.text = f": {environment_temp:.2f}°C"

            # Smooth and resize data for visualization
            smoothed_data = cv2.GaussianBlur(data_matrix, (3, 3), 0)
            screen_width = Window.width
            target_width = screen_width
            target_height = int(target_width * 9 / 16)  # Maintain 16:9 aspect ratio
            resized_data = cv2.resize(smoothed_data, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

            # Normalize the data for colormap application
            normalized_data = cv2.normalize(resized_data, None, 0, 1, cv2.NORM_MINMAX)
            from matplotlib import cm
            # Apply the Inferno colormap from Matplotlib
            inferno_cmap = cm.get_cmap('inferno')
            colormap = (inferno_cmap(normalized_data)[:, :, :3] * 255).astype(np.uint8)

            # Convert to RGB format for Kivy
            colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

            # Adjust bounding boxes to the displayed resolution with calculated offsets and resizing
            display_width, display_height = colormap.shape[1], colormap.shape[0]

            # Calculate the vertical offset based on 2 inches downward position
            camera_distance_in_inches = 2  # Distance in inches between cameras
            pixels_per_inch = display_height / 24  # Assuming thermal camera's height is 24 units
            vertical_offset = int(camera_distance_in_inches * pixels_per_inch)

            # Shrink factor to make bounding boxes slightly smaller
            shrink_factor = 0.8  # 90% of the original size

            for x1, y1, x2, y2 in scaled_bboxes:
                # Scale the bounding box to the displayed resolution
                display_x1 = int(x1 * display_width / 32)
                display_x2 = int(x2 * display_width / 32)
                display_y1 = int(y1 * display_height / 24) + vertical_offset
                display_y2 = int(y2 * display_height / 24) + vertical_offset

                # Calculate the center of the bounding box
                center_x = (display_x1 + display_x2) // 2
                center_y = (display_y1 + display_y2) // 2

                # Apply shrink factor
                new_width = int((display_x2 - display_x1) * shrink_factor)
                new_height = int((display_y2 - display_y1) * shrink_factor)

                # Update bounding box coordinates based on the shrink factor
                display_x1 = center_x - new_width // 2
                display_x2 = center_x + new_width // 2
                display_y1 = center_y - new_height // 2
                display_y2 = center_y + new_height // 2

                # Ensure the adjusted bounding box stays within the displayed frame
                display_x1 = max(0, min(display_width - 1, display_x1))
                display_x2 = max(0, min(display_width - 1, display_x2))
                display_y1 = max(0, min(display_height - 1, display_y1))
                display_y2 = max(0, min(display_height - 1, display_y2))

                # Draw the bounding box

                # Drawing the green bounding box
                cv2.rectangle(colormap, (display_x1, display_y1), (display_x2, display_y2), (0, 255, 0), 3)

                
                # Define text for the temperature (split it into two parts)
                temp_text = f"{dog_temp:.2f}"  # Temperature without the degree symbol
                degree_text = "C"  # Degree symbol and "C" part

                # Font settings
                font = cv2.FONT_HERSHEY_COMPLEX  # Use a different font
                font_scale = 0.7
                font_thickness = 2
                color = (255, 255, 255)  # White color

                # Position text on the left of the bounding box
                text_size_temp = cv2.getTextSize(temp_text, font, font_scale, font_thickness)[0]
                
                padding = 5  # Horizontal and vertical space from the top-left corner of the bounding box

                # Position the text at the top-left corner of the bounding box
                text_x = display_x1 + padding  # Horizontal position (slightly offset from the left)
                text_y = display_y1 - padding  # Vertical position (slightly offset from the top)

                # Draw the temperature (without degree symbol)
                cv2.putText(colormap, temp_text, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)

                # Draw the degree symbol and "C" next to the temperature
                text_x += text_size_temp[0] + 2  # Move x position after temperature
                cv2.putText(colormap, degree_text, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)

                # Flip the image horizontally (flip the bounding box and the text)
                colormap = cv2.flip(colormap, 1)


            # Update the texture for Kivy
            texture = Texture.create(size=(colormap.shape[1], colormap.shape[0]), colorfmt='rgb')
            texture.blit_buffer(colormap.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            texture.flip_vertical()
            texture.flip_horizontal()
            texture.mag_filter = 'linear'  # Enable smoothing for texture scaling

            self.image_widget.texture = texture

        except Exception as e:
            print(f"Error displaying heatmap: {e}")

    def update_thermal_view(self, dt):
        frame_data = self.fetch_thermal_data_from_server()
        if frame_data:
            app = PetWatch.get_running_app()
            dog_bboxes = app.shared_data.get("dog_bboxes", [])
            self.dog_bboxes = dog_bboxes

            print(f"ThermalCameraScreen: Retrieved bounding boxes: {self.dog_bboxes}")

            environment_temp = np.mean(frame_data)  # Calculate environment temperature
            if self.ids.temperature_label:
                self.ids.temperature_label.text = f'Environment Temperature: {environment_temp:.2f}°C'

            self.display_heatmap(frame_data)

    def go_to_live_feed(self):
        self.manager.current = "live_feed"  # Navigate to live feed screen

    def go_to_thermal_feed(self):
        self.manager.current = "thermal_camera"  # Navigate to thermal feed screen

    def profile_tab(self):
        self.manager.current = "profile_tab"

    def home(self):
        self.manager.current = "dashboard"
    
    def pet_tab(self):
        self.manager.current = "pet_tab"
    
    def activity_logs(self):
        self.manager.current = "activity_logs"

class ActivityLogsScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.previous_behavior = None
        self.activity_log = []  # Store persistent logs
        self.backend_url = "http://127.0.0.1:8000/api/get-logs/"  # Fetch from environment or default
        self.log_update_interval = 5  # Fetch logs every 5 seconds

    def update_behavior_bboxes(self, behavior_bboxes, behavior_labels):
        """Update behavior bounding boxes and log the behavior."""
        if not behavior_labels:
            return  # No behavior detected, exit early

        behavior_labels.sort(key=lambda x: x[1], reverse=True)
        # Get the first detected behavior (assumes behaviors are in a prioritized order)
        current_behavior = behavior_labels[0]

        # Check if the behavior has changed
        if self.previous_behavior != current_behavior:
            current_time = datetime.now().strftime("%I:%M:%S %p")
            log_entry = f"{current_time} - {current_behavior}"

            # Only add the log if it isn't already present
            if log_entry not in self.activity_log:
                self.activity_log.insert(0, log_entry)  # Prepend the log entry

            # Update the previous behavior
            self.previous_behavior = current_behavior
            self.update_activity_log_display()

    def update_activity_log_display(self):
        """Update the displayed activity log."""
        # Ensure logs are sorted in reverse chronological order
        log_text = "\n".join(self.activity_log)
        behavior_log_label = self.ids.behavior_log
        behavior_log_label.text = log_text
        behavior_log_label.height = behavior_log_label.texture_size[1]  # Update height
        behavior_log_container = self.ids.behavior_log_container
        behavior_log_container.height = behavior_log_label.height + dp(20)

    def fetch_logs_from_backend(self, *args):
        """Fetch logs from the backend API and update the UI."""
        try:
            # Fetch data from the backend API
            response = requests.get(self.backend_url)
            if response.status_code == 200:
                fetched_logs = response.json()  # Assumes logs are returned as a JSON list
                
                # Get today's date in the format used by the log timestamps
                today = datetime.now().date()
                
                # Filter logs to only include those with today's date
                new_logs = []
                for log in fetched_logs:
                    log_timestamp = datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00'))
                    if log_timestamp.date() == today:  # Check if the log is from today
                        log_entry = f"{self.convert_to_ph_time(log['timestamp'])} - {log['behavior']}"
                        if log_entry not in self.activity_log:
                            new_logs.append(log_entry)
                
                # Add filtered logs to the activity log
                for log in new_logs:
                    self.activity_log.insert(0, log)  # Prepend new logs
                
                # Sort logs explicitly by reverse chronological order
                self.activity_log.sort(reverse=True)
                self.update_activity_log_display()
            else:
                self.ids.behavior_log.text = "Failed to fetch logs from the server."
        except Exception as e:
            self.ids.behavior_log.text = f"Error: {str(e)}"

    def convert_to_ph_time(self, timestamp):
        """Convert timestamp to Philippine Time (UTC+8)."""
        utc_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        ph_timezone = pytz.timezone('Asia/Manila')
        ph_time = utc_time.astimezone(ph_timezone)
        return ph_time.strftime("%I:%M:%S %p")

    def on_enter(self):
        """Fetch logs from backend and update the display when the screen is entered."""
        self.update_date_label()
        Clock.schedule_once(lambda dt: self.fetch_logs_from_backend(), 0.5)
        # Start periodic log fetching
        Clock.schedule_interval(self.fetch_logs_from_backend, self.log_update_interval)

    def update_date_label(self):
        """Update the date label dynamically."""
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        self.ids.date_label.text = current_date

    def go_to_live_feed(self):
        self.manager.current = "live_feed"

    def profile_tab(self):
        self.manager.current = "profile_tab"

    def home(self):
        self.manager.current = "dashboard"
    
    def pet_tab(self):
        self.manager.current = "pet_tab"
    
    def activity_logs(self):
        self.manager.current = "activity_logs"

    def thermal(self):
        self.manager.current = "thermal_camera"

class ScreenManagement(ScreenManager):
    pass

class SignupScreen(Screen):
    def signup(self, first_name, last_name, email, password):
        # Name Validation
        if not first_name.strip() or not last_name.strip():
            self.display_error_message('Please enter both your first name and last name.')
            return

        full_name = f'{first_name} {last_name}'

        # Email Validation
        if not (email.endswith('@yahoo.com') or email.endswith('@gmail.com')):
            self.display_error_message('Please use a @yahoo.com or @gmail.com email.')
            return

        # Password Validation
        if len(password) < 8:
            self.display_error_message('Password must be at least 8 characters long.')
            return

        url = 'http://127.0.0.1:8000/signup/'
        data = {
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'password': password
        }

        try:
            response = requests.post(url, json=data)
            if response.status_code in [201, 226]:
                profile_tab_screen = self.manager.get_screen('profile_tab')
                profile_tab_screen.ids.full_name_label.text = full_name
                profile_tab_screen.ids.email_label.text = email

                self.login_after_signup(full_name, email, password)
            elif response.status_code == 400:
                # Email Validation
                try:
                    error_message = response.json().get('message', 'An error occurred.')
                    if 'Email already in use' in error_message:
                        self.display_error_message('Email already in use. Please use a different email.')
                    else:
                        self.display_error_message(error_message)
                except ValueError:
                    self.display_error_message('Invalid response received from the server.')
                    print(f'Error during signup: Invalid JSON response from server: {response.text}')
            else:
                self.display_error_message('An unexpected error occurred. Please try again.')
                print(f'Unexpected status code: {response.status_code}, response: {response.text}')
        except Exception as e:
            self.display_error_message('An error occurred during signup. Please try again later.')
            print(f'Error during signup: {e}')

    def login_after_signup(self, full_name, email, password):
        url = 'http://127.0.0.1:8000/login/'
        data = {'email': email, 'password': password}

        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                dashboard_screen = self.manager.get_screen('dashboard')
                dashboard_screen.ids.user_label.text = f'Hello, [b]{full_name}[/b]'
                self.manager.current = 'profile'
            else:
                try:
                    error_message = response.json().get('error', 'Unknown error')
                    print(f'Login after signup failed: {error_message}')
                except ValueError:
                    print('Login after signup failed: Invalid JSON response from server.')
        except Exception as e:
            print(f'Error during login after signup: {e}') 

    def display_error_message(self, message):
        self.ids.label_error.text = message 
        print(f'Error: {message}') 

class LoginScreen(Screen):
    def login(self, email, password):
        self.ids.label_error.text = '' 

        url = 'http://127.0.0.1:8000/login/'
        data = {'email': email, 'password': password}

        if not email or not password:
            self.ids.label_error.text = 'Please enter both email and password.'
            return

        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                user_data = response.json()
                full_name = user_data.get('full_name')

                profile_tab_screen = self.manager.get_screen('profile_tab')
                profile_tab_screen.ids.full_name_label.text = full_name
                profile_tab_screen.ids.email_label.text = email

                dashboard_screen = self.manager.get_screen('dashboard')
                dashboard_screen.ids.user_label.text = f'Hello, [b]{full_name}[/b]'

                dashboard_screen.fetch_pet_profile(email)

                self.manager.current = 'dashboard'
            else:
                self.ids.label_error.text = response.json().get('error', 'Login failed, please try again.')

        except Exception as e:
            self.ids.label_error.text = f'Error during login: {e}'


class ProfileScreen(Screen):
    def on_enter(self, *args):
        # Retrieve first_name, last_name, and email from the 'signup' screen
        signup_screen = self.manager.get_screen('signup')
        first_name = signup_screen.ids.first_name.text
        last_name = signup_screen.ids.last_name.text
        email = signup_screen.ids.email.text

        # Update the Labels to display the data
        self.ids.first_name_label.text = first_name
        self.ids.last_name_label.text = last_name
        self.ids.email_label.text = email

class PetProfileScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.selected_gender = None  
        self.selected_age_unit = None  

    def on_enter(self, *args):
        profile_screen = self.manager.get_screen('profile')
        email = profile_screen.ids.email_label.text

        self.ids.email_input.text = email

    def set_gender(self, gender):
        self.selected_gender = gender
        if gender == 'Male':
            self.ids.male_checkbox.active = True
            self.ids.female_checkbox.active = False
        elif gender == 'Female':
            self.ids.male_checkbox.active = False
            self.ids.female_checkbox.active = True

    def open_age_unit_menu(self):
        menu_items = [
            {'viewclass': 'OneLineListItem', 'text': 'Years', 'on_release': lambda: self.set_age_unit('Years')},
            {'viewclass': 'OneLineListItem', 'text': 'Months', 'on_release': lambda: self.set_age_unit('Months')},
        ]
        self.age_unit_menu = MDDropdownMenu(
            caller=self.ids.age_unit_dropdown,
            items=menu_items,
            width_mult=3,
        )
        self.age_unit_menu.open()

    def set_age_unit(self, unit):
        self.selected_age_unit = unit
        self.ids.age_unit_dropdown.text = unit
        self.age_unit_menu.dismiss()

    def save_pet_profile(self):
        # Get values from the input fields
        email = self.ids.email_input.text
        pet_name = self.ids.pet_name_input.text
        pet_breed = self.ids.pet_breed_input.text
        pet_age = self.ids.pet_age_input.text  
        age_unit = self.selected_age_unit  
        gender = self.selected_gender  
        # Validate all fields
        if not email or not pet_name or not pet_breed or not pet_age or not age_unit or not gender:
            print('Please fill in all fields before saving.')
            return

        # Pet Age Validation
        try:
            pet_age = int(pet_age)  
        except ValueError:
            print('Invalid age. Please enter a valid number for age.')
            return

        self.submit_pet_profile_to_api(email, pet_name, pet_breed, pet_age, age_unit, gender)

    def submit_pet_profile_to_api(self, email, pet_name, pet_breed, pet_age, age_unit, gender):
        url = 'http://127.0.0.1:8000/pet_profile/'  # API endpoint
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        data = {
            'email': email,
            'pet_name': pet_name,
            'pet_breed': pet_breed,
            'pet_age': pet_age,  
            'age_unit': age_unit,  
            'gender': gender
        }

        try:
            print(f'Sending request to {url} with data: {data}')  
            response = requests.post(url, json=data, headers=headers)

            print(f'Response status code: {response.status_code}')  
            print(f'Response content: {response.text}')  

            if response.status_code == 201:
                print('Pet profile saved successfully!')

                dashboard_screen = self.manager.get_screen('dashboard')
                dashboard_screen.ids.pet_name_label.text = f'Name: {pet_name}'
                dashboard_screen.ids.pet_breed_label.text = f'Breed: {pet_breed}'
                dashboard_screen.ids.pet_age_label.text = f'Age: {pet_age} {age_unit} old' 
                dashboard_screen.ids.pet_gender_label.text = f'Gender: {gender}'

                self.manager.current = 'dashboard'  # Navigate to dashboard screen
            else:
                print(f'Error: {response.text}')
        except Exception as e:
            print(f'Error during saving pet profile: {str(e)}')

class DashboardScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fetch_pet_profile(self, email):
        url = f'http://127.0.0.1:8000/pet_profile/?email={email}'

        try:
            response = requests.get(url)
            if response.status_code == 200:
                pet_profile = response.json()

                # Check if the response is a list or a dictionary
                if isinstance(pet_profile, list):
                    if len(pet_profile) > 0:
                        pet_profile = pet_profile[0]  
                    else:
                        print('No pet profiles found.')
                        return
                elif not isinstance(pet_profile, dict):
                    print('Unexpected response format. Expected a list or dictionary.')
                    return

                # Update labels with fetched pet profile data
                self.ids.pet_name_label.text = f"Name: {pet_profile.get('pet_name', 'N/A')}"
                self.ids.pet_breed_label.text = f"Breed: {pet_profile.get('breed', 'N/A')}"
                self.ids.pet_age_label.text = f"Age: {pet_profile.get('age', 'N/A')} {pet_profile.get('age_unit', '')} old"
                self.ids.pet_gender_label.text = f"Sex: {pet_profile.get('gender', 'N/A')}"

                print('Pet profile loaded successfully!')
            else:
                print('Failed to load pet profile:', response.json().get('error', 'Unknown error'))
        except Exception as e:
            print(f'Error fetching pet profile: {e}')

    def update_pet_data(self, pet_data):
        self.ids.pet_name_label.text = f"Name: {pet_data.get('pet_name', 'N/A')}"
        self.ids.pet_breed_label.text = f"Breed: {pet_data.get('breed', 'N/A')}"
        self.ids.pet_age_label.text = f"Age: {pet_data.get('age', 'N/A')} {pet_data.get('age_unit', '')} old"
        self.ids.pet_gender_label.text = f"Sex: {pet_data.get('gender', 'N/A')}"


    def go_to_live_feed(self):
        self.manager.current = 'live_feed'

    def profile_tab(self):
        self.manager.current = 'profile_tab'

    def home(self):
        self.manager.current = 'dashboard'

    def pet_tab(self):
        self.manager.current = 'pet_tab'

    def activity_logs(self):
        self.manager.current = 'activity_logs'

    def thermal(self):
        self.manager.current = 'thermal_camera'

class PetTabScreen(Screen):
    dialog = None

    def set_gender(self, gender):
        if gender == 'Male':
            self.ids.female_checkbox.active = False
        elif gender == 'Female':
            self.ids.male_checkbox.active = False

    def open_age_unit_menu(self):
        menu_items = [
            {'viewclass': 'OneLineListItem', 'text': 'Years', 'on_release': lambda: self.set_age_unit('Years')},
            {'viewclass': 'OneLineListItem', 'text': 'Months', 'on_release': lambda: self.set_age_unit('Months')},
        ]
        self.age_unit_menu = MDDropdownMenu(
            caller=self.ids.age_unit_dropdown,
            items=menu_items,
            width_mult=3,
        )
        self.age_unit_menu.open()

    def set_age_unit(self, unit):
        self.ids.age_unit_dropdown.text = unit
        self.age_unit_menu.dismiss()

    def update_pet_profile(self):
        pet_name = self.ids.pet_name_input.text.strip()
        new_pet_name = self.ids.new_pet_name_input.text.strip()
        pet_breed = self.ids.pet_breed_input.text.strip()
        pet_age = self.ids.pet_age_input.text.strip()
        gender = 'Male' if self.ids.male_checkbox.active else 'Female' if self.ids.female_checkbox.active else ''
        age_unit = self.ids.age_unit_dropdown.text.strip()

        # Validate data
        if not all([pet_name, new_pet_name, pet_breed, pet_age, gender, age_unit]):
            print('Error: All fields are required.')
            return

        try:
            pet_age = int(pet_age)
        except ValueError:
            print('Error: Age must be a number.')
            return

        data = {
            'pet_name': pet_name,
            'new_pet_name': new_pet_name,
            'pet_breed': pet_breed,
            'pet_age': pet_age,
            'gender': gender,
            'age_unit': age_unit,
        }

        url = 'http://127.0.0.1:8000/update_pet_profile/'

        try:
            response = requests.put(url, json=data)

            # Handle the response
            if response.status_code == 200:
                updated_data = response.json()
                self.load_pet_data(updated_data) 

                self.show_success_dialog()

                # Pass updated data to the DashboardScreen
                dashboard_screen = self.manager.get_screen("dashboard")
                dashboard_screen.update_pet_data(updated_data)

            else:
                error_message = response.json().get('error', 'An error occurred')
                print('Error:', error_message)

        except requests.exceptions.RequestException as e:
            print('Request failed:', str(e))

    def load_pet_data(self, data):
        self.ids.pet_name_input.text = data.get('pet_name', "")
        self.ids.new_pet_name_input.text = data.get('new_pet_name', "")
        self.ids.pet_breed_input.text = data.get('pet_breed', "")
        self.ids.pet_age_input.text = str(data.get('pet_age', ""))
        self.ids.male_checkbox.active = data.get('gender', "") == 'Male'
        self.ids.female_checkbox.active = data.get('gender', "") == 'Female'
        self.ids.age_unit_dropdown.text = data.get('age_unit', "")

    def show_success_dialog(self):
        if not self.dialog:
            self.dialog = MDDialog(
                text = 'Pet Profile Updated Successfully!',
                size_hint = (0.8, None),
                height = '200dp',
                buttons = [
                    {
                        'text': 'OK',
                        'on_release': self.close_dialog,
                    }
                ],
            )
        self.dialog.open()

    def close_dialog(self, *args):
        if self.dialog:
            self.dialog.dismiss()

    def go_to_live_feed(self):
        self.manager.current = 'live_feed'

    def profile_tab(self):
        self.manager.current = 'profile_tab'

    def pet_tab(self):
        self.manager.current = 'pet_tab'

    def activity_logs(self):
        self.manager.current = 'activity_logs'

    def home(self):
        self.manager.current = 'dashboard'

class ProfileTabScreen(Screen):
    def on_enter(self, *args):
        if not self.ids.full_name_label.text:
            self.ids.full_name_label.text = 'No name available'
        if not self.ids.email_label.text:
            self.ids.email_label.text = 'No email available'
    
    def go_to_live_feed(self):
        self.manager.current = 'live_feed'
    
    def profile_tab(self):
        self.manager.current = 'profile_tab'

    def home(self):
        self.manager.current = 'dashboard'
    
    def pet_tab(self):
        self.manager.current = 'pet_tab'

    def activity_logs(self):
        self.manager.current = 'activity_logs'

class AboutUsScreen(Screen):
    pass

class HelpScreen(Screen):
    pass

class UserGuideScreen(Screen):
    pass

class FAQsScreen(Screen):
    pass

class EditProfileScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_enter(self):
        # Load current user data into the edit fields
        profile_tab_screen = self.manager.get_screen('profile_tab')
        full_name = profile_tab_screen.ids.full_name_label.text
        first_name, last_name = self.split_full_name(full_name)
        self.ids.edit_first_name_input.text = first_name
        self.ids.edit_last_name_input.text = last_name
        self.ids.edit_email_input.text = profile_tab_screen.ids.email_label.text

    def split_full_name(self, full_name):
        """Split the full name into first and last names.
        The last name is always the last word, and the rest is the first name."""
        names = full_name.split()
        if not names:
            return '', ''
        
        # Last name is the last word
        last_name = names[-1]
        
        # First name is everything except the last word
        first_name = ' '.join(names[:-1]) if len(names) > 1 else names[0]
        
        return first_name, last_name

    def save_changes(self):
        current_email = self.manager.get_screen('profile_tab').ids.email_label.text
        new_first_name = self.ids.edit_first_name_input.text.strip()
        new_last_name = self.ids.edit_last_name_input.text.strip()
        new_email = self.ids.edit_email_input.text.strip()

        print(f'Attempting to update profile with first name: {new_first_name}, last name: {new_last_name}, email: {new_email}') 

        # Validate inputs
        if not new_first_name or not new_last_name or not new_email:
            self.show_popup('Please fill in all fields!')
            return

        if not (new_email.endswith('@yahoo.com') or new_email.endswith('@gmail.com')):
            self.show_popup('Please use a valid @yahoo.com or @gmail.com email.')
            return

        url = 'http://127.0.0.1:8000/update_profile/'

        data = {
            'current_email': current_email,
            'new_first_name': new_first_name,
            'new_last_name': new_last_name,
            'new_email': new_email
        }

        try:
            print(f'Sending request to {url} with data: {data}')  
            response = requests.put(url, json=data)
            print(f'Response status: {response.status_code}')  

            if response.status_code == 200:
                new_full_name = f'{new_first_name} {new_last_name}'
                self.update_profile_information(new_full_name, new_email)
                self.show_popup('Profile updated successfully!')
                Clock.schedule_once(lambda dt: self.return_to_profile(), 1)
            else:
                error_message = response.json().get('error', 'Failed to update profile')
                self.show_popup(f'Error: {error_message}')

        except requests.exceptions.RequestException as e:
            print(f'Request exception: {str(e)}') 
            self.show_popup('Network error. Please check your connection.')

        except Exception as e:
            print(f'Unexpected error: {str(e)}')  
            self.show_popup(f'An error occurred: {str(e)}')

    def update_profile_information(self, new_full_name, new_email):
        try:
            # Update ProfileTabScreen
            profile_tab_screen = self.manager.get_screen('profile_tab')
            profile_tab_screen.ids.full_name_label.text = new_full_name
            profile_tab_screen.ids.email_label.text = new_email

            # Update Dashboard
            dashboard_screen = self.manager.get_screen('dashboard')
            dashboard_screen.ids.user_label.text = f'Hello, [b]{new_full_name}[/b]'

            print('Profile information updated successfully')  
        except Exception as e:
            print(f'Error updating profile information: {str(e)}')  
            raise

    def return_to_profile(self):
        self.manager.current = 'profile_tab'

    def show_popup(self, message):
        popup = Popup(
            title='Message',
            content=Label(text=message),
            size_hint=(None, None),
            size=(400, 200),
        )
        popup.open()
        print(f'Showing popup with message: {message}') 

    def cancel_changes(self):
        self.manager.current = 'profile_tab'

class ForgotPasswordScreen(Screen):
    def reset_password(self, email, new_password, confirm_password):
        self.ids.label_error.text = ''

        url = 'http://127.0.0.1:8000/forgot_password/'
        data = {'email': email, 'new_password': new_password, 'confirm_password': confirm_password}

        if not email or not new_password or not confirm_password:
            self.ids.label_error.text = 'All fields are required.'
            return

        if new_password != confirm_password:
            self.ids.label_error.text = 'Passwords do not match.'
            return
        
        if len(new_password) < 8:
            self.ids.label_error.text = 'Password must be at least 8 characters long.'
            return

        if not (email.endswith('@gmail.com') or email.endswith('@yahoo.com')):
            self.ids.label_error.text = 'Email must be a @gmail.com or @yahoo.com address.'
            return

        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                self.ids.label_error.text = "Password reset successful!"
                self.manager.transition.direction = "right"
                self.manager.current = "login"
            else:
                self.ids.label_error.text = response.json().get('error', 'Password reset failed, please try again.')

        except Exception as e:
            self.ids.label_error.text = f'Error during password reset: {e}'

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        print('MainScreen Initialized')

class PetWatch(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dialog = None  
        self.dog_temp = None  
        self.dog_behavior = None  
        self.activity_log = [] 
        self.shared_data = {"dog_bboxes": []}
        self._temperature_update_event = None 

        return Builder.load_file('main.kv')
    
    def build(self):
        screen_manager = ScreenManager(transition=NoTransition())
        screen_manager.add_widget(MainScreen(name='main'))
        screen_manager.add_widget(LoginScreen(name='login'))
        screen_manager.add_widget(SignupScreen(name='signup'))
        screen_manager.add_widget(ProfileScreen(name='profile'))
        screen_manager.add_widget(PetProfileScreen(name='pet_profile'))
        screen_manager.add_widget(DashboardScreen(name='dashboard'))
        screen_manager.add_widget(LiveFeedScreen(name='live_feed'))
        screen_manager.add_widget(PetTabScreen(name='pet_tab'))
        screen_manager.add_widget(ActivityLogsScreen(name='activity_logs'))
        screen_manager.add_widget(ProfileTabScreen(name='profile_tab'))
        screen_manager.add_widget(AboutUsScreen(name='about_us'))
        screen_manager.add_widget(HelpScreen(name='help_button'))
        screen_manager.add_widget(EditProfileScreen(name='edit_profile'))
        screen_manager.add_widget(UserGuideScreen(name='userguide'))
        screen_manager.add_widget(FAQsScreen(name='faqs'))
        screen_manager.add_widget(ThermalCameraScreen(name = 'thermal_camera'))
        screen_manager.add_widget(ForgotPasswordScreen(name='forgot_password'))

        if hasattr(Window, 'metrics'):
            density = Window.metrics['density']
            print(f"Screen density: {density}")
        else:
            print("Window.metrics not available, using default density of 1")
            density = 1

        scaled_padding = 10 * density
        print(f"Scaled padding: {scaled_padding}")

        return screen_manager
    
    def start_temperature_updates(self):
        if not self._temperature_update_event:
            self._temperature_update_event = Clock.schedule_interval(self.update_temperature_data, 1)

    def stop_temperature_updates(self):
        if self._temperature_update_event:
            self._temperature_update_event.cancel()
            self._temperature_update_event = None

    # Inside the update_temperature_data method in PetWatch
    def update_temperature_data(self, dt):
        # Fetch thermal data
        frame_data = self.fetch_thermal_data_from_server()
        if frame_data:
            frame_data = np.array(frame_data).reshape((24, 32))  # Reshape thermal data

            # Get bounding boxes for the dogs
            dog_bboxes = self.shared_data.get("dog_bboxes", [])
            max_dog_temp = None

            # Process each bounding box and calculate the max temperature within it
            for bbox in dog_bboxes:
                # Scale bounding box from live resolution (e.g., 640x480) to thermal resolution (32x24)
                scaled_bbox = scale_bbox_to_thermal(bbox, 2880, 1620, 32, 24)
                x1, y1, x2, y2 = scaled_bbox

                # Extract the region from the thermal data
                dog_region = frame_data[y1:y2, x1:x2]
                if dog_region.size > 0:
                    # Calculate max temperature within the bounding box
                    dog_temp = np.max(dog_region) + 8  # Add some offset if needed
                    if max_dog_temp is None or dog_temp > max_dog_temp:
                        max_dog_temp = dog_temp

            # Update the app with the maximum dog temperature found
            self.dog_body_temp = f"{max_dog_temp:.2f}°C" if max_dog_temp else "Undetected"
            self.update_screens_with_temperature(self.dog_body_temp)  # Update both screens

            # Update the dashboard and status
            app = PetWatch.get_running_app()
            dashboard_screen = app.root.get_screen('dashboard')

            if max_dog_temp:
                dashboard_screen.ids.dashboard_temp.text = f"{max_dog_temp:.2f}°C"

                # Temperature range checks
                if 38.3 <= max_dog_temp <= 39.2:  # Normal range
                    dashboard_screen.ids.status_label.text = 'Normal'
                    dashboard_screen.ids.status_icon.icon = 'check-circle'
                    dashboard_screen.ids.status_icon.icon_color = [34/255, 139/255, 34/255, 1]  # Green
                    dashboard_screen.ids.status_label.text_color = [34/255, 139/255, 34/255, 1]
                elif 37.0 <= max_dog_temp <= 37.9:  # Hypothermia warning range
                    dashboard_screen.ids.status_label.text = 'Warning'
                    dashboard_screen.ids.status_icon.icon = 'alert-circle'
                    dashboard_screen.ids.status_icon.icon_color = [255/255, 165/255, 0, 1]  # Orange
                    dashboard_screen.ids.status_label.text_color = [255/255, 165/255, 0, 1]
                elif 39.3 <= max_dog_temp <= 40.5:  # Hyperthermia warning range
                    dashboard_screen.ids.status_label.text = 'Warning'
                    dashboard_screen.ids.status_icon.icon = 'alert-circle'
                    dashboard_screen.ids.status_icon.icon_color = [255/255, 165/255, 0, 1]  # Orange
                    dashboard_screen.ids.status_label.text_color = [255/255, 165/255, 0, 1]
                elif max_dog_temp < 36.7 or max_dog_temp > 41.0:  # Critical range
                    dashboard_screen.ids.status_label.text = 'Critical'
                    dashboard_screen.ids.status_icon.icon = 'alert'
                    dashboard_screen.ids.status_icon.icon_color = [255/255, 0, 0, 1]  # Red
                    dashboard_screen.ids.status_label.text_color = [255/255, 0, 0, 1]
            else:
                dashboard_screen.ids.dashboard_temp.text = 'Undetected'
                dashboard_screen.ids.status_label.text = 'Undetected'
                dashboard_screen.ids.status_icon.icon = 'cancel'
                dashboard_screen.ids.status_icon.icon_color = [122/255, 118/255, 117/255, 1]  # Gray
                dashboard_screen.ids.status_label.text_color = [122/255, 118/255, 117/255, 1]

    def fetch_thermal_data_from_server(self):
        try:
            #ESP32 IP address
            ESP32_ip = 'http://192.168.0.200' 
            endpoint = '/thermal'

            # Send HTTP GET request to fetch thermal data
            response = requests.get(ESP32_ip + endpoint)

            if response.status_code == 200:
                # Process and return the thermal data
                data = response.text
                return [float(i) for i in data.split(',')]
            else:
                print(f'Error: {response.status_code}')
        except requests.exceptions.RequestException as e:
            print(f'Error fetching data: {e}')
        return None

    def update_screens_with_temperature(self, dog_temp):
        try:
            # Update dashboard temperature label
            dashboard_screen = self.root.get_screen('dashboard')
            if dashboard_screen.ids.dashboard_temp:
                dashboard_screen.ids.dashboard_temp.text = f"{dog_temp}"  # Directly use the dog_temp

            # Update thermal screen temperature label
            thermal_screen = self.root.get_screen('thermal_camera')
            if thermal_screen.ids.dog_temperature_label:
                thermal_screen.ids.dog_temperature_label.text = f": {dog_temp}" if dog_temp else ": Undetected"
        except Exception as e:
            print(f"Error updating screens: {e}")

    def on_start(self):
        self.start_temperature_updates()

    def show_logout_confirmation(self):
        if not self.dialog:
            brown_color = (79/255, 54/255, 48/255, 1)  

            self.dialog = MDDialog(
                text="[color=#6d3b07]Are you sure you want to proceed?[/color]",
                buttons=[
                    MDRaisedButton(
                        text="CANCEL",
                        font_name = 'BPoppins',
                        on_release=self.close_dialog,
                        md_bg_color=brown_color,  
                        size_hint=(None, None),  
                        size=(dp(120), dp(48))   
                    ),
                    MDRaisedButton(
                        text="LOGOUT",
                        font_name = 'BPoppins',
                        on_release=self.logout,
                        md_bg_color=brown_color,  
                        size_hint=(None, None),  
                        size=(dp(120), dp(48))   
                    ),
                ],
            )
        self.dialog.open()

    def close_dialog(self, obj=None):
        if self.dialog:
            self.dialog.dismiss()
            self.dialog = None  

    def logout(self, obj=None):
        self.close_dialog()  
        print('Logging out...')
        self.root.current = 'login'  

    def next_step(self):
        self.root.current = "pet_profile"

    def prev_step(self):
        self.root.current = "profile"

    def next_step_dash(self):
        self.root.current = "dashboard"
    
    def goto_about_us(self):
        self.root.current = "about_us"
    
    def activity_logs(self):
        self.root.current = "activity_logs"
    
    def goto_help(self):
        self.root.current = "help_button"
    
    def cancel_changes(self):
        self.root.current = "profile_tab"

if __name__ == '__main__':
    LabelBase.register(name='BPoppins', fn_regular='fonts/Poppins/Poppins-SemiBold.ttf')
    LabelBase.register(name='MPoppins', fn_regular='fonts/Poppins/Poppins-Medium.ttf')
    PetWatch().run()
