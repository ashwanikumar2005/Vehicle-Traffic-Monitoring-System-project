import cv2
from sort import *
import numpy as np
from ultralytics import YOLO
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import sys
import time
import csv

video_path = None

# Function to open a file dialog and select video file
def choose_file():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
    if video_path:
        file_label.config(text=f"Selected: {video_path}")

# Vehicle monitoring function
def monitor_traffic():
    global running, vehicle_count, car_count, bus_count, truck_count

    # Initialize video capture and YOLO model
    cap = cv2.VideoCapture(video_path)
    model = YOLO('yolov8n.pt')

    # Load class names
    classnames = []
    with open('classes.txt', 'r') as f:
        classnames = f.read().splitlines()

    # Initialize SORT tracker
    tracker = Sort(max_age=30)

    # Vehicle count
    vehicle_count = set()
    car_count = 0
    bus_count = 0
    truck_count = 0

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame for display
        display_width, display_height = 600, 350
        frame = cv2.resize(frame, (display_width, display_height))

        detections = np.empty((0, 5))
        results = model(frame, stream=1)

        # Detect "car", "truck", and "bus" classes
        detected_vehicles = {}
        for info in results:
            boxes = info.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                classindex = box.cls[0]
                object_detected = classnames[int(classindex)]

                if object_detected in ['car', 'truck', 'bus'] and conf > 0.6:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    new_detections = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, new_detections))
                    detected_vehicles[(x1, y1, x2, y2)] = object_detected

        # Update tracker
        track_result = tracker.update(detections)

        # Draw bounding boxes and track vehicles
        for result in track_result:
            x1, y1, x2, y2, obj_id = map(int, result)
            vehicle_type = detected_vehicles.get((x1, y1, x2, y2))

            # Match object ID with vehicle type from detected vehicles
            vehicle_type = None
            for detected_box, detected_type in detected_vehicles.items():
                if (x1 >= detected_box[0] and y1 >= detected_box[1] and
                        x2 <= detected_box[2] and y2 <= detected_box[3]):
                    vehicle_type = detected_type
                    break

            # If vehicle type was found, log the vehicle
            if vehicle_type:
                if obj_id not in vehicle_count:
                    vehicle_count.add(obj_id)
                    log_vehicle(obj_id, vehicle_type)

                    # Update vehicle counts based on the detected type
                    if vehicle_type == 'car':
                        car_count += 1
                    elif vehicle_type == 'bus':
                        bus_count += 1
                    elif vehicle_type == 'truck':
                        truck_count += 1

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        try:
            # Update counts in the UI
            vehicle_count_var.set(len(vehicle_count))
            car_count_var.set(car_count)
            bus_count_var.set(bus_count)
            truck_count_var.set(truck_count)

            # Convert frame to ImageTk format for Tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
            root.update()
        except:
            pass

    cap.release()
    cv2.destroyAllWindows()

# Function to log vehicle entry with timestamp
def log_vehicle(obj_id, vehicle_type):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')  # Get current time
    # Avoid duplicate entries for the same vehicle ID
    for child in log_list.get_children():
        if log_list.item(child)["values"][0] == obj_id:
            return
    log_list.insert("", "end", values=(obj_id, vehicle_type, timestamp))
    vehicle_count_var.set(len(vehicle_count))

# Function to start the traffic monitoring thread
def start_monitoring():
    global running
    if video_path:
        running = True
        threading.Thread(target=monitor_traffic).start()
    else:
        file_label.config(text="Please select a video file first!", fg="red")

# Function to stop monitoring
def stop_monitoring():
    global running
    running = False
    sys.exit(1)

# Function to download logs as CSV
def download_logs():
    # Get the current logs from the Treeview
    logs = []
    for child in log_list.get_children():
        logs.append(log_list.item(child)["values"])

    # Open a file dialog to save the CSV
    save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
    if save_path:
        with open(save_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("Object ID", "Type of Vehicle", "Timestamp"))  # Header
            writer.writerows(logs)  # Log data

# Initialize Tkinter window
root = Tk()
root.title("Vehicle Traffic Monitoring System")
root.geometry("1024x600")
root.resizable(True, True)

# Style configuration
style = ttk.Style()
style.configure("Treeview.Heading", font=("Arial", 10, "bold"))
style.configure("Treeview", font=("Arial", 10))

# Create Notebook for tabs
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

# Tab 1: Video Monitoring
monitor_tab = Frame(notebook)
notebook.add(monitor_tab, text="Monitoring")

# Tab 2: Log Records
log_tab = Frame(notebook)
notebook.add(log_tab, text="Logs")

# Video selection frame
file_frame = Frame(monitor_tab)
file_frame.pack(fill="x", pady=5)

file_label = Label(file_frame, text="No file selected", font=("Arial", 12))
file_label.pack(side="left", padx=10)
file_button = Button(file_frame, text="Choose File", command=choose_file, font=("Arial", 12), bg="blue", fg="white")
file_button.pack(side="right", padx=10)

# Video display frame


video_frame = Frame(monitor_tab, bg="black")
video_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Video frame title label
title_label = Label(video_frame, text="Vehicle Traffic  Monitoring System", font=("Arial", 18, "bold"), bg="black", fg="white")
title_label.pack(fill="x", pady=10)  # Add some padding to create space above the video


video_label = Label(video_frame, bg="black")
video_label.pack(fill="both", expand=True)

# Vehicle count frame
count_frame = Frame(monitor_tab)
count_frame.pack(pady=10)

vehicle_count_var = IntVar(value=0)
car_count_var = IntVar(value=0)
bus_count_var = IntVar(value=0)
truck_count_var = IntVar(value=0)

Label(count_frame, text="Total Vehicles:", font=("Arial", 14)).pack(side="left", padx=5)
Label(count_frame, textvariable=vehicle_count_var, font=("Arial", 14), fg="red").pack(side="left")

Label(count_frame, text="Cars:", font=("Arial", 14)).pack(side="left", padx=10)
Label(count_frame, textvariable=car_count_var, font=("Arial", 14), fg="blue").pack(side="left")

Label(count_frame, text="Buses:", font=("Arial", 14)).pack(side="left", padx=10)
Label(count_frame, textvariable=bus_count_var, font=("Arial", 14), fg="green").pack(side="left")

Label(count_frame, text="Trucks:", font=("Arial", 14)).pack(side="left", padx=10)
Label(count_frame, textvariable=truck_count_var, font=("Arial", 14), fg="purple").pack(side="left")

# Log frame in "Logs" tab
log_list_frame = Frame(log_tab)
log_list_frame.pack(padx=10, pady=10, fill="both", expand=True)

log_list = ttk.Treeview(log_list_frame, columns=("Object ID", "Vehicle Type", "Timestamp"), show="headings")
log_list.heading("Object ID", text="Object ID")
log_list.heading("Vehicle Type", text="Vehicle Type")
log_list.heading("Timestamp", text="Timestamp")
log_list.pack(fill="both", expand=True)

log_buttons_frame = Frame(log_tab)
log_buttons_frame.pack(pady=10)

log_button = Button(log_buttons_frame, text="Download Logs", command=download_logs, font=("Arial", 12), bg="blue", fg="white")
log_button.pack(side="left", padx=10)

# Start and Stop buttons
button_frame = Frame(monitor_tab)
button_frame.pack(pady=20)

start_button = Button(button_frame, text="Start", command=start_monitoring, font=("Arial", 14), bg="green", fg="white")
start_button.pack(side="left", padx=10)

stop_button = Button(button_frame, text="Stop", command=stop_monitoring, font=("Arial", 14), bg="red", fg="white")
stop_button.pack(side="left", padx=10)

root.mainloop()
