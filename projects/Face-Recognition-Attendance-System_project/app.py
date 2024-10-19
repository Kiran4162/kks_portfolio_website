import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# VARIABLES
MESSAGE = "WELCOME. Instruction: to register your attendance kindly click on 'a' on keyboard"

# Defining Flask App
app = Flask(__name__)

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1) if cv2.VideoCapture(1).isOpened() else cv2.VideoCapture(0)

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# Get the total number of registered users
def totalreg():
    return len(os.listdir('static/faces'))

# Extract the face from an image
def extract_faces(img):
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    return []

# Function to identify a face using the model
def identify_face(facearray, threshold=0.8):  # Adding threshold for unknown face identification
    model = tf.keras.models.load_model('static/face_recognition_model.h5')

    # Ensure input shape matches expected model input
    print(f"Shape before reshaping: {facearray.shape}")  # Debugging shape
    if len(facearray.shape) == 3:
        facearray = facearray.reshape(1, 160, 160, 3)  # Add batch dimension

    facearray = facearray / 255.0  # Normalize the image

    predictions = model.predict(facearray)
    max_confidence = np.max(predictions)  # Get the highest probability

    print(f"Model output: {predictions}")  # Debugging model output
    print(f"Max confidence: {max_confidence}")  # Debugging confidence

    if max_confidence < threshold:  # If the max confidence is below the threshold
        print("Unknown face detected.")
        return "unidentified"  # Return a specific value for unidentified faces
    
    identified_person_idx = np.argmax(predictions, axis=1)[0]
    
    # Fetch the list of users (username_userid) based on the directories
    userlist = os.listdir('static/faces')
    
    # Ensure the prediction index doesn't exceed the number of users
    if identified_person_idx < len(userlist):
        return userlist[identified_person_idx]  # Return user string 'name_id'
    else:
        return None

# Define the model architecture using MobileNetV2
def create_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3))

    for layer in base_model.layers:
        layer.trainable = False  # Freeze the base layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)  # Add dropout for regularization
    output_layer = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output_layer)

    return model

# Function to train the model
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')

    for user in userlist:
        user_folder = os.path.join('static/faces', user)
        if os.path.isdir(user_folder):
            for imgname in os.listdir(user_folder):
                img_path = os.path.join(user_folder, imgname)
                img = cv2.imread(img_path)
                resized_face = cv2.resize(img, (160, 160))
                faces.append(resized_face)
                labels.append(user)

    faces = np.array(faces) / 255.0
    labels = np.array(labels)

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)

    num_classes = len(le.classes_)

    if os.path.exists('static/face_recognition_model.h5'):
        model = tf.keras.models.load_model('static/face_recognition_model.h5')
        
        # Modify the output layer to match new number of classes
        x = model.layers[-2].output  # Access the last trainable layer (before output)
        new_output_layer = Dense(num_classes, activation='softmax', name = 'output_dense')(x)
        model = Model(inputs=model.input, outputs=new_output_layer)
        
        for layer in model.layers[:-1]:  # Freeze MobileNetV2 layers
            layer.trainable = False
        
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    else:
        model = create_model(num_classes)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Check shapes
    print(f"Faces shape: {faces.shape}, Labels shape: {labels_categorical.shape}")
    
    model.fit(faces, labels_categorical, epochs=10, batch_size=32)
    model.save('static/face_recognition_model.h5')

# Extract attendance info
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    return names, rolls, times, len(df)

# Add attendance for a user
# def add_attendance(name):
#     if isinstance(name, (str, bytes)):
#         username = name.split('_')[0]
#         userid = name.split('_')[-1]
#         current_time = datetime.now().strftime("%H:%M:%S")
        
#         df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
#         if str(userid) not in list(df['Roll']):
#             with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
#                 f.write(f'\n{username},{userid},{current_time}')
#         else:
#             print("This user has already marked attendance for the day.")
#     else:
#         print(f"Invalid name format: {name}. Expected a string with '_' separator.")

# Add attendance for a user
# def add_attendance(name):
#     if isinstance(name, (str, bytes)):
#         # username = name.split('_')[0]
#         # userid = name.split('_')[-1]
#         username, userid = name.split('_')

#         # Convert userid to integer if possible
#         try:
#             userid = int(userid)
#         except ValueError:
#             print(f"Invalid userid format: {userid}. Expected a numeric string.")
#             return
        
#         current_time = datetime.now().strftime("%H:%M:%S")
        
#         df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')

#         # Check if the user has already marked attendance for today
#         if str(userid) in list(df['Roll']):
#             # Further check if today's date matches
#             if df[df['Roll'] == str(userid)]['Time'].str.contains(datetoday2).any():
#                 print("This user has already marked attendance for today.")
#                 return "Attendance already marked"  # Optional message for UI feedback

#         # If the user hasn't marked attendance, add a new entry
#         with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
#             f.write(f'\n{username},{userid},{current_time}')
#             print(f"Attendance marked for {username} ({userid}) at {current_time}.")
#     else:
#         print(f"Invalid name format: {name}. Expected a string with '_' separator.")

# Add attendance for a user
def add_attendance(name):
    if isinstance(name, (str, bytes)):
        username = name.split('_')[0]
        userid = name.split('_')[-1]
        current_time = datetime.now().strftime("%H:%M:%S")
        
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
        
        # Check if this user has already marked attendance today
        attendance_today = df[(df['Roll'] == str(userid)) & (df['Time'].str.contains(datetoday2))]
        
        if attendance_today.empty:
            # If the user has not marked attendance today, append to the file
            with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
                f.write(f'\n{username},{userid},{current_time}')
            print(f"Attendance marked for {username} ({userid}) at {current_time}.")
        else:
            print("This user has already marked attendance for the day.")
    else:
        print(f"Invalid name format: {name}. Expected a string with '_' separator.")



################## ROUTING FUNCTIONS ##############################

# app = Flask(__name__)
# MESSAGE = "Welcome!"

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/home')
def home():
    names, rolls, times, l = extract_attendance()
    zipped_data = zip(names, rolls, times)
    return render_template('home.html', zipped_data=zipped_data, l=l,
                           totalreg=totalreg(), datetoday2=datetime.now().strftime("%Y-%m-%d"), mess=MESSAGE)

@app.route('/start', methods=['GET'])
def start():
    global MESSAGE  
    ATTENDANCE_MARKED = False

    if 'face_recognition_model.h5' not in os.listdir('static'):
        names, rolls, times, l = extract_attendance()
        MESSAGE = 'This face is not registered with us, kindly register yourself first.'
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
                               totalreg=totalreg(), datetoday2=datetime.now().strftime("%Y-%m-%d"), mess=MESSAGE)

    cap = cv2.VideoCapture(1)  # Changed to 0 for the default camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face = cv2.resize(frame[y:y+h, x:x+w], (160, 160))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
            identified_person = identify_face(face)
            if identified_person == "unidentified":
                cv2.putText(frame, 'Unidentified', (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                username, userid = identified_person.split('_')
                cv2.putText(frame, f'{username} ({userid})', (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # if not ATTENDANCE_MARKED:
                #     add_attendance(identified_person)
                #     ATTENDANCE_MARKED = True
                
                # if cv2.waitKey(1) & 0xFF == ord('a'):  # Check if 'a' is pressed
                #     response = add_attendance(identified_person)
                #     if response == "Attendance already marked":
                #         print("Attendance already marked for today.")
                #         break
                #     else:
                #         current_time_ = datetime.now().strftime("%H:%M:%S")
                #         ATTENDANCE_MARKED = True
                #         print(f"Attendance marked for {identified_person} at {current_time_}")
                #         break
                if cv2.waitKey(1) & 0xFF == ord('a'):  # Check if 'a' is pressed
                    add_attendance(identified_person)
                    current_time_ = datetime.now().strftime("%H:%M:%S")
                    ATTENDANCE_MARKED = True
                    print(f"Attendance marked for {identified_person} at {current_time_}")
                    break

                    names, rolls, times, l = extract_attendance()
                    # return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
                    #                        totalreg=totalreg(), datetoday2=datetime.now().strftime("%Y-%m-%d"), mess=MESSAGE)


        cv2.imshow('Attendance System', frame)
        cv2.putText(frame, 'Press "a" to mark attendance', (30, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255))
        # key = cv2.waitKey(1) & 0xFF

        # if key == ord('q'):
        #     break
        if cv2.waitKey(1) & 0xFF == ord('q') or ATTENDANCE_MARKED:
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
                           totalreg=totalreg(), datetoday2=datetime.now().strftime("%Y-%m-%d"), mess=MESSAGE)

@app.route('/attendance', methods=['GET'])
def attendance():
    names, rolls, times, l = extract_attendance()
    zipped_data = zip(names, rolls, times)
    return render_template('attendance.html', zipped_data=zipped_data, l=l)

@app.route('/add', methods=['GET', 'POST'])
def add():
    global MESSAGE
    
    if request.method == 'POST':
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
        
        # if newusername and newuserid:
        #     user_id = f"{newusername}_{newuserid}"
        #     # user_folder = f'static/faces/{user_id}'
        #     user_folder = userimagefolder
        #     if not os.path.isdir(user_folder):
        #         os.makedirs(user_folder)
        
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        
        cap = cv2.VideoCapture(1)  # Changed to 0 for the default camera
        if not cap.isOpened():
            MESSAGE = "Cannot access the camera."
            return render_template('home.html', mess=MESSAGE)
        
        i, j = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            faces = face_detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.3, 5)
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f'Images Captured: {i}/100', (30, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                
                face = cv2.resize(frame[y:y+h, x:x+w], (160, 160))
                
                cv2.imwrite(os.path.join(userimagefolder, f'{newusername}_{newuserid}_{i}.jpg'), face)
                i += 1
                
                if i >=100:
                    break
            
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == ord('q') or i >= 100:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Training the model with the new images...")
        
        # Train the model with new data
        train_model()
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
                                   totalreg=totalreg(), datetoday2=datetime.now().strftime("%Y-%m-%d"),
                                   mess="Successfully Registered")
    
    return render_template('add.html', mess=MESSAGE)

if __name__ == '__main__':
    app.run(debug=True)
