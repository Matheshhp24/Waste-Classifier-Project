import cv2
from flask import Flask,render_template,request,redirect,url_for,Response,session

from roboflow import Roboflow
from werkzeug.utils import secure_filename



app = Flask(__name__)

namelist = []
numbers = []



@app.template_filter('zip')
def _zip(a, b):
    return zip(a, b)


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/upload', methods=['POST','GET'])
def upload():
    
        
    rf = Roboflow(api_key="n71Ls7tYoQ2qkifAa2a2")
    project = rf.workspace().project("garbage-classification-3")
    model_WCP = project.version(2).model
    
    global namelist
    namelist=[]
    global wasteimage
    wasteimage = []
    global numbers 
    numbers = []
    
    for i in range(1,100):
        numbers.append(i)
        
    global image_extensions
    image_extensions = ['jpeg', 'jpg', 'png', 'gif', 'tiff', 'tif', 'bmp', 'svg', 'webp', 'heic', 'heif', 'raw', 'psd']
    global video_extensions
    video_extensions = ['mp4', 'avi', 'mov', 'wmv', 'mkv', 'flv', 'webm', 'm4v', '3gp', 'mpeg', 'mpg']


    
    if request.method=='POST':
        input_image = request.files["file"]
        global fileExtension
        fileExtension = secure_filename(input_image.filename).split('.')[-1].lower()
        if fileExtension in image_extensions:
            print('Detecting the waste on given image')
            global htmlfile
            htmlfile = 'image.html'
            global input_image_filepath
            input_image_filepath = "static/images/Input_Images/input."+fileExtension
            global output_image_filepath
            output_image_filepath = "static/images/Output_Images/output."+fileExtension
            input_image.save(input_image_filepath)
            my_Prediction =  model_WCP.predict(input_image_filepath, confidence=40, overlap=30).json()
            model_WCP.predict(input_image_filepath, confidence=40, overlap=30).save(output_image_filepath)
            for i in range(0,30):
                try:
                    namelist.append(my_Prediction['predictions'][i]['class'])
                except IndexError:
                    break
        elif fileExtension in video_extensions:
            print('Detecting the waste on given video')
            output_image_filepath = "static/images/Output_Images/output."+fileExtension
            input_image_filepath = "static/images/Input_Images/input."+fileExtension
            print("Analyzing...")
            input_image.save(input_image_filepath)
            
            htmlfile = 'video.html'
            video_frame()
    return render_template(htmlfile ,video_extensions=video_extensions , fileExtension=fileExtension, image_extensions=image_extensions, names=namelist, numbers = numbers, op=output_image_filepath)

@app.route('/results')
def results():
    return render_template('/results.html',names=namelist,numbers=numbers)


def video_frame():
    rf = Roboflow(api_key="n71Ls7tYoQ2qkifAa2a2")
    project = rf.workspace().project("garbage-classification-3")
    model_WCP = project.version(2).model
    cap = cv2.VideoCapture(input_image_filepath)  # Open the webcam (change the index if multiple webcams are available)
    frame_count = 0
    global numbers 
    numbers = []
    global namelist
    namelist=[]
    
    for i in range(1,100):
        numbers.append(i)
    org = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.5
    color = (0, 255, 0)
    thickness = 3
    lineType = cv2.LINE_AA
    while True:
        ret, frame = cap.read() 
        print(ret)# Read frames from the webcam
        if not ret:
            cap.release()
            break
            
        frame_count += 1
        if frame_count %200 !=0:
            continue
        cv2.imwrite('temp.jpg', frame)
        predictions = model_WCP.predict('temp.jpg')
        
        

        for bounding_box in predictions:
            x0 = bounding_box['x'] - bounding_box['width'] / 2
            x1 = bounding_box['x'] + bounding_box['width'] / 2
            y0 = bounding_box['y'] - bounding_box['height'] / 2
            y1 = bounding_box['y'] + bounding_box['height'] / 2
            cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), thickness=3)
            class_name = bounding_box['class']
            # confidence_score = bounding_box['confidence']
            # detection_results = bounding_box
            # class_and_confidence = (class_name, confidence_score)
            # print(class_name, '\n')
            text = str(class_name)
            
            namelist.append(class_name)
            

            
            cv2.putText(frame, text, org, font, fontScale, color, thickness, lineType)

        processed_frame = process_frame(frame)  # Process the frame
        ret, buffer = cv2.imencode('.jpg', processed_frame)  # Encode the processed frame as JPEG

        # Yield the encoded frame as a byte string
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        print(namelist)
    


@app.route('/video_show', methods=['POST','GET'])
def video_show():
    return Response(video_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
       
   
    
def process_frame(frame):
    # Process the frame here (e.g., object detection, image filtering, etc.)
    processed_frame = frame  # Placeholder, replace with your processing logic
    return processed_frame







@app.route('/video_feed', methods=['POST','GET'])
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
       



def generate_frames():
    rf = Roboflow(api_key="n71Ls7tYoQ2qkifAa2a2")
    project = rf.workspace().project("garbage-classification-3")
    model_WCP = project.version(2).model
    print('Detecting the waste on Webcam')
    cap = cv2.VideoCapture(1)  # Open the webcam (change the index if multiple webcams are available)

    global numbers 
    numbers = []
    global namelist
    namelist=[]
    
    for i in range(1,100):
        numbers.append(i)
    while True:
        ret, frame = cap.read()  # Read frames from the webcam
        if not ret:
            break
        cv2.imwrite('temp.jpg', frame)
        predictions = model_WCP.predict('temp.jpg')
        predictions_json = predictions.json()
        # print(predictions_json) 
        
        

        for bounding_box in predictions:
            x0 = bounding_box['x'] - bounding_box['width'] / 2
            x1 = bounding_box['x'] + bounding_box['width'] / 2
            y0 = bounding_box['y'] - bounding_box['height'] / 2
            y1 = bounding_box['y'] + bounding_box['height'] / 2
            cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), thickness=3)
            class_name = bounding_box['class']
            # confidence_score = bounding_box['confidence']
            # detection_results = bounding_box
            # class_and_confidence = (class_name, confidence_score)
            # print(class_name, '\n')
            text = str(class_name)
            org = (50, 50)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1.5
            color = (0, 255, 0)
            thickness = 2
            lineType = cv2.LINE_AA
            namelist.append(class_name)
            

            
            cv2.putText(frame, text, org, font, fontScale, color, thickness, lineType)

        processed_frame = process_frame(frame)  # Process the frame
        ret, buffer = cv2.imencode('.jpg', processed_frame)  # Encode the processed frame as JPEG

        # Yield the encoded frame as a byte string
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        print(namelist)
@app.route('/webcam', methods=['POST','GET'])
def webcam():
    print(namelist)
    generate_frames()
    return render_template('webcam.html',names = namelist )




    
 










        
if __name__ == '__main__':
    app.run(debug=True)
