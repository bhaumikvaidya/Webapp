from flask import render_template, url_for, request, send_file, flash
from forms import FaceForm
from flask import Flask
import cv2
from face_swap import face_swap_dlib,face_swap_mediapipe
import os

app = Flask(__name__)




@app.route("/", methods=['GET', 'POST'])
def homepage():
    form = FaceForm()
    if(form.is_submitted()):
        source = request.files['source']
        destination = request.files['destination']
        source.save(f"{os.getcwd()}\\static\input\{source.filename}")
        destination.save(f"{os.getcwd()}\\static\input\{destination.filename}")
      
        #print("Saved Input Files")
        image1= f"{os.getcwd()}\\static\\input\\{source.filename}"
        image2 = f"{os.getcwd()}\\static\\input\\{destination.filename}"
        face_swap_mediapipe(image1,image2)
        #print("Done Swapping")
        #Give Back Image
        source_facepic = url_for('static', filename=f"input/{source.filename}")
        destination_facepic = url_for('static', filename=f"input/{destination.filename}")
        output_path = source.filename.split(".")[0] + "-" + destination.filename.split(".")[0] + ".jpg"
        output_facepic = url_for('static', filename=f"output/{output_path}")
        return render_template('main.html', form=form, sfpic=source_facepic, dfpic=destination_facepic, ofpic=output_facepic, filename=output_path.split(".")[0])
    return render_template('main.html', form=form)

@app.route("/download/<filename>")
def download(filename):
	#print("Download request recieved")
	return send_file(f"{os.getcwd()}\\static\\output\\{filename}.jpg", as_attachment=True, mimetype="image/jpeg")
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)