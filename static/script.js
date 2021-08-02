//declarations
source_face_display = document.getElementById("sourceface");
source_upload_btn = document.getElementsByClassName("sf")[0];
destination_face_display = document.getElementById("destinationface");
destination_upload_btn = document.getElementsByClassName("df")[0];
merged_face_display = document.getElementById("outputface");
merge_btn = document.getElementById("mergeface");
downloadbtn = document.getElementById("download");

API_URL = "http://localhost:5000";

//load Main Face
source_upload_btn.onchange = (evt) => {
	console.log("Updating Main Face");
	// Display new picture
	var tgt = evt.target || window.event.srcElement,
    files = tgt.files;

	var fr = new FileReader();
	fr.onload = function () {
		source_face_display.src = fr.result;
	}
	fr.readAsDataURL(files[0]);
}

//load Second Face
destination_upload_btn.onchange = (evt) => {
	console.log("Updating Second Face");
	// Display new picture
	var tgt = evt.target || window.event.srcElement,
    files = tgt.files;

	var fr = new FileReader();
	fr.onload = function () {
		destination_face_display.src = fr.result;
	}
	fr.readAsDataURL(files[0]);
}