from   inference   import   get_model
import   supervision   as   sv
import   cv2

# define the image url to use for inference
image_file   =   "test.jpeg" # [1]
image   =   cv2 . imread ( image_file ) # [1]

# load a pre-trained yolov8n model
# Replace "taylor-swift-records/3" with the model ID from your private model [2]
model   =   get_model ( model_id = "erythrocyte_abnormalities_3/1" ) # [1]

# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
results   =   model . infer ( image )[ 0 ] # [1]

print("Type of results:", type(results))
print("Contents of results:", results)

# load the results into the supervision Detections api
detections   =   sv . Detections . from_inference ( results ) # [1]

# create supervision annotators
bounding_box_annotator   =   sv . BoxAnnotator () # [1]
label_annotator   =   sv . LabelAnnotator () # [1]

# annotate the image with our inference results
annotated_image   =   bounding_box_annotator . annotate (
    scene = image ,
    detections = detections ) # [1]
annotated_image   =   label_annotator . annotate (
    scene = annotated_image ,
    detections = detections ) # [1]

# display the image
sv . plot_image ( annotated_image ) # [1]

